//! Posting and reading as tools: in a room with `reply = "tool"`, the
//! agent's final text is not posted; the agent posts with an MCP tool
//! `post` instead, as often as it likes, or not at all. It can also `read`
//! rooms (paginated, filtered) and list the `rooms` open to it; what it may
//! read and post to is its room's [`Access`].
//!
//! Three pieces:
//! - [`Posts`]: parley's register of running turns. Each turn gets a random
//!   token, its access rules, and closures from `main.rs` that send a post
//!   to Matrix and fetch history from it.
//! - [`serve`]: a Unix socket in the data dir. One connection carries one
//!   tool call: a JSON line `{token, tool, args}` in, `{ok, text | error}`
//!   out.
//! - [`mcp_main`]: `parley mcp`, the MCP server Claude Code starts for the
//!   turn (stdio, newline-delimited JSON-RPC). Its tools forward to the
//!   socket, with the token from its environment.

use std::{
    collections::HashMap,
    future::Future,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use crate::{
    access::{Access, Scope},
    read, schedule,
};

/// Where in its room a post goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// The conversation the turn answers: its thread, or the main timeline.
    Here,
    /// The room's main timeline.
    Main,
    /// The thread rooted at this event.
    Thread(String),
}

impl Target {
    /// `None` or "" → here, "main" → main timeline, "$…" → that thread.
    pub fn parse(thread: Option<&str>) -> Result<Self> {
        match thread.map(str::trim) {
            None | Some("") => Ok(Self::Here),
            Some(t) if t.eq_ignore_ascii_case("main") => Ok(Self::Main),
            Some(t) if t.starts_with('$') && !t.contains('/') => Ok(Self::Thread(t.to_owned())),
            Some(t) => {
                bail!("thread must be omitted, \"main\" or a thread's root event ID, not {t:?}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Post {
    /// The room's ID.
    pub room: String,
    pub target: Target,
    pub body: String,
}

type Future_<T> = Pin<Box<dyn Future<Output = T> + Send>>;
/// Sends a post; returns the new event's ID.
pub type SendFn = Arc<dyn Fn(Post) -> Future_<Result<String>> + Send + Sync>;
/// Reads a page of a room (by ID).
pub type ReadFn = Arc<dyn Fn(String, read::Query) -> Future_<Result<read::Page>> + Send + Sync>;

/// What a turn's tools do; `main.rs` fills it in.
#[derive(Clone)]
pub struct Turn {
    pub access: Access,
    pub send: SendFn,
    pub read: ReadFn,
    /// Where the `schedule` tools keep their schedules.
    pub schedules: Arc<schedule::Store>,
    /// The generation a schedule made in this turn gets: 0 in a run a
    /// person started, the firing schedule's plus one in a scheduled run.
    pub depth: u32,
    /// The highest generation this room allows.
    pub max_depth: u32,
}

struct Entry {
    turn: Turn,
    count: Arc<AtomicUsize>,
}

/// The turns that may post right now, by token.
#[derive(Default)]
pub struct Posts {
    turns: Mutex<HashMap<String, Entry>>,
}

/// One turn's right to post; withdrawn when dropped.
pub struct Ticket {
    posts: Arc<Posts>,
    pub token: String,
    count: Arc<AtomicUsize>,
}

impl Ticket {
    /// How many posts went out with this ticket.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.posts.turns.lock().unwrap().remove(&self.token);
    }
}

impl Posts {
    pub fn register(self: &Arc<Self>, turn: Turn) -> Result<Ticket> {
        let token = random_token()?;
        let count = Arc::new(AtomicUsize::new(0));
        self.turns.lock().unwrap().insert(
            token.clone(),
            Entry {
                turn,
                count: count.clone(),
            },
        );
        Ok(Ticket {
            posts: self.clone(),
            token,
            count,
        })
    }

    /// Give a standing client (clients.rs) the tools for as long as parley
    /// runs, under its own token.
    pub fn insert(&self, token: String, turn: Turn) {
        let count = Arc::new(AtomicUsize::new(0));
        self.turns
            .lock()
            .unwrap()
            .insert(token, Entry { turn, count });
    }

    /// Run `tool` with `args` for the turn holding `token`; the tool's answer.
    pub async fn call(&self, token: &str, tool: &str, args: &Value) -> Result<String> {
        let (turn, count) = {
            let turns = self.turns.lock().unwrap();
            let Some(e) = turns.get(token) else {
                bail!("this turn is over or unknown; its tools no longer work");
            };
            (e.turn.clone(), e.count.clone())
        };
        let a = &turn.access;
        let s = |k: &str| args.get(k).and_then(Value::as_str);
        match tool {
            "post" => {
                let body = s("body").unwrap_or("");
                if body.trim().is_empty() {
                    bail!("the body is empty");
                }
                let room = a.room(s("room"))?;
                let target = match Target::parse(s("thread"))? {
                    // "Here" in another room is its main timeline.
                    Target::Here if room != a.here => Target::Main,
                    t => t,
                };
                let scope = match &target {
                    Target::Here => a.here_scope.clone(),
                    Target::Main => Scope::Main,
                    Target::Thread(t) => Scope::Thread(t.clone()),
                };
                a.check(&room, &scope, true)?;
                let id = (turn.send)(Post {
                    room,
                    target,
                    body: body.to_owned(),
                })
                .await?;
                count.fetch_add(1, Ordering::SeqCst);
                Ok(format!("Posted ({id})."))
            }
            "read" => {
                let room = a.room(s("room"))?;
                let q = read::Query::from_args(args, &a.here_scope)?;
                a.check(&room, &q.scope, false)?;
                let page = (turn.read)(room.clone(), q.clone()).await?;
                Ok(read::render(&room, &q, &page))
            }
            "rooms" => Ok(a.describe()),
            "schedule" => {
                let room = a.room(s("room"))?;
                // Where its runs happen: here by default (in another
                // room, its main timeline).
                let thread = match Target::parse(s("thread"))? {
                    Target::Here if room != a.here => None,
                    Target::Here => match &a.here_scope {
                        Scope::Thread(t) => Some(t.clone()),
                        _ => None,
                    },
                    Target::Main => None,
                    Target::Thread(t) => Some(t),
                };
                let scope = match &thread {
                    Some(t) => Scope::Thread(t.clone()),
                    None => Scope::Main,
                };
                a.check(&room, &scope, true)?;
                if turn.depth > turn.max_depth {
                    bail!(
                        "refused: this run was started by a chain of {} schedules, and this \
                         room allows at most {} (`max_depth` in room.toml). Ask Kas to \
                         raise it, or finish here.",
                        turn.depth,
                        turn.max_depth
                    );
                }
                let mut job = schedule::Request::from_args(args)?.job(
                    schedule::new_id()?,
                    &room,
                    thread.as_deref(),
                    "agent",
                    chrono::Utc::now(),
                )?;
                job.depth = turn.depth;
                turn.schedules.create(&job)?;
                Ok(format!("Scheduled: {}", job.describe()))
            }
            "schedules" => {
                let room = a.room(s("room"))?;
                a.check(&room, &Scope::All, false)
                    .or_else(|_| a.check(&room, &a.here_scope, false))?;
                let jobs = turn.schedules.list(&room)?;
                if jobs.is_empty() {
                    return Ok(format!("No schedules in {room}."));
                }
                Ok(jobs
                    .iter()
                    .map(|j| format!("- {}", j.describe()))
                    .collect::<Vec<_>>()
                    .join("\n"))
            }
            "cancel" => {
                let room = a.room(s("room"))?;
                let id = s("id").context("`id` is required")?;
                let job = turn
                    .schedules
                    .list(&room)?
                    .into_iter()
                    .find(|j| j.id == id)
                    .with_context(|| format!("no schedule {id} in {room}"))?;
                let scope = match &job.thread {
                    Some(t) => Scope::Thread(t.clone()),
                    None => Scope::Main,
                };
                a.check(&room, &scope, true)?;
                turn.schedules.cancel(&room, id)?;
                Ok(format!("Cancelled `{id}`; it fires no more."))
            }
            "handoff" => {
                let room = a.room(s("room"))?;
                let id = s("id").context("`id` is required")?;
                a.check(&room, &a.here_scope, true)
                    .or_else(|_| a.check(&room, &Scope::Main, true))?;
                turn.schedules.handoff(&room, id, s("text").unwrap_or(""))?;
                Ok(format!("Saved; the next run of `{id}` gets it."))
            }
            _ => bail!("unknown tool {tool:?}"),
        }
    }
}

fn random_token() -> Result<String> {
    use std::io::Read;
    let mut buf = [0u8; 16];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

#[derive(Deserialize)]
struct Request {
    token: String,
    tool: String,
    #[serde(default)]
    args: Value,
}

/// Bind the socket at `path` (replacing a stale one) and answer tool calls on it
/// until the process ends.
pub fn listen(path: &Path, posts: Arc<Posts>) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::remove_file(path);
    let listener = tokio::net::UnixListener::bind(path)
        .with_context(|| format!("could not bind {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    tokio::spawn(serve(listener, posts));
    Ok(())
}

pub async fn serve(listener: tokio::net::UnixListener, posts: Arc<Posts>) {
    loop {
        let stream = match listener.accept().await {
            Ok((s, _)) => s,
            Err(e) => {
                eprintln!("post socket: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };
        let posts = posts.clone();
        tokio::spawn(async move {
            let (read, mut write) = stream.into_split();
            let mut line = String::new();
            if tokio::io::BufReader::new(read)
                .read_line(&mut line)
                .await
                .is_err()
            {
                return;
            }
            let answer = match handle_request(&posts, &line).await {
                Ok(text) => json!({"ok": true, "text": text}),
                Err(e) => json!({"ok": false, "error": format!("{e:#}")}),
            };
            let _ = write.write_all(format!("{answer}\n").as_bytes()).await;
        });
    }
}

async fn handle_request(posts: &Posts, line: &str) -> Result<String> {
    let req: Request = serde_json::from_str(line).context("bad request")?;
    posts.call(&req.token, &req.tool, &req.args).await
}

/// The `--mcp-config` JSON that starts `exe mcp` with the socket and token.
pub fn mcp_config(exe: &Path, socket: &Path, token: &str) -> String {
    json!({
        "mcpServers": {
            "parley": {
                "type": "stdio",
                "command": exe,
                "args": ["mcp"],
                "env": {"PARLEY_SOCKET": socket, "PARLEY_TOKEN": token},
            }
        }
    })
    .to_string()
}

/// What the system prompt says in a `reply = "tool"` room.
pub const PROMPT_NOTE: &str = "\n\n## Posting\n\n\
Your final answer is **not** posted to the room. To say something, call the \
`post` tool (from the `parley` MCP server): it posts Markdown to this \
conversation right away. You may post several times, for example a short \
note before long work and the result after it, or not at all when nothing \
needs saying. Pass `thread: \"main\"` to post in the room's main timeline \
instead of this thread.\n\n\
When you have posted what you want to say, end with a final answer of just \
`[posted]`. If you choose to stay silent, end with just `[silent]`; parley \
then marks the message with ✓. Any other final answer, when you posted \
nothing, is posted for you as a fallback.\n\n\
## Reading rooms\n\n\
The `read` tool reads messages straight from Matrix, including ones from \
before your context starts: this room by default, or another room that is \
open to you (`rooms` lists them, with what you may read and post). It \
returns a page, newest first unless `order: \"oldest\"`, filtered by \
`thread` (\"here\", \"main\" or a root event ID), `since`/`until` \
(YYYY-MM-DD or unix seconds), `sender` and `contains`, with a cursor for \
the next page. Use it when the past matters instead of guessing. `post` \
takes a `room` too.\n\n\
## Schedules\n\n\
Your run ends with this turn, but you can plan later ones: `schedule` \
starts a run at a time (`at`, `in`), repeatedly (`every`, `cron`, with \
`times`), here or in another room you may post to; `only_if_new` skips a \
fire when nobody wrote since the last run. `schedules` lists them, \
`cancel` ends one. For long work, do a step, `handoff` what the next run \
needs, and schedule it: state lives on, the process does not. A message \
starting with `[schedule <id>: …]` was sent by a schedule, not a person.\n";

/// The final answers that mean "nothing more to post".
pub fn is_marker(text: &str) -> bool {
    matches!(text.trim(), "" | "[posted]" | "[silent]")
}

// ---- the MCP server (`parley mcp`) ----

fn tools() -> Value {
    let room = json!({"type": "string", "description": "Omit for this room; or a room's name or ID from `rooms`."});
    json!([
        {
            "name": "post",
            "description": "Post a Markdown message to a Matrix room. By default it goes to the conversation this turn answers (its thread, or the main timeline). Returns the new event's ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "The message, in Markdown."},
                    "thread": {"type": "string", "description": "Omit for this conversation (in another room: its main timeline); \"main\" for the main timeline; or a thread's root event ID."},
                    "room": room,
                },
                "required": ["body"]
            }
        },
        {
            "name": "read",
            "description": "Read messages of a Matrix room from the server: one page, filtered, with a cursor for the next. Reaches back to the room's first message.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "room": room,
                    "thread": {"type": "string", "description": "Omit for the whole room; \"here\" for the conversation this turn answers; \"main\" for the main timeline only; or a thread's root event ID."},
                    "order": {"type": "string", "enum": ["newest", "oldest"], "description": "Walk back from the newest message (default) or forward from the oldest."},
                    "limit": {"type": "integer", "description": "Messages per page, 1-100 (default 20)."},
                    "cursor": {"type": "string", "description": "From the previous page, to continue it (keep the other arguments the same)."},
                    "since": {"type": "string", "description": "Only messages from then on: YYYY-MM-DD, YYYY-MM-DDTHH:MM (UTC) or unix seconds."},
                    "until": {"type": "string", "description": "Only messages before then (a date includes that whole day)."},
                    "sender": {"type": "string", "description": "Only senders whose user ID contains this, e.g. \"kas\"."},
                    "contains": {"type": "string", "description": "Only messages whose text contains this (any case)."}
                }
            }
        },
        {
            "name": "rooms",
            "description": "List the rooms this turn may read and post to, and which parts of them.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "schedule",
            "description": "Plan a run for later, once or repeatedly. When it fires, an agent run starts in that room and thread with `prompt` as its message (with the previous run's handoff), as if someone had sent it; the run lasts one turn, like any other. Say when with exactly one of: `at` or `in` (once), `every` (repeatedly; optionally starting at `at`/`in`), or `cron`. Use it to follow up later, to check something periodically, or to continue long work in steps (leave state with `handoff`).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What the run should do; written to the future agent, which has no other memory of this request than the room's context."},
                    "at": {"type": "string", "description": "A time: RFC 3339, or local (Europe/Amsterdam) YYYY-MM-DD HH:MM."},
                    "in": {"type": "string", "description": "A delay from now: 90s, 20m, 1h30m, 2d."},
                    "every": {"type": "string", "description": "Repeat with this period (at least 1m), e.g. 30m, 1d. First run after one period, or at `at`/`in`."},
                    "cron": {"type": "string", "description": "Cron with seconds, local time: \"sec min hour day month weekday\", e.g. \"0 0 8 * * Mon-Fri\"."},
                    "times": {"type": "integer", "description": "How many runs in all (default: 1 for at/in, forever for every/cron)."},
                    "only_if_new": {"type": "boolean", "description": "Skip a fire (without starting an agent) when nobody wrote in the room, or the schedule's thread, since its last run."},
                    "thread": {"type": "string", "description": "Omit for this conversation (in another room: its main timeline); \"main\"; or a thread's root event ID."},
                    "room": room,
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "schedules",
            "description": "List a room's live schedules, soonest first, with their IDs.",
            "inputSchema": {"type": "object", "properties": {"room": room}}
        },
        {
            "name": "cancel",
            "description": "End a schedule: it fires no more. (A run already going is not stopped.)",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "room": room}, "required": ["id"]}
        },
        {
            "name": "handoff",
            "description": "Replace a schedule's scratchboard: what its next run is shown first. Use it at the end of a scheduled run to hand over state (what was done, what is next).",
            "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "room": room}, "required": ["id", "text"]}
        }
    ])
}

/// Runs tool `name` with `args`; its answer, or what went wrong.
pub type CallFn<'a> = dyn FnMut(&str, &Value) -> Result<String, String> + 'a;

/// Answer one JSON-RPC message; `None` for notifications. Tool calls go
/// to `call`.
pub fn handle_rpc(msg: &Value, call: &mut CallFn<'_>) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => json!({
            "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "parley", "version": env!("CARGO_PKG_VERSION")},
        }),
        "ping" => json!({}),
        "tools/list" => json!({"tools": tools()}),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let outcome = match name {
                "post" if args.get("body").and_then(Value::as_str).is_none() => {
                    Err("`body` is required".to_owned())
                }
                "post" | "read" | "rooms" | "schedule" | "schedules" | "cancel" | "handoff" => {
                    call(name, &args)
                }
                _ => Err(format!("unknown tool {name:?}")),
            };
            let (text, is_error) = match outcome {
                Ok(t) => (t, false),
                Err(e) if name == "post" => (format!("Not posted: {e}"), true),
                Err(e) => (format!("Failed: {e}"), true),
            };
            json!({"content": [{"type": "text", "text": text}], "isError": is_error})
        }
        _ => {
            return Some(json!({
                "jsonrpc": "2.0", "id": id,
                "error": {"code": -32601, "message": format!("unknown method {method:?}")},
            }));
        }
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// Send one tool call to parley's socket.
fn call_socket(socket: &Path, token: &str, tool: &str, args: &Value) -> Result<String, String> {
    let run = || -> Result<String> {
        let mut stream = std::os::unix::net::UnixStream::connect(socket)
            .with_context(|| format!("parley is not listening on {}", socket.display()))?;
        let req = json!({"token": token, "tool": tool, "args": args});
        stream.write_all(format!("{req}\n").as_bytes())?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line)?;
        let answer: Value = serde_json::from_str(&line).context("no answer from parley")?;
        if answer.get("ok") == Some(&Value::Bool(true)) {
            Ok(answer
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned())
        } else {
            bail!(
                "{}",
                answer
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
            )
        }
    };
    run().map_err(|e| format!("{e:#}"))
}

/// `parley mcp`: serve the tools on stdin/stdout, through the socket at
/// `socket` with `token`.
pub fn mcp_main(socket: PathBuf, token: String) -> Result<()> {
    let mut call = |tool: &str, args: &Value| call_socket(&socket, &token, tool, args);
    let stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle_rpc(&msg, &mut call),
            Err(e) => Some(json!({
                "jsonrpc": "2.0", "id": null,
                "error": {"code": -32700, "message": e.to_string()},
            })),
        };
        if let Some(a) = answer {
            let mut out = stdout.lock();
            writeln!(out, "{a}")?;
            out.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::Grant;

    #[test]
    fn targets_parse() {
        assert_eq!(Target::parse(None).unwrap(), Target::Here);
        assert_eq!(Target::parse(Some(" ")).unwrap(), Target::Here);
        assert_eq!(Target::parse(Some("Main")).unwrap(), Target::Main);
        assert_eq!(
            Target::parse(Some("$abc")).unwrap(),
            Target::Thread("$abc".into())
        );
        assert!(Target::parse(Some("abc")).is_err());
        assert!(Target::parse(Some("$../x")).is_err());
    }

    #[test]
    fn markers() {
        assert!(is_marker(" [silent]\n"));
        assert!(is_marker("[posted]"));
        assert!(is_marker(""));
        assert!(!is_marker("Here is the answer"));
    }

    #[test]
    fn rpc_initialize_list_and_call() {
        let mut sent = Vec::new();
        let mut call = |t: &str, a: &Value| {
            sent.push((t.to_owned(), a.clone()));
            Ok("Posted ($ev).".to_owned())
        };
        let init = handle_rpc(
            &json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
            &mut call,
        )
        .unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert!(
            handle_rpc(
                &json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
                &mut call
            )
            .is_none()
        );
        let list = handle_rpc(
            &json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
            &mut call,
        )
        .unwrap();
        let names: Vec<&str> = list["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "post",
                "read",
                "rooms",
                "schedule",
                "schedules",
                "cancel",
                "handoff"
            ]
        );
        let c = handle_rpc(
            &json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"post","arguments":{"body":"hi","thread":"main"}}}),
            &mut call,
        )
        .unwrap();
        assert_eq!(c["result"]["isError"], false);
        assert_eq!(c["result"]["content"][0]["text"], "Posted ($ev).");
        let bad = handle_rpc(
            &json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"post","arguments":{}}}),
            &mut call,
        )
        .unwrap();
        assert_eq!(bad["result"]["isError"], true);
        let unknown = handle_rpc(
            &json!({"jsonrpc":"2.0","id":4,"method":"resources/list"}),
            &mut call,
        )
        .unwrap();
        assert_eq!(unknown["error"]["code"], -32601);
        assert_eq!(
            sent,
            [("post".to_owned(), json!({"body":"hi","thread":"main"}))]
        );
    }

    #[test]
    fn mcp_config_names_the_server() {
        let c: Value = serde_json::from_str(&mcp_config(
            Path::new("/bin/parley"),
            Path::new("/d/post.sock"),
            "tok",
        ))
        .unwrap();
        let s = &c["mcpServers"]["parley"];
        assert_eq!(s["command"], "/bin/parley");
        assert_eq!(s["args"][0], "mcp");
        assert_eq!(s["env"]["PARLEY_TOKEN"], "tok");
    }

    /// A turn in `!me:x`, thread `$here`, that may read all of `!life:x`
    /// and only read thread `$t` of `!nix:x`.
    fn turn(
        got: Arc<Mutex<Vec<Post>>>,
        reads: Arc<Mutex<Vec<(String, read::Query)>>>,
        rooms: &Path,
    ) -> Turn {
        let names = [("life".to_owned(), "!life:x".to_owned())].into();
        let grants = [
            Grant {
                room: "life".into(),
                thread: None,
                read: true,
                write: false,
            },
            Grant {
                room: "!nix:x".into(),
                thread: Some("$t".into()),
                read: true,
                write: false,
            },
        ];
        let (access, _) = Access::new("!me:x", Some("parley"), Some("$here"), &grants, &names);
        Turn {
            access,
            send: Arc::new(move |p: Post| {
                got.lock().unwrap().push(p);
                Box::pin(async { Ok("$new".to_owned()) })
            }),
            read: Arc::new(move |room, q| {
                reads.lock().unwrap().push((room, q));
                Box::pin(async {
                    Ok(read::Page {
                        msgs: vec![],
                        next: None,
                        scanned: 0,
                    })
                })
            }),
            schedules: Arc::new(schedule::Store::new(rooms.to_owned())),
            depth: 0,
            max_depth: 10,
        }
    }

    #[tokio::test]
    async fn socket_round_trip_checks_access_and_forgets_dropped_tickets() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = tmp.path().join("post.sock");
        let posts = Arc::new(Posts::default());
        listen(&sock, posts.clone()).unwrap();

        let got = Arc::new(Mutex::new(Vec::new()));
        let reads = Arc::new(Mutex::new(Vec::new()));
        let ticket = posts
            .register(turn(got.clone(), reads.clone(), tmp.path()))
            .unwrap();
        let token = ticket.token.clone();

        // The MCP side is blocking; run it off the runtime's threads.
        let call = |tool: &'static str, args: Value| {
            let (s, t) = (sock.clone(), token.clone());
            tokio::task::spawn_blocking(move || call_socket(&s, &t, tool, &args))
        };
        assert_eq!(
            call("post", json!({"body": "hello"}))
                .await
                .unwrap()
                .unwrap(),
            "Posted ($new)."
        );
        let e = call("post", json!({"body": "x", "thread": "nope"}))
            .await
            .unwrap()
            .unwrap_err();
        assert!(e.contains("thread must be"), "{e}");
        let e = call("post", json!({"body": "x", "room": "life"}))
            .await
            .unwrap()
            .unwrap_err();
        assert!(e.contains("may not post to !life:x"), "{e}");
        assert_eq!(ticket.count(), 1);
        assert_eq!(
            *got.lock().unwrap(),
            [Post {
                room: "!me:x".into(),
                target: Target::Here,
                body: "hello".into()
            }]
        );

        let r = call("read", json!({"room": "life", "contains": "x"}))
            .await
            .unwrap()
            .unwrap();
        assert!(r.contains("0 message(s) from !life:x"), "{r}");
        call("read", json!({"room": "!nix:x", "thread": "$t"}))
            .await
            .unwrap()
            .unwrap();
        let e = call("read", json!({"room": "!nix:x"}))
            .await
            .unwrap()
            .unwrap_err();
        assert!(e.contains("only: thread $t"), "{e}");
        call("read", json!({"thread": "here"}))
            .await
            .unwrap()
            .unwrap();
        let reads: Vec<(String, Scope)> = reads
            .lock()
            .unwrap()
            .iter()
            .map(|(r, q)| (r.clone(), q.scope.clone()))
            .collect();
        assert_eq!(
            reads,
            [
                ("!life:x".to_owned(), Scope::All),
                ("!nix:x".to_owned(), Scope::Thread("$t".into())),
                ("!me:x".to_owned(), Scope::Thread("$here".into())),
            ]
        );
        let rooms = call("rooms", json!({})).await.unwrap().unwrap();
        assert!(rooms.contains("life !life:x: whole room, read"), "{rooms}");

        // Schedules: made here (in thread $here), listed, handed off, ended.
        let r = call("schedule", json!({"prompt": "look again", "in": "1h"}))
            .await
            .unwrap()
            .unwrap();
        assert!(r.starts_with("Scheduled: `"), "{r}");
        let id = r.split('`').nth(1).unwrap().to_owned();
        let e = call(
            "schedule",
            json!({"prompt": "x", "in": "1h", "room": "life"}),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(e.contains("may not post to !life:x"), "{e}");
        let list = call("schedules", json!({})).await.unwrap().unwrap();
        assert!(
            list.contains(&id) && list.contains("thread $here"),
            "{list}"
        );
        call("handoff", json!({"id": id, "text": "state"}))
            .await
            .unwrap()
            .unwrap();
        let r = call("cancel", json!({"id": id})).await.unwrap().unwrap();
        assert!(r.contains("fires no more"), "{r}");
        let list = call("schedules", json!({})).await.unwrap().unwrap();
        assert!(list.starts_with("No schedules"), "{list}");

        // A run at the end of a chain of schedules may not make another;
        // one within the limit makes the next generation.
        let deep = |depth| {
            let mut t = turn(got.clone(), Arc::default(), tmp.path());
            t.depth = depth;
            posts.register(t).unwrap()
        };
        let (over, within) = (deep(11), deep(10));
        let call_as = |tok: String, args: Value| {
            let s = sock.clone();
            tokio::task::spawn_blocking(move || call_socket(&s, &tok, "schedule", &args))
        };
        let args = json!({"prompt": "again", "in": "1h"});
        let e = call_as(over.token.clone(), args.clone())
            .await
            .unwrap()
            .unwrap_err();
        assert!(e.contains("max_depth"), "{e}");
        let r = call_as(within.token.clone(), args).await.unwrap().unwrap();
        assert!(r.contains("generation 10"), "{r}");
        drop((over, within));

        drop(ticket);
        let e = call("post", json!({"body": "late"}))
            .await
            .unwrap()
            .unwrap_err();
        assert!(e.contains("no longer work"), "{e}");
    }
}
