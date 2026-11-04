//! Posting as a tool: in a room with `reply = "tool"`, the agent's final
//! text is not posted; the agent posts with an MCP tool `post` instead, as
//! often as it likes, or not at all.
//!
//! Three pieces:
//! - [`Posts`]: parley's register of running turns that may post. Each
//!   turn gets a random token; what a post with that token does (send it
//!   to Matrix, record it) is a closure from `main.rs`.
//! - [`serve`]: a Unix socket in the data dir. One connection carries one
//!   post: a JSON line `{token, thread, body}` in, `{ok, event_id | error}`
//!   out.
//! - [`mcp_main`]: `parley mcp`, the MCP server Claude Code starts for the
//!   turn (stdio, newline-delimited JSON-RPC). Its `post` tool forwards to
//!   the socket, with the token from its environment.

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

/// Where a post goes.
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
    pub target: Target,
    pub body: String,
}

type Future_<T> = Pin<Box<dyn Future<Output = T> + Send>>;
/// Sends a post; returns the new event's ID.
pub type SendFn = Arc<dyn Fn(Post) -> Future_<Result<String>> + Send + Sync>;

struct Entry {
    send: SendFn,
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
    pub fn register(self: &Arc<Self>, send: SendFn) -> Result<Ticket> {
        let token = random_token()?;
        let count = Arc::new(AtomicUsize::new(0));
        self.turns.lock().unwrap().insert(
            token.clone(),
            Entry {
                send,
                count: count.clone(),
            },
        );
        Ok(Ticket {
            posts: self.clone(),
            token,
            count,
        })
    }

    /// Send `post` for the turn holding `token`.
    pub async fn post(&self, token: &str, post: Post) -> Result<String> {
        let (send, count) = {
            let turns = self.turns.lock().unwrap();
            let Some(e) = turns.get(token) else {
                bail!("this turn is over or unknown; it can no longer post");
            };
            (e.send.clone(), e.count.clone())
        };
        let id = send(post).await?;
        count.fetch_add(1, Ordering::SeqCst);
        Ok(id)
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
    thread: Option<String>,
    body: String,
}

/// Bind the socket at `path` (replacing a stale one) and answer posts on it
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
                Ok(id) => json!({"ok": true, "event_id": id}),
                Err(e) => json!({"ok": false, "error": format!("{e:#}")}),
            };
            let _ = write.write_all(format!("{answer}\n").as_bytes()).await;
        });
    }
}

async fn handle_request(posts: &Posts, line: &str) -> Result<String> {
    let req: Request = serde_json::from_str(line).context("bad request")?;
    if req.body.trim().is_empty() {
        bail!("the body is empty");
    }
    let target = Target::parse(req.thread.as_deref())?;
    posts
        .post(
            &req.token,
            Post {
                target,
                body: req.body,
            },
        )
        .await
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
nothing, is posted for you as a fallback.\n";

/// The final answers that mean "nothing more to post".
pub fn is_marker(text: &str) -> bool {
    matches!(text.trim(), "" | "[posted]" | "[silent]")
}

// ---- the MCP server (`parley mcp`) ----

const TOOL_DESCRIPTION: &str = "Post a Markdown message to the Matrix room this turn is answering. \
By default it goes to the conversation the message came from (its thread, or the main timeline). \
Returns the new event's ID.";

fn tools() -> Value {
    json!([{
        "name": "post",
        "description": TOOL_DESCRIPTION,
        "inputSchema": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "The message, in Markdown."},
                "thread": {"type": "string", "description": "Omit for this conversation; \"main\" for the room's main timeline; or a thread's root event ID."}
            },
            "required": ["body"]
        }
    }])
}

/// Posts `(thread, body)`; the event ID, or what went wrong.
pub type PostFn<'a> = dyn FnMut(Option<&str>, &str) -> Result<String, String> + 'a;

/// Answer one JSON-RPC message; `None` for notifications. `post` sends
/// `(thread, body)` on and returns the event ID or an error text.
pub fn handle_rpc(msg: &Value, post: &mut PostFn<'_>) -> Option<Value> {
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
            let outcome = match (name, args.get("body").and_then(Value::as_str)) {
                ("post", Some(body)) => post(args.get("thread").and_then(Value::as_str), body)
                    .map(|id| format!("Posted ({id}).")),
                ("post", None) => Err("`body` is required".to_owned()),
                _ => Err(format!("unknown tool {name:?}")),
            };
            let (text, is_error) = match outcome {
                Ok(t) => (t, false),
                Err(e) => (format!("Not posted: {e}"), true),
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

/// Send one post to parley's socket.
fn post_to_socket(
    socket: &Path,
    token: &str,
    thread: Option<&str>,
    body: &str,
) -> Result<String, String> {
    let run = || -> Result<String> {
        let mut stream = std::os::unix::net::UnixStream::connect(socket)
            .with_context(|| format!("parley is not listening on {}", socket.display()))?;
        let req = json!({"token": token, "thread": thread, "body": body});
        stream.write_all(format!("{req}\n").as_bytes())?;
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line)?;
        let answer: Value = serde_json::from_str(&line).context("no answer from parley")?;
        if answer.get("ok") == Some(&Value::Bool(true)) {
            Ok(answer
                .get("event_id")
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

/// `parley mcp`: serve the `post` tool on stdin/stdout.
pub fn mcp_main() -> Result<()> {
    let socket: PathBuf = std::env::var_os("PARLEY_SOCKET")
        .context("PARLEY_SOCKET is not set")?
        .into();
    let token = std::env::var("PARLEY_TOKEN").context("PARLEY_TOKEN is not set")?;
    let mut post = |thread: Option<&str>, body: &str| post_to_socket(&socket, &token, thread, body);
    let stdout = std::io::stdout();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle_rpc(&msg, &mut post),
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
        let mut post = |t: Option<&str>, b: &str| {
            sent.push((t.map(str::to_owned), b.to_owned()));
            Ok("$ev".to_owned())
        };
        let init = handle_rpc(
            &json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
            &mut post,
        )
        .unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert!(
            handle_rpc(
                &json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
                &mut post
            )
            .is_none()
        );
        let list = handle_rpc(
            &json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
            &mut post,
        )
        .unwrap();
        assert_eq!(list["result"]["tools"][0]["name"], "post");
        let call = handle_rpc(
            &json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"post","arguments":{"body":"hi","thread":"main"}}}),
            &mut post,
        )
        .unwrap();
        assert_eq!(call["result"]["isError"], false);
        assert_eq!(call["result"]["content"][0]["text"], "Posted ($ev).");
        let bad = handle_rpc(
            &json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"post","arguments":{}}}),
            &mut post,
        )
        .unwrap();
        assert_eq!(bad["result"]["isError"], true);
        let unknown = handle_rpc(
            &json!({"jsonrpc":"2.0","id":4,"method":"resources/list"}),
            &mut post,
        )
        .unwrap();
        assert_eq!(unknown["error"]["code"], -32601);
        assert_eq!(sent, [(Some("main".to_owned()), "hi".to_owned())]);
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

    #[tokio::test]
    async fn socket_round_trip_counts_posts_and_forgets_dropped_tickets() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = tmp.path().join("post.sock");
        let posts = Arc::new(Posts::default());
        listen(&sock, posts.clone()).unwrap();

        let got = Arc::new(Mutex::new(Vec::new()));
        let g = got.clone();
        let ticket = posts
            .register(Arc::new(move |p: Post| {
                g.lock().unwrap().push(p);
                Box::pin(async { Ok("$new".to_owned()) })
            }))
            .unwrap();
        let token = ticket.token.clone();

        // The MCP side is blocking; run it off the runtime's threads.
        let s = sock.clone();
        let t = token.clone();
        let r = tokio::task::spawn_blocking(move || post_to_socket(&s, &t, None, "hello"))
            .await
            .unwrap();
        assert_eq!(r.unwrap(), "$new");
        let s = sock.clone();
        let t = token.clone();
        let r = tokio::task::spawn_blocking(move || post_to_socket(&s, &t, Some("nope"), "x"))
            .await
            .unwrap();
        assert!(r.unwrap_err().contains("thread must be"));
        assert_eq!(ticket.count(), 1);
        assert_eq!(
            *got.lock().unwrap(),
            [Post {
                target: Target::Here,
                body: "hello".into()
            }]
        );

        drop(ticket);
        let s = sock.clone();
        let r = tokio::task::spawn_blocking(move || post_to_socket(&s, &token, None, "late"))
            .await
            .unwrap();
        assert!(r.unwrap_err().contains("no longer post"));
    }
}
