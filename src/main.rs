//! parley — a Matrix bot that lets people talk to home-lab agents.
//!
//! The bot joins rooms it is invited to (only by people on the allow list)
//! and answers every text message with an agent run (see `agent.rs`). The
//! bot is an ordinary room member with its own device, so it receives the
//! room keys like any other member and can read encrypted rooms while the
//! server cannot.
//!
//! Configuration (environment):
//!   PARLEY_HOMESERVER     default https://chat.lab.artintel.ai
//!   PARLEY_PASSWORD_FILE  file whose 1st line is the user ID, 2nd the password
//!                         (only read on first start; afterwards the saved
//!                         session is reused)
//!   PARLEY_ALLOWED        comma-separated user IDs allowed to invite the bot
//!   PARLEY_DATA_DIR       default <data dir>/parley
//!   PARLEY_CLAUDE         the claude program, default `claude`
//!
//! Each room's system prompt is `PARLEY_DATA_DIR/rooms/<room id>/SYSTEM.md`,
//! written with a default on the room's first message. Edit it any time; the
//! next message uses the new text.
//!
//! Verification: start "Verify" on @agent in Element. The bot accepts
//! requests from people on the allow list and prints the emoji to its log.
//! The bot confirms its side by itself; *you* are the check: compare the log
//! with what Element shows, and only press "They match" if they do.
//!
//! Other state in PARLEY_DATA_DIR: `session.json` (access token, device ID)
//! and `store/` (encrypted SQLite with the bot's encryption keys). Losing the
//! store means losing the keys: delete both and log in again as a new device.

mod access;
mod agent;
mod breaker;
mod layered;
mod memory;
mod post;
mod read;
mod schedule;

use std::{
    env,
    future::Future,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use matrix_sdk::{
    Client, LoopCtrl, Room, RoomMemberships, RoomState,
    authentication::matrix::MatrixSession,
    config::SyncSettings,
    encryption::{
        CrossSigningResetAuthType,
        verification::{
            SasState, SasVerification, Verification, VerificationRequest, VerificationRequestState,
            format_emojis,
        },
    },
    event_handler::Ctx,
    ruma::{
        OwnedEventId, OwnedRoomId, OwnedUserId, RoomId,
        api::{client::uiaa, error::ErrorKind},
        events::key::verification::request::ToDeviceKeyVerificationRequestEvent,
        events::reaction::ReactionEventContent,
        events::relation::{Annotation, Thread},
        events::room::{
            member::StrippedRoomMemberEvent,
            message::{
                MessageType, OriginalSyncRoomMessageEvent, Relation, RoomMessageEventContent,
            },
        },
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    agent::{Agent, ClaudeCode},
    breaker::{Breaker, State},
};

/// What we save after the first login, so every later start is the same device.
#[derive(Serialize, Deserialize)]
struct SavedSession {
    homeserver: String,
    /// Passphrase for the SQLite store that holds the encryption keys.
    store_passphrase: String,
    matrix: MatrixSession,
}

struct Config {
    homeserver: String,
    password_file: Option<PathBuf>,
    allowed: Vec<OwnedUserId>,
    data_dir: PathBuf,
    claude: String,
}

impl Config {
    fn from_env() -> Result<Self> {
        let allowed =
            parse_allowed(&env::var("PARLEY_ALLOWED").unwrap_or_else(|_| {
                "@kas:chat.lab.artintel.ai,@camille:chat.lab.artintel.ai".into()
            }))?;
        let data_dir = match env::var_os("PARLEY_DATA_DIR") {
            Some(d) => d.into(),
            None => dirs::data_dir().context("no data dir")?.join("parley"),
        };
        Ok(Self {
            homeserver: env::var("PARLEY_HOMESERVER")
                .unwrap_or_else(|_| "https://chat.lab.artintel.ai".into()),
            password_file: env::var_os("PARLEY_PASSWORD_FILE").map(Into::into),
            allowed,
            data_dir,
            claude: env::var("PARLEY_CLAUDE").unwrap_or_else(|_| "claude".into()),
        })
    }
}

/// The user IDs in a comma-separated `PARLEY_ALLOWED` value.
fn parse_allowed(list: &str) -> Result<Vec<OwnedUserId>> {
    list.split(',')
        .map(|s| {
            s.trim()
                .try_into()
                .with_context(|| format!("bad user ID {s:?}"))
        })
        .collect()
}

/// How long to wait after the `failures`-th sync failure in a row (≥ 1):
/// 1, 2, 4, 8 s, then 8 s.
fn sync_backoff(failures: u32) -> Duration {
    Duration::from_secs(1 << (failures - 1).min(3))
}

#[tokio::main]
async fn main() -> Result<()> {
    // `parley mcp`: the `post` tool's MCP server, started by Claude Code for
    // one turn. Its stdout is the protocol, so nothing else may print.
    if env::args().nth(1).as_deref() == Some("mcp") {
        return post::mcp_main();
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parley=info,matrix_sdk=warn".into()),
        )
        .init();

    let config = Config::from_env()?;
    std::fs::create_dir_all(&config.data_dir)?;
    let session_file = config.data_dir.join("session.json");
    let store_dir = config.data_dir.join("store");

    let client = if session_file.exists() {
        let saved: SavedSession = serde_json::from_str(&std::fs::read_to_string(&session_file)?)?;
        let client = Client::builder()
            .homeserver_url(&saved.homeserver)
            .sqlite_store(&store_dir, Some(&saved.store_passphrase))
            .build()
            .await?;
        client.restore_session(saved.matrix).await?;
        println!("restored session for {}", client.user_id().unwrap());
        client
    } else {
        first_login(&config, &session_file, &store_dir).await?
    };

    // Catch up once without handlers, so old messages are not answered again.
    let response = client.sync_once(SyncSettings::default()).await?;

    let allowed = config.allowed.clone();
    client.add_event_handler(
        move |ev: StrippedRoomMemberEvent, client: Client, room: Room| {
            let allowed = allowed.clone();
            async move { on_invite(ev, client, room, &allowed).await }
        },
    );
    let agent = Arc::new(Agent::new(
        ClaudeCode {
            program: config.claude.clone(),
        },
        config.data_dir.join("rooms"),
    ));
    client.add_event_handler_context(agent.clone());
    // The socket the `post` tool talks to (see post.rs).
    let tooling = Arc::new(Tooling {
        posts: Arc::new(post::Posts::default()),
        socket: config.data_dir.join("post.sock"),
        exe: env::current_exe().context("cannot find parley's own program")?,
        schedules: Arc::new(schedule::Store::new(config.data_dir.join("rooms"))),
        runs: Runs::default(),
    });
    post::listen(&tooling.socket, tooling.posts.clone())?;
    client.add_event_handler_context(tooling.clone());
    client.add_event_handler(on_message);
    let allowed = config.allowed.clone();
    client.add_event_handler(
        move |ev: ToDeviceKeyVerificationRequestEvent, client: Client| {
            let allowed = allowed.clone();
            async move {
                if !allowed.contains(&ev.sender) {
                    println!("ignoring verification request from {}", ev.sender);
                    return;
                }
                if let Some(request) = client
                    .encryption()
                    .get_verification_request(&ev.sender, &ev.content.transaction_id)
                    .await
                {
                    tokio::spawn(handle_verification_request(request));
                }
            }
        },
    );

    tokio::spawn(scheduler(client.clone(), agent, tooling));
    println!("listening");
    sync_forever(&client, SyncSettings::default().token(response.next_batch)).await
}

/// Sync until the access token stops working.
///
/// The SDK's sync stream keeps its place between calls, so a failed request
/// loses nothing. Failures back off (1, 2, 4, 8 s); after 5 in a row the
/// circuit opens and the bot waits 5 minutes before one probe. A revoked or
/// unknown token is not retried: that needs a new login, not patience.
async fn sync_forever(client: &Client, settings: SyncSettings) -> Result<()> {
    let breaker = Arc::new(Mutex::new(Breaker::new(5, Duration::from_secs(300))));
    client
        .sync_with_result_callback(settings, |result| {
            let breaker = breaker.clone();
            async move {
                let wait = match result {
                    Ok(_) => {
                        let mut b = breaker.lock().unwrap();
                        if b.failures() > 0 {
                            println!("sync recovered after {} failures", b.failures());
                        }
                        b.success();
                        return Ok(LoopCtrl::Continue);
                    }
                    Err(e) => {
                        if let Some(ErrorKind::UnknownToken(_)) = e.client_api_error_kind() {
                            eprintln!("access token rejected; log in again");
                            return Err(e);
                        }
                        let mut b = breaker.lock().unwrap();
                        let opened = b.failure();
                        match b.state() {
                            State::Open(left) => {
                                if opened {
                                    eprintln!(
                                        "sync circuit open after {} failures ({e}); next try in {}s",
                                        b.failures(),
                                        left.as_secs()
                                    );
                                }
                                left
                            }
                            _ => {
                                let wait = sync_backoff(b.failures());
                                eprintln!("sync failed ({e}); retrying in {}s", wait.as_secs());
                                wait
                            }
                        }
                    }
                };
                tokio::time::sleep(wait).await;
                Ok(LoopCtrl::Continue)
            }
        })
        .await?;
    Ok(())
}

/// Log in with the password once, save the session, and set up cross-signing
/// so the bot's device is signed by its own identity.
async fn first_login(
    config: &Config,
    session_file: &PathBuf,
    store_dir: &PathBuf,
) -> Result<Client> {
    let Some(path) = &config.password_file else {
        bail!("no saved session; set PARLEY_PASSWORD_FILE for the first login");
    };
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let (Some(user), Some(password)) = (lines.next(), lines.next()) else {
        bail!(
            "{} must hold the user ID and the password on two lines",
            path.display()
        );
    };

    let store_passphrase = random_hex(32)?;
    let client = Client::builder()
        .homeserver_url(&config.homeserver)
        .sqlite_store(store_dir, Some(&store_passphrase))
        .build()
        .await?;
    client
        .matrix_auth()
        .login_username(user, password)
        .initial_device_display_name("parley")
        .await?;

    let saved = SavedSession {
        homeserver: config.homeserver.clone(),
        store_passphrase,
        matrix: client
            .matrix_auth()
            .session()
            .context("logged in but no session")?,
    };
    write_private(session_file, &serde_json::to_string_pretty(&saved)?)?;
    println!(
        "logged in as {user}; session saved to {}",
        session_file.display()
    );

    // A first sync uploads the device keys; then create the cross-signing
    // identity. The server asks for the password again (user-interactive auth).
    client.sync_once(SyncSettings::default()).await?;
    if let Some(handle) = client.encryption().reset_cross_signing().await? {
        if let CrossSigningResetAuthType::Uiaa(uiaa) = handle.auth_type() {
            let mut auth =
                uiaa::Password::new(client.user_id().unwrap().to_owned().into(), password.into());
            auth.session = uiaa.session.clone();
            handle.auth(Some(uiaa::AuthData::Password(auth))).await?;
        }
    }
    println!("cross-signing identity created");
    Ok(client)
}

/// Join rooms we are invited to, but only when the inviter is on the allow list.
async fn on_invite(
    ev: StrippedRoomMemberEvent,
    client: Client,
    room: Room,
    allowed: &[OwnedUserId],
) {
    if ev.state_key != client.user_id().unwrap() {
        return;
    }
    if !allowed.contains(&ev.sender) {
        println!("ignoring invite to {} from {}", room.room_id(), ev.sender);
        let _ = room.leave().await;
        return;
    }
    match room.join().await {
        Ok(_) => println!("joined {} (invited by {})", room.room_id(), ev.sender),
        Err(e) => eprintln!("could not join {}: {e}", room.room_id()),
    }
}

/// What runs need besides the agent: posting tools, schedules, and the
/// runs going on now (for `!stop`).
struct Tooling {
    posts: Arc<post::Posts>,
    socket: PathBuf,
    /// This program, started as `parley mcp` for each run.
    exe: PathBuf,
    schedules: Arc<schedule::Store>,
    runs: Runs,
}

/// The runs going on (or waiting for their room's turn), so `!stop` can
/// kill them. Aborting a run's task drops its `claude` child, which kills it.
#[derive(Default)]
struct Runs {
    next: std::sync::atomic::AtomicU64,
    #[allow(clippy::type_complexity)]
    live: Mutex<
        std::collections::HashMap<u64, (OwnedRoomId, Option<String>, tokio::task::AbortHandle)>,
    >,
}

impl Runs {
    /// Run `fut` as the run of `room`'s conversation `thread`, until it ends
    /// or is stopped.
    async fn track(
        &self,
        room: OwnedRoomId,
        thread: Option<String>,
        fut: impl Future<Output = ()> + Send + 'static,
    ) {
        let task = tokio::spawn(fut);
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.live
            .lock()
            .unwrap()
            .insert(id, (room, thread, task.abort_handle()));
        let _ = task.await;
        self.live.lock().unwrap().remove(&id);
    }

    /// Kill the runs of `room`'s conversation `thread`; how many there were.
    fn stop(&self, room: &RoomId, thread: Option<&str>) -> usize {
        let mut live = self.live.lock().unwrap();
        let ids: Vec<u64> = live
            .iter()
            .filter(|(_, (r, t, _))| r == room && t.as_deref() == thread)
            .map(|(id, _)| *id)
            .collect();
        for id in &ids {
            if let Some((_, _, h)) = live.remove(id) {
                h.abort();
            }
        }
        ids.len()
    }
}

/// Shows "typing…" in a room until dropped (a notice lasts ~4 s).
struct Typing(tokio::task::JoinHandle<()>);

impl Typing {
    fn start(room: &Room) -> Self {
        let room = room.clone();
        Self(tokio::spawn(async move {
            loop {
                let _ = room.typing_notice(true).await;
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }))
    }
}

impl Drop for Typing {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The content of a post to `target`, for a turn in the conversation
/// `here` (a thread root, or the main timeline) answering `reply_to` (a
/// message, or nothing for a schedule's run).
fn post_content(
    body: String,
    target: &post::Target,
    here: Option<&OwnedEventId>,
    reply_to: Option<&OwnedEventId>,
) -> Result<(RoomMessageEventContent, Option<String>)> {
    let mut content = RoomMessageEventContent::text_markdown(body);
    let root = match target {
        post::Target::Here => here.cloned(),
        post::Target::Main => None,
        post::Target::Thread(t) => {
            Some(OwnedEventId::try_from(t.as_str()).context("bad thread ID")?)
        }
    };
    let thread = root.as_ref().map(|r| r.as_str().to_owned());
    if let Some(root) = root {
        // In its own thread a post follows the message it answers; elsewhere
        // (or without one) the thread's root.
        let latest = match (target, reply_to) {
            (post::Target::Here, Some(r)) => r.clone(),
            _ => root.clone(),
        };
        content.relates_to = Some(Relation::Thread(Thread::plain(root, latest)));
    }
    Ok((content, thread))
}

/// The tools of a turn in `room`, in the conversation `here` (a thread
/// root, or the main timeline), answering `reply_to`: its access rules,
/// posting, reading and schedules.
async fn tool_turn(
    client: &Client,
    agent: &Arc<Agent<ClaudeCode>>,
    tooling: &Tooling,
    room: &Room,
    here: &Option<OwnedEventId>,
    reply_to: &Option<OwnedEventId>,
    depth: u32,
) -> Result<post::Turn> {
    let thread_id = here.as_ref().map(|t| t.as_str());
    let max_depth = agent.max_depth(room.room_id()).await;
    let (access, problems) = agent.access(room.room_id(), thread_id).await?;
    for p in problems {
        eprintln!("[{}] access: {p}", room.room_id());
    }
    let (c, agent, here, reply_to, home) = (
        client.clone(),
        agent.clone(),
        here.clone(),
        reply_to.clone(),
        room.room_id().to_owned(),
    );
    let send: post::SendFn = Arc::new(move |p: post::Post| {
        let (c, agent, here, reply_to, home) = (
            c.clone(),
            agent.clone(),
            here.clone(),
            reply_to.clone(),
            home.clone(),
        );
        Box::pin(async move {
            let room = joined(&c, &p.room)?;
            let body = p.body.clone();
            let (content, thread) = if room.room_id() == home {
                post_content(p.body, &p.target, here.as_ref(), reply_to.as_ref())?
            } else {
                // Another room: `Here` was made `Main` already; there is no
                // message to reply to there.
                let target = match p.target {
                    post::Target::Here => post::Target::Main,
                    t => t,
                };
                post_content(p.body, &target, None, None)?
            };
            let sent = room.send(content).await?;
            if let Err(e) = agent
                .record(room.room_id(), thread.as_deref(), "agent", &body)
                .await
            {
                eprintln!("could not write transcript: {e}");
            }
            Ok(sent.response.event_id.to_string())
        })
    });
    let c = client.clone();
    let read: post::ReadFn = Arc::new(move |room_id: String, q: read::Query| {
        let c = c.clone();
        Box::pin(async move {
            let room = joined(&c, &room_id)?;
            read_page(&room, &q).await
        })
    });
    Ok(post::Turn {
        access,
        send,
        read,
        schedules: tooling.schedules.clone(),
        depth,
        max_depth,
    })
}
/// The joined room with this ID.
fn joined(client: &Client, id: &str) -> Result<Room> {
    let id: &RoomId = id.try_into().context("bad room ID")?;
    match client.get_room(id) {
        Some(r) if r.state() == RoomState::Joined => Ok(r),
        _ => anyhow::bail!("parley is not in room {id}"),
    }
}

/// One page of `room`'s history for the `read` tool. A thread is read
/// through its relations (plus its root), the rest through `/messages`.
async fn read_page(room: &Room, q: &read::Query) -> Result<read::Page> {
    use matrix_sdk::room::{IncludeRelations, MessagesOptions, RelationsOptions};
    use matrix_sdk::ruma::{UInt, api::Direction, events::relation::RelationType};
    let raw = |ev: &matrix_sdk::deserialized_responses::TimelineEvent| {
        serde_json::from_str::<serde_json::Value>(ev.raw().json().get()).ok()
    };
    let dir = if q.oldest_first {
        Direction::Forward
    } else {
        Direction::Backward
    };
    let root = match &q.scope {
        access::Scope::Thread(t) => {
            Some(OwnedEventId::try_from(t.as_str()).context("bad thread ID")?)
        }
        _ => None,
    };
    read::collect(q, |token: Option<String>| {
        let (room, root) = (room.clone(), root.clone());
        async move {
            let Some(root) = root else {
                let mut opts = MessagesOptions::new(dir).from(token.as_deref());
                opts.limit = UInt::from(100u32);
                let m = room.messages(opts).await?;
                let next = if m.chunk.is_empty() { None } else { m.end };
                return Ok(read::Chunk {
                    events: m.chunk.iter().filter_map(raw).collect(),
                    next,
                });
            };
            // A thread: its root comes first walking forward, last walking
            // back. Tokens: "root", or "rel:<relations token>".
            let root_chunk = || async {
                let ev = room.event(&root, None).await?;
                anyhow::Ok(raw(&ev).into_iter().collect::<Vec<_>>())
            };
            match token.as_deref() {
                None if q.oldest_first => Ok(read::Chunk {
                    events: root_chunk().await?,
                    next: Some("rel:".into()),
                }),
                Some("root") => Ok(read::Chunk {
                    events: root_chunk().await?,
                    next: None,
                }),
                t => {
                    let from = t
                        .and_then(|t| t.strip_prefix("rel:"))
                        .filter(|t| !t.is_empty())
                        .map(str::to_owned);
                    let rel = room
                        .relations(
                            root.clone(),
                            RelationsOptions {
                                from,
                                dir,
                                limit: Some(UInt::from(100u32)),
                                include_relations: IncludeRelations::RelationsOfType(
                                    RelationType::Thread,
                                ),
                                recurse: false,
                            },
                        )
                        .await?;
                    let next = match rel.next_batch_token {
                        Some(n) => Some(format!("rel:{n}")),
                        None if !q.oldest_first => Some("root".into()),
                        None => None,
                    };
                    Ok(read::Chunk {
                        events: rel.chunk.iter().filter_map(raw).collect(),
                        next,
                    })
                }
            }
        }
    })
    .await
}

/// Answer every text message from someone else with an agent run (or, in
/// a `trigger = "schedule"` room, only record it).
async fn on_message(
    ev: OriginalSyncRoomMessageEvent,
    room: Room,
    client: Client,
    Ctx(agent): Ctx<Arc<Agent<ClaudeCode>>>,
    Ctx(tooling): Ctx<Arc<Tooling>>,
) {
    if room.state() != RoomState::Joined || Some(ev.sender.as_ref()) == client.user_id() {
        return;
    }
    if let MessageType::VerificationRequest(_) = &ev.content.msgtype {
        // Element verifies a *user* through a message in a room. The bot only
        // sees rooms it was invited to by allowed people, so no extra check.
        if let Some(request) = client
            .encryption()
            .get_verification_request(&ev.sender, &ev.event_id)
            .await
        {
            tokio::spawn(handle_verification_request(request));
        }
        return;
    }
    let MessageType::Text(text) = &ev.content.msgtype else {
        return;
    };
    println!("[{}] {}: {}", room.room_id(), ev.sender, text.body);
    let body = text.body.clone();
    // A Matrix thread is its own conversation (own session); the main
    // timeline is another.
    let thread = match &ev.content.relates_to {
        Some(Relation::Thread(t)) => Some(t.event_id.clone()),
        _ => None,
    };
    let reply_to = ev.event_id.clone();
    let command = Command::parse(&body);

    // Runs take seconds; don't hold up the sync loop meanwhile.
    tokio::spawn(async move {
        // A room with `threads` answers a main-timeline message in a new
        // thread rooted at that message (but commands stay where they are).
        let thread = match thread {
            None if command.is_none() && agent.starts_threads(room.room_id()).await => {
                Some(reply_to.clone())
            }
            t => t,
        };
        let thread_id = thread.as_ref().map(|t| t.as_str().to_owned());
        if let Err(e) = agent
            .record(
                room.room_id(),
                thread_id.as_deref(),
                ev.sender.as_str(),
                &body,
            )
            .await
        {
            eprintln!("could not write transcript: {e}");
        }

        if let Some(command) = command {
            let text = run_command(
                &agent,
                &tooling,
                room.room_id(),
                thread_id.as_deref(),
                command,
            )
            .await;
            send_text(&room, text, thread.as_ref(), Some(&reply_to)).await;
            return;
        }
        if !agent.runs_on_messages(room.room_id()).await {
            return;
        }

        let sender = match room.get_member_no_sync(&ev.sender).await {
            Ok(Some(member)) => member.name().to_owned(),
            _ => ev.sender.localpart().to_owned(),
        };
        let room_id = room.room_id().to_owned();
        let t = tooling.clone();
        tooling
            .runs
            .track(
                room_id,
                thread_id,
                answer(
                    client,
                    agent,
                    t,
                    room,
                    thread,
                    Some(reply_to),
                    sender,
                    body,
                    0,
                ),
            )
            .await;
    });
}

/// A `!` command: handled by parley itself, without an agent.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// `!new`: forget this conversation's session.
    New,
    /// `!context [mode]`: show or switch the room's context mode.
    Context(Option<String>),
    /// `!stop`: kill this conversation's run.
    Stop,
    /// `!schedules`: list the room's schedules.
    Schedules,
    /// `!cancel <id>`: end a schedule.
    Cancel(String),
}

impl Command {
    fn parse(body: &str) -> Option<Self> {
        let body = body.trim();
        let (word, rest) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
        let rest = rest.trim();
        let arg = Some(rest.to_owned()).filter(|r| !r.is_empty());
        match word.to_ascii_lowercase().as_str() {
            "!new" if arg.is_none() => Some(Self::New),
            "!context" => Some(Self::Context(arg)),
            "!stop" if arg.is_none() => Some(Self::Stop),
            "!schedules" if arg.is_none() => Some(Self::Schedules),
            "!cancel" => Some(Self::Cancel(rest.to_owned())),
            _ => None,
        }
    }
}

/// What parley answers to `command`, sent in `room`'s conversation `thread`.
async fn run_command(
    agent: &Agent<ClaudeCode>,
    tooling: &Tooling,
    room: &RoomId,
    thread: Option<&str>,
    command: Command,
) -> String {
    match command {
        Command::New => match agent.reset(room, thread).await {
            Ok(Some(old)) => match old.messages {
                Some(1) => "_New session (the old one had 1 message): the next message starts fresh._".to_owned(),
                Some(n) => format!("_New session (the old one had {n} messages): the next message starts fresh._"),
                None => "_New session (the old one's message count is unknown): the next message starts fresh._".to_owned(),
            },
            Ok(None) => "_There was no session yet; the next message starts fresh._".to_owned(),
            Err(e) => format!("_Could not reset the session: {e}_"),
        },
        Command::Context(arg) => match agent.context(room, arg.as_deref()).await {
            Ok(text) => text,
            Err(e) => format!("_{e}_\n\n{}", memory::Mode::HELP),
        },
        Command::Stop => match tooling.runs.stop(room, thread) {
            0 => "_Nothing is running here._".to_owned(),
            1 => "_Stopped the run._".to_owned(),
            n => format!("_Stopped {n} runs._"),
        },
        Command::Schedules => match tooling.schedules.list(room.as_str()) {
            Ok(jobs) if jobs.is_empty() => "_No schedules in this room._".to_owned(),
            Ok(jobs) => jobs
                .iter()
                .map(|j| format!("- {}", j.describe()))
                .collect::<Vec<_>>()
                .join("\n"),
            Err(e) => format!("_Could not list the schedules: {e}_"),
        },
        Command::Cancel(id) => match tooling.schedules.cancel(room.as_str(), &id) {
            Ok(_) => format!("_Cancelled `{id}`; it fires no more._"),
            Err(e) => format!("_Could not cancel: {e}_"),
        },
    }
}

/// Post `text` in `room`'s conversation `thread`, after `reply_to`.
async fn send_text(
    room: &Room,
    text: String,
    thread: Option<&OwnedEventId>,
    reply_to: Option<&OwnedEventId>,
) {
    let content = match post_content(text, &post::Target::Here, thread, reply_to) {
        Ok((c, _)) => c,
        Err(e) => {
            eprintln!("bad post: {e}");
            return;
        }
    };
    if let Err(e) = room.send(content).await {
        eprintln!("send failed: {e}");
    }
}

/// One agent run in `room`'s conversation `thread`: for the message
/// `reply_to` from `sender`, or (with none) a schedule's.
#[allow(clippy::too_many_arguments)]
async fn answer(
    client: Client,
    agent: Arc<Agent<ClaudeCode>>,
    tooling: Arc<Tooling>,
    room: Room,
    thread: Option<OwnedEventId>,
    reply_to: Option<OwnedEventId>,
    sender: String,
    body: String,
    depth: u32,
) {
    let thread_id = thread.as_ref().map(|t| t.as_str().to_owned());
    let me = client.user_id().map(ToOwned::to_owned);
    let typing = Typing::start(&room);
    // Everyone who is or may soon be in the room, except the bot.
    let members: Vec<OwnedUserId> = match room.members(RoomMemberships::ACTIVE).await {
        Ok(list) => list
            .iter()
            .map(|m| m.user_id().to_owned())
            .filter(|id| Some(id) != me.as_ref())
            .collect(),
        Err(e) => {
            eprintln!("could not list members: {e}");
            let _ = room
                .send(RoomMessageEventContent::text_plain(
                    "Could not check who is in this room; not running.",
                ))
                .await;
            return;
        }
    };
    // A `reply = "tool"` room posts (and reads) with tools while it runs.
    let ticket = if agent.replies_with_tool(room.room_id()).await {
        match tool_turn(&client, &agent, &tooling, &room, &thread, &reply_to, depth)
            .await
            .and_then(|turn| tooling.posts.register(turn))
        {
            Ok(t) => Some(t),
            Err(e) => {
                eprintln!("could not set up the tools, answering as text: {e:#}");
                None
            }
        }
    } else {
        None
    };
    let mcp_config = ticket
        .as_ref()
        .map(|t| post::mcp_config(&tooling.exe, &tooling.socket, &t.token));
    let answer = agent
        .reply_with(
            room.room_id(),
            thread_id.as_deref(),
            &sender,
            &body,
            &members,
            mcp_config.as_deref(),
        )
        .await;
    drop(typing);
    let _ = room.typing_notice(false).await;

    match ticket {
        Some(t) if t.count() > 0 => {
            if !post::is_marker(&answer) {
                println!(
                    "[{}] posted {}; final text not posted: {answer}",
                    room.room_id(),
                    t.count()
                );
            }
        }
        Some(_) if post::is_marker(&answer) => {
            // Silence, made visible (a schedule's run has nothing to mark).
            if let Some(r) = &reply_to {
                let mark = ReactionEventContent::new(Annotation::new(r.clone(), "✓".into()));
                if let Err(e) = room.send(mark).await {
                    eprintln!("could not react: {e}");
                }
            }
        }
        // Text mode, or a tool-mode run that posted nothing but said
        // something (an error, or a model that forgot the tool).
        _ => {
            if let Err(e) = agent
                .record(room.room_id(), thread_id.as_deref(), "agent", &answer)
                .await
            {
                eprintln!("could not write transcript: {e}");
            }
            send_text(&room, answer, thread.as_ref(), reply_to.as_ref()).await;
        }
    }
    // A `digest` room updates its notes, a `layered` one its summaries, now that the answer is out.
    agent.reflect(room.room_id()).await;

    // An agent working on parley itself asks for a restart by creating
    // `restart` in the data dir; exit only after its reply is out, and
    // let systemd (Restart=always) start the new build.
    if agent.take_restart_request() {
        println!("restart requested by an agent; exiting");
        std::process::exit(0);
    }
}

/// Fire due schedules, forever: each starts a run in its room and thread
/// with the schedule's prompt and handoff as the message.
async fn scheduler(client: Client, agent: Arc<Agent<ClaudeCode>>, tooling: Arc<Tooling>) {
    loop {
        let now = chrono::Utc::now();
        for job in tooling.schedules.take_due(now) {
            tokio::spawn(fire(client.clone(), agent.clone(), tooling.clone(), job));
        }
        // Wake for the soonest schedule, a new one, or every 30 s at most
        // (schedules can also be added or edited by hand).
        let wait = tooling
            .schedules
            .soonest()
            .and_then(|t| (t - chrono::Utc::now()).to_std().ok())
            .unwrap_or(Duration::ZERO)
            .clamp(Duration::from_secs(1), Duration::from_secs(30));
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            _ = tooling.schedules.wake.notified() => {}
        }
    }
}

async fn fire(
    client: Client,
    agent: Arc<Agent<ClaudeCode>>,
    tooling: Arc<Tooling>,
    job: schedule::Job,
) {
    let room = match joined(&client, &job.room) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("schedule {} cannot fire: {e}", job.id);
            return;
        }
    };
    let thread = match job.thread.as_deref().map(OwnedEventId::try_from) {
        None => None,
        Some(Ok(t)) => Some(t),
        Some(Err(e)) => {
            eprintln!("schedule {} has a bad thread: {e}", job.id);
            return;
        }
    };
    println!(
        "[{}] schedule {} fires (run {})",
        job.room, job.id, job.fired
    );
    let board = tooling
        .schedules
        .board(&job.room, &job.id)
        .ok()
        .and_then(|b| std::fs::read_to_string(b).ok());
    let body = schedule::fire_message(&job, board.as_deref());
    let who = format!("schedule {}", job.id);
    // Schedules this run makes are one generation further.
    let depth = job.depth + 1;
    if let Err(e) = agent
        .record(room.room_id(), job.thread.as_deref(), &who, &body)
        .await
    {
        eprintln!("could not write transcript: {e}");
    }
    let t = tooling.clone();
    tooling
        .runs
        .track(
            room.room_id().to_owned(),
            job.thread.clone(),
            answer(client, agent, t, room, thread, None, who, body, depth),
        )
        .await;
}

/// Accept a verification request and drive it through emoji (SAS) comparison.
async fn handle_verification_request(request: VerificationRequest) {
    println!("verification requested by {}", request.other_user_id());
    if let Err(e) = request.accept().await {
        eprintln!("could not accept verification: {e}");
        return;
    }
    let mut changes = request.changes();
    while let Some(state) = changes.next().await {
        match state {
            VerificationRequestState::Transitioned {
                verification: Verification::SasV1(sas),
            } => {
                handle_sas(sas).await;
                break;
            }
            VerificationRequestState::Done | VerificationRequestState::Cancelled(_) => break,
            _ => {}
        }
    }
}

async fn handle_sas(sas: SasVerification) {
    if let Err(e) = sas.accept().await {
        eprintln!("could not accept SAS: {e}");
        return;
    }
    let mut changes = sas.changes();
    while let Some(state) = changes.next().await {
        match state {
            SasState::KeysExchanged {
                emojis: Some(emojis),
                ..
            } => {
                println!(
                    "\nVERIFY with {} {}: compare with Element, press \"They match\" only if equal:\n{}\n",
                    sas.other_device().user_id(),
                    sas.other_device().device_id(),
                    format_emojis(emojis.emojis)
                );
                if let Err(e) = sas.confirm().await {
                    eprintln!("could not confirm: {e}");
                }
            }
            SasState::Done { .. } => {
                println!(
                    "verified {} {}",
                    sas.other_device().user_id(),
                    sas.other_device().device_id()
                );
                break;
            }
            SasState::Cancelled(info) => {
                println!("verification cancelled: {}", info.reason());
                break;
            }
            _ => {}
        }
    }
}

fn random_hex(bytes: usize) -> Result<String> {
    use std::io::Read;
    let mut buf = vec![0u8; bytes];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

/// Write a file readable only by the owner (it holds an access token).
fn write_private(path: &PathBuf, contents: &str) -> Result<()> {
    use std::{fs::OpenOptions, io::Write, os::unix::fs::OpenOptionsExt};
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    f.write_all(contents.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_backoff_doubles_up_to_eight_seconds() {
        let waits: Vec<u64> = (1..=7).map(|n| sync_backoff(n).as_secs()).collect();
        assert_eq!(waits, [1, 2, 4, 8, 8, 8, 8]);
        assert_eq!(sync_backoff(u32::MAX), Duration::from_secs(8));
    }

    #[test]
    fn parse_allowed_reads_the_default_list() {
        let allowed =
            parse_allowed("@kas:chat.lab.artintel.ai,@camille:chat.lab.artintel.ai").unwrap();
        let ids: Vec<&str> = allowed.iter().map(|u| u.as_str()).collect();
        assert_eq!(
            ids,
            ["@kas:chat.lab.artintel.ai", "@camille:chat.lab.artintel.ai"]
        );
    }

    #[test]
    fn parse_allowed_trims_whitespace() {
        let allowed = parse_allowed(" @a:example.org ,\t@b:example.org\n").unwrap();
        let ids: Vec<&str> = allowed.iter().map(|u| u.as_str()).collect();
        assert_eq!(ids, ["@a:example.org", "@b:example.org"]);
    }

    #[test]
    fn parse_allowed_single_id() {
        let allowed = parse_allowed("@a:example.org").unwrap();
        assert_eq!(allowed.len(), 1);
    }

    #[test]
    fn parse_allowed_rejects_a_bad_id_and_names_it() {
        let err = parse_allowed("@a:example.org,kas").unwrap_err();
        assert!(format!("{err}").contains("\"kas\""), "{err}");
    }

    #[test]
    fn parse_allowed_rejects_empty_entries() {
        // An empty value or a trailing comma is an error, not "nobody allowed".
        assert!(parse_allowed("").is_err());
        assert!(parse_allowed("@a:example.org,").is_err());
    }

    #[test]
    fn commands() {
        assert_eq!(Command::parse(" !new "), Some(Command::New));
        assert_eq!(Command::parse("!new please"), None);
        assert_eq!(Command::parse("!context"), Some(Command::Context(None)));
        assert_eq!(
            Command::parse("!context layered 10"),
            Some(Command::Context(Some("layered 10".into())))
        );
        assert_eq!(Command::parse("!contextual"), None);
        assert_eq!(Command::parse("!STOP"), Some(Command::Stop));
        assert_eq!(Command::parse("!schedules"), Some(Command::Schedules));
        assert_eq!(
            Command::parse("!cancel ab12"),
            Some(Command::Cancel("ab12".into()))
        );
        assert_eq!(Command::parse("stop"), None);
    }

    #[test]
    fn posts_follow_their_message_or_the_root() {
        let root: OwnedEventId = "$root".try_into().unwrap();
        let msg: OwnedEventId = "$msg".try_into().unwrap();
        let latest = |c: &RoomMessageEventContent| match &c.relates_to {
            Some(Relation::Thread(t)) => Some((
                t.event_id.to_string(),
                t.in_reply_to.as_ref().map(|r| r.event_id.to_string()),
            )),
            _ => None,
        };
        let (c, t) =
            post_content("x".into(), &post::Target::Here, Some(&root), Some(&msg)).unwrap();
        assert_eq!(t.as_deref(), Some("$root"));
        assert_eq!(latest(&c), Some(("$root".into(), Some("$msg".into()))));
        // A schedule's run has no message: it follows the root.
        let (c, _) = post_content("x".into(), &post::Target::Here, Some(&root), None).unwrap();
        assert_eq!(latest(&c), Some(("$root".into(), Some("$root".into()))));
        let (c, t) =
            post_content("x".into(), &post::Target::Main, Some(&root), Some(&msg)).unwrap();
        assert_eq!((latest(&c), t), (None, None));
    }

    #[tokio::test]
    async fn stop_kills_only_its_conversation() {
        let runs = Arc::new(Runs::default());
        let room: OwnedRoomId = "!r:x".try_into().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let r = runs.clone();
        let rm = room.clone();
        let a = tokio::spawn(async move {
            r.track(rm, Some("$a".into()), async move {
                tokio::time::sleep(Duration::from_secs(60)).await;
            })
            .await
        });
        let r = runs.clone();
        let rm = room.clone();
        let b = tokio::spawn(async move {
            r.track(rm, None, async move {
                let _ = rx.await;
            })
            .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(runs.stop(&room, Some("$b")), 0);
        assert_eq!(runs.stop(&room, Some("$a")), 1);
        tokio::time::timeout(Duration::from_secs(1), a)
            .await
            .unwrap()
            .unwrap();
        tx.send(()).unwrap();
        b.await.unwrap();
        assert!(runs.live.lock().unwrap().is_empty());
    }
}
