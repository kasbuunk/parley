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

mod agent;
mod breaker;

use std::{
    env,
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
        OwnedUserId,
        api::{client::uiaa, error::ErrorKind},
        events::key::verification::request::ToDeviceKeyVerificationRequestEvent,
        events::relation::Thread,
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
    let agent = Agent::new(
        ClaudeCode {
            program: config.claude.clone(),
        },
        config.data_dir.join("rooms"),
    );
    client.add_event_handler_context(Arc::new(agent));
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

/// Answer every text message from someone else with an agent run.
async fn on_message(
    ev: OriginalSyncRoomMessageEvent,
    room: Room,
    client: Client,
    Ctx(agent): Ctx<Arc<Agent<ClaudeCode>>>,
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
    let me = client.user_id().map(ToOwned::to_owned);
    let is_new = body.trim().eq_ignore_ascii_case("!new");

    // Runs take seconds; don't hold up the sync loop meanwhile.
    tokio::spawn(async move {
        // A room with `threads` answers a main-timeline message in a new
        // thread rooted at that message (but `!new` stays where it is).
        let thread = match thread {
            None if !is_new && agent.starts_threads(room.room_id()).await => Some(reply_to.clone()),
            t => t,
        };
        let send = |text: String| {
            let mut content = RoomMessageEventContent::text_markdown(text);
            if let Some(root) = &thread {
                content.relates_to = Some(Relation::Thread(Thread::plain(
                    root.clone(),
                    reply_to.clone(),
                )));
            }
            let room = room.clone();
            async move {
                if let Err(e) = room.send(content).await {
                    eprintln!("send failed: {e}");
                }
            }
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

        if is_new {
            let text = match agent.reset(room.room_id(), thread_id.as_deref()).await {
                Ok(true) => "_New session: the next message starts fresh._".to_owned(),
                Ok(false) => {
                    "_There was no session yet; the next message starts fresh._".to_owned()
                }
                Err(e) => format!("_Could not reset the session: {e}_"),
            };
            send(text).await;
            return;
        }

        let sender = match room.get_member_no_sync(&ev.sender).await {
            Ok(Some(member)) => member.name().to_owned(),
            _ => ev.sender.localpart().to_owned(),
        };

        // Show "typing…" until the answer is there (a notice lasts ~4 s).
        let typing_room = room.clone();
        let typing = tokio::spawn(async move {
            loop {
                let _ = typing_room.typing_notice(true).await;
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        });
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
                typing.abort();
                return;
            }
        };
        let answer = agent
            .reply(
                room.room_id(),
                thread_id.as_deref(),
                &sender,
                &body,
                &members,
            )
            .await;
        typing.abort();
        let _ = room.typing_notice(false).await;

        if let Err(e) = agent
            .record(room.room_id(), thread_id.as_deref(), "agent", &answer)
            .await
        {
            eprintln!("could not write transcript: {e}");
        }
        send(answer).await;

        // An agent working on parley itself asks for a restart by creating
        // `restart` in the data dir; exit only after its reply is out, and
        // let systemd (Restart=always) start the new build.
        if agent.take_restart_request() {
            println!("restart requested by an agent; exiting");
            std::process::exit(0);
        }
    });
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
}
