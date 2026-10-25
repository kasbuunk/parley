//! parley — a Matrix bot that lets people talk to home-lab agents.
//!
//! Step 1 is an end-to-end encrypted echo: the bot joins rooms it is invited
//! to (only by people on the allow list) and answers every text message with
//! the same text. The bot is an ordinary room member with its own device, so
//! it receives the room keys like any other member and can read encrypted
//! rooms while the server cannot.
//!
//! Configuration (environment):
//!   PARLEY_HOMESERVER     default https://chat.lab.artintel.ai
//!   PARLEY_PASSWORD_FILE  file whose 1st line is the user ID, 2nd the password
//!                         (only read on first start; afterwards the saved
//!                         session is reused)
//!   PARLEY_ALLOWED        comma-separated user IDs allowed to invite the bot
//!   PARLEY_DATA_DIR       default <data dir>/parley
//!
//! Verification: start "Verify" on @agent in Element. The bot accepts
//! requests from people on the allow list and prints the emoji to its log.
//! The bot confirms its side by itself; *you* are the check: compare the log
//! with what Element shows, and only press "They match" if they do.
//!
//! State lives in PARLEY_DATA_DIR: `session.json` (access token, device ID)
//! and `store/` (encrypted SQLite with the bot's encryption keys). Losing the
//! store means losing the keys: delete both and log in again as a new device.

use std::{env, path::PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use matrix_sdk::{
    Client, Room, RoomState,
    authentication::matrix::MatrixSession,
    config::SyncSettings,
    encryption::{
        CrossSigningResetAuthType,
        verification::{
            SasState, SasVerification, Verification, VerificationRequest,
            VerificationRequestState, format_emojis,
        },
    },
    ruma::{
        OwnedUserId,
        api::client::uiaa,
        events::key::verification::request::ToDeviceKeyVerificationRequestEvent,
        events::room::{
            member::StrippedRoomMemberEvent,
            message::{MessageType, OriginalSyncRoomMessageEvent, RoomMessageEventContent},
        },
    },
};
use serde::{Deserialize, Serialize};

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
}

impl Config {
    fn from_env() -> Result<Self> {
        let allowed = env::var("PARLEY_ALLOWED")
            .unwrap_or_else(|_| {
                "@kas:chat.lab.artintel.ai,@camille:chat.lab.artintel.ai".into()
            })
            .split(',')
            .map(|s| s.trim().try_into().with_context(|| format!("bad user ID {s:?}")))
            .collect::<Result<_>>()?;
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
        })
    }
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

    // Catch up once without handlers, so old messages are not echoed again.
    let response = client.sync_once(SyncSettings::default()).await?;

    let allowed = config.allowed.clone();
    client.add_event_handler(move |ev: StrippedRoomMemberEvent, client: Client, room: Room| {
        let allowed = allowed.clone();
        async move { on_invite(ev, client, room, &allowed).await }
    });
    client.add_event_handler(on_message);
    let allowed = config.allowed.clone();
    client.add_event_handler(move |ev: ToDeviceKeyVerificationRequestEvent, client: Client| {
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
    });

    println!("listening");
    client.sync(SyncSettings::default().token(response.next_batch)).await?;
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
        bail!("{} must hold the user ID and the password on two lines", path.display());
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
        matrix: client.matrix_auth().session().context("logged in but no session")?,
    };
    write_private(session_file, &serde_json::to_string_pretty(&saved)?)?;
    println!("logged in as {user}; session saved to {}", session_file.display());

    // A first sync uploads the device keys; then create the cross-signing
    // identity. The server asks for the password again (user-interactive auth).
    client.sync_once(SyncSettings::default()).await?;
    if let Some(handle) = client.encryption().reset_cross_signing().await? {
        if let CrossSigningResetAuthType::Uiaa(uiaa) = handle.auth_type() {
            let mut auth = uiaa::Password::new(client.user_id().unwrap().to_owned().into(), password.into());
            auth.session = uiaa.session.clone();
            handle.auth(Some(uiaa::AuthData::Password(auth))).await?;
        }
    }
    println!("cross-signing identity created");
    Ok(client)
}

/// Join rooms we are invited to, but only when the inviter is on the allow list.
async fn on_invite(ev: StrippedRoomMemberEvent, client: Client, room: Room, allowed: &[OwnedUserId]) {
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

/// Echo every text message from someone else.
async fn on_message(ev: OriginalSyncRoomMessageEvent, room: Room, client: Client) {
    if room.state() != RoomState::Joined || Some(ev.sender.as_ref()) == client.user_id() {
        return;
    }
    if let MessageType::VerificationRequest(_) = &ev.content.msgtype {
        // Element verifies a *user* through a message in a room. The bot only
        // sees rooms it was invited to by allowed people, so no extra check.
        if let Some(request) =
            client.encryption().get_verification_request(&ev.sender, &ev.event_id).await
        {
            tokio::spawn(handle_verification_request(request));
        }
        return;
    }
    let MessageType::Text(text) = &ev.content.msgtype else { return };
    println!("[{}] {}: {}", room.room_id(), ev.sender, text.body);
    let reply = RoomMessageEventContent::text_plain(format!("echo: {}", text.body));
    if let Err(e) = room.send(reply).await {
        eprintln!("send failed: {e}");
    }
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
            VerificationRequestState::Transitioned { verification: Verification::SasV1(sas) } => {
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
            SasState::KeysExchanged { emojis: Some(emojis), .. } => {
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
                println!("verified {} {}", sas.other_device().user_id(), sas.other_device().device_id());
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
    let mut f = OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
    f.write_all(contents.as_bytes())?;
    Ok(())
}
