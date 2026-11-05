//! Turning a room message into an agent reply.
//!
//! Every room has a folder `rooms/<room id>/` holding:
//! - `SYSTEM.md`: the system prompt (a default is written on first use);
//! - `room.toml`: optional profile: where the agent runs, which tools and
//!   settings it gets, who may be in the room, whether it keeps a session.
//!   Without it a room is locked down: no tools, no settings, no memory.
//! - `session`: the harness session ID for the room's main timeline, when
//!   the profile says `resume`; `threads/<root event id>.session` for each
//!   Matrix thread. A thread is its own conversation, so unrelated topics
//!   don't share one context. The file holds the ID, then on a second
//!   line how many messages the session has had. `!new` clears the
//!   session it is sent in and says how many messages it had.
//!   With `threads`, every main-timeline message starts a new thread.
//!
//! - `context`: the context mode `!context` set, overriding room.toml;
//!   `MEMORY.md`: the room's notes in `digest` mode (see `memory.rs`);
//!   `summaries/`: day and week summaries in `layered` mode (`layered.rs`),
//!   unless room.toml puts them elsewhere (`summaries`, e.g. in the vault).
//! - `transcript.jsonl`: every text message in the room and every reply,
//!   one JSON object per line (`ts`, `sender`, `thread`, `body`). Not sent
//!   to the model as a whole; `window` and `digest` modes quote its end, and
//!   an agent with file tools can read or grep it to look back.
//!
//! Each message (prefixed with the sender's name) is one user turn. The
//! model and harness sit behind [`Harness`]; Claude Code is the first
//! implementation, Pi can be a second.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use matrix_sdk::ruma::{OwnedRoomId, OwnedUserId, RoomId};
use serde::Deserialize;
use tokio::{process::Command, sync::Mutex};

use crate::{
    breaker::{Breaker, State},
    layered,
    memory::{self, Mode},
};

const DEFAULT_SYSTEM_PROMPT: &str = "\
You are the agent in a private Matrix chat on Kas's home lab. People write to \
you from their phones and laptops. Each message starts with the sender's name \
in brackets, for example `[Kas] …`.

You only see the latest message, not the earlier conversation. You have no \
tools and no access to files, the lab or anyone's data; say so plainly when a \
question needs them.

Answer briefly, in the language of the message. Your answer is posted to the \
room as Markdown.
";

/// A room's `room.toml`. Every field is optional; the defaults are the
/// locked-down room.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Profile {
    /// Where the agent runs. Default: the room's own folder.
    pub workdir: Option<PathBuf>,
    /// Built-in tools: "" for none, "default" for all, or a list like "Read,Grep".
    pub tools: String,
    /// Claude Code setting sources to load: "user", "project", "local".
    /// With "project", a repo's CLAUDE.md and .claude/ apply.
    pub settings: Vec<String>,
    /// Load the user's MCP servers too.
    pub mcp: bool,
    /// For example "bypassPermissions": tool calls run without asking.
    pub permission_mode: Option<String>,
    /// If set, the agent only runs while every member of the room is listed.
    pub members: Option<Vec<OwnedUserId>>,
    /// Keep one session per room, so the agent remembers the conversation.
    /// Shorthand for `context = "session"`.
    pub resume: bool,
    /// What a turn remembers of earlier ones (see `memory.rs`). Default:
    /// `session` with `resume`, else `fresh`. `!context` overrides it.
    pub context: Option<Mode>,
    /// Answer a main-timeline message in a new thread started from it, so
    /// every topic gets its own thread and session.
    pub threads: bool,
    /// Whether SYSTEM.md replaces the harness's own system prompt or is
    /// appended to it.
    pub prompt: PromptMode,
    pub model: Option<String>,
    pub timeout_secs: u64,
    /// A readable name for the room, used in the names of its summary
    /// notes (`parley 2026-10-25.md`) and their tags.
    pub name: Option<String>,
    /// Folder for the `layered` mode's summary notes. Default: the room's
    /// `summaries/`. Kas's vault: `…/life/ai/rooms/<name>`.
    pub summaries: Option<PathBuf>,
    /// How the agent speaks: `text` posts the run's final answer; `tool`
    /// gives it a `post` tool instead (see `post.rs`).
    pub reply: ReplyMode,
    /// Other rooms (or parts of them) the `read` and `post` tools may use,
    /// as `[[access]]` tables (see `access.rs`). The room itself is always open.
    pub access: Vec<crate::access::Grant>,
    /// What starts a run: every message (`message`, the default), or only
    /// schedules (`schedule`; messages are recorded, and commands work).
    pub trigger: Trigger,
    /// How many generations of schedules may make schedules (see
    /// `schedule::Job::depth`). Default 10.
    pub max_depth: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Trigger {
    #[default]
    Message,
    Schedule,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReplyMode {
    #[default]
    Text,
    Tool,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            workdir: None,
            tools: String::new(),
            settings: Vec::new(),
            mcp: false,
            permission_mode: None,
            members: None,
            resume: false,
            context: None,
            threads: false,
            prompt: PromptMode::Replace,
            model: None,
            timeout_secs: 300,
            name: None,
            summaries: None,
            reply: ReplyMode::Text,
            access: Vec::new(),
            trigger: Trigger::Message,
            max_depth: 10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    Replace,
    Append,
}

/// One agent run.
pub struct Run<'a> {
    pub profile: &'a Profile,
    pub system_prompt: &'a str,
    pub workdir: &'a Path,
    pub input: &'a str,
    /// Session to continue, if any.
    pub session: Option<&'a str>,
    /// Whether the harness should keep the session for a later turn.
    pub persist: bool,
    /// Extra MCP servers for this run (`--mcp-config` JSON), e.g. parley's
    /// `post` tool.
    pub mcp_config: Option<&'a str>,
}

pub struct Output {
    pub text: String,
    /// Session ID to continue next time, if the harness keeps sessions.
    pub session: Option<String>,
}

/// A session file: the harness session ID and how many messages it has had.
#[derive(Debug, Default, PartialEq)]
pub struct Saved {
    pub id: String,
    /// `None` for files written before parley counted messages.
    pub messages: Option<u64>,
}

impl Saved {
    async fn read(file: &Path) -> Option<Self> {
        let text = tokio::fs::read_to_string(file).await.ok()?;
        let mut lines = text.lines();
        let id = lines.next()?.trim().to_owned();
        if id.is_empty() {
            return None;
        }
        let messages = lines.next().and_then(|l| l.trim().parse().ok());
        Some(Self { id, messages })
    }
}

impl std::fmt::Display for Saved {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.id)?;
        if let Some(n) = self.messages {
            writeln!(f, "{n}")?;
        }
        Ok(())
    }
}

pub trait Harness: Send + Sync {
    fn run(&self, run: &Run<'_>) -> impl Future<Output = Result<Output>> + Send;
}

/// Claude Code in headless mode, on the logged-in subscription. It never
/// asks questions: whatever the profile does not allow is refused.
pub struct ClaudeCode {
    pub program: String,
}

impl ClaudeCode {
    fn args(run: &Run<'_>) -> Vec<String> {
        let p = run.profile;
        let mut args: Vec<String> = vec!["-p".into(), run.input.into()];
        args.push(match p.prompt {
            PromptMode::Replace => "--system-prompt".into(),
            PromptMode::Append => "--append-system-prompt".into(),
        });
        args.push(run.system_prompt.into());
        args.extend(["--tools".into(), p.tools.clone()]);
        if !p.mcp {
            args.push("--strict-mcp-config".into());
        }
        if let Some(config) = run.mcp_config {
            args.extend(["--mcp-config".into(), config.into()]);
            args.extend(["--allowedTools".into(), "mcp__parley__post".into()]);
        }
        args.extend(["--setting-sources".into(), p.settings.join(",")]);
        if p.settings.is_empty() {
            args.push("--disable-slash-commands".into());
        }
        if !run.persist {
            args.push("--no-session-persistence".into());
        }
        if let Some(mode) = &p.permission_mode {
            args.extend(["--permission-mode".into(), mode.clone()]);
        }
        if let Some(model) = &p.model {
            args.extend(["--model".into(), model.clone()]);
        }
        if let Some(session) = run.session {
            args.extend(["--resume".into(), session.into()]);
        }
        args.extend(["--output-format".into(), "json".into()]);
        args
    }
}

impl Harness for ClaudeCode {
    async fn run(&self, run: &Run<'_>) -> Result<Output> {
        let timeout = Duration::from_secs(run.profile.timeout_secs);
        let child = Command::new(&self.program)
            .args(Self::args(run))
            .current_dir(run.workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("could not start {}", self.program))?;

        let output = tokio::time::timeout(timeout, child.wait_with_output())
            .await
            .with_context(|| format!("no answer within {}s", timeout.as_secs()))??;

        #[derive(Deserialize)]
        struct Reply {
            result: Option<String>,
            #[serde(default)]
            is_error: bool,
            session_id: Option<String>,
        }
        let reply: Reply = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "claude exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        })?;
        match reply.result {
            Some(text) if !reply.is_error => Ok(Output {
                text,
                session: reply.session_id,
            }),
            other => bail!("claude reported an error: {}", other.unwrap_or_default()),
        }
    }
}

/// A conversation: a room's main timeline (`None`), or one of its threads.
pub type Conv = (OwnedRoomId, Option<String>);

pub struct Agent<H> {
    harness: H,
    rooms_dir: PathBuf,
    breaker: std::sync::Mutex<Breaker>,
    /// One lock per conversation (a room's main timeline, or one of its
    /// threads), so its turns come one after the other; conversations run
    /// side by side.
    conv_locks: std::sync::Mutex<HashMap<Conv, Arc<Mutex<()>>>>,
    /// One lock per room, for what the room's conversations share: its
    /// context mode and its memory notes.
    room_locks: std::sync::Mutex<HashMap<OwnedRoomId, Arc<Mutex<()>>>>,
    /// Per room, the last exchange a `digest` room still has to take into
    /// its memory (see [`Agent::reflect`]).
    pending_digests: std::sync::Mutex<HashMap<OwnedRoomId, Exchange>>,
    /// Rooms whose summaries are being written, so two replies in a row
    /// do not write the same ones twice.
    summarising: std::sync::Mutex<std::collections::HashSet<OwnedRoomId>>,
}

/// A message a turn answers: several when they came in while the
/// conversation's previous turn ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Said {
    pub sender: String,
    pub body: String,
}

impl Said {
    pub fn new(sender: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            sender: sender.into(),
            body: body.into(),
        }
    }
}

/// One message (or several) and its answer.
struct Exchange {
    sender: String,
    body: String,
    answer: String,
}

/// Where a room's context mode comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Command,
    Profile,
    Default,
}

impl<H: Harness> Agent<H> {
    pub fn new(harness: H, rooms_dir: PathBuf) -> Self {
        Self {
            harness,
            rooms_dir,
            breaker: std::sync::Mutex::new(Breaker::new(3, Duration::from_secs(300))),
            conv_locks: Default::default(),
            room_locks: Default::default(),
            pending_digests: Default::default(),
            summarising: Default::default(),
        }
    }

    /// The reply to post for `body`, sent by `sender` in `room_id` (in the
    /// thread rooted at `thread`, if any), whose human members (everyone but
    /// the bot) are `members`.
    #[cfg(test)]
    pub async fn reply(
        &self,
        room_id: &RoomId,
        thread: Option<&str>,
        sender: &str,
        body: &str,
        members: &[OwnedUserId],
    ) -> String {
        self.reply_with(room_id, thread, &[Said::new(sender, body)], members, None)
            .await
    }

    /// [`Agent::reply`] to one or more messages, giving the run the extra
    /// MCP servers in `mcp_config`; with one, the system prompt tells the
    /// agent to post with the `post` tool.
    pub async fn reply_with(
        &self,
        room_id: &RoomId,
        thread: Option<&str>,
        said: &[Said],
        members: &[OwnedUserId],
        mcp_config: Option<&str>,
    ) -> String {
        let lock = self
            .conv_locks
            .lock()
            .unwrap()
            .entry((room_id.to_owned(), thread.map(str::to_owned)))
            .or_default()
            .clone();
        let _turn = lock.lock().await;

        if let State::Open(left) = self.breaker.lock().unwrap().state() {
            return format!(
                "_Agent paused after repeated failures; trying again in {} min._",
                left.as_secs().div_ceil(60)
            );
        }

        match self.run(room_id, thread, said, members, mcp_config).await {
            Ok(text) => {
                self.breaker.lock().unwrap().success();
                text
            }
            Err(e) => {
                eprintln!("[{room_id}] agent failed: {e:#}");
                let opened = self.breaker.lock().unwrap().failure();
                if opened {
                    eprintln!("agent circuit open: pausing agent runs");
                }
                format!("_Agent error: {e}_")
            }
        }
    }

    fn room_lock(&self, room_id: &RoomId) -> Arc<Mutex<()>> {
        self.room_locks
            .lock()
            .unwrap()
            .entry(room_id.to_owned())
            .or_default()
            .clone()
    }

    /// The room's context mode and where it was set.
    async fn mode(dir: &Path, profile: &Profile) -> Result<(Mode, Source)> {
        if let Some(mode) = memory::read_override(dir)
            .await
            .context("the `context` file is invalid")?
        {
            return Ok((mode, Source::Command));
        }
        Ok(match profile.context {
            Some(mode) => (mode, Source::Profile),
            None if profile.resume => (Mode::Session, Source::Default),
            None => (Mode::Fresh, Source::Default),
        })
    }

    /// `!context` (with `arg` `None`): the room's context mode, and its
    /// notes in `digest` mode. `!context <mode>`: switch to it from the
    /// next message on; `default` goes back to room.toml.
    pub async fn context(&self, room_id: &RoomId, arg: Option<&str>) -> Result<String> {
        let dir = self.room_dir(room_id)?;
        let lock = self.room_lock(room_id);
        let _turn = lock.lock().await;
        let profile = Self::profile(&dir).await?;
        let file = dir.join("context");
        match arg.map(str::trim).filter(|a| !a.is_empty()) {
            None => {
                let (mode, source) = Self::mode(&dir, &profile).await?;
                let from = match source {
                    Source::Command => "set with `!context`",
                    Source::Profile => "from room.toml",
                    Source::Default => "the default",
                };
                let mut text = format!("Context: **{mode}** ({from}).");
                if let Mode::Digest(_) = mode {
                    match tokio::fs::read_to_string(dir.join("MEMORY.md")).await {
                        Ok(notes) if !notes.trim().is_empty() => {
                            text.push_str(&format!("\n\nThe room's notes:\n\n{}", notes.trim()))
                        }
                        _ => text.push_str("\n\nNo notes yet."),
                    }
                }
                text.push_str(&format!(
                    "\n\nSwitch with `!context <mode>`:\n{}",
                    Mode::HELP
                ));
                Ok(text)
            }
            Some(a) if a.eq_ignore_ascii_case("default") => {
                match tokio::fs::remove_file(&file).await {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
                let (mode, _) = Self::mode(&dir, &profile).await?;
                Ok(format!(
                    "Context back to the room's default: **{mode}**, from the next message."
                ))
            }
            Some(a) => {
                let mode: Mode = a.parse()?;
                tokio::fs::create_dir_all(&dir).await?;
                tokio::fs::write(&file, format!("{mode}\n")).await?;
                let note = match mode {
                    Mode::Session => " Sessions saved earlier carry on; `!new` starts one over.",
                    Mode::Digest(_) => {
                        " After every reply a short extra run updates the room's notes; `!context` shows them."
                    }
                    Mode::Layered(_) => {
                        " Earlier days get summarised after replies, a few at a time; they are kept as notes in the room's `summaries/` folder, or where room.toml's `summaries` says."
                    }
                    _ => " Saved sessions are kept for when you switch back to `session`.",
                };
                Ok(format!(
                    "Context is now **{mode}**, from the next message.{note}"
                ))
            }
        }
    }

    /// Call after a reply is out, so the extra runs do not delay it. In
    /// `digest` mode: fold the room's last exchange into its notes
    /// (`MEMORY.md`). In `layered` mode: write the day and week summaries
    /// that are missing.
    pub async fn reflect(&self, room_id: &RoomId) {
        let pending = self.pending_digests.lock().unwrap().remove(room_id);
        if let Some(ex) = pending {
            let lock = self.room_lock(room_id);
            let _turn = lock.lock().await;
            if let Err(e) = self.digest(room_id, &ex).await {
                eprintln!("[{room_id}] could not update the room's notes: {e:#}");
            }
        }
        if !self.summarising.lock().unwrap().insert(room_id.to_owned()) {
            return;
        }
        if let Err(e) = self.summarise(room_id).await {
            eprintln!("[{room_id}] could not write summaries: {e:#}");
        }
        self.summarising.lock().unwrap().remove(room_id);
    }

    /// In `layered` mode: write the room's missing summaries, days first,
    /// then the weeks those complete. Without the room lock: it only adds
    /// files that replies read when they are there.
    async fn summarise(&self, room_id: &RoomId) -> Result<()> {
        let dir = self.room_dir(room_id)?;
        let profile = Self::profile(&dir).await?;
        let (Mode::Layered(_), _) = Self::mode(&dir, &profile).await? else {
            return Ok(());
        };
        let transcript = tokio::fs::read_to_string(dir.join("transcript.jsonl"))
            .await
            .unwrap_or_default();
        let msgs = layered::parse(&transcript, &[]);
        let today = layered::day_of(now());
        let (out_dir, prefix) = summaries_of(&dir, &profile);
        let name = profile.name.as_deref().unwrap_or(room_id.as_str());
        tokio::fs::create_dir_all(&out_dir).await?;
        // Locked down and small: it only writes text.
        let small = Profile {
            model: Some("haiku".into()),
            timeout_secs: 120,
            ..Profile::default()
        };
        loop {
            let have = read_summaries(&out_dir, &prefix).await;
            let jobs = layered::jobs(&msgs, today, &have);
            if jobs.is_empty() {
                return Ok(());
            }
            for job in jobs {
                let out = self
                    .harness
                    .run(&Run {
                        profile: &small,
                        system_prompt: job.system_prompt,
                        workdir: &dir,
                        input: &job.input,
                        session: None,
                        persist: false,
                        mcp_config: None,
                    })
                    .await?;
                if out.text.trim().is_empty() {
                    bail!("the summary run for {} returned nothing", job.key);
                }
                let file = out_dir.join(format!("{prefix}{}.md", job.key));
                let note = layered::note(&job, name, room_id.as_str(), &prefix, today, &out.text);
                tokio::fs::write(&file, note).await?;
                println!("[{room_id}] wrote {}", file.display());
            }
        }
    }

    async fn digest(&self, room_id: &RoomId, ex: &Exchange) -> Result<()> {
        let dir = self.room_dir(room_id)?;
        let file = dir.join("MEMORY.md");
        let notes = tokio::fs::read_to_string(&file).await.ok();
        let input = memory::digest_input(notes.as_deref(), &ex.sender, &ex.body, &ex.answer);
        // Locked down and small: it only rewrites text.
        let profile = Profile {
            model: Some("haiku".into()),
            timeout_secs: 120,
            ..Profile::default()
        };
        let out = self
            .harness
            .run(&Run {
                profile: &profile,
                system_prompt: memory::DIGEST_SYSTEM_PROMPT,
                workdir: &dir,
                input: &input,
                session: None,
                persist: false,
                mcp_config: None,
            })
            .await?;
        if out.text.trim().is_empty() {
            bail!("the digest run returned nothing");
        }
        tokio::fs::write(&file, out.text.trim().to_owned() + "\n").await?;
        Ok(())
    }

    /// Forget the session of the room's main timeline, or of one thread.
    /// Returns `None` if there was none, else the session's message count
    /// (unknown for sessions saved before parley counted them).
    pub async fn reset(&self, room_id: &RoomId, thread: Option<&str>) -> Result<Option<Saved>> {
        let file = self.session_file(&self.room_dir(room_id)?, thread)?;
        let saved = Saved::read(&file).await;
        match tokio::fs::remove_file(&file).await {
            Ok(()) => Ok(Some(saved.unwrap_or_default())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Append one line to the room's transcript.
    pub async fn record(
        &self,
        room_id: &RoomId,
        thread: Option<&str>,
        sender: &str,
        body: &str,
    ) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let dir = self.room_dir(room_id)?;
        tokio::fs::create_dir_all(&dir).await?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or_default();
        let mut line = serde_json::to_string(&serde_json::json!({
            "ts": ts, "sender": sender, "thread": thread, "body": body,
        }))?;
        line.push('\n');
        let mut f = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("transcript.jsonl"))
            .await?;
        f.write_all(line.as_bytes()).await?;
        // tokio writes in the background; without this the line may not be
        // on disk yet when the next one is appended or someone reads it.
        f.flush().await?;
        Ok(())
    }

    /// Whether an agent asked parley to restart (by creating the file
    /// `restart` next to `rooms/`). Consumes the request.
    pub fn take_restart_request(&self) -> bool {
        let Some(base) = self.rooms_dir.parent() else {
            return false;
        };
        std::fs::remove_file(base.join("restart")).is_ok()
    }

    /// Whether the room's profile says to post with the `post` tool rather
    /// than the run's final text. A missing or broken profile says no.
    /// The room's `max_depth` (the default for a room without a profile).
    pub async fn max_depth(&self, room_id: &RoomId) -> u32 {
        match self.room_dir(room_id) {
            Ok(dir) => Self::profile(&dir).await.unwrap_or_default().max_depth,
            Err(_) => Profile::default().max_depth,
        }
    }

    pub async fn replies_with_tool(&self, room_id: &RoomId) -> bool {
        match self.room_dir(room_id) {
            Ok(dir) => Self::profile(&dir)
                .await
                .is_ok_and(|p| p.reply == ReplyMode::Tool),
            Err(_) => false,
        }
    }

    /// Whether a message starts a run (not in a `trigger = "schedule"`
    /// room). A missing or broken profile says yes, so its error shows.
    pub async fn runs_on_messages(&self, room_id: &RoomId) -> bool {
        match self.room_dir(room_id) {
            Ok(dir) => Self::profile(&dir)
                .await
                .map_or(true, |p| p.trigger == Trigger::Message),
            Err(_) => true,
        }
    }

    /// Whether the room's profile says to answer main-timeline messages in
    /// a new thread. A missing or broken profile says no.
    pub async fn starts_threads(&self, room_id: &RoomId) -> bool {
        match self.room_dir(room_id) {
            Ok(dir) => Self::profile(&dir).await.is_ok_and(|p| p.threads),
            Err(_) => false,
        }
    }

    /// What a turn in `room_id`, answering `thread`, may read and post to,
    /// and what was wrong with the room's grants. Grants name rooms by ID,
    /// or by the `name` in their room.toml.
    pub async fn access(
        &self,
        room_id: &RoomId,
        thread: Option<&str>,
    ) -> Result<(crate::access::Access, Vec<String>)> {
        let profile = Self::profile(&self.room_dir(room_id)?).await?;
        let names = self.room_names().await;
        Ok(crate::access::Access::new(
            room_id.as_str(),
            profile.name.as_deref(),
            thread,
            &profile.access,
            &names,
        ))
    }

    /// Room names (the `name` in each room.toml) to room IDs.
    pub async fn room_names(&self) -> HashMap<String, String> {
        let mut names = HashMap::new();
        if let Ok(mut entries) = tokio::fs::read_dir(&self.rooms_dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let id = entry.file_name().to_string_lossy().into_owned();
                if !id.starts_with('!') {
                    continue;
                }
                if let Ok(Profile { name: Some(n), .. }) = Self::profile(&entry.path()).await {
                    names.insert(n, id);
                }
            }
        }
        names
    }

    async fn profile(dir: &Path) -> Result<Profile> {
        match tokio::fs::read_to_string(dir.join("room.toml")).await {
            Ok(text) => toml::from_str(&text).context("room.toml is invalid"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Profile::default()),
            Err(e) => Err(e).context("could not read room.toml"),
        }
    }

    fn room_dir(&self, room_id: &RoomId) -> Result<PathBuf> {
        // Room IDs come from the homeserver; still, never let one name a
        // folder outside rooms/.
        let id = room_id.as_str();
        if id.contains('/') || id.contains("..") {
            bail!("refusing room ID {id:?}");
        }
        Ok(self.rooms_dir.join(id))
    }

    fn session_file(&self, dir: &Path, thread: Option<&str>) -> Result<PathBuf> {
        Ok(match thread {
            None => dir.join("session"),
            Some(root) => {
                if root.contains('/') || root.contains("..") {
                    bail!("refusing thread ID {root:?}");
                }
                dir.join("threads").join(format!("{root}.session"))
            }
        })
    }

    async fn run(
        &self,
        room_id: &RoomId,
        thread: Option<&str>,
        said: &[Said],
        members: &[OwnedUserId],
        mcp_config: Option<&str>,
    ) -> Result<String> {
        let dir = self.room_dir(room_id)?;
        tokio::fs::create_dir_all(&dir).await?;

        let profile = Self::profile(&dir).await?;

        // Fail closed: a capable room stops working as soon as someone
        // who is not on its list is in it.
        if let Some(allowed) = &profile.members {
            let strangers: Vec<_> = members.iter().filter(|m| !allowed.contains(m)).collect();
            if !strangers.is_empty() {
                let names: Vec<_> = strangers.iter().map(|m| m.as_str()).collect();
                return Ok(format!(
                    "_Not running: this room's profile does not allow {}._",
                    names.join(", ")
                ));
            }
        }

        let prompt_file = dir.join("SYSTEM.md");
        let system_prompt = match tokio::fs::read_to_string(&prompt_file).await {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Write the default out, so it is there to read and edit.
                tokio::fs::write(&prompt_file, DEFAULT_SYSTEM_PROMPT).await?;
                println!("[{room_id}] wrote default {}", prompt_file.display());
                DEFAULT_SYSTEM_PROMPT.to_owned()
            }
            Err(e) => return Err(e).context("could not read SYSTEM.md"),
        };
        let system_prompt = match mcp_config {
            Some(_) => system_prompt + crate::post::PROMPT_NOTE,
            None => system_prompt,
        };

        let (mode, _) = Self::mode(&dir, &profile).await?;
        let session_file = self.session_file(&dir, thread)?;
        let saved = if mode.resumes() {
            Saved::read(&session_file).await
        } else {
            None
        };
        let session = saved.as_ref().map(|s| s.id.clone());

        let message = said
            .iter()
            .map(|s| format!("[{}] {}", s.sender, s.body))
            .collect::<Vec<_>>()
            .join("\n\n");
        let bodies: Vec<&str> = said.iter().map(|s| s.body.as_str()).collect();
        let input = match mode {
            Mode::Session | Mode::Fresh => message.clone(),
            Mode::Window(n) | Mode::Digest(n) => {
                let transcript = tokio::fs::read_to_string(dir.join("transcript.jsonl"))
                    .await
                    .unwrap_or_default();
                let history = memory::window(&transcript, n, &bodies);
                let notes = match mode {
                    Mode::Digest(_) => tokio::fs::read_to_string(dir.join("MEMORY.md")).await.ok(),
                    _ => None,
                };
                memory::with_history(notes.as_deref(), &history, &message)
            }
            Mode::Layered(n) => {
                let path = dir.join("transcript.jsonl");
                let transcript = tokio::fs::read_to_string(&path).await.unwrap_or_default();
                let msgs = layered::parse(&transcript, &bodies);
                let (sum_dir, prefix) = summaries_of(&dir, &profile);
                let have = read_summaries(&sum_dir, &prefix).await;
                layered::input(
                    &msgs,
                    thread,
                    n,
                    layered::day_of(now()),
                    &have,
                    &path.display().to_string(),
                    &message,
                )
            }
        };
        let workdir = profile.workdir.clone().unwrap_or_else(|| dir.clone());
        let run = Run {
            profile: &profile,
            system_prompt: &system_prompt,
            workdir: &workdir,
            input: &input,
            session: session.as_deref(),
            persist: mode.resumes(),
            mcp_config,
        };
        match self.harness.run(&run).await {
            Ok(out) => {
                if let Mode::Digest(_) = mode {
                    self.pending_digests.lock().unwrap().insert(
                        room_id.to_owned(),
                        Exchange {
                            sender: said.last().map(|s| s.sender.clone()).unwrap_or_default(),
                            body: match said {
                                [one] => one.body.clone(),
                                _ => message.clone(),
                            },
                            answer: out.text.clone(),
                        },
                    );
                }
                if let (true, Some(id)) = (mode.resumes(), &out.session) {
                    if let Some(parent) = session_file.parent() {
                        tokio::fs::create_dir_all(parent).await?;
                    }
                    // A resumed session may come back under a new ID; it is
                    // still the same conversation, so the count carries on.
                    let messages = match &saved {
                        Some(s) => s.messages.map(|n| n + 1),
                        None => Some(1),
                    };
                    let saved = Saved {
                        id: id.clone(),
                        messages,
                    };
                    tokio::fs::write(&session_file, saved.to_string()).await?;
                }
                Ok(out.text)
            }
            Err(e) => {
                if session.is_some() {
                    // A broken session would fail every turn; start over next time.
                    let _ = tokio::fs::remove_file(&session_file).await;
                    return Err(e.context("session reset; the next message starts fresh"));
                }
                Err(e)
            }
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Where a room's summary notes go, and the prefix of their file names.
fn summaries_of(dir: &Path, profile: &Profile) -> (PathBuf, String) {
    let folder = profile
        .summaries
        .clone()
        .unwrap_or_else(|| dir.join("summaries"));
    let prefix = profile
        .name
        .as_ref()
        .map(|n| format!("{n} "))
        .unwrap_or_default();
    (folder, prefix)
}

/// The summaries in `dir` named `<prefix><key>.md`, by key, without their
/// frontmatter.
async fn read_summaries(dir: &Path, prefix: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return out;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let Some(key) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".md"))
            .and_then(|n| n.strip_prefix(prefix))
        else {
            continue;
        };
        if let Ok(text) = tokio::fs::read_to_string(&path).await {
            out.insert(key.to_owned(), layered::strip_frontmatter(&text).to_owned());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::VecDeque, time::Instant};

    use anyhow::anyhow;
    use matrix_sdk::ruma::OwnedRoomId;

    fn room(id: &str) -> OwnedRoomId {
        RoomId::parse(id).unwrap()
    }

    // ---- ClaudeCode against fake programs ----

    /// A fake `claude`: an executable shell script with `body` in `dir`.
    fn script(dir: &Path, body: &str) -> String {
        // Written by a separate `sh`, not by this process: tests run in
        // parallel threads, and a child forked by another test while we held
        // the file open for writing makes Linux refuse to exec it
        // ("Text file busy").
        use std::io::Write;
        let path = dir.join("fake-claude");
        let mut sh = std::process::Command::new("/bin/sh")
            .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
            .arg(&path)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        sh.stdin
            .take()
            .unwrap()
            .write_all(format!("#!/bin/sh\n{body}\n").as_bytes())
            .unwrap();
        assert!(sh.wait().unwrap().success());
        path.to_str().unwrap().to_owned()
    }

    /// A fake-program harness plus the timeout its runs get.
    struct Cc(ClaudeCode, Duration);

    fn claude(program: String, timeout: Duration) -> Cc {
        Cc(ClaudeCode { program }, timeout)
    }

    impl Cc {
        /// Run with the default (locked-down) profile. Profiles count
        /// whole seconds; tests want less, so the timeout is applied here.
        async fn run(&self, system_prompt: &str, workdir: &Path, input: &str) -> Result<String> {
            let profile = Profile {
                timeout_secs: 60,
                ..Profile::default()
            };
            let run = Run {
                profile: &profile,
                system_prompt,
                workdir,
                input,
                session: None,
                persist: false,
                mcp_config: None,
            };
            match tokio::time::timeout(self.1, self.0.run(&run)).await {
                Ok(r) => r.map(|o| o.text),
                Err(_) => bail!("no answer within {:?}", self.1),
            }
        }
    }

    #[tokio::test]
    async fn claude_code_returns_result() {
        let tmp = tempfile::tempdir().unwrap();
        let cc = claude(
            script(tmp.path(), r#"echo '{"result":"hi","is_error":false}'"#),
            Duration::from_secs(10),
        );
        assert_eq!(
            cc.run("sys", tmp.path(), "[Kas] hello").await.unwrap(),
            "hi"
        );
    }

    #[tokio::test]
    async fn claude_code_is_error_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cc = claude(
            script(tmp.path(), r#"echo '{"result":"boom","is_error":true}'"#),
            Duration::from_secs(10),
        );
        let err = cc.run("sys", tmp.path(), "x").await.unwrap_err();
        assert!(format!("{err:#}").contains("boom"), "{err:#}");
    }

    #[tokio::test]
    async fn claude_code_missing_result_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let cc = claude(
            script(tmp.path(), r#"echo '{"is_error":false}'"#),
            Duration::from_secs(10),
        );
        assert!(cc.run("sys", tmp.path(), "x").await.is_err());
    }

    #[tokio::test]
    async fn claude_code_non_json_reports_status_and_stderr() {
        let tmp = tempfile::tempdir().unwrap();
        let cc = claude(
            script(tmp.path(), "echo 'not json'; echo 'it broke' >&2; exit 1"),
            Duration::from_secs(10),
        );
        let err = format!("{:#}", cc.run("sys", tmp.path(), "x").await.unwrap_err());
        assert!(err.contains("it broke"), "{err}");
        assert!(err.contains("exit status: 1"), "{err}");
    }

    #[tokio::test]
    async fn claude_code_times_out() {
        let tmp = tempfile::tempdir().unwrap();
        let cc = claude(
            script(tmp.path(), "exec sleep 10"),
            Duration::from_millis(200),
        );
        let start = Instant::now();
        let err = format!("{:#}", cc.run("sys", tmp.path(), "x").await.unwrap_err());
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "took {:?}",
            start.elapsed()
        );
        assert!(err.contains("no answer within"), "{err}");
    }

    #[tokio::test]
    async fn claude_code_missing_program_fails_to_start() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope").to_str().unwrap().to_owned();
        let err = format!(
            "{:#}",
            claude(missing, Duration::from_secs(1))
                .run("s", tmp.path(), "x")
                .await
                .unwrap_err()
        );
        assert!(err.contains("could not start"), "{err}");
    }

    #[tokio::test]
    async fn claude_code_passes_lockdown_args_and_runs_in_workdir() {
        let tmp = tempfile::tempdir().unwrap();
        let workdir = tmp.path().join("room");
        std::fs::create_dir(&workdir).unwrap();
        let dump = tmp.path().join("dump");
        let body = format!(
            r#"pwd -P > '{d}'
for a in "$@"; do printf '<%s>\n' "$a"; done >> '{d}'
echo '{{"result":"ok"}}'"#,
            d = dump.display()
        );
        let cc = claude(script(tmp.path(), &body), Duration::from_secs(10));
        let reply = cc
            .run("be brief", &workdir, "[Kas] hi there")
            .await
            .unwrap();
        assert_eq!(reply, "ok"); // is_error defaults to false

        let dump = std::fs::read_to_string(dump).unwrap();
        let mut lines = dump.lines();
        let pwd = lines.next().unwrap();
        assert_eq!(Path::new(pwd), workdir.canonicalize().unwrap());
        let args: Vec<&str> = lines.collect();
        assert_eq!(
            args,
            [
                "<-p>",
                "<[Kas] hi there>",
                "<--system-prompt>",
                "<be brief>",
                "<--tools>",
                "<>",
                "<--strict-mcp-config>",
                "<--setting-sources>",
                "<>",
                "<--disable-slash-commands>",
                "<--no-session-persistence>",
                "<--output-format>",
                "<json>",
            ]
        );
    }

    #[test]
    fn mcp_config_is_passed_with_the_post_tool_allowed() {
        let profile = Profile::default();
        let run = Run {
            profile: &profile,
            system_prompt: "s",
            workdir: Path::new("/"),
            input: "x",
            session: None,
            persist: false,
            mcp_config: Some("{}"),
        };
        let args = ClaudeCode::args(&run);
        let at = args.iter().position(|a| a == "--mcp-config").unwrap();
        assert_eq!(args[at + 1], "{}");
        assert!(args.contains(&"mcp__parley__post".to_owned()));
    }

    // ---- Agent with a fake harness ----

    #[derive(Debug, Clone)]
    struct Call {
        system_prompt: String,
        workdir: PathBuf,
        input: String,
        session: Option<String>,
        persist: bool,
    }

    /// Records calls; answers from `results` (then "ok"). With `sleepy`, a
    /// body that is a number makes the run take that many milliseconds and
    /// log "start <n>" / "end <n>".
    #[derive(Default)]
    struct Fake {
        calls: std::sync::Mutex<Vec<Call>>,
        results: std::sync::Mutex<VecDeque<Result<String>>>,
        log: std::sync::Mutex<Vec<String>>,
        sleepy: bool,
    }

    impl Fake {
        fn failing(n: usize) -> Self {
            let f = Fake::default();
            for i in 0..n {
                f.results
                    .lock()
                    .unwrap()
                    .push_back(Err(anyhow!("fail {i}")));
            }
            f
        }
        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Harness for Fake {
        async fn run(&self, run: &Run<'_>) -> Result<Output> {
            let input = run.input;
            self.calls.lock().unwrap().push(Call {
                system_prompt: run.system_prompt.to_owned(),
                workdir: run.workdir.to_owned(),
                input: input.to_owned(),
                session: run.session.map(ToOwned::to_owned),
                persist: run.persist,
            });
            if self.sleepy {
                let n = input.rsplit(' ').next().unwrap().to_owned();
                self.log.lock().unwrap().push(format!("start {n}"));
                tokio::time::sleep(Duration::from_millis(n.parse().unwrap())).await;
                self.log.lock().unwrap().push(format!("end {n}"));
            }
            let text = self
                .results
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Ok("ok".into()))?;
            let n = self.calls.lock().unwrap().len();
            Ok(Output {
                text,
                session: Some(format!("sess-{n}")),
            })
        }
    }

    #[tokio::test]
    async fn writes_default_system_prompt_and_passes_it() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().join("rooms"));
        let r = room("!abc:example.org");
        assert_eq!(agent.reply(&r, None, "Kas", "hello", &[]).await, "ok");

        let dir = tmp.path().join("rooms").join("!abc:example.org");
        let on_disk = std::fs::read_to_string(dir.join("SYSTEM.md")).unwrap();
        assert_eq!(on_disk, DEFAULT_SYSTEM_PROMPT);
        let calls = agent.harness.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].system_prompt, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(calls[0].workdir, dir);
        assert_eq!(calls[0].input, "[Kas] hello");
    }

    #[tokio::test]
    async fn edited_system_prompt_is_used_next_time() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!abc:example.org");
        agent.reply(&r, None, "Kas", "one", &[]).await;
        std::fs::write(
            tmp.path().join(r.as_str()).join("SYSTEM.md"),
            "Be a pirate.",
        )
        .unwrap();
        agent.reply(&r, None, "Camille", "two", &[]).await;

        let calls = agent.harness.calls();
        assert_eq!(calls[1].system_prompt, "Be a pirate.");
        assert_eq!(calls[1].input, "[Camille] two");
    }

    #[tokio::test]
    async fn rooms_have_separate_folders() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        agent
            .reply(&room("!a:example.org"), None, "Kas", "x", &[])
            .await;
        agent
            .reply(&room("!b:example.org"), None, "Kas", "y", &[])
            .await;
        let calls = agent.harness.calls();
        assert_ne!(calls[0].workdir, calls[1].workdir);
        assert!(tmp.path().join("!a:example.org/SYSTEM.md").exists());
        assert!(tmp.path().join("!b:example.org/SYSTEM.md").exists());
    }

    #[tokio::test]
    async fn input_is_sender_in_brackets_then_body() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        agent
            .reply(
                &room("!a:example.org"),
                None,
                "Kas Buunk",
                "multi\nline",
                &[],
            )
            .await;
        assert_eq!(agent.harness.calls()[0].input, "[Kas Buunk] multi\nline");
    }

    #[tokio::test]
    async fn messages_answered_together_are_one_input() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let said = [Said::new("Kas", "one"), Said::new("Camille", "two")];
        agent
            .reply_with(&room("!a:example.org"), None, &said, &[], None)
            .await;
        assert_eq!(agent.harness.calls()[0].input, "[Kas] one\n\n[Camille] two");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn threads_of_one_room_run_concurrently() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Arc::new(Agent::new(
            Fake {
                sleepy: true,
                ..Default::default()
            },
            tmp.path().to_owned(),
        ));
        let r = room("!a:example.org");
        let a = {
            let (agent, r) = (agent.clone(), r.clone());
            tokio::spawn(async move { agent.reply(&r, Some("$a"), "Kas", "150", &[]).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let b = {
            let (agent, r) = (agent.clone(), r.clone());
            tokio::spawn(async move { agent.reply(&r, Some("$b"), "Kas", "10", &[]).await })
        };
        a.await.unwrap();
        b.await.unwrap();
        let log = agent.harness.log.lock().unwrap().clone();
        assert_eq!(log, ["start 150", "start 10", "end 10", "end 150"]);
    }

    #[tokio::test]
    async fn harness_error_becomes_error_reply() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::failing(1), tmp.path().to_owned());
        let reply = agent
            .reply(&room("!a:example.org"), None, "Kas", "x", &[])
            .await;
        assert_eq!(reply, "_Agent error: fail 0_");
    }

    #[tokio::test]
    async fn breaker_opens_after_three_failures_and_skips_harness() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::failing(3), tmp.path().to_owned());
        let r = room("!a:example.org");
        for _ in 0..3 {
            assert!(
                agent
                    .reply(&r, None, "Kas", "x", &[])
                    .await
                    .starts_with("_Agent error")
            );
        }
        let reply = agent.reply(&r, None, "Kas", "x", &[]).await;
        assert_eq!(
            reply,
            "_Agent paused after repeated failures; trying again in 5 min._"
        );
        assert_eq!(agent.harness.calls().len(), 3);
        // The breaker is shared: other rooms are paused too.
        let other = agent
            .reply(&room("!b:example.org"), None, "Kas", "x", &[])
            .await;
        assert!(other.contains("paused"), "{other}");
        assert_eq!(agent.harness.calls().len(), 3);
    }

    #[tokio::test]
    async fn success_resets_the_failure_count() {
        let tmp = tempfile::tempdir().unwrap();
        let fake = Fake::default();
        {
            let mut results = fake.results.lock().unwrap();
            results.push_back(Err(anyhow!("a")));
            results.push_back(Err(anyhow!("b")));
            results.push_back(Ok("fine".into()));
            results.push_back(Err(anyhow!("c")));
            results.push_back(Err(anyhow!("d")));
        }
        let agent = Agent::new(fake, tmp.path().to_owned());
        let r = room("!a:example.org");
        let mut replies = Vec::new();
        for _ in 0..6 {
            replies.push(agent.reply(&r, None, "Kas", "x", &[]).await);
        }
        assert_eq!(replies[2], "fine");
        // Two failures after the success do not open the breaker: call 6 runs.
        assert_eq!(replies[5], "ok");
        assert_eq!(agent.harness.calls().len(), 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn replies_in_one_conversation_are_serialized_in_arrival_order() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Arc::new(Agent::new(
            Fake {
                sleepy: true,
                ..Default::default()
            },
            tmp.path().to_owned(),
        ));
        let r = room("!a:example.org");
        let mut tasks = Vec::new();
        // Long first, short later: without the room lock they would overlap.
        for ms in ["150", "50", "0"] {
            let (agent, r) = (agent.clone(), r.clone());
            tasks.push(tokio::spawn(async move {
                agent.reply(&r, None, "Kas", ms, &[]).await
            }));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        for t in tasks {
            assert_eq!(t.await.unwrap(), "ok");
        }
        let log = agent.harness.log.lock().unwrap().clone();
        assert_eq!(
            log,
            [
                "start 150",
                "end 150",
                "start 50",
                "end 50",
                "start 0",
                "end 0"
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn different_rooms_run_concurrently() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Arc::new(Agent::new(
            Fake {
                sleepy: true,
                ..Default::default()
            },
            tmp.path().to_owned(),
        ));
        let a = {
            let agent = agent.clone();
            tokio::spawn(async move {
                agent
                    .reply(&room("!a:example.org"), None, "Kas", "150", &[])
                    .await
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let b = {
            let agent = agent.clone();
            tokio::spawn(async move {
                agent
                    .reply(&room("!b:example.org"), None, "Kas", "10", &[])
                    .await
            })
        };
        a.await.unwrap();
        b.await.unwrap();
        let log = agent.harness.log.lock().unwrap().clone();
        assert_eq!(log, ["start 150", "start 10", "end 10", "end 150"]);
    }

    // ---- sessions: main timeline, threads, !new ----

    fn resume_room(rooms: &Path, id: &str) {
        let dir = rooms.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("room.toml"), "resume = true\n").unwrap();
    }

    #[tokio::test]
    async fn starts_threads_only_when_the_profile_says_so() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        for (id, toml) in [
            ("!on:example.org", Some("threads = true\n")),
            ("!off:example.org", Some("resume = true\n")),
            ("!broken:example.org", Some("threads = \"yes\"\n")),
            ("!none:example.org", None),
        ] {
            if let Some(toml) = toml {
                let dir = tmp.path().join(id);
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join("room.toml"), toml).unwrap();
            }
        }
        assert!(agent.starts_threads(&room("!on:example.org")).await);
        assert!(!agent.starts_threads(&room("!off:example.org")).await);
        assert!(!agent.starts_threads(&room("!broken:example.org")).await);
        assert!(!agent.starts_threads(&room("!none:example.org")).await);
    }

    #[tokio::test]
    async fn main_timeline_and_threads_keep_separate_sessions() {
        let tmp = tempfile::tempdir().unwrap();
        resume_room(tmp.path(), "!a:example.org");
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");

        agent.reply(&r, None, "Kas", "one", &[]).await; // main: new -> sess-1
        agent.reply(&r, Some("$t1"), "Kas", "two", &[]).await; // thread: new -> sess-2
        agent.reply(&r, None, "Kas", "three", &[]).await; // main resumes sess-1
        agent.reply(&r, Some("$t1"), "Kas", "four", &[]).await; // thread resumes sess-2

        let sessions: Vec<_> = agent
            .harness
            .calls()
            .into_iter()
            .map(|c| c.session)
            .collect();
        assert_eq!(
            sessions,
            [None, None, Some("sess-1".into()), Some("sess-2".into())]
        );
        let dir = tmp.path().join("!a:example.org");
        assert!(dir.join("session").exists());
        assert!(dir.join("threads").join("$t1.session").exists());
    }

    #[tokio::test]
    async fn reset_clears_only_that_session() {
        let tmp = tempfile::tempdir().unwrap();
        resume_room(tmp.path(), "!a:example.org");
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");
        agent.reply(&r, None, "Kas", "one", &[]).await;
        agent.reply(&r, Some("$t1"), "Kas", "two", &[]).await;

        assert!(agent.reset(&r, None).await.unwrap().is_some());
        assert!(agent.reset(&r, None).await.unwrap().is_none()); // nothing left to clear
        agent.reply(&r, None, "Kas", "fresh", &[]).await;
        agent.reply(&r, Some("$t1"), "Kas", "still", &[]).await;

        let calls = agent.harness.calls();
        assert_eq!(calls[2].session, None); // main started over
        assert_eq!(calls[3].session.as_deref(), Some("sess-2")); // thread kept
    }

    #[tokio::test]
    async fn reset_reports_the_message_count() {
        let tmp = tempfile::tempdir().unwrap();
        resume_room(tmp.path(), "!a:example.org");
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");
        for body in ["one", "two", "three"] {
            agent.reply(&r, None, "Kas", body, &[]).await;
        }
        let cleared = agent.reset(&r, None).await.unwrap().unwrap();
        assert_eq!(cleared.messages, Some(3));
    }

    #[tokio::test]
    async fn legacy_session_file_resumes_with_unknown_count() {
        let tmp = tempfile::tempdir().unwrap();
        resume_room(tmp.path(), "!a:example.org");
        let dir = tmp.path().join("!a:example.org");
        std::fs::write(dir.join("session"), "old-id").unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");
        agent.reply(&r, None, "Kas", "hi", &[]).await;
        assert_eq!(agent.harness.calls()[0].session.as_deref(), Some("old-id"));
        let cleared = agent.reset(&r, None).await.unwrap().unwrap();
        assert_eq!(cleared.messages, None);
    }

    #[tokio::test]
    async fn thread_ids_cannot_escape_the_room_folder() {
        let tmp = tempfile::tempdir().unwrap();
        resume_room(tmp.path(), "!a:example.org");
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let reply = agent
            .reply(&room("!a:example.org"), Some("$../../x"), "Kas", "hi", &[])
            .await;
        assert!(reply.starts_with("_Agent error"), "{reply}");
        assert!(
            agent
                .reset(&room("!a:example.org"), Some("a/b"))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn transcript_appends_json_lines() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().join("rooms"));
        let r = room("!a:example.org");
        agent.record(&r, None, "@kas:x", "hi").await.unwrap();
        agent
            .record(&r, Some("$t"), "agent", "hello\nthere")
            .await
            .unwrap();
        let text =
            std::fs::read_to_string(tmp.path().join("rooms/!a:example.org/transcript.jsonl"))
                .unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["sender"], "@kas:x");
        assert!(lines[0]["thread"].is_null());
        assert_eq!(lines[1]["thread"], "$t");
        assert_eq!(lines[1]["body"], "hello\nthere");
    }

    // ---- context modes ----

    fn room_toml(rooms: &Path, id: &str, toml: &str) {
        let dir = rooms.join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("room.toml"), toml).unwrap();
    }

    /// A message as main.rs handles it: recorded first, then answered.
    async fn say(agent: &Agent<Fake>, r: &RoomId, body: &str) -> String {
        agent.record(r, None, "@kas:x", body).await.unwrap();
        let answer = agent.reply(r, None, "Kas", body, &[]).await;
        agent.record(r, None, "agent", &answer).await.unwrap();
        agent.reflect(r).await;
        answer
    }

    #[tokio::test]
    async fn mode_comes_from_command_then_profile_then_resume() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");
        let status = agent.context(&r, None).await.unwrap();
        assert!(
            status.starts_with("Context: **fresh** (the default)."),
            "{status}"
        );

        room_toml(tmp.path(), r.as_str(), "resume = true\n");
        let status = agent.context(&r, None).await.unwrap();
        assert!(
            status.starts_with("Context: **session** (the default)."),
            "{status}"
        );

        room_toml(
            tmp.path(),
            r.as_str(),
            "resume = true\ncontext = \"window 4\"\n",
        );
        let status = agent.context(&r, None).await.unwrap();
        assert!(
            status.starts_with("Context: **window 4** (from room.toml)."),
            "{status}"
        );

        let set = agent.context(&r, Some("digest 3")).await.unwrap();
        assert!(set.starts_with("Context is now **digest 3**"), "{set}");
        let status = agent.context(&r, None).await.unwrap();
        assert!(
            status.starts_with("Context: **digest 3** (set with `!context`)."),
            "{status}"
        );
        assert!(status.contains("No notes yet."), "{status}");

        let back = agent.context(&r, Some("default")).await.unwrap();
        assert!(back.contains("**window 4**"), "{back}");
        assert!(agent.context(&r, Some("bogus")).await.is_err());
        assert!(
            agent
                .context(&r, None)
                .await
                .unwrap()
                .contains("**window 4**")
        );
    }

    #[tokio::test]
    async fn switching_away_from_session_runs_fresh_and_back_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        resume_room(tmp.path(), "!a:example.org");
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");
        say(&agent, &r, "one").await; // session sess-1
        agent.context(&r, Some("fresh")).await.unwrap();
        say(&agent, &r, "two").await;
        agent.context(&r, Some("session")).await.unwrap();
        say(&agent, &r, "three").await;

        let calls = agent.harness.calls();
        let got: Vec<_> = calls
            .iter()
            .map(|c| (c.session.clone(), c.persist))
            .collect();
        assert_eq!(
            got,
            [(None, true), (None, false), (Some("sess-1".into()), true)]
        );
        assert_eq!(calls[1].input, "[Kas] two");
    }

    #[tokio::test]
    async fn window_mode_quotes_recent_messages() {
        let tmp = tempfile::tempdir().unwrap();
        room_toml(tmp.path(), "!a:example.org", "context = \"window 2\"\n");
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        let r = room("!a:example.org");
        say(&agent, &r, "first").await;
        say(&agent, &r, "second").await;

        let calls = agent.harness.calls();
        assert_eq!(calls[0].input, "[Kas] first"); // nothing before it
        let input = &calls[1].input;
        assert!(input.contains("[kas] first\n\n[agent] ok"), "{input}");
        assert!(
            input.ends_with("</recent-messages>\n\n[Kas] second"),
            "{input}"
        );
        assert!(calls.iter().all(|c| c.session.is_none() && !c.persist));
    }

    #[tokio::test]
    async fn digest_mode_keeps_notes_after_each_reply() {
        let tmp = tempfile::tempdir().unwrap();
        room_toml(tmp.path(), "!a:example.org", "context = \"digest 1\"\n");
        let fake = Fake::default();
        {
            let mut results = fake.results.lock().unwrap();
            results.push_back(Ok("hello".into())); // reply to "hi"
            results.push_back(Ok("- Kas said hi".into())); // digest
            results.push_back(Ok("fine".into())); // reply to "again"
            results.push_back(Ok("- Kas said hi twice".into())); // digest
        }
        let agent = Agent::new(fake, tmp.path().to_owned());
        let r = room("!a:example.org");
        assert_eq!(say(&agent, &r, "hi").await, "hello");
        let dir = tmp.path().join("!a:example.org");
        assert_eq!(
            std::fs::read_to_string(dir.join("MEMORY.md")).unwrap(),
            "- Kas said hi\n"
        );
        say(&agent, &r, "again").await;

        let calls = agent.harness.calls();
        assert_eq!(calls.len(), 4); // reply, digest, reply, digest
        assert_eq!(calls[1].system_prompt, memory::DIGEST_SYSTEM_PROMPT);
        assert!(
            calls[1].input.contains("[Kas] hi\n\n[agent] hello"),
            "{}",
            calls[1].input
        );
        let input = &calls[2].input;
        assert!(input.contains("<room-memory>"), "{input}");
        assert!(input.contains("- Kas said hi"), "{input}");
        assert!(input.contains("[agent] hello"), "{input}");
        assert!(!input.contains("[kas] hi"), "window is 1: {input}");
        assert!(
            agent
                .context(&r, None)
                .await
                .unwrap()
                .contains("- Kas said hi twice")
        );
    }

    #[tokio::test]
    async fn layered_mode_summarises_past_days_and_quotes_the_thread() {
        let tmp = tempfile::tempdir().unwrap();
        let id = "!a:example.org";
        let vault = tmp.path().join("vault");
        room_toml(
            tmp.path(),
            id,
            &format!(
                "context = \"layered 5\"\nname = \"garden\"\nsummaries = {:?}\n",
                vault.display().to_string()
            ),
        );
        let dir = tmp.path().join(id);
        // Two messages from three days ago; the transcript is chronological.
        let old = now() - 3 * 86_400;
        let lines: String = [("@kas:x", "plan the garden"), ("agent", "dig first")]
            .iter()
            .map(|(s, b)| {
                serde_json::json!({"ts": old, "sender": s, "thread": null, "body": b}).to_string()
                    + "\n"
            })
            .collect();
        std::fs::write(dir.join("transcript.jsonl"), lines).unwrap();

        let fake = Fake::default();
        {
            let mut results = fake.results.lock().unwrap();
            results.push_back(Ok("thread answer".into())); // reply
            results.push_back(Ok("- garden planned".into())); // day summary
        }
        let agent = Agent::new(fake, tmp.path().to_owned());
        let r = room(id);
        agent
            .record(&r, Some("$t"), "@kas:x", "and now?")
            .await
            .unwrap();
        let answer = agent.reply(&r, Some("$t"), "Kas", "and now?", &[]).await;
        assert_eq!(answer, "thread answer");
        agent
            .record(&r, Some("$t"), "agent", &answer)
            .await
            .unwrap();
        agent.reflect(&r).await;

        let calls = agent.harness.calls();
        assert!(!calls[0].persist);
        let input = &calls[0].input;
        assert!(input.contains("(not summarised yet)"), "{input}");
        assert!(input.contains("transcript.jsonl"), "{input}");
        assert!(
            input.contains("[kas] plan the garden\n\n[agent] dig first"),
            "{input}"
        );
        assert!(input.ends_with("[Kas] and now?"), "{input}");
        assert_eq!(calls[1].system_prompt, layered::DAY_PROMPT);
        assert!(calls[1].input.contains("[kas] plan the garden"));
        assert_eq!(calls.len(), 2);
        let day = layered::date(layered::day_of(old));
        let note = std::fs::read_to_string(vault.join(format!("garden {day}.md"))).unwrap();
        assert!(note.starts_with("---\ntype: room-summary\n"), "{note}");
        assert!(note.ends_with("---\n\n- garden planned\n"), "{note}");

        // The next message reads the summary and quotes the thread.
        agent
            .record(&r, Some("$t"), "@kas:x", "more")
            .await
            .unwrap();
        agent.reply(&r, Some("$t"), "Kas", "more", &[]).await;
        let input = &agent.harness.calls()[2].input;
        assert!(input.contains("- garden planned"), "{input}");
        assert!(
            input.contains("so far:\n\n[kas] and now?\n\n[agent] thread answer\n</this-thread>"),
            "{input}"
        );
    }

    #[tokio::test]
    async fn failed_reply_is_not_digested() {
        let tmp = tempfile::tempdir().unwrap();
        room_toml(tmp.path(), "!a:example.org", "context = \"digest\"\n");
        let agent = Agent::new(Fake::failing(1), tmp.path().to_owned());
        let r = room("!a:example.org");
        assert!(say(&agent, &r, "hi").await.starts_with("_Agent error"));
        assert_eq!(agent.harness.calls().len(), 1);
        assert!(!tmp.path().join("!a:example.org/MEMORY.md").exists());
    }

    #[tokio::test]
    async fn restart_request_is_consumed_once() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().join("rooms"));
        assert!(!agent.take_restart_request());
        std::fs::write(tmp.path().join("restart"), "").unwrap();
        assert!(agent.take_restart_request());
        assert!(!agent.take_restart_request());
    }
}
