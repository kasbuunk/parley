//! Turning a room message into an agent reply.
//!
//! Step 2 (v0): every room has a folder `rooms/<room id>/` with a
//! `SYSTEM.md`. Each message starts a fresh agent run with that system prompt
//! and the message (prefixed with the sender's name) as its only user turn.
//! No history, no tools, no data. The folder is the run's working directory,
//! so later steps can put context there (AGENTS.md, files, a transcript).
//!
//! The model and harness sit behind [`Harness`]; Claude Code is the first
//! implementation, Pi can be a second.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use matrix_sdk::ruma::{OwnedRoomId, RoomId};
use tokio::{process::Command, sync::Mutex};

use crate::breaker::{Breaker, State};

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

/// One agent run: a system prompt, a working directory, one user turn.
pub trait Harness: Send + Sync {
    fn run(
        &self,
        system_prompt: &str,
        workdir: &Path,
        input: &str,
    ) -> impl Future<Output = Result<String>> + Send;
}

/// Claude Code in headless mode, on the logged-in subscription.
///
/// Locked down: no tools, no MCP servers, no user/project settings (so no
/// hooks or CLAUDE.md), no slash commands, nothing saved as a session.
pub struct ClaudeCode {
    pub program: String,
    pub timeout: Duration,
}

impl Harness for ClaudeCode {
    async fn run(&self, system_prompt: &str, workdir: &Path, input: &str) -> Result<String> {
        let child = Command::new(&self.program)
            .args(["-p", input, "--system-prompt", system_prompt])
            .args(["--tools", "", "--strict-mcp-config", "--setting-sources", ""])
            .args(["--disable-slash-commands", "--no-session-persistence"])
            .args(["--output-format", "json"])
            .current_dir(workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("could not start {}", self.program))?;

        let output = tokio::time::timeout(self.timeout, child.wait_with_output())
            .await
            .with_context(|| format!("no answer within {}s", self.timeout.as_secs()))??;

        #[derive(serde::Deserialize)]
        struct Reply {
            result: Option<String>,
            #[serde(default)]
            is_error: bool,
        }
        let reply: Reply = serde_json::from_slice(&output.stdout).with_context(|| {
            format!(
                "claude exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        })?;
        match reply.result {
            Some(text) if !reply.is_error => Ok(text),
            other => bail!("claude reported an error: {}", other.unwrap_or_default()),
        }
    }
}

pub struct Agent<H> {
    harness: H,
    rooms_dir: PathBuf,
    breaker: std::sync::Mutex<Breaker>,
    /// One lock per room, so replies come out in the order messages came in.
    room_locks: std::sync::Mutex<HashMap<OwnedRoomId, Arc<Mutex<()>>>>,
}

impl<H: Harness> Agent<H> {
    pub fn new(harness: H, rooms_dir: PathBuf) -> Self {
        Self {
            harness,
            rooms_dir,
            breaker: std::sync::Mutex::new(Breaker::new(3, Duration::from_secs(300))),
            room_locks: Default::default(),
        }
    }

    /// The reply to post for `body`, sent by `sender` in `room_id`.
    pub async fn reply(&self, room_id: &RoomId, sender: &str, body: &str) -> String {
        let lock = self.room_locks.lock().unwrap().entry(room_id.to_owned()).or_default().clone();
        let _turn = lock.lock().await;

        if let State::Open(left) = self.breaker.lock().unwrap().state() {
            return format!(
                "_Agent paused after repeated failures; trying again in {} min._",
                left.as_secs().div_ceil(60)
            );
        }

        match self.run(room_id, sender, body).await {
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

    async fn run(&self, room_id: &RoomId, sender: &str, body: &str) -> Result<String> {
        let dir = self.rooms_dir.join(room_id.as_str());
        tokio::fs::create_dir_all(&dir).await?;
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
        self.harness.run(&system_prompt, &dir, &format!("[{sender}] {body}")).await
    }
}
