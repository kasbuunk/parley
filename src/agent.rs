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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::VecDeque, os::unix::fs::PermissionsExt, time::Instant};

    use anyhow::anyhow;
    use matrix_sdk::ruma::OwnedRoomId;

    fn room(id: &str) -> OwnedRoomId {
        RoomId::parse(id).unwrap()
    }

    // ---- ClaudeCode against fake programs ----

    /// A fake `claude`: an executable shell script with `body` in `dir`.
    fn script(dir: &Path, body: &str) -> String {
        let path = dir.join("fake-claude");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_str().unwrap().to_owned()
    }

    fn claude(program: String, timeout: Duration) -> ClaudeCode {
        ClaudeCode { program, timeout }
    }

    #[tokio::test]
    async fn claude_code_returns_result() {
        let tmp = tempfile::tempdir().unwrap();
        let cc = claude(
            script(tmp.path(), r#"echo '{"result":"hi","is_error":false}'"#),
            Duration::from_secs(10),
        );
        assert_eq!(cc.run("sys", tmp.path(), "[Kas] hello").await.unwrap(), "hi");
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
        let cc = claude(script(tmp.path(), r#"echo '{"is_error":false}'"#), Duration::from_secs(10));
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
        let cc = claude(script(tmp.path(), "exec sleep 10"), Duration::from_millis(200));
        let start = Instant::now();
        let err = format!("{:#}", cc.run("sys", tmp.path(), "x").await.unwrap_err());
        assert!(start.elapsed() < Duration::from_secs(5), "took {:?}", start.elapsed());
        assert!(err.contains("no answer within"), "{err}");
    }

    #[tokio::test]
    async fn claude_code_missing_program_fails_to_start() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nope").to_str().unwrap().to_owned();
        let err = format!("{:#}", claude(missing, Duration::from_secs(1)).run("s", tmp.path(), "x").await.unwrap_err());
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
        let reply = cc.run("be brief", &workdir, "[Kas] hi there").await.unwrap();
        assert_eq!(reply, "ok"); // is_error defaults to false

        let dump = std::fs::read_to_string(dump).unwrap();
        let mut lines = dump.lines();
        let pwd = lines.next().unwrap();
        assert_eq!(Path::new(pwd), workdir.canonicalize().unwrap());
        let args: Vec<&str> = lines.collect();
        assert_eq!(
            args,
            [
                "<-p>", "<[Kas] hi there>", "<--system-prompt>", "<be brief>",
                "<--tools>", "<>", "<--strict-mcp-config>", "<--setting-sources>", "<>",
                "<--disable-slash-commands>", "<--no-session-persistence>",
                "<--output-format>", "<json>",
            ]
        );
    }

    // ---- Agent with a fake harness ----

    #[derive(Debug, Clone)]
    struct Call {
        system_prompt: String,
        workdir: PathBuf,
        input: String,
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
                f.results.lock().unwrap().push_back(Err(anyhow!("fail {i}")));
            }
            f
        }
        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl Harness for Fake {
        async fn run(&self, system_prompt: &str, workdir: &Path, input: &str) -> Result<String> {
            self.calls.lock().unwrap().push(Call {
                system_prompt: system_prompt.to_owned(),
                workdir: workdir.to_owned(),
                input: input.to_owned(),
            });
            if self.sleepy {
                let n = input.rsplit(' ').next().unwrap().to_owned();
                self.log.lock().unwrap().push(format!("start {n}"));
                tokio::time::sleep(Duration::from_millis(n.parse().unwrap())).await;
                self.log.lock().unwrap().push(format!("end {n}"));
            }
            self.results.lock().unwrap().pop_front().unwrap_or_else(|| Ok("ok".into()))
        }
    }

    #[tokio::test]
    async fn writes_default_system_prompt_and_passes_it() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().join("rooms"));
        let r = room("!abc:example.org");
        assert_eq!(agent.reply(&r, "Kas", "hello").await, "ok");

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
        agent.reply(&r, "Kas", "one").await;
        std::fs::write(tmp.path().join(r.as_str()).join("SYSTEM.md"), "Be a pirate.").unwrap();
        agent.reply(&r, "Camille", "two").await;

        let calls = agent.harness.calls();
        assert_eq!(calls[1].system_prompt, "Be a pirate.");
        assert_eq!(calls[1].input, "[Camille] two");
    }

    #[tokio::test]
    async fn rooms_have_separate_folders() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        agent.reply(&room("!a:example.org"), "Kas", "x").await;
        agent.reply(&room("!b:example.org"), "Kas", "y").await;
        let calls = agent.harness.calls();
        assert_ne!(calls[0].workdir, calls[1].workdir);
        assert!(tmp.path().join("!a:example.org/SYSTEM.md").exists());
        assert!(tmp.path().join("!b:example.org/SYSTEM.md").exists());
    }

    #[tokio::test]
    async fn input_is_sender_in_brackets_then_body() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        agent.reply(&room("!a:example.org"), "Kas Buunk", "multi\nline").await;
        assert_eq!(agent.harness.calls()[0].input, "[Kas Buunk] multi\nline");
    }

    #[tokio::test]
    async fn harness_error_becomes_error_reply() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::failing(1), tmp.path().to_owned());
        let reply = agent.reply(&room("!a:example.org"), "Kas", "x").await;
        assert_eq!(reply, "_Agent error: fail 0_");
    }

    #[tokio::test]
    async fn breaker_opens_after_three_failures_and_skips_harness() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::failing(3), tmp.path().to_owned());
        let r = room("!a:example.org");
        for _ in 0..3 {
            assert!(agent.reply(&r, "Kas", "x").await.starts_with("_Agent error"));
        }
        let reply = agent.reply(&r, "Kas", "x").await;
        assert_eq!(reply, "_Agent paused after repeated failures; trying again in 5 min._");
        assert_eq!(agent.harness.calls().len(), 3);
        // The breaker is shared: other rooms are paused too.
        let other = agent.reply(&room("!b:example.org"), "Kas", "x").await;
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
            replies.push(agent.reply(&r, "Kas", "x").await);
        }
        assert_eq!(replies[2], "fine");
        // Two failures after the success do not open the breaker: call 6 runs.
        assert_eq!(replies[5], "ok");
        assert_eq!(agent.harness.calls().len(), 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn replies_in_one_room_are_serialized_in_arrival_order() {
        let tmp = tempfile::tempdir().unwrap();
        let agent =
            Arc::new(Agent::new(Fake { sleepy: true, ..Default::default() }, tmp.path().to_owned()));
        let r = room("!a:example.org");
        let mut tasks = Vec::new();
        // Long first, short later: without the room lock they would overlap.
        for ms in ["150", "50", "0"] {
            let (agent, r) = (agent.clone(), r.clone());
            tasks.push(tokio::spawn(async move { agent.reply(&r, "Kas", ms).await }));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        for t in tasks {
            assert_eq!(t.await.unwrap(), "ok");
        }
        let log = agent.harness.log.lock().unwrap().clone();
        assert_eq!(log, ["start 150", "end 150", "start 50", "end 50", "start 0", "end 0"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn different_rooms_run_concurrently() {
        let tmp = tempfile::tempdir().unwrap();
        let agent =
            Arc::new(Agent::new(Fake { sleepy: true, ..Default::default() }, tmp.path().to_owned()));
        let a = {
            let agent = agent.clone();
            tokio::spawn(async move { agent.reply(&room("!a:example.org"), "Kas", "150").await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let b = {
            let agent = agent.clone();
            tokio::spawn(async move { agent.reply(&room("!b:example.org"), "Kas", "10").await })
        };
        a.await.unwrap();
        b.await.unwrap();
        let log = agent.harness.log.lock().unwrap().clone();
        assert_eq!(log, ["start 150", "start 10", "end 10", "end 150"]);
    }
}
