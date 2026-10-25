//! Turning a room message into an agent reply.
//!
//! Every room has a folder `rooms/<room id>/` holding:
//! - `SYSTEM.md`: the system prompt (a default is written on first use);
//! - `room.toml`: optional profile: where the agent runs, which tools and
//!   settings it gets, who may be in the room, whether it keeps a session.
//!   Without it a room is locked down: no tools, no settings, no memory.
//! - `session`: the harness session ID, when the profile says `resume`.
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
    pub resume: bool,
    /// Whether SYSTEM.md replaces the harness's own system prompt or is
    /// appended to it.
    pub prompt: PromptMode,
    pub model: Option<String>,
    pub timeout_secs: u64,
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
            prompt: PromptMode::Replace,
            model: None,
            timeout_secs: 300,
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
}

pub struct Output {
    pub text: String,
    /// Session ID to continue next time, if the harness keeps sessions.
    pub session: Option<String>,
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
        args.extend(["--setting-sources".into(), p.settings.join(",")]);
        if p.settings.is_empty() {
            args.push("--disable-slash-commands".into());
        }
        if !p.resume {
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
            Some(text) if !reply.is_error => Ok(Output { text, session: reply.session_id }),
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

    /// The reply to post for `body`, sent by `sender` in `room_id`, whose
    /// human members (everyone but the bot) are `members`.
    pub async fn reply(
        &self,
        room_id: &RoomId,
        sender: &str,
        body: &str,
        members: &[OwnedUserId],
    ) -> String {
        let lock = self.room_locks.lock().unwrap().entry(room_id.to_owned()).or_default().clone();
        let _turn = lock.lock().await;

        if let State::Open(left) = self.breaker.lock().unwrap().state() {
            return format!(
                "_Agent paused after repeated failures; trying again in {} min._",
                left.as_secs().div_ceil(60)
            );
        }

        match self.run(room_id, sender, body, members).await {
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

    async fn run(
        &self,
        room_id: &RoomId,
        sender: &str,
        body: &str,
        members: &[OwnedUserId],
    ) -> Result<String> {
        // Room IDs come from the homeserver; still, never let one name a
        // folder outside rooms/.
        let id = room_id.as_str();
        if id.contains('/') || id.contains("..") {
            bail!("refusing room ID {id:?}");
        }
        let dir = self.rooms_dir.join(id);
        tokio::fs::create_dir_all(&dir).await?;

        let profile = match tokio::fs::read_to_string(dir.join("room.toml")).await {
            Ok(text) => toml::from_str(&text).context("room.toml is invalid")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Profile::default(),
            Err(e) => return Err(e).context("could not read room.toml"),
        };

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

        let session_file = dir.join("session");
        let session = if profile.resume {
            tokio::fs::read_to_string(&session_file).await.ok().map(|s| s.trim().to_owned())
        } else {
            None
        };

        let input = format!("[{sender}] {body}");
        let workdir = profile.workdir.clone().unwrap_or_else(|| dir.clone());
        let run = Run {
            profile: &profile,
            system_prompt: &system_prompt,
            workdir: &workdir,
            input: &input,
            session: session.as_deref(),
        };
        match self.harness.run(&run).await {
            Ok(out) => {
                if let (true, Some(id)) = (profile.resume, &out.session) {
                    tokio::fs::write(&session_file, id).await?;
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

    /// A fake-program harness plus the timeout its runs get.
    struct Cc(ClaudeCode, Duration);

    fn claude(program: String, timeout: Duration) -> Cc {
        Cc(ClaudeCode { program }, timeout)
    }

    impl Cc {
        /// Run with the default (locked-down) profile. Profiles count
        /// whole seconds; tests want less, so the timeout is applied here.
        async fn run(&self, system_prompt: &str, workdir: &Path, input: &str) -> Result<String> {
            let profile = Profile { timeout_secs: 60, ..Profile::default() };
            let run = Run { profile: &profile, system_prompt, workdir, input, session: None };
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
        async fn run(&self, run: &Run<'_>) -> Result<Output> {
            let input = run.input;
            self.calls.lock().unwrap().push(Call {
                system_prompt: run.system_prompt.to_owned(),
                workdir: run.workdir.to_owned(),
                input: input.to_owned(),
            });
            if self.sleepy {
                let n = input.rsplit(' ').next().unwrap().to_owned();
                self.log.lock().unwrap().push(format!("start {n}"));
                tokio::time::sleep(Duration::from_millis(n.parse().unwrap())).await;
                self.log.lock().unwrap().push(format!("end {n}"));
            }
            let text = self.results.lock().unwrap().pop_front().unwrap_or_else(|| Ok("ok".into()))?;
            Ok(Output { text, session: None })
        }
    }

    #[tokio::test]
    async fn writes_default_system_prompt_and_passes_it() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().join("rooms"));
        let r = room("!abc:example.org");
        assert_eq!(agent.reply(&r, "Kas", "hello", &[]).await, "ok");

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
        agent.reply(&r, "Kas", "one", &[]).await;
        std::fs::write(tmp.path().join(r.as_str()).join("SYSTEM.md"), "Be a pirate.").unwrap();
        agent.reply(&r, "Camille", "two", &[]).await;

        let calls = agent.harness.calls();
        assert_eq!(calls[1].system_prompt, "Be a pirate.");
        assert_eq!(calls[1].input, "[Camille] two");
    }

    #[tokio::test]
    async fn rooms_have_separate_folders() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        agent.reply(&room("!a:example.org"), "Kas", "x", &[]).await;
        agent.reply(&room("!b:example.org"), "Kas", "y", &[]).await;
        let calls = agent.harness.calls();
        assert_ne!(calls[0].workdir, calls[1].workdir);
        assert!(tmp.path().join("!a:example.org/SYSTEM.md").exists());
        assert!(tmp.path().join("!b:example.org/SYSTEM.md").exists());
    }

    #[tokio::test]
    async fn input_is_sender_in_brackets_then_body() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::default(), tmp.path().to_owned());
        agent.reply(&room("!a:example.org"), "Kas Buunk", "multi\nline", &[]).await;
        assert_eq!(agent.harness.calls()[0].input, "[Kas Buunk] multi\nline");
    }

    #[tokio::test]
    async fn harness_error_becomes_error_reply() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::failing(1), tmp.path().to_owned());
        let reply = agent.reply(&room("!a:example.org"), "Kas", "x", &[]).await;
        assert_eq!(reply, "_Agent error: fail 0_");
    }

    #[tokio::test]
    async fn breaker_opens_after_three_failures_and_skips_harness() {
        let tmp = tempfile::tempdir().unwrap();
        let agent = Agent::new(Fake::failing(3), tmp.path().to_owned());
        let r = room("!a:example.org");
        for _ in 0..3 {
            assert!(agent.reply(&r, "Kas", "x", &[]).await.starts_with("_Agent error"));
        }
        let reply = agent.reply(&r, "Kas", "x", &[]).await;
        assert_eq!(reply, "_Agent paused after repeated failures; trying again in 5 min._");
        assert_eq!(agent.harness.calls().len(), 3);
        // The breaker is shared: other rooms are paused too.
        let other = agent.reply(&room("!b:example.org"), "Kas", "x", &[]).await;
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
            replies.push(agent.reply(&r, "Kas", "x", &[]).await);
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
            tasks.push(tokio::spawn(async move { agent.reply(&r, "Kas", ms, &[]).await }));
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
            tokio::spawn(async move { agent.reply(&room("!a:example.org"), "Kas", "150", &[]).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        let b = {
            let agent = agent.clone();
            tokio::spawn(async move { agent.reply(&room("!b:example.org"), "Kas", "10", &[]).await })
        };
        a.await.unwrap();
        b.await.unwrap();
        let log = agent.harness.log.lock().unwrap().clone();
        assert_eq!(log, ["start 150", "start 10", "end 10", "end 150"]);
    }
}
