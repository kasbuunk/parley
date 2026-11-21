//! Runs a turn in the agent platform: `runtime = "agent-platform"` in
//! room.toml starts one Kubernetes Job per turn (gVisor, Cilium) through the
//! platform's `bin/spawn`, instead of a local `claude`.
//!
//! The pod can reach only Anthropic, so it gets no MCP tools and no
//! `post.sock`: parley posts the run's final answer. The pod is thrown away,
//! so there is no session to resume; context comes from a prompt mode.
//!
//! Job specs stay readable to whoever may read Jobs, so the message, its
//! history and SYSTEM.md are not in them. The Job holds only fixed flags, its
//! container waits on stdin, and parley sends the text with `kubectl attach
//! -i`. That also keeps it out of any process's argv. A Secret per turn was
//! the alternative, but parley's kubeconfig deliberately has no access to
//! Secrets.
//!
//! The Job is deleted when the turn ends in any way, `!stop` included: a
//! guard does it on drop. If parley itself dies, `activeDeadlineSeconds` ends
//! the Job and the template's TTL removes it.
//!
//! It fails closed: anything wrong here fails the turn. It never falls back
//! to a local run.

use std::{
    path::PathBuf,
    process::Stdio,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::{io::AsyncWriteExt, process::Command};

use crate::agent::{Output, Profile, ReplyMode, Run, parse_reply};

const NAMESPACE: &str = "agents";
const SECRET: &str = "agent-claude";
const RUNTIME_CLASS: &str = "gvisor";
/// Pod states that will not get better by waiting.
const STUCK: &[&str] = &[
    "ErrImagePull",
    "ImagePullBackOff",
    "InvalidImageName",
    "CreateContainerConfigError",
    "CreateContainerError",
];

/// What the room's profile may not combine with this runtime.
pub fn check(profile: &Profile, resumes: bool, mcp_config: bool) -> Result<()> {
    if profile.run_as.is_some() {
        bail!("room.toml: `runtime` cannot be combined with `run_as`");
    }
    if profile.workdir.is_some() {
        bail!("room.toml: `workdir` has no meaning in a pod");
    }
    if profile.trace.is_some() {
        bail!("room.toml: `trace` cannot read a pod's session");
    }
    if profile.mcp || mcp_config || profile.reply != ReplyMode::Text {
        bail!("room.toml: a pod has no MCP tools; use `reply = \"text\"` and no `mcp`");
    }
    if resumes {
        bail!("this room's context resumes a session, and a pod is thrown away; use `window N`");
    }
    Ok(())
}

/// How to reach the platform.
pub struct Platform {
    pub spawn: PathBuf,
    pub kubectl: String,
    pub kubeconfig: PathBuf,
    /// How often to look at the pod.
    pub poll: Duration,
}

impl Default for Platform {
    fn default() -> Self {
        let home = std::env::var_os("HOME").unwrap_or_default();
        Self {
            spawn: PathBuf::from(home).join("git/github.com/kasbuunk/agent-platform/bin/spawn"),
            kubectl: "kubectl".into(),
            kubeconfig: PathBuf::from("/var/lib/agent-platform/parley.kubeconfig"),
            poll: Duration::from_secs(1),
        }
    }
}

/// Deletes the Job when the turn ends, however it ends. From `Drop`, so it
/// must not wait: a thread sees the delete through.
struct JobGuard {
    kubectl: String,
    kubeconfig: PathBuf,
    name: String,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        let mut delete = std::process::Command::new(&self.kubectl);
        delete
            .args(["delete", "job", &self.name, "-n", NAMESPACE])
            .args(["--ignore-not-found", "--wait=false"])
            .env_clear()
            .envs(base_env())
            .env("KUBECONFIG", &self.kubeconfig)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let name = std::mem::take(&mut self.name);
        std::thread::spawn(move || match delete.output() {
            Ok(out) if out.status.success() => {}
            Ok(out) => eprintln!(
                "could not delete job {name}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
            Err(e) => eprintln!("could not delete job {name}: {e}"),
        });
    }
}

/// All a child gets of parley's environment, which holds its secrets.
fn base_env() -> impl Iterator<Item = (&'static str, std::ffi::OsString)> {
    ["PATH", "HOME"]
        .into_iter()
        .filter_map(|k| std::env::var_os(k).map(|v| (k, v)))
}

#[derive(Debug, PartialEq)]
enum Pod {
    Pending,
    Running(String),
    Succeeded(String),
    Failed(String),
}

#[derive(Deserialize)]
struct PodList {
    items: Vec<serde_json::Value>,
}

fn pod_state(list: &[u8]) -> Result<Pod> {
    let list: PodList = serde_json::from_slice(list).context("unreadable pod list")?;
    let Some(pod) = list.items.first() else {
        return Ok(Pod::Pending);
    };
    let name = pod["metadata"]["name"]
        .as_str()
        .context("pod without a name")?
        .to_owned();
    if let Some(reason) = pod["status"]["containerStatuses"][0]["state"]["waiting"]["reason"]
        .as_str()
        .filter(|r| STUCK.contains(r))
    {
        bail!("the pod cannot start: {reason}");
    }
    Ok(match pod["status"]["phase"].as_str() {
        Some("Running") => Pod::Running(name),
        Some("Succeeded") => Pod::Succeeded(name),
        Some("Failed") => Pod::Failed(name),
        _ => Pod::Pending,
    })
}

/// The Job as `spawn --print` made it, made to wait for its input on stdin.
fn with_stdin(job: &[u8], deadline: u64) -> Result<Vec<u8>> {
    let mut job: serde_json::Value = serde_json::from_slice(job).context("spawn printed no Job")?;
    job["spec"]["activeDeadlineSeconds"] = deadline.into();
    job["metadata"]["labels"]["app.kubernetes.io/managed-by"] = "parley".into();
    // Nothing in the cluster enforces gVisor, so a Job without it is refused
    // here rather than run on the host's kernel.
    let runtime = job["spec"]["template"]["spec"]["runtimeClassName"].as_str();
    if runtime != Some(RUNTIME_CLASS) {
        bail!("the Job's runtime class is {runtime:?}, not {RUNTIME_CLASS}");
    }
    let container = job["spec"]["template"]["spec"]["containers"]
        .get_mut(0)
        .context("the Job has no container")?;
    container["stdin"] = true.into();
    container["stdinOnce"] = true.into();
    container["tty"] = false.into();
    // The image is loaded into containerd; there is no registry to pull from.
    container["imagePullPolicy"] = "Never".into();
    // The sandbox is killed at once on this cluster when requests are below
    // limits.
    let limits = container["resources"]["limits"].clone();
    if !limits.is_object() {
        bail!("the Job's container has no resource limits");
    }
    container["resources"]["requests"] = limits;
    Ok(serde_json::to_vec(&job)?)
}

/// The fixed flags of a run; nothing of the conversation.
fn claude_args(profile: &Profile) -> Vec<String> {
    let mut args: Vec<String> = vec!["-p".into(), "--tools".into(), profile.tools.clone()];
    args.extend(
        [
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
        ]
        .map(String::from),
    );
    if let Some(mode) = &profile.permission_mode {
        args.extend(["--permission-mode".into(), mode.clone()]);
    }
    if let Some(model) = &profile.model {
        args.extend(["--model".into(), model.clone()]);
    }
    args.extend(["--output-format".into(), "json".into()]);
    args
}

/// What goes over stdin. The system prompt travels with the message, since a
/// flag for it would put it in the Job spec.
fn payload(system_prompt: &str, input: &str) -> String {
    format!("<room-instructions>\n{system_prompt}\n</room-instructions>\n\n{input}")
}

fn job_name() -> Result<String> {
    use std::io::Read;
    let mut b = [0u8; 6];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
    Ok(format!("parley-{hex}"))
}

impl Platform {
    fn kubectl(&self) -> Command {
        let mut c = Command::new(&self.kubectl);
        c.env_clear()
            .envs(base_env())
            .env("KUBECONFIG", &self.kubeconfig)
            .stdin(Stdio::null())
            .kill_on_drop(true);
        c
    }

    async fn pod(&self, job: &str) -> Result<Pod> {
        let out = self
            .kubectl()
            .args(["get", "pods", "-n", NAMESPACE, "-o", "json", "-l"])
            .arg(format!("job-name={job}"))
            .output()
            .await
            .context("could not run kubectl")?;
        if !out.status.success() {
            bail!(
                "could not read the pod: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        pod_state(&out.stdout)
    }

    async fn logs(&self, pod: &str) -> Result<Vec<u8>> {
        let out = self
            .kubectl()
            .args(["logs", pod, "-c", "agent", "-n", NAMESPACE])
            .output()
            .await
            .context("could not run kubectl")?;
        if !out.status.success() {
            bail!(
                "could not read the pod's output: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(out.stdout)
    }

    async fn failure(&self, pod: &str) -> anyhow::Error {
        let tail = match self.logs(pod).await {
            Ok(logs) => {
                let text = String::from_utf8_lossy(&logs);
                let text = text.trim();
                let start = text.len().saturating_sub(500);
                let start = (start..=text.len())
                    .find(|i| text.is_char_boundary(*i))
                    .unwrap_or(start);
                text[start..].to_owned()
            }
            Err(e) => format!("{e:#}"),
        };
        anyhow::anyhow!("the pod failed: {tail}")
    }

    pub async fn run(&self, run: &Run<'_>) -> Result<Output> {
        check(run.profile, run.session.is_some(), run.mcp_config.is_some())?;
        // Opened, not only looked at: one parley cannot read is no kubeconfig.
        std::fs::File::open(&self.kubeconfig).with_context(|| {
            format!(
                "not running: no usable kubeconfig at {}",
                self.kubeconfig.display()
            )
        })?;
        let name = job_name()?;
        let timeout = Duration::from_secs(run.profile.timeout_secs);
        let spawned = Command::new(&self.spawn)
            .arg(&name)
            .args([
                "--namespace",
                NAMESPACE,
                "--secret",
                SECRET,
                "--runtime-class",
                RUNTIME_CLASS,
                "--print",
                "--",
            ])
            .args(claude_args(run.profile))
            .env_clear()
            .envs(base_env())
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .with_context(|| format!("could not start {}", self.spawn.display()))?;
        if !spawned.status.success() {
            bail!(
                "spawn failed: {}",
                String::from_utf8_lossy(&spawned.stderr).trim()
            );
        }
        // Outlives parley's own timeout a little, so that one reports first.
        let job = with_stdin(&spawned.stdout, run.profile.timeout_secs + 60)?;

        // Before the create: a create that fails halfway leaves nothing.
        let _guard = JobGuard {
            kubectl: self.kubectl.clone(),
            kubeconfig: self.kubeconfig.clone(),
            name: name.clone(),
        };
        let text = payload(run.system_prompt, run.input);
        tokio::time::timeout(timeout, self.turn(&name, &job, &text))
            .await
            .with_context(|| format!("no answer from the pod within {}s", timeout.as_secs()))?
    }

    async fn turn(&self, name: &str, job: &[u8], text: &str) -> Result<Output> {
        let mut create = self
            .kubectl()
            .args(["create", "-f", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("could not run kubectl")?;
        let mut stdin = create.stdin.take().context("kubectl has no stdin")?;
        stdin.write_all(job).await.context("kubectl create")?;
        drop(stdin);
        let created = create.wait_with_output().await?;
        if !created.status.success() {
            bail!(
                "could not create the Job: {}",
                String::from_utf8_lossy(&created.stderr).trim()
            );
        }

        let pod = loop {
            match self.pod(name).await? {
                Pod::Running(pod) => break pod,
                Pod::Succeeded(pod) | Pod::Failed(pod) => return Err(self.failure(&pod).await),
                Pod::Pending => tokio::time::sleep(self.poll).await,
            }
        };

        // The Job is deleted after the turn; the log keeps which pod ran it.
        println!("platform: job {name} runs in pod {pod}");

        let mut attach = self
            .kubectl()
            .args(["attach", "-i", &pod, "-c", "agent", "-n", NAMESPACE])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("could not run kubectl attach")?;
        let mut stdin = attach.stdin.take().context("kubectl has no stdin")?;
        // A failed write means attach died; its exit status below says why.
        let _ = stdin.write_all(text.as_bytes()).await;
        drop(stdin);

        let mut attached = Box::pin(attach.wait_with_output());
        let mut attaching = true;
        let started = Instant::now();
        loop {
            if attaching {
                tokio::select! {
                    done = &mut attached => {
                        attaching = false;
                        let done = done.context("kubectl attach")?;
                        if !done.status.success()
                            && !matches!(self.pod(name).await?, Pod::Succeeded(_) | Pod::Failed(_))
                        {
                            bail!(
                                "could not send the message to the pod: {}",
                                String::from_utf8_lossy(&done.stderr).trim()
                            );
                        }
                    }
                    _ = tokio::time::sleep(self.poll) => {}
                }
            } else {
                tokio::time::sleep(self.poll).await;
            }
            match self.pod(name).await? {
                Pod::Succeeded(pod) => {
                    let logs = self.logs(&pod).await?;
                    // Only the last line is Claude Code's JSON.
                    let last = logs
                        .split(|b| *b == b'\n')
                        .rfind(|l| !l.iter().all(u8::is_ascii_whitespace))
                        .unwrap_or_default();
                    let mut out = parse_reply(last, || {
                        format!("the pod's output after {:?}", started.elapsed())
                    })?;
                    // Nothing to resume: the pod is gone.
                    out.session = None;
                    return Ok(out);
                }
                Pod::Failed(pod) => return Err(self.failure(&pod).await),
                Pod::Running(_) | Pod::Pending => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{PromptMode, Run};
    use std::path::Path;

    fn write_script(path: &Path, body: &str) {
        // Through `sh`, for the reason in agent.rs's `script`: no "Text file
        // busy" from a child forked by a parallel test.
        use std::io::Write;
        let mut sh = std::process::Command::new("/bin/sh")
            .args(["-c", "cat > \"$1\" && chmod 755 \"$1\"", "sh"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        sh.stdin
            .take()
            .unwrap()
            .write_all(format!("#!/bin/sh\n{body}\n").as_bytes())
            .unwrap();
        assert!(sh.wait().unwrap().success());
    }

    const JOB: &str = r#"{"metadata":{"labels":{}},"spec":{"template":{"spec":{"runtimeClassName":"gvisor","containers":[{"name":"agent","imagePullPolicy":"IfNotPresent","resources":{"requests":{"cpu":"250m","memory":"512Mi"},"limits":{"cpu":"2","memory":"2Gi"}}}]}}}}"#;

    #[test]
    fn the_job_cannot_be_pulled_or_leave_gvisor() {
        let job: serde_json::Value =
            serde_json::from_slice(&with_stdin(JOB.as_bytes(), 90).unwrap()).unwrap();
        let container = &job["spec"]["template"]["spec"]["containers"][0];
        assert_eq!(container["imagePullPolicy"], "Never");
        assert_eq!(
            container["resources"]["requests"],
            container["resources"]["limits"]
        );

        let without = JOB.replace(r#""runtimeClassName":"gvisor","#, "");
        assert!(with_stdin(without.as_bytes(), 90).is_err());
        let other = JOB.replace(r#""gvisor""#, r#""runc""#);
        assert!(with_stdin(other.as_bytes(), 90).is_err());
    }

    /// A fake `spawn` and `kubectl` in `dir`; `phase` is what the pod reports
    /// once attached, `waiting` a container waiting reason.
    fn fake(dir: &Path, phase: &str, waiting: &str) -> Platform {
        let d = dir.display();
        write_script(
            &dir.join("spawn"),
            &format!("printf '%s\\n' \"$@\" > {d}/spawn.args\nprintf '%s' '{JOB}'"),
        );
        write_script(
            &dir.join("kubectl"),
            &format!(
                r#"echo "KUBECONFIG=$KUBECONFIG SECRET=${{PARLEY_SECRET:-none}}" >> {d}/env
case "$1" in
create) cat > {d}/job.json ;;
get)
  if [ -f {d}/attached ]; then p={phase}; else p=Running; fi
  printf '{{"items":[{{"metadata":{{"name":"pod-1"}},"status":{{"phase":"%s","containerStatuses":[{{"state":{{"waiting":{{"reason":"{waiting}"}}}}}}]}}}}]}}' "$p" ;;
attach) cat > {d}/stdin; touch {d}/attached ;;
logs) echo 'some noise'; echo '{{"result":"from the pod","session_id":"s1"}}' ;;
delete) echo "$*" >> {d}/deleted ;;
esac"#
            ),
        );
        let kubeconfig = dir.join("kubeconfig");
        std::fs::write(&kubeconfig, "fake").unwrap();
        Platform {
            spawn: dir.join("spawn"),
            kubectl: dir.join("kubectl").display().to_string(),
            kubeconfig,
            poll: Duration::from_millis(10),
        }
    }

    fn profile(timeout_secs: u64) -> Profile {
        Profile {
            timeout_secs,
            prompt: PromptMode::Replace,
            ..Profile::default()
        }
    }

    fn run<'a>(p: &'a Profile, dir: &'a Path) -> Run<'a> {
        Run {
            profile: p,
            system_prompt: "SECRET-SYSTEM",
            workdir: dir,
            input: "[kas] SECRET-MESSAGE",
            session: None,
            new_session: None,
            persist: false,
            mcp_config: None,
        }
    }

    async fn deleted(dir: &Path) -> String {
        for _ in 0..200 {
            if let Ok(text) = std::fs::read_to_string(dir.join("deleted")) {
                return text;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the Job was not deleted");
    }

    #[tokio::test]
    async fn answers_from_the_pod_with_the_text_only_on_stdin() {
        let tmp = tempfile::tempdir().unwrap();
        let p = profile(30);
        let out = fake(tmp.path(), "Succeeded", "")
            .run(&run(&p, tmp.path()))
            .await
            .unwrap();
        assert_eq!(out.text, "from the pod");
        assert_eq!(out.session, None);

        let stdin = std::fs::read_to_string(tmp.path().join("stdin")).unwrap();
        assert!(stdin.contains("SECRET-SYSTEM") && stdin.contains("SECRET-MESSAGE"));
        for file in ["job.json", "spawn.args"] {
            let text = std::fs::read_to_string(tmp.path().join(file)).unwrap();
            assert!(!text.contains("SECRET-"), "{file}: {text}");
        }
        let job: serde_json::Value =
            serde_json::from_slice(&std::fs::read(tmp.path().join("job.json")).unwrap()).unwrap();
        let container = &job["spec"]["template"]["spec"]["containers"][0];
        assert_eq!(container["stdin"], true);
        assert_eq!(container["stdinOnce"], true);
        assert_eq!(job["spec"]["activeDeadlineSeconds"], 90);
        let args = std::fs::read_to_string(tmp.path().join("spawn.args")).unwrap();
        assert!(args.contains("agent-claude") && args.contains("--print"));
        assert!(deleted(tmp.path()).await.contains("delete job parley-"));
    }

    #[tokio::test]
    async fn no_kubeconfig_fails_without_starting_anything() {
        let tmp = tempfile::tempdir().unwrap();
        let mut platform = fake(tmp.path(), "Succeeded", "");
        platform.kubeconfig = tmp.path().join("missing");
        let p = profile(30);
        let e = platform.run(&run(&p, tmp.path())).await.err().unwrap();
        assert!(format!("{e:#}").contains("no usable kubeconfig"), "{e:#}");
        assert!(!tmp.path().join("spawn.args").exists());
        assert!(!tmp.path().join("env").exists());
    }

    #[tokio::test]
    async fn a_failed_pod_is_an_error_and_is_deleted() {
        let tmp = tempfile::tempdir().unwrap();
        let p = profile(30);
        let e = fake(tmp.path(), "Failed", "")
            .run(&run(&p, tmp.path()))
            .await
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("the pod failed"), "{e:#}");
        deleted(tmp.path()).await;
    }

    #[tokio::test]
    async fn a_pod_that_cannot_start_fails_at_once() {
        let tmp = tempfile::tempdir().unwrap();
        let p = profile(30);
        let e = fake(tmp.path(), "Running", "ImagePullBackOff")
            .run(&run(&p, tmp.path()))
            .await
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("ImagePullBackOff"), "{e:#}");
        deleted(tmp.path()).await;
    }

    #[tokio::test]
    async fn a_timeout_deletes_the_job() {
        let tmp = tempfile::tempdir().unwrap();
        let p = profile(1);
        let e = fake(tmp.path(), "Running", "")
            .run(&run(&p, tmp.path()))
            .await
            .err()
            .unwrap();
        assert!(format!("{e:#}").contains("within 1s"), "{e:#}");
        deleted(tmp.path()).await;
    }

    #[tokio::test]
    async fn a_stopped_turn_deletes_the_job() {
        let tmp = tempfile::tempdir().unwrap();
        let platform = fake(tmp.path(), "Running", "");
        let p = profile(30);
        let r = run(&p, tmp.path());
        // `!stop` aborts the task, which drops this future.
        let stopped = tokio::time::timeout(Duration::from_millis(300), platform.run(&r)).await;
        assert!(stopped.is_err());
        deleted(tmp.path()).await;
    }

    #[test]
    fn refuses_what_a_pod_cannot_do() {
        let ok = Profile::default();
        check(&ok, false, false).unwrap();
        assert!(check(&ok, true, false).is_err());
        assert!(check(&ok, false, true).is_err());
        for bad in [
            Profile {
                run_as: Some("paul".into()),
                ..Profile::default()
            },
            Profile {
                workdir: Some("/x".into()),
                ..Profile::default()
            },
            Profile {
                trace: Some("https://x".into()),
                ..Profile::default()
            },
            Profile {
                mcp: true,
                ..Profile::default()
            },
            Profile {
                reply: ReplyMode::Tool,
                ..Profile::default()
            },
        ] {
            assert!(check(&bad, false, false).is_err());
        }
    }
}
