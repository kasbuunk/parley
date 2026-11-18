//! Every agent run in a cgroup of its own, below parley's.
//!
//! On 15 Nov 2026 systemd-oomd killed all of `parley.service` (five runs at
//! once, 8 GB): the unit was one cgroup, so the heavy run took the bot and
//! every other conversation with it. With `Delegate=memory pids` on the
//! unit, parley owns its cgroup: it moves itself to `main/` and starts each
//! run in `run-<n>/` with `memory.max` and `memory.swap.max`. Then the
//! kernel stops a run at its own limit, and oomd, which only picks leaf
//! cgroups, picks the run, not parley.
//!
//! Without delegation (the cgroup is not ours to write) runs share
//! parley's cgroup as before; [`Pool::init`] says why in the log.

use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The limits every run gets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// `memory.max`, e.g. "6G".
    pub memory: String,
    /// `memory.swap.max`, e.g. "2G".
    pub swap: String,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            memory: "6G".into(),
            swap: "2G".into(),
        }
    }
}

/// parley's own (delegated) cgroup, where runs get their cgroups.
#[derive(Debug)]
pub struct Pool {
    base: PathBuf,
    limits: Limits,
    next: AtomicU64,
}

impl Pool {
    /// The cgroup this process is in, made ready for runs: parley moves
    /// to `main/`, and the memory and pids controllers are enabled for the
    /// children.
    pub fn init(limits: Limits) -> Result<Self> {
        let own = fs::read_to_string("/proc/self/cgroup").context("no /proc/self/cgroup")?;
        let path = own
            .lines()
            .find_map(|l| l.strip_prefix("0::"))
            .context("not on cgroup v2")?;
        Self::at(
            &Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/')),
            limits,
        )
    }

    /// [`Pool::init`] with the cgroup at `base` (a plain folder in tests).
    pub fn at(base: &Path, limits: Limits) -> Result<Self> {
        let pool = Self {
            base: base.to_owned(),
            limits,
            next: AtomicU64::new(0),
        };
        // Leftovers of an earlier parley: stop what still runs there.
        if let Ok(entries) = fs::read_dir(base) {
            for e in entries.flatten() {
                if e.file_name().to_string_lossy().starts_with("run-") {
                    let _ = fs::write(e.path().join("cgroup.kill"), "1");
                    let _ = fs::remove_dir(e.path());
                }
            }
        }
        let main = base.join("main");
        match fs::create_dir(&main) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("cannot make cgroups in {} (no Delegate=?)", base.display())
                });
            }
        }
        // A cgroup with processes of its own cannot hand controllers to
        // its children, so parley leaves for `main/` first.
        fs::write(main.join("cgroup.procs"), std::process::id().to_string())
            .context("cannot move parley to its main cgroup")?;
        fs::write(base.join("cgroup.subtree_control"), "+memory +pids")
            .context("cannot enable the memory controller")?;
        Ok(pool)
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// A new cgroup for one run, with the limits set.
    pub fn scope(&self) -> Result<Scope> {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let dir = self.base.join(format!("run-{n}"));
        fs::create_dir(&dir).with_context(|| format!("cannot make {}", dir.display()))?;
        let scope = Scope { dir };
        fs::write(scope.dir.join("memory.max"), &self.limits.memory)
            .context("cannot set memory.max")?;
        fs::write(scope.dir.join("memory.swap.max"), &self.limits.swap)
            .context("cannot set memory.swap.max")?;
        Ok(scope)
    }
}

/// One run's cgroup. Dropping it stops whatever still runs in it (the
/// run's leftovers, such as a server it started) and removes it.
#[derive(Debug)]
pub struct Scope {
    dir: PathBuf,
}

impl Scope {
    /// `cgroup.procs`, opened for the child to write itself into between
    /// fork and exec (see [`ClaudeCode`](crate::agent::ClaudeCode)).
    pub fn procs(&self) -> Result<fs::File> {
        fs::OpenOptions::new()
            .write(true)
            // Always there on cgroupfs; made here for plain folders (tests).
            .create(true)
            .open(self.dir.join("cgroup.procs"))
            .context("cannot open cgroup.procs")
    }

    /// How often the kernel killed a process here for want of memory.
    pub fn oom_kills(&self) -> u64 {
        fs::read_to_string(self.dir.join("memory.events"))
            .ok()
            .and_then(|events| {
                events.lines().find_map(|l| {
                    l.strip_prefix("oom_kill ")
                        .and_then(|n| n.trim().parse().ok())
                })
            })
            .unwrap_or(0)
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let _ = fs::write(self.dir.join("cgroup.kill"), "1");
        // The kill takes a moment; a cgroup with processes cannot go yet.
        let dir = std::mem::take(&mut self.dir);
        let remove = move || {
            for _ in 0..50 {
                if fs::remove_dir(&dir).is_ok() || !dir.exists() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            eprintln!("could not remove cgroup {}", dir.display());
        };
        std::thread::spawn(remove);
    }
}

/// Parse a size like "6G", "512M" or "max" as `memory.max` takes it.
pub fn check_size(s: &str) -> Result<()> {
    let s = s.trim();
    if s == "max" {
        return Ok(());
    }
    let digits = s.trim_end_matches(['K', 'M', 'G', 'T']);
    if digits.is_empty()
        || !digits.chars().all(|c| c.is_ascii_digit())
        || s.len() - digits.len() > 1
    {
        bail!("not a memory size: {s:?} (like 6G, 512M or max)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pool_moves_parley_to_main_and_enables_memory() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("run-7")).unwrap();
        let pool = Pool::at(dir.path(), Limits::default()).unwrap();
        assert_eq!(
            fs::read_to_string(dir.path().join("main/cgroup.procs")).unwrap(),
            std::process::id().to_string()
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("cgroup.subtree_control")).unwrap(),
            "+memory +pids"
        );
        // An earlier parley's run was told to stop.
        assert_eq!(
            fs::read_to_string(dir.path().join("run-7/cgroup.kill")).unwrap(),
            "1"
        );
        let scope = pool.scope().unwrap();
        let run = dir.path().join("run-0");
        assert_eq!(fs::read_to_string(run.join("memory.max")).unwrap(), "6G");
        assert_eq!(
            fs::read_to_string(run.join("memory.swap.max")).unwrap(),
            "2G"
        );
        assert_eq!(scope.oom_kills(), 0);
        fs::write(
            run.join("memory.events"),
            "low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n",
        )
        .unwrap();
        assert_eq!(scope.oom_kills(), 1);
        drop(scope);
        assert_eq!(fs::read_to_string(run.join("cgroup.kill")).unwrap(), "1");
        assert!(pool.scope().is_ok());
        assert!(dir.path().join("run-1").is_dir());
    }

    #[test]
    fn without_delegation_there_is_no_pool() {
        let dir = tempfile::tempdir().unwrap();
        let err = Pool::at(&dir.path().join("not-ours/deeper"), Limits::default()).unwrap_err();
        assert!(format!("{err:#}").contains("Delegate"), "{err:#}");
    }

    #[test]
    fn sizes() {
        for ok in ["6G", "512M", "max", "1073741824"] {
            assert!(check_size(ok).is_ok(), "{ok}");
        }
        for bad in ["", "G", "6GB", "lots", "-1"] {
            assert!(check_size(bad).is_err(), "{bad}");
        }
    }
}
