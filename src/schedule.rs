//! Schedules: runs that start without a message. An agent (through the
//! `schedule` tool) or Kas (by asking one) plans a prompt for later, once
//! or repeatedly; when it is due, parley starts a run in that room and
//! thread as if the prompt had been sent there.
//!
//! The schedule kinds come from transmit (github.com/kasbuunk/transmit):
//! delayed (`at`), interval (`every`, from a first time) and cron, each
//! with a number of iterations or none (forever).
//!
//! Each schedule is a JSON file `rooms/<room>/schedules/<id>.json`, so they
//! live on the NAS and survive restarts; a finished or cancelled one moves
//! to `schedules/done/`. Next to it, `<id>.md` is the schedule's
//! scratchboard: a run ends, but what it hands off (`handoff` tool) is
//! shown to the next run of the same schedule.
//!
//! With `only_if_new`, parley first checks deterministically whether
//! anyone wrote in the room (or the schedule's thread) since the last run;
//! if not, the fire is skipped and no agent starts.

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Local, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// When a schedule fires.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum When {
    /// Once, at this time (transmit's `Delayed`).
    At(DateTime<Utc>),
    /// From `first` on, every `secs` seconds (transmit's `Interval`).
    Every { first: DateTime<Utc>, secs: u64 },
    /// At the times of a cron expression in local time, with seconds:
    /// `sec min hour day month weekday [year]` (transmit's `Cron`).
    Cron { expr: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    /// The room's ID.
    pub room: String,
    /// The thread's root event, or `None` for the main timeline.
    pub thread: Option<String>,
    pub prompt: String,
    pub when: When,
    /// How many runs in all; `None` is forever (transmit's `Iterate`).
    pub times: Option<u32>,
    /// Runs so far.
    #[serde(default)]
    pub fired: u32,
    /// The next fire, or `None` once the schedule is over.
    pub next: Option<DateTime<Utc>>,
    /// Skip a fire when nobody wrote since the last run (or since the
    /// schedule was made).
    #[serde(default)]
    pub only_if_new: bool,
    /// When it last ran (or was made): what `only_if_new` compares with.
    pub since: DateTime<Utc>,
    pub created: DateTime<Utc>,
    /// Who made it: a user ID, or `agent`.
    pub by: String,
}

/// What the `schedule` tool asks for, before it is a [`Job`].
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub prompt: String,
    /// RFC 3339 time, or local `YYYY-MM-DD HH:MM`.
    pub at: Option<String>,
    /// A delay like `90s`, `20m`, `1h30m`, `2d`.
    pub after: Option<String>,
    pub every: Option<String>,
    pub cron: Option<String>,
    pub times: Option<u32>,
    pub only_if_new: bool,
}

impl Request {
    pub fn from_args(args: &Value) -> Result<Self> {
        let s = |k: &str| {
            args.get(k)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
        };
        let times = match args.get("times") {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .context("`times` must be a whole number ≥ 1")?,
            ),
        };
        Ok(Self {
            prompt: s("prompt").context("`prompt` is required")?,
            at: s("at"),
            after: s("in"),
            every: s("every"),
            cron: s("cron"),
            times,
            only_if_new: args
                .get("only_if_new")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

    /// The job this asks for, made `now` in `room`/`thread` by `by`.
    pub fn job(
        self,
        id: String,
        room: &str,
        thread: Option<&str>,
        by: &str,
        now: DateTime<Utc>,
    ) -> Result<Job> {
        let start = match (&self.at, &self.after) {
            (Some(_), Some(_)) => bail!("give `at` or `in`, not both"),
            (Some(at), None) => Some(parse_time(at)?),
            (None, Some(d)) => Some(now + chrono::Duration::seconds(parse_duration(d)? as i64)),
            (None, None) => None,
        };
        let (when, times) = match (&self.every, &self.cron) {
            (Some(_), Some(_)) => bail!("give `every` or `cron`, not both"),
            (Some(every), None) => {
                let secs = parse_duration(every)?;
                if secs < 60 {
                    bail!("`every` must be at least a minute");
                }
                let first = start.unwrap_or(now + chrono::Duration::seconds(secs as i64));
                (When::Every { first, secs }, self.times)
            }
            (None, Some(expr)) => {
                if start.is_some() {
                    bail!("`cron` takes no `at` or `in`");
                }
                expr.parse::<cron::Schedule>()
                    .map_err(|e| anyhow::anyhow!("bad cron expression {expr:?}: {e}"))?;
                (When::Cron { expr: expr.clone() }, self.times)
            }
            (None, None) => match start {
                Some(t) => (When::At(t), Some(1)),
                None => bail!("say when: `at`, `in`, `every` or `cron`"),
            },
        };
        let mut job = Job {
            id,
            room: room.to_owned(),
            thread: thread.map(str::to_owned),
            prompt: self.prompt,
            when,
            times,
            fired: 0,
            next: None,
            only_if_new: self.only_if_new,
            since: now,
            created: now,
            by: by.to_owned(),
        };
        job.next = job.first();
        if job.next.is_none() {
            bail!("this schedule never fires");
        }
        Ok(job)
    }
}

impl Job {
    fn over(&self) -> bool {
        self.times.is_some_and(|t| self.fired >= t)
    }

    /// The first fire.
    fn first(&self) -> Option<DateTime<Utc>> {
        match &self.when {
            When::At(t) => Some(*t),
            When::Every { first, .. } => Some(*first),
            When::Cron { .. } => self.after(Utc::now()),
        }
    }

    /// The next fire strictly after `now`, given the runs so far. Fires
    /// missed while parley was down are not made up, only the next one.
    pub fn after(&self, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
        if self.over() {
            return None;
        }
        match &self.when {
            When::At(_) => None,
            When::Every { first, secs } => {
                if now < *first {
                    return Some(*first);
                }
                let k = (now - *first).num_seconds() as u64 / secs + 1;
                Some(*first + chrono::Duration::seconds((k * secs) as i64))
            }
            When::Cron { expr } => {
                let s: cron::Schedule = expr.parse().ok()?;
                s.after(&now.with_timezone(&Local))
                    .next()
                    .map(|t| t.with_timezone(&Utc))
            }
        }
    }

    /// One line about it, for lists.
    pub fn describe(&self) -> String {
        let local = |t: &DateTime<Utc>| t.with_timezone(&Local).format("%Y-%m-%d %H:%M");
        let when = match &self.when {
            When::At(t) => format!("at {}", local(t)),
            When::Every { secs, .. } => format!("every {}", show_duration(*secs)),
            When::Cron { expr } => format!("cron `{expr}`"),
        };
        let runs = match self.times {
            Some(t) => format!("{} of {t} runs", self.fired),
            None => format!("{} runs", self.fired),
        };
        let next = match &self.next {
            Some(t) => format!("next {}", local(t)),
            None => "over".into(),
        };
        let place = match &self.thread {
            Some(t) => format!("thread {t}"),
            None => "main timeline".into(),
        };
        let gate = if self.only_if_new {
            ", only if new messages"
        } else {
            ""
        };
        format!(
            "`{}` {when}, {runs}, {next}{gate} ({place}): {}",
            self.id,
            self.prompt.lines().next().unwrap_or("")
        )
    }
}

/// `90s`, `20m`, `1h30m`, `2d`, `1w`, or plain seconds.
pub fn parse_duration(text: &str) -> Result<u64> {
    let text = text.trim();
    if let Ok(n) = text.parse::<u64>() {
        return Ok(n);
    }
    let (mut total, mut num) = (0u64, String::new());
    for c in text.chars() {
        if c.is_ascii_digit() {
            num.push(c);
            continue;
        }
        if c.is_whitespace() {
            continue;
        }
        let unit = match c {
            's' => 1,
            'm' => 60,
            'h' => 3600,
            'd' => 86_400,
            'w' => 604_800,
            _ => bail!("bad duration {text:?}: use e.g. 90s, 20m, 1h30m, 2d"),
        };
        let n: u64 = num
            .parse()
            .with_context(|| format!("bad duration {text:?}"))?;
        total += n * unit;
        num.clear();
    }
    if !num.is_empty() || total == 0 && !text.starts_with('0') {
        bail!("bad duration {text:?}: use e.g. 90s, 20m, 1h30m, 2d");
    }
    Ok(total)
}

fn show_duration(secs: u64) -> String {
    let mut out = String::new();
    let mut left = secs;
    for (unit, size) in [("d", 86_400), ("h", 3600), ("m", 60), ("s", 1)] {
        if left >= size {
            out.push_str(&format!("{}{unit}", left / size));
            left %= size;
        }
    }
    if out.is_empty() { "0s".into() } else { out }
}

/// RFC 3339, or local `YYYY-MM-DD HH:MM` (or with a `T`).
pub fn parse_time(text: &str) -> Result<DateTime<Utc>> {
    if let Ok(t) = DateTime::parse_from_rfc3339(text) {
        return Ok(t.with_timezone(&Utc));
    }
    for f in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(t) = chrono::NaiveDateTime::parse_from_str(text, f)
            && let Some(t) = Local.from_local_datetime(&t).earliest()
        {
            return Ok(t.with_timezone(&Utc));
        }
    }
    bail!("bad time {text:?}: use RFC 3339 or local YYYY-MM-DD HH:MM")
}

/// The schedules of all rooms, on disk.
pub struct Store {
    rooms_dir: PathBuf,
    /// Firing, cancelling and making schedules read and rewrite files;
    /// one at a time, so a cancel is never undone by a fire.
    lock: Mutex<()>,
    /// Woken when a schedule is made, so one due now does not wait a tick.
    pub wake: tokio::sync::Notify,
}

impl Store {
    pub fn new(rooms_dir: PathBuf) -> Self {
        Self {
            rooms_dir,
            lock: Mutex::new(()),
            wake: tokio::sync::Notify::new(),
        }
    }

    fn dir(&self, room: &str) -> Result<PathBuf> {
        if room.contains('/') || room.contains("..") || !room.starts_with('!') {
            bail!("refusing room ID {room:?}");
        }
        Ok(self.rooms_dir.join(room).join("schedules"))
    }

    fn check_id(id: &str) -> Result<()> {
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric()) {
            bail!("no schedule {id:?}");
        }
        Ok(())
    }

    fn save(&self, job: &Job) -> Result<()> {
        let dir = self.dir(&job.room)?;
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(format!(".{}.json", job.id));
        std::fs::write(&tmp, serde_json::to_string_pretty(job)?)?;
        std::fs::rename(tmp, dir.join(format!("{}.json", job.id)))?;
        Ok(())
    }

    /// Move a job to `done/`, as it is now.
    fn retire(&self, job: &Job) -> Result<()> {
        let dir = self.dir(&job.room)?;
        let done = dir.join("done");
        std::fs::create_dir_all(&done)?;
        std::fs::write(
            done.join(format!("{}.json", job.id)),
            serde_json::to_string_pretty(job)?,
        )?;
        std::fs::remove_file(dir.join(format!("{}.json", job.id)))?;
        Ok(())
    }

    pub fn create(&self, job: &Job) -> Result<()> {
        let _g = self.lock.lock().unwrap();
        self.save(job)?;
        self.wake.notify_one();
        Ok(())
    }

    fn load(dir: &Path) -> Vec<Job> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut jobs: Vec<Job> = entries
            .flatten()
            .filter(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n.ends_with(".json") && !n.starts_with('.')
            })
            .filter_map(|e| {
                let text = std::fs::read_to_string(e.path()).ok()?;
                match serde_json::from_str(&text) {
                    Ok(j) => Some(j),
                    Err(err) => {
                        eprintln!("bad schedule {}: {err}", e.path().display());
                        None
                    }
                }
            })
            .collect();
        jobs.sort_by_key(|j: &Job| j.next);
        jobs
    }

    /// The room's live schedules, soonest first.
    pub fn list(&self, room: &str) -> Result<Vec<Job>> {
        Ok(Self::load(&self.dir(room)?))
    }

    /// End a schedule: it fires no more.
    pub fn cancel(&self, room: &str, id: &str) -> Result<Job> {
        Self::check_id(id)?;
        let _g = self.lock.lock().unwrap();
        let file = self.dir(room)?.join(format!("{id}.json"));
        let text = std::fs::read_to_string(&file).with_context(|| format!("no schedule {id}"))?;
        let mut job: Job = serde_json::from_str(&text)?;
        job.next = None;
        self.retire(&job)?;
        Ok(job)
    }

    /// The scratchboard a schedule's runs hand to each other.
    pub fn board(&self, room: &str, id: &str) -> Result<PathBuf> {
        Self::check_id(id)?;
        Ok(self.dir(room)?.join(format!("{id}.md")))
    }

    pub fn handoff(&self, room: &str, id: &str, text: &str) -> Result<()> {
        let board = self.board(room, id)?;
        if !board.with_extension("json").exists() {
            bail!("no live schedule {id} in this room");
        }
        std::fs::write(board, text)?;
        Ok(())
    }

    /// The soonest next fire of any schedule.
    pub fn soonest(&self) -> Option<DateTime<Utc>> {
        self.rooms()
            .iter()
            .flat_map(|d| Self::load(&d.join("schedules")))
            .filter_map(|j| j.next)
            .min()
    }

    fn rooms(&self) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(&self.rooms_dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with('!'))
            .map(|e| e.path())
            .collect()
    }

    /// The schedules due at `now`, each moved on to its next fire (or
    /// retired) before it is handed out, so a crash mid-run does not fire
    /// it twice. A due `only_if_new` schedule without new messages is moved
    /// on without a run.
    pub fn take_due(&self, now: DateTime<Utc>) -> Vec<Job> {
        let _g = self.lock.lock().unwrap();
        let mut due = Vec::new();
        for room in self.rooms() {
            for mut job in Self::load(&room.join("schedules")) {
                if !job.next.is_some_and(|t| t <= now) {
                    continue;
                }
                let run = !job.only_if_new
                    || new_messages(
                        &room.join("transcript.jsonl"),
                        job.since,
                        job.thread.as_deref(),
                    ) > 0;
                if run {
                    job.fired += 1;
                    job.since = now;
                }
                job.next = job.after(now);
                let saved = match job.next {
                    Some(_) => self.save(&job),
                    None => self.retire(&job),
                };
                if let Err(e) = saved {
                    eprintln!("could not move schedule {} on: {e:#}", job.id);
                    continue;
                }
                if run {
                    due.push(job);
                }
            }
        }
        due
    }
}

/// How many messages people (not parley, not schedules) wrote after
/// `since`, in the thread rooted at `thread` or, without one, anywhere in
/// the room.
pub fn new_messages(transcript: &Path, since: DateTime<Utc>, thread: Option<&str>) -> usize {
    let Ok(text) = std::fs::read_to_string(transcript) else {
        return 0;
    };
    let since = since.timestamp();
    text.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["ts"].as_i64().is_some_and(|ts| ts >= since))
        .filter(|v| v["sender"].as_str().is_some_and(|s| s.starts_with('@')))
        .filter(|v| thread.is_none_or(|t| v["thread"].as_str() == Some(t)))
        .count()
}

/// A short random ID.
pub fn new_id() -> Result<String> {
    use std::io::Read;
    let mut buf = [0u8; 4];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

/// The message a fire of `job` starts its run with: the prompt, where the
/// schedule stands, and the previous run's handoff.
pub fn fire_message(job: &Job, board: Option<&str>) -> String {
    let runs = match job.times {
        Some(t) => format!("run {} of {t}", job.fired),
        None => format!("run {}", job.fired),
    };
    let next = match &job.next {
        Some(t) => format!(
            "next {}",
            t.with_timezone(&Local).format("%Y-%m-%d %H:%M %Z")
        ),
        None => "this is the last run".into(),
    };
    let board = match board.map(str::trim).filter(|b| !b.is_empty()) {
        Some(b) => format!("Handoff from the previous run:\n\n{b}"),
        None => "No handoff from a previous run yet.".into(),
    };
    format!(
        "[schedule {id}: {runs}, {next}. Nobody sent this message; the schedule did. \
         Leave what the next run needs with `handoff(id: \"{id}\", text)`; end the \
         schedule with `cancel(id: \"{id}\")`.]\n\n{prompt}\n\n{board}",
        id = job.id,
        prompt = job.prompt,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn t(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    fn req(args: Value) -> Request {
        Request::from_args(&args).unwrap()
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("90s").unwrap(), 90);
        assert_eq!(parse_duration("20m").unwrap(), 1200);
        assert_eq!(parse_duration("1h30m").unwrap(), 5400);
        assert_eq!(parse_duration("2d").unwrap(), 172_800);
        assert_eq!(parse_duration("45").unwrap(), 45);
        assert_eq!(parse_duration("0s").unwrap(), 0);
        assert!(parse_duration("soon").is_err());
        assert!(parse_duration("5").is_ok());
        assert!(parse_duration("5x").is_err());
        assert!(parse_duration("1h5").is_err());
        assert_eq!(show_duration(5400), "1h30m");
    }

    #[test]
    fn delayed_fires_once() {
        let now = t("2026-11-04T20:00:00Z");
        let job = req(json!({"prompt": "p", "in": "20m"}))
            .job("a".into(), "!r", None, "agent", now)
            .unwrap();
        assert_eq!(job.when, When::At(t("2026-11-04T20:20:00Z")));
        assert_eq!(job.next, Some(t("2026-11-04T20:20:00Z")));
        let mut j = job.clone();
        j.fired = 1;
        assert_eq!(j.after(t("2026-11-04T20:20:00Z")), None);
    }

    #[test]
    fn interval_counts_and_skips_missed_fires() {
        let now = t("2026-11-04T20:00:00Z");
        let mut job = req(json!({"prompt": "p", "every": "1h", "times": 3}))
            .job("a".into(), "!r", None, "agent", now)
            .unwrap();
        assert_eq!(job.next, Some(t("2026-11-04T21:00:00Z")));
        job.fired = 1;
        assert_eq!(
            job.after(t("2026-11-04T21:00:00Z")),
            Some(t("2026-11-04T22:00:00Z"))
        );
        // Down from 21:00 to 23:30: the next fire is 00:00, not three at once.
        assert_eq!(
            job.after(t("2026-11-04T23:30:00Z")),
            Some(t("2026-11-05T00:00:00Z"))
        );
        job.fired = 3;
        assert_eq!(job.after(t("2026-11-04T23:30:00Z")), None);
    }

    #[test]
    fn cron_and_bad_requests() {
        let now = Utc::now();
        let job = req(json!({"prompt": "p", "cron": "0 0 8 * * *"}))
            .job("a".into(), "!r", None, "agent", now)
            .unwrap();
        assert!(job.next.unwrap() > now);
        assert_eq!(job.times, None);
        let bad = |a: Value| {
            Request::from_args(&a)
                .and_then(|r| r.job("a".into(), "!r", None, "x", now))
                .is_err()
        };
        assert!(bad(json!({"prompt": "p"})));
        assert!(bad(json!({"in": "5m"})));
        assert!(bad(json!({"prompt": "p", "cron": "nonsense"})));
        assert!(bad(json!({"prompt": "p", "every": "10s"})));
        assert!(bad(
            json!({"prompt": "p", "at": "2026-11-04 21:00", "in": "5m"})
        ));
        assert!(bad(json!({"prompt": "p", "in": "5m", "times": 0})));
    }

    #[test]
    fn store_fires_retires_cancels_and_gates() {
        let dir = tempfile::tempdir().unwrap();
        let room = "!room:x";
        std::fs::create_dir_all(dir.path().join(room)).unwrap();
        let store = Store::new(dir.path().to_owned());
        let now = t("2026-11-04T20:00:00Z");
        let once = req(json!({"prompt": "once", "in": "1m"}))
            .job("once".into(), room, None, "agent", now)
            .unwrap();
        let gated = req(json!({"prompt": "look", "every": "5m", "only_if_new": true}))
            .job("gated".into(), room, Some("$t"), "agent", now)
            .unwrap();
        let other = req(json!({"prompt": "x", "every": "5m"}))
            .job("other".into(), room, None, "agent", now)
            .unwrap();
        for j in [&once, &gated, &other] {
            store.create(j).unwrap();
        }
        assert_eq!(store.list(room).unwrap().len(), 3);
        assert_eq!(store.soonest(), Some(t("2026-11-04T20:01:00Z")));

        // Not due yet.
        assert!(store.take_due(t("2026-11-04T20:00:30Z")).is_empty());
        // 20:05: `once` (due since 20:01) runs and retires, `other` runs,
        // `gated` has nothing new in its thread and skips.
        let due = store.take_due(t("2026-11-04T20:05:00Z"));
        let mut ids: Vec<_> = due.iter().map(|j| j.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, ["once", "other"]);
        let due: Vec<_> = due.into_iter().filter(|j| j.id == "once").collect();
        assert_eq!(due[0].next, None);
        assert!(
            dir.path()
                .join(room)
                .join("schedules/done/once.json")
                .exists()
        );
        let left = store.list(room).unwrap();
        assert_eq!(left.len(), 2);
        let g = left.iter().find(|j| j.id == "gated").unwrap();
        assert_eq!((g.fired, g.next), (0, Some(t("2026-11-04T20:10:00Z"))));

        // Someone writes in the thread (and the agent, which does not count).
        let ts = t("2026-11-04T20:07:00Z").timestamp();
        std::fs::write(
            dir.path().join(room).join("transcript.jsonl"),
            format!(
                "{}\n{}\n",
                json!({"ts": ts, "sender": "@kas:x", "thread": "$t", "body": "hi"}),
                json!({"ts": ts, "sender": "agent", "thread": "$t", "body": "hey"}),
            ),
        )
        .unwrap();
        let due = store.take_due(t("2026-11-04T20:10:00Z"));
        let mut ids: Vec<_> = due.iter().map(|j| j.id.as_str()).collect();
        ids.sort();
        assert_eq!(ids, ["gated", "other"]);

        store.handoff(room, "gated", "step 2 next").unwrap();
        assert_eq!(
            std::fs::read_to_string(store.board(room, "gated").unwrap()).unwrap(),
            "step 2 next"
        );
        assert!(store.handoff(room, "once", "x").is_err());
        assert!(store.handoff(room, "../x", "x").is_err());

        store.cancel(room, "gated").unwrap();
        assert!(store.cancel(room, "gated").is_err());
        assert_eq!(store.list(room).unwrap().len(), 1);
        // Nothing new since 20:10: a gate would skip, but `other` is ungated.
        assert_eq!(store.take_due(t("2026-11-04T20:15:00Z")).len(), 1);
    }

    #[test]
    fn fire_message_shows_handoff() {
        let now = t("2026-11-04T20:00:00Z");
        let mut job = req(json!({"prompt": "check the build", "every": "1h", "times": 2}))
            .job("ab12".into(), "!r", None, "agent", now)
            .unwrap();
        job.fired = 1;
        let m = fire_message(&job, Some("half done"));
        assert!(m.contains("run 1 of 2"));
        assert!(m.contains("check the build"));
        assert!(m.contains("half done"));
        assert!(fire_message(&job, None).contains("No handoff"));
    }
}
