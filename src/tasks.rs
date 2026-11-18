//! Tasks for people: what an agent needs a person to do, kept where it
//! does not scroll away. The `task` tool writes one note per task into
//! Kas's Obsidian vault (frontmatter `status: open|done`) and posts it in
//! the turn's conversation, with the people's names (so it notifies) and
//! an `obsidian://` link to the note. A ✅ reaction on that post, or
//! `status: done` in the note, closes it; `!taken` lists what is open.
//!
//! Configured by `tasks.toml` in the data dir:
//! `dir` (where the notes go), `root` (the vault's folder) and `vault`
//! (its name in Obsidian, for links). Without it there is no `task` tool.
//! With `daily` and `ai_daily` set, a new task is also listed in the AI
//! daily note of its day, which Kas's daily note links to (`ai-notes` in
//! its frontmatter; made if missing). The vault's own sync (life-sync on
//! the lab) commits the notes.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Store {
    /// Folder of the task notes.
    pub dir: PathBuf,
    /// The vault's folder; links are relative to it.
    pub root: PathBuf,
    /// The vault's name in Obsidian.
    pub vault: String,
    /// Folder of Kas's daily notes (`<date>.md`), relative to `root`. With
    /// `ai_daily`, a new task is listed in the AI daily note of its day,
    /// and the daily note points at that one in its frontmatter
    /// (`ai-notes`), as the vault's CLAUDE.md wants; a missing daily note
    /// is made, with only that frontmatter.
    #[serde(default)]
    pub daily: Option<PathBuf>,
    /// Folder of the AI daily notes (`<date>-ai.md`), relative to `root`.
    #[serde(default)]
    pub ai_daily: Option<PathBuf>,
}

/// A task as written in its note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Task {
    pub title: String,
    pub body: String,
    /// Who asks: the agent's name.
    pub from: String,
    /// The room's name.
    pub room: String,
    /// YYYY-MM-DD, local.
    pub created: String,
    pub due: Option<String>,
    /// The Matrix post's event ID.
    pub event: Option<String>,
    pub room_id: String,
    pub done: bool,
}

/// What `!taken` and the tool show of an open task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Open {
    pub title: String,
    pub from: String,
    pub room: String,
    pub created: String,
    pub due: Option<String>,
    pub link: String,
}

impl Store {
    /// `tasks.toml` in `data_dir`, if there is one.
    pub fn load(data_dir: &Path) -> Result<Option<Self>> {
        let path = data_dir.join("tasks.toml");
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(Some(
                toml::from_str(&text).with_context(|| format!("reading {}", path.display()))?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// A free path for a new note on `created` named after `title`.
    pub fn path_for(&self, created: &str, title: &str) -> Result<PathBuf> {
        let name = file_name(title);
        if name.is_empty() {
            bail!("the title is empty");
        }
        let base = format!("{created} {name}");
        let mut path = self.dir.join(format!("{base}.md"));
        let mut n = 2;
        while path.exists() {
            path = self.dir.join(format!("{base} ({n}).md"));
            n += 1;
        }
        Ok(path)
    }

    /// The `obsidian://` link that opens the note at `path`.
    pub fn link(&self, path: &Path) -> String {
        let rel = path.strip_prefix(&self.root).unwrap_or(path);
        let rel = rel.to_string_lossy();
        let rel = rel.strip_suffix(".md").unwrap_or(&rel);
        format!(
            "obsidian://open?vault={}&file={}",
            encode(&self.vault),
            encode(rel)
        )
    }

    pub fn write(&self, path: &Path, task: &Task) -> Result<()> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        std::fs::write(path, task.note()).with_context(|| format!("writing {}", path.display()))
    }

    /// List the task at `path` in the AI daily note of its day, and link
    /// that from Kas's daily note (frontmatter only). Off without
    /// `daily` and `ai_daily`.
    pub fn link_daily(&self, path: &Path, task: &Task) -> Result<()> {
        let (Some(daily), Some(ai_daily)) = (&self.daily, &self.ai_daily) else {
            return Ok(());
        };
        let date = &task.created;
        let ai_name = format!("{date}-ai");
        let ai_path = self.root.join(ai_daily).join(format!("{ai_name}.md"));
        let mut text = match std::fs::read_to_string(&ai_path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => format!(
                "---\ncreated: {date}\nupdated: {date}\ntype: ai-journal\nauthor: claude\ntags: [taak]\ndaily: \"[[{date}]]\"\n---\n"
            ),
            Err(e) => return Err(e).with_context(|| format!("reading {}", ai_path.display())),
        };
        let rel = path.strip_prefix(&self.root).unwrap_or(path);
        let rel = rel.to_string_lossy();
        let rel = rel.strip_suffix(".md").unwrap_or(&rel);
        let due = task
            .due
            .as_deref()
            .map(|d| format!(", uiterlijk {d}"))
            .unwrap_or_default();
        if !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!(
            "\n## 📌 Taak: {}\n\n[[{rel}|{}]], van {} ({}){due}.\n",
            task.title, task.title, task.from, task.room
        ));
        write_file(&ai_path, &text)?;

        let day_path = self.root.join(daily).join(format!("{date}.md"));
        let day = match std::fs::read_to_string(&day_path) {
            Ok(t) => Some(t),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).with_context(|| format!("reading {}", day_path.display())),
        };
        let line = format!("ai-notes: \"[[{ai_name}]]\"");
        let new = match day {
            None => Some(format!("---\n{line}\n---\n")),
            Some(t) => with_frontmatter_line(&t, "ai-notes", &line),
        };
        if let Some(new) = new {
            write_file(&day_path, &new)?;
        }
        Ok(())
    }

    /// Every task note, with its path, oldest first.
    fn all(&self) -> Vec<(PathBuf, Task)> {
        let Ok(dir) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut tasks: Vec<(PathBuf, Task)> = dir
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "md"))
            .filter_map(|p| {
                let text = std::fs::read_to_string(&p).ok()?;
                Some((p, Task::parse(&text)?))
            })
            .collect();
        tasks.sort_by(|a, b| a.0.cmp(&b.0));
        tasks
    }

    /// The open tasks, oldest first.
    pub fn open(&self) -> Vec<Open> {
        self.all()
            .into_iter()
            .filter(|(_, t)| !t.done)
            .map(|(p, t)| Open {
                title: t.title,
                from: t.from,
                room: t.room,
                created: t.created,
                due: t.due,
                link: self.link(&p),
            })
            .collect()
    }

    /// Close the task posted as `event` in `room_id`; its title, if it was
    /// open.
    pub fn close(&self, room_id: &str, event: &str, today: &str) -> Result<Option<String>> {
        for (path, task) in self.all() {
            if task.event.as_deref() == Some(event) && task.room_id == room_id {
                if task.done {
                    return Ok(None);
                }
                let text = std::fs::read_to_string(&path)?;
                std::fs::write(&path, mark_done(&text, today))?;
                return Ok(Some(task.title));
            }
        }
        Ok(None)
    }
}

impl Open {
    pub fn line(&self) -> String {
        let due = self
            .due
            .as_deref()
            .map(|d| format!(", uiterlijk {d}"))
            .unwrap_or_default();
        format!(
            "- **{}** ({}, {}, {}{due})\n  {}",
            self.title, self.from, self.room, self.created, self.link
        )
    }
}

/// `open` as the answer to `!taken`.
pub fn list(open: &[Open]) -> String {
    if open.is_empty() {
        return "_Geen open taken._".to_owned();
    }
    let lines: Vec<String> = open.iter().map(Open::line).collect();
    format!(
        "**Open taken ({}):**\n{}\n\n_Afvinken: ✅ op de taak, of `status: done` in de notitie._",
        open.len(),
        lines.join("\n")
    )
}

impl Task {
    /// The note: frontmatter, then the body.
    pub fn note(&self) -> String {
        let q = |s: &str| serde_json::to_string(s).unwrap();
        let mut fm = vec![
            format!("status: {}", if self.done { "done" } else { "open" }),
            format!("title: {}", q(&self.title)),
            format!("from: {}", q(&self.from)),
            format!("room: {}", q(&self.room)),
            format!("created: {}", self.created),
        ];
        if let Some(d) = &self.due {
            fm.push(format!("due: {d}"));
        }
        fm.push(format!("room_id: {}", q(&self.room_id)));
        if let Some(e) = &self.event {
            fm.push(format!("event: {}", q(e)));
            fm.push(format!(
                "matrix: {}",
                q(&format!("https://matrix.to/#/{}/{e}", self.room_id))
            ));
        }
        fm.push("tags: [taak]".to_owned());
        format!("---\n{}\n---\n\n{}\n", fm.join("\n"), self.body.trim())
    }

    /// A note written by [`Task::note`] (and maybe edited in Obsidian).
    pub fn parse(text: &str) -> Option<Self> {
        let rest = text.strip_prefix("---\n")?;
        let (fm, body) = rest.split_once("\n---")?;
        let get = |k: &str| -> Option<String> {
            fm.lines().find_map(|l| {
                let v = l.strip_prefix(k)?.strip_prefix(':')?.trim();
                Some(serde_json::from_str::<String>(v).unwrap_or_else(|_| v.to_owned()))
            })
        };
        Some(Self {
            title: get("title")?,
            body: body.trim().to_owned(),
            from: get("from").unwrap_or_default(),
            room: get("room").unwrap_or_default(),
            created: get("created").unwrap_or_default(),
            due: get("due").filter(|d| !d.is_empty()),
            event: get("event"),
            room_id: get("room_id").unwrap_or_default(),
            done: get("status").is_some_and(|s| s.eq_ignore_ascii_case("done")),
        })
    }
}

/// The note `text` with `status: done` and `done: <today>`.
fn mark_done(text: &str, today: &str) -> String {
    let mut out = Vec::new();
    for l in text.lines() {
        if l.starts_with("status:") {
            out.push("status: done".to_owned());
            out.push(format!("done: {today}"));
        } else {
            out.push(l.to_owned());
        }
    }
    out.join("\n") + "\n"
}

fn write_file(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// `text` with `line` in its frontmatter (made if missing); none if it
/// already has `key`. The body stays as it is.
fn with_frontmatter_line(text: &str, key: &str, line: &str) -> Option<String> {
    let Some(rest) = text.strip_prefix("---\n") else {
        return Some(format!("---\n{line}\n---\n{text}"));
    };
    let Some(end) = rest.find("\n---") else {
        return Some(format!("---\n{line}\n---\n{text}"));
    };
    let fm = &rest[..end];
    if fm
        .lines()
        .any(|l| l.strip_prefix(key).is_some_and(|r| r.starts_with(':')))
    {
        return None;
    }
    let fm = if fm.is_empty() {
        line.to_owned()
    } else {
        format!("{fm}\n{line}")
    };
    Some(format!("---\n{fm}{}", &rest[end..]))
}

/// `title` as a file name: no characters Obsidian or file systems refuse,
/// at most 80 characters.
fn file_name(title: &str) -> String {
    let clean: String = title
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '#' | '^' | '[' | ']' => ' ',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();
    let words: Vec<&str> = clean.split_whitespace().collect();
    let mut name = String::new();
    for w in words {
        if name.chars().count() + w.chars().count() + 1 > 80 {
            break;
        }
        if !name.is_empty() {
            name.push(' ');
        }
        name.push_str(w);
    }
    name.trim_start_matches('.').to_owned()
}

/// Percent-encoding for a URL's query value.
fn encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// The post that hands the task to `people` (Matrix ID, name).
pub fn post_body(task: &Task, people: &[(String, String)], link: &str) -> String {
    let who = if people.is_empty() {
        "jou".to_owned()
    } else {
        people
            .iter()
            .map(|(id, name)| format!("[{name}](https://matrix.to/#/{id})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let due = task
        .due
        .as_deref()
        .map(|d| format!("**Uiterlijk:** {d}\n\n"))
        .unwrap_or_default();
    format!(
        "📌 **Taak voor {who}: {}**\n\n{}\n\n{due}_✅ hierop als het klaar is · `!taken` toont wat openstaat_\n\n{link}",
        task.title,
        task.body.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dir: &Path) -> Store {
        Store {
            dir: dir.join("vault/tasks"),
            root: dir.join("vault"),
            vault: "General".into(),
            daily: Some("daily".into()),
            ai_daily: Some("ai/daily".into()),
        }
    }

    #[test]
    fn links_from_the_dailies() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(tmp.path());
        let root = tmp.path().join("vault");
        let p = s.path_for("2026-11-18", "one").unwrap();
        // No notes yet: both are made.
        s.link_daily(&p, &task()).unwrap();
        let ai = std::fs::read_to_string(root.join("ai/daily/2026-11-18-ai.md")).unwrap();
        assert!(ai.starts_with("---\ncreated: 2026-11-18\n"));
        assert!(ai.contains("daily: \"[[2026-11-18]]\"\n---\n"));
        assert!(ai.contains(
            "## 📌 Taak: Delegate in parley.nix: check\n\n[[tasks/2026-11-18 one|Delegate in parley.nix: check]], van Reinout (parley), uiterlijk 2026-11-20.\n"
        ));
        let day = root.join("daily/2026-11-18.md");
        assert_eq!(
            std::fs::read_to_string(&day).unwrap(),
            "---\nai-notes: \"[[2026-11-18-ai]]\"\n---\n"
        );
        // A second task appends; the daily stays as it is.
        s.link_daily(&p, &task()).unwrap();
        let ai2 = std::fs::read_to_string(root.join("ai/daily/2026-11-18-ai.md")).unwrap();
        assert_eq!(ai2.matches("## 📌 Taak").count(), 2);
        assert!(ai2.starts_with(&ai));
        assert_eq!(
            std::fs::read_to_string(&day).unwrap(),
            "---\nai-notes: \"[[2026-11-18-ai]]\"\n---\n"
        );
        // Kas's text without frontmatter, or with other fields.
        std::fs::write(&day, "Mijn dag.\n").unwrap();
        s.link_daily(&p, &task()).unwrap();
        assert_eq!(
            std::fs::read_to_string(&day).unwrap(),
            "---\nai-notes: \"[[2026-11-18-ai]]\"\n---\nMijn dag.\n"
        );
        std::fs::write(&day, "---\nmood: ok\n---\nMijn dag.\n").unwrap();
        s.link_daily(&p, &task()).unwrap();
        assert_eq!(
            std::fs::read_to_string(&day).unwrap(),
            "---\nmood: ok\nai-notes: \"[[2026-11-18-ai]]\"\n---\nMijn dag.\n"
        );
        // Off without the folders.
        let off = Store {
            daily: None,
            ..s.clone()
        };
        let q = off.path_for("2026-11-19", "x").unwrap();
        off.link_daily(
            &q,
            &Task {
                created: "2026-11-19".into(),
                ..task()
            },
        )
        .unwrap();
        assert!(!root.join("ai/daily/2026-11-19-ai.md").exists());
    }

    fn task() -> Task {
        Task {
            title: "Delegate in parley.nix: check".into(),
            body: "Kijk of `journalctl` het zegt.".into(),
            from: "Reinout".into(),
            room: "parley".into(),
            created: "2026-11-18".into(),
            due: Some("2026-11-20".into()),
            event: Some("$ev".into()),
            room_id: "!r:x".into(),
            done: false,
        }
    }

    #[test]
    fn note_round_trips() {
        let t = task();
        assert_eq!(Task::parse(&t.note()).unwrap(), t);
        let t = Task {
            due: None,
            event: None,
            ..task()
        };
        assert_eq!(Task::parse(&t.note()).unwrap(), t);
    }

    #[test]
    fn names_and_links() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(tmp.path());
        let p = s.path_for("2026-11-18", "a/b: c?").unwrap();
        assert!(p.ends_with("tasks/2026-11-18 a b c.md"));
        assert_eq!(
            s.link(&p),
            "obsidian://open?vault=General&file=tasks%2F2026-11-18%20a%20b%20c"
        );
        assert!(s.path_for("2026-11-18", " ?? ").is_err());
        s.write(&p, &task()).unwrap();
        assert!(
            s.path_for("2026-11-18", "a b c")
                .unwrap()
                .ends_with("2026-11-18 a b c (2).md")
        );
    }

    #[test]
    fn open_and_close() {
        let tmp = tempfile::tempdir().unwrap();
        let s = store(tmp.path());
        assert!(s.open().is_empty());
        let p = s.path_for("2026-11-18", "one").unwrap();
        s.write(&p, &task()).unwrap();
        let open = s.open();
        assert_eq!(open.len(), 1);
        assert!(list(&open).contains("uiterlijk 2026-11-20"));
        assert_eq!(s.close("!other:x", "$ev", "2026-11-19").unwrap(), None);
        assert_eq!(
            s.close("!r:x", "$ev", "2026-11-19").unwrap().as_deref(),
            Some("Delegate in parley.nix: check")
        );
        assert!(s.open().is_empty());
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("status: done\ndone: 2026-11-19\n"));
        assert_eq!(s.close("!r:x", "$ev", "2026-11-19").unwrap(), None);
        assert_eq!(list(&[]), "_Geen open taken._");
    }

    #[test]
    fn post_names_people_and_links() {
        let b = post_body(
            &task(),
            &[("@kas:x".into(), "Kas".into())],
            "obsidian://open?x",
        );
        assert!(b.starts_with("📌 **Taak voor [Kas](https://matrix.to/#/@kas:x): Delegate"));
        assert!(b.contains("**Uiterlijk:** 2026-11-20"));
        assert!(b.trim_end().ends_with("obsidian://open?x"));
    }
}
