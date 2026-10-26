//! What a turn remembers of the turns before it: the room's context mode.
//!
//! - `session`: one harness session per main timeline or thread, resumed
//!   every turn (the harness compacts it when it gets long). `!new` clears it.
//! - `fresh`: no memory; every message stands alone.
//! - `window N`: a fresh run each turn, with the room's last N transcript
//!   messages put in front of the message.
//! - `digest N`: like `window N`, plus the room's `MEMORY.md`, which a small
//!   extra run rewrites after every reply: what to keep, what is settled,
//!   what is open.
//! - `layered N`: a fresh run each turn, with summaries of earlier days and
//!   weeks, the room's last N messages outside the thread, the thread so
//!   far, and where to look up the rest (see `layered.rs`).
//!
//! A room's mode is, in order: the `context` file in its folder (written by
//! `!context`), `context` in room.toml, or `session` if the profile says
//! `resume` and else `fresh`.

use std::{fmt, path::Path, str::FromStr};

use anyhow::{Result, bail};
use serde::Deserialize;

/// Messages a window holds when the mode does not say.
pub const DEFAULT_WINDOW: usize = 20;
/// Messages a digest room gets besides its memory, when the mode does not say.
pub const DEFAULT_DIGEST_WINDOW: usize = 6;
/// Longest a single transcript message is quoted, in characters.
pub const MAX_MESSAGE_CHARS: usize = 2000;
/// Longest the quoted history gets in total, in characters.
const MAX_HISTORY_CHARS: usize = 40_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum Mode {
    Session,
    Fresh,
    Window(usize),
    Digest(usize),
    Layered(usize),
}

impl Mode {
    /// Whether the harness keeps and resumes a session.
    pub fn resumes(self) -> bool {
        self == Mode::Session
    }

    pub const HELP: &str = "\
- `session`: one ongoing session per timeline or thread (`!new` clears it)
- `fresh`: no memory, every message stands alone
- `window N`: the room's last N messages (default 20) come with each message
- `digest N`: the room's memory notes, kept up to date after every reply, plus the last N messages (default 6)
- `layered N`: summaries of earlier days and weeks, the last N messages outside the thread (default 10), the whole thread, and a pointer to the full transcript
- `default`: back to what room.toml says";
}

impl FromStr for Mode {
    type Err = anyhow::Error;

    /// `session`, `fresh`, `window`, `window 20`, `window:20`, `digest 6`.
    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        let (name, n) = match s.split_once([' ', ':', '=']) {
            Some((name, n)) => {
                let n: usize = n
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("{:?} is not a number", n.trim()))?;
                if n == 0 {
                    bail!("a window of 0 messages is `fresh`");
                }
                (name, Some(n))
            }
            None => (s.as_str(), None),
        };
        Ok(match (name, n) {
            ("session", None) => Mode::Session,
            ("fresh", None) => Mode::Fresh,
            ("window", n) => Mode::Window(n.unwrap_or(DEFAULT_WINDOW)),
            ("digest", n) => Mode::Digest(n.unwrap_or(DEFAULT_DIGEST_WINDOW)),
            ("layered", n) => Mode::Layered(n.unwrap_or(crate::layered::DEFAULT_RECENT)),
            ("session" | "fresh", Some(_)) => bail!("`{name}` takes no number"),
            _ => bail!("unknown context mode {s:?}"),
        })
    }
}

impl TryFrom<String> for Mode {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mode::Session => write!(f, "session"),
            Mode::Fresh => write!(f, "fresh"),
            Mode::Window(n) => write!(f, "window {n}"),
            Mode::Digest(n) => write!(f, "digest {n}"),
            Mode::Layered(n) => write!(f, "layered {n}"),
        }
    }
}

#[derive(Deserialize)]
struct Line {
    sender: String,
    body: String,
}

/// The last `n` messages of a transcript, oldest first, as `[sender] body`
/// lines, leaving out the message being answered (the transcript's last
/// line, when it is `current`). Empty if there are none.
pub fn window(transcript: &str, n: usize, current: &str) -> String {
    let mut lines: Vec<Line> = transcript
        .lines()
        .filter_map(|l| serde_json::from_str::<Line>(l).ok())
        // Commands like `!context` are not part of the conversation.
        .filter(|l| !l.body.trim_start().starts_with('!'))
        .collect();
    if lines
        .last()
        .is_some_and(|l| l.sender != "agent" && l.body == current)
    {
        lines.pop();
    }
    let start = lines.len().saturating_sub(n);
    let mut out: Vec<String> = lines[start..]
        .iter()
        .map(|l| quote(&l.sender, &l.body, MAX_MESSAGE_CHARS))
        .collect();
    // Keep the newest messages when the whole is too long.
    while out.len() > 1 && out.iter().map(String::len).sum::<usize>() > MAX_HISTORY_CHARS {
        out.remove(0);
    }
    out.join("\n\n")
}

/// One transcript message as `[sender] body`, the body cut at `max`
/// characters.
pub fn quote(sender: &str, body: &str, max: usize) -> String {
    // "@kas:example.org" reads better as "kas".
    let who = sender
        .strip_prefix('@')
        .and_then(|s| s.split(':').next())
        .unwrap_or(sender);
    if body.chars().count() > max {
        let cut: String = body.chars().take(max).collect();
        format!("[{who}] {cut} […]")
    } else {
        format!("[{who}] {body}")
    }
}

/// The input for a fresh run: room memory and recent messages, if any,
/// then the message itself.
pub fn with_history(memory: Option<&str>, history: &str, input: &str) -> String {
    let mut out = String::new();
    if let Some(memory) = memory.map(str::trim).filter(|m| !m.is_empty()) {
        out.push_str(&format!(
            "<room-memory>\nNotes kept from earlier conversations in this room:\n\n{memory}\n</room-memory>\n\n"
        ));
    }
    if !history.is_empty() {
        out.push_str(&format!(
            "<recent-messages>\nThe room's latest messages before this one, oldest first, for context:\n\n{history}\n</recent-messages>\n\n"
        ));
    }
    out.push_str(input);
    out
}

pub const DIGEST_SYSTEM_PROMPT: &str = "\
You keep the memory notes of a Matrix chat room between a person and an AI \
agent. You get the current notes and the latest exchange, and answer with the \
complete new notes, nothing else.

Keep what will matter in later conversations: facts about the people, \
decisions, preferences, things that are settled, open questions and promises. \
Drop chit-chat and anything the exchange made obsolete. Short Markdown bullets \
under a few headings, at most about 400 words; merge and prune rather than \
append. If nothing worth keeping happened, return the notes unchanged.";

/// The input for a digest run.
pub fn digest_input(memory: Option<&str>, sender: &str, body: &str, answer: &str) -> String {
    let memory = memory
        .map(str::trim)
        .filter(|m| !m.is_empty())
        .unwrap_or("(none yet)");
    format!(
        "<current-notes>\n{memory}\n</current-notes>\n\n<latest-exchange>\n[{sender}] {body}\n\n[agent] {answer}\n</latest-exchange>"
    )
}

/// The room's override, if `!context` set one.
pub async fn read_override(dir: &Path) -> Result<Option<Mode>> {
    match tokio::fs::read_to_string(dir.join("context")).await {
        Ok(text) => Ok(Some(text.parse()?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_parse_and_print() {
        for (text, mode, shown) in [
            ("session", Mode::Session, "session"),
            (" Fresh\n", Mode::Fresh, "fresh"),
            ("window", Mode::Window(DEFAULT_WINDOW), "window 20"),
            ("window 5", Mode::Window(5), "window 5"),
            ("window:5", Mode::Window(5), "window 5"),
            ("digest", Mode::Digest(DEFAULT_DIGEST_WINDOW), "digest 6"),
            ("digest 10", Mode::Digest(10), "digest 10"),
            ("layered", Mode::Layered(10), "layered 10"),
            ("layered 4", Mode::Layered(4), "layered 4"),
        ] {
            let parsed: Mode = text.parse().unwrap();
            assert_eq!(parsed, mode, "{text:?}");
            assert_eq!(parsed.to_string(), shown);
            assert_eq!(shown.parse::<Mode>().unwrap(), mode);
        }
        for bad in ["", "sessions", "window x", "window 0", "fresh 3"] {
            assert!(bad.parse::<Mode>().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn only_session_resumes() {
        assert!(Mode::Session.resumes());
        assert!(!Mode::Fresh.resumes());
        assert!(!Mode::Window(3).resumes());
        assert!(!Mode::Digest(3).resumes());
        assert!(!Mode::Layered(3).resumes());
    }

    fn transcript(lines: &[(&str, &str)]) -> String {
        lines
            .iter()
            .map(|(s, b)| {
                serde_json::json!({"ts": 0, "sender": s, "thread": null, "body": b}).to_string()
                    + "\n"
            })
            .collect()
    }

    #[test]
    fn window_takes_the_last_messages_without_the_current_one() {
        let t = transcript(&[
            ("@kas:x", "one"),
            ("agent", "two"),
            ("@kas:x", "three"),
            ("agent", "four"),
            ("@kas:x", "now"),
        ]);
        assert_eq!(window(&t, 2, "now"), "[kas] three\n\n[agent] four");
        assert_eq!(window(&t, 0, "now"), "");
        // The last line is only skipped when it is the message being answered.
        assert_eq!(window(&t, 1, "other"), "[kas] now");
        let t = transcript(&[
            ("@kas:x", "hi"),
            ("@kas:x", "!context window 5"),
            ("@kas:x", "q"),
        ]);
        assert_eq!(window(&t, 5, "q"), "[kas] hi");
    }

    #[test]
    fn window_skips_broken_lines_and_cuts_long_messages() {
        let long = "x".repeat(MAX_MESSAGE_CHARS + 10);
        let t = format!(
            "not json\n{}",
            transcript(&[("@kas:x", &long), ("agent", "ok")])
        );
        let w = window(&t, 10, "q");
        assert!(w.starts_with("[kas] xxx"), "{w}");
        assert!(w.contains(" […]\n\n[agent] ok"), "{w}");
    }

    #[test]
    fn window_drops_the_oldest_when_too_long() {
        let big = "y".repeat(MAX_MESSAGE_CHARS);
        let many: Vec<(&str, &str)> = (0..40).map(|_| ("agent", big.as_str())).collect();
        let mut t = transcript(&many);
        t.push_str(&transcript(&[("agent", "last")]));
        let w = window(&t, 100, "q");
        assert!(w.len() <= MAX_HISTORY_CHARS + 100, "{}", w.len());
        assert!(w.ends_with("[agent] last"));
    }

    #[test]
    fn with_history_leaves_out_empty_parts() {
        assert_eq!(with_history(None, "", "[Kas] hi"), "[Kas] hi");
        assert_eq!(with_history(Some("  \n"), "", "[Kas] hi"), "[Kas] hi");
        let full = with_history(Some("- likes tea"), "[kas] before", "[Kas] hi");
        assert!(full.starts_with("<room-memory>"), "{full}");
        assert!(full.contains("- likes tea"));
        assert!(full.contains("<recent-messages>"));
        assert!(full.contains("[kas] before"));
        assert!(full.ends_with("</recent-messages>\n\n[Kas] hi"));
    }

    #[test]
    fn digest_input_has_notes_and_exchange() {
        let i = digest_input(None, "Kas", "q", "a");
        assert!(i.contains("(none yet)"));
        assert!(i.contains("[Kas] q\n\n[agent] a"));
    }
}
