//! Questions to people as Matrix polls. Claude Code's own
//! `AskUserQuestion` tool (questions with labelled options, one or many
//! picks, and an "Other" answer) cannot ask anyone in a headless run, so
//! parley takes its place: Claude Code hands the call to parley's
//! `approve` tool (`--permission-prompt-tool`), which posts each question
//! as a poll in the turn's conversation, mentioning the people in the
//! room, and waits for a vote. The answers go back to the model as if
//! picked in the terminal.
//!
//! "Anders, namelijk…" stands for "Other": a poll has no text field, so
//! the answer is typed in the conversation. Any text message there while
//! a question waits answers it, voted or not.
//!
//! The run waits at most [`WAIT`]. Then the model is told to end its turn,
//! and the poll stays open: a later vote starts a new turn with
//! `🗳️ <question> → <answer>` as the message. Polls are kept per room in
//! `polls/<event id>.json` until answered.

use std::{
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::oneshot;

/// How long a run waits for an answer before it is told to end its turn.
pub const WAIT: Duration = Duration::from_secs(10 * 60);

/// The poll answer that means "Other: I'll type it".
pub const OTHER: &str = "Anders, namelijk… (typ het hier)";
const OTHER_ID: &str = "other";

/// One question, as `AskUserQuestion` puts it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    pub question: String,
    #[serde(default)]
    pub header: String,
    /// (label, description)
    pub options: Vec<(String, String)>,
    #[serde(default)]
    pub multi: bool,
}

/// The questions of an `AskUserQuestion` call's input.
pub fn questions(input: &Value) -> Result<Vec<Question>> {
    let list = input
        .get("questions")
        .and_then(Value::as_array)
        .context("no `questions`")?;
    let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
    let mut out = Vec::new();
    for q in list {
        let question = s(q, "question");
        if question.trim().is_empty() {
            bail!("a question is empty");
        }
        let options: Vec<(String, String)> = q
            .get("options")
            .and_then(Value::as_array)
            .map(|o| {
                o.iter()
                    .map(|o| (s(o, "label"), s(o, "description")))
                    .collect()
            })
            .unwrap_or_default();
        if options.len() > 19 {
            bail!("a poll takes at most 19 options plus \"Other\"");
        }
        out.push(Question {
            question,
            header: s(q, "header"),
            options,
            multi: q
                .get("multiSelect")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
    }
    if out.is_empty() {
        bail!("no questions");
    }
    Ok(out)
}

/// The `AskUserQuestion` input with the answers filled in, by question.
pub fn answered(input: &Value, answers: &[(String, String)]) -> Value {
    let mut input = input.clone();
    let map: serde_json::Map<String, Value> = answers
        .iter()
        .map(|(q, a)| (q.clone(), Value::String(a.clone())))
        .collect();
    if let Some(o) = input.as_object_mut() {
        o.insert("answers".into(), Value::Object(map));
    }
    input
}

impl Question {
    /// The `org.matrix.msc3381.poll.start` content: the options plus
    /// "Other", mentioning `people`, in the thread `root` (after `latest`).
    pub fn poll(&self, people: &[String], thread: Option<(&str, &str)>) -> Value {
        let title = if self.header.trim().is_empty() {
            format!("❓ {}", self.question)
        } else {
            format!("❓ {}: {}", self.header.trim(), self.question)
        };
        let mut answers: Vec<Value> = self
            .options
            .iter()
            .enumerate()
            .map(|(i, (label, desc))| {
                let text = if desc.trim().is_empty() {
                    label.clone()
                } else {
                    format!("{label} — {desc}")
                };
                json!({"id": format!("a{i}"), "org.matrix.msc1767.text": text})
            })
            .collect();
        answers.push(json!({"id": OTHER_ID, "org.matrix.msc1767.text": OTHER}));
        let fallback = std::iter::once(title.clone())
            .chain(answers.iter().enumerate().map(|(i, a)| {
                format!(
                    "{}. {}",
                    i + 1,
                    a["org.matrix.msc1767.text"].as_str().unwrap_or("")
                )
            }))
            .collect::<Vec<_>>()
            .join("\n");
        let max = if self.multi { answers.len() } else { 1 };
        let mut content = json!({
            "org.matrix.msc3381.poll.start": {
                "question": {"org.matrix.msc1767.text": title},
                "kind": "org.matrix.msc3381.poll.disclosed",
                "max_selections": max,
                "answers": answers,
            },
            "org.matrix.msc1767.text": fallback,
            // Highlights the poll for them, like a mention in a message.
            "m.mentions": {"user_ids": people},
        });
        if let Some((root, latest)) = thread {
            content["m.relates_to"] = json!({
                "rel_type": "m.thread",
                "event_id": root,
                "is_falling_back": true,
                "m.in_reply_to": {"event_id": latest},
            });
        }
        content
    }

    /// The answer the votes `ids` stand for; None for "Other" alone (the
    /// answer is still to be typed).
    pub fn answer(&self, ids: &[String]) -> Option<String> {
        let labels: Vec<&str> = ids
            .iter()
            .filter_map(|id| id.strip_prefix('a')?.parse::<usize>().ok())
            .filter_map(|i| self.options.get(i).map(|(l, _)| l.as_str()))
            .collect();
        (!labels.is_empty()).then(|| labels.join(", "))
    }
}

/// What a vote did.
#[derive(Debug, PartialEq, Eq)]
pub enum Vote {
    /// A waiting run got the answer.
    Answered,
    /// "Other": the answer will be typed; nothing to do yet.
    Other,
    /// Nobody waits any more: start a turn in `thread` with `message`.
    Late {
        thread: Option<String>,
        message: String,
    },
    /// Not one of ours, or answered already.
    Unknown,
}

/// A poll on file (`polls/<event id>.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Asked {
    thread: Option<String>,
    question: Question,
}

struct Waiting {
    room: String,
    thread: Option<String>,
    poll: String,
    question: Question,
    tx: oneshot::Sender<String>,
}

/// The polls a run waits on, and those on file.
pub struct Asks {
    rooms_dir: PathBuf,
    waiting: Mutex<Vec<Waiting>>,
}

impl Asks {
    pub fn new(rooms_dir: PathBuf) -> Self {
        Self {
            rooms_dir,
            waiting: Mutex::default(),
        }
    }

    fn file(&self, room: &str, poll: &str) -> Result<PathBuf> {
        if [room, poll]
            .iter()
            .any(|s| s.contains('/') || s.contains(".."))
        {
            bail!("refusing {room:?} / {poll:?}");
        }
        Ok(self
            .rooms_dir
            .join(room)
            .join("polls")
            .join(format!("{poll}.json")))
    }

    /// Note the poll `poll` posted in `room`'s conversation `thread`, and
    /// wait for its answer on the receiver.
    pub fn wait(
        &self,
        room: &str,
        thread: Option<&str>,
        poll: &str,
        question: &Question,
    ) -> Result<oneshot::Receiver<String>> {
        let file = self.file(room, poll)?;
        write(
            &file,
            &Asked {
                thread: thread.map(str::to_owned),
                question: question.clone(),
            },
        )?;
        let (tx, rx) = oneshot::channel();
        self.waiting.lock().unwrap().push(Waiting {
            room: room.to_owned(),
            thread: thread.map(str::to_owned),
            poll: poll.to_owned(),
            question: question.clone(),
            tx,
        });
        Ok(rx)
    }

    /// A vote `ids` on `poll` in `room`.
    pub fn vote(&self, room: &str, poll: &str, ids: &[String]) -> Vote {
        let mut waiting = self.waiting.lock().unwrap();
        let live = waiting
            .iter()
            .position(|w| w.room == room && w.poll == poll && !w.tx.is_closed());
        if let Some(i) = live {
            let Some(answer) = waiting[i].question.answer(ids) else {
                return Vote::Other;
            };
            let w = waiting.remove(i);
            self.done(room, poll);
            return match w.tx.send(answer) {
                Ok(()) => Vote::Answered,
                Err(answer) => late(w.thread, &w.question, &answer),
            };
        }
        waiting.retain(|w| !(w.room == room && w.poll == poll));
        drop(waiting);
        let Ok(file) = self.file(room, poll) else {
            return Vote::Unknown;
        };
        let Some(asked) = std::fs::read_to_string(&file)
            .ok()
            .and_then(|t| serde_json::from_str::<Asked>(&t).ok())
        else {
            return Vote::Unknown;
        };
        let Some(answer) = asked.question.answer(ids) else {
            return Vote::Other;
        };
        self.done(room, poll);
        late(asked.thread, &asked.question, &answer)
    }

    /// A text message in `room`'s conversation `thread`: the answer to the
    /// oldest question a run there waits on. The poll it answered, if any.
    pub fn text(&self, room: &str, thread: Option<&str>, body: &str) -> Option<String> {
        let mut waiting = self.waiting.lock().unwrap();
        waiting.retain(|w| !w.tx.is_closed());
        let i = waiting
            .iter()
            .position(|w| w.room == room && w.thread.as_deref() == thread)?;
        let w = waiting.remove(i);
        let poll = w.poll.clone();
        w.tx.send(body.to_owned()).ok()?;
        drop(waiting);
        self.done(room, &poll);
        Some(poll)
    }

    /// The question is answered: off file.
    fn done(&self, room: &str, poll: &str) {
        if let Ok(f) = self.file(room, poll) {
            let _ = std::fs::remove_file(f);
        }
    }

    /// The open questions of `room` (not yet answered), for `!vragen`.
    pub fn open(&self, room: &str) -> Vec<String> {
        let Ok(dir) = self.file(room, "x").map(|f| f.with_file_name("")) else {
            return Vec::new();
        };
        let mut found: Vec<(std::time::SystemTime, String)> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let when = e.metadata().ok()?.modified().ok()?;
                let asked: Asked =
                    serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()?;
                Some((when, asked.question.question))
            })
            .collect();
        found.sort();
        found.into_iter().map(|(_, q)| q).collect()
    }
}

fn late(thread: Option<String>, q: &Question, answer: &str) -> Vote {
    Vote::Late {
        thread,
        message: format!("🗳️ {} → {answer}", q.question),
    }
}

fn write(file: &Path, asked: &Asked) -> Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(file, serde_json::to_string_pretty(asked)?)
        .with_context(|| format!("could not write {}", file.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Value {
        json!({"questions": [{
            "question": "Which colour?",
            "header": "Colour",
            "options": [
                {"label": "Red", "description": "warm"},
                {"label": "Blue", "description": ""}
            ],
            "multiSelect": false
        }]})
    }

    #[test]
    fn parses_and_fills_in_answers() {
        let qs = questions(&input()).unwrap();
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].options[1], ("Blue".into(), "".into()));
        let out = answered(&input(), &[("Which colour?".into(), "Blue".into())]);
        assert_eq!(out["answers"]["Which colour?"], "Blue");
        assert_eq!(out["questions"], input()["questions"]);
        assert!(questions(&json!({"questions": []})).is_err());
    }

    #[test]
    fn poll_has_other_mentions_and_thread() {
        let q = &questions(&input()).unwrap()[0];
        let p = q.poll(&["@kas:x".into()], Some(("$root", "$last")));
        let start = &p["org.matrix.msc3381.poll.start"];
        assert_eq!(
            start["question"]["org.matrix.msc1767.text"],
            "❓ Colour: Which colour?"
        );
        let answers = start["answers"].as_array().unwrap();
        assert_eq!(answers.len(), 3);
        assert_eq!(answers[0]["org.matrix.msc1767.text"], "Red — warm");
        assert_eq!(answers[2]["id"], "other");
        assert_eq!(start["max_selections"], 1);
        assert_eq!(p["m.mentions"]["user_ids"][0], "@kas:x");
        assert_eq!(p["m.relates_to"]["event_id"], "$root");
        assert!(q.poll(&[], None).get("m.relates_to").is_none());
    }

    #[test]
    fn answers_from_votes() {
        let mut q = questions(&input()).unwrap().remove(0);
        assert_eq!(q.answer(&["a1".into()]).as_deref(), Some("Blue"));
        assert_eq!(q.answer(&["other".into()]), None);
        q.multi = true;
        assert_eq!(
            q.answer(&["a0".into(), "a1".into(), "other".into()])
                .as_deref(),
            Some("Red, Blue")
        );
    }

    #[tokio::test]
    async fn a_vote_or_a_text_answers_the_waiting_run() {
        let tmp = tempfile::tempdir().unwrap();
        let asks = Asks::new(tmp.path().to_owned());
        let q = questions(&input()).unwrap().remove(0);

        let rx = asks.wait("!r:x", Some("$t"), "$p1", &q).unwrap();
        assert_eq!(asks.open("!r:x"), ["Which colour?"]);
        assert_eq!(asks.vote("!r:x", "$p1", &["other".into()]), Vote::Other);
        assert_eq!(asks.vote("!r:x", "$p1", &["a0".into()]), Vote::Answered);
        assert_eq!(rx.await.unwrap(), "Red");
        assert!(asks.open("!r:x").is_empty());
        assert_eq!(asks.vote("!r:x", "$p1", &["a0".into()]), Vote::Unknown);

        let rx = asks.wait("!r:x", Some("$t"), "$p2", &q).unwrap();
        assert_eq!(asks.text("!r:x", None, "elsewhere"), None);
        assert_eq!(
            asks.text("!r:x", Some("$t"), "green").as_deref(),
            Some("$p2")
        );
        assert_eq!(rx.await.unwrap(), "green");
        assert_eq!(asks.text("!r:x", Some("$t"), "no question"), None);
    }

    #[test]
    fn a_vote_after_the_run_gave_up_starts_a_turn() {
        let tmp = tempfile::tempdir().unwrap();
        let asks = Asks::new(tmp.path().to_owned());
        let q = questions(&input()).unwrap().remove(0);
        drop(asks.wait("!r:x", Some("$t"), "$p", &q).unwrap());
        // A text no longer goes to the run that stopped waiting.
        assert_eq!(asks.text("!r:x", Some("$t"), "hi"), None);
        // parley restarted: only the file is left.
        let asks = Asks::new(tmp.path().to_owned());
        assert_eq!(
            asks.vote("!r:x", "$p", &["a1".into()]),
            Vote::Late {
                thread: Some("$t".into()),
                message: "🗳️ Which colour? → Blue".into()
            }
        );
        assert_eq!(asks.vote("!r:x", "$p", &["a1".into()]), Vote::Unknown);
    }
}
