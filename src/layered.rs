//! The `layered N` context mode: what a fresh run is told of the room's
//! past, from most to least verbatim.
//!
//! 1. The message itself.
//! 2. The thread it is in, word for word, and the room's last N messages
//!    outside that thread.
//! 3. Summaries of earlier days: one per closed day of this and last week,
//!    one per closed week before that (up to [`MAX_WEEKS`]). Days are UTC.
//!    A closed day never changes, so its summary is written once by a
//!    small run (after a reply, see [`jobs`]) and kept as a note (see
//!    [`note`]); a week's summary is made from its days'. The notes are
//!    Obsidian-style Markdown, linked day to week in their frontmatter, so
//!    they can live in Kas's vault.
//! 4. Where the full transcript is, so the agent can look things up itself
//!    when the past matters more than the summaries show.
//!
//! A first try: the layers and sizes are meant to be tuned as we see what
//! the agent actually reaches for.

use std::collections::HashMap;

use serde::Deserialize;

use crate::memory;

/// Messages outside the thread a layered room gets, when the mode does not say.
pub const DEFAULT_RECENT: usize = 10;
/// Weeks before last week that get a summary.
pub const MAX_WEEKS: i64 = 8;
/// Longest the quoted thread gets, in characters.
const MAX_THREAD_CHARS: usize = 40_000;
/// Longest the quoted recent messages get, in characters.
const MAX_RECENT_CHARS: usize = 20_000;
/// Longest one message is quoted in a day's summary input, in characters.
const MAX_DAY_MESSAGE_CHARS: usize = 1000;
/// Longest a day's summary input gets, in characters.
const MAX_DAY_CHARS: usize = 80_000;
const SECS_PER_DAY: u64 = 86_400;

#[derive(Debug, Clone, Deserialize)]
pub struct Msg {
    #[serde(default)]
    pub ts: u64,
    pub sender: String,
    #[serde(default)]
    pub thread: Option<String>,
    pub body: String,
}

impl Msg {
    fn day(&self) -> i64 {
        (self.ts / SECS_PER_DAY) as i64
    }

    fn quote(&self, max: usize) -> String {
        memory::quote(&self.sender, &self.body, max)
    }
}

/// The transcript's messages, oldest first, without commands like `!new`
/// and without the messages being answered (`current`, see
/// [`memory::drop_current`]).
pub fn parse(transcript: &str, current: &[&str]) -> Vec<Msg> {
    let mut msgs: Vec<Msg> = transcript
        .lines()
        .filter_map(|l| serde_json::from_str::<Msg>(l).ok())
        .filter(|m| !m.body.trim_start().starts_with('!'))
        .collect();
    memory::drop_current(&mut msgs, current, |m| (&m.sender, &m.body));
    msgs
}

/// The day (days since 1970-01-01, UTC) of a unix time.
pub fn day_of(secs: u64) -> i64 {
    (secs / SECS_PER_DAY) as i64
}

/// Weeks run Monday to Sunday; 1970-01-01 was a Thursday.
fn week_of(day: i64) -> i64 {
    (day + 3).div_euclid(7)
}

fn monday(week: i64) -> i64 {
    week * 7 - 3
}

/// Year, month and day of a day number.
fn civil(day: i64) -> (i64, i64, i64) {
    // Howard Hinnant's civil_from_days.
    let z = day + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The day number of January 1st of `year`.
fn new_year(year: i64) -> i64 {
    // Howard Hinnant's days_from_civil, for March 1st of the year before,
    // plus the 306 days from then to January 1st.
    let y = year - 1;
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doe = yoe * 365 + yoe / 4 - yoe / 100;
    era * 146_097 + doe - 719_468 + 306
}

/// `2026-10-26` for a day number.
pub fn date(day: i64) -> String {
    let (y, m, d) = civil(day);
    format!("{y:04}-{m:02}-{d:02}")
}

fn weekday(day: i64) -> &'static str {
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][(day + 3).rem_euclid(7) as usize]
}

/// The key of a day's summary: `2026-10-25`.
pub fn day_key(day: i64) -> String {
    date(day)
}

/// The key of a week's summary, its ISO week: `2026-W43`.
pub fn week_key(week: i64) -> String {
    let thursday = monday(week) + 3;
    let (year, _, _) = civil(thursday);
    format!("{year}-W{:02}", (thursday - new_year(year)) / 7 + 1)
}

/// The time span a summary covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Span {
    Day(i64),
    Week(i64),
}

/// Which days get their own summary and which weeks one together, as of
/// `today`: closed days of this and last week; closed weeks before that.
fn spans(msgs: &[Msg], today: i64) -> (Vec<i64>, Vec<i64>) {
    let this_week = week_of(today);
    let mut days: Vec<i64> = msgs.iter().map(Msg::day).filter(|&d| d < today).collect();
    days.dedup();
    let (recent, older): (Vec<i64>, Vec<i64>) =
        days.into_iter().partition(|&d| week_of(d) >= this_week - 1);
    let mut weeks: Vec<i64> = older
        .iter()
        .map(|&d| week_of(d))
        .filter(|&w| w >= this_week - 1 - MAX_WEEKS)
        .collect();
    weeks.dedup();
    (recent, weeks)
}

/// A summary still to write: its key, and the input for the summary run.
#[derive(Debug)]
pub struct Job {
    pub key: String,
    pub span: Span,
    /// The keys of the summaries this one is made from (a week's days).
    pub parts: Vec<String>,
    pub system_prompt: &'static str,
    pub input: String,
}

pub const DAY_PROMPT: &str = "\
You summarise one day of a Matrix chat room between people and an AI agent. \
The agent reads your summary later as background, in place of the messages. \
Answer with the summary only: short Markdown bullets, at most about 120 words: \
the topics, decisions, what was built or changed, open questions and promises. \
Name concrete things (files, commands, people, places) that help find the \
details later. No heading, no preamble.";

pub const WEEK_PROMPT: &str = "\
You summarise one week of a Matrix chat room between people and an AI agent, \
from the summaries of its days. The agent reads yours later as background. \
Answer with the summary only: short Markdown bullets, at most about 150 words: \
the main threads of the week, what was decided or done, what is still open. \
Keep concrete names that help find the details later. No heading, no preamble.";

/// The summaries still missing, days first. A week's job only appears once
/// all its days have one, so call again after writing the days'.
pub fn jobs(msgs: &[Msg], today: i64, have: &HashMap<String, String>) -> Vec<Job> {
    let this_week = week_of(today);
    let oldest_week = this_week - 1 - MAX_WEEKS;
    let mut by_day: Vec<(i64, Vec<&Msg>)> = Vec::new();
    for m in msgs {
        let d = m.day();
        if d >= today || week_of(d) < oldest_week {
            continue;
        }
        match by_day.last_mut() {
            Some((day, list)) if *day == d => list.push(m),
            _ => by_day.push((d, vec![m])),
        }
    }
    let mut out = Vec::new();
    for (day, list) in &by_day {
        let key = day_key(*day);
        if have.contains_key(&key) {
            continue;
        }
        let mut input = format!(
            "The messages of {} ({}), UTC:\n\n",
            date(*day),
            weekday(*day)
        );
        for m in list {
            let line = format!("{} {}\n\n", clock(m.ts), m.quote(MAX_DAY_MESSAGE_CHARS));
            if input.len() + line.len() > MAX_DAY_CHARS {
                input.push_str("[… the rest of the day is left out]\n");
                break;
            }
            input.push_str(&line);
        }
        out.push(Job {
            key,
            span: Span::Day(*day),
            parts: Vec::new(),
            system_prompt: DAY_PROMPT,
            input,
        });
    }
    let (_, weeks) = spans(msgs, today);
    for week in weeks {
        let key = week_key(week);
        if have.contains_key(&key) {
            continue;
        }
        let days: Vec<i64> = by_day
            .iter()
            .map(|(d, _)| *d)
            .filter(|&d| week_of(d) == week)
            .collect();
        let Some(parts) = days
            .iter()
            .map(|&d| {
                have.get(&day_key(d))
                    .map(|s| format!("## {} ({})\n\n{}", date(d), weekday(d), s.trim()))
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        out.push(Job {
            key,
            span: Span::Week(week),
            parts: days.iter().map(|&d| day_key(d)).collect(),
            system_prompt: WEEK_PROMPT,
            input: format!(
                "The days of the week of {}:\n\n{}",
                date(monday(week)),
                parts.join("\n\n")
            ),
        });
    }
    out
}

/// A summary as a vault note: frontmatter, then the summary. Notes link by
/// name (`[[parley 2026-W43]]`), so file names are `<prefix><key>.md`;
/// the prefix (the room's name) keeps them apart from Kas's daily notes.
/// Links go in the frontmatter only, as the vault's conventions ask.
pub fn note(job: &Job, room: &str, room_id: &str, prefix: &str, today: i64, body: &str) -> String {
    let link = |key: &str| format!("\"[[{prefix}{key}]]\"");
    let mut fm = vec![
        "type: room-summary".to_owned(),
        "author: claude".to_owned(),
        format!("room: {room}"),
        format!("room-id: \"{room_id}\""),
    ];
    match job.span {
        Span::Day(day) => {
            fm.push("period: day".into());
            fm.push(format!("date: {}", date(day)));
            fm.push(format!("daily: \"[[{}]]\"", date(day)));
            fm.push(format!("ai-notes: \"[[{}-ai]]\"", date(day)));
            fm.push(format!("up: {}", link(&week_key(week_of(day)))));
        }
        Span::Week(week) => {
            fm.push("period: week".into());
            fm.push(format!("date: {}", date(monday(week))));
            fm.push(format!(
                "down: [{}]",
                job.parts
                    .iter()
                    .map(|k| link(k))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    fm.push(format!("created: {}", date(today)));
    let tag: String = room
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    fm.push(format!("tags: [room-summary, {tag}]"));
    format!("---\n{}\n---\n\n{}\n", fm.join("\n"), body.trim())
}

/// A note's text without its frontmatter.
pub fn strip_frontmatter(text: &str) -> &str {
    text.strip_prefix("---\n")
        .and_then(|rest| rest.split_once("\n---\n"))
        .map_or(text, |(_, body)| body)
        .trim()
}

fn clock(ts: u64) -> String {
    let s = ts % SECS_PER_DAY;
    format!("{:02}:{:02}", s / 3600, s % 3600 / 60)
}

/// Quote `msgs` as `[sender] body` blocks, dropping the oldest (but the
/// first, with `keep_first`) while longer than `max` characters.
fn quote_all(msgs: &[&Msg], max: usize, keep_first: bool) -> String {
    let mut out: Vec<String> = msgs
        .iter()
        .map(|m| m.quote(memory::MAX_MESSAGE_CHARS))
        .collect();
    let first = usize::from(keep_first);
    while out.len() > first + 1 && out.iter().map(String::len).sum::<usize>() > max {
        out.remove(first);
    }
    out.join("\n\n")
}

/// The input for a layered run: the room's past as summaries, the recent
/// messages outside the thread, the thread so far, then the message.
pub fn input(
    msgs: &[Msg],
    thread: Option<&str>,
    recent: usize,
    today: i64,
    have: &HashMap<String, String>,
    transcript_path: Option<&str>,
    message: &str,
) -> String {
    let mut out = String::new();

    let (days, weeks) = spans(msgs, today);
    let mut past = Vec::new();
    for week in weeks {
        let text = have
            .get(&week_key(week))
            .map(|s| s.trim().to_owned())
            .unwrap_or_else(|| "(not summarised yet)".into());
        past.push(format!("### Week of {}\n\n{text}", date(monday(week))));
    }
    for day in days {
        let text = have
            .get(&day_key(day))
            .map(|s| s.trim().to_owned())
            .unwrap_or_else(|| "(not summarised yet)".into());
        past.push(format!("### {} ({})\n\n{text}", date(day), weekday(day)));
    }
    out.push_str("<room-history>\n");
    if past.is_empty() {
        out.push_str("No earlier days in this room to summarise.\n");
    } else {
        out.push_str("Summaries of earlier days in this room, oldest first (UTC days):\n\n");
        out.push_str(&past.join("\n\n"));
        out.push('\n');
    }
    // A run as another user cannot read the transcript: no path for it.
    let source = match transcript_path {
        Some(path) => format!(
            "Everything said in this room is in {path} (one JSON object per \
             line: ts, sender, thread, body). When the past matters more than \
             these notes show, look it up there, or in any other source you \
             can reach, rather than guessing."
        ),
        None => "When the past matters more than these notes show, look it up \
                 with the `read` tool, if you have it, rather than guessing."
            .to_owned(),
    };
    out.push_str(&format!(
        "\nToday is {} ({}). {source}\n</room-history>\n\n",
        date(today),
        weekday(today)
    ));

    let in_thread = |m: &&Msg| thread.is_some() && m.thread.as_deref() == thread;
    let others: Vec<&Msg> = msgs.iter().filter(|m| !in_thread(m)).collect();
    let others = &others[others.len().saturating_sub(recent)..];
    if !others.is_empty() {
        out.push_str(&format!(
            "<recent-messages>\nThe room's latest messages outside this thread, oldest first:\n\n{}\n</recent-messages>\n\n",
            quote_all(others, MAX_RECENT_CHARS, false)
        ));
    }
    let own: Vec<&Msg> = msgs.iter().filter(in_thread).collect();
    if !own.is_empty() {
        out.push_str(&format!(
            "<this-thread>\nThe thread this message is in, so far:\n\n{}\n</this-thread>\n\n",
            quote_all(&own, MAX_THREAD_CHARS, true)
        ));
    }
    out.push_str(message);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(ts: u64, sender: &str, thread: Option<&str>, body: &str) -> Msg {
        Msg {
            ts,
            sender: sender.into(),
            thread: thread.map(Into::into),
            body: body.into(),
        }
    }

    // 2026-10-26 is a Monday.
    const MON: i64 = 20_752;

    #[test]
    fn dates_and_weeks() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(MON), "2026-10-26");
        assert_eq!(weekday(MON), "Mon");
        assert_eq!(date(MON - 1), "2026-10-25");
        assert_eq!(date(11_016), "2000-02-29");
        assert_eq!(monday(week_of(MON + 6)), MON);
        assert_eq!(monday(week_of(MON - 1)), MON - 7);
        assert_eq!(week_key(week_of(MON + 3)), "2026-W44");
        assert_eq!(week_key(week_of(MON - 1)), "2026-W43");
        // ISO weeks belong to the year of their Thursday.
        assert_eq!(new_year(2026), 20_454);
        assert_eq!(date(new_year(2000)), "2000-01-01");
        assert_eq!(week_key(week_of(20_453)), "2026-W01"); // 2025-12-31
        assert_eq!(week_key(week_of(20_454 - 365 + 3)), "2025-W01"); // 2025-01-04
        assert_eq!(week_key(week_of(new_year(2021) + 2)), "2020-W53"); // 2021-01-03
        assert_eq!(clock(MON as u64 * 86_400 + 3_723), "01:02");
    }

    fn at(day: i64, secs: u64) -> u64 {
        day as u64 * SECS_PER_DAY + secs
    }

    #[test]
    fn parse_drops_commands_and_the_current_message() {
        let t = [
            r#"{"ts":1,"sender":"@kas:x","thread":null,"body":"hi"}"#,
            "broken",
            r#"{"ts":2,"sender":"@kas:x","thread":null,"body":"!context"}"#,
            r#"{"ts":3,"sender":"@kas:x","thread":"$t","body":"now"}"#,
        ]
        .join("\n");
        let m = parse(&t, &["now"]);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].body, "hi");
        assert_eq!(parse(&t, &["other"]).len(), 2);
    }

    #[test]
    fn days_then_weeks_are_summarised_once() {
        let today = MON + 2; // Wednesday
        let msgs = vec![
            msg(at(MON - 30, 10), "@kas:x", None, "long ago"), // 5 weeks back
            msg(at(MON - 29, 10), "agent", None, "reply"),
            msg(at(MON - 3, 60), "@kas:x", None, "last week"),
            msg(at(MON + 1, 70), "@kas:x", Some("$t"), "yesterday"),
            msg(at(today, 80), "@kas:x", None, "today"),
            msg(at(MON - 7 * (MAX_WEEKS + 3), 0), "@kas:x", None, "too old"),
        ];
        let mut have = HashMap::new();
        let keys = |j: &[Job]| j.iter().map(|j| j.key.clone()).collect::<Vec<_>>();
        let first = jobs(&msgs, today, &have);
        // No week yet: its days have no summaries.
        assert_eq!(
            keys(&first),
            ["2026-09-26", "2026-09-27", "2026-10-23", "2026-10-27"]
        );
        assert!(
            first[0].input.contains("00:00 [kas] long ago"),
            "{}",
            first[0].input
        );
        for j in &first {
            have.insert(j.key.clone(), format!("- sum of {}", j.key));
        }
        let second = jobs(&msgs, today, &have);
        assert_eq!(keys(&second), ["2026-W39"]);
        assert_eq!(second[0].parts, ["2026-09-26", "2026-09-27"]);
        assert_eq!(second[0].system_prompt, WEEK_PROMPT);
        assert!(
            second[0]
                .input
                .contains("## 2026-09-27 (Sun)\n\n- sum of 2026-09-27")
        );
        have.insert("2026-W39".into(), "- the week".into());
        assert!(jobs(&msgs, today, &have).is_empty());
    }

    #[test]
    fn input_has_history_recent_thread_and_message() {
        let today = MON + 2;
        let msgs = vec![
            msg(at(MON - 30, 10), "@kas:x", None, "long ago"),
            msg(at(MON + 1, 10), "@kas:x", None, "yesterday"),
            msg(at(today, 10), "@kas:x", Some("$t"), "root"),
            msg(at(today, 20), "@kas:x", None, "elsewhere 1"),
            msg(at(today, 30), "@kas:x", None, "elsewhere 2"),
            msg(at(today, 40), "agent", Some("$t"), "answer"),
        ];
        let have = HashMap::from([
            ("2026-W39".to_owned(), "- the week".to_owned()),
            ("2026-10-27".to_owned(), "- yesterday's gist".to_owned()),
        ]);
        let i = input(
            &msgs,
            Some("$t"),
            1,
            today,
            &have,
            Some("/r/t.jsonl"),
            "[Kas] q",
        );
        assert!(i.starts_with("<room-history>\n"), "{i}");
        assert!(
            i.contains(
                "### Week of 2026-09-21\n\n- the week\n\n### 2026-10-27 (Tue)\n\n- yesterday's gist"
            ),
            "{i}"
        );
        assert!(
            i.contains("Today is 2026-10-28 (Wed). Everything said in this room is in /r/t.jsonl")
        );
        // Only the last 1 message outside the thread.
        assert!(
            i.contains("oldest first:\n\n[kas] elsewhere 2\n</recent-messages>"),
            "{i}"
        );
        assert!(!i.contains("elsewhere 1"));
        assert!(
            i.contains("so far:\n\n[kas] root\n\n[agent] answer\n</this-thread>\n\n[Kas] q"),
            "{i}"
        );
        assert!(i.ends_with("[Kas] q"));

        // Missing summaries say so; no thread means no thread block.
        let i = input(
            &msgs,
            None,
            2,
            today,
            &HashMap::new(),
            Some("/r/t.jsonl"),
            "[Kas] q",
        );
        assert!(
            i.contains("### 2026-10-27 (Tue)\n\n(not summarised yet)"),
            "{i}"
        );
        assert!(!i.contains("<this-thread>"));
        assert!(
            i.contains("[kas] elsewhere 2\n\n[agent] answer\n</recent-messages>"),
            "{i}"
        );
    }

    #[test]
    fn notes_link_days_and_weeks_in_their_frontmatter() {
        let today = MON + 2;
        let msgs = vec![
            msg(at(MON - 30, 10), "@kas:x", None, "long ago"),
            msg(at(MON - 29, 10), "@kas:x", None, "next day"),
        ];
        let days = jobs(&msgs, today, &HashMap::new());
        let n = note(&days[0], "parley", "!r:x", "parley ", today, "- gist\n");
        assert_eq!(
            n,
            "---\ntype: room-summary\nauthor: claude\nroom: parley\nroom-id: \"!r:x\"\n\
             period: day\ndate: 2026-09-26\ndaily: \"[[2026-09-26]]\"\n\
             ai-notes: \"[[2026-09-26-ai]]\"\nup: \"[[parley 2026-W39]]\"\n\
             created: 2026-10-28\ntags: [room-summary, parley]\n---\n\n- gist\n"
        );
        assert_eq!(strip_frontmatter(&n), "- gist");
        assert_eq!(strip_frontmatter("- plain\n"), "- plain");

        let have: HashMap<_, _> = days.iter().map(|j| (j.key.clone(), "- d".into())).collect();
        let week = &jobs(&msgs, today, &have)[0];
        let n = note(week, "My Room", "!r:x", "My Room ", today, "- w");
        assert!(
            n.contains("period: week\ndate: 2026-09-21\ndown: [\"[[My Room 2026-09-26]]\", \"[[My Room 2026-09-27]]\"]\n"),
            "{n}"
        );
        assert!(n.contains("tags: [room-summary, my-room]"), "{n}");
    }

    #[test]
    fn without_a_path_it_points_at_the_read_tool() {
        let i = input(&[], None, 5, MON, &HashMap::new(), None, "[Kas] hi");
        assert!(!i.contains("transcript") && !i.contains("jsonl"), "{i}");
        assert!(i.contains("`read` tool"), "{i}");
    }

    #[test]
    fn empty_room_still_points_at_the_transcript() {
        let i = input(
            &[],
            Some("$t"),
            5,
            MON,
            &HashMap::new(),
            Some("/r/t.jsonl"),
            "[Kas] hi",
        );
        assert!(i.contains("No earlier days"));
        assert!(i.contains("/r/t.jsonl"));
        assert!(!i.contains("<recent-messages>"));
        assert!(i.ends_with("</room-history>\n\n[Kas] hi"));
    }

    #[test]
    fn long_threads_keep_their_root_and_newest_messages() {
        let big = "z".repeat(memory::MAX_MESSAGE_CHARS);
        let mut msgs = vec![msg(at(MON, 0), "@kas:x", Some("$t"), "root")];
        for _ in 0..40 {
            msgs.push(msg(at(MON, 1), "agent", Some("$t"), &big));
        }
        msgs.push(msg(at(MON, 2), "agent", Some("$t"), "last"));
        let i = input(&msgs, Some("$t"), 0, MON, &HashMap::new(), Some("/t"), "q");
        assert!(i.contains("so far:\n\n[kas] root\n\n"), "{}", &i[..300]);
        assert!(i.contains("[agent] last\n</this-thread>"));
        assert!(i.len() < MAX_THREAD_CHARS + 2_000, "{}", i.len());
    }
}
