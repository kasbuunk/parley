//! The `read` tool: messages of a room (or a part of it) from the Matrix
//! server, newest or oldest first, filtered, a page at a time.
//!
//! The server hands out events in chunks with opaque tokens. [`collect`]
//! walks chunks until it has `limit` matching messages, and returns a
//! cursor (`<skip>:<token>`) that resumes right after the last one.

use std::future::Future;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::access::Scope;

/// Most messages one call returns.
pub const MAX_LIMIT: usize = 100;
/// Most chunks one call walks before handing back a cursor.
pub const MAX_CHUNKS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    pub scope: Scope,
    pub limit: usize,
    pub oldest_first: bool,
    pub cursor: Cursor,
    /// Unix seconds, inclusive.
    pub since: Option<u64>,
    /// Unix seconds, exclusive.
    pub until: Option<u64>,
    /// Part of the sender's user ID, any case.
    pub sender: Option<String>,
    /// Part of the body, any case.
    pub contains: Option<String>,
}

/// Where a page starts: a chunk's token (`None`: the room's newest, or
/// oldest, end) and how many of its events were already returned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cursor {
    pub token: Option<String>,
    pub skip: usize,
}

impl Cursor {
    pub fn encode(&self) -> String {
        format!("{}:{}", self.skip, self.token.as_deref().unwrap_or(""))
    }

    pub fn decode(s: &str) -> Result<Self> {
        let (skip, token) = s.split_once(':').context("not a cursor from `read`")?;
        Ok(Self {
            skip: skip.parse().context("not a cursor from `read`")?,
            token: Some(token).filter(|t| !t.is_empty()).map(str::to_owned),
        })
    }
}

impl Query {
    /// From the tool's arguments (all optional).
    pub fn from_args(args: &Value, here: &Scope) -> Result<Self> {
        let s = |k: &str| args.get(k).and_then(Value::as_str);
        let limit = match args.get("limit") {
            None | Some(Value::Null) => 20,
            Some(v) => v.as_u64().context("limit must be a number")? as usize,
        }
        .clamp(1, MAX_LIMIT);
        let oldest_first = match s("order").map(str::trim) {
            None | Some("") | Some("newest") => false,
            Some("oldest") => true,
            Some(o) => bail!("order must be \"newest\" or \"oldest\", not {o:?}"),
        };
        Ok(Self {
            scope: Scope::parse(s("thread"), here)?,
            limit,
            oldest_first,
            cursor: match s("cursor") {
                Some(c) if !c.is_empty() => Cursor::decode(c)?,
                _ => Cursor::default(),
            },
            since: s("since").map(|t| parse_time(t, false)).transpose()?,
            until: s("until").map(|t| parse_time(t, true)).transpose()?,
            sender: s("sender").filter(|x| !x.is_empty()).map(str::to_lowercase),
            contains: s("contains")
                .filter(|x| !x.is_empty())
                .map(str::to_lowercase),
        })
    }

    fn matches(&self, m: &Msg) -> bool {
        let in_scope = match &self.scope {
            Scope::All => true,
            Scope::Main => m.thread.is_none(),
            Scope::Thread(t) => m.thread.as_deref() == Some(t) || &m.event_id == t,
        };
        in_scope
            && self.since.is_none_or(|s| m.ts_ms >= s * 1000)
            && self.until.is_none_or(|u| m.ts_ms < u * 1000)
            && self
                .sender
                .as_ref()
                .is_none_or(|s| m.sender.to_lowercase().contains(s))
            && self
                .contains
                .as_ref()
                .is_none_or(|c| m.body.to_lowercase().contains(c))
    }

    /// Past the time bound in the walking direction: nothing further can match.
    fn beyond(&self, m: &Msg) -> bool {
        if self.oldest_first {
            self.until.is_some_and(|u| m.ts_ms >= u * 1000)
        } else {
            self.since.is_some_and(|s| m.ts_ms < s * 1000)
        }
    }
}

/// Unix seconds, or `YYYY-MM-DD` (UTC; with `end`, the end of that day), or
/// `YYYY-MM-DDTHH:MM` (UTC).
pub fn parse_time(s: &str, end: bool) -> Result<u64> {
    let s = s.trim();
    if let Ok(n) = s.parse::<u64>() {
        return Ok(n);
    }
    let bad = || format!("{s:?} is not unix seconds, YYYY-MM-DD or YYYY-MM-DDTHH:MM");
    let (date, time) = match s.split_once(['T', ' ']) {
        Some((d, t)) => (d, Some(t.trim_end_matches('Z'))),
        None => (s, None),
    };
    let mut parts = date.splitn(3, '-').map(str::parse::<i64>);
    let (Some(Ok(y)), Some(Ok(m)), Some(Ok(d))) = (parts.next(), parts.next(), parts.next()) else {
        bail!(bad());
    };
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        bail!(bad());
    }
    let day = days_from_civil(y, m, d);
    let secs = match time {
        None => (day + i64::from(end)) * 86_400,
        Some(t) => {
            let mut hm = t.splitn(3, ':').map(str::parse::<i64>);
            let (Some(Ok(h)), Some(Ok(mi))) = (hm.next(), hm.next()) else {
                bail!(bad());
            };
            day * 86_400 + h * 3600 + mi * 60
        }
    };
    u64::try_from(secs).map_err(|_| anyhow::anyhow!(bad()))
}

/// Howard Hinnant's days_from_civil.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// One message, as `read` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msg {
    pub event_id: String,
    pub ts_ms: u64,
    pub sender: String,
    /// The thread's root, for a message in a thread.
    pub thread: Option<String>,
    pub body: String,
}

/// A message out of a raw (decrypted) timeline event; `None` for anything
/// else (state, reactions, edits).
pub fn parse_event(v: &Value) -> Option<Msg> {
    let content = v.get("content")?;
    let relates = content.get("m.relates_to");
    let rel = relates
        .and_then(|r| r.get("rel_type"))
        .and_then(Value::as_str);
    let body = match v.get("type")?.as_str()? {
        "m.room.message" if rel == Some("m.replace") => return None,
        "m.room.message" => {
            let body = content.get("body").and_then(Value::as_str).unwrap_or("");
            match content.get("msgtype").and_then(Value::as_str) {
                Some("m.text" | "m.notice" | "m.emote") | None => body.to_owned(),
                Some(kind) => format!("[{kind}] {body}"),
            }
        }
        "m.room.encrypted" => "[encrypted; parley has no key for it]".to_owned(),
        _ => return None,
    };
    Some(Msg {
        event_id: v.get("event_id")?.as_str()?.to_owned(),
        ts_ms: v.get("origin_server_ts")?.as_u64()?,
        sender: v.get("sender")?.as_str()?.to_owned(),
        thread: (rel == Some("m.thread"))
            .then(|| relates?.get("event_id")?.as_str().map(str::to_owned))
            .flatten(),
        body,
    })
}

/// One chunk from the server: its events, in walking order, and the token
/// of the next chunk (`None`: the end).
pub struct Chunk {
    pub events: Vec<Value>,
    pub next: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Page {
    /// In walking order (newest first, unless `oldest_first`).
    pub msgs: Vec<Msg>,
    /// Where the next page starts; `None` when there is nothing more.
    pub next: Option<Cursor>,
    pub scanned: usize,
}

/// Walk chunks from `q.cursor` (`fetch(token)` gets one) until `q.limit`
/// messages match, the history ends, the time bound is passed, or
/// [`MAX_CHUNKS`] chunks were walked.
pub async fn collect<F, Fut>(q: &Query, mut fetch: F) -> Result<Page>
where
    F: FnMut(Option<String>) -> Fut,
    Fut: Future<Output = Result<Chunk>>,
{
    let mut token = q.cursor.token.clone();
    let mut skip = q.cursor.skip;
    let mut msgs = Vec::new();
    let mut scanned = 0;
    for _ in 0..MAX_CHUNKS {
        let chunk = fetch(token.clone()).await?;
        for (i, ev) in chunk.events.iter().enumerate().skip(skip) {
            scanned += 1;
            let Some(m) = parse_event(ev) else { continue };
            if q.beyond(&m) {
                return Ok(Page {
                    msgs,
                    next: None,
                    scanned,
                });
            }
            if q.matches(&m) {
                msgs.push(m);
                if msgs.len() == q.limit {
                    return Ok(Page {
                        msgs,
                        next: Some(Cursor { token, skip: i + 1 }),
                        scanned,
                    });
                }
            }
        }
        match chunk.next {
            Some(n) if !chunk.events.is_empty() || Some(&n) != token.as_ref() => {
                token = Some(n);
                skip = 0;
            }
            _ => {
                return Ok(Page {
                    msgs,
                    next: None,
                    scanned,
                });
            }
        }
    }
    Ok(Page {
        msgs,
        next: Some(Cursor { token, skip: 0 }),
        scanned,
    })
}

/// What the tool returns: the page, oldest first, then how to go on.
pub fn render(room: &str, q: &Query, page: &Page) -> String {
    let mut out = format!(
        "{} message(s) from {room} ({} first, {} events scanned):\n",
        page.msgs.len(),
        if q.oldest_first { "oldest" } else { "newest" },
        page.scanned
    );
    let mut msgs: Vec<&Msg> = page.msgs.iter().collect();
    if !q.oldest_first {
        msgs.reverse();
    }
    for m in msgs {
        let secs = m.ts_ms / 1000;
        let day = crate::layered::day_of(secs);
        let rest = secs % 86_400;
        let thread = m
            .thread
            .as_deref()
            .map(|t| format!(" in thread {t}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "\n--- {} {:02}:{:02} UTC, {}{thread} ({})\n{}\n",
            crate::layered::date(day),
            rest / 3600,
            rest % 3600 / 60,
            m.sender,
            m.event_id,
            m.body
        ));
    }
    match &page.next {
        Some(c) => out.push_str(&format!(
            "\nMore {}: call `read` again with the same filters and cursor \"{}\".\n",
            if q.oldest_first {
                "after these"
            } else {
                "before these"
            },
            c.encode()
        )),
        None => out.push_str("\nNothing more matches.\n"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(id: &str, ts: u64, sender: &str, body: &str, thread: Option<&str>) -> Value {
        let mut content = json!({"msgtype": "m.text", "body": body});
        if let Some(t) = thread {
            content["m.relates_to"] = json!({"rel_type": "m.thread", "event_id": t});
        }
        json!({"type": "m.room.message", "event_id": id, "origin_server_ts": ts * 1000,
               "sender": sender, "content": content})
    }

    /// Newest first, in chunks of 3: tokens "t0" (None), "t1", "t2".
    fn history() -> Vec<Value> {
        (0..8u64)
            .rev()
            .map(|i| {
                let thread = (i % 2 == 1).then_some("$e1");
                ev(
                    &format!("$e{i}"),
                    100 + i,
                    if i % 3 == 0 { "@kas:x" } else { "@agent:x" },
                    &format!("msg {i}"),
                    thread,
                )
            })
            .collect()
    }

    async fn run(q: &Query) -> Page {
        let all = history();
        collect(q, |token: Option<String>| {
            let n: usize = token.map(|t| t[1..].parse().unwrap()).unwrap_or(0);
            let events: Vec<Value> = all.iter().skip(n * 3).take(3).cloned().collect();
            let next = (n * 3 + 3 < all.len()).then(|| format!("t{}", n + 1));
            async move { Ok(Chunk { events, next }) }
        })
        .await
        .unwrap()
    }

    fn query(args: Value) -> Query {
        Query::from_args(&args, &Scope::Thread("$e1".into())).unwrap()
    }

    fn ids(p: &Page) -> Vec<&str> {
        p.msgs.iter().map(|m| m.event_id.as_str()).collect()
    }

    #[tokio::test]
    async fn pages_resume_where_they_stopped() {
        let mut q = query(json!({"limit": 4}));
        let p = run(&q).await;
        assert_eq!(ids(&p), ["$e7", "$e6", "$e5", "$e4"]);
        let c = p.next.unwrap();
        assert_eq!(c.encode(), "1:t1");
        q.cursor = Cursor::decode(&c.encode()).unwrap();
        let p = run(&q).await;
        assert_eq!(ids(&p), ["$e3", "$e2", "$e1", "$e0"]);
        q.cursor = p.next.unwrap();
        let p = run(&q).await;
        assert!(p.msgs.is_empty());
        assert_eq!(p.next, None);
    }

    #[tokio::test]
    async fn filters_threads_senders_text_and_time() {
        let p = run(&query(json!({"thread": "here"}))).await;
        assert_eq!(ids(&p), ["$e7", "$e5", "$e3", "$e1"]);
        let p = run(&query(json!({"thread": "main"}))).await;
        assert_eq!(ids(&p), ["$e6", "$e4", "$e2", "$e0"]);
        let p = run(&query(json!({"sender": "KAS"}))).await;
        assert_eq!(ids(&p), ["$e6", "$e3", "$e0"]);
        let p = run(&query(json!({"contains": "MSG 5"}))).await;
        assert_eq!(ids(&p), ["$e5"]);
        // Stops walking once past `since`.
        let p = run(&query(json!({"since": "104"}))).await;
        assert_eq!(ids(&p), ["$e7", "$e6", "$e5", "$e4"]);
        assert_eq!(p.scanned, 5);
        let p = run(&query(json!({"until": "102"}))).await;
        assert_eq!(ids(&p), ["$e1", "$e0"]);
    }

    #[test]
    fn renders_oldest_first_with_a_cursor() {
        let q = query(json!({}));
        let page = Page {
            msgs: vec![
                parse_event(&ev("$b", 1_761_480_000 + 60, "@kas:x", "second", None)).unwrap(),
                parse_event(&ev("$a", 1_761_480_000, "@kas:x", "first", Some("$r"))).unwrap(),
            ],
            next: Some(Cursor {
                token: Some("tok".into()),
                skip: 2,
            }),
            scanned: 9,
        };
        let text = render("!r:x", &q, &page);
        let first = text.find("first").unwrap();
        assert!(first < text.find("second").unwrap(), "{text}");
        assert!(
            text.contains("2025-10-26 12:00 UTC, @kas:x in thread $r ($a)"),
            "{text}"
        );
        assert!(text.contains("cursor \"2:tok\""), "{text}");
    }

    #[test]
    fn events_parse() {
        assert!(parse_event(&json!({"type": "m.reaction", "event_id": "$x", "origin_server_ts": 1, "sender": "@a:x", "content": {}})).is_none());
        let edit = json!({"type": "m.room.message", "event_id": "$x", "origin_server_ts": 1, "sender": "@a:x",
            "content": {"body": "* x", "m.relates_to": {"rel_type": "m.replace", "event_id": "$y"}}});
        assert!(parse_event(&edit).is_none());
        let img = json!({"type": "m.room.message", "event_id": "$x", "origin_server_ts": 1, "sender": "@a:x",
            "content": {"msgtype": "m.image", "body": "cat.png"}});
        assert_eq!(parse_event(&img).unwrap().body, "[m.image] cat.png");
    }

    #[test]
    fn times_parse() {
        assert_eq!(parse_time("1761436800", false).unwrap(), 1_761_436_800);
        assert_eq!(parse_time("2025-10-26", false).unwrap(), 1_761_436_800);
        assert_eq!(
            parse_time("2025-10-26", true).unwrap(),
            1_761_436_800 + 86_400
        );
        assert_eq!(
            parse_time("2025-10-26T01:30", false).unwrap(),
            1_761_436_800 + 5400
        );
        assert_eq!(parse_time("1970-01-01", false).unwrap(), 0);
        assert!(parse_time("yesterday", false).is_err());
        assert!(parse_time("2025-13-01", false).is_err());
    }
}
