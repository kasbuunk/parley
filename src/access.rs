//! What a turn may read and where it may post, beyond its own room.
//!
//! A room's own room is always readable and writable, all of it. Other
//! rooms come from `[[access]]` tables in room.toml:
//!
//! ```toml
//! [[access]]
//! room = "life"          # another room's `name`, or its ID ("!…:server")
//! read = true            # whole room
//!
//! [[access]]
//! room = "nixos"
//! thread = "$rootid"     # only this thread ("main": only the main timeline)
//! read = true
//! write = true
//! ```
//!
//! A grant on a whole room covers its main timeline and every thread; a
//! grant on the main timeline or one thread covers only that.

use std::collections::HashMap;

use anyhow::{Result, bail};
use serde::Deserialize;

/// One `[[access]]` table, as written.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub room: String,
    #[serde(default)]
    pub thread: Option<String>,
    #[serde(default)]
    pub read: bool,
    #[serde(default)]
    pub write: bool,
}

/// A part of a room.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// The main timeline and every thread.
    All,
    /// The main timeline only.
    Main,
    /// The thread rooted at this event (the root included).
    Thread(String),
}

impl Scope {
    /// `None` → all, "main" → main timeline, "$…" → that thread, "here" →
    /// `here` (the conversation the turn answers).
    pub fn parse(arg: Option<&str>, here: &Scope) -> Result<Self> {
        match arg.map(str::trim) {
            None | Some("") => Ok(Self::All),
            Some(t) if t.eq_ignore_ascii_case("main") => Ok(Self::Main),
            Some(t) if t.eq_ignore_ascii_case("here") => Ok(here.clone()),
            Some(t) if t.starts_with('$') && !t.contains('/') => Ok(Self::Thread(t.to_owned())),
            Some(t) => bail!(
                "thread must be omitted, \"main\", \"here\" or a thread's root event ID, not {t:?}"
            ),
        }
    }

    fn covers(&self, other: &Scope) -> bool {
        matches!(self, Scope::All) || self == other
    }

    fn describe(&self) -> String {
        match self {
            Scope::All => "whole room".into(),
            Scope::Main => "main timeline".into(),
            Scope::Thread(t) => format!("thread {t}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Rule {
    room: String,
    name: Option<String>,
    scope: Scope,
    read: bool,
    write: bool,
}

/// The rooms one turn may read and post to.
#[derive(Debug, Clone)]
pub struct Access {
    /// The room the turn answers.
    pub here: String,
    /// The conversation it answers: a thread, or the main timeline.
    pub here_scope: Scope,
    rules: Vec<Rule>,
}

impl Access {
    /// Rules for a turn in `here` (named `name`), answering `thread` (or the
    /// main timeline), with the room's grants. `names` maps room names to
    /// IDs; a grant naming an unknown room is skipped, and reported back.
    pub fn new(
        here: &str,
        name: Option<&str>,
        thread: Option<&str>,
        grants: &[Grant],
        names: &HashMap<String, String>,
    ) -> (Self, Vec<String>) {
        let id_name: HashMap<&str, &str> = names
            .iter()
            .map(|(n, id)| (id.as_str(), n.as_str()))
            .collect();
        let mut rules = vec![Rule {
            room: here.to_owned(),
            name: name.map(str::to_owned),
            scope: Scope::All,
            read: true,
            write: true,
        }];
        let mut problems = Vec::new();
        for g in grants {
            let room = if g.room.starts_with('!') {
                g.room.clone()
            } else if let Some(id) = names.get(&g.room) {
                id.clone()
            } else {
                problems.push(format!("no room is named {:?}", g.room));
                continue;
            };
            let scope = match Scope::parse(g.thread.as_deref(), &Scope::All) {
                Ok(s) if g.thread.as_deref().map(str::trim) != Some("here") => s,
                Ok(_) => {
                    problems.push(format!(
                        "{}: thread = \"here\" means nothing in room.toml",
                        g.room
                    ));
                    continue;
                }
                Err(e) => {
                    problems.push(format!("{}: {e}", g.room));
                    continue;
                }
            };
            rules.push(Rule {
                name: id_name.get(room.as_str()).map(|n| (*n).to_owned()),
                room,
                scope,
                read: g.read,
                write: g.write,
            });
        }
        let here_scope = match thread {
            Some(t) => Scope::Thread(t.to_owned()),
            None => Scope::Main,
        };
        (
            Self {
                here: here.to_owned(),
                here_scope,
                rules,
            },
            problems,
        )
    }

    /// The room ID for a tool's `room` argument: omitted or "here" → this
    /// room; an ID; or the name of a room this turn has a grant on.
    pub fn room(&self, arg: Option<&str>) -> Result<String> {
        match arg.map(str::trim) {
            None | Some("") => Ok(self.here.clone()),
            Some(r) if r.eq_ignore_ascii_case("here") => Ok(self.here.clone()),
            Some(r) if r.starts_with('!') => Ok(r.to_owned()),
            Some(r) => match self.rules.iter().find(|x| x.name.as_deref() == Some(r)) {
                Some(rule) => Ok(rule.room.clone()),
                None => bail!("no room named {r:?} is open to this turn; see the `rooms` tool"),
            },
        }
    }

    /// Whether this turn may read (or, with `write`, post to) `scope` of `room`.
    pub fn check(&self, room: &str, scope: &Scope, write: bool) -> Result<()> {
        let ok = self.rules.iter().any(|r| {
            r.room == room && (if write { r.write } else { r.read }) && r.scope.covers(scope)
        });
        if ok {
            return Ok(());
        }
        let what = if write { "post to" } else { "read" };
        let open: Vec<String> = self
            .rules
            .iter()
            .filter(|r| r.room == room && (if write { r.write } else { r.read }))
            .map(|r| r.scope.describe())
            .collect();
        if open.is_empty() {
            bail!("this turn may not {what} {room}; see the `rooms` tool")
        }
        bail!(
            "this turn may not {what} the {} of {room}, only: {}",
            scope.describe(),
            open.join(", ")
        )
    }

    /// What the `rooms` tool says.
    pub fn describe(&self) -> String {
        let mut out = String::from("Rooms open to this turn:\n");
        for r in &self.rules {
            let rights = match (r.read, r.write) {
                (true, true) => "read, post",
                (true, false) => "read",
                (false, true) => "post",
                (false, false) => continue,
            };
            let name = r
                .name
                .as_deref()
                .map(|n| format!("{n} "))
                .unwrap_or_default();
            let here = if r.room == self.here {
                " (this room)"
            } else {
                ""
            };
            out.push_str(&format!(
                "- {name}{}{here}: {}, {rights}\n",
                r.room,
                r.scope.describe()
            ));
        }
        out.push_str(&format!(
            "This turn answers the {} of this room.\n",
            self.here_scope.describe()
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> HashMap<String, String> {
        [("life", "!life:x"), ("nixos", "!nix:x")]
            .into_iter()
            .map(|(a, b)| (a.to_owned(), b.to_owned()))
            .collect()
    }

    fn grants(toml_text: &str) -> Vec<Grant> {
        #[derive(Deserialize)]
        struct T {
            access: Vec<Grant>,
        }
        toml::from_str::<T>(toml_text).unwrap().access
    }

    #[test]
    fn own_room_is_open_and_grants_are_scoped() {
        let g = grants(
            r#"
            [[access]]
            room = "life"
            read = true

            [[access]]
            room = "!nix:x"
            thread = "$t"
            read = true
            write = true

            [[access]]
            room = "nope"
            read = true
            "#,
        );
        let (a, problems) = Access::new("!me:x", Some("parley"), Some("$here"), &g, &names());
        assert_eq!(problems, ["no room is named \"nope\""]);
        assert_eq!(a.here_scope, Scope::Thread("$here".into()));

        a.check("!me:x", &Scope::All, true).unwrap();
        a.check("!life:x", &Scope::All, false).unwrap();
        a.check("!life:x", &Scope::Thread("$any".into()), false)
            .unwrap();
        assert!(a.check("!life:x", &Scope::Main, true).is_err());
        a.check("!nix:x", &Scope::Thread("$t".into()), true)
            .unwrap();
        let e = a
            .check("!nix:x", &Scope::All, false)
            .unwrap_err()
            .to_string();
        assert!(e.contains("only: thread $t"), "{e}");
        assert!(a.check("!other:x", &Scope::Main, false).is_err());

        assert_eq!(a.room(None).unwrap(), "!me:x");
        assert_eq!(a.room(Some("here")).unwrap(), "!me:x");
        assert_eq!(a.room(Some("life")).unwrap(), "!life:x");
        assert_eq!(a.room(Some("nixos")).unwrap(), "!nix:x");
        assert!(a.room(Some("artintel")).is_err());

        let d = a.describe();
        assert!(
            d.contains("parley !me:x (this room): whole room, read, post"),
            "{d}"
        );
        assert!(d.contains("life !life:x: whole room, read"), "{d}");
        assert!(d.contains("nixos !nix:x: thread $t, read, post"), "{d}");
    }

    #[test]
    fn scopes_parse() {
        let here = Scope::Thread("$h".into());
        assert_eq!(Scope::parse(None, &here).unwrap(), Scope::All);
        assert_eq!(Scope::parse(Some("Main"), &here).unwrap(), Scope::Main);
        assert_eq!(Scope::parse(Some("here"), &here).unwrap(), here);
        assert_eq!(
            Scope::parse(Some("$x"), &here).unwrap(),
            Scope::Thread("$x".into())
        );
        assert!(Scope::parse(Some("x"), &here).is_err());
    }
}
