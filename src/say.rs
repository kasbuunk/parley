//! `parley say`: post one message as some other account and exit.
//!
//! For bots that only announce, like `@lab-alert` (the nightly update's
//! alarm on the lab). They are not parley: they answer nothing and run no
//! agent. But parley answers *them*, like anyone in a room's `members`, so
//! an alert can start an agent run by being an ordinary message.
//!
//!   parley say [--thread <event id>] [--notice] <room id>   (body on stdin)
//!
//! Prints the new event's ID. The account is the one in PARLEY_DATA_DIR
//! (its own, not parley's): `session.json` + `store/`, made on the first
//! run from PARLEY_PASSWORD_FILE like parley's. Rooms are end-to-end
//! encrypted, which is why this is a client with a store and not a curl.
//! An invite to the room is accepted first (that is how the bot joins).
//!
//! `--notice` sends an `m.notice`: parley answers only `m.text`, so a
//! notice is shown but starts no run.

use anyhow::{Context, Result, bail};
use matrix_sdk::{
    RoomState,
    config::SyncSettings,
    ruma::{
        OwnedEventId, OwnedRoomId,
        events::relation::Thread,
        events::room::message::{Relation, RoomMessageEventContent},
    },
};
use std::io::Read;

/// What `parley say` was asked to do.
#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub room: String,
    pub thread: Option<String>,
    pub notice: bool,
}

impl Args {
    /// The arguments after `say`.
    pub fn parse(args: &[String]) -> Result<Self> {
        let (mut room, mut thread, mut notice) = (None, None, false);
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--thread" => {
                    thread = Some(it.next().context("--thread needs an event ID")?.clone())
                }
                "--notice" => notice = true,
                s if s.starts_with("--") => bail!("unknown option {s}"),
                s if room.is_none() => room = Some(s.to_owned()),
                s => bail!("unexpected argument {s:?}"),
            }
        }
        Ok(Self {
            room: room.context("usage: parley say [--thread <event id>] [--notice] <room id>")?,
            thread,
            notice,
        })
    }
}

pub async fn main(args: Args) -> Result<()> {
    let mut body = String::new();
    std::io::stdin().read_to_string(&mut body)?;
    if body.trim().is_empty() {
        bail!("nothing to say: the body (stdin) is empty");
    }
    let room_id = OwnedRoomId::try_from(args.room.as_str()).context("bad room ID")?;

    let client = crate::login().await?;
    client.sync_once(SyncSettings::default()).await?;
    let room = client
        .get_room(&room_id)
        .with_context(|| format!("not in and not invited to {room_id}"))?;
    if room.state() == RoomState::Invited {
        room.join().await?;
        client.sync_once(SyncSettings::default()).await?;
    }
    let room = client
        .get_room(&room_id)
        .context("room gone after joining")?;
    if room.state() != RoomState::Joined {
        bail!("{room_id}: {:?}, not joined", room.state());
    }

    let mut content = if args.notice {
        RoomMessageEventContent::notice_markdown(body)
    } else {
        RoomMessageEventContent::text_markdown(body)
    };
    if let Some(t) = args.thread {
        let root = OwnedEventId::try_from(t.as_str()).context("bad thread ID")?;
        content.relates_to = Some(Relation::Thread(Thread::plain(root.clone(), root)));
    }
    let sent = room.send(content).await?;
    println!("{}", sent.response.event_id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Result<Args> {
        Args::parse(&s.iter().map(|a| a.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn parses_room_thread_and_notice() {
        assert_eq!(
            args(&["--thread", "$e", "!r:x", "--notice"]).unwrap(),
            Args {
                room: "!r:x".into(),
                thread: Some("$e".into()),
                notice: true
            }
        );
    }

    #[test]
    fn needs_a_room_and_no_extras() {
        assert!(args(&[]).is_err());
        assert!(args(&["!r:x", "!s:x"]).is_err());
        assert!(args(&["--thread"]).is_err());
        assert!(args(&["--loud", "!r:x"]).is_err());
    }
}
