# parley

A Matrix bot (`@agent:chat.lab.artintel.ai`) that lets people talk to agents
from any Matrix client. Kas writes to it from Element on his phone and Mac. It
runs on the home lab (NixOS) as `parley.service`, user `kasbuunk`.

You may be running *inside* parley right now: the "parley" chat room is bound
to this repo, so you can change the bot you are talking through. Breaking it
is acceptable (Kas is around), but keep the way back open; see "Deploying".

## How it works

- `src/main.rs`: Matrix side. Login and saved session, end-to-end encryption,
  emoji (SAS) verification, invites (only from the allow list), and the
  message handler. A text message is recorded in the room transcript, then
  answered by an agent run in a spawned task (typing notice while it runs).
  A message in a Matrix thread is answered in that thread. `!new` clears the
  session of the main timeline or thread it is sent in. With `threads = true`
  in room.toml, a main-timeline message is answered in a new thread on it.
- `src/agent.rs`: one agent turn. Reads the room's folder, builds a `Run`,
  calls the `Harness`. `ClaudeCode` is the only harness: headless
  `claude -p … --output-format json`. Per-room lock keeps replies in order;
  a circuit breaker pauses runs after 3 failures in a row.
- `src/memory.rs`: context modes, what a turn remembers of earlier ones:
  `session` (resume one Claude session), `fresh`, `window N` (quote the
  transcript's last N messages), `digest N` (window plus `MEMORY.md`
  notes, rewritten by a small Haiku run after each reply), `layered N`
  (`src/layered.rs`: cached day/week summaries, last N messages outside the
  thread, the whole thread, and a pointer to the transcript). Summaries are
  vault notes (frontmatter, day ↔ week links); room.toml's `name` and
  `summaries` put them in Kas's vault as `ai/rooms/<name>/<name> <period>.md`. Set with
  `context` in room.toml or `!context <mode>` in the room (saved in the
  room's `context` file); `!context` alone shows the mode.
- `src/breaker.rs`: the circuit breaker (also used for the Matrix sync loop).

## State on the lab

`~/.local/share/parley/`:

- `session.json`, `store/`: the bot's login and encryption keys. Never delete
  or move these: losing them makes `@agent` a new, unverified device, and
  every person has to verify it again.
- `rooms/` is a symlink to `/mnt/nas/data/parley/rooms`, so restic backs it
  up with the rest of the NAS. `rooms/<room id>/`:
  - `room.toml`: the room's profile (fields in `Profile`, `src/agent.rs`).
    Without it a room is locked down: no tools, no settings, no memory.
  - `SYSTEM.md`: system prompt (replaces or is appended, per `prompt`).
  - `session`, `threads/<root event id>.session`: Claude session IDs.
  - `transcript.jsonl`: every message and reply, one JSON object per line
    (`ts` unix seconds, `sender`, `thread`, `body`). Not sent to the model.
    Read or grep it to look back at earlier conversations, in this room or
    others.
- `restart`: create this file to ask parley to restart (see below).

Room IDs and names: `journalctl -u parley | grep joined` shows which room is
which. The rooms Kas uses: nixos (`!GQiDox…`), life (`!76kJZ…`), and this one.

## Deploying a change

1. Edit, then `cargo test` (all tests must pass: the Nix build runs them).
2. Commit and push (`git push`; this clone has a read-write deploy key).
3. Build: `nix build -o ~/.config/parley/current`. If it fails, the link
   still points at the old build, so nothing is broken yet. Fix and retry.
4. Restart: `touch ~/.local/share/parley/restart`, then finish your turn.
   After your reply is sent, parley exits and systemd starts the new build.
   Do **not** `systemctl restart parley` yourself: you run as a child of
   parley, so that kills you before your reply goes out.
5. Kas checks the result in the chat; `journalctl -u parley` shows the log.

To roll back: `git revert`, build, restart, or point
`~/.config/parley/current` at an earlier `/nix/store/…-parley-0.1.0`.

The NixOS side (the systemd unit) lives in `~/.config/nixos/modules/parley.nix`.

## Conventions

- Rust edition 2024, `cargo fmt`, no new clippy warnings.
- Tests live next to the code; the Claude harness is tested against fake
  shell scripts, never the real `claude`.
- Commit messages say why, not only what.
