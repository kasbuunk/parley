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
  Conversations are actors: the main timeline and each thread take one
  turn at a time, side by side. Messages that come in during a turn wait
  (`Inbox`) and are answered together in the next one; a schedule's fire
  gets a turn of its own. `!! <message>` stops the running turn and
  answers at once (the next turn is told it was stopped); `!stop` stops
  it and drops what waits.
- `src/agent.rs`: one agent turn. Reads the room's folder, builds a `Run`,
  calls the `Harness`. `ClaudeCode` is the only harness: headless
  `claude -p … --output-format json`. Per-conversation lock keeps turns in order;
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
- `src/post.rs`: posting and reading as tools. With `reply = "tool"` in
  room.toml the run's final text is not posted; the agent gets MCP tools
  (served by `parley mcp`, which Claude Code starts per run and which
  forwards to parley over `post.sock` in the data dir, with a per-run
  token): `post(body, thread?, room?)`, `read(room?, thread?, order?,
  limit?, cursor?, since?, until?, sender?, contains?)` and `rooms()`.
  It may post several times or not at all. A final answer of `[silent]`
  (no posts) gets a ✓ reaction; any other final text with no posts is
  posted as a fallback, so errors still show.
- `src/ask.rs`: questions as Matrix polls. Claude Code's own
  `AskUserQuestion` is handed to parley's MCP tool `approve`
  (`--permission-prompt-tool`, hidden from the model with
  `--disallowedTools`); parley posts each question as a poll (options plus
  "Anders, namelijk…", mentioning the room's people) in the turn's
  conversation and waits up to 10 min for a vote, or for any text message
  there (not `!!`), which then answers it instead of starting a turn. The
  answers go back as `AskUserQuestion`'s `answers`. Unanswered, the model
  is told to end its turn; a later vote starts a turn with
  `🗳️ <question> → <answer>`. Open polls: `polls/<event id>.json` per
  room; `!vragen` lists them.
- `src/access.rs`: what a turn may read and post to. Its own room always;
  other rooms, their main timeline, or single threads through
  `[[access]]` tables in room.toml (`room` = ID or another room's `name`,
  `thread`, `read`, `write`).
- `src/read.rs`: the `read` tool's paging and filters. It reads from the
  Matrix server (`/messages`, or `/relations` for a thread), so it reaches
  back to a room's first message; a cursor (`<skip>:<token>`) resumes a page.
- `src/say.rs`: `parley say [--thread <id>] [--notice] <room id>` (body on
  stdin) posts one message as another account (its own PARLEY_DATA_DIR)
  and exits. For alarm bots like `@lab-alert`, whose messages parley then
  answers like any member's; `--notice` posts without starting a run.
- `src/schedule.rs`: schedules, runs that start without a message. Kinds
  from transmit (`at`/`in` once, `every`, `cron` in local time with
  seconds; `times` or forever). MCP tools `schedule`, `schedules`,
  `cancel`, `handoff`; room commands `!schedules`, `!cancel <id>`, and
  `!stop` (kills this conversation's run). Stored per room as
  `schedules/<id>.json` (done ones in `schedules/done/`), with `<id>.md`
  as the handoff scratchboard shown to the next run. `only_if_new` skips
  a fire, without an agent, when no person wrote since the last run. A
  scheduler loop in `main.rs` fires them through the same path as a
  message. `trigger = "schedule"` in room.toml: messages start no run;
  `trigger = "mention"` with `names = ["melchior"]`: only a message that
  mentions the bot (Matrix pill or a name as a word) or follows up in a
  thread it spoke in starts one; the rest is only recorded.
  Loop guard: each schedule has a generation (`depth`: 0 when made in a
  run a person started, the firing schedule's plus one in a scheduled
  run); `schedule` refuses above room.toml's `max_depth` (default 10).
  At most 20 live schedules per room, fires at least a minute apart.
- `src/clients.rs`: standing clients, the same tools for agents outside
  parley's runs (a Claude Code session on the lab, or on the Mac over
  ssh). One file per client in `clients/<name>.toml` (0600): `home` room,
  `token`, `[[access]]` grants as in room.toml. Loaded at start; the
  client runs `parley mcp --client <name>`. Current: `claude-lab`,
  `claude-mac` (home nixos, read/write on every room).
- `src/voice.rs`: voice messages. An `m.audio` event is downloaded
  (decrypted by the SDK) to the room's `media/<event id>.<ext>`, run
  through `parley-transcribe` (`transcribe/parley-transcribe.py`,
  faster-whisper large-v3-turbo on the CPU, the model room-transcribe
  uses; built next to `parley` by the flake), and then handled as the
  text message `🎤 <transcript>`. `voice` in room.toml: `reply` (default:
  the transcript is also posted as a reply, where the answer will come),
  `quiet`, or `off`. A 5-second clip takes about 20 s.
- `src/speech.rs`: speaking. The MCP tool `speak(text, thread?, room?)`
  posts a Matrix voice message with the text as caption. The voice is an
  outside program: `<program> <out.ogg>`, text on stdin, Ogg Opus out.
  Default `parley-speak` (Piper, Dutch voice `nl_NL-alex-medium`, pinned in
  the flake); set `PARLEY_SPEAK` to swap it, for example for ElevenLabs.
- Tracing: `trace = "https://agents.lab.artintel.ai"` in room.toml makes
  every run keep its Claude Code session file (a new session gets its ID
  up front with `--session-id`) and puts `[🔍 sessie](<trace>/sessions/<id>)`
  under the turn's first post here (or under a text answer). The viewer
  is agentsview (`~/.config/nixos/modules/agentsview.nix`), which reads
  `~/.claude/projects` live; on in life, nixos, artintel and parley.
- `src/cgroup.rs`: every Claude Code run in a cgroup of its own. With
  `Delegate=memory pids` on the unit, parley moves itself to
  `parley.service/main/` and starts each run in `run-<n>/` with
  `memory.max` 6G and `memory.swap.max` 2G (`PARLEY_RUN_MEMORY`,
  `PARLEY_RUN_SWAP`), so the kernel or oomd stops the run that is too
  big, not parley (15 Nov: oomd killed all of parley). The run's error
  then says it ran out of memory. When a run ends, what it left running
  is killed. At most `PARLEY_MAX_RUNS` (4) runs at once over all rooms;
  the rest wait (a run waiting on a poll holds its place). Without
  delegation, runs share parley's cgroup (logged at start).
- `src/confine.rs`: `run_as = "<user>"` in room.toml runs the room's
  harness (summaries and digests too) as that Linux user:
  `sudo -n -u <user> /run/current-system/sw/bin/<user>-claude <args>`,
  cleared env, cwd `/` (the wrapper picks the workdir, so `workdir` is
  refused). The message goes over stdin and the MCP config is a file, so
  neither is in sudo's log. Each turn gets its own socket in
  `/run/parley-<user>` that accepts only its token. No `image`, `task`,
  `tasks`, trace link or transcript path. It fails closed: no sudo, no
  wrapper or no cgroup means no run.
- `display_name = "Paul"` in room.toml: the bot's name in that room only
  (its `m.room.member` event, keeping avatar and the rest; the global
  profile stays `agent`). Applied at start, on joining and at each turn,
  only when it differs; a failure is logged and the room goes on.
- `src/platform.rs`: `runtime = "agent-platform"` in room.toml runs each
  turn as one Kubernetes Job (gVisor, Cilium, only Anthropic reachable)
  through agent-platform's `bin/spawn`, with
  `/var/lib/agent-platform/parley.kubeconfig`. The platform owns the pod
  (image, resources, sandbox, egress labels) in a profile, a PodTemplate
  in its `profiles/`; `pod_profile` in room.toml picks one. The message and system
  prompt go over `kubectl attach -i`, never in argv or the Job spec. No
  MCP tools, no session: use `reply = "text"` and `window N`. The Job is
  deleted when the turn ends, and a failure never falls back to a local run.
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
  - `schedules/`: the room's schedules (see `src/schedule.rs`).
  - `transcript.jsonl`: every message and reply, one JSON object per line
    (`ts` unix seconds, `sender`, `thread`, `body`). Not sent to the model.
    Read or grep it to look back at earlier conversations, in this room or
    others.
- `clients/`: standing clients' files with their tokens (see `src/clients.rs`).
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

Never `cargo run` (or start any second parley) on the lab: it logs in
with the same session and store, and rebinds `post.sock`, so the running
bot's tools stop working until it restarts.

## Conventions

- Rust edition 2024, `cargo fmt`, no new clippy warnings.
- Tests live next to the code; the Claude harness is tested against fake
  shell scripts, never the real `claude`.
- Commit messages say why, not only what.
