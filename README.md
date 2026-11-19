# parley

A Matrix bot that puts AI agents in group chats. People and agents talk in
the same rooms, from any Matrix client (Element on a phone, a laptop, a
terminal). Each room is bound to an agent with its own working directory,
tools, system prompt and memory. The agent harness is
[Claude Code](https://docs.claude.com/en/docs/claude-code) running headless.

It runs on one home-lab machine (NixOS, systemd) and has been used daily for
infrastructure work, a personal assistant, a software project, and a family
room. This README explains the design for engineers who want to build
something similar: a multiplayer chat with several people and several agents.

## Why Matrix

- **Clients for free.** Notifications, threads, reactions, polls, voice
  messages, read receipts and end-to-end encryption already exist on every
  platform. The bot only has to be a good room member.
- **People and agents are the same kind of thing.** The bot is an ordinary
  member with its own device. It receives room keys like everyone else, so
  rooms stay encrypted and the homeserver never sees plaintext.
- **Rooms are the unit of scope.** One room = one agent profile = one
  working directory and permission set. Inviting someone into a room is the
  access control.

## Architecture

```
 Element / any client
        │  Matrix (E2EE)
        ▼
 ┌─────────────── parley (one process) ───────────────┐
 │ sync loop ─► message handler ─► conversation actor │
 │                                   (one per thread) │
 │                                        │           │
 │   scheduler ──────────────────────────►│           │
 │                                        ▼           │
 │                         agent run (cgroup, ≤ N)    │
 │                         claude -p … --resume …     │
 │                                        │           │
 │   post.sock ◄── parley mcp ◄───────────┘           │
 │   (post, read, schedule, task, approve …)          │
 └────────────────────────────────────────────────────┘
        │
        ▼
 rooms/<room id>/  room.toml, SYSTEM.md, transcript.jsonl,
                   sessions, schedules, summaries
```

| File | What it does |
| --- | --- |
| `src/main.rs` | Login, E2EE, emoji verification, invites (allow list), message handler, conversation actors, scheduler loop |
| `src/agent.rs` | One agent turn: reads the room profile, builds the run, calls the harness, circuit breaker |
| `src/memory.rs`, `src/layered.rs` | Context modes (see below) |
| `src/post.rs` | MCP tools over a Unix socket: `post`, `read`, `rooms` |
| `src/read.rs` | Paged, filtered reading of room history from the homeserver |
| `src/access.rs` | Which other rooms and threads a turn may read or post to |
| `src/schedule.rs` | Runs that start without a message (once, interval, cron) |
| `src/ask.rs` | The agent's questions become Matrix polls |
| `src/tasks.rs` | Tasks for people, closed with a ✅ reaction |
| `src/voice.rs` | Voice messages transcribed locally with faster-whisper |
| `src/cgroup.rs` | Each run in its own cgroup with memory limits |
| `src/clients.rs` | The same tools for agents outside parley (e.g. a terminal session) |
| `src/say.rs` | Post as another account (alert bots) |

## The multiplayer problems, and how parley handles them

### Who is talking

Every message is passed to the agent as `[Name] text`. The agent sees who
said what, and a room's system prompt can say who the people are and how to
treat them. A room can also require that only listed members are present
(`members` in room.toml); if a stranger joins, the agent stops answering.

### Turns, concurrency and interruptions

Each conversation (the main timeline, and each Matrix thread) is an
**actor** that runs one turn at a time. Threads run side by side.

- Messages that arrive during a turn wait in an inbox and are answered
  together in the next turn, so three quick messages make one answer, not
  three.
- `!! <message>` stops the running turn and answers at once; the next turn
  is told it was interrupted.
- `!stop` stops the turn and drops what waits.
- With `threads = true`, every main-timeline message starts a new thread,
  so each topic gets its own conversation and its own session.

### Agents that speak when they want to

With `reply = "tool"`, the agent's final output is **not** posted. Instead
it gets a `post` tool. It can post a short "on it" note, do ten minutes of
work, post the result, post in another thread, or say nothing at all
(`[silent]` gives the message a ✓ reaction). This turns the agent from a
request/response endpoint into a participant.

### Agents in several rooms

A room's `[[access]]` grants let its agent `read` and `post` in other rooms
or single threads. To make another room's agent *act*, an agent schedules a
run there with a self-contained prompt. Messages from the bot never start a
run by themselves, which prevents agents from looping on each other.

### Loop guards

Agents can schedule runs, and scheduled runs can schedule more. Each
schedule has a generation number; above `max_depth` (default 10) scheduling
is refused. At most 20 live schedules per room, fires at least a minute
apart, and `only_if_new` skips a fire when no person wrote since the last
run. A circuit breaker pauses a room after three failed runs in a row.

### Asking people things

Claude Code's built-in `AskUserQuestion` is routed (via
`--permission-prompt-tool`) to parley, which posts it as a **Matrix poll**
that mentions the room's people. A vote, or any text reply in the thread,
answers it. If nobody answers in ten minutes, the agent ends its turn and a
later vote starts a new turn with the answer.

### Tasks for people

When the agent needs a person to do something (sign in, flip a switch), the
`task` tool posts it prominently and keeps it as a note. A ✅ reaction closes
it; `!taken` lists what is open.

### Resource isolation

Agents run real tools (shells, builds, tests). Every run gets its own cgroup
(`memory.max`, `memory.swap.max`), so the kernel kills the run that grows
too big, not the bot. At most `PARLEY_MAX_RUNS` (default 4) runs at once;
the rest wait. Whatever a run leaves behind is killed when it ends.

### Who may trigger the agent

`trigger` in room.toml: `message` (every message), `mention` (only when the
bot is mentioned or named, or in a thread it already spoke in) or
`schedule` (only schedules; messages are only recorded). `mention` is what
you want in a room that is mostly people talking to each other.

## Context window management

This is the core design question for long-lived chat agents. A room lives
for months; a model's context does not. parley records every message in the
room's `transcript.jsonl` and lets each room choose what a turn sees. Set it
with `context` in room.toml, or `!context <mode>` in the room.

| Mode | What the agent sees | Good for |
| --- | --- | --- |
| `session` | One Claude Code session per conversation, resumed every turn | Short-lived threads, coding work where tool results matter |
| `fresh` | Only the new message | Stateless commands, alert handling |
| `window N` | The last N messages of the transcript, quoted | Cheap, predictable cost |
| `digest N` | Window, plus a `MEMORY.md` of notes that a small model (Haiku) rewrites after each reply | Long-running assistant without big contexts |
| `layered N` | See below | Rooms that live for months |

**`layered N`** builds the context like human memory, coarse far away and
detailed nearby:

1. Week summaries of older weeks, day summaries of recent days. Each is
   written once by a small model and cached as a Markdown note (with
   frontmatter and day ↔ week links, so they double as a browsable archive,
   e.g. in Obsidian).
2. The last N messages of the room outside the current thread.
3. The current thread in full.
4. A pointer to the full transcript file and the `read` tool, with the
   instruction to look things up instead of guessing.

Trade-offs we found:

- **Sessions grow without bound** and carry stale tool output; a resumed
  session is great for an hour of coding and bad for a room that lives a
  year.
- **Summaries lose detail**, so the agent must be able to get the detail
  back. The transcript on disk and the `read` tool (which pages through the
  homeserver's history with filters on sender, date, text and thread) are
  that escape hatch.
- **Threads are the natural boundary.** One thread = one topic = one full
  context; everything else is summarised.
- Summaries are made lazily, once per day or week, so the cost is close to
  zero per turn.

## Other features

- **Voice messages** are transcribed on the CPU with faster-whisper
  (`large-v3-turbo`) and handled as text (`🎤 …`). The transcript is posted
  back so people see what the agent heard.
- **Tracing.** With `trace` set, each answer links to the run's session in a
  session viewer, so you can watch what the agent did.
- **Standing clients.** An agent in a terminal (outside parley) can use the
  same `post`/`read`/`schedule` tools via `parley mcp --client <name>`, so it
  can report to a room.
- **Alert bots.** `parley say` posts as another account; parley then treats
  the alert like any member's message and the room's agent can act on it.
- **Room commands.** `!new` (new session), `!context`, `!schedules`,
  `!cancel <id>`, `!stop`, `!! …`, `!vragen` (open polls), `!taken` (open
  tasks).

## A room profile

`rooms/<room id>/room.toml`. A room without one gets no tools and no memory.

```toml
name = "nixos"
workdir = "/home/me/.config/nixos"
tools = "default"
settings = ["user", "project"]   # load the repo's CLAUDE.md
permission_mode = "bypassPermissions"
members = ["@me:example.org"]    # only run while nobody else is in the room
context = "layered 10"
threads = true
reply = "tool"
trigger = "message"
trace = "https://agents.example.org"

[[access]]
room = "life"      # another room's name or ID
read = true
write = true
```

`SYSTEM.md` next to it is the room's system prompt. By default it replaces
Claude Code's own; with `prompt = "append"` it is added to it.

## Running it

Requirements: a Matrix homeserver and an account for the bot, Claude Code
installed and logged in for the user that runs parley, and Nix (or Cargo).

```sh
nix build            # or: cargo build --release
```

Environment:

| Variable | Meaning |
| --- | --- |
| `PARLEY_HOMESERVER` | Homeserver URL |
| `PARLEY_PASSWORD_FILE` | First line user ID, second line password; only used on first start |
| `PARLEY_ALLOWED` | Comma-separated user IDs that may invite the bot |
| `PARLEY_DATA_DIR` | State: login, encryption store, rooms |
| `PARLEY_CLAUDE` | The `claude` binary |
| `PARLEY_MAX_RUNS`, `PARLEY_RUN_MEMORY`, `PARLEY_RUN_SWAP` | Run limits |

Run it as a systemd service with `Delegate=memory pids` to get per-run
cgroups. Invite the bot from an allowed account, then verify it with emoji
(the emoji appear in the bot's log). Keep `session.json` and `store/` in the
data dir safe: they are the bot's device and keys.

Security note: a room profile with `tools = "default"` and
`bypassPermissions` gives everyone in that room a shell as the bot's user.
Use `members` and the invite allow list accordingly.

## Development

```sh
cargo test     # the Claude harness is tested against fake shell scripts
cargo fmt
cargo clippy
```

This is a personal project, shaped around one household and one machine;
some defaults reflect that. Issues and pull requests are welcome.

## License

[MIT](LICENSE)
