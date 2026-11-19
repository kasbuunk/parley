# parley

parley is a Matrix bot that puts AI agents into ordinary chat rooms, next to
the people in them. You talk to an agent from any Matrix client (Element on
a phone, a laptop, the web). Several people and several agents share rooms,
threads, schedules and tasks. Each agent run is a headless
[Claude Code](https://docs.claude.com/en/docs/claude-code) process.

This README is for engineers who want to build something like it: a chat
system where several people and agents work together. It explains the
design and why it looks the way it does. Read the code for the details;
every module starts with a doc comment that explains it.

It runs as one systemd service on a NixOS home lab. It is about 10k lines of
Rust (edition 2024, `matrix-sdk`, `tokio`) with about 120 tests.

## Contents

- [Why Matrix](#why-matrix)
- [The core loop](#the-core-loop)
- [Rooms are profiles](#rooms-are-profiles)
- [Conversations are actors](#conversations-are-actors)
- [Context window management](#context-window-management)
- [Agents as room members: tools instead of replies](#agents-as-room-members-tools-instead-of-replies)
- [Multiple agents and rooms](#multiple-agents-and-rooms)
- [Schedules: runs nobody started](#schedules-runs-nobody-started)
- [Asking people: polls and tasks](#asking-people-polls-and-tasks)
- [Voice](#voice)
- [Operations: isolation, failure, tracing](#operations-isolation-failure-tracing)
- [Module map](#module-map)
- [Running it](#running-it)
- [Lessons](#lessons)

## Why Matrix

We did not want to build a chat client, accounts, push notifications or
end-to-end encryption. Matrix already has all of that, and it is
self-hostable (we run our own homeserver). The bot is one more account
(`@agent:…`) with its own device:

- **Every client works at once.** Phone, desktop, web. Notifications,
  threads, reactions, polls, mentions and voice messages come from the
  clients you already have.
- **Encryption stays end to end.** The bot is a room member and gets the
  room keys like anyone else. The homeserver never sees plain text. People
  verify the bot's device with emoji (SAS) once.
- **People and agents are the same kind of thing.** A person, the bot and
  an alert script (`parley say`, which posts as another account) are all
  room members. Most of the multi-party behaviour comes from that one
  choice.
- **Invites are the permission model.** The bot only joins rooms when
  someone on its allow list (`PARLEY_ALLOWED`) invites it. A room can also
  require that *every* member is on a list (`members` in `room.toml`).
  Otherwise the agent stays silent, so a stranger who is added later gets
  no shell.

## The core loop

```
Matrix event ─► record in transcript.jsonl ─► conversation actor (queue)
                                                   │
                                                   ▼
                           build Run: room profile + system prompt
                                      + context (see below) + message
                                                   │
                                                   ▼
                       Harness: `claude -p … --output-format json`
                       (in its own cgroup, with parley's MCP tools)
                                                   │
                                                   ▼
                       posts to Matrix (final answer or `post` tool calls)
```

1. Every text message (and every transcribed voice message) goes into the
   room's `transcript.jsonl` first: `ts`, `sender`, `thread`, `body`. That
   file is the source of truth for history. It is never sent to the model
   as a whole.
2. The message goes to its *conversation*: the room's main timeline or one
   thread.
3. A turn builds a `Run` from the room's folder (profile, system prompt,
   context) and hands it to a `Harness`. Today there is one harness,
   `ClaudeCode`. The trait exists so another agent runtime can be added
   without touching the Matrix side.
4. While the run works the room shows a typing notice. The answer goes back
   to the same thread.

## Rooms are profiles

A room is configured by a folder: `rooms/<room id>/`. In it:

| File | What it is |
|---|---|
| `room.toml` | The profile: tools, working directory, permissions, context mode, triggers, access to other rooms. Without it the room is locked down: no tools, no memory. |
| `SYSTEM.md` | The system prompt, either replacing Claude Code's own or appended to it (`prompt = "append"`). |
| `transcript.jsonl` | Everything said in the room. |
| `session`, `threads/*.session` | Claude Code session IDs, for the `session` context mode. |
| `schedules/`, `polls/` | Pending schedules and open polls. |

So "an agent" in parley is not a separate service. It is a room profile:
which repo it works in, which tools it gets, what it may read, what it
remembers. Our rooms include an ops agent bound to the NixOS config
(`nixos`), a personal assistant on a notes vault (`life`), a work agent
(`artintel`), and this repo itself (`parley`). The bot in that last room can
change, build and restart itself. Each agent has its own name and persona
in its `SYSTEM.md`.

A short example:

```toml
workdir = "/home/me/git/my-repo"
tools = "default"                 # or "" for none, or "Read,Grep"
settings = ["user", "project"]    # load the repo's CLAUDE.md and .claude/
permission_mode = "bypassPermissions"
members = ["@me:example.org"]     # only run while every member is listed
threads = true                    # answer each new message in a new thread
context = "layered 10"            # see below
reply = "tool"                    # agent posts via a tool, see below
trigger = "message"               # or "mention" / "schedule"

[[access]]
room = "ops"                      # another room, by name or ID
read = true
write = true
```

All fields are in `Profile` in `src/agent.rs`.

## Conversations are actors

With more than one person, messages arrive while the agent is still busy.
Our first version took a per-room lock and answered every message on its
own, which produced several overlapping answers. Now:

- **Each conversation is an actor.** The main timeline and every thread
  take one turn at a time. Different threads run side by side.
- **Messages that arrive during a turn wait in an inbox.** They are
  answered *together* in the next turn. Three quick follow-ups get one
  answer.
- **`!! <message>`** stops the running turn and answers right away. The
  next turn is told that it was stopped. **`!stop`** stops the turn and
  drops what is waiting.
- **`threads = true`** answers every main-timeline message in a new thread
  started from it. Each topic gets its own thread, its own actor and (in
  `session` mode) its own session. This mattered more than we expected: the
  main timeline stays readable, and parallel work does not mix.

A schedule's fire gets a turn of its own, so it never merges with what a
person just wrote.

## Context window management

This is the design question with the most choices. In a chat, a "turn" is a
new message, sometimes days after the previous one, sometimes in a
different thread. What should the model see? parley does not pick one
answer. Each room chooses a **context mode** (`context` in `room.toml`, or
`!context <mode>` in the room at runtime):

| Mode | Each turn sees | Good for | Cost |
|---|---|---|---|
| `session` | One Claude Code session per timeline or thread, resumed each turn (`--resume`). Claude Code compacts it when it gets long. `!new` starts over. | Long, focused work in one thread, where tool results matter | Grows without limit; compaction loses detail without telling you; nothing from outside the thread |
| `fresh` | Only the message | Stateless helpers, alert triage | No memory at all |
| `window N` | The room's last N transcript messages, quoted before the message | Small, chatty rooms | Anything older than N is gone |
| `digest N` | `window N` plus a `MEMORY.md` of notes (what is settled, what is open). A small Haiku run rewrites it after every reply. | Rooms with ongoing decisions | One extra small run per reply; notes drift if nobody checks them |
| `layered N` | See below | Rooms that live for weeks with many threads | A few extra small runs per day |

### `layered N`, our default

Each turn starts a **fresh** process. The prompt is built from layers,
from most to least literal:

1. **The message itself.**
2. **The whole thread it is in**, word for word, and **the room's last N
   messages outside that thread**, so the agent knows what else is going
   on.
3. **Summaries of earlier days and weeks.** One per finished day of this
   week and last week, one per finished week before that (up to 8 weeks).
   A finished day never changes, so its summary is written **once** by a
   small model run and cached. A week's summary is made from its days'
   summaries. Days are UTC, so "finished" is the same everywhere.
4. **A pointer to the full transcript** and the instruction to search it
   (and other rooms' transcripts) when the past matters more than the
   summaries show.

Why this works for us:

- **Bounded, and roughly constant.** The prompt size depends on the length
  of the thread and on N, not on the age of the room. Each layer has a
  character cap.
- **Cheap.** Summaries are made once per day and per week, not per turn.
  They are cached as files.
- **The agent can look things up.** The summaries are an index, not the
  memory. The agent has file tools and the transcript's path, so it can
  `grep` for an exact quote. This "summary plus pointer" pattern is the
  main idea. Older approaches either stuff everything into the prompt or
  hide it behind a retrieval step that the model cannot control.
- **People can read the summaries.** They are Markdown notes with
  frontmatter and day ↔ week links. With `summaries = <dir>` they go into
  a personal Obsidian vault, so the same summaries are useful to people
  too.
- **No hidden compaction.** In `session` mode the harness decides what to
  forget. In `layered` mode the code decides, and you can test it.

What it costs: a fresh process does not have the tool results of earlier
turns (file contents it read, command output). In a long coding thread it
has to read files again. For that kind of work, `!context session` in that
room is the better choice. The mode can change at any time without losing
anything, because the transcript is always the source of truth.

### Things that helped regardless of mode

- **Keep one transcript, outside the model.** Every mode is a view of
  `transcript.jsonl`. Switching modes or rebuilding summaries never loses
  data.
- **Tell the model where it is.** The prompt says whether a message is in
  the main timeline or a thread, who sent it, and in which room.
- **Cap everything in characters,** per message and per layer. One pasted
  log should not push the rest of the context out.
- **Leave commands out** (`!new`, `!context`) of the history.

## Agents as room members: tools instead of replies

The simple model is that the final answer of a run is posted as the
reply. That breaks once an agent works for a long time or works with
others. With `reply = "tool"` the final answer is **not** posted. The run
gets MCP tools instead, served by parley itself (`parley mcp`, which Claude
Code starts for each run and which talks to parley over a Unix socket with
a per-run token):

- `post(body, thread?, room?)`: post now, here or in another room. Can be
  called several times ("starting, this takes ten minutes" … "done").
- `read(room?, thread?, since?, until?, sender?, contains?, …)`: read
  Matrix history directly from the server, paged, back to the first
  message.
- `rooms()`: which rooms this run may read or post to.

A run may also post nothing. A final answer of `[silent]` puts a ✓
reaction on the message instead. This matters in rooms where an agent
watches alerts or where several people talk among themselves: the agent
does not have to answer every message.

If a run posts nothing and its final text is not a marker, that text is
posted anyway, so errors are never lost.

## Multiple agents and rooms

- **Access grants.** `[[access]]` tables in `room.toml` give a room's
  agent read or write access to other rooms, to their main timeline only,
  or to one thread. The room itself is always open.
- **Handing work over.** An agent that finds work for another room posts
  there (to inform) and schedules a run there (to make that agent act),
  with a self-contained prompt. The bot's own messages never start a run,
  so two agents cannot reply to each other forever. The prompts set the
  rule "one hop": do not hand work back, do not ask for a reply.
- **Mention-only rooms.** `trigger = "mention"` with `names = ["…"]`: in a
  room where people mostly talk to each other, the agent only answers when
  it is mentioned (a Matrix pill or its name as a word), or when someone
  follows up in a thread it already spoke in. Everything else is still
  recorded, so it knows the context when someone asks.
- **Standing clients.** The same tools are available outside parley's
  runs. A Claude Code session on a laptop can `read` and `post` to rooms
  through `parley mcp --client <name>` (`src/clients.rs`).
- **Other accounts.** `parley say` posts as another Matrix account (for
  example an alert bot). parley answers those messages like any member's.

## Schedules: runs nobody started

Agents can plan their own future work with the MCP tools `schedule`,
`schedules`, `cancel` and `handoff`:

- Once (`at`, `in`), repeating (`every`, `cron` in local time), with an
  optional count (`times`).
- `only_if_new`: skip a fire, without starting a model, if no person wrote
  since the last run.
- `handoff`: a scratchpad (`schedules/<id>.md`) that the next run of that
  schedule sees. This makes long jobs work: do a step, write down where you
  are, schedule the next step. The state lives on, the process does not.
- `trigger = "schedule"`: a room where messages start no run and only
  schedules do.

Agents that schedule agents need limits on loops. Each schedule has a
generation: 0 when made by a run that a person started, +1 when made by a
scheduled run. Above `max_depth` (default 10), `schedule` is refused. Also:
at most 20 live schedules per room, and fires at least a minute apart.
Room commands: `!schedules`, `!cancel <id>`.

## Asking people: polls and tasks

An agent that needs a decision should not hide the question in paragraph
four of a long post.

- **Questions are Matrix polls.** Claude Code's own `AskUserQuestion` tool
  is routed to parley (through `--permission-prompt-tool`). parley posts
  each question as a poll with the options plus "Other…", mentions the
  room's people so they get a notification, and waits up to 10 minutes. A
  vote, or any plain text reply in that thread, is the answer. With no
  answer, the model is told to end its turn. A vote that comes later
  starts a new turn with the answer. `!vragen` (or `!questions`) lists open polls.
- **Tasks are for things only a person can do** (sign in, flip a switch,
  check something). The `task` tool posts a 📌 message with a mention and
  writes a note to a Markdown folder (in our case an Obsidian vault, also
  listed in that day's note). A ✅ reaction closes it (parley adds 🏁).
  `!taken` lists open tasks.

## Voice

An `m.audio` message is downloaded (the SDK decrypts it), transcribed
locally with faster-whisper (`transcribe/parley-transcribe.py`, CPU only)
and then handled as the text message `🎤 <transcript>`. With
`voice = "reply"` the transcript is also posted, so the speaker can check
what the agent heard. A 5-second clip takes about 20 seconds on our
hardware.

## Operations: isolation, failure, tracing

Agents run real tools on a real machine, so the bot has to survive them.

- **A cgroup per run.** parley moves itself into `parley.service/main/` and
  starts each run in `run-<n>/` with `memory.max` (6G) and
  `memory.swap.max` (2G). If a run uses too much memory, the kernel kills
  that run, not the bot. (Before this, systemd-oomd once killed all of
  parley, with five parallel runs and their queues.) Whatever a run leaves
  running is killed when it ends. At most 4 runs at once over all rooms;
  the rest wait.
- **Circuit breakers.** After 3 failed runs in a row, all runs pause for
  5 minutes instead of burning tokens on a broken setup (an expired login
  breaks every room at once). The Matrix sync loop
  has its own breaker.
- **Tracing.** With `trace = <url>`, every run keeps its Claude Code
  session file, and the turn's first post links to it in a session viewer
  (we use agentsview). You can follow a long run live from your phone.
- **Self-deploy.** The bot can rebuild itself: edit, `cargo test`, commit,
  `nix build -o ~/.config/parley/current`, then touch a `restart` file.
  parley exits after the current reply has been sent, and systemd starts
  the new build. A failed build leaves the old link in place. Rolling back
  means pointing the link at an older store path.

## Module map

| File | What it does |
|---|---|
| `src/main.rs` | Matrix: login, E2EE, verification, invites, the message handler, conversation actors, inbox, scheduler loop |
| `src/agent.rs` | One turn: `Profile`, `Run`, the `Harness` trait, `ClaudeCode` |
| `src/memory.rs` | Context modes `session`, `fresh`, `window`, `digest` |
| `src/layered.rs` | The `layered` mode: day/week summaries, layers, caps |
| `src/post.rs` | The MCP server (`parley mcp`) and the `post`/`read`/`rooms` tools |
| `src/read.rs` | Paging and filters for `read` |
| `src/access.rs` | `[[access]]` grants: what a run may read and post to |
| `src/schedule.rs` | Schedules, handoffs, loop limits |
| `src/ask.rs` | `AskUserQuestion` as Matrix polls |
| `src/tasks.rs` | Tasks for people |
| `src/voice.rs` | Voice messages to text |
| `src/clients.rs` | Standing MCP clients outside parley's runs |
| `src/say.rs` | `parley say`: post as another account |
| `src/cgroup.rs` | One cgroup per run, concurrency limit |
| `src/breaker.rs` | Circuit breaker |

## Running it

parley is built with Nix (`nix build`), which also builds the
transcriber. `cargo build` and `cargo test` work for the Rust part alone.
The tests use fake shell scripts in place of `claude`, so they need no API
access.

Configuration is in environment variables (see the top of `src/main.rs`):
`PARLEY_HOMESERVER`, `PARLEY_PASSWORD_FILE` (only read on first login),
`PARLEY_ALLOWED`, `PARLEY_DATA_DIR`, `PARLEY_CLAUDE`, `PARLEY_MAX_RUNS`,
`PARLEY_RUN_MEMORY`, `PARLEY_RUN_SWAP`. Our systemd unit lives in a
separate NixOS config. It needs `Delegate=memory pids` for the per-run
cgroups.

Keep the data directory's `session.json` and `store/` safe. They hold the
bot's device and encryption keys. If you lose them, the bot is a new
device and everyone has to verify it again. Never run two instances on the
same data directory.

## Lessons

What we would tell someone starting on a system like this:

1. **Make the agent a member, not a backend.** Once it posts through a
   tool, can stay silent, can read other rooms and can schedule itself, a
   lot of features follow without extra work.
2. **One conversation, one turn at a time, with an inbox.** It fixes the
   problem of several people writing at once.
3. **Threads are the unit of work.** One topic, one thread, one actor, one
   context.
4. **Store everything and choose views of it.** A plain transcript plus
   context modes you can switch was better than any single memory design.
   Summaries plus a pointer to the source beat both "send everything" and
   hidden retrieval.
5. **Plan for loops from the start.** Bot messages start no runs,
   schedules have generations, handoffs go one way.
6. **Ask people with the client's own features.** Polls, mentions,
   reactions. Do not put a question at the end of a long post.
7. **Isolate every run.** Memory limits, kill leftovers, a concurrency
   limit, circuit breakers. Agents will try `cargo build` on everything at
   once.
