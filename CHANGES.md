# Computer Agent — Autonomy & Background Operation Upgrade

*Date: 2026-07-03*

This document describes the gap analysis performed against the product
requirements and the changes implemented to close them.

---

## 1. Background: the requirement

A personal autonomous agent that:

- runs on its own on the computer and uses it like a human (GUI, browser, files, shell)
- handles routine tasks on a schedule (check email, news feeds, important messages) and **notifies** the user of important items
- **waits for the user's approval** on important/consequential actions
- supports **autonomy modes** — low / medium / high — controlling how independently it acts
- works with **any LLM** and supports **pluggable skills/tools** installed later
- lets the user **interact while tasks run**: ask what it's focusing on, get progress, change priority, hold/resume, or terminate any task (including the currently running one)

## 2. What already existed (unchanged)

| Capability | Where |
|---|---|
| Multi-LLM support (Anthropic, OpenAI, Gemini, LiteLLM/Ollama/Bedrock/100+) | `computer_agent/llm/` |
| Pluggable skills (`skill.yaml` drop-in) and tools (entry-points) | `computer_agent/skills/`, `computer_agent/tools/` |
| Computer use: 40+ tools — screen/mouse/keyboard, Playwright browser, filesystem, shell, HTTP | `computer_agent/tools/` |
| Rule-based safety gates (ALLOW / REQUIRE_HITL / BLOCK) | `computer_agent/abilities/` |
| HITL checkpoint state machine | `computer_agent/hitl/` |
| Semantic memory (PostgreSQL + pgvector) and user preferences | `computer_agent/memory/` |
| Event bus (23+ lifecycle event types) | `computer_agent/runtime/event_bus.py` |

## 3. Gaps found

1. **No autonomy modes** — the same static rules always applied; no trust dial.
2. **No autonomous operation** — no long-lived process, scheduler, task queue, or worker; the agent only ran while a CLI command was in the foreground.
3. **No general notifications** — only HITL approvals notified.
4. **No task management** — no progress querying, priorities, pause/hold, or termination; a running task could only be stopped with Ctrl-C.
5. **No interaction while running** — the chat REPL blocked during execution.
6. **Bug (confirmed):** `computer-agent approve/deny` could never work — it ran in a separate process from the waiting agent, whose in-memory `asyncio.Event` was unreachable, so every approval silently expired after 300 s.

A migration to Google ADK was considered and rejected: the framework already
meets the any-LLM and pluggable-skills requirements natively, and ADK provides
none of the missing pieces above.

## 4. What was built

### 4.1 Autonomy modes (`low` / `medium` / `high`)

- **New** `computer_agent/abilities/autonomy.py` — `AutonomyLevel`, `AutonomyManager`
  (persisted in the `user_preferences` table, switchable at runtime), and the
  decision matrix combining each tool's declared `RiskLevel` with the mode:

  | Tool risk | LOW mode | MEDIUM mode | HIGH mode |
  |-----------|----------|-------------|-----------|
  | low       | allow    | allow       | allow |
  | medium    | ask      | allow (+ report) | allow |
  | high      | ask      | ask         | allow unless `always_hitl` |
  | critical  | ask      | ask         | ask |

- `AbilityRule` gained `always_hitl: true` — rules that never auto-approve even
  in HIGH mode (file deletion, sending email/messages, credential access).
  `BLOCK` rules (financial operations) always win.
- The Coordinator now passes each tool's risk level into the gate, and its
  system prompt tells the LLM which mode is active and how to behave in it.
- CLI: `computer-agent mode [low|medium|high]` (persists across restarts).

### 4.2 Daemon — the agent as a long-lived process

- **New** `computer_agent/daemon/` — FastAPI app (`127.0.0.1:8765` by default)
  hosting the task worker, scheduler, HITL manager, notifier, and progress
  tracker in **one process**. Endpoints: `/chat`, `/tasks` (+ pause/resume/
  cancel/priority), `/hitl/pending`, `/hitl/{id}/resolve`, `/mode`, `/status`,
  `/routines`, `/events` (live SSE stream).
- `computer-agent daemon` starts it; the CLI (`chat`, `task`, `routine`,
  `approve`, `deny`, `pending`, `mode`) becomes a thin HTTP client.
- **Fixes the approval bug structurally**: approve/deny now resolves the real
  in-memory checkpoint in the daemon process. HITL resolutions are also
  persisted, and approval-requested/expired events are emitted.
- New config: `DAEMON_HOST`, `DAEMON_PORT`, `HITL_TIMEOUT_ACTION`
  (`pause` = wait indefinitely for the user, default; `expire` = legacy auto-deny).

### 4.3 Task management — queue, priority, hold, terminate, progress

- **New** `computer_agent/taskmgr/` — `TaskRecord` (goal, status, priority,
  source, progress log, timestamps), `TaskControl` (cooperative pause/resume/
  cancel), and `TaskManager` (priority queue + background worker; concurrency 1
  by default since mouse/keyboard can't be shared; hard-cancel fallback after a
  10 s grace period).
- The Coordinator checks pause/cancel signals **between every turn and before
  every tool call**, so even the currently running task can be held or
  terminated at a safe point mid-execution.
- Per-turn progress events are recorded to each task and persisted
  (`agent_tasks`, `task_events` tables — migration
  `computer_agent/memory/migrations/002_tasks.sql`).
- CLI: `task submit | list | show | status | pause | resume | cancel | priority`.
- `run --background/-b` queues a one-off task on the daemon.

### 4.4 Scheduler + notifications — autonomous routine work

- **New** `computer_agent/scheduler/` — cron routines (APScheduler) stored in
  the `scheduled_routines` table, loaded at daemon start; each trigger submits
  a normal task (same HITL gating, progress, notifications).
  CLI: `routine add "Check my email for urgent messages" -c "*/30 * * * *"`,
  plus `list | remove | enable | disable`.
- **New** `computer_agent/notify/` — `Notifier` with pluggable channels
  (macOS notification, terminal; external channels via the
  `computer_agent.notifiers` entry-point group). The daemon notifies on
  background-task completion/failure and approval requests.
- **New tool** `notify_user(title, message)` — lets routine tasks proactively
  surface important findings ("urgent email from X").

### 4.5 Interactive chat while tasks run + in-band approvals

- **New** `computer_agent/daemon/chat.py` — persistent chat sessions with:
  - **fast-path intents** (no LLM round-trip): `approve` / `deny` (resolves the
    newest pending checkpoint), `mode <level>`, `status` / "what are you
    working on?"
  - **13 agent-control tools** for natural phrasing: `get_task_status`,
    `list_tasks`, `get_task`, `start_background_task`, `pause_task`,
    `resume_task`, `cancel_task`, `set_task_priority`, `set_autonomy_mode`,
    `list_pending_approvals`, `approve_pending_action`, `deny_pending_action`,
    `notify_user`.
- `computer-agent chat` now streams live daemon events while you type:
  approval prompts appear inline ("reply 'approve' or 'deny'"), background
  task progress/completions/failures show as status lines. Falls back to the
  old in-process REPL when the daemon isn't running.

### 4.6 Other fixes

- `set_level()` crashed on Python 3.12 when passed an enum (str-enum `str()`
  behavior change) — caught by the smoke test, fixed.
- HITL snapshot statuses are now actually updated on resolution
  (`update_snapshot_status` was previously dead code).

## 5. New system capabilities (summary)

The agent can now:

1. **Run autonomously** as a daemon, executing scheduled routines
   (email checks, news feeds, monitoring) without any terminal in the foreground.
2. **Notify the user** of important findings, completed/failed background
   tasks, and actions awaiting permission.
3. **Respect a trust dial**: `low` asks for everything with side effects,
   `medium` self-handles simple reversible work and reports back, `high`
   self-handles almost everything — while irreversible/sensitive actions
   (delete, send email/message, credentials) *always* ask, and financial
   actions are *always* blocked.
4. **Be interrupted and directed at any time**: ask what it's focusing on,
   see step-by-step progress, reprioritize the queue, hold/resume, or
   terminate any task — including the one currently executing.
5. **Take approvals in-band**: reply "approve"/"deny" in chat, use
   `computer-agent approve <id>`, or the HTTP API — all of which now work.
6. Everything remains **LLM-agnostic** (switch via `PRIMARY_MODEL`) and
   **pluggable** (skills, tools, ability rules, and now notification channels
   via entry points).

## 6. Files added / changed

**New packages/files**

```
computer_agent/abilities/autonomy.py       autonomy levels + decision matrix
computer_agent/taskmgr/{models,control,manager}.py
computer_agent/daemon/{app,api,client,chat}.py
computer_agent/scheduler/service.py
computer_agent/notify/notifier.py
computer_agent/runtime/progress.py         event → task progress persistence
computer_agent/tools/agent_control.py      13 self-management tools
computer_agent/memory/migrations/002_tasks.sql
```

**Modified**

```
computer_agent/abilities/engine.py         autonomy-aware evaluate(), always_hitl
computer_agent/abilities/rules/default.yaml
computer_agent/coordinator.py              task control checkpoints, progress
                                           events, mode-aware system prompt
computer_agent/hitl/checkpoint.py          events, pause-instead-of-expire,
                                           resolution persistence
computer_agent/hitl/approval_ui.py         delegates to Notifier
computer_agent/memory/store.py             task/routine persistence
computer_agent/runtime/event_bus.py        TASK_PAUSED/RESUMED/PROGRESS
computer_agent/main.py                     daemon/mode/task/routine commands,
                                           daemon-backed chat + approvals
computer_agent/config.py, .env.example     new settings
pyproject.toml                             + fastapi, uvicorn, apscheduler
tests/test_abilities.py                    + autonomy matrix tests (25 pass)
```

**Migration note:** the new tables only auto-apply on a fresh Postgres volume.
For an existing database run:

```bash
docker exec -i $(docker ps -qf name=postgres) \
  psql -U agent -d computer_agent < computer_agent/memory/migrations/002_tasks.sql
```

## 7. Verification status

- 25/25 abilities tests pass (including the 12-case autonomy matrix).
- Smoke-tested against a live app instance (stubbed LLM): all API endpoints,
  full task lifecycle (submit → running → pause → resume → priority → cancel,
  including cancelling a *running* task), and chat fast-paths.
- Full-suite, DB-backed, and end-to-end manual testing: see the testing guide
  (unit → DB migration → approval flow → autonomy modes → interactive control
  → routines → regression).
