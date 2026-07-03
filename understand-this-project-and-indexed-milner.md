# ComputerAgent — Gap Analysis & Plan to Meet Requirements

## Context

The user's requirement is a personal autonomous agent that runs on its own on the computer, uses the computer like a human (GUI, browser, files, shell), handles routine tasks (checking emails, news feeds, important messages) on its own schedule, notifies the user of important items, and pauses for user approval on consequential decisions. It must support **autonomy modes** (low / medium / high), **work with any LLM**, support **pluggable skills/tools** (like Claude Code skills), and let the user **interact while tasks run** — ask what it's focusing on, get progress, change priority, hold/resume, or terminate any task (including the one currently executing).

The existing project (`computer_agent/`, ~5,160 LOC, custom framework — **not** Google ADK) already delivers a large portion of this. This plan identifies what's done, what's missing, and how to close the gaps.

## What already exists (verified working)

| Requirement | Status | Where |
|---|---|---|
| Works with any LLM | ✅ Done | `computer_agent/llm/` — registry pattern-matches model name → Anthropic / OpenAI / Google Gemini / LiteLLM (Ollama, Bedrock, 100+); lazy SDK loading; runtime-registerable providers |
| Pluggable skills (install later) | ✅ Done | `computer_agent/skills/loader.py` — drop a `skill.yaml` into `skills/` and it auto-loads as a macro tool; external plugins via Python entry-points (`computer_agent.tools/.skills/.abilities`) |
| Use the computer like a human | ✅ Done | 40 tools: screen/mouse/keyboard (pyautogui + OCR), browser (Playwright, 11 tools), filesystem, shell, HTTP — all risk-tagged and gated |
| Permission gates on risky actions | ✅ Done (binary) | `abilities/` rule engine (ALLOW / REQUIRE_HITL / BLOCK) + `hitl/` checkpoint state machine with macOS-notification approval UI |
| Memory / preferences | ✅ Done | `memory/store.py` — Postgres + pgvector: execution traces w/ semantic search, HITL snapshots, `user_preferences` table |
| User interaction | ⚠️ Partial | `main.py` CLI: single-shot `run`, blocking `chat` REPL, `approve`/`deny` |
| Event/progress infrastructure | ⚠️ Foundation only | `runtime/event_bus.py` — 23 event types emitted, but nothing subscribes/persists them |

## The gaps (requirement → what's missing)

1. **Autonomy modes (low/medium/high)** — ❌ No `autonomy_level` concept anywhere. The abilities engine is static: the same rules always apply regardless of trust level. Needs a mode dimension that maps tool `RiskLevel` × mode → auto-allow / notify-after / require-approval, switchable at runtime and persisted.
2. **Runs on its own (daemon + schedules)** — ❌ No long-lived process, no scheduler, no task queue, no background worker. Today the agent only runs while a CLI command is in the foreground. "Check email every 30 min and notify me" is impossible.
3. **Notifications** — ⚠️ Only HITL approvals trigger a macOS notification; no general "important email arrived" / "task finished" notifier.
4. **Task management (progress / priority / hold / terminate)** — ❌ No task records, no priority, no pause/resume, no cancellation. The coordinator loop has no cancellation checks; a running task cannot be stopped except Ctrl-C of the whole process.
5. **Interact while tasks run** — ❌ `chat` blocks during execution; you cannot ask "what are you working on?" mid-task. Also, `approve`/`deny` run in a *separate process* from the waiting agent — the in-memory `asyncio` wait can't be resolved cross-process (latent bug; a daemon architecture fixes this properly).
6. **In-band approvals** — ❌ Approvals only via `computer-agent approve <id>`, not by replying "yes, send it" in chat.

## Google ADK question (user asked for a recommendation)

**Recommendation: keep the custom framework; do not migrate to Google ADK.**
- The spec's hard requirements — *any LLM* and *pluggable skills* — are already fully met here; ADK is Gemini-first (other providers only via LiteLLM wrappers) and migration would discard the working abilities/HITL engine, tool registry, and memory layer for no functional gain.
- The remaining gaps (scheduler, task manager, daemon, autonomy modes) are things ADK does not provide out of the box either — they'd have to be built regardless.

## Bug found during analysis (worth knowing even if nothing else is built)

`computer-agent approve <id>` / `deny <id>` **cannot work today**: they start a fresh process whose `HITLManager._states` dict is empty ([hitl/checkpoint.py:175](computer_agent/hitl/checkpoint.py:175)), so `resolve()` always returns "not found" while the agent process blocks on its in-memory `asyncio.Event` until the 300s timeout expires the checkpoint. Approvals only work in-process. The daemon architecture below fixes this structurally.

## Implementation plan

**Target architecture:** one long-lived daemon process (`computer-agent daemon`) hosting FastAPI on `127.0.0.1:8765`, a TaskManager (priority queue + worker), APScheduler, the HITLManager, a Notifier, and an event-bus→Postgres persistence subscriber. The CLI becomes a thin HTTP client. New deps: `fastapi`, `uvicorn`, `apscheduler` (no Redis/Celery — Postgres-backed queue is enough for a single-user machine).

### Phase 1 — Autonomy modes (low/medium/high)
- **New** `computer_agent/abilities/autonomy.py`: `AutonomyLevel` enum + `AutonomyManager` singleton — loads/persists the level via the existing `user_preferences` table (`memory_store.save_preference()`), runtime-switchable. Decision matrix combining tool `RiskLevel` × mode:

  | Tool risk | LOW mode | MEDIUM mode | HIGH mode |
  |---|---|---|---|
  | low | allow | allow | allow |
  | medium | ask | allow + report | allow |
  | high | ask | ask | allow unless rule `always_hitl` |
  | critical | ask | ask | ask |

  Rule `BLOCK` always wins; `always_hitl` rules never downgrade.
- **Modify** [abilities/engine.py](computer_agent/abilities/engine.py): add `always_hitl` field to `AbilityRule`; `evaluate()` gains optional `risk_level` param and applies the matrix.
- **Modify** [abilities/rules/default.yaml](computer_agent/abilities/rules/default.yaml): mark `credential_block`, `delete_file_guard`, `send_email_guard` as `always_hitl: true`.
- **Modify** [coordinator.py](computer_agent/coordinator.py) `_execute_tool_calls`: pass the tool's `risk_level` into `abilities_engine.evaluate()`.
- **Modify** [config.py](computer_agent/config.py): `AUTONOMY_LEVEL` (default `medium`); [main.py](computer_agent/main.py): `computer-agent mode [low|medium|high]` command.

### Phase 2 — Daemon + HTTP API (fixes the HITL bug)
- **New** `computer_agent/daemon/` package: `app.py` (FastAPI app factory; lifespan bootstraps registries, memory store, autonomy, worker, scheduler), `api.py` (routes: `POST /chat`, `POST/GET /tasks`, `POST /tasks/{id}/pause|resume|cancel`, `PATCH /tasks/{id}` priority, `GET /hitl/pending`, `POST /hitl/{id}/resolve`, `GET/PUT /mode`, `GET /status`, `GET /events` SSE), `client.py` (`DaemonClient` over httpx — already a dependency).
- **Modify** [main.py](computer_agent/main.py): new `daemon` command; `approve`/`deny`/`pending` route through `DaemonClient` (clear error if daemon down instead of the silently-broken path).
- **Modify** [hitl/checkpoint.py](computer_agent/hitl/checkpoint.py): emit `HITL_APPROVAL_REQUESTED`/`EXPIRED` events (currently never emitted) and call `memory_store.update_snapshot_status()` on resolution (currently never called).

### Phase 3 — TaskManager: queue, priority, pause/hold, terminate, progress
- **New** `computer_agent/taskmgr/`: `models.py` (`TaskStatus`: queued/running/paused/awaiting_approval/completed/failed/cancelled; `TaskRecord` with priority + source user/schedule/chat), `control.py` (`TaskControl` — pause via cleared `asyncio.Event`, cancel flag, `await checkpoint()` raises `TaskCancelledError` or blocks while paused), `manager.py` (`TaskManager` singleton: `submit()` persists + pushes to priority heap; worker loop pops highest priority → runs a `Coordinator`; concurrency 1 — mouse/keyboard can't be shared; `pause/resume/cancel/set_priority/list/status`; hard `asyncio.Task.cancel()` fallback after grace period).
- **Modify** [coordinator.py](computer_agent/coordinator.py): accept `task_id`/`task_control`; call `await task_control.checkpoint()` at each turn and before each tool call (this is what makes hold/terminate of the *currently running* task possible); emit per-turn `STEP_COMPLETED` progress events.
- **New** `computer_agent/runtime/progress.py`: event-bus subscriber persisting `TASK_*`/`STEP_*`/`HITL_*` events → `task_events` table and updating task status.
- **New migration** `computer_agent/memory/migrations/002_tasks.sql`: `agent_tasks`, `task_events`, `scheduled_routines` tables.
- **Modify** [runtime/event_bus.py](computer_agent/runtime/event_bus.py): add `TASK_PAUSED/RESUMED/PROGRESS` event types; [main.py](computer_agent/main.py): `task submit|list|show|pause|resume|cancel|priority` sub-commands.

### Phase 4 — Scheduler + notifications (autonomous routine work)
- **New** `computer_agent/scheduler/service.py`: APScheduler `AsyncIOScheduler` loading routines from `scheduled_routines`; each trigger → `task_manager.submit(goal, source="schedule")`. CLI: `routine add --cron "*/30 * * * *" "Check my email for urgent messages"`, `routine list|remove|enable|disable`.
- **New** `computer_agent/notify/notifier.py`: channel abstraction (`MacOSChannel` extracted from the existing osascript code in [hitl/approval_ui.py](computer_agent/hitl/approval_ui.py), `TerminalChannel`; entry-point group `computer_agent.notifiers` for future Slack/email). Daemon subscribes it to `TASK_COMPLETED/FAILED` (the "report back" behavior) and `HITL_APPROVAL_REQUESTED`.
- **New** `computer_agent/tools/agent_control.py`: `@tool notify_user(title, message)` so routine tasks can proactively surface important items ("urgent email from X").
- **Config**: `hitl_timeout_action: "pause"` — background tasks awaiting approval pause instead of auto-expiring at 300s.

### Phase 5 — Interactive chat while tasks run + in-band approvals
- **New** `computer_agent/daemon/chat.py`: `ChatSession` — fast-path regex intents (bare "approve"/"deny", "set mode high", "pause/cancel task", "what are you working on") handled without an LLM call; everything else goes to a chat `Coordinator` that has agent-control tools registered (`list_tasks`, `pause_task`, `cancel_task`, `set_task_priority`, `start_background_task`, `set_autonomy_mode`, `approve/deny_pending_action`) so natural phrasing works too.
- **Modify** [main.py](computer_agent/main.py) `_chat_session`: rewrite as daemon client — input loop POSTing `/chat` plus concurrent SSE listener printing live progress and inline HITL prompts ("Task #3 wants to run `delete_file(...)` — reply 'approve' or 'deny'"). Keep in-process REPL as no-daemon fallback.
- **Modify** coordinator system prompt: inject current autonomy mode + control-tool description per run.

## Verification

- **Phase 1**: extend `tests/test_abilities.py` with the (risk × mode) matrix parametrized; assert BLOCK/`always_hitl` never downgrade. Manual: `mode high` → `write_file` runs unprompted; `mode low` → prompts; restart → mode persisted.
- **Phase 2**: `tests/test_daemon_api.py` with FastAPI `TestClient`. End-to-end: daemon in one terminal, `delete_file` task, `approve <id>` from another terminal → executes (impossible today).
- **Phase 3**: unit tests for `TaskControl.checkpoint()` (cancel raises, pause blocks) and priority ordering with a stubbed Coordinator. Manual: pause a long-running task mid-flight, resume, cancel; check `agent_tasks`/`task_events` rows.
- **Phase 4**: a `* * * * *` routine enqueues every minute and fires a macOS notification on completion; survives daemon restart.
- **Phase 5**: chat while a background task runs; ask "what are you working on"; reply "approve" to an inline HITL prompt; "set mode high" changes gating live.

## Build order
1 → 2 → 3 → 4 → 5. Each phase ships independently useful behavior; Phase 2 alone fixes the broken approve/deny flow.
