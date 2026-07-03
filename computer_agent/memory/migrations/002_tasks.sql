-- Task manager, task event log, and scheduled routines
-- Run via: docker-compose up (auto-applied on first start) or psql directly

-- -------------------------------------------------------------------
-- Agent Tasks
-- Every unit of work submitted to the TaskManager (user, chat, schedule).
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_tasks (
    id              UUID PRIMARY KEY,
    goal            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    priority        INTEGER NOT NULL DEFAULT 5,
    source          TEXT NOT NULL DEFAULT 'user',      -- user | schedule | chat
    session_id      TEXT NOT NULL,
    schedule_id     UUID,
    result          TEXT,
    error           TEXT,
    progress        JSONB NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON agent_tasks(priority DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON agent_tasks(created_at DESC);

-- -------------------------------------------------------------------
-- Task Events
-- Append-only log of lifecycle/progress events per task (from the event bus).
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS task_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id         UUID NOT NULL,
    event_type      TEXT NOT NULL,
    data            JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id, created_at);

-- -------------------------------------------------------------------
-- Scheduled Routines
-- Recurring background jobs (check email, news feeds, ...) run by the daemon.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scheduled_routines (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT UNIQUE NOT NULL,
    cron            TEXT NOT NULL,                     -- standard 5-field cron expression
    goal            TEXT NOT NULL,                     -- natural-language task for the agent
    priority        INTEGER NOT NULL DEFAULT 5,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    notify          BOOLEAN NOT NULL DEFAULT TRUE,     -- notify user on completion
    last_run_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
