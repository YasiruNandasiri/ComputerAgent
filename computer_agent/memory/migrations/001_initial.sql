-- Initial schema for Computer Agent memory layer
-- Run via: docker-compose up (auto-applied) or psql directly

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -------------------------------------------------------------------
-- Execution Traces
-- Stores full execution logs for every completed task.
-- Indexed by embedding for similarity-based trace retrieval.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS execution_traces (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      TEXT NOT NULL,
    skill_name      TEXT,                          -- Skill used, or NULL for ad-hoc
    goal            TEXT NOT NULL,                 -- Natural language description of the task
    steps_json      JSONB NOT NULL DEFAULT '[]',   -- Array of executed steps with results
    result          TEXT,                          -- Final outcome summary
    success         BOOLEAN NOT NULL DEFAULT FALSE,
    duration_ms     INTEGER,
    token_count     INTEGER,
    embedding       VECTOR(1536),                  -- Embedded from `goal` text
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON execution_traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_skill ON execution_traces(skill_name);
CREATE INDEX IF NOT EXISTS idx_traces_success ON execution_traces(success);
CREATE INDEX IF NOT EXISTS idx_traces_created ON execution_traces(created_at DESC);
-- pgvector IVFFlat index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_traces_embedding ON execution_traces
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- -------------------------------------------------------------------
-- User Preferences
-- Key/value store for learned user preferences and settings.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_preferences (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key         TEXT NOT NULL UNIQUE,
    value       JSONB NOT NULL,
    context     TEXT,               -- Free-form description of when this preference applies
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------------
-- Learned Shortcuts
-- Maps task trigger patterns to optimized execution steps.
-- Agent learns faster paths over time.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learned_shortcuts (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trigger_pattern     TEXT NOT NULL,
    optimized_steps     JSONB NOT NULL,
    confidence          FLOAT NOT NULL DEFAULT 0.5,
    use_count           INTEGER NOT NULL DEFAULT 0,
    last_used_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------------
-- HITL State Snapshots
-- Serialized agent state saved at every approval checkpoint.
-- Allows the agent to resume after human review.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS state_snapshots (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          TEXT NOT NULL,
    checkpoint_id       TEXT NOT NULL UNIQUE,
    serialized_state    JSONB NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | denied | expired
    proposed_action     TEXT,
    risk_reason         TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ,
    resolved_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_snapshots_session ON state_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_status ON state_snapshots(status);
CREATE INDEX IF NOT EXISTS idx_snapshots_checkpoint ON state_snapshots(checkpoint_id);

-- -------------------------------------------------------------------
-- Skill Cache
-- Tracks skill performance metrics for auto-optimization.
-- -------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_cache (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    skill_name      TEXT NOT NULL UNIQUE,
    version         TEXT NOT NULL DEFAULT '0.0.0',
    last_used_at    TIMESTAMPTZ,
    avg_duration_ms FLOAT,
    success_rate    FLOAT DEFAULT 1.0,
    use_count       INTEGER NOT NULL DEFAULT 0,
    parameter_defaults JSONB DEFAULT '{}'
);
