"""
Memory Store — async PostgreSQL + pgvector client.

Provides:
  - save_trace(trace) — persist an execution trace with embedding
  - search_similar_traces(goal, top_k) — semantic search over past traces
  - save_snapshot / get_snapshot — HITL state checkpoint persistence
  - save_preference / get_preference — user preferences
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from computer_agent.config import settings
from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


class MemoryStore:
    """Async PostgreSQL + pgvector memory store."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or settings.database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            return
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            # Register pgvector codec
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("memory_store_connected", dsn=self._dsn.split("@")[-1])
        except Exception as e:
            logger.warning(
                "memory_store_unavailable",
                error=str(e),
                hint="Start PostgreSQL with: docker-compose up -d",
            )
            self._pool = None

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _available(self) -> bool:
        return self._pool is not None

    # ------------------------------------------------------------------
    # Execution Traces
    # ------------------------------------------------------------------

    async def save_trace(
        self,
        session_id: str,
        goal: str,
        steps: list[dict[str, Any]],
        result: str,
        success: bool,
        duration_ms: int | None = None,
        token_count: int | None = None,
        skill_name: str | None = None,
        embedding: list[float] | None = None,
    ) -> str | None:
        """Save an execution trace. Returns the trace ID, or None if DB unavailable."""
        if not self._available():
            return None

        trace_id = str(uuid.uuid4())
        embedding_str = f"[{','.join(str(v) for v in embedding)}]" if embedding else None

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO execution_traces
                    (id, session_id, skill_name, goal, steps_json, result,
                     success, duration_ms, token_count, embedding)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9,
                        $10::vector)
                """,
                trace_id,
                session_id,
                skill_name,
                goal,
                json.dumps(steps),
                result,
                success,
                duration_ms,
                token_count,
                embedding_str,
            )
        logger.debug("trace_saved", trace_id=trace_id, success=success)
        return trace_id

    async def search_similar_traces(
        self,
        embedding: list[float],
        top_k: int | None = None,
        min_similarity: float | None = None,
        success_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most similar execution traces using cosine similarity.
        Returns a list of trace dicts sorted by similarity (descending).
        """
        if not self._available() or not embedding:
            return []

        k = top_k or settings.memory_top_k
        threshold = min_similarity or settings.memory_similarity_threshold
        embedding_str = f"[{','.join(str(v) for v in embedding)}]"

        success_filter = "AND success = TRUE" if success_only else ""

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    id, session_id, skill_name, goal, steps_json, result,
                    success, duration_ms, created_at,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM execution_traces
                WHERE embedding IS NOT NULL
                  {success_filter}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                embedding_str,
                k * 2,  # Fetch extra, filter by threshold below
            )

        results = []
        for row in rows:
            similarity = float(row["similarity"])
            if similarity >= threshold:
                results.append({
                    "id": str(row["id"]),
                    "session_id": row["session_id"],
                    "skill_name": row["skill_name"],
                    "goal": row["goal"],
                    "steps": json.loads(row["steps_json"]),
                    "result": row["result"],
                    "success": row["success"],
                    "duration_ms": row["duration_ms"],
                    "created_at": row["created_at"].isoformat(),
                    "similarity": similarity,
                })
            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------
    # HITL State Snapshots
    # ------------------------------------------------------------------

    async def save_snapshot(
        self,
        session_id: str,
        checkpoint_id: str,
        serialized_state: dict[str, Any],
        proposed_action: str,
        risk_reason: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Persist agent state at a HITL checkpoint."""
        if not self._available():
            return

        ttl = ttl_seconds or settings.hitl_approval_timeout
        expires_at = datetime.now(UTC) + timedelta(seconds=ttl)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO state_snapshots
                    (session_id, checkpoint_id, serialized_state, proposed_action,
                     risk_reason, expires_at)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6)
                ON CONFLICT (checkpoint_id) DO UPDATE
                    SET serialized_state = EXCLUDED.serialized_state,
                        status = 'pending',
                        expires_at = EXCLUDED.expires_at
                """,
                session_id,
                checkpoint_id,
                json.dumps(serialized_state),
                proposed_action,
                risk_reason,
                expires_at,
            )
        logger.info("snapshot_saved", checkpoint_id=checkpoint_id, session=session_id)

    async def get_snapshot(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Retrieve a state snapshot by checkpoint ID."""
        if not self._available():
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM state_snapshots WHERE checkpoint_id = $1",
                checkpoint_id,
            )
        if not row:
            return None

        return {
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "checkpoint_id": row["checkpoint_id"],
            "state": json.loads(row["serialized_state"]),
            "status": row["status"],
            "proposed_action": row["proposed_action"],
            "risk_reason": row["risk_reason"],
            "created_at": row["created_at"].isoformat(),
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
        }

    async def update_snapshot_status(self, checkpoint_id: str, status: str) -> None:
        """Update snapshot status (approved | denied | expired)."""
        if not self._available():
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE state_snapshots
                SET status = $1, resolved_at = NOW()
                WHERE checkpoint_id = $2
                """,
                status,
                checkpoint_id,
            )

    # ------------------------------------------------------------------
    # Agent Tasks
    # ------------------------------------------------------------------

    async def save_task(self, task: dict[str, Any]) -> None:
        """Upsert a task record (dict form of TaskRecord)."""
        if not self._available():
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_tasks
                    (id, goal, status, priority, source, session_id, schedule_id,
                     result, error, progress, created_at, started_at, finished_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb,
                        $11::timestamptz, $12::timestamptz, $13::timestamptz)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    priority = EXCLUDED.priority,
                    result = EXCLUDED.result,
                    error = EXCLUDED.error,
                    progress = EXCLUDED.progress,
                    started_at = EXCLUDED.started_at,
                    finished_at = EXCLUDED.finished_at
                """,
                task["id"],
                task["goal"],
                task["status"],
                task["priority"],
                task["source"],
                task["session_id"],
                task.get("schedule_id"),
                task.get("result"),
                task.get("error"),
                json.dumps(task.get("progress", [])),
                task["created_at"],
                task.get("started_at"),
                task.get("finished_at"),
            )

    async def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List persisted task records, newest first."""
        if not self._available():
            return []

        status_filter = "WHERE status = $2" if status else ""
        args: list[Any] = [limit] + ([status] if status else [])
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM agent_tasks {status_filter}
                ORDER BY created_at DESC LIMIT $1
                """,
                *args,
            )
        return [dict(row) for row in rows]

    async def append_task_event(
        self, task_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Append an event to the task's event log."""
        if not self._available():
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO task_events (task_id, event_type, data)
                VALUES ($1, $2, $3::jsonb)
                """,
                task_id,
                event_type,
                json.dumps(data, default=str),
            )

    # ------------------------------------------------------------------
    # Scheduled Routines
    # ------------------------------------------------------------------

    async def save_routine(self, routine: dict[str, Any]) -> None:
        """Upsert a scheduled routine by name."""
        if not self._available():
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO scheduled_routines (id, name, cron, goal, priority, enabled, notify)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (name) DO UPDATE SET
                    cron = EXCLUDED.cron,
                    goal = EXCLUDED.goal,
                    priority = EXCLUDED.priority,
                    enabled = EXCLUDED.enabled,
                    notify = EXCLUDED.notify
                """,
                routine["id"],
                routine["name"],
                routine["cron"],
                routine["goal"],
                routine.get("priority", 5),
                routine.get("enabled", True),
                routine.get("notify", True),
            )

    async def list_routines(self) -> list[dict[str, Any]]:
        if not self._available():
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM scheduled_routines ORDER BY name")
        return [dict(row) for row in rows]

    async def delete_routine(self, name: str) -> bool:
        if not self._available():
            return False

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM scheduled_routines WHERE name = $1", name
            )
        return result.endswith("1")

    async def mark_routine_run(self, name: str) -> None:
        if not self._available():
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE scheduled_routines SET last_run_at = NOW() WHERE name = $1", name
            )

    # ------------------------------------------------------------------
    # User Preferences
    # ------------------------------------------------------------------

    async def save_preference(self, key: str, value: Any, context: str = "") -> None:
        """Upsert a user preference."""
        if not self._available():
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_preferences (key, value, context, updated_at)
                VALUES ($1, $2::jsonb, $3, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        context = EXCLUDED.context,
                        updated_at = NOW()
                """,
                key,
                json.dumps(value),
                context,
            )

    async def get_preference(self, key: str) -> Any:
        """Retrieve a user preference value."""
        if not self._available():
            return None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM user_preferences WHERE key = $1", key
            )
        return json.loads(row["value"]) if row else None

    async def get_all_preferences(self) -> dict[str, Any]:
        """Retrieve all user preferences as a dict."""
        if not self._available():
            return {}

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT key, value FROM user_preferences")
        return {row["key"]: json.loads(row["value"]) for row in rows}


# Module-level singleton
memory_store = MemoryStore()
