"""
TaskManager — priority queue + background worker executing Coordinator runs.

The manager is the single authority over task lifecycle:

    queued → running → completed | failed | cancelled
              ↕ paused / awaiting_approval

Records are kept in memory (source of truth for the daemon's lifetime) and
persisted best-effort to PostgreSQL so history survives restarts. Pause and
cancel are cooperative: the Coordinator awaits TaskControl.checkpoint()
between turns and before every tool call. A hard asyncio cancel is used as a
fallback when a cancelled task doesn't stop within the grace period.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
from typing import Any

from computer_agent.config import settings
from computer_agent.logging_setup import get_logger
from computer_agent.runtime.event_bus import Event, EventType, event_bus
from computer_agent.taskmgr.control import TaskCancelledError, TaskControl
from computer_agent.taskmgr.models import TaskRecord, TaskStatus

logger = get_logger(__name__)

_CANCEL_GRACE_SECONDS = 10.0


class TaskManager:
    """Singleton owning the task queue and the background worker."""

    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._controls: dict[str, TaskControl] = {}
        # heap entries: (-priority, seq, task_id, priority_at_push)
        self._heap: list[tuple[int, int, str, int]] = []
        self._seq = itertools.count()
        self._wakeup = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    async def start(self) -> None:
        """Start the background worker (called from the daemon lifespan)."""
        if self.is_running:
            return
        self._worker = asyncio.create_task(self._worker_loop(), name="taskmgr-worker")
        logger.info("task_worker_started")

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        for t in list(self._running_tasks.values()):
            t.cancel()
        logger.info("task_worker_stopped")

    # ------------------------------------------------------------------
    # Submission & lookup
    # ------------------------------------------------------------------

    async def submit(
        self,
        goal: str,
        priority: int = 5,
        source: str = "user",
        schedule_id: str | None = None,
    ) -> TaskRecord:
        record = TaskRecord(goal=goal, priority=priority, source=source, schedule_id=schedule_id)
        self._records[record.id] = record
        self._controls[record.id] = TaskControl()
        self._push(record)
        await self._persist(record)
        self._wakeup.set()
        logger.info("task_submitted", task_id=record.id, goal=goal[:80], priority=priority)
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        record = self._records.get(task_id)
        if record:
            return record
        # Allow prefix matching (CLI convenience for short ids)
        matches = [r for tid, r in self._records.items() if tid.startswith(task_id)]
        return matches[0] if len(matches) == 1 else None

    def list(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        records = sorted(
            self._records.values(),
            key=lambda r: (r.status.value, -r.priority, r.created_at),
        )
        if status:
            records = [r for r in records if r.status == status]
        return records

    def status_summary(self) -> dict[str, Any]:
        """What is the agent focusing on right now?"""
        running = [r for r in self._records.values() if r.status == TaskStatus.RUNNING]
        awaiting = [r for r in self._records.values() if r.status == TaskStatus.AWAITING_APPROVAL]
        paused = [r for r in self._records.values() if r.status == TaskStatus.PAUSED]
        queued = [r for r in self._records.values() if r.status == TaskStatus.QUEUED]
        return {
            "current": running[0].summary() if running else None,
            "awaiting_approval": [r.summary() for r in awaiting],
            "paused": [r.summary() for r in paused],
            "queued": [r.summary() for r in sorted(queued, key=lambda r: -r.priority)],
        }

    # ------------------------------------------------------------------
    # User control
    # ------------------------------------------------------------------

    async def pause(self, task_id: str) -> TaskRecord | None:
        record = self.get(task_id)
        if not record or record.status.is_terminal():
            return None
        control = self._controls.get(record.id)
        if control:
            control.pause()
        if record.status in (TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.AWAITING_APPROVAL):
            record.status = TaskStatus.PAUSED
        await self._persist(record)
        await event_bus.emit(Event(
            type=EventType.TASK_PAUSED,
            session_id=record.session_id,
            data={"task_id": record.id, "goal": record.goal},
        ))
        return record

    async def resume(self, task_id: str) -> TaskRecord | None:
        record = self.get(task_id)
        if not record or record.status != TaskStatus.PAUSED:
            return None
        control = self._controls.get(record.id)
        started = record.started_at is not None
        record.status = TaskStatus.RUNNING if started else TaskStatus.QUEUED
        if control:
            control.resume()
        if not started:
            self._push(record)
            self._wakeup.set()
        await self._persist(record)
        await event_bus.emit(Event(
            type=EventType.TASK_RESUMED,
            session_id=record.session_id,
            data={"task_id": record.id, "goal": record.goal},
        ))
        return record

    async def cancel(self, task_id: str) -> TaskRecord | None:
        record = self.get(task_id)
        if not record or record.status.is_terminal():
            return None
        control = self._controls.get(record.id)
        if control:
            control.cancel()
        if record.started_at is None:
            # Never started — finalize immediately
            await self._finalize(record, TaskStatus.CANCELLED, result="Cancelled before start")
        else:
            asyncio.get_running_loop().create_task(self._enforce_cancel(record.id))
        return record

    async def set_priority(self, task_id: str, priority: int) -> TaskRecord | None:
        record = self.get(task_id)
        if not record or record.status.is_terminal():
            return None
        record.priority = priority
        if record.status == TaskStatus.QUEUED:
            self._push(record)  # stale heap entries are skipped on pop
            self._wakeup.set()
        await self._persist(record)
        return record

    async def _enforce_cancel(self, task_id: str) -> None:
        """Hard-cancel the asyncio task if cooperative cancel doesn't land."""
        await asyncio.sleep(_CANCEL_GRACE_SECONDS)
        record = self._records.get(task_id)
        runner = self._running_tasks.get(task_id)
        if record and not record.status.is_terminal() and runner and not runner.done():
            logger.warning("task_hard_cancel", task_id=task_id)
            runner.cancel()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _push(self, record: TaskRecord) -> None:
        heapq.heappush(
            self._heap, (-record.priority, next(self._seq), record.id, record.priority)
        )

    def _pop_next(self) -> TaskRecord | None:
        while self._heap:
            _, _, task_id, priority_at_push = heapq.heappop(self._heap)
            record = self._records.get(task_id)
            if not record or record.status != TaskStatus.QUEUED:
                continue  # cancelled/paused/finished or stale
            if record.priority != priority_at_push:
                continue  # stale entry — a fresher one exists
            return record
        return None

    async def _worker_loop(self) -> None:
        max_concurrent = max(1, settings.max_concurrent_tasks)
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()

            while len(self._running_tasks) < max_concurrent:
                record = self._pop_next()
                if record is None:
                    break
                runner = asyncio.create_task(
                    self._run_task(record), name=f"task-{record.id[:8]}"
                )
                self._running_tasks[record.id] = runner
                runner.add_done_callback(
                    lambda t, tid=record.id: self._on_runner_done(tid)
                )

    def _on_runner_done(self, task_id: str) -> None:
        self._running_tasks.pop(task_id, None)
        self._wakeup.set()  # a slot freed up — check the queue

    async def _run_task(self, record: TaskRecord) -> None:
        from computer_agent.coordinator import Coordinator
        from computer_agent.taskmgr.models import _now

        control = self._controls[record.id]
        record.status = TaskStatus.RUNNING
        record.started_at = _now()
        await self._persist(record)
        logger.info("task_started", task_id=record.id, goal=record.goal[:80])

        coordinator = Coordinator(
            session_id=record.session_id,
            task_id=record.id,
            task_control=control,
        )

        try:
            result = await coordinator.run(record.goal)
            if control.is_cancelled:
                await self._finalize(record, TaskStatus.CANCELLED, result=result)
            else:
                await self._finalize(record, TaskStatus.COMPLETED, result=result)
        except TaskCancelledError:
            await self._finalize(record, TaskStatus.CANCELLED, result="Cancelled by user")
        except asyncio.CancelledError:
            await self._finalize(record, TaskStatus.CANCELLED, result="Terminated")
        except Exception as e:
            logger.error("task_failed", task_id=record.id, error=str(e))
            await self._finalize(record, TaskStatus.FAILED, error=str(e))

    async def _finalize(
        self,
        record: TaskRecord,
        status: TaskStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        from computer_agent.taskmgr.models import _now

        record.status = status
        record.result = result
        record.error = error
        record.finished_at = _now()
        await self._persist(record)

        if status == TaskStatus.CANCELLED:
            await event_bus.emit(Event(
                type=EventType.TASK_CANCELLED,
                session_id=record.session_id,
                data={"task_id": record.id, "goal": record.goal},
            ))
        logger.info("task_finished", task_id=record.id, status=status.value)

    # ------------------------------------------------------------------
    # Persistence (best-effort)
    # ------------------------------------------------------------------

    async def _persist(self, record: TaskRecord) -> None:
        try:
            from computer_agent.memory.store import memory_store
            await memory_store.save_task(record.to_dict())
        except Exception as e:
            logger.debug("task_persist_failed", task_id=record.id, error=str(e))


# Module-level singleton
task_manager = TaskManager()
