"""
SchedulerService — cron-style recurring routines executed as background tasks.

Routines ("check my email for urgent messages every 30 minutes") are stored in
the scheduled_routines table and loaded at daemon start. Each trigger submits
a task to the TaskManager, so routine runs get the same lifecycle, HITL gating,
progress tracking, and notifications as any other task.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from computer_agent.logging_setup import get_logger
from computer_agent.memory.store import memory_store

logger = get_logger(__name__)


class SchedulerService:
    """Wraps APScheduler's AsyncIOScheduler; persists routines in Postgres."""

    def __init__(self) -> None:
        self._scheduler: Any = None
        # In-memory mirror so the scheduler works without a database
        self._routines: dict[str, dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    async def start(self) -> None:
        """Start the scheduler and load persisted routines."""
        if self.is_running:
            return

        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        self._scheduler = AsyncIOScheduler()
        self._scheduler.start()

        for routine in await memory_store.list_routines():
            r = {
                "id": str(routine["id"]),
                "name": routine["name"],
                "cron": routine["cron"],
                "goal": routine["goal"],
                "priority": routine["priority"],
                "enabled": routine["enabled"],
                "notify": routine["notify"],
            }
            self._routines[r["name"]] = r
            if r["enabled"]:
                self._schedule_job(r)

        logger.info("scheduler_started", routines=len(self._routines))

    async def stop(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

    # ------------------------------------------------------------------
    # Routine CRUD
    # ------------------------------------------------------------------

    async def add_routine(
        self,
        name: str,
        cron: str,
        goal: str,
        priority: int = 5,
        notify: bool = True,
    ) -> dict[str, Any]:
        from apscheduler.triggers.cron import CronTrigger

        CronTrigger.from_crontab(cron)  # validate; raises ValueError on bad cron

        routine = {
            "id": self._routines.get(name, {}).get("id", str(uuid.uuid4())),
            "name": name,
            "cron": cron,
            "goal": goal,
            "priority": priority,
            "enabled": True,
            "notify": notify,
        }
        self._routines[name] = routine
        await memory_store.save_routine(routine)
        if self.is_running:
            self._schedule_job(routine)
        logger.info("routine_added", name=name, cron=cron)
        return routine

    async def remove_routine(self, name: str) -> bool:
        routine = self._routines.pop(name, None)
        await memory_store.delete_routine(name)
        if routine and self.is_running:
            self._unschedule_job(routine)
        return routine is not None

    async def set_enabled(self, name: str, enabled: bool) -> dict[str, Any] | None:
        routine = self._routines.get(name)
        if not routine:
            return None
        routine["enabled"] = enabled
        await memory_store.save_routine(routine)
        if self.is_running:
            if enabled:
                self._schedule_job(routine)
            else:
                self._unschedule_job(routine)
        return routine

    def list_routines(self) -> list[dict[str, Any]]:
        return sorted(self._routines.values(), key=lambda r: r["name"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _schedule_job(self, routine: dict[str, Any]) -> None:
        from apscheduler.triggers.cron import CronTrigger

        self._scheduler.add_job(
            self._fire,
            CronTrigger.from_crontab(routine["cron"]),
            id=routine["id"],
            args=[routine["name"]],
            replace_existing=True,
        )

    def _unschedule_job(self, routine: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._scheduler.remove_job(routine["id"])

    async def _fire(self, name: str) -> None:
        """Cron trigger fired — enqueue the routine as a background task."""
        routine = self._routines.get(name)
        if not routine or not routine["enabled"]:
            return

        from computer_agent.taskmgr.manager import task_manager

        record = await task_manager.submit(
            goal=routine["goal"],
            priority=routine["priority"],
            source="schedule",
            schedule_id=routine["id"],
        )
        try:
            await memory_store.mark_routine_run(name)
        except Exception as e:
            logger.debug("routine_mark_run_failed", error=str(e))
        logger.info("routine_fired", name=name, task_id=record.id)


# Module-level singleton
scheduler_service = SchedulerService()
