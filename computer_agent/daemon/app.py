"""
Daemon application — FastAPI app hosting the agent's long-lived services:
task worker, scheduler, HITL manager, notifier subscriptions, and the API.

Run with: computer-agent daemon
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


def _register_notifications() -> None:
    """Notify the user when background/scheduled tasks finish or need approval."""
    from computer_agent.notify.notifier import notifier
    from computer_agent.runtime.event_bus import Event, EventType, event_bus

    async def _on_task_done(event: Event) -> None:
        data = event.data
        if not data.get("task_id"):
            return  # ad-hoc coordinator runs (chat turns) don't notify
        goal = (data.get("goal") or "")[:80]
        if event.type == EventType.TASK_COMPLETED:
            await notifier.notify(
                title="Task completed",
                message=data.get("response", "")[:200] or "Done.",
                subtitle=goal,
            )
        elif event.type == EventType.TASK_FAILED:
            await notifier.notify(
                title="Task failed",
                message=data.get("error", "Unknown error")[:200],
                subtitle=goal,
            )

    event_bus.subscribe(EventType.TASK_COMPLETED, _on_task_done)
    event_bus.subscribe(EventType.TASK_FAILED, _on_task_done)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from computer_agent.abilities.autonomy import autonomy_manager
    from computer_agent.config import settings
    from computer_agent.memory.store import memory_store
    from computer_agent.runtime.progress import register_progress_subscriber
    from computer_agent.scheduler.service import scheduler_service
    from computer_agent.skills.loader import skill_registry
    from computer_agent.taskmgr.manager import task_manager
    from computer_agent.tools.registry import registry

    # Bootstrap
    registry.discover()
    skill_registry.discover()
    skill_registry.register_with_tool_registry()
    await memory_store.connect()
    await autonomy_manager.load()
    register_progress_subscriber()
    _register_notifications()
    await task_manager.start()
    if settings.scheduler_enabled:
        await scheduler_service.start()

    logger.info(
        "daemon_started",
        host=settings.daemon_host,
        port=settings.daemon_port,
        autonomy=autonomy_manager.level.value,
    )
    yield

    # Shutdown
    await scheduler_service.stop()
    await task_manager.stop()
    await memory_store.disconnect()
    logger.info("daemon_stopped")


def create_app() -> FastAPI:
    from computer_agent.daemon.api import router

    app = FastAPI(
        title="Computer Agent Daemon",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.include_router(router)
    return app
