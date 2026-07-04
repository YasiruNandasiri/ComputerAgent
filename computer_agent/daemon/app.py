"""
Daemon application — FastAPI app hosting the agent's long-lived services:
task worker, scheduler, HITL manager, notifier subscriptions, and the API.

Run with: computer-agent daemon
"""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

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
    from computer_agent.config import settings
    from computer_agent.daemon.api import router

    app = FastAPI(
        title="Computer Agent Daemon",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Generate a random token at startup; the UI fetches it via GET /startup-token.
    token = secrets.token_hex(32)
    app.state.startup_token = token
    logger.info("daemon_startup_token_ready", hint="fetch GET /startup-token from localhost")

    app.add_middleware(
        _LocalhostGuardMiddleware,
        token=token,
        daemon_host=settings.daemon_host,
    )
    app.include_router(router)
    return app


class _LocalhostGuardMiddleware(BaseHTTPMiddleware):
    """DNS-rebinding guard + startup-token enforcement for mutating requests."""

    def __init__(self, app: FastAPI, token: str, daemon_host: str) -> None:
        super().__init__(app)
        self._token = token
        self._allowed_hosts = {daemon_host, "localhost", "127.0.0.1"}

    async def dispatch(self, request, call_next):  # type: ignore[override]
        host_header = request.headers.get("host", "")
        if host_header:
            host_name = host_header.split(":")[0]
            if host_name not in self._allowed_hosts:
                return JSONResponse({"error": "Forbidden: invalid Host header"}, status_code=403)

        # Exempt safe/idempotent methods from token check
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {self._token}":
                return JSONResponse({"error": "Unauthorized: bearer token required"}, status_code=401)

        return await call_next(request)
