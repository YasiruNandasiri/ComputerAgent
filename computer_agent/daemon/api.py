"""
Daemon HTTP API — the single control surface for the CLI (and future UIs).

Because these routes run in the same process as the TaskManager, HITLManager,
and Coordinators, approve/deny resolves the real in-memory asyncio.Event —
fixing the old broken cross-process approval flow.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from computer_agent.abilities.autonomy import AutonomyLevel, autonomy_manager
from computer_agent.hitl.checkpoint import hitl_manager
from computer_agent.runtime.event_bus import Event, EventType, event_bus
from computer_agent.taskmgr.manager import task_manager
from computer_agent.taskmgr.models import TaskStatus

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SubmitTaskRequest(BaseModel):
    goal: str
    priority: int = 5
    source: str = "user"


class ResolveRequest(BaseModel):
    approved: bool
    note: str = ""


class ModeRequest(BaseModel):
    level: str


class PriorityRequest(BaseModel):
    priority: int


class RoutineRequest(BaseModel):
    name: str
    cron: str
    goal: str
    priority: int = 5
    notify: bool = True


# ---------------------------------------------------------------------------
# Status & mode
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_status() -> dict[str, Any]:
    from computer_agent.scheduler.service import scheduler_service

    return {
        "ok": True,
        "autonomy_level": autonomy_manager.level.value,
        "worker_running": task_manager.is_running,
        "scheduler_running": scheduler_service.is_running,
        "tasks": task_manager.status_summary(),
        "pending_approvals": len(hitl_manager.get_pending_checkpoints()),
    }


@router.get("/mode")
async def get_mode() -> dict[str, str]:
    return {"level": autonomy_manager.level.value}


@router.put("/mode")
async def set_mode(body: ModeRequest) -> dict[str, str]:
    try:
        level = AutonomyLevel(body.level.lower())
    except ValueError as e:
        raise HTTPException(400, f"Invalid level '{body.level}'. Use low, medium, or high.") from e
    await autonomy_manager.set_level(level)
    return {"level": level.value}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(body: ChatRequest) -> dict[str, str]:
    from computer_agent.daemon.chat import chat_sessions

    session = chat_sessions.get_or_create(body.session_id)
    response = await session.handle(body.message)
    return {"response": response, "session_id": session.session_id}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@router.post("/tasks")
async def submit_task(body: SubmitTaskRequest) -> dict[str, Any]:
    record = await task_manager.submit(
        goal=body.goal, priority=body.priority, source=body.source
    )
    return record.summary()


@router.get("/tasks")
async def list_tasks(status: str | None = None) -> list[dict[str, Any]]:
    status_filter = TaskStatus(status) if status else None
    return [r.summary() for r in task_manager.list(status=status_filter)]


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    record = task_manager.get(task_id)
    if not record:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return record.to_dict()


@router.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str) -> dict[str, Any]:
    record = await task_manager.pause(task_id)
    if not record:
        raise HTTPException(404, f"Task '{task_id}' not found or already finished")
    return record.summary()


@router.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str) -> dict[str, Any]:
    record = await task_manager.resume(task_id)
    if not record:
        raise HTTPException(404, f"Task '{task_id}' not found or not paused")
    return record.summary()


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    record = await task_manager.cancel(task_id)
    if not record:
        raise HTTPException(404, f"Task '{task_id}' not found or already finished")
    return record.summary()


@router.patch("/tasks/{task_id}")
async def set_task_priority(task_id: str, body: PriorityRequest) -> dict[str, Any]:
    record = await task_manager.set_priority(task_id, body.priority)
    if not record:
        raise HTTPException(404, f"Task '{task_id}' not found or already finished")
    return record.summary()


# ---------------------------------------------------------------------------
# HITL approvals
# ---------------------------------------------------------------------------

@router.get("/hitl/pending")
async def pending_approvals() -> list[dict[str, Any]]:
    return [
        {
            "checkpoint_id": s.checkpoint_id,
            "session_id": s.session_id,
            "tool": s.proposed_tool,
            "params": s.proposed_parameters,
            "message": s.approval_message,
            "goal": s.goal,
            "created_at": s.created_at,
        }
        for s in hitl_manager.get_pending_checkpoints()
    ]


@router.post("/hitl/{checkpoint_id}/resolve")
async def resolve_checkpoint(checkpoint_id: str, body: ResolveRequest) -> dict[str, Any]:
    found = hitl_manager.resolve(checkpoint_id, approved=body.approved, user_note=body.note)
    if not found:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found")
    return {"checkpoint_id": checkpoint_id, "approved": body.approved}


# ---------------------------------------------------------------------------
# Routines (scheduled tasks)
# ---------------------------------------------------------------------------

@router.get("/routines")
async def list_routines() -> list[dict[str, Any]]:
    from computer_agent.scheduler.service import scheduler_service
    return scheduler_service.list_routines()


@router.post("/routines")
async def add_routine(body: RoutineRequest) -> dict[str, Any]:
    from computer_agent.scheduler.service import scheduler_service
    try:
        return await scheduler_service.add_routine(
            name=body.name,
            cron=body.cron,
            goal=body.goal,
            priority=body.priority,
            notify=body.notify,
        )
    except ValueError as e:
        raise HTTPException(400, f"Invalid cron expression: {e}") from e


@router.delete("/routines/{name}")
async def remove_routine(name: str) -> dict[str, Any]:
    from computer_agent.scheduler.service import scheduler_service
    if not await scheduler_service.remove_routine(name):
        raise HTTPException(404, f"Routine '{name}' not found")
    return {"removed": name}


@router.post("/routines/{name}/enable")
async def enable_routine(name: str) -> dict[str, Any]:
    from computer_agent.scheduler.service import scheduler_service
    routine = await scheduler_service.set_enabled(name, True)
    if not routine:
        raise HTTPException(404, f"Routine '{name}' not found")
    return routine


@router.post("/routines/{name}/disable")
async def disable_routine(name: str) -> dict[str, Any]:
    from computer_agent.scheduler.service import scheduler_service
    routine = await scheduler_service.set_enabled(name, False)
    if not routine:
        raise HTTPException(404, f"Routine '{name}' not found")
    return routine


# ---------------------------------------------------------------------------
# Live event stream (SSE)
# ---------------------------------------------------------------------------

_STREAMED_EVENTS = (
    EventType.TASK_STARTED,
    EventType.TASK_COMPLETED,
    EventType.TASK_FAILED,
    EventType.TASK_CANCELLED,
    EventType.TASK_PAUSED,
    EventType.TASK_RESUMED,
    EventType.STEP_COMPLETED,
    EventType.HITL_APPROVAL_REQUESTED,
    EventType.HITL_APPROVAL_GRANTED,
    EventType.HITL_APPROVAL_DENIED,
    EventType.HITL_APPROVAL_EXPIRED,
)


@router.get("/events")
async def event_stream() -> StreamingResponse:
    """Server-Sent Events stream of task/HITL lifecycle events."""
    queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=200)

    def _handler(event: Event) -> None:
        with contextlib.suppress(asyncio.QueueFull):  # slow consumer — drop rather than block the bus
            queue.put_nowait(event)

    for event_type in _STREAMED_EVENTS:
        event_bus.subscribe(event_type, _handler)

    async def _generate():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                payload = json.dumps({
                    "type": event.type.value,
                    "session_id": event.session_id,
                    "timestamp": event.timestamp,
                    "data": event.data,
                }, default=str)
                yield f"data: {payload}\n\n"
        finally:
            for event_type in _STREAMED_EVENTS:
                event_bus.unsubscribe(event_type, _handler)

    return StreamingResponse(_generate(), media_type="text/event-stream")
