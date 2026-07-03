"""
Progress subscriber — persists task lifecycle events from the event bus into
the task manager's records (and the task_events table), so "what are you
working on?" and `task show` reflect live progress.

Registered once in the daemon lifespan via register_progress_subscriber().
"""

from __future__ import annotations

from computer_agent.logging_setup import get_logger
from computer_agent.runtime.event_bus import Event, EventType, event_bus
from computer_agent.taskmgr.models import TaskStatus

logger = get_logger(__name__)

_TRACKED = (
    EventType.STEP_COMPLETED,
    EventType.TOOL_INVOKED,
    EventType.TOOL_FAILED,
    EventType.HITL_APPROVAL_REQUESTED,
    EventType.HITL_APPROVAL_GRANTED,
    EventType.HITL_APPROVAL_DENIED,
    EventType.HITL_APPROVAL_EXPIRED,
)


async def _on_event(event: Event) -> None:
    task_id = event.data.get("task_id")
    if not task_id:
        return

    from computer_agent.memory.store import memory_store
    from computer_agent.taskmgr.manager import task_manager

    record = task_manager.get(task_id)
    if record:
        if event.type == EventType.STEP_COMPLETED:
            tools = ", ".join(event.data.get("tools", []))
            record.add_progress(f"turn {event.data.get('turn')}: {tools}")
        elif event.type == EventType.HITL_APPROVAL_REQUESTED:
            record.add_progress(
                f"waiting for approval: {event.data.get('tool')} "
                f"(checkpoint {str(event.data.get('checkpoint_id'))[:8]})"
            )
            if record.status == TaskStatus.RUNNING:
                record.status = TaskStatus.AWAITING_APPROVAL
        elif event.type in (EventType.HITL_APPROVAL_GRANTED, EventType.HITL_APPROVAL_DENIED):
            verdict = "approved" if event.type == EventType.HITL_APPROVAL_GRANTED else "denied"
            record.add_progress(f"user {verdict}: {event.data.get('tool')}")
            if record.status == TaskStatus.AWAITING_APPROVAL:
                record.status = TaskStatus.RUNNING
        elif event.type == EventType.TOOL_FAILED:
            record.add_progress(f"tool failed: {event.data.get('tool')}")

    try:
        await memory_store.append_task_event(task_id, event.type.value, event.data)
    except Exception as e:
        logger.debug("task_event_persist_failed", error=str(e))


def register_progress_subscriber() -> None:
    for event_type in _TRACKED:
        event_bus.subscribe(event_type, _on_event)
    logger.info("progress_subscriber_registered")
