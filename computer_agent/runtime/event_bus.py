"""
Event Bus — lightweight async pub/sub for inter-agent communication.

All significant runtime events are emitted through the bus so that:
  - Observability can trace the full execution lifecycle
  - Agents can react to each other's state changes
  - The UI can subscribe to receive live updates

Usage:
    event_bus.emit(Event(type=EventType.TASK_STEP_COMPLETED, data={...}))
    event_bus.subscribe(EventType.HITL_APPROVAL_REQUESTED, my_handler)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    # Task lifecycle
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"
    TASK_PAUSED = "task.paused"
    TASK_RESUMED = "task.resumed"
    TASK_PROGRESS = "task.progress"

    # Step lifecycle
    STEP_STARTED = "task.step.started"
    STEP_COMPLETED = "task.step.completed"
    STEP_FAILED = "task.step.failed"
    STEP_RETRYING = "task.step.retrying"

    # HITL
    HITL_APPROVAL_REQUESTED = "hitl.approval.requested"
    HITL_APPROVAL_GRANTED = "hitl.approval.granted"
    HITL_APPROVAL_DENIED = "hitl.approval.denied"
    HITL_APPROVAL_EXPIRED = "hitl.approval.expired"

    # Memory
    TRACE_SAVED = "memory.trace.saved"
    TRACE_RETRIEVED = "memory.trace.retrieved"

    # Tools
    TOOL_INVOKED = "tool.invoked"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"

    # Agent
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"

    # System
    SKILL_REGISTERED = "skill.registered"
    ERROR = "system.error"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Simple async event bus with typed subscriptions."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}
        self._history: list[Event] = []
        self._max_history: int = 1000

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    async def emit(self, event: Event) -> None:
        """Emit an event, calling all registered handlers."""
        # Store in history (ring buffer)
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Handlers must not crash the bus

    def get_history(
        self,
        event_type: EventType | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        return events[-limit:]


# Module-level singleton
event_bus = EventBus()
