"""Task records — the unit of work managed by the TaskManager."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


def _now() -> str:
    return datetime.now(UTC).isoformat()


_MAX_PROGRESS = 50


class TaskRecord(BaseModel):
    """A single unit of agent work, queued and executed by the TaskManager."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    status: TaskStatus = TaskStatus.QUEUED
    priority: int = 5          # higher = more urgent
    source: str = "user"       # user | schedule | chat
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schedule_id: str | None = None
    result: str | None = None
    error: str | None = None
    progress: list[str] = []   # recent step notes, newest last (capped)
    created_at: str = Field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None

    def add_progress(self, note: str) -> None:
        self.progress.append(note)
        if len(self.progress) > _MAX_PROGRESS:
            del self.progress[: len(self.progress) - _MAX_PROGRESS]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def summary(self) -> dict[str, Any]:
        """Compact view for listings and chat answers."""
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status.value,
            "priority": self.priority,
            "source": self.source,
            "created_at": self.created_at,
            "last_progress": self.progress[-1] if self.progress else None,
        }
