"""
HITL Checkpoint — state machine for Human-in-the-Loop approval gates.

When the Abilities Engine flags an action as REQUIRE_HITL:
1. The agent's full execution context is serialized to the memory store.
2. The user is notified via ApprovalUI.
3. The agent suspends the current step and awaits a decision.
4. On approval → resume from checkpoint; on denial → cancel and inform user.

State transitions:
    RUNNING → AWAITING_APPROVAL → APPROVED → RUNNING
                                └→ DENIED → CANCELLED
                                └→ EXPIRED → CANCELLED
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from computer_agent.config import settings
from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class CheckpointStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class CheckpointState(BaseModel):
    """Complete serialized state of an agent execution at a HITL gate."""

    session_id: str
    checkpoint_id: str
    created_at: str = ""

    # Task context
    goal: str
    remaining_steps: list[dict[str, Any]] = []
    completed_steps: list[dict[str, Any]] = []
    working_memory: dict[str, Any] = {}

    # LLM context
    conversation_history: list[dict[str, Any]] = []

    # The action that triggered the checkpoint
    proposed_tool: str
    proposed_parameters: dict[str, Any] = {}
    risk_reason: str
    approval_message: str

    # Resolution
    status: CheckpointStatus = CheckpointStatus.PENDING
    resolved_at: str | None = None
    user_note: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# HITL Manager
# ---------------------------------------------------------------------------

class HITLManager:
    """
    Manages Human-in-the-Loop approval checkpoints.
    Uses an asyncio.Event per checkpoint for non-blocking wait.
    """

    def __init__(self) -> None:
        # checkpoint_id → asyncio.Event (set when resolved)
        self._pending: dict[str, asyncio.Event] = {}
        # checkpoint_id → CheckpointState
        self._states: dict[str, CheckpointState] = {}

    async def request_approval(
        self,
        session_id: str,
        goal: str,
        proposed_tool: str,
        proposed_parameters: dict[str, Any],
        risk_reason: str,
        approval_message: str,
        conversation_history: list[dict[str, Any]],
        working_memory: dict[str, Any],
        remaining_steps: list[dict[str, Any]],
        completed_steps: list[dict[str, Any]],
        task_id: str | None = None,
    ) -> CheckpointState:
        """
        Serialize state, notify user, and wait for their decision.
        Returns the resolved CheckpointState with status APPROVED or DENIED.
        """
        checkpoint_id = str(uuid.uuid4())
        event = asyncio.Event()

        state = CheckpointState(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(UTC).isoformat(),
            goal=goal,
            proposed_tool=proposed_tool,
            proposed_parameters=proposed_parameters,
            risk_reason=risk_reason,
            approval_message=approval_message,
            conversation_history=conversation_history,
            working_memory=working_memory,
            remaining_steps=remaining_steps,
            completed_steps=completed_steps,
        )

        self._pending[checkpoint_id] = event
        self._states[checkpoint_id] = state

        # Persist to memory store if available
        await self._persist_state(state)

        # Notify user
        from computer_agent.hitl.approval_ui import approval_ui
        await approval_ui.notify(state)

        from computer_agent.runtime.event_bus import Event, EventType, event_bus
        await event_bus.emit(Event(
            type=EventType.HITL_APPROVAL_REQUESTED,
            session_id=session_id,
            data={
                "checkpoint_id": checkpoint_id,
                "tool": proposed_tool,
                "params": proposed_parameters,
                "message": approval_message,
                "task_id": task_id,
            },
        ))

        logger.info(
            "hitl_waiting_approval",
            checkpoint_id=checkpoint_id,
            tool=proposed_tool,
            session=session_id,
        )

        # Wait for resolution. With hitl_timeout_action="pause" the wait is
        # indefinite — the task simply stays in awaiting_approval until the
        # user decides. With "expire" the legacy timeout applies.
        if settings.hitl_timeout_action == "pause":
            await event.wait()
        else:
            try:
                await asyncio.wait_for(
                    event.wait(),
                    timeout=float(settings.hitl_approval_timeout),
                )
            except TimeoutError:
                state.status = CheckpointStatus.EXPIRED
                logger.warning("hitl_approval_expired", checkpoint_id=checkpoint_id)
                await event_bus.emit(Event(
                    type=EventType.HITL_APPROVAL_EXPIRED,
                    session_id=session_id,
                    data={"checkpoint_id": checkpoint_id, "tool": proposed_tool},
                ))
                await self._update_snapshot(checkpoint_id, state.status.value)

        # Clean up
        self._pending.pop(checkpoint_id, None)
        return state

    def resolve(
        self,
        checkpoint_id: str,
        approved: bool,
        user_note: str = "",
    ) -> bool:
        """
        Called externally (CLI/UI) when the user approves or denies.
        Returns True if the checkpoint was found, False otherwise.
        """
        state = self._states.get(checkpoint_id)
        if not state:
            logger.warning("hitl_resolve_unknown", checkpoint_id=checkpoint_id)
            return False

        state.status = CheckpointStatus.APPROVED if approved else CheckpointStatus.DENIED
        state.resolved_at = datetime.now(UTC).isoformat()
        state.user_note = user_note

        event = self._pending.get(checkpoint_id)
        if event:
            event.set()

        # Persist the resolution when called from an async context (daemon)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._update_snapshot(checkpoint_id, state.status.value))
        except RuntimeError:
            pass  # No running loop (bare CLI process) — snapshot stays pending

        logger.info(
            "hitl_resolved",
            checkpoint_id=checkpoint_id,
            approved=approved,
            note=user_note,
        )
        return True

    def get_pending_checkpoints(self) -> list[CheckpointState]:
        """Return all currently pending approval requests."""
        return [
            s for s in self._states.values()
            if s.status == CheckpointStatus.PENDING
        ]

    async def _update_snapshot(self, checkpoint_id: str, status: str) -> None:
        """Best-effort persistence of a checkpoint resolution."""
        try:
            from computer_agent.memory.store import memory_store
            await memory_store.update_snapshot_status(checkpoint_id, status)
        except Exception as e:
            logger.debug("hitl_snapshot_update_failed", error=str(e))

    async def _persist_state(self, state: CheckpointState) -> None:
        """Persist checkpoint state to the memory store."""
        try:
            from computer_agent.memory.store import memory_store
            await memory_store.save_snapshot(
                session_id=state.session_id,
                checkpoint_id=state.checkpoint_id,
                serialized_state=state.to_dict(),
                proposed_action=f"{state.proposed_tool}({state.proposed_parameters})",
                risk_reason=state.risk_reason,
            )
        except Exception as e:
            logger.warning("hitl_persist_failed", error=str(e))


# Module-level singleton
hitl_manager = HITLManager()
