"""
Approval UI — notifies the user when HITL approval is required.

Delivery channels (in order of availability):
  1. macOS native notification (via osascript) — macOS only
  2. Terminal print — always available (fallback)
  3. Future: mobile push, Slack DM, email

The UI also provides a CLI-compatible resolve interface.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from computer_agent.logging_setup import get_logger

if TYPE_CHECKING:
    from computer_agent.hitl.checkpoint import CheckpointState

logger = get_logger(__name__)


class ApprovalUI:
    """Sends approval requests to the user and collects responses."""

    async def notify(self, state: CheckpointState) -> None:
        """Dispatch a notification for the given checkpoint state."""
        if sys.platform == "darwin":
            await self._notify_macos(state)
        else:
            self._notify_terminal(state)

    async def _notify_macos(self, state: CheckpointState) -> None:
        """Send a macOS notification via the shared Notifier service."""
        try:
            from computer_agent.notify.notifier import notifier
            await notifier.notify(
                title="Computer Agent — Approval Required",
                message=state.approval_message[:200],
                subtitle=f"Tool: {state.proposed_tool}",
            )
            logger.debug("approval_notification_sent", checkpoint_id=state.checkpoint_id)
        except Exception as e:
            logger.debug("approval_notification_failed", error=str(e))

        # Always also print to terminal for visibility
        self._notify_terminal(state)

    def _notify_terminal(self, state: CheckpointState) -> None:
        """Print approval request to terminal."""
        sep = "=" * 70
        print(f"\n{sep}")
        print("  COMPUTER AGENT — APPROVAL REQUIRED")
        print(sep)
        print(f"  Checkpoint : {state.checkpoint_id}")
        print(f"  Session    : {state.session_id}")
        print(f"  Goal       : {state.goal}")
        print(f"  Tool       : {state.proposed_tool}")
        print(f"  Reason     : {state.risk_reason}")
        print(f"\n  {state.approval_message}")
        print(f"\n  Parameters : {state.proposed_parameters}")
        print("\nTo respond, run:")
        print(f"  computer-agent approve {state.checkpoint_id}")
        print(f"  computer-agent deny    {state.checkpoint_id}")
        print(sep + "\n")

    async def prompt_inline(self, state: CheckpointState) -> bool:
        """
        Interactive inline approval prompt (for terminal/CLI sessions).
        Blocks until user responds with y/n.
        """
        self._notify_terminal(state)
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: input("\n  Approve? [y/N]: ").strip().lower(),
            )
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False


# Module-level singleton
approval_ui = ApprovalUI()
