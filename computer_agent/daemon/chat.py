"""
ChatSession — conversational interface to the daemon.

Fast-path intents (regex, no LLM round-trip) handle approvals, mode changes,
and task control instantly; everything else goes to a persistent Coordinator
that also has the agent-control tools, so natural phrasing ("hold that
download for now") works too.
"""

from __future__ import annotations

import re
from typing import Any

from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)

_CHAT_EXTRA_PROMPT = """## Conversation mode
You are chatting with the user while background tasks may be running.
- For long-running work, call start_background_task instead of doing it inline,
  then tell the user the task id and that you'll keep working in the background.
- Use get_task_status / list_tasks / get_task to answer questions about progress.
- Use pause_task / resume_task / cancel_task / set_task_priority when the user
  asks to hold, resume, stop, or reprioritize work.
- Only call approve_pending_action / deny_pending_action when the user clearly
  approves or declines a specific pending action in this conversation."""

_APPROVE_RE = re.compile(r"^\s*(approve|yes|yep|ok(ay)?|do it|go ahead)\s*[.!]?\s*$", re.I)
_DENY_RE = re.compile(r"^\s*(deny|no|nope|don'?t|stop|reject|cancel that)\s*[.!]?\s*$", re.I)
_MODE_RE = re.compile(r"^\s*(?:set\s+)?mode\s+(low|medium|high)\s*$", re.I)
_STATUS_RE = re.compile(
    r"^\s*(status|what are you (working on|doing|focusing on)\??|progress\??)\s*$", re.I
)


class ChatSession:
    """One conversational session with its own Coordinator and history."""

    def __init__(self, session_id: str | None = None) -> None:
        from computer_agent.coordinator import Coordinator

        self._coordinator = Coordinator(
            session_id=session_id,
            extra_system_prompt=_CHAT_EXTRA_PROMPT,
        )
        self.session_id = self._coordinator.session_id

    async def handle(self, message: str) -> str:
        fast = await self._fast_path(message)
        if fast is not None:
            return fast
        return await self._coordinator.run(message)

    # ------------------------------------------------------------------
    # Fast-path intents (no LLM call)
    # ------------------------------------------------------------------

    async def _fast_path(self, message: str) -> str | None:
        from computer_agent.hitl.checkpoint import hitl_manager

        m = _MODE_RE.match(message)
        if m:
            from computer_agent.abilities.autonomy import AutonomyLevel, autonomy_manager
            level = await autonomy_manager.set_level(AutonomyLevel(m.group(1).lower()))
            return f"Autonomy level set to {level.value}."

        if _STATUS_RE.match(message):
            return self._format_status()

        pending = hitl_manager.get_pending_checkpoints()
        if pending:
            newest = max(pending, key=lambda s: s.created_at)
            if _APPROVE_RE.match(message):
                hitl_manager.resolve(newest.checkpoint_id, approved=True, user_note="via chat")
                return f"Approved: {newest.proposed_tool} — resuming the task."
            if _DENY_RE.match(message):
                hitl_manager.resolve(newest.checkpoint_id, approved=False, user_note="via chat")
                return f"Denied: {newest.proposed_tool} — the task will not run that action."

        return None

    def _format_status(self) -> str:
        from computer_agent.taskmgr.manager import task_manager

        summary = task_manager.status_summary()
        lines: list[str] = []

        current: dict[str, Any] | None = summary["current"]
        if current:
            lines.append(f"Currently working on: {current['goal']} (task {current['id'][:8]})")
            if current.get("last_progress"):
                lines.append(f"  Latest step: {current['last_progress']}")
        else:
            lines.append("No task is running right now.")

        for r in summary["awaiting_approval"]:
            lines.append(f"Awaiting your approval: {r['goal']} (task {r['id'][:8]})")
        for r in summary["paused"]:
            lines.append(f"Paused: {r['goal']} (task {r['id'][:8]})")
        if summary["queued"]:
            lines.append("Queued:")
            for r in summary["queued"]:
                lines.append(f"  [{r['priority']}] {r['goal']} (task {r['id'][:8]})")

        return "\n".join(lines)


class ChatSessionManager:
    """Holds live chat sessions keyed by session id."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: str | None) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        session = ChatSession(session_id=session_id)
        self._sessions[session.session_id] = session
        return session


# Module-level singleton
chat_sessions = ChatSessionManager()
