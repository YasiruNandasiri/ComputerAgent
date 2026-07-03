"""
Agent-control tools — let the agent notify the user, manage its own task
queue, and adjust settings when asked in natural language ("hold that
download", "what are you working on?", "set mode high").

All tools are thin wrappers over the TaskManager / AutonomyManager / HITL
singletons and are RiskLevel.LOW: they only affect the agent itself, never
the user's computer or external services.
"""

from __future__ import annotations

from typing import Any

from computer_agent.tools.base import RiskLevel, ToolResult, tool


@tool(
    description="Send a notification to the user (system notification). Use this to "
                "proactively surface important findings, e.g. an urgent email or a "
                "completed background task.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def notify_user(title: str, message: str) -> ToolResult:
    from computer_agent.notify.notifier import notifier
    await notifier.notify(title=title, message=message)
    return ToolResult.ok(output=f"Notification sent: {title}")


@tool(
    description="Get the agent's current focus: the running task, tasks awaiting "
                "approval, paused tasks, and the queue. Use when the user asks what "
                "you are working on or for overall progress.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
def get_task_status() -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager
    return ToolResult.ok(output=task_manager.status_summary())


@tool(
    description="List background tasks with id, goal, status, priority and latest progress.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
def list_tasks(status: str = "") -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager
    from computer_agent.taskmgr.models import TaskStatus

    status_filter = TaskStatus(status) if status else None
    records = task_manager.list(status=status_filter)
    return ToolResult.ok(output=[r.summary() for r in records])


@tool(
    description="Show one task in detail: full progress log, result, timestamps. "
                "Accepts a full task id or unique prefix.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
def get_task(task_id: str) -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager

    record = task_manager.get(task_id)
    if not record:
        return ToolResult.fail(error=f"Task '{task_id}' not found")
    return ToolResult.ok(output=record.to_dict())


@tool(
    description="Submit a new background task to the queue. Higher priority runs first "
                "(default 5). Use this for long-running work so the conversation stays "
                "responsive.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def start_background_task(goal: str, priority: int = 5) -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager

    if not task_manager.is_running:
        return ToolResult.fail(
            error="Background worker is not running. Start the daemon: computer-agent daemon"
        )
    record = await task_manager.submit(goal=goal, priority=priority, source="chat")
    return ToolResult.ok(output={"task_id": record.id, "status": record.status.value})


@tool(
    description="Pause (hold) a task. A running task stops at its next safe point and "
                "waits; a queued task is held back. Resume later with resume_task.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def pause_task(task_id: str) -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager

    record = await task_manager.pause(task_id)
    if not record:
        return ToolResult.fail(error=f"Task '{task_id}' not found or already finished")
    return ToolResult.ok(output={"task_id": record.id, "status": record.status.value})


@tool(
    description="Resume a paused task.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def resume_task(task_id: str) -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager

    record = await task_manager.resume(task_id)
    if not record:
        return ToolResult.fail(error=f"Task '{task_id}' not found or not paused")
    return ToolResult.ok(output={"task_id": record.id, "status": record.status.value})


@tool(
    description="Terminate (cancel) a task permanently, including the currently "
                "running one.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def cancel_task(task_id: str) -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager

    record = await task_manager.cancel(task_id)
    if not record:
        return ToolResult.fail(error=f"Task '{task_id}' not found or already finished")
    return ToolResult.ok(output={"task_id": record.id, "status": record.status.value})


@tool(
    description="Change a task's priority (higher runs first).",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def set_task_priority(task_id: str, priority: int) -> ToolResult:
    from computer_agent.taskmgr.manager import task_manager

    record = await task_manager.set_priority(task_id, priority)
    if not record:
        return ToolResult.fail(error=f"Task '{task_id}' not found or already finished")
    return ToolResult.ok(output={"task_id": record.id, "priority": record.priority})


@tool(
    description="Change the agent's autonomy level: 'low' (ask before everything), "
                "'medium' (handle simple things, ask for major ones), or 'high' "
                "(handle most things). Only do this when the user explicitly asks.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
async def set_autonomy_mode(level: str) -> ToolResult:
    from computer_agent.abilities.autonomy import AutonomyLevel, autonomy_manager

    try:
        new_level = AutonomyLevel(level.lower())
    except ValueError:
        return ToolResult.fail(error=f"Invalid level '{level}'. Use low, medium, or high.")
    await autonomy_manager.set_level(new_level)
    return ToolResult.ok(output=f"Autonomy level set to {new_level.value}")


@tool(
    description="List pending approval requests (actions waiting for the user's "
                "permission).",
    risk_level=RiskLevel.LOW,
    category="agent",
)
def list_pending_approvals() -> ToolResult:
    from computer_agent.hitl.checkpoint import hitl_manager

    pending: list[dict[str, Any]] = [
        {
            "checkpoint_id": s.checkpoint_id,
            "tool": s.proposed_tool,
            "message": s.approval_message,
            "goal": s.goal,
        }
        for s in hitl_manager.get_pending_checkpoints()
    ]
    return ToolResult.ok(output=pending)


@tool(
    description="Approve a pending action on the user's behalf. ONLY call this when the "
                "user has explicitly said yes to that specific action in this "
                "conversation.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
def approve_pending_action(checkpoint_id: str, note: str = "") -> ToolResult:
    from computer_agent.hitl.checkpoint import hitl_manager

    if hitl_manager.resolve(checkpoint_id, approved=True, user_note=note):
        return ToolResult.ok(output=f"Approved checkpoint {checkpoint_id}")
    return ToolResult.fail(error=f"Checkpoint '{checkpoint_id}' not found")


@tool(
    description="Deny a pending action on the user's behalf. ONLY call this when the "
                "user has explicitly declined that specific action in this conversation.",
    risk_level=RiskLevel.LOW,
    category="agent",
)
def deny_pending_action(checkpoint_id: str, note: str = "") -> ToolResult:
    from computer_agent.hitl.checkpoint import hitl_manager

    if hitl_manager.resolve(checkpoint_id, approved=False, user_note=note):
        return ToolResult.ok(output=f"Denied checkpoint {checkpoint_id}")
    return ToolResult.fail(error=f"Checkpoint '{checkpoint_id}' not found")
