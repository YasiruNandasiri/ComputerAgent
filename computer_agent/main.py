"""
Computer Agent — CLI entry point.

Commands:
  computer-agent run "Take a screenshot and describe what you see"
  computer-agent chat                  # Interactive REPL mode
  computer-agent approve <checkpoint>  # Approve a pending HITL request
  computer-agent deny   <checkpoint>   # Deny a pending HITL request
  computer-agent pending               # List pending HITL requests
  computer-agent tools                 # List all registered tools
  computer-agent skills                # List all loaded skills
  computer-agent version               # Print version info
"""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="computer-agent",
    help="Computer Agent — AI executive assistant that controls your computer",
    add_completion=False,
)

console = Console()


def _bootstrap() -> None:
    """Initialize all singletons (registry, skills, memory)."""
    from computer_agent.skills.loader import skill_registry
    from computer_agent.tools.registry import registry

    registry.discover()
    skill_registry.discover()
    skill_registry.register_with_tool_registry()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    request: str = typer.Argument(..., help="The task for the agent to perform"),
    session_id: str = typer.Option(None, "--session", "-s", help="Session ID to resume"),
    background: bool = typer.Option(
        False, "--background", "-b",
        help="Queue on the daemon and return immediately (requires the daemon)",
    ),
    priority: int = typer.Option(5, "--priority", "-p", help="Priority for --background"),
):
    """Run the agent on a single task (in-process, or queued with --background)."""
    if background:
        result = _daemon_client().submit_task(request, priority=priority)
        console.print(
            f"[green]✓ Task queued:[/green] {result['id'][:8]} — follow it with "
            f"[bold]computer-agent task show {result['id'][:8]}[/bold] or in chat."
        )
        return
    _bootstrap()
    asyncio.run(_run_task(request, session_id))


async def _run_task(request: str, session_id: str | None) -> None:
    from computer_agent.abilities.autonomy import autonomy_manager
    from computer_agent.coordinator import Coordinator
    from computer_agent.memory.store import memory_store

    await memory_store.connect()
    await autonomy_manager.load()

    coordinator = Coordinator(session_id=session_id)

    console.print(Panel(
        f"[bold cyan]Task:[/bold cyan] {request}",
        title="[bold]Computer Agent[/bold]",
        border_style="blue",
    ))

    try:
        response = await coordinator.run(request)
        console.print(Panel(
            Text(response, style="green"),
            title="[bold]Result[/bold]",
            border_style="green",
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Task interrupted by user.[/yellow]")
    finally:
        await memory_store.disconnect()


@app.command()
def chat(
    session_id: str = typer.Option(None, "--session", "-s", help="Session ID to continue"),
):
    """
    Chat with the agent. When the daemon is running, tasks execute in the
    background and you can keep talking (ask progress, approve actions, pause
    or cancel tasks). Without the daemon, falls back to a blocking REPL.
    """
    from computer_agent.daemon.client import DaemonClient

    client = DaemonClient()
    if client.is_running():
        _chat_via_daemon(client, session_id)
    else:
        console.print(
            "[dim]Daemon not running — using in-process mode "
            "(no background tasks; start the daemon for the full experience).[/dim]"
        )
        _bootstrap()
        asyncio.run(_chat_session(session_id))


def _chat_via_daemon(client, session_id: str | None) -> None:
    """Interactive chat against the daemon, with a live event stream."""
    import threading

    console.print(Panel(
        "[bold]Computer Agent — Interactive Mode (daemon)[/bold]\n"
        "Background tasks keep running while you chat. Try: "
        "[bold]status[/bold], [bold]mode high[/bold], or just describe a task.\n"
        "Type [bold]exit[/bold] to leave (the daemon keeps working).",
        border_style="blue",
    ))

    stop = threading.Event()

    def _event_listener() -> None:
        while not stop.is_set():
            try:
                for event in client.stream_events():
                    if stop.is_set():
                        return
                    _print_daemon_event(event)
            except Exception:
                if stop.is_set():
                    return
                stop.wait(2.0)  # reconnect after transient errors

    listener = threading.Thread(target=_event_listener, daemon=True)
    listener.start()

    try:
        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye — the daemon keeps running.[/yellow]")
                break
            try:
                result = client.chat(user_input, session_id)
                session_id = result["session_id"]
                console.print(f"\n[bold green]Agent:[/bold green] {result['response']}")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
    finally:
        stop.set()


def _print_daemon_event(event: dict) -> None:
    """Render a daemon SSE event as a live status line."""
    etype = event.get("type", "")
    data = event.get("data", {})
    task_id = data.get("task_id")
    task_ref = f"task {str(task_id)[:8]}" if task_id else "task"

    if etype == "hitl.approval.requested":
        console.print(
            f"\n[bold yellow]⚠ Approval needed[/bold yellow] ({task_ref}): "
            f"{data.get('message') or data.get('tool')}\n"
            f"[dim]Reply 'approve' or 'deny' — or: computer-agent approve "
            f"{str(data.get('checkpoint_id'))[:8]}...[/dim]"
        )
    elif etype == "task.completed" and task_id:
        console.print(
            f"\n[green]✓ Background {task_ref} finished:[/green] "
            f"{(data.get('goal') or '')[:60]}"
        )
    elif etype == "task.failed" and task_id:
        console.print(
            f"\n[red]✗ Background {task_ref} failed:[/red] {(data.get('error') or '')[:80]}"
        )
    elif etype == "task.paused":
        console.print(f"\n[yellow]⏸ {task_ref} paused[/yellow]")
    elif etype == "task.resumed":
        console.print(f"\n[green]▶ {task_ref} resumed[/green]")
    elif etype == "task.step.completed" and task_id:
        tools = ", ".join(data.get("tools", []))
        console.print(f"[dim]  {task_ref} · turn {data.get('turn')}: {tools}[/dim]")


async def _chat_session(session_id: str | None) -> None:
    from computer_agent.abilities.autonomy import autonomy_manager
    from computer_agent.coordinator import Coordinator
    from computer_agent.memory.store import memory_store

    await memory_store.connect()
    await autonomy_manager.load()
    coordinator = Coordinator(session_id=session_id)

    console.print(Panel(
        "[bold]Computer Agent — Interactive Mode[/bold]\n"
        "Type your request and press Enter. Type [bold]exit[/bold] or [bold]quit[/bold] to stop.",
        border_style="blue",
    ))
    console.print(f"[dim]Session: {coordinator.session_id}[/dim]")

    try:
        while True:
            try:
                user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye.[/yellow]")
                break

            console.print("[dim]Thinking...[/dim]")
            try:
                response = await coordinator.run(user_input)
                console.print(f"\n[bold green]Agent:[/bold green] {response}")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
    finally:
        await memory_store.disconnect()


def _daemon_client():
    """Return a connected DaemonClient or exit with a helpful error."""
    from computer_agent.daemon.client import DaemonClient

    client = DaemonClient()
    if not client.is_running():
        console.print(
            "[red]The daemon is not running.[/red] Approvals and task management "
            "need the long-lived agent process.\n"
            "Start it with: [bold]computer-agent daemon[/bold]"
        )
        raise typer.Exit(code=1)
    return client


@app.command()
def daemon(
    host: str = typer.Option(None, "--host", help="Bind host (default from config)"),
    port: int = typer.Option(None, "--port", help="Bind port (default from config)"),
):
    """Run the agent daemon: task worker, scheduler, notifications, and HTTP API."""
    import uvicorn

    from computer_agent.config import settings
    from computer_agent.daemon.app import create_app

    console.print(Panel(
        "[bold]Computer Agent — Daemon[/bold]\n"
        f"API: http://{host or settings.daemon_host}:{port or settings.daemon_port}\n"
        f"Web UI: http://{host or settings.daemon_host}:{port or settings.daemon_port}/\n"
        "Interact via: [bold]computer-agent chat[/bold] / [bold]task[/bold] / "
        "[bold]routine[/bold] / [bold]approve[/bold]",
        border_style="blue",
    ))
    uvicorn.run(
        create_app(),
        host=host or settings.daemon_host,
        port=port or settings.daemon_port,
        log_level="warning",
    )


@app.command()
def approve(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID to approve"),
    note: str = typer.Option("", "--note", "-n", help="Optional approval note"),
):
    """Approve a pending HITL checkpoint (requires the daemon)."""
    client = _daemon_client()
    try:
        client.resolve(checkpoint_id, approved=True, note=note)
        console.print(f"[green]✓ Approved checkpoint:[/green] {checkpoint_id}")
    except RuntimeError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def deny(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID to deny"),
    note: str = typer.Option("", "--note", "-n", help="Reason for denial"),
):
    """Deny a pending HITL checkpoint (requires the daemon)."""
    client = _daemon_client()
    try:
        client.resolve(checkpoint_id, approved=False, note=note)
        console.print(f"[yellow]✗ Denied checkpoint:[/yellow] {checkpoint_id}")
    except RuntimeError as e:
        console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def pending():
    """List all pending HITL approval requests (requires the daemon)."""
    client = _daemon_client()
    checkpoints = client.pending()
    if not checkpoints:
        console.print("[green]No pending approval requests.[/green]")
        return

    table = Table(title="Pending HITL Approvals", show_header=True)
    table.add_column("Checkpoint ID", style="cyan", no_wrap=True)
    table.add_column("Session", style="dim")
    table.add_column("Tool")
    table.add_column("Message")

    for cp in checkpoints:
        message = cp["message"]
        table.add_row(
            cp["checkpoint_id"][:8] + "...",
            cp["session_id"][:8] + "...",
            cp["tool"],
            message[:60] + "..." if len(message) > 60 else message,
        )

    console.print(table)


@app.command()
def tools():
    """List all registered tools."""
    _bootstrap()
    from computer_agent.tools.registry import registry

    all_tools = registry.get_all_tools()

    table = Table(title=f"Registered Tools ({len(all_tools)})", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Risk", style="red")
    table.add_column("Description")

    for t in sorted(all_tools, key=lambda x: (x.category, x.name)):
        risk_color = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }.get(t.risk_level.value, "white")

        table.add_row(
            t.name,
            t.category,
            f"[{risk_color}]{t.risk_level.value}[/{risk_color}]",
            t.description[:70],
        )

    console.print(table)


@app.command()
def skills():
    """List all loaded skills."""
    _bootstrap()
    from computer_agent.skills.loader import skill_registry

    skill_names = skill_registry.list_skills()
    if not skill_names:
        console.print("[yellow]No skills loaded. Drop skill.yaml files into the skills/ directory.[/yellow]")
        return

    table = Table(title=f"Loaded Skills ({len(skill_names)})", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="dim")
    table.add_column("Description")

    for name in skill_names:
        manifest = skill_registry.get_skill(name)
        if manifest:
            table.add_row(manifest.name, manifest.version, manifest.description)

    console.print(table)


_MODE_DESCRIPTIONS = {
    "low": "ask before anything with side effects",
    "medium": "handle simple reversible actions, ask for major ones",
    "high": "handle most things, ask only for critical actions",
}


@app.command()
def mode(
    level: str = typer.Argument(None, help="New autonomy level: low | medium | high"),
):
    """Show or set the agent's autonomy level (low / medium / high)."""
    if level is not None and level.lower() not in _MODE_DESCRIPTIONS:
        console.print(f"[red]Invalid level '{level}'. Use: low, medium, or high.[/red]")
        raise typer.Exit(code=1)

    # Prefer the daemon so the running agent picks the change up immediately
    from computer_agent.daemon.client import DaemonClient
    client = DaemonClient()
    if client.is_running():
        current = client.set_mode(level.lower()) if level else client.get_mode()
    else:
        current = asyncio.run(_mode_local(level))

    if level:
        console.print(f"[green]✓ Autonomy level set to:[/green] [bold]{current}[/bold]")
    else:
        console.print(
            f"Autonomy level: [bold cyan]{current}[/bold cyan] "
            f"— {_MODE_DESCRIPTIONS[current]}"
        )


async def _mode_local(level: str | None) -> str:
    from computer_agent.abilities.autonomy import autonomy_manager
    from computer_agent.memory.store import memory_store

    await memory_store.connect()
    try:
        await autonomy_manager.load()
        if level:
            await autonomy_manager.set_level(level.lower())
        return autonomy_manager.level.value
    finally:
        await memory_store.disconnect()


# ---------------------------------------------------------------------------
# Task management (requires the daemon)
# ---------------------------------------------------------------------------

task_app = typer.Typer(name="task", help="Manage background tasks (requires the daemon).")
app.add_typer(task_app)


def _print_task_rows(tasks: list) -> None:
    table = Table(show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Pri", justify="right")
    table.add_column("Status", style="yellow")
    table.add_column("Source", style="dim")
    table.add_column("Goal")
    table.add_column("Last progress", style="dim")
    for t in tasks:
        table.add_row(
            t["id"][:8],
            str(t["priority"]),
            t["status"],
            t["source"],
            t["goal"][:50],
            (t.get("last_progress") or "")[:40],
        )
    console.print(table)


@task_app.command("submit")
def task_submit(
    goal: str = typer.Argument(..., help="What the agent should do"),
    priority: int = typer.Option(5, "--priority", "-p", help="Higher runs first"),
):
    """Queue a new background task."""
    result = _daemon_client().submit_task(goal, priority=priority)
    console.print(f"[green]✓ Task queued:[/green] {result['id'][:8]} — {result['goal'][:60]}")


@task_app.command("list")
def task_list(
    status: str = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """List background tasks."""
    tasks = _daemon_client().list_tasks(status=status)
    if not tasks:
        console.print("[dim]No tasks.[/dim]")
        return
    _print_task_rows(tasks)


@task_app.command("show")
def task_show(task_id: str = typer.Argument(..., help="Task ID (or unique prefix)")):
    """Show a task's full detail and progress log."""
    t = _daemon_client().get_task(task_id)
    console.print(Panel(
        f"[bold]Goal:[/bold] {t['goal']}\n"
        f"[bold]Status:[/bold] {t['status']}   [bold]Priority:[/bold] {t['priority']}   "
        f"[bold]Source:[/bold] {t['source']}\n"
        f"[bold]Created:[/bold] {t['created_at']}\n"
        f"[bold]Started:[/bold] {t.get('started_at') or '—'}   "
        f"[bold]Finished:[/bold] {t.get('finished_at') or '—'}\n"
        + (f"[bold]Result:[/bold] {t['result']}\n" if t.get("result") else "")
        + (f"[bold red]Error:[/bold red] {t['error']}\n" if t.get("error") else ""),
        title=f"Task {t['id'][:8]}",
        border_style="blue",
    ))
    if t.get("progress"):
        console.print("[bold]Progress:[/bold]")
        for note in t["progress"]:
            console.print(f"  • {note}")


@task_app.command("pause")
def task_pause(task_id: str):
    """Pause (hold) a task — including the currently running one."""
    t = _daemon_client().pause_task(task_id)
    console.print(f"[yellow]⏸ Paused:[/yellow] {t['id'][:8]} — {t['goal'][:60]}")


@task_app.command("resume")
def task_resume(task_id: str):
    """Resume a paused task."""
    t = _daemon_client().resume_task(task_id)
    console.print(f"[green]▶ Resumed:[/green] {t['id'][:8]} — {t['goal'][:60]}")


@task_app.command("cancel")
def task_cancel(task_id: str):
    """Terminate a task permanently."""
    t = _daemon_client().cancel_task(task_id)
    console.print(f"[red]✗ Cancelled:[/red] {t['id'][:8]} — {t['goal'][:60]}")


@task_app.command("priority")
def task_priority(task_id: str, priority: int = typer.Argument(..., help="Higher runs first")):
    """Change a task's priority."""
    t = _daemon_client().set_task_priority(task_id, priority)
    console.print(f"[green]✓ Priority set to {t['priority']}:[/green] {t['id'][:8]}")


@task_app.command("status")
def task_status():
    """Show what the agent is focusing on right now."""
    status = _daemon_client().status()
    tasks = status["tasks"]
    current = tasks.get("current")
    console.print(f"Autonomy: [bold cyan]{status['autonomy_level']}[/bold cyan]")
    if current:
        console.print(
            f"[bold green]Working on:[/bold green] {current['goal']} "
            f"(task {current['id'][:8]})"
        )
        if current.get("last_progress"):
            console.print(f"  [dim]{current['last_progress']}[/dim]")
    else:
        console.print("[dim]No task running.[/dim]")
    for r in tasks.get("awaiting_approval", []):
        console.print(f"[yellow]Awaiting approval:[/yellow] {r['goal']} (task {r['id'][:8]})")
    for r in tasks.get("paused", []):
        console.print(f"[dim]Paused: {r['goal']} (task {r['id'][:8]})[/dim]")
    queued = tasks.get("queued", [])
    if queued:
        console.print(f"[dim]{len(queued)} task(s) queued.[/dim]")


# ---------------------------------------------------------------------------
# Scheduled routines (requires the daemon)
# ---------------------------------------------------------------------------

routine_app = typer.Typer(name="routine", help="Recurring background routines (cron).")
app.add_typer(routine_app)


@routine_app.command("add")
def routine_add(
    goal: str = typer.Argument(..., help="What the agent should do on each run"),
    cron: str = typer.Option(..., "--cron", "-c", help='Cron expression, e.g. "*/30 * * * *"'),
    name: str = typer.Option(None, "--name", "-n", help="Routine name (default: derived)"),
    priority: int = typer.Option(5, "--priority", "-p"),
    no_notify: bool = typer.Option(False, "--no-notify", help="Skip completion notification"),
):
    """Add a recurring routine, e.g. check email every 30 minutes."""
    routine_name = name or "-".join(goal.lower().split()[:4])
    r = _daemon_client().add_routine(
        name=routine_name, cron=cron, goal=goal, priority=priority, notify=not no_notify
    )
    console.print(f"[green]✓ Routine added:[/green] {r['name']} ({r['cron']})")


@routine_app.command("list")
def routine_list():
    """List all scheduled routines."""
    routines = _daemon_client().list_routines()
    if not routines:
        console.print("[dim]No routines. Add one with: computer-agent routine add[/dim]")
        return
    table = Table(show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Cron")
    table.add_column("Enabled")
    table.add_column("Pri", justify="right")
    table.add_column("Goal")
    for r in routines:
        table.add_row(
            r["name"],
            r["cron"],
            "[green]yes[/green]" if r["enabled"] else "[dim]no[/dim]",
            str(r["priority"]),
            r["goal"][:50],
        )
    console.print(table)


@routine_app.command("remove")
def routine_remove(name: str):
    """Remove a routine."""
    _daemon_client().remove_routine(name)
    console.print(f"[green]✓ Removed routine:[/green] {name}")


@routine_app.command("enable")
def routine_enable(name: str):
    """Enable a routine."""
    _daemon_client().set_routine_enabled(name, True)
    console.print(f"[green]✓ Enabled:[/green] {name}")


@routine_app.command("disable")
def routine_disable(name: str):
    """Disable a routine (kept, but won't fire)."""
    _daemon_client().set_routine_enabled(name, False)
    console.print(f"[yellow]✓ Disabled:[/yellow] {name}")


@app.command()
def version():
    """Print version and configuration info."""
    import platform

    from computer_agent.config import settings

    console.print(Panel(
        f"[bold]Computer Agent[/bold] v0.1.0\n\n"
        f"Platform    : {platform.system()} {platform.release()}\n"
        f"Python      : {sys.version.split()[0]}\n"
        f"Model       : {settings.primary_model}\n"
        f"DB          : {settings.database_url.split('@')[-1]}\n"
        f"Browser     : {settings.browser_type}",
        border_style="blue",
    ))


if __name__ == "__main__":
    app()
