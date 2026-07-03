# Computer Agent

A modular, semi-autonomous Computer Controlling Agent for personal productivity. Controls your Mac/Linux desktop and web browser via a hybrid Vision+API execution model, with strict Human-in-the-Loop safety gates.

## Quick Start

```bash
# 0. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Copy environment config and add your ANTHROPIC_API_KEY
cp .env.example .env

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Start the database (PostgreSQL + pgvector)
docker-compose up -d

# 4. Run a single task
uv run computer-agent run "Take a screenshot and tell me what's on my screen"

# 5. Interactive chat mode
uv run computer-agent chat
```

## The Daemon (recommended)

Run the agent as a long-lived background process — it executes queued and
scheduled tasks, sends notifications, and lets you chat while it works:

```bash
uv run computer-agent daemon         # Terminal 1: worker + scheduler + API
uv run computer-agent chat           # Terminal 2: chat while tasks run
```

In chat you can say things like `status`, `mode high`, `approve` / `deny`
(when an action is waiting), or "pause that task" / "what are you working on?".

## Common Commands

```bash
uv run computer-agent mode [low|medium|high]   # Show/set autonomy level
uv run computer-agent run "..." [-b]           # One-off task (-b queues on the daemon)

uv run computer-agent task submit "..."        # Queue a background task
uv run computer-agent task list|show|status    # Inspect tasks & progress
uv run computer-agent task pause|resume|cancel <id>
uv run computer-agent task priority <id> <n>   # Higher runs first

uv run computer-agent routine add "Check my email for urgent messages" -c "*/30 * * * *"
uv run computer-agent routine list|remove|enable|disable <name>

uv run computer-agent pending        # List pending HITL approval requests
uv run computer-agent approve <id>   # Approve a HITL checkpoint
uv run computer-agent deny <id>      # Deny a HITL checkpoint
uv run computer-agent tools          # List all registered tools
uv run computer-agent skills         # List all loaded skills
uv run computer-agent version        # Print version/config info

uv run pytest                        # Run tests
uv run ruff check .                  # Lint
uv run mypy computer_agent/          # Type-check
```

## Autonomy Levels

| Level | Behavior |
|-------|----------|
| `low` | Asks before anything with side effects |
| `medium` (default) | Handles simple reversible actions, asks for major ones |
| `high` | Handles most things; only critical/flagged actions ask |

Rules marked `always_hitl` (deleting files, sending email/messages,
credential access) always require approval regardless of level. Financial
operations are always blocked.

## Adding Skills

Drop a `skill.yaml` file into any subdirectory of `skills/` and it will be
auto-loaded on the next run. See `skills/web_research/skill.yaml` for an example.

## Architecture

See the architecture documentation in the project for full details.
