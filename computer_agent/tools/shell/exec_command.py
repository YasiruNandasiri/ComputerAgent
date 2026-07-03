"""
Shell execution tool.
Commands are validated against an allowlist and run in a subprocess with a timeout.
Output is captured and returned as a ToolResult.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from computer_agent.config import settings
from computer_agent.tools.base import RiskLevel, ToolResult, tool


def _validate_command(command: str) -> None:
    """Ensure the base command is in the allowlist."""
    base_cmd = shlex.split(command)[0] if command.strip() else ""
    # Strip path prefix (e.g. /usr/bin/ls → ls)
    base_cmd = Path(base_cmd).name

    if base_cmd not in settings.allowed_shell_commands:
        raise PermissionError(
            f"Command '{base_cmd}' is not in the allowed shell commands list. "
            f"Add it to ALLOWED_SHELL_COMMANDS in your .env to permit it."
        )


@tool(name="run_shell_command", risk_level=RiskLevel.MEDIUM, category="shell",
      description="Execute a shell command and return its stdout, stderr, and exit code.")
def run_shell_command(
    command: str,
    working_directory: str = "~",
    timeout: int = 30,
) -> ToolResult:
    """
    Run a shell command in a subprocess.

    command: The shell command to execute (must be in allowed commands list)
    working_directory: Directory to run the command in (default: home directory)
    timeout: Maximum seconds to wait for the command to complete
    """
    try:
        _validate_command(command)
    except PermissionError as e:
        return ToolResult.fail(error=str(e))

    cwd = Path(working_directory).expanduser().resolve()
    if not cwd.exists():
        return ToolResult.fail(error=f"Working directory not found: {cwd}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            env=_safe_env(),
        )
        return ToolResult.ok(
            output={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            },
            command=command,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.fail(error=f"Command timed out after {timeout}s: {command}")
    except Exception as e:
        return ToolResult.fail(error=f"Command execution failed: {e}")


@tool(name="run_python_snippet", risk_level=RiskLevel.MEDIUM, category="shell",
      description="Execute a small Python code snippet and return the output.")
def run_python_snippet(code: str, timeout: int = 30) -> ToolResult:
    """
    Run a Python code snippet in a subprocess.

    code: Python code to execute
    timeout: Maximum seconds to wait
    """
    # Validate that python is allowed
    if "python" not in settings.allowed_shell_commands and "python3" not in settings.allowed_shell_commands:
        return ToolResult.fail(error="Python execution is not in the allowed commands list.")

    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_safe_env(),
        )
        return ToolResult.ok(
            output={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        return ToolResult.fail(error=f"Python snippet timed out after {timeout}s")
    except Exception as e:
        return ToolResult.fail(error=str(e))


def _safe_env() -> dict[str, str]:
    """
    Return a sanitized environment for subprocess execution.
    Strips secrets that might be in the parent environment.
    """
    import os
    env = dict(os.environ)
    # Remove common secret keys from the subprocess environment
    for key in list(env.keys()):
        lower = key.lower()
        if any(s in lower for s in ("api_key", "secret", "password", "token", "credential")):
            del env[key]
    return env
