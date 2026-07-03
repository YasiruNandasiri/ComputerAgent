"""
Filesystem tools: read_file, write_file, list_directory, file_exists, delete_file.
All path operations are validated against the configured allowed paths.
"""

from __future__ import annotations

from pathlib import Path

from computer_agent.config import settings
from computer_agent.tools.base import RiskLevel, ToolResult, tool


def _validate_read_path(path: str) -> Path:
    """Resolve and validate that the path is within an allowed read directory."""
    resolved = Path(path).expanduser().resolve()
    for allowed in settings.allowed_read_paths:
        try:
            resolved.relative_to(allowed.resolve())
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Path '{resolved}' is outside allowed read paths: {settings.allowed_read_paths}"
    )


def _validate_write_path(path: str) -> Path:
    """Resolve and validate that the path is within an allowed write directory."""
    resolved = Path(path).expanduser().resolve()
    for allowed in settings.allowed_write_paths:
        try:
            resolved.relative_to(allowed.resolve())
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Path '{resolved}' is outside allowed write paths: {settings.allowed_write_paths}"
    )


@tool(name="read_file", risk_level=RiskLevel.LOW, category="fs",
      description="Read the contents of a text file from the filesystem.")
def read_file(path: str, max_chars: int = 50000) -> ToolResult:
    """
    Read a file and return its text content.

    path: Absolute or home-relative path to the file (e.g. ~/Documents/report.md)
    max_chars: Maximum characters to return (default 50000 to protect context window)
    """
    try:
        resolved = _validate_read_path(path)
        if not resolved.exists():
            return ToolResult.fail(error=f"File not found: {resolved}")
        if not resolved.is_file():
            return ToolResult.fail(error=f"Path is not a file: {resolved}")

        content = resolved.read_text(encoding="utf-8", errors="replace")
        truncated = len(content) > max_chars
        return ToolResult.ok(
            output=content[:max_chars],
            path=str(resolved),
            size_bytes=resolved.stat().st_size,
            truncated=truncated,
        )
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=f"Read error: {e}")


@tool(name="write_file", risk_level=RiskLevel.MEDIUM, category="fs",
      description="Write text content to a file, creating it if it does not exist.")
def write_file(path: str, content: str, overwrite: bool = True) -> ToolResult:
    """
    Write content to a file.

    path: Target file path (must be within allowed write paths)
    content: Text content to write
    overwrite: If False and file exists, return an error instead of overwriting
    """
    try:
        resolved = _validate_write_path(path)
        if resolved.exists() and not overwrite:
            return ToolResult.fail(error=f"File already exists: {resolved}. Set overwrite=True.")

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return ToolResult.ok(
            output=f"Written {len(content)} chars to {resolved}",
            path=str(resolved),
            bytes_written=len(content.encode()),
        )
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=f"Write error: {e}")


@tool(name="append_file", risk_level=RiskLevel.MEDIUM, category="fs",
      description="Append text to the end of an existing file.")
def append_file(path: str, content: str) -> ToolResult:
    """
    Append content to an existing file.

    path: Target file path
    content: Text to append
    """
    try:
        resolved = _validate_write_path(path)
        with resolved.open("a", encoding="utf-8") as f:
            f.write(content)
        return ToolResult.ok(output=f"Appended {len(content)} chars to {resolved}")
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=f"Append error: {e}")


@tool(name="list_directory", risk_level=RiskLevel.LOW, category="fs",
      description="List files and subdirectories in a given directory path.")
def list_directory(path: str, show_hidden: bool = False) -> ToolResult:
    """
    List contents of a directory.

    path: Directory path to list
    show_hidden: Whether to include hidden files/directories (starting with .)
    """
    try:
        resolved = _validate_read_path(path)
        if not resolved.is_dir():
            return ToolResult.fail(error=f"Not a directory: {resolved}")

        entries = []
        for item in sorted(resolved.iterdir()):
            if not show_hidden and item.name.startswith("."):
                continue
            stat = item.stat()
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size_bytes": stat.st_size if item.is_file() else None,
                "modified": stat.st_mtime,
            })

        return ToolResult.ok(output=entries, path=str(resolved), count=len(entries))
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="file_exists", risk_level=RiskLevel.LOW, category="fs",
      description="Check whether a file or directory exists at the given path.")
def file_exists(path: str) -> ToolResult:
    """
    Check if a path exists.

    path: File or directory path to check
    """
    try:
        resolved = _validate_read_path(path)
        exists = resolved.exists()
        return ToolResult.ok(
            output=exists,
            path=str(resolved),
            is_file=resolved.is_file() if exists else None,
            is_dir=resolved.is_dir() if exists else None,
        )
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="delete_file", risk_level=RiskLevel.HIGH, category="fs",
      description="Delete a file. This action is irreversible. Requires HITL approval.")
def delete_file(path: str) -> ToolResult:
    """
    Delete a file permanently.

    path: Path of the file to delete
    """
    try:
        resolved = _validate_write_path(path)
        if not resolved.exists():
            return ToolResult.fail(error=f"File not found: {resolved}")
        if resolved.is_dir():
            return ToolResult.fail(error=f"Use delete_directory for directories: {resolved}")

        resolved.unlink()
        return ToolResult.ok(output=f"Deleted: {resolved}")
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="search_files", risk_level=RiskLevel.LOW, category="fs",
      description="Search for files matching a name pattern within a directory.")
def search_files(directory: str, pattern: str, max_results: int = 50) -> ToolResult:
    """
    Recursively search for files matching a glob pattern.

    directory: Root directory to search in
    pattern: Glob pattern to match (e.g. '*.pdf', 'report_*.csv')
    max_results: Maximum number of results to return
    """
    try:
        resolved = _validate_read_path(directory)
        if not resolved.is_dir():
            return ToolResult.fail(error=f"Not a directory: {resolved}")

        matches = []
        for match in resolved.rglob(pattern):
            if len(matches) >= max_results:
                break
            matches.append({
                "path": str(match),
                "name": match.name,
                "size_bytes": match.stat().st_size if match.is_file() else None,
            })

        return ToolResult.ok(
            output=matches,
            pattern=pattern,
            count=len(matches),
            truncated=len(matches) >= max_results,
        )
    except PermissionError as e:
        return ToolResult.fail(error=str(e))
    except Exception as e:
        return ToolResult.fail(error=str(e))
