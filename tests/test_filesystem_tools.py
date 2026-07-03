"""
Tests for filesystem tools — path validation and read/write operations.
"""

from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture
def allowed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch settings to allow read/write within tmp_path."""
    test_dir = tmp_path / "workspace"
    test_dir.mkdir()

    from computer_agent import config as cfg_module
    original_settings = cfg_module.settings

    # Patch allowed paths
    monkeypatch.setattr(cfg_module.settings, "allowed_read_paths_raw", str(test_dir))
    monkeypatch.setattr(cfg_module.settings, "allowed_write_paths_raw", str(test_dir))

    return test_dir


class TestFilesystemTools:
    def test_write_and_read_file(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import write_file, read_file

        target = str(allowed_dir / "hello.txt")
        write_result = write_file(path=target, content="Hello, World!")
        assert write_result.success, write_result.error

        read_result = read_file(path=target)
        assert read_result.success
        assert read_result.output == "Hello, World!"

    def test_read_nonexistent_file(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import read_file

        result = read_file(path=str(allowed_dir / "ghost.txt"))
        assert not result.success
        assert "not found" in result.error.lower()

    def test_list_directory(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import write_file, list_directory

        write_file(path=str(allowed_dir / "a.txt"), content="a")
        write_file(path=str(allowed_dir / "b.txt"), content="b")

        result = list_directory(path=str(allowed_dir))
        assert result.success
        names = [e["name"] for e in result.output]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_path_traversal_blocked(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import read_file

        # Attempt to escape the allowed directory
        result = read_file(path=str(allowed_dir / ".." / ".." / "etc" / "passwd"))
        assert not result.success
        assert "outside allowed" in result.error.lower()

    def test_file_exists_true(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import write_file, file_exists

        target = str(allowed_dir / "exists.txt")
        write_file(path=target, content="yes")

        result = file_exists(path=target)
        assert result.success
        assert result.output is True

    def test_file_exists_false(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import file_exists

        result = file_exists(path=str(allowed_dir / "nope.txt"))
        assert result.success
        assert result.output is False

    def test_no_overwrite_protection(self, allowed_dir: Path) -> None:
        from computer_agent.tools.filesystem.operations import write_file

        target = str(allowed_dir / "locked.txt")
        write_file(path=target, content="original")

        result = write_file(path=target, content="new", overwrite=False)
        assert not result.success
        assert "already exists" in result.error.lower()
