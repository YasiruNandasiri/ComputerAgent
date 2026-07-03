"""
Notifier — outbound notification service with pluggable channels.

Built-in channels: macOS (osascript) and terminal. External packages can
contribute channels via the entry-point group "computer_agent.notifiers":

    [project.entry-points."computer_agent.notifiers"]
    slack = "my_package.notify:SlackChannel"

where the entry point resolves to a class with an async
`send(title, message, subtitle, sound)` method.
"""

from __future__ import annotations

import asyncio
import platform
from typing import Protocol

from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


class NotificationChannel(Protocol):
    async def send(self, title: str, message: str, subtitle: str, sound: bool) -> None: ...


class MacOSChannel:
    """Native macOS notification via osascript."""

    async def send(self, title: str, message: str, subtitle: str = "", sound: bool = True) -> None:
        def _esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"')

        script = f'display notification "{_esc(message[:200])}" with title "{_esc(title)}"'
        if subtitle:
            script += f' subtitle "{_esc(subtitle)}"'
        if sound:
            script += ' sound name "Glass"'

        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()


class TerminalChannel:
    """Plain terminal print fallback."""

    async def send(self, title: str, message: str, subtitle: str = "", sound: bool = True) -> None:
        line = f"🔔 [{title}] {subtitle + ': ' if subtitle else ''}{message}"
        print(line, flush=True)


class Notifier:
    """Dispatches a notification to all configured channels."""

    def __init__(self) -> None:
        self._channels: list[NotificationChannel] = []
        if platform.system() == "Darwin":
            self._channels.append(MacOSChannel())
        self._channels.append(TerminalChannel())
        self._load_entry_point_channels()

    def _load_entry_point_channels(self) -> None:
        try:
            from importlib.metadata import entry_points
            for ep in entry_points(group="computer_agent.notifiers"):
                try:
                    channel_cls = ep.load()
                    self._channels.append(channel_cls())
                    logger.info("notifier_channel_loaded", name=ep.name)
                except Exception as e:
                    logger.warning("notifier_channel_failed", name=ep.name, error=str(e))
        except Exception as e:
            logger.debug("entry_point_scan_failed", group="computer_agent.notifiers", error=str(e))

    def add_channel(self, channel: NotificationChannel) -> None:
        self._channels.append(channel)

    async def notify(
        self, title: str, message: str, subtitle: str = "", sound: bool = True
    ) -> None:
        for channel in self._channels:
            try:
                await channel.send(title, message, subtitle, sound)
            except Exception as e:
                logger.debug("notify_channel_error", channel=type(channel).__name__, error=str(e))


# Module-level singleton
notifier = Notifier()
