"""
Cooperative task control — pause / resume / cancel signals checked by the
Coordinator between agent turns and before each tool invocation.
"""

from __future__ import annotations

import asyncio


class TaskCancelledError(Exception):
    """Raised inside the agent loop when the user terminates the task."""


class TaskControl:
    """
    Shared handle between the TaskManager (which flips the signals) and the
    Coordinator (which awaits checkpoint() at safe points).
    """

    def __init__(self) -> None:
        self._resume = asyncio.Event()
        self._resume.set()  # running by default
        self._cancelled = False

    @property
    def is_paused(self) -> bool:
        return not self._resume.is_set() and not self._cancelled

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._resume.set()  # unblock a paused waiter so it can observe the cancel

    async def checkpoint(self) -> None:
        """
        Await this at safe points. Raises TaskCancelledError if the task was
        terminated; blocks while the task is paused.
        """
        if self._cancelled:
            raise TaskCancelledError("Task cancelled by user")
        if not self._resume.is_set():
            await self._resume.wait()
            if self._cancelled:
                raise TaskCancelledError("Task cancelled by user")
