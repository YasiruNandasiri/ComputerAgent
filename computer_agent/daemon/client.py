"""
DaemonClient — thin synchronous HTTP client the CLI uses to talk to the
running daemon (httpx).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from computer_agent.config import settings


class DaemonNotRunningError(RuntimeError):
    pass


class DaemonClient:
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self._base = f"http://{host or settings.daemon_host}:{port or settings.daemon_port}"
        self._client = httpx.Client(base_url=self._base, timeout=httpx.Timeout(600.0, connect=2.0))

    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        try:
            return self._client.get("/status").status_code == 200
        except httpx.HTTPError:
            return False

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as e:
            raise DaemonNotRunningError(
                "Daemon is not running. Start it with: computer-agent daemon"
            ) from e
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(detail)
        return response.json()

    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def chat(self, message: str, session_id: str | None = None) -> dict[str, str]:
        return self._request(
            "POST", "/chat", json={"message": message, "session_id": session_id}
        )

    def get_mode(self) -> str:
        return self._request("GET", "/mode")["level"]

    def set_mode(self, level: str) -> str:
        return self._request("PUT", "/mode", json={"level": level})["level"]

    # Tasks -------------------------------------------------------------

    def submit_task(self, goal: str, priority: int = 5) -> dict[str, Any]:
        return self._request("POST", "/tasks", json={"goal": goal, "priority": priority})

    def list_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        params = {"status": status} if status else None
        return self._request("GET", "/tasks", params=params)

    def get_task(self, task_id: str) -> dict[str, Any]:
        return self._request("GET", f"/tasks/{task_id}")

    def pause_task(self, task_id: str) -> dict[str, Any]:
        return self._request("POST", f"/tasks/{task_id}/pause")

    def resume_task(self, task_id: str) -> dict[str, Any]:
        return self._request("POST", f"/tasks/{task_id}/resume")

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        return self._request("POST", f"/tasks/{task_id}/cancel")

    def set_task_priority(self, task_id: str, priority: int) -> dict[str, Any]:
        return self._request("PATCH", f"/tasks/{task_id}", json={"priority": priority})

    # HITL ---------------------------------------------------------------

    def pending(self) -> list[dict[str, Any]]:
        return self._request("GET", "/hitl/pending")

    def resolve(self, checkpoint_id: str, approved: bool, note: str = "") -> dict[str, Any]:
        return self._request(
            "POST", f"/hitl/{checkpoint_id}/resolve",
            json={"approved": approved, "note": note},
        )

    # Routines ------------------------------------------------------------

    def list_routines(self) -> list[dict[str, Any]]:
        return self._request("GET", "/routines")

    def add_routine(
        self, name: str, cron: str, goal: str, priority: int = 5, notify: bool = True
    ) -> dict[str, Any]:
        return self._request("POST", "/routines", json={
            "name": name, "cron": cron, "goal": goal,
            "priority": priority, "notify": notify,
        })

    def remove_routine(self, name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/routines/{name}")

    def set_routine_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        action = "enable" if enabled else "disable"
        return self._request("POST", f"/routines/{name}/{action}")

    # Events ----------------------------------------------------------------

    def stream_events(self) -> Iterator[dict[str, Any]]:
        """Yield decoded SSE events from the daemon (blocking generator)."""
        with self._client.stream("GET", "/events", timeout=None) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
