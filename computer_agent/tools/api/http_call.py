"""
API tools: http_get, http_post, http_put, http_delete.
All requests are validated against an allowed-domain allowlist.
Credentials are never passed through LLM context — use the vault integration.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from computer_agent.config import settings
from computer_agent.tools.base import RiskLevel, ToolResult, tool


def _validate_url(url: str) -> None:
    """Ensure the request targets an allowed domain."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    # Strip port if present (e.g. localhost:8080 → localhost)
    domain = domain.split(":")[0]

    # Always allow localhost for local dev
    if domain in ("localhost", "127.0.0.1", "::1"):
        return

    for allowed in settings.allowed_api_domains:
        if domain == allowed or domain.endswith("." + allowed):
            return

    raise PermissionError(
        f"Domain '{domain}' is not in the allowed API domains list. "
        f"Add it to ALLOWED_API_DOMAINS in your .env to permit it."
    )


@tool(name="http_get", risk_level=RiskLevel.MEDIUM, category="api",
      description="Make an HTTP GET request to a URL and return the response.")
def http_get(
    url: str,
    headers: dict | None = None,
    timeout: int = 30,
) -> ToolResult:
    """
    Perform an HTTP GET request.

    url: The full URL to request
    headers: Optional dictionary of request headers
    timeout: Request timeout in seconds
    """
    try:
        _validate_url(url)
    except PermissionError as e:
        return ToolResult.fail(error=str(e))

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers or {})
        return _build_result(response)
    except httpx.TimeoutException:
        return ToolResult.fail(error=f"Request timed out after {timeout}s: {url}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="http_post", risk_level=RiskLevel.MEDIUM, category="api",
      description="Make an HTTP POST request with a JSON body.")
def http_post(
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
) -> ToolResult:
    """
    Perform an HTTP POST request with a JSON payload.

    url: The full URL to POST to
    body: JSON-serializable dict to send as the request body
    headers: Optional dictionary of request headers
    timeout: Request timeout in seconds
    """
    try:
        _validate_url(url)
    except PermissionError as e:
        return ToolResult.fail(error=str(e))

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(url, json=body or {}, headers=headers or {})
        return _build_result(response)
    except httpx.TimeoutException:
        return ToolResult.fail(error=f"Request timed out after {timeout}s: {url}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="http_put", risk_level=RiskLevel.MEDIUM, category="api",
      description="Make an HTTP PUT request with a JSON body.")
def http_put(
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
) -> ToolResult:
    """
    Perform an HTTP PUT request.

    url: The full URL to PUT to
    body: JSON-serializable dict to send as the request body
    headers: Optional request headers
    timeout: Request timeout in seconds
    """
    try:
        _validate_url(url)
    except PermissionError as e:
        return ToolResult.fail(error=str(e))

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.put(url, json=body or {}, headers=headers or {})
        return _build_result(response)
    except httpx.TimeoutException:
        return ToolResult.fail(error=f"PUT timed out: {url}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="http_delete", risk_level=RiskLevel.HIGH, category="api",
      description="Make an HTTP DELETE request. This may permanently delete remote resources.")
def http_delete(
    url: str,
    headers: dict | None = None,
    timeout: int = 30,
) -> ToolResult:
    """
    Perform an HTTP DELETE request.

    url: The resource URL to delete
    headers: Optional request headers
    timeout: Request timeout in seconds
    """
    try:
        _validate_url(url)
    except PermissionError as e:
        return ToolResult.fail(error=str(e))

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.delete(url, headers=headers or {})
        return _build_result(response)
    except Exception as e:
        return ToolResult.fail(error=str(e))


def _build_result(response: httpx.Response) -> ToolResult:
    """Build a ToolResult from an httpx Response object."""
    # Attempt to parse JSON, fall back to raw text
    try:
        body = response.json()
    except Exception:
        body = response.text

    success = 200 <= response.status_code < 300
    if success:
        return ToolResult.ok(
            output=body,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
    return ToolResult.fail(
        error=f"HTTP {response.status_code}: {response.text[:500]}",
        status_code=response.status_code,
    )
