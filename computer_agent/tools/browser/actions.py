"""
Browser tools powered by Playwright (CDP-based).
These tools control a persistent Chromium/Firefox/WebKit session.
The browser session is managed by BrowserSessionManager (singleton).
"""

from __future__ import annotations

import base64

from computer_agent.config import settings
from computer_agent.logging_setup import get_logger
from computer_agent.tools.base import RiskLevel, ToolResult, tool

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Browser Session Manager (singleton)
# ---------------------------------------------------------------------------

_session: _BrowserSession | None = None


class _BrowserSession:
    """Wraps a persistent Playwright browser context."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def start(self) -> None:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        browser_cls = getattr(self._playwright, settings.browser_type)

        user_data = settings.browser_user_data_path
        user_data.mkdir(parents=True, exist_ok=True)

        self._context = await browser_cls.launch_persistent_context(
            user_data_dir=str(user_data),
            headless=settings.browser_headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )

        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = await self._context.new_page()

        logger.info("browser_session_started", browser=settings.browser_type)

    async def page(self):
        if self._page is None:
            await self.start()
        return self._page

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
        self._page = None
        self._context = None
        self._browser = None
        logger.info("browser_session_closed")


async def _get_session() -> _BrowserSession:
    global _session
    if _session is None:
        _session = _BrowserSession()
        await _session.start()
    return _session


# ---------------------------------------------------------------------------
# Browser Tools
# ---------------------------------------------------------------------------

@tool(name="browser_navigate", risk_level=RiskLevel.MEDIUM, category="browser",
      description="Navigate the browser to a URL.")
async def browser_navigate(url: str) -> ToolResult:
    """
    Navigate the browser to the given URL and wait for page load.

    url: Full URL to navigate to (e.g. https://www.google.com)
    """
    try:
        session = await _get_session()
        page = await session.page()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        status = response.status if response else 0
        title = await page.title()
        return ToolResult.ok(
            output={"url": page.url, "title": title, "status": status},
            url=page.url,
        )
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_screenshot", risk_level=RiskLevel.LOW, category="browser",
      description="Take a screenshot of the current browser page.")
async def browser_screenshot() -> ToolResult:
    """Capture a full-page screenshot of the active browser tab."""
    try:
        session = await _get_session()
        page = await session.page()
        screenshot_bytes = await page.screenshot(full_page=False, type="png")
        b64 = base64.b64encode(screenshot_bytes).decode()
        return ToolResult.ok(output=b64, format="base64_png", url=page.url)
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_get_url", risk_level=RiskLevel.LOW, category="browser",
      description="Get the current URL and page title of the active browser tab.")
async def browser_get_url() -> ToolResult:
    """Return the current page URL and title."""
    try:
        session = await _get_session()
        page = await session.page()
        return ToolResult.ok(output={"url": page.url, "title": await page.title()})
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_click", risk_level=RiskLevel.MEDIUM, category="browser",
      description="Click an element in the browser matching a CSS selector or text.")
async def browser_click(selector: str, use_text: bool = False) -> ToolResult:
    """
    Click a browser element.

    selector: CSS selector or visible text of the element
    use_text: If True, treats selector as visible text to match
    """
    try:
        session = await _get_session()
        page = await session.page()
        if use_text:
            await page.get_by_text(selector).first.click(timeout=10000)
        else:
            await page.click(selector, timeout=10000)
        return ToolResult.ok(output=f"Clicked: {selector}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_type", risk_level=RiskLevel.MEDIUM, category="browser",
      description="Type text into a browser input field matching a CSS selector.")
async def browser_type(selector: str, text: str, clear_first: bool = True) -> ToolResult:
    """
    Type text into a browser input field.

    selector: CSS selector of the input element
    text: Text to type
    clear_first: Whether to clear the field before typing
    """
    try:
        session = await _get_session()
        page = await session.page()
        if clear_first:
            await page.fill(selector, "", timeout=10000)
        await page.type(selector, text, delay=50)
        return ToolResult.ok(output=f"Typed into {selector}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_get_text", risk_level=RiskLevel.LOW, category="browser",
      description="Extract the visible text content from the current browser page.")
async def browser_get_text(selector: str = "body") -> ToolResult:
    """
    Get text content from a page element.

    selector: CSS selector of the element (default: body = full page text)
    """
    try:
        session = await _get_session()
        page = await session.page()
        text = await page.inner_text(selector, timeout=10000)
        return ToolResult.ok(output=text, selector=selector, char_count=len(text))
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_get_dom", risk_level=RiskLevel.LOW, category="browser",
      description="Get the HTML source of the current page or a specific element.")
async def browser_get_dom(selector: str = "body", max_chars: int = 20000) -> ToolResult:
    """
    Get the HTML DOM of the current page or element.

    selector: CSS selector (default: full body)
    max_chars: Maximum characters to return
    """
    try:
        session = await _get_session()
        page = await session.page()
        html = await page.inner_html(selector, timeout=10000)
        truncated = len(html) > max_chars
        return ToolResult.ok(
            output=html[:max_chars],
            truncated=truncated,
            url=page.url,
        )
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_wait_for", risk_level=RiskLevel.LOW, category="browser",
      description="Wait for a CSS selector to appear on the page.")
async def browser_wait_for(selector: str, timeout: int = 10) -> ToolResult:
    """
    Wait for an element to appear in the DOM.

    selector: CSS selector to wait for
    timeout: Maximum seconds to wait
    """
    try:
        session = await _get_session()
        page = await session.page()
        await page.wait_for_selector(selector, timeout=timeout * 1000)
        return ToolResult.ok(output=f"Element appeared: {selector}")
    except Exception as e:
        return ToolResult.fail(error=f"Timeout waiting for '{selector}': {e}")


@tool(name="browser_evaluate", risk_level=RiskLevel.MEDIUM, category="browser",
      description="Execute JavaScript in the browser context and return the result.")
async def browser_evaluate(script: str) -> ToolResult:
    """
    Run a JavaScript expression in the page context.

    script: JavaScript expression to evaluate (e.g. 'document.title')
    """
    try:
        session = await _get_session()
        page = await session.page()
        result = await page.evaluate(script)
        return ToolResult.ok(output=result)
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_go_back", risk_level=RiskLevel.LOW, category="browser",
      description="Navigate back one step in the browser history.")
async def browser_go_back() -> ToolResult:
    """Go back one page in browser history."""
    try:
        session = await _get_session()
        page = await session.page()
        await page.go_back(wait_until="domcontentloaded", timeout=15000)
        return ToolResult.ok(output={"url": page.url, "title": await page.title()})
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="browser_close", risk_level=RiskLevel.LOW, category="browser",
      description="Close the browser session.")
async def browser_close() -> ToolResult:
    """Close and clean up the browser session."""
    global _session
    if _session:
        await _session.close()
        _session = None
    return ToolResult.ok(output="Browser session closed")
