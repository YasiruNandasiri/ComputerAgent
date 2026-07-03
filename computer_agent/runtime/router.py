"""
Execution Router — selects the best execution strategy for a task step.

Priority:
  1. API_DIRECT   — fast API call (if a known API mapping exists)
  2. STRUCTURAL   — DOM / accessibility tree (if element can be found programmatically)
  3. VISUAL       — screenshot + LLM vision (fallback, always possible)

The router also implements the fallback cascade:
  execute(strategy_1) → on failure → execute(strategy_2) → ... → escalate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from computer_agent.logging_setup import get_logger
from computer_agent.tools.base import ExecutionStrategy
from computer_agent.tools.registry import registry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known API Mappings
# Maps high-level action descriptions to direct API tool names.
# Populated by skills that declare `api_equivalent` in their manifests.
# ---------------------------------------------------------------------------

_API_MAPPINGS: dict[str, str] = {
    "search the web": "http_get",
    "read email": "http_get",  # → Gmail/Outlook API
    "send email": "http_post",
    "create calendar event": "http_post",
    "get weather": "http_get",
    "search github": "http_get",
}


@dataclass
class RoutingDecision:
    strategy: ExecutionStrategy
    tool_name: str | None = None
    confidence: float = 1.0
    reason: str = ""


class ExecutionRouter:
    """
    Selects the optimal execution strategy for a given task description
    and provides a fallback cascade when strategies fail.
    """

    def select_strategy(
        self,
        task_description: str,
        context: dict[str, Any],
    ) -> RoutingDecision:
        """
        Decide how to execute this task step.
        Returns a RoutingDecision with the selected strategy and tool.
        """
        desc_lower = task_description.lower()

        # Priority 1: Check for direct API mapping
        for trigger, tool_name in _API_MAPPINGS.items():
            if trigger in desc_lower and tool_name in registry.tool_names():
                return RoutingDecision(
                    strategy=ExecutionStrategy.API_DIRECT,
                    tool_name=tool_name,
                    confidence=0.9,
                    reason=f"Matched API mapping: {trigger} → {tool_name}",
                )

        # Priority 2: Browser DOM / structural access for web tasks
        browser_keywords = [
            "navigate", "click", "fill", "form", "browser", "website",
            "page", "url", "web", "http", "search on", "open tab",
        ]
        if any(kw in desc_lower for kw in browser_keywords):
            return RoutingDecision(
                strategy=ExecutionStrategy.STRUCTURAL,
                tool_name="browser_navigate",
                confidence=0.8,
                reason="Browser/web action detected",
            )

        # Priority 3: Desktop accessibility for native app tasks
        desktop_keywords = [
            "open app", "launch", "close window", "menu", "dialog",
            "application", "finder", "spotlight",
        ]
        if any(kw in desc_lower for kw in desktop_keywords):
            return RoutingDecision(
                strategy=ExecutionStrategy.STRUCTURAL,
                tool_name=None,  # Desktop agent will select specific tool
                confidence=0.7,
                reason="Desktop/native app action detected",
            )

        # Default: Visual computer-use
        return RoutingDecision(
            strategy=ExecutionStrategy.VISUAL,
            tool_name="take_screenshot",
            confidence=0.6,
            reason="Fallback to visual computer-use",
        )

    def get_fallback_sequence(
        self, primary: ExecutionStrategy
    ) -> list[ExecutionStrategy]:
        """Return the fallback cascade from a primary strategy."""
        cascades = {
            ExecutionStrategy.API_DIRECT: [
                ExecutionStrategy.API_DIRECT,
                ExecutionStrategy.STRUCTURAL,
                ExecutionStrategy.VISUAL,
            ],
            ExecutionStrategy.STRUCTURAL: [
                ExecutionStrategy.STRUCTURAL,
                ExecutionStrategy.VISUAL,
            ],
            ExecutionStrategy.VISUAL: [
                ExecutionStrategy.VISUAL,
            ],
        }
        return cascades.get(primary, [ExecutionStrategy.VISUAL])

    def register_api_mapping(self, trigger: str, tool_name: str) -> None:
        """Register a new API mapping at runtime (e.g., from a skill manifest)."""
        _API_MAPPINGS[trigger.lower()] = tool_name
        logger.debug("api_mapping_registered", trigger=trigger, tool=tool_name)


# Module-level singleton
router = ExecutionRouter()
