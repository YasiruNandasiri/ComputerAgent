"""
BrickRegistry — plug-and-play registry for Tools, Skills, and Abilities.

The registry auto-discovers tool modules from the `tools/` directory tree,
loads skill manifests from the `skills/` plugin directory, and exposes
LLM-compatible schemas for function-calling.

Usage:
    registry = BrickRegistry()
    registry.discover()
    tools = registry.get_all_tools()
    schema = registry.to_anthropic_tool_schemas()
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

from computer_agent.logging_setup import get_logger
from computer_agent.tools.base import RiskLevel, ToolDefinition, ToolResult

logger = get_logger(__name__)


class BrickRegistry:
    """Central registry for all agent capabilities."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._discovered: bool = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> None:
        """
        Auto-discover all tools by:
          1. Scanning the built-in `computer_agent.tools` package (any @tool-decorated function)
          2. Loading external tool packages via entry-points (group: "computer_agent.tools")

        External packages declare tools in their pyproject.toml:
            [project.entry-points."computer_agent.tools"]
            my_tools = "my_package.tools:register"
        where `register` is a callable that receives the BrickRegistry instance.
        """
        if self._discovered:
            return

        tools_pkg_path = Path(__file__).parent
        self._scan_package("computer_agent.tools", tools_pkg_path)
        self._load_entry_point_tools()
        self._discovered = True
        logger.info("tool_discovery_complete", count=len(self._tools))

    def _load_entry_point_tools(self) -> None:
        """Load external tools registered via Python entry-points."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="computer_agent.tools")
            for ep in eps:
                try:
                    register_fn = ep.load()
                    if callable(register_fn):
                        register_fn(self)
                        logger.info("entry_point_tools_loaded", name=ep.name)
                    else:
                        logger.warning("entry_point_not_callable", name=ep.name)
                except Exception as e:
                    logger.warning("entry_point_tool_load_failed", name=ep.name, error=str(e))
        except Exception as e:
            logger.debug("entry_point_scan_failed", group="computer_agent.tools", error=str(e))

    def _scan_package(self, package_name: str, package_path: Path) -> None:
        for _finder, module_name, _is_pkg in pkgutil.walk_packages(
            path=[str(package_path)],
            prefix=package_name + ".",
            onerror=lambda name: logger.warning("module_import_error", module=name),
        ):
            # Skip the base module and registry itself
            if module_name.endswith((".base", ".registry")):
                continue
            try:
                module = importlib.import_module(module_name)
                self._register_from_module(module)
            except Exception as e:
                logger.warning("module_load_failed", module=module_name, error=str(e))

    def _register_from_module(self, module: Any) -> None:
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and hasattr(obj, "_tool_meta"):
                tool_def: ToolDefinition = obj._tool_meta
                if tool_def.name not in self._tools:
                    self._tools[tool_def.name] = tool_def
                    logger.debug("tool_registered", name=tool_def.name, category=tool_def.category)

    # ------------------------------------------------------------------
    # Manual registration (for testing / runtime injection)
    # ------------------------------------------------------------------

    def register_tool(self, tool_def: ToolDefinition) -> None:
        """Manually register a ToolDefinition."""
        self._tools[tool_def.name] = tool_def
        logger.debug("tool_registered_manual", name=tool_def.name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry. Available: {self.tool_names()}")
        return self._tools[name]

    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_all_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.category == category]

    def get_tools_by_risk(self, max_risk: RiskLevel) -> list[ToolDefinition]:
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_idx = order.index(max_risk)
        return [t for t in self._tools.values() if order.index(t.risk_level) <= max_idx]

    # ------------------------------------------------------------------
    # LLM Schema Generation
    # ------------------------------------------------------------------

    def to_anthropic_tool_schemas(
        self,
        categories: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas in Anthropic's tool_use format."""
        tools = self.get_all_tools()
        if categories:
            tools = [t for t in tools if t.category in categories]
        if exclude_names:
            tools = [t for t in tools if t.name not in exclude_names]
        return [t.to_anthropic_tool() for t in tools]

    def to_openai_tool_schemas(
        self,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas in OpenAI/Gemini function-calling format."""
        tools = self.get_all_tools()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_openai_tool() for t in tools]

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Invoke a tool by name with the given keyword arguments.
        Handles both sync and async tool functions.
        """
        tool_def = self.get_tool(tool_name)

        try:
            if inspect.iscoroutinefunction(tool_def.func):
                result = await tool_def.func(**kwargs)
            else:
                result = tool_def.func(**kwargs)

            if not isinstance(result, ToolResult):
                result = ToolResult.ok(output=result)

            return result

        except Exception as e:
            logger.error(
                "tool_invocation_failed",
                tool=tool_name,
                error=str(e),
                kwargs=kwargs,
            )
            return ToolResult.fail(error=str(e), tool=tool_name)

    def __repr__(self) -> str:
        return f"BrickRegistry(tools={len(self._tools)})"


# Module-level singleton
registry = BrickRegistry()
