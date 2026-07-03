"""
Tests for the BrickRegistry — tool auto-discovery and invocation.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from computer_agent.tools.base import RiskLevel, ToolDefinition, ToolResult, tool
from computer_agent.tools.registry import BrickRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_registry() -> BrickRegistry:
    return BrickRegistry()


def make_tool(name: str, risk: RiskLevel = RiskLevel.LOW) -> ToolDefinition:
    @tool(name=name, risk_level=risk, category="test", description=f"Test tool {name}")
    def dummy_tool(message: str) -> ToolResult:
        """A dummy test tool."""
        return ToolResult.ok(output=f"echo: {message}")

    return dummy_tool._tool_meta


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------

class TestBrickRegistry:
    def test_manual_registration(self, empty_registry: BrickRegistry) -> None:
        tool_def = make_tool("test_echo")
        empty_registry.register_tool(tool_def)
        assert "test_echo" in empty_registry.tool_names()

    def test_get_tool_returns_definition(self, empty_registry: BrickRegistry) -> None:
        tool_def = make_tool("test_get")
        empty_registry.register_tool(tool_def)
        retrieved = empty_registry.get_tool("test_get")
        assert retrieved.name == "test_get"

    def test_get_unknown_tool_raises(self, empty_registry: BrickRegistry) -> None:
        with pytest.raises(KeyError, match="unknown_tool"):
            empty_registry.get_tool("unknown_tool")

    def test_filter_by_category(self, empty_registry: BrickRegistry) -> None:
        @tool(name="cat_tool_1", category="screen", risk_level=RiskLevel.LOW)
        def t1(x: int) -> ToolResult:
            return ToolResult.ok()

        @tool(name="cat_tool_2", category="api", risk_level=RiskLevel.MEDIUM)
        def t2(x: int) -> ToolResult:
            return ToolResult.ok()

        empty_registry.register_tool(t1._tool_meta)
        empty_registry.register_tool(t2._tool_meta)

        screen_tools = empty_registry.get_tools_by_category("screen")
        assert all(t.category == "screen" for t in screen_tools)

    def test_filter_by_risk_max(self, empty_registry: BrickRegistry) -> None:
        low_tool = make_tool("low_risk", RiskLevel.LOW)
        high_tool = make_tool("high_risk", RiskLevel.HIGH)
        empty_registry.register_tool(low_tool)
        empty_registry.register_tool(high_tool)

        safe_tools = empty_registry.get_tools_by_risk(RiskLevel.MEDIUM)
        tool_names = [t.name for t in safe_tools]
        assert "low_risk" in tool_names
        assert "high_risk" not in tool_names

    def test_anthropic_schema_format(self, empty_registry: BrickRegistry) -> None:
        empty_registry.register_tool(make_tool("schema_test"))
        schemas = empty_registry.to_anthropic_tool_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Tests: Invocation
# ---------------------------------------------------------------------------

class TestToolInvocation:
    @pytest.mark.asyncio
    async def test_invoke_sync_tool(self, empty_registry: BrickRegistry) -> None:
        @tool(name="invoke_sync", category="test", risk_level=RiskLevel.LOW)
        def sync_tool(value: str) -> ToolResult:
            return ToolResult.ok(output=f"got: {value}")

        empty_registry.register_tool(sync_tool._tool_meta)
        result = await empty_registry.invoke("invoke_sync", value="hello")
        assert result.success
        assert result.output == "got: hello"

    @pytest.mark.asyncio
    async def test_invoke_async_tool(self, empty_registry: BrickRegistry) -> None:
        import asyncio

        @tool(name="invoke_async", category="test", risk_level=RiskLevel.LOW)
        async def async_tool(count: int) -> ToolResult:
            await asyncio.sleep(0)
            return ToolResult.ok(output=count * 2)

        empty_registry.register_tool(async_tool._tool_meta)
        result = await empty_registry.invoke("invoke_async", count=5)
        assert result.success
        assert result.output == 10

    @pytest.mark.asyncio
    async def test_invoke_failing_tool(self, empty_registry: BrickRegistry) -> None:
        @tool(name="failing_tool", category="test", risk_level=RiskLevel.LOW)
        def bad_tool() -> ToolResult:
            raise RuntimeError("Intentional failure")

        empty_registry.register_tool(bad_tool._tool_meta)
        result = await empty_registry.invoke("failing_tool")
        assert not result.success
        assert "Intentional failure" in result.error
