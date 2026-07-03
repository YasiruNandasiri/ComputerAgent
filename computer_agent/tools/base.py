"""
Base classes for all tools in the Computer Agent framework.

A Tool is an atomic, non-decomposable function the LLM can invoke.
Every tool:
  - Has a typed input schema (generated from the function signature)
  - Returns a ToolResult
  - Declares a RiskLevel (drives HITL gate logic)
  - Is registered in the BrickRegistry
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel


class RiskLevel(StrEnum):
    LOW = "low"        # Read-only, no side effects (screenshot, read_file)
    MEDIUM = "medium"  # Limited side effects, reversible (write_file, click)
    HIGH = "high"      # Significant side effects (send_email, delete_file)
    CRITICAL = "critical"  # Irreversible or financial (payment, rm -rf)


class ExecutionStrategy(StrEnum):
    API_DIRECT = "api_direct"
    STRUCTURAL = "structural"   # DOM / accessibility tree
    VISUAL = "visual"           # Screenshot + LLM vision


class ToolResult(BaseModel):
    """Unified result type returned by every tool invocation."""
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def ok(cls, output: Any = None, **metadata: Any) -> ToolResult:
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> ToolResult:
        return cls(success=False, error=error, metadata=metadata)


@dataclass
class ToolDefinition:
    """Metadata for a registered tool, used to bind it to the LLM."""
    name: str
    description: str
    func: Callable[..., ToolResult]
    risk_level: RiskLevel
    input_schema: dict[str, Any]          # JSON Schema for parameters
    category: str = "general"             # screen / browser / fs / shell / api
    requires_hitl: bool = False           # Override from abilities engine
    platform: str | None = None          # "darwin" | "linux" | None (any)

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Serialize to Anthropic tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Serialize to OpenAI / Gemini function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


F = TypeVar("F", bound=Callable[..., Any])


def tool(
    name: str | None = None,
    description: str | None = None,
    risk_level: RiskLevel = RiskLevel.LOW,
    category: str = "general",
    platform: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator that marks a function as a Tool and attaches metadata.

    Usage:
        @tool(name="read_file", risk_level=RiskLevel.LOW, category="fs")
        def read_file(path: str) -> ToolResult:
            ...
    """

    def decorator(func: F) -> F:
        _name = name or func.__name__
        _description = description or (inspect.getdoc(func) or "").split("\n")[0]
        _schema = _build_input_schema(func)

        # Attach metadata to the function object so registry can pick it up
        func._tool_meta = ToolDefinition(  # type: ignore[attr-defined]
            name=_name,
            description=_description,
            func=func,
            risk_level=risk_level,
            input_schema=_schema,
            category=category,
            platform=platform,
        )
        return func

    return decorator


# ---------------------------------------------------------------------------
# Internal: build a JSON Schema dict from a Python function's type hints
# ---------------------------------------------------------------------------

_PY_TYPE_TO_JSON: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "bytes": "string",
}


def _build_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    hints = {k: v for k, v in func.__annotations__.items() if k != "return"}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        json_type = _resolve_json_type(hints.get(param_name))
        prop: dict[str, Any] = {"type": json_type}

        # Extract docstring param descriptions (basic heuristic)
        doc = inspect.getdoc(func) or ""
        for line in doc.splitlines():
            line = line.strip()
            if line.lower().startswith(f"{param_name}:") or line.lower().startswith(
                f":param {param_name}:"
            ):
                prop["description"] = line.split(":", 1)[-1].strip()
                break

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _resolve_json_type(annotation: Any) -> str:
    if annotation is None:
        return "string"
    type_name = getattr(annotation, "__name__", str(annotation))
    # Handle Optional[X] → extract X
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        # e.g. list[str], dict[str,str]
        origin_name = getattr(origin, "__name__", str(origin))
        return _PY_TYPE_TO_JSON.get(origin_name, "string")
    return _PY_TYPE_TO_JSON.get(type_name, "string")
