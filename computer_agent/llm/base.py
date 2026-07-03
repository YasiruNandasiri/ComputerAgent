"""
LLM Provider Abstraction Layer — inspired by Google ADK's BaseLlm pattern.

Defines a unified interface all LLM providers must implement so the
Coordinator can swap models (Anthropic, OpenAI, Gemini, local Ollama, etc.)
purely through configuration without touching orchestration logic.

Usage:
    provider = LLMRegistry.resolve("claude-sonnet-4-20250514")
    response = await provider.generate(messages, system=SYSTEM_PROMPT, tools=tools)
    for call in response.tool_calls:
        result = await registry.invoke(call.name, **call.arguments)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Unified message types (provider-neutral)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultMessage:
    """The result of a tool invocation, sent back to the LLM."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class LLMUsage:
    """Token usage statistics from a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """
    Unified response from any LLM provider.

    The coordinator only works with LLMResponse objects — never with
    provider-specific types like anthropic.types.Message.
    """
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens"
    usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_done(self) -> bool:
        return self.stop_reason in ("end_turn", "stop", "stop_sequence")


# ---------------------------------------------------------------------------
# Message format (provider-neutral conversation history)
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in the conversation history."""
    role: str  # "user" | "assistant"
    # Content can be a plain string, or a list of typed blocks for multi-turn tool use
    content: str | list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Subclasses must implement `generate()`. The `stream()` method has a
    default implementation that calls `generate()` and yields the result.

    Providers register themselves in LLMRegistry via `supported_models()`.
    """

    @classmethod
    @abstractmethod
    def supported_models(cls) -> list[str]:
        """
        Return a list of regex patterns for model names this provider handles.

        Example: ["claude-.*", "claude-haiku-.*"]
        Used by LLMRegistry for auto-resolution.
        """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Call the LLM and return a unified LLMResponse.

        Args:
            messages: Conversation history in provider-specific format
                      (providers normalize internally).
            system: System prompt string.
            tools: List of tool schemas in the provider's native format.
                   Use LLMToolAdapter.format_tools() to generate these.
            max_tokens: Maximum tokens in the response.
            model: Override model name (falls back to the instance's model).
        """

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> AsyncIterator[LLMResponse]:
        """
        Stream LLM responses. Default: calls generate() and yields the result.
        Override for true streaming.
        """
        response = await self.generate(
            messages,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            model=model,
        )
        yield response

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
