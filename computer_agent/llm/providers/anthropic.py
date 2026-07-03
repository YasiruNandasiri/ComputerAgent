"""
Anthropic Claude provider adapter.

Wraps anthropic.AsyncAnthropic and translates to/from the unified LLM types.
Supports all Claude models (claude-3, claude-3-5, claude-4, etc.).
"""

from __future__ import annotations

import json
from typing import Any

from computer_agent.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolCall
from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None  # lazy-loaded

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"claude-.*"]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "The 'anthropic' package is required for Claude models. "
                    "Install it with: uv add anthropic"
                ) from e
        return self._client

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        model_name = model or self._model

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Convert anthropic.types.Message → LLMResponse."""
        text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input or {},
                ))

        stop_reason = self._normalize_stop_reason(response.stop_reason)

        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
        )

    @staticmethod
    def _normalize_stop_reason(reason: str | None) -> str:
        mapping = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
            "stop_sequence": "end_turn",
        }
        return mapping.get(reason or "", "end_turn")

    # ------------------------------------------------------------------
    # Message format helpers (used by coordinator)
    # ------------------------------------------------------------------

    @staticmethod
    def format_tool_schemas(tool_definitions: list[Any]) -> list[dict[str, Any]]:
        """Convert ToolDefinition list → Anthropic tool schema list."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tool_definitions
        ]

    @staticmethod
    def format_tool_result_messages(
        tool_calls: list[ToolCall],
        results: list[Any],  # list[ToolResult]
    ) -> list[dict[str, Any]]:
        """
        Build the 'tool_result' content blocks for the next user turn.
        In Anthropic's API, tool results are sent as a user message
        containing an array of tool_result blocks.
        """
        content = []
        for tc, result in zip(tool_calls, results, strict=True):
            output = result.output
            if isinstance(output, (dict, list)):
                content_str = json.dumps(output, indent=2, default=str)
            else:
                content_str = str(output) if output is not None else "Success"

            if not result.success:
                content_str = f"Error: {result.error}"

            content.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": content_str,
            })
        return content

    @staticmethod
    def assistant_message_from_response(response: Any, raw_content: list[Any]) -> dict[str, Any]:
        """
        Build the assistant message dict to append to conversation history.
        raw_content is the list of Anthropic content blocks from the raw response.
        """
        return {"role": "assistant", "content": raw_content}
