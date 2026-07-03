"""
OpenAI provider adapter.

Wraps openai.AsyncOpenAI and translates to/from the unified LLM types.
Supports GPT-4, GPT-4o, o1, o3, and any OpenAI-compatible endpoint
(Azure OpenAI, local LM Studio, etc.) via base_url override.
"""

from __future__ import annotations

import json
from typing import Any

from computer_agent.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolCall
from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)

_AZURE_PREFIX = "azure/"


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None  # lazy-loaded

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gpt-.*", r"o1-.*", r"o3-.*", r"o4-.*", r"openai/.*"]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
                kwargs: dict[str, Any] = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = openai.AsyncOpenAI(**kwargs)
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for OpenAI models. "
                    "Install it with: uv add openai"
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

        # Prepend system message in OpenAI format
        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": full_messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Convert openai.types.chat.ChatCompletion → LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        stop_reason = self._normalize_stop_reason(choice.finish_reason)

        usage = LLMUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
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
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "end_turn",
            "function_call": "tool_use",
        }
        return mapping.get(reason or "", "end_turn")

    # ------------------------------------------------------------------
    # Message format helpers (used by coordinator)
    # ------------------------------------------------------------------

    @staticmethod
    def format_tool_schemas(tool_definitions: list[Any]) -> list[dict[str, Any]]:
        """Convert ToolDefinition list → OpenAI function-calling schema list."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tool_definitions
        ]

    @staticmethod
    def format_tool_result_messages(
        tool_calls: list[ToolCall],
        results: list[Any],  # list[ToolResult]
    ) -> list[dict[str, Any]]:
        """
        Build tool result messages for the next turn.
        OpenAI requires one 'tool' role message per tool call.
        """
        messages = []
        for tc, result in zip(tool_calls, results, strict=True):
            output = result.output
            if isinstance(output, (dict, list)):
                content_str = json.dumps(output, indent=2, default=str)
            else:
                content_str = str(output) if output is not None else "Success"

            if not result.success:
                content_str = f"Error: {result.error}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content_str,
            })
        return messages

    @staticmethod
    def assistant_message_from_tool_calls(tool_calls: list[ToolCall]) -> dict[str, Any]:
        """Build the assistant message dict that requested the tool calls."""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        }
