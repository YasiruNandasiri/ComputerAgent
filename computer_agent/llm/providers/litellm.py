"""
LiteLLM provider adapter — the universal fallback.

Wraps litellm.acompletion() which supports 100+ LLM providers through
a single OpenAI-compatible interface. This is the recommended adapter for:
  - Local models via Ollama     (ollama/llama3, ollama/mistral, ...)
  - HuggingFace TGI/Inference   (huggingface/...)
  - AWS Bedrock                 (bedrock/...)
  - Azure OpenAI                (azure/...)
  - OpenRouter                  (openrouter/...)
  - Any OpenAI-compatible API   (openai/custom)

Model name format: "<provider>/<model_name>" e.g. "ollama/llama3"
"""

from __future__ import annotations

import json
from typing import Any

from computer_agent.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolCall
from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


class LiteLLMProvider(BaseLLMProvider):
    """
    Universal LLM provider adapter via LiteLLM.
    Falls back to this provider when no other provider matches the model name.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    @classmethod
    def supported_models(cls) -> list[str]:
        # Explicit prefixes for common local/open providers
        return [
            r"ollama/.*",
            r"ollama_chat/.*",
            r"huggingface/.*",
            r"bedrock/.*",
            r"azure/.*",
            r"openrouter/.*",
            r"together_ai/.*",
            r"replicate/.*",
            r"vertex_ai/.*",
            r"litellm/.*",
        ]

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> LLMResponse:
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "The 'litellm' package is required for this model provider. "
                "Install it with: uv add litellm"
            ) from e

        model_name = model or self._model

        # Prepend system message (OpenAI compatible format)
        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": full_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base

        response = await litellm.acompletion(**kwargs)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal parsers (LiteLLM returns OpenAI-compatible format)
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Convert LiteLLM response (OpenAI-compatible) → LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls: list[ToolCall] = []

        if hasattr(message, "tool_calls") and message.tool_calls:
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
            input_tokens=getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
            output_tokens=getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
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
            "function_call": "tool_use",
        }
        return mapping.get(reason or "", "end_turn")

    # ------------------------------------------------------------------
    # Message format helpers (OpenAI-compatible, same as OpenAI provider)
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
        """Build tool result messages for the next turn (OpenAI tool role format)."""
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
