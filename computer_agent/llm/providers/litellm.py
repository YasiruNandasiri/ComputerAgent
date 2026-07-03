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

import asyncio
import json
import random
from typing import Any

from computer_agent.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolCall
from computer_agent.llm.errors import LLMContextWindowError, LLMError, LLMRateLimitError
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

        response = await self._acompletion_with_retry(litellm, kwargs)
        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Retry with exponential backoff + jitter
    # ------------------------------------------------------------------

    async def _acompletion_with_retry(self, litellm_mod: Any, kwargs: dict[str, Any]) -> Any:
        from computer_agent.config import settings

        err: Exception | None = None
        retry_after_hint: float | None = None

        for attempt in range(settings.llm_max_retries + 1):
            try:
                return await litellm_mod.acompletion(**kwargs)
            except litellm_mod.ContextWindowExceededError as e:
                # Never retried — a bigger prompt won't shrink by waiting
                raise LLMContextWindowError(str(e)) from e
            except litellm_mod.RateLimitError as e:
                err = e
                retry_after_hint = self._extract_retry_after(e)
            except (
                litellm_mod.APIConnectionError,
                litellm_mod.ServiceUnavailableError,
                litellm_mod.InternalServerError,
                litellm_mod.Timeout,
            ) as e:
                err = e
                retry_after_hint = None
            except litellm_mod.AuthenticationError as e:
                raise LLMError(f"Authentication failed: {e}") from e
            except litellm_mod.BadRequestError as e:
                raise LLMError(f"Bad request: {e}") from e

            if attempt == settings.llm_max_retries:
                raise LLMRateLimitError(
                    f"LLM call failed after {attempt} retries: {err}",
                    retry_after=retry_after_hint,
                ) from err

            delay = min(
                settings.llm_retry_base_delay * (2 ** attempt),
                settings.llm_retry_max_delay,
            )
            delay *= 0.5 + random.random() * 0.5  # jitter: 50–100%
            if retry_after_hint is not None:
                delay = max(delay, retry_after_hint)

            logger.warning(
                "llm_retry",
                attempt=attempt + 1,
                delay=round(delay, 1),
                error=type(err).__name__,
                retry_after=retry_after_hint,
            )
            await asyncio.sleep(delay)

        # Should never reach here, but satisfy type checker
        raise LLMRateLimitError(f"LLM call failed after {settings.llm_max_retries} retries: {err}")

    def _extract_retry_after(self, e: Any) -> float | None:
        headers = getattr(getattr(e, "response", None), "headers", None) or {}
        val = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None

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
        """Build tool result messages for the next turn (OpenAI tool role format).

        Screenshots are returned as image_url content blocks in a follow-up user
        message (the OpenAI API does not accept images inside role:tool messages).
        """
        from computer_agent.config import settings
        from computer_agent.llm.context_manager import truncate_middle

        messages: list[dict[str, Any]] = []
        image_followups: list[dict[str, Any]] = []  # (text_label, b64, media_type)

        for tc, result in zip(tool_calls, results, strict=True):
            output = result.output
            fmt = result.metadata.get("format", "")

            if not result.success:
                content_str = f"Error: {result.error}"
            elif fmt in ("base64_png", "base64_jpeg"):
                # Emit a short placeholder in the tool message; actual image follows
                w = result.metadata.get("width", "?")
                h = result.metadata.get("height", "?")
                ow = result.metadata.get("original_width", w)
                oh = result.metadata.get("original_height", h)
                media_type = "image/jpeg" if fmt == "base64_jpeg" else "image/png"
                scale = f"{ow / w:.2f}" if isinstance(w, int) and w else "1.00"
                content_str = (
                    f"Screenshot captured ({w}x{h} {'JPEG' if fmt == 'base64_jpeg' else 'PNG'}, "
                    f"screen is {ow}x{oh} — multiply coordinates by {scale}). "
                    f"Image attached in the next message."
                )
                image_followups.append({
                    "label": f"[Screenshot from {tc.name}]",
                    "b64": str(output),
                    "media_type": media_type,
                })
            elif isinstance(output, (dict, list)):
                content_str = truncate_middle(
                    json.dumps(output, separators=(",", ":"), default=str),
                    settings.tool_result_max_chars,
                )
            else:
                content_str = truncate_middle(
                    str(output) if output is not None else "Success",
                    settings.tool_result_max_chars,
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content_str,
            })

        # Attach images in a single follow-up user message after all tool messages
        if image_followups:
            content_parts: list[dict[str, Any]] = []
            for img in image_followups:
                content_parts.append({"type": "text", "text": img["label"]})
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img['media_type']};base64,{img['b64']}",
                    },
                })
            messages.append({"role": "user", "content": content_parts})

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

