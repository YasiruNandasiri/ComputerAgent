"""
Google Gemini provider adapter.

Wraps google.genai (the Google Gen AI SDK) and translates to/from
unified LLM types. Supports all Gemini models (gemini-2.5-flash,
gemini-2.5-pro, gemini-2.0-flash, etc.).

Note: This is the direct SDK adapter. For ADK-based orchestration,
you would use google-adk as a full framework — but here we use the
google-genai SDK directly so our HITL, abilities engine, and memory
layers remain in control.
"""

from __future__ import annotations

import json
from typing import Any

from computer_agent.llm.base import BaseLLMProvider, LLMResponse, LLMUsage, ToolCall
from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)


class GoogleProvider(BaseLLMProvider):
    """LLM provider backed by the Google Gen AI (Gemini) SDK."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None  # lazy-loaded

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"gemini-.*", r"google/.*"]

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import google.genai as genai
                if self._api_key:
                    self._client = genai.Client(api_key=self._api_key)
                else:
                    self._client = genai.Client()
            except ImportError as e:
                raise ImportError(
                    "The 'google-genai' package is required for Gemini models. "
                    "Install it with: uv add google-genai"
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
        from google.genai import types as gtypes

        client = self._get_client()
        model_name = model or self._model

        # Convert unified messages to Gemini Content objects
        contents = self._build_contents(messages)

        config = gtypes.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
        )

        # Attach function declarations if tools provided
        if tools:
            config.tools = [self._build_tools_config(tools)]

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_contents(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Convert unified message list to Gemini Content objects."""
        try:
            from google.genai import types as gtypes
        except ImportError:
            return messages  # type: ignore

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, str):
                contents.append(gtypes.Content(
                    role=role,
                    parts=[gtypes.Part(text=content)],
                ))
            elif isinstance(content, list):
                # Could be tool_result blocks — convert to text representation
                text = json.dumps(content, default=str)
                contents.append(gtypes.Content(
                    role=role,
                    parts=[gtypes.Part(text=text)],
                ))
        return contents

    def _build_tools_config(self, tools: list[dict[str, Any]]) -> Any:
        """Build a Gemini Tool from OpenAI-style function schemas."""
        try:
            from google.genai import types as gtypes
        except ImportError:
            return None

        declarations = []
        for t in tools:
            func = t.get("function", t)  # handle both formats
            declarations.append(gtypes.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters", func.get("input_schema", {})),
            ))
        return gtypes.Tool(function_declarations=declarations)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Convert google.genai GenerateContentResponse → LLMResponse."""
        text = ""
        tool_calls: list[ToolCall] = []

        try:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=fc.id if hasattr(fc, "id") else fc.name,
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

            finish_reason = candidate.finish_reason
            stop_reason = self._normalize_stop_reason(str(finish_reason))
        except (AttributeError, IndexError):
            stop_reason = "end_turn"

        usage = LLMUsage(
            input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0)
            if response.usage_metadata else 0,
            output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0)
            if response.usage_metadata else 0,
        )

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else stop_reason,
            usage=usage,
        )

    @staticmethod
    def _normalize_stop_reason(reason: str | None) -> str:
        reason_lower = (reason or "").lower()
        if "stop" in reason_lower or "max_token" in reason_lower:
            return "end_turn"
        if "tool" in reason_lower or "function" in reason_lower:
            return "tool_use"
        return "end_turn"

    # ------------------------------------------------------------------
    # Message format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_tool_schemas(tool_definitions: list[Any]) -> list[dict[str, Any]]:
        """Convert ToolDefinition list to OpenAI function format (used to build FunctionDeclaration)."""
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
        """Build tool result messages for the next turn."""
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
