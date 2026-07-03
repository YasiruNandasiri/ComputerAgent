"""
Tests for LLM provider adapters.
All tests use mocks — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from computer_agent.llm.base import LLMResponse, LLMUsage, ToolCall
from computer_agent.tools.base import RiskLevel, ToolDefinition, ToolResult


# ---------------------------------------------------------------------------
# Helpers: minimal ToolDefinition for schema generation tests
# ---------------------------------------------------------------------------

def make_tool_def(name: str, desc: str = "A test tool") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=desc,
        func=lambda: ToolResult.ok(),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        category="test",
    )


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_format_tool_schemas(self):
        from computer_agent.llm.providers.anthropic import AnthropicProvider

        tools = [make_tool_def("read_file", "Read a file")]
        schemas = AnthropicProvider.format_tool_schemas(tools)

        assert len(schemas) == 1
        assert schemas[0]["name"] == "read_file"
        assert schemas[0]["description"] == "Read a file"
        assert "input_schema" in schemas[0]
        assert schemas[0]["input_schema"]["type"] == "object"

    def test_parse_response_text_only(self):
        from computer_agent.llm.providers.anthropic import AnthropicProvider

        # Build a mock Anthropic Message
        block = MagicMock()
        block.text = "Hello!"
        block.type = "text"

        mock_response = MagicMock()
        mock_response.content = [block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        result = provider._parse_response(mock_response)

        assert result.text == "Hello!"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_parse_response_tool_use(self):
        from computer_agent.llm.providers.anthropic import AnthropicProvider

        block = MagicMock()
        block.type = "tool_use"
        block.id = "toolu_01"
        block.name = "read_file"
        block.input = {"path": "/tmp/test.txt"}
        del block.text  # ensure no text attribute

        mock_response = MagicMock()
        mock_response.content = [block]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 30

        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        result = provider._parse_response(mock_response)

        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_01"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].arguments == {"path": "/tmp/test.txt"}
        assert result.stop_reason == "tool_use"

    def test_format_tool_result_messages_success(self):
        from computer_agent.llm.providers.anthropic import AnthropicProvider

        tc = ToolCall(id="toolu_01", name="read_file", arguments={"path": "/tmp/x"})
        result = ToolResult.ok(output="file contents here")

        content = AnthropicProvider.format_tool_result_messages([tc], [result])
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "toolu_01"
        assert content[0]["content"] == "file contents here"

    def test_format_tool_result_messages_error(self):
        from computer_agent.llm.providers.anthropic import AnthropicProvider

        tc = ToolCall(id="toolu_02", name="delete_file", arguments={"path": "/tmp/x"})
        result = ToolResult.fail(error="Permission denied")

        content = AnthropicProvider.format_tool_result_messages([tc], [result])
        assert content[0]["content"] == "Error: Permission denied"

    @pytest.mark.asyncio
    async def test_generate_calls_client(self):
        from computer_agent.llm.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(model="claude-sonnet-4-20250514", api_key="test-key")

        # Mock the Anthropic client
        block = MagicMock()
        block.text = "Done."
        block.type = "text"
        mock_resp = MagicMock()
        mock_resp.content = [block]
        mock_resp.stop_reason = "end_turn"
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 3

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_resp)
        provider._client = mock_client

        response = await provider.generate(
            [{"role": "user", "content": "hello"}],
            system="You are an assistant.",
        )

        assert response.text == "Done."
        assert response.is_done is True
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["system"] == "You are an assistant."


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_format_tool_schemas(self):
        from computer_agent.llm.providers.openai import OpenAIProvider

        tools = [make_tool_def("browser_click", "Click at coordinates")]
        schemas = OpenAIProvider.format_tool_schemas(tools)

        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "browser_click"
        assert "parameters" in schemas[0]["function"]

    def test_parse_response_text_only(self):
        from computer_agent.llm.providers.openai import OpenAIProvider

        mock_message = MagicMock()
        mock_message.content = "Here is the answer."
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 8

        provider = OpenAIProvider(model="gpt-4o")
        result = provider._parse_response(mock_response)

        assert result.text == "Here is the answer."
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"

    def test_parse_response_tool_calls(self):
        import json
        from computer_agent.llm.providers.openai import OpenAIProvider

        mock_tc = MagicMock()
        mock_tc.id = "call_abc"
        mock_tc.function.name = "list_directory"
        mock_tc.function.arguments = json.dumps({"path": "/tmp"})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 20

        provider = OpenAIProvider(model="gpt-4o")
        result = provider._parse_response(mock_response)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "list_directory"
        assert result.tool_calls[0].arguments == {"path": "/tmp"}
        assert result.stop_reason == "tool_use"

    def test_format_tool_result_messages(self):
        from computer_agent.llm.providers.openai import OpenAIProvider

        tcs = [ToolCall(id="call_1", name="list_directory", arguments={"path": "/tmp"})]
        results = [ToolResult.ok(output=["a.txt", "b.txt"])]

        messages = OpenAIProvider.format_tool_result_messages(tcs, results)

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"
        assert "a.txt" in messages[0]["content"]

    def test_assistant_message_from_tool_calls(self):
        import json
        from computer_agent.llm.providers.openai import OpenAIProvider

        tcs = [ToolCall(id="call_x", name="run_shell_command", arguments={"command": "ls"})]
        msg = OpenAIProvider.assistant_message_from_tool_calls(tcs)

        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["id"] == "call_x"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------

class TestLiteLLMProvider:
    def test_format_tool_schemas(self):
        from computer_agent.llm.providers.litellm import LiteLLMProvider

        tools = [make_tool_def("http_get", "Make an HTTP GET request")]
        schemas = LiteLLMProvider.format_tool_schemas(tools)

        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "http_get"

    def test_supported_models(self):
        from computer_agent.llm.providers.litellm import LiteLLMProvider

        patterns = LiteLLMProvider.supported_models()
        assert any("ollama" in p for p in patterns)
        assert any("huggingface" in p for p in patterns)

    def test_parse_response_text(self):
        from computer_agent.llm.providers.litellm import LiteLLMProvider

        mock_message = MagicMock()
        mock_message.content = "LiteLLM response"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10)

        provider = LiteLLMProvider(model="ollama/llama3")
        result = provider._parse_response(mock_resp)

        assert result.text == "LiteLLM response"
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_generate_calls_litellm(self):
        from computer_agent.llm.providers.litellm import LiteLLMProvider

        mock_message = MagicMock()
        mock_message.content = "Local LLM says hi"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = MagicMock(prompt_tokens=3, completion_tokens=7)

        # Mock litellm at the module level inside the provider module
        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            provider = LiteLLMProvider(model="ollama/llama3")
            response = await provider.generate(
                [{"role": "user", "content": "hello"}],
                system="You are an assistant",
            )

        assert response.text == "Local LLM says hi"
        assert response.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# GoogleProvider
# ---------------------------------------------------------------------------

class TestGoogleProvider:
    def test_format_tool_schemas(self):
        from computer_agent.llm.providers.google import GoogleProvider

        tools = [make_tool_def("take_screenshot", "Capture the screen")]
        schemas = GoogleProvider.format_tool_schemas(tools)

        assert schemas[0]["function"]["name"] == "take_screenshot"

    def test_supported_models(self):
        from computer_agent.llm.providers.google import GoogleProvider
        import re

        patterns = GoogleProvider.supported_models()
        assert any(re.match(p, "gemini-2.5-flash") for p in patterns)
        assert any(re.match(p, "gemini-2.0-flash") for p in patterns)

    def test_normalize_stop_reason(self):
        from computer_agent.llm.providers.google import GoogleProvider

        assert GoogleProvider._normalize_stop_reason("STOP") == "end_turn"
        assert GoogleProvider._normalize_stop_reason("MAX_TOKENS") == "end_turn"
        assert GoogleProvider._normalize_stop_reason("FUNCTION_CALL") == "tool_use"


# ---------------------------------------------------------------------------
# BaseLLMProvider — ABC contract
# ---------------------------------------------------------------------------

class TestBaseLLMProvider:
    def test_cannot_instantiate_abc(self):
        from computer_agent.llm.base import BaseLLMProvider

        with pytest.raises(TypeError):
            BaseLLMProvider()  # type: ignore

    def test_llm_response_properties(self):
        response = LLMResponse(
            text="hello",
            tool_calls=[],
            stop_reason="end_turn",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
        assert response.is_done is True
        assert response.has_tool_calls is False
        assert response.usage.total_tokens == 15

    def test_llm_response_has_tool_calls(self):
        tc = ToolCall(id="x", name="foo", arguments={})
        response = LLMResponse(
            text="",
            tool_calls=[tc],
            stop_reason="tool_use",
        )
        assert response.has_tool_calls is True
        assert response.is_done is False
