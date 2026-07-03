"""
Tests for LLMRegistry — model name resolution to provider instances.
"""

from __future__ import annotations

import re

import pytest

from computer_agent.llm.registry import LLMRegistry, _CUSTOM_PROVIDERS


class TestLLMRegistryResolution:
    def test_claude_resolves_to_anthropic(self):
        provider = LLMRegistry.resolve("claude-sonnet-4-20250514")
        assert provider.__class__.__name__ == "AnthropicProvider"

    def test_claude_haiku_resolves_to_anthropic(self):
        provider = LLMRegistry.resolve("claude-haiku-4-20250514")
        assert provider.__class__.__name__ == "AnthropicProvider"

    def test_gpt4o_resolves_to_openai(self):
        provider = LLMRegistry.resolve("gpt-4o")
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_gpt4_resolves_to_openai(self):
        provider = LLMRegistry.resolve("gpt-4-turbo")
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_o1_resolves_to_openai(self):
        provider = LLMRegistry.resolve("o1-mini")
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_o3_resolves_to_openai(self):
        provider = LLMRegistry.resolve("o3-mini")
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_gemini_resolves_to_google(self):
        provider = LLMRegistry.resolve("gemini-2.5-flash")
        assert provider.__class__.__name__ == "GoogleProvider"

    def test_gemini_pro_resolves_to_google(self):
        provider = LLMRegistry.resolve("gemini-2.5-pro")
        assert provider.__class__.__name__ == "GoogleProvider"

    def test_ollama_resolves_to_litellm(self):
        provider = LLMRegistry.resolve("ollama/llama3")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_ollama_chat_resolves_to_litellm(self):
        provider = LLMRegistry.resolve("ollama_chat/mistral")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_huggingface_resolves_to_litellm(self):
        provider = LLMRegistry.resolve("huggingface/meta-llama/Llama-3.1-8B")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_bedrock_resolves_to_litellm(self):
        provider = LLMRegistry.resolve("bedrock/anthropic.claude-3-sonnet")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_azure_resolves_to_litellm(self):
        provider = LLMRegistry.resolve("azure/my-gpt4-deployment")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_openrouter_resolves_to_litellm(self):
        provider = LLMRegistry.resolve("openrouter/meta-llama/llama-3.1-8b")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_unknown_model_falls_back_to_litellm(self):
        provider = LLMRegistry.resolve("my-custom-model-xyz")
        assert provider.__class__.__name__ == "LiteLLMProvider"

    def test_api_key_passed_to_provider(self):
        provider = LLMRegistry.resolve("claude-sonnet-4-20250514", api_key="test-key-123")
        assert provider._api_key == "test-key-123"

    def test_api_base_passed_to_provider(self):
        provider = LLMRegistry.resolve("gpt-4o", api_base="http://localhost:8080/v1")
        # OpenAIProvider stores api_base as _base_url
        assert provider._base_url == "http://localhost:8080/v1"  # type: ignore[attr-defined]


class TestLLMRegistryCustomProviders:
    def setup_method(self):
        """Clear custom providers before each test."""
        _CUSTOM_PROVIDERS.clear()

    def teardown_method(self):
        """Clear custom providers after each test."""
        _CUSTOM_PROVIDERS.clear()

    def test_register_custom_provider_with_pattern(self):
        from computer_agent.llm.base import BaseLLMProvider, LLMResponse
        from typing import Any

        class MyCustomProvider(BaseLLMProvider):
            def __init__(self, model: str, **kwargs: Any) -> None:
                self._model = model

            @classmethod
            def supported_models(cls) -> list[str]:
                return [r"my-custom-.*"]

            async def generate(self, messages, **kwargs) -> LLMResponse:  # type: ignore
                return LLMResponse(text="custom")

        LLMRegistry.register(MyCustomProvider, pattern=r"my-custom-.*")
        provider = LLMRegistry.resolve("my-custom-v1")
        assert provider.__class__.__name__ == "MyCustomProvider"

    def test_custom_provider_takes_priority_over_builtin(self):
        from computer_agent.llm.base import BaseLLMProvider, LLMResponse
        from typing import Any

        class OverrideAnthropicProvider(BaseLLMProvider):
            def __init__(self, model: str, **kwargs: Any) -> None:
                self._model = model

            @classmethod
            def supported_models(cls) -> list[str]:
                return [r"claude-.*"]

            async def generate(self, messages, **kwargs) -> LLMResponse:  # type: ignore
                return LLMResponse(text="overridden")

        LLMRegistry.register(OverrideAnthropicProvider, pattern=r"claude-.*")
        provider = LLMRegistry.resolve("claude-sonnet-4-20250514")
        assert provider.__class__.__name__ == "OverrideAnthropicProvider"

    def test_register_uses_supported_models_when_no_pattern(self):
        from computer_agent.llm.base import BaseLLMProvider, LLMResponse
        from typing import Any

        class AnotherProvider(BaseLLMProvider):
            def __init__(self, model: str, **kwargs: Any) -> None:
                self._model = model

            @classmethod
            def supported_models(cls) -> list[str]:
                return [r"another-model-.*"]

            async def generate(self, messages, **kwargs) -> LLMResponse:  # type: ignore
                return LLMResponse(text="another")

        LLMRegistry.register(AnotherProvider)
        assert any(p == r"another-model-.*" for p, _ in _CUSTOM_PROVIDERS)

    def test_register_raises_on_empty_supported_models(self):
        from computer_agent.llm.base import BaseLLMProvider, LLMResponse
        from typing import Any

        class BadProvider(BaseLLMProvider):
            def __init__(self, model: str, **kwargs: Any) -> None:
                self._model = model

            @classmethod
            def supported_models(cls) -> list[str]:
                return []

            async def generate(self, messages, **kwargs) -> LLMResponse:  # type: ignore
                return LLMResponse(text="bad")

        with pytest.raises(ValueError, match="no supported_models"):
            LLMRegistry.register(BadProvider)


class TestLLMRegistryPatterns:
    def test_list_patterns_includes_builtin(self):
        patterns = LLMRegistry.list_patterns()
        assert any("claude" in p for p in patterns)
        assert any("gpt" in p for p in patterns)
        assert any("gemini" in p for p in patterns)
        assert any("ollama" in p for p in patterns)

    def test_all_builtin_patterns_are_valid_regex(self):
        patterns = LLMRegistry.list_patterns()
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern '{pattern}': {e}")
