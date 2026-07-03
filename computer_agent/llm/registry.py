"""
LLM Registry — resolves model names to provider instances.

Inspired by Google ADK's LLMRegistry pattern:
  - Providers register themselves with regex patterns for model name matching
  - Model names are matched against patterns in priority order
  - Unknown prefixes fall back to LiteLLM (which supports 100+ providers)
  - External providers can self-register via LLMRegistry.register()

Usage:
    provider = LLMRegistry.resolve("claude-sonnet-4-20250514")
    provider = LLMRegistry.resolve("gpt-4o")
    provider = LLMRegistry.resolve("ollama/llama3")
    provider = LLMRegistry.resolve("gemini-2.5-flash")

    # Register a custom provider at runtime
    LLMRegistry.register(MyCustomProvider)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from computer_agent.logging_setup import get_logger

if TYPE_CHECKING:
    from computer_agent.llm.base import BaseLLMProvider

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Registry entries: (pattern, module_path, class_name, priority)
# Providers are lazily imported — their SDK is only loaded when needed.
# ---------------------------------------------------------------------------

_LAZY_ENTRIES: list[tuple[str, str, str, int]] = [
    # (regex_pattern, module_path, class_name, priority)
    # Higher priority (lower number) wins when multiple patterns match
    (r"claude-.*",        "computer_agent.llm.providers.anthropic", "AnthropicProvider", 10),
    (r"gpt-.*",           "computer_agent.llm.providers.openai",    "OpenAIProvider",    10),
    (r"o1-.*",            "computer_agent.llm.providers.openai",    "OpenAIProvider",    10),
    (r"o3-.*",            "computer_agent.llm.providers.openai",    "OpenAIProvider",    10),
    (r"o4-.*",            "computer_agent.llm.providers.openai",    "OpenAIProvider",    10),
    (r"openai/.*",        "computer_agent.llm.providers.openai",    "OpenAIProvider",    10),
    (r"gemini-.*",        "computer_agent.llm.providers.google",    "GoogleProvider",    10),
    (r"google/.*",        "computer_agent.llm.providers.google",    "GoogleProvider",    10),
    (r"ollama/.*",        "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"ollama_chat/.*",   "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"huggingface/.*",   "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"bedrock/.*",       "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"azure/.*",         "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"openrouter/.*",    "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"together_ai/.*",   "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"replicate/.*",     "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"vertex_ai/.*",     "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    (r"litellm/.*",       "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   20),
    # Catch-all: any unrecognized model goes to LiteLLM
    (r".*",               "computer_agent.llm.providers.litellm",   "LiteLLMProvider",   99),
]

# Runtime-registered custom provider classes {pattern: class}
_CUSTOM_PROVIDERS: list[tuple[str, type]] = []


class LLMRegistry:
    """
    Global registry for LLM provider resolution.

    Usage:
        provider = LLMRegistry.resolve("claude-sonnet-4-20250514")

    Model name format:
        "claude-sonnet-4-20250514"   → AnthropicProvider
        "gpt-4o"                     → OpenAIProvider
        "gemini-2.5-flash"           → GoogleProvider
        "ollama/llama3"              → LiteLLMProvider
        "anything-else"              → LiteLLMProvider (catch-all)
    """

    @staticmethod
    def resolve(
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> BaseLLMProvider:
        """
        Resolve a model name string to an instantiated provider.

        Args:
            model: Model name, e.g. "claude-sonnet-4-20250514", "gpt-4o",
                   "ollama/llama3", "gemini-2.5-flash".
            api_key: Override API key (falls back to environment variable).
            api_base: Override base URL (for self-hosted / proxy endpoints).

        Returns:
            An instantiated BaseLLMProvider ready to call.

        Raises:
            ImportError: If the required SDK for the matched provider is not installed.
        """
        # Check custom providers first (highest priority)
        for pattern, provider_cls in _CUSTOM_PROVIDERS:
            if re.match(pattern, model, re.IGNORECASE):
                logger.info("llm_resolved", model=model, provider=provider_cls.__name__, source="custom")
                return LLMRegistry._instantiate(provider_cls, model, api_key, api_base)

        # Match against lazy entries sorted by priority
        entries_sorted = sorted(_LAZY_ENTRIES, key=lambda e: e[3])
        for pattern, module_path, class_name, _priority in entries_sorted:
            if re.match(pattern, model, re.IGNORECASE):
                provider_cls = LLMRegistry._load_provider_class(module_path, class_name)
                logger.info(
                    "llm_resolved",
                    model=model,
                    provider=class_name,
                    pattern=pattern,
                )
                return LLMRegistry._instantiate(provider_cls, model, api_key, api_base)

        # Should never reach here — ".*" catch-all always matches
        raise ValueError(f"Could not resolve LLM provider for model: '{model}'")

    @staticmethod
    def register(
        provider_cls: type,
        pattern: str | None = None,
    ) -> None:
        """
        Register a custom provider class.

        Args:
            provider_cls: A class implementing BaseLLMProvider.
            pattern: Regex pattern for model names this provider handles.
                     If None, uses provider_cls.supported_models()[0].

        Example:
            from my_package import MyAzureProvider
            LLMRegistry.register(MyAzureProvider, pattern=r"my-azure-.*")
        """
        if pattern is None:
            patterns = provider_cls.supported_models()
            if not patterns:
                raise ValueError(f"{provider_cls.__name__} has no supported_models() patterns")
            pattern = patterns[0]

        _CUSTOM_PROVIDERS.insert(0, (pattern, provider_cls))
        logger.info("llm_provider_registered", provider=provider_cls.__name__, pattern=pattern)

    @staticmethod
    def _load_provider_class(module_path: str, class_name: str) -> type:
        """Lazily import and return a provider class."""
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @staticmethod
    def _instantiate(
        provider_cls: type,
        model: str,
        api_key: str | None,
        api_base: str | None,
    ) -> BaseLLMProvider:
        """Instantiate a provider class with the appropriate arguments."""
        import inspect
        sig = inspect.signature(provider_cls.__init__)
        params = set(sig.parameters.keys()) - {"self"}

        kwargs: dict[str, object] = {"model": model}
        if "api_key" in params and api_key is not None:
            kwargs["api_key"] = api_key
        # OpenAIProvider uses base_url; LiteLLMProvider uses api_base
        if "base_url" in params and api_base is not None:
            kwargs["base_url"] = api_base
        elif "api_base" in params and api_base is not None:
            kwargs["api_base"] = api_base

        return provider_cls(**kwargs)

    @staticmethod
    def list_patterns() -> list[str]:
        """Return all registered model name patterns (for debugging)."""
        custom = [p for p, _ in _CUSTOM_PROVIDERS]
        builtin = [p for p, _, _, _ in _LAZY_ENTRIES]
        return custom + builtin
