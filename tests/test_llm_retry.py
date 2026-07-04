"""
Tests for LiteLLM provider retry logic.

No real API calls — litellm.acompletion is monkeypatched with async fakes.
asyncio.sleep is also patched to avoid real delays and record call args.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from computer_agent.llm.errors import LLMContextWindowError, LLMRateLimitError
from computer_agent.llm.providers.litellm import LiteLLMProvider


# ---------------------------------------------------------------------------
# Helpers to build fake litellm module
# ---------------------------------------------------------------------------

def _make_litellm_fake():
    """Return a minimal fake litellm module with the exception hierarchy."""
    mod = types.ModuleType("litellm_fake")

    class LiteLLMException(Exception): pass
    class RateLimitError(LiteLLMException): pass
    class ContextWindowExceededError(LiteLLMException): pass
    class BadRequestError(LiteLLMException): pass
    class APIConnectionError(LiteLLMException): pass
    class ServiceUnavailableError(LiteLLMException): pass
    class InternalServerError(LiteLLMException): pass
    class Timeout(LiteLLMException): pass
    class AuthenticationError(LiteLLMException): pass

    mod.RateLimitError = RateLimitError
    mod.ContextWindowExceededError = ContextWindowExceededError
    mod.BadRequestError = BadRequestError
    mod.APIConnectionError = APIConnectionError
    mod.ServiceUnavailableError = ServiceUnavailableError
    mod.InternalServerError = InternalServerError
    mod.Timeout = Timeout
    mod.AuthenticationError = AuthenticationError
    return mod


def _fake_response():
    """Minimal object that looks like a litellm acompletion response."""
    choice = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _patch_settings(**kwargs):
    """Patch multiple attributes on the settings singleton using patch.object."""
    import computer_agent.config as cfg
    return [patch.object(cfg.settings, k, v) for k, v in kwargs.items()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def provider():
    return LiteLLMProvider(model="azure/gpt-4o", api_key="fake")


@pytest.fixture()
def fake_litellm():
    return _make_litellm_fake()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_then_success(provider, fake_litellm):
    """RateLimitError twice then success → 3 total calls, increasing delays."""
    call_count = 0
    delays: list[float] = []

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise fake_litellm.RateLimitError("rate limit")
        return _fake_response()

    async def fake_sleep(secs):
        delays.append(secs)

    fake_litellm.acompletion = fake_acompletion

    patches = _patch_settings(llm_max_retries=3, llm_retry_base_delay=2.0, llm_retry_max_delay=60.0)
    with patches[0], patches[1], patches[2], patch("asyncio.sleep", side_effect=fake_sleep):
        resp = await provider._acompletion_with_retry(fake_litellm, {"model": "x", "messages": []})

    assert call_count == 3
    assert len(delays) == 2
    assert all(d > 0 for d in delays)


@pytest.mark.asyncio
async def test_always_rate_limit_raises(provider, fake_litellm):
    """Always RateLimitError → LLMRateLimitError after max_retries+1 calls."""
    call_count = 0

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        raise fake_litellm.RateLimitError("always")

    fake_litellm.acompletion = fake_acompletion

    async def fake_sleep(_):
        pass

    patches = _patch_settings(llm_max_retries=3, llm_retry_base_delay=0.01, llm_retry_max_delay=1.0)
    with patches[0], patches[1], patches[2], patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(LLMRateLimitError):
            await provider._acompletion_with_retry(fake_litellm, {"model": "x", "messages": []})

    assert call_count == 4  # 1 initial + 3 retries


@pytest.mark.asyncio
async def test_retry_after_header_respected(provider, fake_litellm):
    """retry-after header → delay is at least that value."""
    delays: list[float] = []

    async def fake_acompletion(**kwargs):
        err = fake_litellm.RateLimitError("limited")
        err.response = MagicMock()
        err.response.headers = {"retry-after": "10"}
        raise err

    async def fake_sleep(secs):
        delays.append(secs)

    fake_litellm.acompletion = fake_acompletion

    patches = _patch_settings(llm_max_retries=1, llm_retry_base_delay=0.01, llm_retry_max_delay=120.0)
    with patches[0], patches[1], patches[2], patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(LLMRateLimitError):
            await provider._acompletion_with_retry(fake_litellm, {"model": "x", "messages": []})

    assert delays[0] >= 10.0


@pytest.mark.asyncio
async def test_context_window_not_retried(provider, fake_litellm):
    """ContextWindowExceededError → exactly 1 call, LLMContextWindowError raised."""
    call_count = 0

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        raise fake_litellm.ContextWindowExceededError("too big")

    fake_litellm.acompletion = fake_acompletion

    async def fake_sleep(_):
        pass

    patches = _patch_settings(llm_max_retries=3)
    with patches[0], patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(LLMContextWindowError):
            await provider._acompletion_with_retry(fake_litellm, {"model": "x", "messages": []})

    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_after_capped_at_max_delay(provider, fake_litellm):
    """Server sends retry-after: 3600 -> delay capped at llm_retry_max_delay (60s)."""
    delays: list[float] = []

    async def fake_acompletion(**kwargs):
        err = fake_litellm.RateLimitError("quota")
        err.response = MagicMock()
        err.response.headers = {"retry-after": "3600"}
        raise err

    async def fake_sleep(secs):
        delays.append(secs)

    fake_litellm.acompletion = fake_acompletion

    patches = _patch_settings(llm_max_retries=1, llm_retry_base_delay=2.0, llm_retry_max_delay=60.0)
    with patches[0], patches[1], patches[2], patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(LLMRateLimitError):
            await provider._acompletion_with_retry(fake_litellm, {"model": "x", "messages": []})

    assert delays[0] <= 60.0, f"Delay was {delays[0]}s, expected <= 60s (capped, not 3600)"


@pytest.mark.asyncio
async def test_retry_after_ms_header_parsed(provider, fake_litellm):
    """Azure retry-after-ms: 5000 -> delay is at least 5.0 seconds."""
    delays: list[float] = []

    async def fake_acompletion(**kwargs):
        err = fake_litellm.RateLimitError("quota")
        err.response = MagicMock()
        err.response.headers = {"retry-after-ms": "5000"}
        raise err

    async def fake_sleep(secs):
        delays.append(secs)

    fake_litellm.acompletion = fake_acompletion

    patches = _patch_settings(llm_max_retries=1, llm_retry_base_delay=0.01, llm_retry_max_delay=120.0)
    with patches[0], patches[1], patches[2], patch("asyncio.sleep", side_effect=fake_sleep):
        with pytest.raises(LLMRateLimitError):
            await provider._acompletion_with_retry(fake_litellm, {"model": "x", "messages": []})

    assert delays[0] >= 5.0, f"Delay was {delays[0]}s, expected >= 5.0 (from retry-after-ms: 5000)"
