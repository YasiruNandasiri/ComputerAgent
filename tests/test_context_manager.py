"""
Unit tests for ContextManager.

Pure unit tests — no real LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from computer_agent.llm.context_manager import (
    ContextManager,
    _mechanical_digest,
    truncate_middle,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic conversations
# ---------------------------------------------------------------------------

def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _assistant_tool_calls_openai(tool_id: str, tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    }


def _tool_result_openai(tool_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_id, "content": content}


def _image_followup_user(label: str = "screenshot") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"[{label}]"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/abc"}},
        ],
    }


def _assistant_tool_use_anthropic(tool_id: str, tool_name: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
        ],
    }


def _tool_result_anthropic(tool_id: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
    }


def _image_tool_result_anthropic(tool_id: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
                    {"type": "text", "text": "Screenshot captured (100x100)"},
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# truncate_middle
# ---------------------------------------------------------------------------

def test_truncate_middle_short():
    s = "hello"
    assert truncate_middle(s, 100) == s


def test_truncate_middle_long():
    s = "a" * 200
    result = truncate_middle(s, 100)
    assert len(result) <= 140  # 100 chars + truncation notice overhead
    assert "truncated" in result
    assert result.startswith("a" * 70)   # head is 70% of 100
    assert result.endswith("a" * 30)     # tail is 30% of 100


# ---------------------------------------------------------------------------
# Group splitting — OpenAI
# ---------------------------------------------------------------------------

def test_split_groups_openai_simple():
    cm = ContextManager("gpt-4o", "openai")
    msgs = [
        _user("goal"),
        _assistant_tool_calls_openai("t1", "take_screenshot"),
        _tool_result_openai("t1", "base64..."),
        _image_followup_user("screenshot"),
        _assistant_text("done"),
    ]
    groups = cm.split_into_groups(msgs)
    assert len(groups) == 3
    # group 0: user goal (singleton)
    assert groups[0] == [msgs[0]]
    # group 1: assistant + tool result + image followup (3 messages)
    assert len(groups[1]) == 3
    assert groups[1][0] == msgs[1]
    assert groups[1][1] == msgs[2]
    assert groups[1][2] == msgs[3]
    # group 2: assistant text (singleton)
    assert groups[2] == [msgs[4]]


def test_split_groups_openai_no_image_followup():
    cm = ContextManager("gpt-4o", "openai")
    msgs = [
        _user("goal"),
        _assistant_tool_calls_openai("t1", "run_command"),
        _tool_result_openai("t1", "output"),
        _assistant_text("done"),
    ]
    groups = cm.split_into_groups(msgs)
    assert len(groups) == 3
    assert len(groups[1]) == 2  # assistant + tool result only


# ---------------------------------------------------------------------------
# Group splitting — Anthropic
# ---------------------------------------------------------------------------

def test_split_groups_anthropic():
    cm = ContextManager("claude-3-5-sonnet-20241022", "anthropic")
    msgs = [
        _user("goal"),
        _assistant_tool_use_anthropic("t1", "take_screenshot"),
        _tool_result_anthropic("t1", "ok"),
        _assistant_text("done"),
    ]
    groups = cm.split_into_groups(msgs)
    assert len(groups) == 3
    assert len(groups[1]) == 2  # assistant tool_use + user tool_result


# ---------------------------------------------------------------------------
# Pairing invariants helper
# ---------------------------------------------------------------------------

def _assert_valid_openai_pairing(messages: list[dict]) -> None:
    """Assert that every OpenAI tool_calls assistant msg is immediately followed by its tool results."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            expected_ids = {tc["id"] for tc in msg["tool_calls"]}
            i += 1
            found_ids = set()
            while i < len(messages) and messages[i].get("role") == "tool":
                found_ids.add(messages[i]["tool_call_id"])
                i += 1
            assert found_ids == expected_ids, (
                f"tool_call_id mismatch: expected {expected_ids}, got {found_ids}"
            )
        else:
            i += 1


def _assert_valid_anthropic_pairing(messages: list[dict]) -> None:
    """Assert that every Anthropic tool_use assistant msg is followed by tool_result user msg."""
    i = 0
    while i < len(messages):
        msg = messages[i]
        content = msg.get("content")
        if msg.get("role") == "assistant" and isinstance(content, list):
            tool_use_ids = {b["id"] for b in content if b.get("type") == "tool_use"}
            if tool_use_ids:
                assert i + 1 < len(messages), "tool_use message has no following message"
                next_msg = messages[i + 1]
                next_content = next_msg.get("content", [])
                result_ids = {
                    b["tool_use_id"]
                    for b in next_content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                }
                assert result_ids == tool_use_ids
                i += 2
                continue
        i += 1


# ---------------------------------------------------------------------------
# Compaction — OpenAI
# ---------------------------------------------------------------------------

def _make_long_openai_conversation(n_rounds: int = 15) -> list[dict]:
    msgs = [_user("Do a complex task")]
    for i in range(n_rounds):
        tid = f"t{i}"
        msgs.append(_assistant_tool_calls_openai(tid, "some_tool"))
        msgs.append(_tool_result_openai(tid, f"result {i}"))
    msgs.append(_assistant_text("All done."))
    return msgs


@pytest.mark.asyncio
async def test_compact_openai_preserves_pairing():
    cm = ContextManager("gpt-4o", "openai")
    msgs = _make_long_openai_conversation(15)

    stub_llm = MagicMock()
    stub_llm.generate = AsyncMock(return_value=MagicMock(text="SUMMARY"))

    import computer_agent.config as cfg
    with patch.object(cfg.settings, "context_keep_recent_groups", 4):
        compacted = await cm.compact(msgs, stub_llm)

    assert compacted[0]["role"] == "user"
    _assert_valid_openai_pairing(compacted)
    # goal text should be preserved somewhere in first message
    assert "complex task" in compacted[0]["content"].lower() or "summary" in compacted[0]["content"].lower()


@pytest.mark.asyncio
async def test_compact_no_op_when_short():
    cm = ContextManager("gpt-4o", "openai")
    msgs = _make_long_openai_conversation(3)  # only 3 rounds

    stub_llm = MagicMock()
    stub_llm.generate = AsyncMock(return_value=MagicMock(text="SUMMARY"))

    import computer_agent.config as cfg
    with patch.object(cfg.settings, "context_keep_recent_groups", 6):
        compacted = await cm.compact(msgs, stub_llm)

    assert compacted == msgs  # unchanged


@pytest.mark.asyncio
async def test_compact_falls_back_to_digest_on_llm_error():
    cm = ContextManager("gpt-4o", "openai")
    msgs = _make_long_openai_conversation(15)

    stub_llm = MagicMock()
    stub_llm.generate = AsyncMock(side_effect=RuntimeError("LLM offline"))

    import computer_agent.config as cfg
    with patch.object(cfg.settings, "context_keep_recent_groups", 4):
        # Should not raise
        compacted = await cm.compact(msgs, stub_llm)

    assert compacted[0]["role"] == "user"
    _assert_valid_openai_pairing(compacted)


# ---------------------------------------------------------------------------
# Compaction — Anthropic
# ---------------------------------------------------------------------------

def _make_long_anthropic_conversation(n_rounds: int = 15) -> list[dict]:
    msgs = [_user("Do a complex task")]
    for i in range(n_rounds):
        tid = f"t{i}"
        msgs.append(_assistant_tool_use_anthropic(tid, "some_tool"))
        msgs.append(_tool_result_anthropic(tid, f"result {i}"))
    msgs.append(_assistant_text("All done."))
    return msgs


@pytest.mark.asyncio
async def test_compact_anthropic_preserves_pairing():
    cm = ContextManager("claude-3-5-sonnet", "anthropic")
    msgs = _make_long_anthropic_conversation(15)

    stub_llm = MagicMock()
    stub_llm.generate = AsyncMock(return_value=MagicMock(text="SUMMARY"))

    import computer_agent.config as cfg
    with patch.object(cfg.settings, "context_keep_recent_groups", 4):
        compacted = await cm.compact(msgs, stub_llm)

    assert compacted[0]["role"] == "user"
    _assert_valid_anthropic_pairing(compacted)


# ---------------------------------------------------------------------------
# Image pruning
# ---------------------------------------------------------------------------

def _make_conversation_with_images(n_images: int) -> list[dict]:
    msgs = [_user("goal")]
    for i in range(n_images):
        tid = f"it{i}"
        msgs.append(_assistant_tool_calls_openai(tid, "take_screenshot"))
        msgs.append(_tool_result_openai(tid, "see next"))
        msgs.append(_image_followup_user(f"shot{i}"))
    return msgs


def test_prune_images_keeps_newest():
    cm = ContextManager("gpt-4o", "openai")
    msgs = _make_conversation_with_images(5)

    import computer_agent.config as cfg
    with patch.object(cfg.settings, "max_images_in_context", 2):
        pruned = cm.prune_old_images(msgs)

    assert pruned == 3  # 5 - 2 = 3 removed
    # Count remaining image parts
    remaining_images = sum(
        1 for m in msgs
        if isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
    )
    assert remaining_images == 2


def test_prune_images_noop_when_within_limit():
    cm = ContextManager("gpt-4o", "openai")
    msgs = _make_conversation_with_images(2)

    import computer_agent.config as cfg
    with patch.object(cfg.settings, "max_images_in_context", 2):
        pruned = cm.prune_old_images(msgs)

    assert pruned == 0


# ---------------------------------------------------------------------------
# Token estimation fallback
# ---------------------------------------------------------------------------

def test_token_estimation_falls_back_to_heuristic():
    cm = ContextManager("unknown-azure-deployment", "openai")
    msgs = [_user("hello world"), _assistant_text("hi")]

    with patch("litellm.token_counter", side_effect=Exception("unknown model")):
        est = cm.estimate_tokens(msgs, "system prompt", [])

    assert est > 0  # heuristic produced something


# ---------------------------------------------------------------------------
# Mechanical digest
# ---------------------------------------------------------------------------

def test_mechanical_digest_extracts_tool_names():
    msgs = [
        _assistant_tool_calls_openai("t1", "take_screenshot"),
        _tool_result_openai("t1", "ok"),
        _assistant_tool_calls_openai("t2", "run_command"),
        _tool_result_openai("t2", "Error: command not found"),
    ]
    digest = _mechanical_digest(msgs)
    assert "take_screenshot" in digest
    assert "run_command" in digest
    assert "error" in digest.lower()
