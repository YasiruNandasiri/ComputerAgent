"""
ContextManager — token estimation, image pruning, and conversation compaction.

Responsible for keeping the conversation within the model's context window by:
  1. Pruning stale screenshot images every turn (cheap, no LLM call).
  2. Compacting old turns into a summary before each LLM call when usage
     approaches the threshold.

Provider-format awareness:
  "openai"    — assistant has tool_calls list; results are role:tool messages;
                screenshots follow as a role:user image_url message.
  "anthropic" — assistant content is a list of blocks (tool_use); results are a
                role:user message with tool_result blocks.
  "plain"     — no structured tool calls; all messages are singletons.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from computer_agent.logging_setup import get_logger

if TYPE_CHECKING:
    from computer_agent.llm.base import BaseLLMProvider

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public utility — used by providers too
# ---------------------------------------------------------------------------

def truncate_middle(s: str, max_chars: int) -> str:
    """Truncate a string to max_chars, keeping head and tail with a note in between."""
    if len(s) <= max_chars:
        return s
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return s[:head] + f"\n... [truncated {len(s) - max_chars} chars] ...\n" + s[-tail:]


# ---------------------------------------------------------------------------
# Summarization prompt (module constant)
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = (
    "Summarize this agent-execution transcript in under 400 words. "
    "Preserve: (1) the user's goal, (2) what has been accomplished step by step, "
    "(3) key facts discovered (file paths, URLs, ids, on-screen values), "
    "(4) errors hit and what was tried, (5) what remains to be done. "
    "Do not invent details."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_assistant_with_tool_calls(msg: dict[str, Any], fmt: str) -> bool:
    if fmt == "openai":
        return (
            msg.get("role") == "assistant"
            and bool(msg.get("tool_calls"))
        )
    if fmt == "anthropic":
        content = msg.get("content")
        return (
            msg.get("role") == "assistant"
            and isinstance(content, list)
            and any(b.get("type") == "tool_use" for b in content)
        )
    return False


def _is_tool_result_message(msg: dict[str, Any], fmt: str) -> bool:
    if fmt == "openai":
        return msg.get("role") == "tool"
    if fmt == "anthropic":
        content = msg.get("content")
        return (
            msg.get("role") == "user"
            and isinstance(content, list)
            and any(b.get("type") == "tool_result" for b in content)
        )
    return False


def _is_image_followup_user(msg: dict[str, Any]) -> bool:
    """True for an OpenAI follow-up user message that carries image_url parts."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(p.get("type") == "image_url" for p in content)


def _has_image_part(msg: dict[str, Any]) -> bool:
    """Return True if the message contains any image content."""
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if part.get("type") in ("image_url", "image"):
                return True
    return False


def _replace_image_parts_with_stub(msg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of msg with image content replaced by a stub string."""
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    new_parts: list[dict[str, Any]] = []
    changed = False
    for part in content:
        if part.get("type") in ("image_url", "image"):
            new_parts.append({
                "type": "text",
                "text": "[Screenshot removed from history to save context — take a new one if needed.]",
            })
            changed = True
        else:
            new_parts.append(part)
    if not changed:
        return msg
    return {**msg, "content": new_parts}


def _text_of(msg: dict[str, Any]) -> str:
    """Extract plain-text content from any message format."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("type", "")
                if t == "text":
                    parts.append(part.get("text", ""))
                elif t == "tool_result":
                    inner = part.get("content", "")
                    if isinstance(inner, str):
                        parts.append(inner)
        return "\n".join(p for p in parts if p)
    return ""


def _render_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render a message list as plain text for the summariser, capped at ~40k chars."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("type", "")
                if t == "text":
                    lines.append(f"{role}: {part.get('text', '')[:500]}")
                elif t == "tool_use":
                    lines.append(f"tool_call {part.get('name')}: {json.dumps(part.get('input', {}), separators=(',', ':'))[:200]}")
                elif t in ("image_url", "image"):
                    lines.append(f"{role}: [screenshot]")
                elif t == "tool_result":
                    inner = part.get("content", "")
                    lines.append(f"tool_result: {str(inner)[:300]}")
        elif isinstance(content, str):
            lines.append(f"{role}: {content[:500]}")
        # tool role messages
        if msg.get("role") == "tool":
            lines.append(f"tool({msg.get('tool_call_id', '')[:8]}): {str(msg.get('content', ''))[:300]}")

    full = "\n".join(lines)
    return truncate_middle(full, 40_000)


def _mechanical_digest(messages: list[dict[str, Any]]) -> str:
    """Fallback summary that never calls an LLM — extracts tool names, errors, paths/URLs."""
    tools_called: list[str] = []
    errors: list[str] = []
    artifacts: list[str] = []

    for msg in messages:
        content = msg.get("content")
        role = msg.get("role", "")

        # Collect tool names from OpenAI assistant messages
        for tc in msg.get("tool_calls") or []:
            name = tc.get("function", {}).get("name") or tc.get("name", "")
            if name:
                tools_called.append(name)

        # Collect from Anthropic assistant blocks
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    tools_called.append(part.get("name", ""))

        # Collect errors and file/URL artefacts from text content
        text = _text_of(msg) if role in ("tool", "user", "assistant") else ""
        if "error" in text.lower() or "failed" in text.lower():
            errors.append(text[:200])
        # Naive path/URL extraction
        for m in re.findall(r"(?:/[\w./\-]+|https?://\S+)", text):
            artifacts.append(m[:80])

    parts = []
    if tools_called:
        parts.append("Tools called: " + ", ".join(tools_called[-20:]))
    if errors:
        parts.append("Last errors: " + "; ".join(errors[-3:]))
    if artifacts:
        parts.append("Key artifacts: " + ", ".join(dict.fromkeys(artifacts)[:10]))
    return "\n".join(parts) or "No tool calls or errors recorded."


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Manages conversation context: token estimation, image pruning, compaction.

    provider_format: "openai" | "anthropic" | "plain"
    """

    def __init__(self, model: str, provider_format: str) -> None:
        self._model = model
        self._fmt = provider_format
        self._context_window: int | None = None  # resolved lazily

    # ------------------------------------------------------------------
    # Context window resolution
    # ------------------------------------------------------------------

    def context_window(self) -> int:
        if self._context_window is not None:
            return self._context_window

        from computer_agent.config import settings

        if settings.context_window_tokens > 0:
            self._context_window = settings.context_window_tokens
            logger.debug("context_window_from_config", tokens=self._context_window)
            return self._context_window

        try:
            import litellm
            info = litellm.get_model_info(self._model)
            val = info.get("max_input_tokens") or info.get("max_tokens")
            if val and isinstance(val, int) and val > 0:
                self._context_window = val
                logger.debug("context_window_from_litellm", tokens=self._context_window, model=self._model)
                return self._context_window
        except Exception:
            pass

        self._context_window = 128_000
        logger.warning(
            "context_window_fallback",
            tokens=self._context_window,
            model=self._model,
            hint="Set CONTEXT_WINDOW_TOKENS env var to override.",
        )
        return self._context_window

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        last_actual: int = 0,
    ) -> int:
        primary = self._litellm_token_count(messages, system)
        if primary is None:
            primary = self._heuristic_token_count(messages, system, tools)
        # Trust whichever is larger: our estimate or the last API-reported value
        return max(primary, last_actual)

    def _litellm_token_count(
        self,
        messages: list[dict[str, Any]],
        system: str,
    ) -> int | None:
        try:
            import litellm
            sys_msg = [{"role": "system", "content": system}] if system else []
            count = litellm.token_counter(model=self._model, messages=sys_msg + messages)
            return int(count)
        except Exception:
            return None

    def _heuristic_token_count(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
    ) -> int:
        total = len(system) // 4
        total += 6_000  # flat tool schema budget
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                        total += 1_600  # flat per-image budget
                    else:
                        total += len(json.dumps(part, default=str)) // 4
            else:
                total += len(json.dumps(msg, default=str)) // 4
        return total

    # ------------------------------------------------------------------
    # Compaction threshold check
    # ------------------------------------------------------------------

    def needs_compaction(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]],
        last_actual: int = 0,
    ) -> bool:
        from computer_agent.config import settings
        estimated = self.estimate_tokens(messages, system, tools, last_actual=last_actual)
        threshold = self.context_window() * settings.context_compact_threshold
        return estimated >= threshold

    # ------------------------------------------------------------------
    # Image pruning (cheap, no LLM)
    # ------------------------------------------------------------------

    def prune_old_images(
        self,
        messages: list[dict[str, Any]],
        keep: int | None = None,
    ) -> int:
        """Replace old screenshot image parts with a stub, keeping the newest `keep`.

        Returns the number of images pruned.
        """
        from computer_agent.config import settings
        max_keep = keep if keep is not None else settings.max_images_in_context

        # Walk end→start, count images, stub out extras
        image_count = 0
        pruned = 0
        # First pass: count total images
        for msg in messages:
            if _has_image_part(msg):
                image_count += 1

        if image_count <= max_keep:
            return 0

        # Second pass end→start: keep newest max_keep, stub the rest
        images_kept = 0
        for i in range(len(messages) - 1, -1, -1):
            if _has_image_part(messages[i]):
                if images_kept < max_keep:
                    images_kept += 1
                else:
                    messages[i] = _replace_image_parts_with_stub(messages[i])
                    pruned += 1

        if pruned:
            logger.debug("images_pruned", count=pruned, kept=images_kept)
        return pruned

    # ------------------------------------------------------------------
    # Message grouping
    # ------------------------------------------------------------------

    def split_into_groups(
        self, messages: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Split messages into atomic groups that must not be split across compaction."""
        groups: list[list[dict[str, Any]]] = []
        i = 0
        fmt = self._fmt
        while i < len(messages):
            if _is_assistant_with_tool_calls(messages[i], fmt):
                j = i + 1
                # Collect all contiguous tool-result messages
                while j < len(messages) and _is_tool_result_message(messages[j], fmt):
                    j += 1
                # For OpenAI: also collect an immediately-following image user message
                if fmt == "openai" and j < len(messages) and _is_image_followup_user(messages[j]):
                    j += 1
                groups.append(messages[i:j])
                i = j
            else:
                groups.append([messages[i]])
                i += 1
        return groups

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def compact(
        self,
        messages: list[dict[str, Any]],
        llm: "BaseLLMProvider",
        aggressive: bool = False,
    ) -> list[dict[str, Any]]:
        """Compact conversation history by summarizing old turns.

        Never raises — falls back to a mechanical digest if the summarizer fails.
        """
        from computer_agent.config import settings

        groups = self.split_into_groups(messages)
        keep = 2 if aggressive else settings.context_keep_recent_groups
        if len(groups) <= keep + 1:
            return messages  # nothing to fold

        goal_text = _text_of(groups[0][0])
        middle = [m for g in groups[1:-keep] for m in g]
        tail = [m for g in groups[-keep:] for m in g]

        transcript = _render_for_summary(middle)
        summary = await self._summarize(llm, transcript)

        # Rebuild: single user message = goal + summary
        first_content = (
            goal_text
            + "\n\n--- Progress summary (earlier steps compacted) ---\n"
            + summary
        )

        # Anthropic alternation safety: if tail starts with a plain user message,
        # merge it into first to avoid two consecutive user messages
        if (
            self._fmt == "anthropic"
            and tail
            and tail[0].get("role") == "user"
            and isinstance(tail[0].get("content"), str)
        ):
            first_content += "\n\n" + tail[0]["content"]
            tail = tail[1:]

        new_history = [{"role": "user", "content": first_content}] + tail

        logger.info(
            "context_compacted",
            before=len(messages),
            after=len(new_history),
            aggressive=aggressive,
        )
        return new_history

    async def _summarize(self, llm: "BaseLLMProvider", transcript: str) -> str:
        """Call LLM to summarize; fall back to mechanical digest on any failure."""
        from computer_agent.config import settings

        model_override = settings.compaction_model or None
        try:
            resp = await llm.generate(
                messages=[{"role": "user", "content": _SUMMARY_PROMPT + "\n\n" + transcript}],
                system="",
                tools=None,
                max_tokens=1024,
                **({"model": model_override} if model_override else {}),
            )
            return resp.text or _mechanical_digest([])
        except Exception as e:
            logger.warning("compaction_summarizer_failed", error=str(e))
            return _mechanical_digest([])
