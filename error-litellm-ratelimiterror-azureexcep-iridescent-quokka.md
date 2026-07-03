# Plan: Rate-limit retries, context-window management, token optimization & Web chat UI

## Context

Running the agent against Azure OpenAI `gpt-4o` (128k context) via LiteLLM produced two fatal errors:

1. `litellm.RateLimitError` — Azure 429s are not retried anywhere; the generic `except Exception` in `Coordinator.run()` ([coordinator.py:158](computer_agent/coordinator.py#L158)) turns them into "I encountered an error" and the task dies.
2. `litellm.ContextWindowExceededError` — the conversation grew to **398,603 tokens** against a 128k window. `Coordinator._conversation` grows unboundedly; nothing truncates, summarizes, or budget-checks it.

**Root cause of the token blowup (most important finding):** `take_screenshot` ([screenshot.py:16](computer_agent/tools/screen/screenshot.py#L16)) returns the screenshot as a base64 PNG **string**, and the tool-result formatters inject it into the conversation as **plain text**. A 1920×1080 PNG ≈ 800KB base64 ≈ **~300k tokens as text** (base64 tokenizes at ~1 token per 2.5 chars). One screenshot alone explains the 398k blowup. Sent properly as an `image_url` content block, the same screenshot costs gpt-4o ~1–1.7k tokens. **This is the single biggest fix in the plan.**

There is also no persistent interface: `computer-agent run` exits after one task. **Decisions made with the user:** build a **browser-based chat UI** served by the existing FastAPI daemon; compaction/summarization uses the same `PRIMARY_MODEL`.

---

## Part 0 — Background for the implementer (read first)

- `Coordinator._run_agent_loop()` ([coordinator.py:187](computer_agent/coordinator.py#L187)) loops: call LLM → execute tool calls → append results → call LLM again (up to 50 turns). Every call sends the **entire** `self._conversation` + system prompt + ~56 tool schemas (~6k tokens).
- History format is provider-dependent. OpenAI/LiteLLM: `{"role":"assistant","content":None,"tool_calls":[...]}` followed by one `{"role":"tool","tool_call_id":...,"content":...}` per call. **The OpenAI API rejects any request where an assistant `tool_calls` message is not immediately followed by its matching `tool` messages** — every history-editing operation must treat these as an atomic unit. Anthropic's equivalent: `tool_use` blocks in an assistant message + a user message with `tool_result` blocks.
- `LLMResponse.usage` already reports exact `input_tokens`/`output_tokens` per call — a free, exact token meter we'll exploit.
- The daemon (`taskmgr`) constructs Coordinators too, so all fixes inside Coordinator/providers apply to daemon tasks automatically.

### Files changed / added (overview)

| # | Area | Files |
|---|------|-------|
| 1 | Config settings | [config.py](computer_agent/config.py) |
| 2 | Unified LLM exceptions | **NEW** `computer_agent/llm/errors.py` |
| 3 | Retry with exponential backoff | [llm/providers/litellm.py](computer_agent/llm/providers/litellm.py) |
| 4 | Screenshot downscale + JPEG | [tools/screen/screenshot.py](computer_agent/tools/screen/screenshot.py) |
| 5 | Image blocks, truncation, compact JSON | [coordinator.py](computer_agent/coordinator.py), [llm/providers/litellm.py](computer_agent/llm/providers/litellm.py), mirror in openai.py/anthropic.py |
| 6 | ContextManager (compaction) | **NEW** `computer_agent/llm/context_manager.py` |
| 7 | Coordinator integration | [coordinator.py](computer_agent/coordinator.py) |
| 8 | Web chat UI | **NEW** `computer_agent/daemon/web/index.html`, [daemon/api.py](computer_agent/daemon/api.py), [main.py](computer_agent/main.py) |
| 9 | Tests | **NEW** `tests/test_llm_retry.py`, `tests/test_context_manager.py` |

---

## Step 1 — Config settings ([config.py](computer_agent/config.py))

New section `# --- LLM Resilience & Context ---` in `Settings`:

```python
# Retry
llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
llm_retry_base_delay: float = Field(default=2.0, alias="LLM_RETRY_BASE_DELAY")
llm_retry_max_delay: float = Field(default=60.0, alias="LLM_RETRY_MAX_DELAY")

# Context management
context_window_tokens: int = Field(default=0, alias="CONTEXT_WINDOW_TOKENS")   # 0 = auto-detect via litellm
context_compact_threshold: float = Field(default=0.75, alias="CONTEXT_COMPACT_THRESHOLD")
context_keep_recent_groups: int = Field(default=6, alias="CONTEXT_KEEP_RECENT_GROUPS")
compaction_model: str = Field(default="", alias="COMPACTION_MODEL")            # "" = primary model

# Token optimization
max_images_in_context: int = Field(default=2, alias="MAX_IMAGES_IN_CONTEXT")
tool_result_max_chars: int = Field(default=8000, alias="TOOL_RESULT_MAX_CHARS")
screenshot_max_dimension: int = Field(default=1440, alias="SCREENSHOT_MAX_DIMENSION")
screenshot_jpeg_quality: int = Field(default=60, alias="SCREENSHOT_JPEG_QUALITY")
```

`CONTEXT_WINDOW_TOKENS` exists as an override because Azure deployment names (`azure/<deployment>`) are often missing from litellm's model map, so auto-detection can fail.

## Step 2 — Unified exceptions (NEW `computer_agent/llm/errors.py`)

The coordinator is provider-agnostic and must not import litellm exception classes. Providers translate native errors into:

```python
class LLMError(Exception): ...
class LLMRateLimitError(LLMError):          # raised only after retries are exhausted
    def __init__(self, message, retry_after: float | None = None): ...
class LLMContextWindowError(LLMError): ...  # prompt too big — retrying is pointless
class LLMTransientError(LLMError): ...      # connection / 5xx / timeout
```

Re-export from `computer_agent/llm/__init__.py`.

## Step 3 — Exponential-backoff retry in `LiteLLMProvider.generate()`

**Where:** inside the provider (replacing the bare `litellm.acompletion(**kwargs)` at [litellm.py:97](computer_agent/llm/providers/litellm.py#L97)), NOT in the coordinator — the litellm exception types can only be caught precisely here. Do **not** also set litellm's `num_retries` (would multiply retries 3×3).

**What to retry:** `litellm.RateLimitError` (Azure 429), `APIConnectionError`, `ServiceUnavailableError`, `InternalServerError`, `Timeout`.
**What NOT to retry:** `ContextWindowExceededError` → translate immediately to `LLMContextWindowError` (an oversized prompt never shrinks by waiting); `AuthenticationError`/`BadRequestError` → translate to `LLMError`.
⚠️ **Gotcha:** `litellm.ContextWindowExceededError` is a *subclass* of `litellm.BadRequestError` — its `except` clause must come first.

### Algorithm (exponential backoff + jitter + Retry-After)

```python
async def _acompletion_with_retry(self, litellm_mod, kwargs):
    for attempt in range(settings.llm_max_retries + 1):     # 1 try + 3 retries
        try:
            return await litellm_mod.acompletion(**kwargs)
        except litellm_mod.ContextWindowExceededError as e:
            raise LLMContextWindowError(str(e)) from e       # never retried
        except litellm_mod.RateLimitError as e:
            err, retry_after = e, self._extract_retry_after(e)
        except (litellm_mod.APIConnectionError, litellm_mod.ServiceUnavailableError,
                litellm_mod.InternalServerError, litellm_mod.Timeout) as e:
            err, retry_after = e, None

        if attempt == settings.llm_max_retries:
            raise LLMRateLimitError(f"LLM call failed after {attempt} retries: {err}") from err

        delay = min(settings.llm_retry_base_delay * 2 ** attempt,   # 2, 4, 8
                    settings.llm_retry_max_delay)                    # cap 60s
        delay *= 0.5 + random.random() * 0.5                         # jitter: 50–100%
        if retry_after is not None:
            delay = max(delay, retry_after)                          # server knows best
        logger.warning("llm_retry", attempt=attempt + 1, delay=round(delay, 1),
                       error=type(err).__name__, retry_after=retry_after)
        await asyncio.sleep(delay)

def _extract_retry_after(self, e) -> float | None:
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    val = headers.get("retry-after") or headers.get("Retry-After")
    try: return float(val) if val else None
    except ValueError: return None
```

**Why exponential + jitter:** doubling the wait (2→4→8s) gives Azure's per-minute token bucket time to refill; randomizing the delay prevents multiple concurrent tasks from retrying at the same instant and colliding again (thundering herd). Honoring `Retry-After` uses Azure's own hint for exactly when capacity returns.

**After final failure:** `LLMRateLimitError` propagates to `Coordinator.run()`. Add an `except LLMRateLimitError` branch there (before the generic handler) that returns a **task-state message** built from the conversation: the goal + a mechanical digest of tools called so far + "Azure rate limit persisted after 3 retries — resend to continue." (Reuse `mechanical_digest()` from Step 6.) This satisfies "summary of the task at hand when the limit is hit."

*(Optional follow-up, not in scope now: mirror the same translation in openai.py/anthropic.py — the user runs litellm.)*

## Step 4 — Screenshot downscale + JPEG ([screenshot.py](computer_agent/tools/screen/screenshot.py))

In `take_screenshot` and `take_region_screenshot`, before encoding:

```python
max_dim = settings.screenshot_max_dimension            # 1440
if max(screenshot.width, screenshot.height) > max_dim:
    scale = max_dim / max(screenshot.width, screenshot.height)
    screenshot = screenshot.resize((int(screenshot.width*scale), int(screenshot.height*scale)))
screenshot = screenshot.convert("RGB")                 # JPEG has no alpha channel
screenshot.save(buf, format="JPEG", quality=settings.screenshot_jpeg_quality)
return ToolResult.ok(output=b64, format="base64_jpeg",
                     width=w, height=h, original_width=W, original_height=H)
```

Effect: ~800KB PNG → ~60–120KB JPEG. ⚠️ **Coordinate scaling is critical:** mouse tools use physical screen pixels (and Retina Macs screenshot at 2×). The placeholder text (Step 5a) must include: `"Image scaled to {w}x{h}; actual screen is {W}x{H} — multiply coordinates by {W/w:.2f}."`

## Step 5 — Serialization fixes (the big token wins)

### 5a. Send screenshots as image content blocks — the critical fix

Detection at serialization time is trivial: `result.metadata.get("format") in ("base64_png", "base64_jpeg")` (the tool already sets it).

**OpenAI/LiteLLM format** (`LiteLLMProvider.format_tool_result_messages`, [litellm.py:168](computer_agent/llm/providers/litellm.py#L168), mirror in openai.py): the API does **not** accept images inside `role:"tool"` messages. Standard pattern:
1. The `tool` message content becomes a short placeholder: `"Screenshot captured (1440x900 JPEG, screen is 2880x1800 — multiply coords by 2.00). Image attached in the next message."`
2. After ALL tool messages of that turn, append one `user` message:
   ```python
   {"role": "user", "content": [
       {"type": "text", "text": "[Screenshot from take_screenshot]"},
       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}
   ```
   Pairing stays intact: tool messages remain contiguous after the assistant message; the image user-message follows them.

**Anthropic format** (in `Coordinator._append_tool_results`, [coordinator.py:320](computer_agent/coordinator.py#L320) branch): `tool_result` natively supports image blocks:
```python
{"type": "tool_result", "tool_use_id": tc.id, "content": [
    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    {"type": "text", "text": "Screenshot captured (WxH)"}]}
```

Result: ~300k text tokens → ~1–1.7k image tokens per screenshot.

### 5b. Sliding window of images

Old screenshots are stale (the screen changed) — keep only the newest `max_images_in_context` (2). Every turn, `ContextManager.prune_old_images()` walks messages end→start, counts image parts, and replaces older ones with the stub `"[Screenshot removed from history to save context — take a new one if needed.]"`. Cheap (no LLM call), runs every turn, independent of full compaction.

### 5c. Tool-result truncation (head + tail)

Helper in `context_manager.py`, applied to `content_str` at all three serialization sites (coordinator.py:325 & :345, litellm.py:177; skip image outputs):

```python
def truncate_middle(s: str, max_chars: int) -> str:      # default 8000
    if len(s) <= max_chars: return s
    head = int(max_chars * 0.7); tail = max_chars - head
    return s[:head] + f"\n... [truncated {len(s)-max_chars} chars] ...\n" + s[-tail:]
```

Head+tail beats plain head-truncation because errors/summaries often sit at the end of command output.

### 5d. Compact JSON

Every `json.dumps(output, indent=2, default=str)` → `json.dumps(output, separators=(",", ":"), default=str)` (coordinator.py:325, litellm.py:177, check openai.py/anthropic.py). Pretty-printed JSON wastes ~20–30% tokens on whitespace.

### 5e. Tool schemas & Azure prompt caching — deliberately leave alone

56 tools ≈ ~6k tokens/call. Azure applies **automatic prefix caching** when tools + system prompt are byte-identical across calls (>1024 tokens) — already the case here. Rules for the implementer: do NOT reorder/subset tools per call, do NOT inject timestamps/session ids into the system prompt — either would break the cache prefix. Tool-subset routing isn't worth it (5% of window, risks missing tools mid-task); flag as future work. `max_tokens=4096` stays.

## Step 6 — ContextManager (NEW `computer_agent/llm/context_manager.py`)

```python
class ContextManager:
    def __init__(self, model: str, provider_format: str): ...  # "openai"|"anthropic"|"plain"
    def context_window(self) -> int: ...
    def estimate_tokens(self, messages, system, tools) -> int: ...
    def needs_compaction(self, messages, system, tools) -> bool: ...
    def prune_old_images(self, messages, keep=None) -> int: ...
    def split_into_groups(self, messages) -> list[list[dict]]: ...
    async def compact(self, messages, llm, aggressive=False) -> list[dict]: ...
```

`provider_format` is derived once in the Coordinator with the same branching already used in `_append_tool_results` (class name `AnthropicProvider` → "anthropic"; has `format_tool_result_messages` → "openai"; else "plain").

**Context window resolution** (cached at init): `settings.context_window_tokens` if >0 → else `litellm.get_model_info(model)["max_input_tokens"]` in try/except → else 128,000 with a warning log naming the source used.

### Token estimation (layered, cheapest-accurate first)

1. **Primary:** `litellm.token_counter(model=..., messages=[system]+messages)` in try/except (raises on unknown Azure deployments / image parts).
2. **Fallback heuristic:** `sum(len(json.dumps(m)) for m in messages)//4 + len(system)//4 + 6000` (tools), counting each image part as a flat 1,600 tokens rather than its base64 length.
3. **Cross-check:** the coordinator stores `response.usage.input_tokens` from the previous call (`_last_input_tokens`) — exact data the API gave us for free; trust the larger of estimate vs. last-actual.

### Threshold

`needs_compaction` ⇔ estimate ≥ `context_window() * 0.75` (~96k of 128k) — leaves room for the 4k reply plus summary overhead. This check runs **before every LLM call** (i.e., before the next tool execution round), satisfying "summarize just before a tool call when the limit is too high."

### Message grouping — the pairing-safety core

A **group** is an atomic unit that must never be split:
- **OpenAI:** assistant-with-`tool_calls` + ALL contiguous following `role:"tool"` messages + (from 5a) an immediately-following user message carrying image blocks. Everything else is a singleton.
- **Anthropic:** assistant message containing `tool_use` blocks + the next user message containing `tool_result` blocks. Others singletons.
- **Plain:** all singletons.

```python
def split_into_groups(messages):
    groups, i = [], 0
    while i < len(messages):
        if is_assistant_with_tool_calls(messages[i]):
            j = i + 1
            while j < len(messages) and is_tool_result_message(messages[j]): j += 1
            if fmt == "openai" and j < len(messages) and is_image_followup(messages[j]): j += 1
            groups.append(messages[i:j]); i = j
        else:
            groups.append([messages[i]]); i += 1
    return groups
```

### Compaction algorithm

```python
async def compact(messages, llm, aggressive=False):
    groups = split_into_groups(messages)
    keep = 2 if aggressive else settings.context_keep_recent_groups   # 6
    if len(groups) <= keep + 1: return messages          # nothing to fold

    goal_text  = text_of(groups[0][0])                   # original user goal
    middle     = flatten(groups[1:-keep])                # whole groups only → pairing safe
    tail       = flatten(groups[-keep:])

    # 1. Render `middle` as plain text: role + text; tool msgs →
    #    "tool <name> → <first 300 chars>"; image parts → "[screenshot]";
    #    whole render capped at ~40k chars (head+tail).
    transcript = render_for_summary(middle)

    # 2. Summarize with the primary provider (no tools, max_tokens=1024,
    #    model=settings.compaction_model or None). On ANY failure fall back to
    #    mechanical_digest(middle): tool names called, last errors, files/URLs
    #    touched — compaction must never throw.
    summary = await summarize_or_digest(llm, transcript)

    # 3. Rebuild: merge goal + summary into ONE user message so message[0]
    #    stays role=user and no same-role adjacency is created (Anthropic
    #    alternation safety):
    first = {"role": "user", "content": goal_text +
             "\n\n--- Progress summary (earlier steps compacted) ---\n" + summary}
    if fmt == "anthropic" and tail and tail[0]["role"] == "user" and is_plain_text(tail[0]):
        first["content"] += "\n\n" + tail[0]["content"]; tail = tail[1:]
    logger.info("context_compacted", before=len(messages), after=1+len(tail))
    return [first] + tail
```

**Summarization prompt** (module constant): *"Summarize this agent-execution transcript in under 400 words. Preserve: (1) the user's goal, (2) what has been accomplished step by step, (3) key facts discovered (file paths, URLs, ids, on-screen values), (4) errors hit and what was tried, (5) what remains to be done. Do not invent details."*

**Post-conditions (assert in tests):**
1. `rebuilt[0]["role"] == "user"`.
2. Every OpenAI `tool` message's `tool_call_id` appears in the contiguously-preceding assistant's `tool_calls`.
3. Every Anthropic `tool_result.tool_use_id` matches a `tool_use` in the immediately preceding assistant message.
4. The tail is a whole-group suffix (no group split).

## Step 7 — Coordinator integration ([coordinator.py](computer_agent/coordinator.py))

1. `__init__`: `self._context = ContextManager(settings.primary_model, self._provider_format())`; `self._last_input_tokens = 0`.
2. Top of the `while` body in `_run_agent_loop` (before `_call_llm`):
   ```python
   self._context.prune_old_images(self._conversation)
   if self._context.needs_compaction(self._conversation, self._system_prompt(), tool_schemas) \
           or self._last_input_tokens >= self._context.context_window() * settings.context_compact_threshold:
       self._conversation = await self._context.compact(self._conversation, self._llm)
   ```
3. After each response (~line 203): `self._last_input_tokens = response.usage.input_tokens`.
4. **Reactive fallback** in `_call_llm`: catch `LLMContextWindowError` → `prune_old_images(keep=1)` + `compact(aggressive=True)` → retry the call **once**; if it fails again, re-raise.
5. `run()`: add `except LLMRateLimitError` with the task-state message (Step 3).
6. `_append_tool_results`: image blocks (Anthropic branch), `truncate_middle`, compact JSON.

Known acceptable side effect: HITL checkpoints snapshot `conversation_history` (coordinator.py:436); after compaction the snapshot may differ from live history — intentional.

## Step 8 — Web chat UI on the daemon

Verified foundation: `POST /chat` ([daemon/api.py:102](computer_agent/daemon/api.py#L102), `{message, session_id?}` → `{response, session_id}`, Coordinator-per-session in memory), SSE `GET /events` ([api.py:267](computer_agent/daemon/api.py#L267)) with 15s keepalives, `GET /hitl/pending` + `POST /hitl/{id}/resolve`, `GET /status`. No static files/CORS today; daemon on `127.0.0.1:8765`.

**Design — one self-contained page, no build step, no framework:**
1. **NEW** `computer_agent/daemon/web/index.html` — single file, inline CSS + vanilla JS.
2. **New route** in [daemon/api.py](computer_agent/daemon/api.py): `GET /` → `FileResponse(Path(__file__).parent / "web" / "index.html")`. Same origin ⇒ **no CORS needed**; no `StaticFiles` mount for one file.
3. Page behavior (existing endpoints only):
   - Chat pane: send → `fetch("/chat", ...)`; persist `session_id` in `localStorage` so a page refresh keeps the conversation. Spinner while the request is in flight (responses are synchronous and can take minutes).
   - Live activity feed: `new EventSource("/events")` → render `task.step.completed` ("turn 3: take_screenshot, click"), `task.completed` / `task.failed` lines — this is what makes the long synchronous wait transparent.
   - HITL: on `hitl.approval.requested`, show an Approve/Deny banner wired to `POST /hitl/{id}/resolve`.
   - Status header: from events (fallback: poll `GET /status` every 10s) — running/queued tasks, autonomy mode.
   - "New session" button: clears stored `session_id` (also the user's escape hatch for a bloated context).
4. In `main.py` daemon command, print `Web UI: http://127.0.0.1:8765/` on startup.

**Why this solves "it closes itself":** the daemon is long-lived and sessions persist across messages — start `computer-agent daemon` once, talk in the browser; nothing exits when a task finishes. (The terminal `computer-agent chat` also remains available.)

## Step 9 — Tests & verification

**NEW `tests/test_llm_retry.py`** (no real API; monkeypatch `litellm.acompletion` with async fakes and `asyncio.sleep` to record delays):
- raises `RateLimitError` twice then succeeds → 3 calls, delays grow ~2→4 (within jitter bounds).
- always raises → `LLMRateLimitError` after `1 + llm_max_retries` calls.
- exception with `response.headers={"retry-after": "10"}` → recorded delay ≥ 10.
- raises `ContextWindowExceededError` → exactly 1 call, `LLMContextWindowError` raised (not retried).

**NEW `tests/test_context_manager.py`** (pure unit tests):
- group splitting for OpenAI and Anthropic synthetic conversations.
- compaction pairing invariants: stub summarizer LLM returns "SUMMARY"; compact a 30-message conversation; assert the four post-conditions (write an `assert_valid_openai_pairing(messages)` helper).
- compaction no-op when short; goal + tail preserved; stub-LLM-raises → mechanical digest used (never throws).
- `prune_old_images` keeps last N (both formats); `truncate_middle`; token-estimation fallback (monkeypatch `litellm.token_counter` to raise → heuristic used, image parts counted flat).

**Coordinator reactive path:** fake provider whose `generate` raises `LLMContextWindowError` once then succeeds → `_call_llm` compacts and returns (inject via monkeypatching `_resolve_provider`).

**Manual verification:** start `computer-agent daemon`, open `http://127.0.0.1:8765/`, run a screenshot-heavy task; watch logs for `llm_retry` / `context_compacted`; verify a click after a downscaled screenshot lands correctly (Retina coordinate scaling). Per the user's preference, Claude runs only the targeted test files (`pytest tests/test_llm_retry.py tests/test_context_manager.py -q`); the user runs the full suite and real Azure end-to-end himself.

## Ordered implementation sequence

1. config.py settings → 2. llm/errors.py (+ re-exports) → 3. retry in litellm provider + its tests → 4. screenshot downscale/JPEG → 5. serialization fixes (image blocks, truncation, compact JSON) → 6. context_manager.py + its tests → 7. coordinator integration → 8. web UI → 9. targeted test pass.

## Risks / gotchas summary

- `ContextWindowExceededError` subclasses `BadRequestError` — except-clause order matters.
- Compaction must treat the image-follow-up user message as part of its tool group, or it strands a "see attached image" placeholder.
- `litellm.get_model_info` often fails on Azure deployment names → `CONTEXT_WINDOW_TOKENS` override + 128k default is the safety net.
- Downscaled screenshots break click coordinates unless the scale note is in the placeholder text (Retina 2× compounds this).
- The summarizer call goes through the same provider retry and falls back to a mechanical digest — compaction never throws.
- Nothing may inject per-turn dynamic content into system prompt / tool order, or Azure prompt-cache hits are lost.
