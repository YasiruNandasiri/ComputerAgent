# Code review of commit 402f04e + remediation plan

## Context

Commit 402f04e implemented the resilience plan (retry, ContextManager, image blocks, web UI). The architecture matches the plan and all 17 new tests pass — **but I found 2 confirmed crashes, 2 correctness bugs on the exact paths the feature was built for, and several production gaps**. Both crashes were verified by executing the code. The remediation below fixes them in severity order.

---

## P0 — Confirmed bugs (fix first)

### 1. `_mechanical_digest` crashes on any path/URL — KeyError: slice
[context_manager.py:218](computer_agent/llm/context_manager.py#L218): `dict.fromkeys(artifacts)[:10]` — `dict.fromkeys()` returns a dict; slicing a dict raises `KeyError(slice(...))`. **Verified by execution.** Any conversation containing a file path or URL (i.e., nearly all) triggers it. Worse, it's called from the new `except LLMRateLimitError` handler in [coordinator.py:~171](computer_agent/coordinator.py#L171), so the crash happens *inside the except block* → the graceful "progress so far" message never appears and the task dies with an unhandled KeyError — the exact scenario the feature exists for.

**Fix:** `list(dict.fromkeys(artifacts))[:10]` (dict.fromkeys is the order-preserving dedup idiom; it just needs `list()` before slicing). Also wrap the digest call in the coordinator's except-branch in its own try/except — a fallback formatter must never be able to kill the handler it serves.

### 2. Compaction fallback summarizes an empty list — silent total data loss
[context_manager.py:476,479](computer_agent/llm/context_manager.py#L476): both fallback paths call `_mechanical_digest([])` instead of digesting the folded `middle` messages. So whenever the summarizer LLM fails (e.g. it's rate-limited too — likely, same deployment), the entire folded history is replaced by the constant string "No tool calls or errors recorded." The agent forgets everything mid-task with no error.

**Fix:** thread the messages through: `_summarize(llm, transcript, fallback_messages=middle)` and on failure return `_mechanical_digest(fallback_messages)`. Same for the `resp.text or ...` branch. Also pass the plan's system prompt (`"You compress agent execution logs. Be factual and terse."`) instead of `system=""`.

### 3. Retina coordinate instruction makes the model click in the wrong place
[litellm.py:265-270](computer_agent/llm/providers/litellm.py#L265): the placeholder says "screen is {original_px} — multiply coordinates by {original_px/img}". On macOS Retina (this user's platform), `pyautogui.screenshot()` returns **pixels** (2880×1800) but `pyautogui` mouse coordinates are in **points** (1440×900 = `pyautogui.size()`). After downscaling to 1440, the image is already 1:1 with mouse space — telling the model to multiply by 2.0 doubles every click coordinate.

**Fix (method):** make image space ≡ mouse space so no arithmetic is needed:
- In `take_screenshot`, also record `point_w, point_h = pyautogui.size()` in metadata.
- Resize target = `min(settings.screenshot_max_dimension, point_w)` derived scale — on a 2× Retina this lands exactly on point dimensions.
- Placeholder text becomes: "Mouse coordinates map 1:1 to this image" when img == point size, else "multiply image coordinates by {point_w/img_w:.2f}" — **computed against `pyautogui.size()`, never against raw screenshot pixels.**
- Regression test: metadata from a fake 2880×1800 capture with size()=(1440,900) must yield scale 1.0.

### 4. Server `Retry-After` is uncapped — one bad header sleeps the task for an hour
[litellm.py:146-147](computer_agent/llm/providers/litellm.py#L146): `delay = max(delay, retry_after_hint)` with no ceiling. Azure can send large Retry-After values on quota exhaustion.

**Fix:** `delay = min(max(delay, retry_after_hint), settings.llm_retry_max_delay)`. Also read Azure's `retry-after-ms` header (milliseconds) before `retry-after`.

---

## P1 — Correctness / security gaps

### 5. Image pruning is a silent no-op for Anthropic format
Anthropic screenshots are nested **inside** `tool_result` blocks' inner content list ([coordinator.py:~390](computer_agent/coordinator.py#L390)), but `_has_image_part` / `_replace_image_parts_with_stub` ([context_manager.py:99-127](computer_agent/llm/context_manager.py#L99)) only scan top-level parts. With a Claude model the original unbounded-image bug is fully back.
**Fix (method):** one recursive part-walker used by both functions: `iter_image_parts(content)` that descends into `part["content"]` when `part["type"] == "tool_result"`. Add an Anthropic-format pruning test (the current tests only cover OpenAI format).

### 6. Repeated compactions accumulate summaries forever
`compact()` takes `_text_of(groups[0][0])` as the goal; after the first compaction that message is `goal + "--- Progress summary ---" + s1`. The next compaction appends `s2` → the first message grows monotonically and old stale summaries are never dropped.
**Fix (carry-forward algorithm):** treat the marker as structure, not text:
```
goal, prev_summary = first_msg.split(MARKER, 1)      # prev_summary may be ""
transcript = prev_summary + render_for_summary(middle)  # old knowledge feeds the new summary
new_first  = goal + MARKER + new_summary                # REPLACE, never append
```
Information carries forward through the summarizer; the first message stays bounded (goal + ≤400 words).

### 7. DOM XSS in the web UI activity feed → HITL approval bypass
[index.html:297](computer_agent/daemon/web/index.html#L297): `addActivity` uses `innerHTML` with event-derived text — which includes LLM output (`STEP_COMPLETED.note = response.text[:200]`) and task goals. A malicious page the agent browses can steer its output to contain `<img onerror=...>`, which then executes in the UI origin — and that origin can `fetch("/hitl/{id}/resolve", {approved:true})`, i.e. self-approve dangerous actions. For a computer-use agent this is a real escalation path, not a cosmetic bug.
**Fix:** build the node with `document.createElement` + `textContent` (as `addMessage` already correctly does) and append the timestamp as a separate `<span>`. Grep the file for every other `innerHTML` with dynamic input (the spinner literal is fine).

### 8. Token estimate misses tool schemas on the primary path
`_litellm_token_count` counts system + messages only; the flat 6k tool budget exists only in the heuristic fallback → ~6k systematic underestimate when litellm counting works. (Verified separately that `litellm.token_counter` handles `image_url` parts correctly — 95 tokens, not 40k — so no change needed there.)
**Fix:** compute tool-schema tokens once in `__init__` (`litellm.token_counter(model, text=json.dumps(tools))`, fallback `len(json.dumps(tools))//4`), cache it, and add it in `estimate_tokens` regardless of path.

### 9. "Plain" provider branch still ships base64 garbage
The fallback branch of `_append_tool_results` ([coordinator.py:~430](computer_agent/coordinator.py#L430)) `truncate_middle`s the raw base64 to 8k chars — no longer catastrophic, but 8k chars of useless base64 per screenshot for providers without `format_tool_result_messages` (e.g. Google).
**Fix:** detect `fmt in ("base64_png","base64_jpeg")` there too and emit a text stub: `"[Screenshot captured (WxH) — this provider path does not support images]"`.

---

## P2 — Production hardening (smaller)

10. **Misleading exhaustion error:** connection/5xx failures also surface as `LLMRateLimitError("...")`; `LLMTransientError` is defined but never raised. Raise `LLMTransientError` when the last error wasn't a 429 (message text stays useful either way), and delete it if not adopted — dead code in an error hierarchy misleads maintainers.
11. **Resize quality:** `screenshot.resize(...)` without `resample=` — pass `Image.Resampling.LANCZOS`; screen text legibility directly affects the vision model's OCR ability. Consider JPEG quality 70–75 (60 visibly smears small text).
12. **Retry visibility:** emit `EventType.STEP_RETRYING` (already exists) from the coordinator around `_call_llm` retries, or at least have the provider accept an optional callback — today the web UI shows nothing during a 14-second backoff sequence.
13. **Web UI concurrency:** input is disabled during the whole in-flight `/chat` request; the daemon supports background tasks, so allow queuing a second message (or at minimum a visible "task running since Xs" timer). SSE events are also not filtered by session — every browser tab sees all sessions' events.
14. **Daemon exposure:** the UI increases the attack surface of the unauthenticated localhost API; cheap mitigations: bind check on `Host` header (DNS-rebinding guard) and a startup-generated token required by non-GET endpoints. Flag as follow-up, don't block.

## Meta-finding: the tests pass while the code crashes

17/17 green with two confirmed crash bugs is the review's most important lesson. The fixtures were written to satisfy the code path, not to challenge it: the digest test contains no paths/URLs (so line 218 never ran), and the fallback test asserts only "did not raise" + pairing (so an empty digest passed).
**Method for the fix-up tests:** assert on *content*, not just absence of exceptions —
- digest test fixture must include a file path and a URL; assert both appear in the output;
- fallback test: assert the digest names the tools that were folded (`"take_screenshot" in compacted[0]["content"]`);
- Retina test: fake metadata 2880×1800 pixels / 1440×900 points → placeholder must say scale 1.0 (or "map 1:1");
- Anthropic pruning test (currently missing entirely);
- repeated-compaction test: compact twice, assert exactly one summary marker in message[0].

## Implementation order

1. P0-1, P0-2 (context_manager.py — two-line + signature fix) + their content-asserting tests.
2. P0-3 (screenshot.py + litellm.py placeholder, point-space algorithm) + Retina test.
3. P0-4 (retry cap + retry-after-ms) + test.
4. P1-5 recursive image walker + Anthropic test; P1-6 marker carry-forward + double-compaction test.
5. P1-7 XSS fix (index.html, mechanical); P1-8 cached tool-token count; P1-9 plain-branch stub.
6. P2 items as a final pass, each optional.

## Verification

`uv run pytest tests/test_context_manager.py tests/test_llm_retry.py -q` (targeted only — the user runs the full suite himself). Manual: start daemon, open `http://127.0.0.1:8765/`, run a screenshot task on the Retina display and confirm a subsequent click lands correctly; check log for `context_compacted` and confirm message[0] has a single summary marker after two compactions.
