"""
Coordinator Agent — the root orchestrator of the Computer Agent system.

Responsibilities:
  1. Receive user requests
  2. Consult memory for similar past traces (fast-replay opportunity)
  3. Resolve the configured LLM provider via LLMRegistry
  4. Run the agentic loop: LLM → tool calls → execute → observe → repeat
  5. Gate each tool call through the Abilities Engine (ALLOW / HITL / BLOCK)
  6. Pause at HITL checkpoints and resume on human approval
  7. Log complete execution traces to memory
  8. Return the final response to the user

The coordinator is LLM-provider agnostic — it works with any provider
(Anthropic Claude, OpenAI GPT, Google Gemini, local Ollama/LiteLLM) by
using the unified BaseLLMProvider interface. Switch providers by changing
PRIMARY_MODEL in .env — no code changes needed.

Architecture mirrors Google ADK's root_agent pattern (coordinator +
sub-agents) but with OS-level integration and the HITL layer built in.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from computer_agent.abilities.engine import abilities_engine
from computer_agent.config import settings
from computer_agent.hitl.checkpoint import CheckpointStatus, hitl_manager
from computer_agent.llm import LLMRegistry, LLMResponse, ToolCall
from computer_agent.llm.base import BaseLLMProvider
from computer_agent.logging_setup import get_logger
from computer_agent.memory.embeddings import embed_text
from computer_agent.memory.store import memory_store
from computer_agent.runtime.event_bus import Event, EventType, event_bus
from computer_agent.taskmgr.control import TaskCancelledError, TaskControl
from computer_agent.tools.base import ToolDefinition, ToolResult
from computer_agent.tools.registry import registry

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a Computer Agent — an AI executive assistant that controls a computer on behalf of the user.

You have access to a curated set of tools for:
- Taking screenshots and reading the screen (OCR / visual)
- Controlling the mouse and keyboard
- Automating web browsers (Playwright)
- Reading and writing files
- Running shell commands
- Making HTTP API calls

## Core Principles
1. PREFER PRECISION OVER SPEED: Always verify your actions completed successfully before proceeding to the next step.
2. PREFER APIs OVER CLICKING: If a task can be accomplished via an API call, prefer that over GUI automation.
3. OBSERVE BEFORE ACTING: Take a screenshot or read the DOM before clicking to confirm the current state.
4. ONE STEP AT A TIME: Complete one action, verify it worked, then proceed.
5. REPORT CLEARLY: After completing a task, summarize what was done and the outcome.

## When tools fail
- Try an alternative approach (e.g., if a CSS selector fails, try text-based selection)
- After 3 failures on the same step, stop and explain the situation to the user
- Never hallucinate — if you cannot verify something happened, say so

## Safety
- Some tools require human approval before execution. The system will automatically pause and notify the user.
- Never attempt to access credentials from memory or screenshots — credentials are handled by the secure vault.
"""


class Coordinator:
    """
    Root orchestrator that runs the main agent loop.

    The LLM provider is resolved from settings.primary_model via LLMRegistry
    and can be any supported provider (Anthropic, OpenAI, Google, LiteLLM).
    Switch models by setting PRIMARY_MODEL in .env.
    """

    def __init__(
        self,
        session_id: str | None = None,
        task_id: str | None = None,
        task_control: TaskControl | None = None,
        extra_system_prompt: str = "",
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.task_id = task_id
        self._task_control = task_control
        self._extra_system_prompt = extra_system_prompt
        self._llm: BaseLLMProvider = self._resolve_provider()
        self._conversation: list[dict[str, Any]] = []
        self._working_memory: dict[str, Any] = {}
        self._start_time: float = 0.0
        self._token_count: int = 0

        logger.info(
            "coordinator_initialized",
            session_id=self.session_id,
            model=settings.primary_model,
            provider=self._llm.__class__.__name__,
        )

    def _resolve_provider(self) -> BaseLLMProvider:
        """Resolve the LLM provider from config settings."""
        api_key = self._pick_api_key(settings.primary_model)
        api_base = settings.llm_api_base or None
        return LLMRegistry.resolve(
            settings.primary_model,
            api_key=api_key or None,
            api_base=api_base,
        )

    def _pick_api_key(self, model: str) -> str:
        """Return the appropriate API key based on the model name prefix."""
        m = model.lower()
        if m.startswith("claude"):
            return settings.anthropic_api_key
        if m.startswith(("gpt-", "o1-", "o3-", "o4-", "openai/")):
            return settings.openai_api_key
        if m.startswith(("gemini", "google/")):
            return settings.google_api_key
        return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_request: str) -> str:
        """Process a user request end-to-end. Returns the agent's final text response."""
        self._start_time = time.time()

        await event_bus.emit(Event(
            type=EventType.TASK_STARTED,
            session_id=self.session_id,
            data={"goal": user_request, "task_id": self.task_id},
        ))

        similar_traces = await self._retrieve_similar_traces(user_request)
        context_hint = self._build_context_hint(similar_traces)

        user_message = user_request
        if context_hint:
            user_message = f"{user_request}\n\n{context_hint}"

        self._conversation.append({"role": "user", "content": user_message})

        try:
            final_response = await self._run_agent_loop(user_request)
        except TaskCancelledError:
            # Propagate so the TaskManager finalizes the task as cancelled
            logger.info("coordinator_cancelled", session=self.session_id, task_id=self.task_id)
            raise
        except Exception as e:
            logger.error("coordinator_error", error=str(e), session=self.session_id)
            final_response = f"I encountered an error: {e}"
            await event_bus.emit(Event(
                type=EventType.TASK_FAILED,
                session_id=self.session_id,
                data={"error": str(e), "task_id": self.task_id},
            ))

        duration_ms = int((time.time() - self._start_time) * 1000)
        await self._save_trace(user_request, final_response, duration_ms)

        await event_bus.emit(Event(
            type=EventType.TASK_COMPLETED,
            session_id=self.session_id,
            data={
                "goal": user_request,
                "response": final_response[:500],
                "duration_ms": duration_ms,
                "task_id": self.task_id,
            },
        ))

        return final_response

    # ------------------------------------------------------------------
    # Agent Loop
    # ------------------------------------------------------------------

    async def _run_agent_loop(self, goal: str) -> str:
        """
        Core agentic loop (provider-agnostic):
          LLM → tool_use → execute → observe → LLM → ... → text response
        """
        tool_defs = registry.get_all_tools()
        tool_schemas = self._format_tools(tool_defs)
        turn = 0

        while turn < settings.max_conversation_turns:
            turn += 1

            if self._task_control:
                await self._task_control.checkpoint()

            response: LLMResponse = await self._call_llm(tool_schemas)
            self._token_count += response.usage.total_tokens

            if response.is_done and not response.has_tool_calls:
                self._conversation.append({"role": "assistant", "content": response.text})
                return response.text

            if response.has_tool_calls:
                self._append_assistant_turn(response)
                tool_results = await self._execute_tool_calls(response.tool_calls, goal)
                self._append_tool_results(response.tool_calls, tool_results)
                await event_bus.emit(Event(
                    type=EventType.STEP_COMPLETED,
                    session_id=self.session_id,
                    data={
                        "task_id": self.task_id,
                        "turn": turn,
                        "tools": [tc.name for tc in response.tool_calls],
                        "note": response.text[:200] if response.text else "",
                    },
                ))
                continue

            if response.text:
                self._conversation.append({"role": "assistant", "content": response.text})
                return response.text

            logger.warning("unexpected_llm_response", stop_reason=response.stop_reason, turn=turn)
            break

        return "I've reached the maximum number of steps for this task. Please review the progress."

    def _system_prompt(self) -> str:
        """Compose the system prompt with the current autonomy mode."""
        from computer_agent.abilities.autonomy import autonomy_manager

        mode = autonomy_manager.level.value
        mode_notes = {
            "low": "You must be maximally cautious: every action with side effects "
                   "pauses for the user's approval. Prefer suggesting over doing.",
            "medium": "You may perform simple, reversible actions on your own, but "
                      "always report what you did. Major or risky actions pause for approval.",
            "high": "You may handle most actions autonomously. Only critical or "
                    "flagged actions pause for approval. Keep the user informed of "
                    "everything you do in your final report.",
        }
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n## Autonomy Mode\n"
            f"Current autonomy level: {mode}. {mode_notes.get(mode, '')}"
        )
        if self._extra_system_prompt:
            prompt += f"\n\n{self._extra_system_prompt}"
        return prompt

    async def _call_llm(self, tool_schemas: list[dict[str, Any]]) -> LLMResponse:
        """Call the configured LLM provider with current conversation state."""
        return await self._llm.generate(
            self._conversation,
            system=self._system_prompt(),
            tools=tool_schemas if tool_schemas else None,
            max_tokens=4096,
        )

    def _format_tools(self, tool_defs: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tool definitions for the current provider."""
        provider_cls = type(self._llm)
        if hasattr(provider_cls, "format_tool_schemas"):
            return provider_cls.format_tool_schemas(tool_defs)
        # Generic fallback — OpenAI function format works for most providers
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tool_defs
        ]

    def _append_assistant_turn(self, response: LLMResponse) -> None:
        """Append the assistant's tool-requesting turn to conversation history."""
        provider_cls = type(self._llm)

        if provider_cls.__name__ == "AnthropicProvider":
            blocks: list[dict[str, Any]] = []
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            self._conversation.append({"role": "assistant", "content": blocks})

        elif hasattr(provider_cls, "assistant_message_from_tool_calls"):
            msg = provider_cls.assistant_message_from_tool_calls(response.tool_calls)
            self._conversation.append(msg)

        else:
            tool_summary = ", ".join(tc.name for tc in response.tool_calls)
            self._conversation.append({
                "role": "assistant",
                "content": response.text or f"[Calling tools: {tool_summary}]",
            })

    def _append_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: list[ToolResult],
    ) -> None:
        """Append tool results to conversation history in the provider's expected format."""
        import json as _json
        provider_cls = type(self._llm)

        if provider_cls.__name__ == "AnthropicProvider":
            content = []
            for tc, result in zip(tool_calls, results, strict=True):
                output = result.output
                if isinstance(output, (dict, list)):
                    content_str = _json.dumps(output, indent=2, default=str)
                else:
                    content_str = str(output) if output is not None else "Success"
                if not result.success:
                    content_str = f"Error: {result.error}"
                content.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": content_str,
                })
            self._conversation.append({"role": "user", "content": content})

        elif hasattr(provider_cls, "format_tool_result_messages"):
            messages = provider_cls.format_tool_result_messages(tool_calls, results)
            self._conversation.extend(messages)

        else:
            lines = []
            for tc, result in zip(tool_calls, results, strict=True):
                output = result.output
                output_str = _json.dumps(output, default=str) if isinstance(output, (dict, list)) else str(output or "Success")
                if not result.success:
                    output_str = f"Error: {result.error}"
                lines.append(f"Tool '{tc.name}' result: {output_str}")
            self._conversation.append({"role": "user", "content": "\n".join(lines)})

    # ------------------------------------------------------------------
    # Tool Execution with Abilities Gate
    # ------------------------------------------------------------------

    async def _execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        goal: str,
    ) -> list[ToolResult]:
        """Execute all tool calls from an LLM response, gated by the abilities engine."""
        results: list[ToolResult] = []

        for tc in tool_calls:
            if self._task_control:
                await self._task_control.checkpoint()

            await event_bus.emit(Event(
                type=EventType.TOOL_INVOKED,
                session_id=self.session_id,
                data={"tool": tc.name, "params": tc.arguments, "task_id": self.task_id},
            ))

            try:
                tool_risk = registry.get_tool(tc.name).risk_level
            except KeyError:
                tool_risk = None
            decision = abilities_engine.evaluate(tc.name, tc.arguments, risk_level=tool_risk)

            if decision.is_blocked():
                result = ToolResult.fail(
                    error=f"Action blocked by safety policy: {decision.format_message()}"
                )
            elif decision.requires_hitl():
                result = await self._handle_hitl_checkpoint(tc, decision, goal)
            else:
                result = await self._invoke_tool(tc.name, tc.arguments)

            event_type = EventType.TOOL_COMPLETED if result.success else EventType.TOOL_FAILED
            await event_bus.emit(Event(
                type=event_type,
                session_id=self.session_id,
                data={"tool": tc.name, "success": result.success, "error": result.error},
            ))
            results.append(result)

        return results

    async def _invoke_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Invoke a tool with retry logic and per-step timeout."""
        for attempt in range(1, settings.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    registry.invoke(tool_name, **params),
                    timeout=float(settings.step_timeout_seconds),
                )
                if result.success or attempt == settings.max_retries:
                    return result
                logger.warning("tool_retry", tool=tool_name, attempt=attempt, error=result.error)
                await asyncio.sleep(1.0 * attempt)
            except TimeoutError:
                if attempt == settings.max_retries:
                    return ToolResult.fail(
                        error=f"Tool '{tool_name}' timed out after {settings.step_timeout_seconds}s"
                    )
                logger.warning("tool_timeout_retry", tool=tool_name, attempt=attempt)
                await asyncio.sleep(1.0)

        return ToolResult.fail(error=f"Tool '{tool_name}' failed after {settings.max_retries} attempts")

    async def _handle_hitl_checkpoint(
        self,
        tc: ToolCall,
        decision: Any,
        goal: str,
    ) -> ToolResult:
        """Pause execution, request human approval, and resume if approved."""
        logger.info("hitl_checkpoint_triggered", tool=tc.name, session=self.session_id)

        checkpoint_state = await hitl_manager.request_approval(
            session_id=self.session_id,
            goal=goal,
            proposed_tool=tc.name,
            proposed_parameters=tc.arguments,
            risk_reason=decision.rule_name or "policy_rule",
            approval_message=decision.format_message(),
            conversation_history=self._conversation.copy(),
            working_memory=self._working_memory.copy(),
            remaining_steps=[],
            completed_steps=[],
            task_id=self.task_id,
        )

        if checkpoint_state.status == CheckpointStatus.APPROVED:
            logger.info("hitl_approved", tool=tc.name, session=self.session_id)
            await event_bus.emit(Event(
                type=EventType.HITL_APPROVAL_GRANTED,
                session_id=self.session_id,
                data={
                    "tool": tc.name,
                    "checkpoint": checkpoint_state.checkpoint_id,
                    "task_id": self.task_id,
                },
            ))
            return await self._invoke_tool(tc.name, tc.arguments)
        else:
            reason = checkpoint_state.status.value
            note = checkpoint_state.user_note or "No reason provided"
            await event_bus.emit(Event(
                type=EventType.HITL_APPROVAL_DENIED,
                session_id=self.session_id,
                data={"tool": tc.name, "reason": reason, "task_id": self.task_id},
            ))
            return ToolResult.fail(
                error=f"Action denied by user ({reason}): {note}. Tool: {tc.name}"
            )

    # ------------------------------------------------------------------
    # Memory Integration
    # ------------------------------------------------------------------

    async def _retrieve_similar_traces(self, goal: str) -> list[dict[str, Any]]:
        try:
            embedding = await embed_text(goal)
            if not embedding:
                return []
            return await memory_store.search_similar_traces(
                embedding=embedding,
                top_k=settings.memory_top_k,
                success_only=True,
            )
        except Exception as e:
            logger.debug("trace_retrieval_failed", error=str(e))
            return []

    def _build_context_hint(self, traces: list[dict[str, Any]]) -> str:
        if not traces:
            return ""
        lines = ["[Memory: Similar past tasks that may help:]"]
        for t in traces[:3]:
            sim = t.get("similarity", 0.0)
            lines.append(
                f"- Task: {t['goal']} (similarity: {sim:.0%}, "
                f"duration: {t.get('duration_ms', '?')}ms)"
            )
        lines.append("[Use these as reference, not strict templates]")
        return "\n".join(lines)

    async def _save_trace(self, goal: str, result: str, duration_ms: int) -> None:
        try:
            embedding = await embed_text(goal)
            await memory_store.save_trace(
                session_id=self.session_id,
                goal=goal,
                steps=[],
                result=result[:1000],
                success="error" not in result.lower()[:100],
                duration_ms=duration_ms,
                token_count=self._token_count,
                embedding=embedding,
            )
        except Exception as e:
            logger.debug("trace_save_failed", error=str(e))
