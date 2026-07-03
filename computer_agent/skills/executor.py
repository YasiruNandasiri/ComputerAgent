"""
Skill Executor — runs a skill's steps in sequence, resolving parameters
and handling failures according to each step's `on_failure` directive.
"""

from __future__ import annotations

import re
from typing import Any

from computer_agent.logging_setup import get_logger
from computer_agent.tools.base import ToolResult
from computer_agent.tools.registry import registry

logger = get_logger(__name__)


class SkillExecutor:
    """
    Executes a SkillManifest step-by-step.

    Parameter interpolation: step params may reference prior step outputs
    using {step_N.output} or user-provided kwargs using {param_name}.
    """

    def __init__(self, manifest: Any) -> None:
        self._manifest = manifest
        self._step_results: list[ToolResult] = []

    async def run(self, kwargs: dict[str, Any]) -> ToolResult:
        """
        Execute all steps of the skill.
        Returns a ToolResult with the final step's output, or failure on abort.
        """
        logger.info("skill_starting", name=self._manifest.name)

        for i, step in enumerate(self._manifest.steps):
            # Resolve parameter templates
            resolved_params = self._resolve_params(step.params, kwargs, i)

            logger.debug(
                "skill_step_starting",
                skill=self._manifest.name,
                step=i + 1,
                tool=step.tool,
            )

            # Invoke the tool
            result = await registry.invoke(step.tool, **resolved_params)
            self._step_results.append(result)

            if not result.success:
                if step.on_failure == "abort":
                    logger.warning(
                        "skill_step_failed_abort",
                        skill=self._manifest.name,
                        step=i + 1,
                        error=result.error,
                    )
                    return ToolResult.fail(
                        error=f"Skill '{self._manifest.name}' failed at step {i + 1} "
                              f"({step.tool}): {result.error}"
                    )
                elif step.on_failure == "continue":
                    logger.warning(
                        "skill_step_failed_continue",
                        skill=self._manifest.name,
                        step=i + 1,
                        error=result.error,
                    )
                    continue
                # on_failure == "retry" is handled by the coordinator's retry logic

            logger.debug(
                "skill_step_completed",
                skill=self._manifest.name,
                step=i + 1,
                tool=step.tool,
            )

        # Return the last step's output as the skill result
        if self._step_results:
            last = self._step_results[-1]
            return ToolResult.ok(
                output=last.output,
                skill=self._manifest.name,
                steps_completed=len(self._step_results),
            )
        return ToolResult.ok(output="Skill completed with no steps", skill=self._manifest.name)

    def _resolve_params(
        self,
        params: dict[str, Any],
        kwargs: dict[str, Any],
        step_index: int,
    ) -> dict[str, Any]:
        """
        Resolve parameter templates in step params.
        Supports:
          {param_name}         → from user-provided kwargs
          {step_0.output}      → from a previous step's output
          {step_0.metadata.x}  → from step metadata
        """
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str):
                resolved[key] = self._interpolate(value, kwargs, step_index)
            else:
                resolved[key] = value
        return resolved

    def _interpolate(self, template: str, kwargs: dict[str, Any], step_index: int) -> str:
        """Interpolate {variable} references in a template string."""
        def replace(match: re.Match) -> str:
            ref = match.group(1)
            # Step output reference: step_N.output or step_N.metadata.key
            if ref.startswith("step_"):
                parts = ref.split(".", 1)
                step_n = int(parts[0].replace("step_", ""))
                if step_n < len(self._step_results):
                    result = self._step_results[step_n]
                    if len(parts) == 1 or parts[1] == "output":
                        return str(result.output or "")
                    if parts[1].startswith("metadata."):
                        meta_key = parts[1].replace("metadata.", "")
                        return str(result.metadata.get(meta_key, ""))
                return match.group(0)  # Unresolved, leave as-is

            # User kwarg reference
            if ref in kwargs:
                return str(kwargs[ref])

            return match.group(0)  # Unresolved

        return re.sub(r"\{([^}]+)\}", replace, template)
