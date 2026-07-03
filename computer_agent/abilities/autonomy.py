"""
Autonomy Levels — a user-controlled dial over how independently the agent acts.

The autonomy level combines with each tool's declared RiskLevel and the
matched ability rule to produce the final gate decision:

    Tool risk  | LOW mode      | MEDIUM mode      | HIGH mode
    -----------|---------------|------------------|---------------------------
    low        | allow         | allow            | allow
    medium     | ask           | allow (+ report) | allow
    high       | ask           | ask              | allow unless always_hitl
    critical   | ask           | ask              | ask

A rule with action BLOCK always blocks. A rule flagged `always_hitl: true`
never has its REQUIRE_HITL downgraded, regardless of mode.

The level is persisted in the user_preferences table so it survives restarts,
and can be changed at runtime (CLI `computer-agent mode`, chat, or API).
"""

from __future__ import annotations

from enum import StrEnum

from computer_agent.logging_setup import get_logger
from computer_agent.tools.base import RiskLevel

logger = get_logger(__name__)

_PREF_KEY = "autonomy_level"


class AutonomyLevel(StrEnum):
    LOW = "low"        # Ask before anything with side effects
    MEDIUM = "medium"  # Handle simple, reversible actions; ask for major ones
    HIGH = "high"      # Handle most things; ask only for critical/flagged actions


class AutonomyManager:
    """Holds the current autonomy level and persists it as a user preference."""

    def __init__(self) -> None:
        self._level: AutonomyLevel | None = None

    @property
    def level(self) -> AutonomyLevel:
        if self._level is not None:
            return self._level
        return self._from_settings()

    def _from_settings(self) -> AutonomyLevel:
        from computer_agent.config import settings
        try:
            return AutonomyLevel(settings.autonomy_level.lower())
        except ValueError:
            logger.warning("invalid_autonomy_level", value=settings.autonomy_level)
            return AutonomyLevel.MEDIUM

    async def load(self) -> AutonomyLevel:
        """Load the persisted level from user preferences (call at bootstrap)."""
        try:
            from computer_agent.memory.store import memory_store
            saved = await memory_store.get_preference(_PREF_KEY)
            if saved:
                self._level = AutonomyLevel(str(saved).lower())
        except Exception as e:
            logger.debug("autonomy_load_failed", error=str(e))
        return self.level

    async def set_level(self, level: AutonomyLevel | str) -> AutonomyLevel:
        """Change the level at runtime and persist it."""
        value = level.value if isinstance(level, AutonomyLevel) else str(level).lower()
        self._level = AutonomyLevel(value)
        try:
            from computer_agent.memory.store import memory_store
            await memory_store.save_preference(
                _PREF_KEY, self._level.value, context="agent autonomy mode"
            )
        except Exception as e:
            logger.debug("autonomy_persist_failed", error=str(e))
        logger.info("autonomy_level_changed", level=self._level.value)
        return self._level


def resolve_action(
    rule_action: AbilityAction,  # type: ignore[name-defined]  # noqa: F821
    risk_level: RiskLevel | None,
    level: AutonomyLevel,
    always_hitl: bool = False,
) -> AbilityAction:  # type: ignore[name-defined]  # noqa: F821
    """
    Combine a matched rule's action with the tool's risk level and the current
    autonomy level into the final decision. With risk_level=None the rule
    action passes through unchanged (legacy behavior).
    """
    from computer_agent.abilities.engine import AbilityAction

    if rule_action == AbilityAction.BLOCK:
        return AbilityAction.BLOCK
    if always_hitl and rule_action == AbilityAction.REQUIRE_HITL:
        return AbilityAction.REQUIRE_HITL
    if risk_level is None:
        return rule_action
    if risk_level == RiskLevel.CRITICAL:
        return AbilityAction.REQUIRE_HITL

    if level == AutonomyLevel.LOW:
        # Never downgrade an explicit HITL rule in the most cautious mode
        if rule_action == AbilityAction.REQUIRE_HITL:
            return AbilityAction.REQUIRE_HITL
        if risk_level == RiskLevel.LOW:
            return AbilityAction.ALLOW
        return AbilityAction.REQUIRE_HITL

    if level == AutonomyLevel.MEDIUM:
        if risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
            return AbilityAction.ALLOW
        return AbilityAction.REQUIRE_HITL

    # HIGH: allow everything below critical unless the rule is always_hitl
    return AbilityAction.ALLOW


# Module-level singleton
autonomy_manager = AutonomyManager()
