"""
Tests for the Abilities Engine — rule evaluation and HITL triggering.
"""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import textwrap

from computer_agent.abilities.engine import AbilitiesEngine, AbilityAction, AbilityRule


@pytest.fixture
def engine_with_rules(tmp_path: Path) -> AbilitiesEngine:
    """Create an engine with a minimal test rule set."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "test.yaml").write_text(textwrap.dedent("""
        rules:
          - name: "delete_guard"
            trigger: "delete_file"
            action: "require_hitl"
            message: "Deleting {path}"
          - name: "financial_block"
            trigger: "payment|transfer"
            action: "block"
            message: "Financial ops blocked"
          - name: "read_allow"
            trigger: "read_file|take_screenshot"
            action: "allow"
            message: ""
    """))
    return AbilitiesEngine(rules_dir=rules_dir)


class TestAbilitiesEngine:
    def test_hitl_triggered_for_delete(self, engine_with_rules: AbilitiesEngine) -> None:
        decision = engine_with_rules.evaluate("delete_file", {"path": "~/test.txt"})
        assert decision.requires_hitl()
        assert decision.rule_name == "delete_guard"

    def test_block_for_payment(self, engine_with_rules: AbilitiesEngine) -> None:
        decision = engine_with_rules.evaluate("payment_service", {})
        assert decision.is_blocked()

    def test_allow_for_read(self, engine_with_rules: AbilitiesEngine) -> None:
        decision = engine_with_rules.evaluate("read_file", {"path": "~/docs/a.txt"})
        assert decision.is_allowed()

    def test_allow_for_screenshot(self, engine_with_rules: AbilitiesEngine) -> None:
        decision = engine_with_rules.evaluate("take_screenshot", {})
        assert decision.is_allowed()

    def test_default_allow_for_unknown(self, engine_with_rules: AbilitiesEngine) -> None:
        decision = engine_with_rules.evaluate("some_unknown_tool", {})
        assert decision.is_allowed()
        assert decision.rule_name is None

    def test_message_interpolation(self, engine_with_rules: AbilitiesEngine) -> None:
        decision = engine_with_rules.evaluate("delete_file", {"path": "/tmp/data.csv"})
        msg = decision.format_message()
        assert "/tmp/data.csv" in msg

    def test_dynamic_rule_insertion(self, engine_with_rules: AbilitiesEngine) -> None:
        new_rule = AbilityRule(
            name="custom_guard",
            trigger="my_custom_tool",
            action=AbilityAction.REQUIRE_HITL,
            message="Custom tool requires approval",
        )
        engine_with_rules.add_rule(new_rule)
        decision = engine_with_rules.evaluate("my_custom_tool", {})
        assert decision.requires_hitl()
        assert decision.rule_name == "custom_guard"


class TestAutonomyModes:
    """Autonomy level × tool risk level decision matrix."""

    @pytest.fixture(autouse=True)
    def _reset_autonomy(self):
        from computer_agent.abilities.autonomy import autonomy_manager
        yield
        autonomy_manager._level = None

    def _set_level(self, level: str) -> None:
        from computer_agent.abilities.autonomy import AutonomyLevel, autonomy_manager
        autonomy_manager._level = AutonomyLevel(level)

    @pytest.mark.parametrize(
        ("mode", "risk", "expected"),
        [
            # LOW mode: only risk-free actions run unattended
            ("low", "low", AbilityAction.ALLOW),
            ("low", "medium", AbilityAction.REQUIRE_HITL),
            ("low", "high", AbilityAction.REQUIRE_HITL),
            ("low", "critical", AbilityAction.REQUIRE_HITL),
            # MEDIUM mode: simple reversible actions run, major ones ask
            ("medium", "low", AbilityAction.ALLOW),
            ("medium", "medium", AbilityAction.ALLOW),
            ("medium", "high", AbilityAction.REQUIRE_HITL),
            ("medium", "critical", AbilityAction.REQUIRE_HITL),
            # HIGH mode: everything below critical runs
            ("high", "low", AbilityAction.ALLOW),
            ("high", "medium", AbilityAction.ALLOW),
            ("high", "high", AbilityAction.ALLOW),
            ("high", "critical", AbilityAction.REQUIRE_HITL),
        ],
    )
    def test_matrix_for_unmatched_tool(
        self, engine_with_rules: AbilitiesEngine, mode: str, risk: str, expected: AbilityAction
    ) -> None:
        from computer_agent.tools.base import RiskLevel

        self._set_level(mode)
        decision = engine_with_rules.evaluate(
            "some_unknown_tool", {}, risk_level=RiskLevel(risk)
        )
        assert decision.action == expected

    def test_block_never_downgraded(self, engine_with_rules: AbilitiesEngine) -> None:
        from computer_agent.tools.base import RiskLevel

        self._set_level("high")
        decision = engine_with_rules.evaluate("payment_service", {}, risk_level=RiskLevel.LOW)
        assert decision.is_blocked()

    def test_always_hitl_never_downgraded(self, engine_with_rules: AbilitiesEngine) -> None:
        from computer_agent.tools.base import RiskLevel

        engine_with_rules.add_rule(AbilityRule(
            name="email_guard",
            trigger="send_email",
            action=AbilityAction.REQUIRE_HITL,
            always_hitl=True,
        ))
        self._set_level("high")
        decision = engine_with_rules.evaluate("send_email", {}, risk_level=RiskLevel.HIGH)
        assert decision.requires_hitl()

    def test_plain_hitl_rule_downgraded_in_high_mode(
        self, engine_with_rules: AbilitiesEngine
    ) -> None:
        from computer_agent.tools.base import RiskLevel

        self._set_level("high")
        decision = engine_with_rules.evaluate(
            "delete_file", {"path": "~/x"}, risk_level=RiskLevel.HIGH
        )
        assert decision.is_allowed()

    def test_hitl_rule_kept_in_low_mode_even_for_low_risk(
        self, engine_with_rules: AbilitiesEngine
    ) -> None:
        from computer_agent.tools.base import RiskLevel

        self._set_level("low")
        decision = engine_with_rules.evaluate(
            "delete_file", {"path": "~/x"}, risk_level=RiskLevel.LOW
        )
        assert decision.requires_hitl()

    def test_no_risk_level_keeps_legacy_behavior(
        self, engine_with_rules: AbilitiesEngine
    ) -> None:
        self._set_level("high")
        decision = engine_with_rules.evaluate("delete_file", {"path": "~/x"})
        assert decision.requires_hitl()

    def test_autonomy_hitl_gets_fallback_message(
        self, engine_with_rules: AbilitiesEngine
    ) -> None:
        from computer_agent.tools.base import RiskLevel

        self._set_level("low")
        decision = engine_with_rules.evaluate("some_unknown_tool", {}, risk_level=RiskLevel.HIGH)
        assert decision.requires_hitl()
        assert "low" in decision.format_message()
