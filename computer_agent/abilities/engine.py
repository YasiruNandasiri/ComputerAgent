"""
Abilities Engine — evaluates rule-based constraints before every tool invocation.

Abilities are system-wide behavioral parameters: security guards, persona rules,
and HITL triggers. They operate at the middleware layer between the LLM's
tool-call decision and the actual tool execution.

Usage:
    engine = AbilitiesEngine()
    decision = engine.evaluate("delete_file", {"path": "~/Documents/invoice.pdf"})
    if decision.action == AbilityAction.REQUIRE_HITL:
        # Pause and ask human for approval
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel

from computer_agent.logging_setup import get_logger

if TYPE_CHECKING:
    from computer_agent.tools.base import RiskLevel

logger = get_logger(__name__)

_RULES_DIR = Path(__file__).parent / "rules"


class AbilityAction(StrEnum):
    ALLOW = "allow"
    REQUIRE_HITL = "require_hitl"
    BLOCK = "block"


class AbilityRule(BaseModel):
    name: str
    trigger: str          # regex pattern
    action: AbilityAction
    message: str = ""
    always_hitl: bool = False  # never downgrade REQUIRE_HITL, even in HIGH autonomy


class AbilityDecision(BaseModel):
    action: AbilityAction
    rule_name: str | None = None
    message: str = ""
    tool_name: str = ""
    parameters: dict[str, Any] = {}

    def is_allowed(self) -> bool:
        return self.action == AbilityAction.ALLOW

    def requires_hitl(self) -> bool:
        return self.action == AbilityAction.REQUIRE_HITL

    def is_blocked(self) -> bool:
        return self.action == AbilityAction.BLOCK

    def format_message(self) -> str:
        """Interpolate parameter values into the rule message template."""
        try:
            return self.message.format(**self.parameters, tool=self.tool_name)
        except KeyError:
            return self.message


class AbilitiesEngine:
    """
    Evaluates tool invocations against a rule set loaded from YAML files.
    Rules are loaded from computer_agent/abilities/rules/*.yaml.
    """

    def __init__(self, rules_dir: Path = _RULES_DIR) -> None:
        self._rules: list[AbilityRule] = []
        self._rules_dir = rules_dir
        self._load_rules()

    def _load_rules(self) -> None:
        """
        Load ability rules from:
          1. Built-in rules/*.yaml files
          2. External packages via entry-points (group: "computer_agent.abilities")

        External packages declare extra rule files in their pyproject.toml:
            [project.entry-points."computer_agent.abilities"]
            my_rules = "my_package:RULES_YAML"
        where RULES_YAML is a path to a YAML file with the standard rules format.
        """
        # 1. Built-in rules directory
        for yaml_file in sorted(self._rules_dir.glob("*.yaml")):
            self._load_rules_from_yaml(yaml_file)

        # 2. Entry-point registered rule files
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="computer_agent.abilities")
            for ep in eps:
                try:
                    rules_path_attr = ep.load()
                    rules_path = Path(str(rules_path_attr))
                    if rules_path.is_file():
                        self._load_rules_from_yaml(rules_path)
                        logger.info("ep_abilities_loaded", name=ep.name, file=str(rules_path))
                    elif rules_path.is_dir():
                        for yaml_file in sorted(rules_path.glob("*.yaml")):
                            self._load_rules_from_yaml(yaml_file)
                        logger.info("ep_abilities_dir_loaded", name=ep.name)
                except Exception as e:
                    logger.warning("entry_point_abilities_failed", name=ep.name, error=str(e))
        except Exception as e:
            logger.debug("entry_point_scan_failed", group="computer_agent.abilities", error=str(e))

        logger.info("abilities_loaded", count=len(self._rules))

    def _load_rules_from_yaml(self, yaml_file: Path) -> None:
        """Parse a single YAML rules file and append rules to self._rules."""
        try:
            data = yaml.safe_load(yaml_file.read_text())
            for rule_data in data.get("rules", []):
                rule = AbilityRule(
                    name=rule_data["name"],
                    trigger=rule_data["trigger"],
                    action=AbilityAction(rule_data["action"]),
                    message=rule_data.get("message", ""),
                    always_hitl=bool(rule_data.get("always_hitl", False)),
                )
                self._rules.append(rule)
        except Exception as e:
            logger.warning("ability_rule_load_failed", file=str(yaml_file), error=str(e))

    def evaluate(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        risk_level: RiskLevel | None = None,
    ) -> AbilityDecision:
        """
        Evaluate a tool invocation against all ability rules.
        The first matching rule provides the base action (default ALLOW); when
        risk_level is given, the current autonomy level adjusts the final
        decision (see abilities/autonomy.py). Without risk_level the rule
        action is returned unchanged.
        """
        from computer_agent.abilities.autonomy import autonomy_manager, resolve_action

        matched: AbilityRule | None = None
        for rule in self._rules:
            if re.search(rule.trigger, tool_name, re.IGNORECASE):
                matched = rule
                break

        base_action = matched.action if matched else AbilityAction.ALLOW
        final_action = resolve_action(
            base_action,
            risk_level=risk_level,
            level=autonomy_manager.level,
            always_hitl=matched.always_hitl if matched else False,
        )

        if matched:
            logger.debug(
                "ability_matched",
                tool=tool_name,
                rule=matched.name,
                action=base_action,
                final_action=final_action,
                autonomy=autonomy_manager.level.value,
            )

        message = matched.message if matched else ""
        if final_action == AbilityAction.REQUIRE_HITL and not message:
            message = (
                f"Autonomy mode '{autonomy_manager.level.value}' requires approval "
                f"before running '{tool_name}'."
            )

        return AbilityDecision(
            action=final_action,
            rule_name=matched.name if matched else None,
            message=message,
            tool_name=tool_name,
            parameters=parameters,
        )

    def add_rule(self, rule: AbilityRule) -> None:
        """Dynamically add a rule (e.g., from a skill manifest)."""
        self._rules.insert(0, rule)  # Higher priority than defaults

    def reload(self) -> None:
        """Reload all rules from disk."""
        self._rules.clear()
        self._load_rules()


# Module-level singleton
abilities_engine = AbilitiesEngine()
