"""
Skill Loader — discovers and loads skill manifests from the plugins directory.

A Skill is a composed workflow: a YAML manifest that chains multiple tools
together to accomplish a specific business logic. Skills are loaded at startup
and presented to the LLM as "macro tools" in the tool schema.

Skill manifest format (skill_name/skill.yaml):
    name: "generate_monthly_expense_report"
    version: "1.0.0"
    description: "Generate a monthly expense report from bank exports"
    triggers:
      - "expense report"
      - "monthly spending"
    parameters:
      - name: month
        type: string
        required: true
    hitl_checkpoints:
      - before: "send_email"
        message: "About to email report to {recipient}"
    steps:
      - tool: read_file
        params:
          path: "~/Downloads/bank_{month}.csv"
      - tool: run_python_snippet
        params:
          code: "..."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from computer_agent.logging_setup import get_logger
from computer_agent.tools.base import RiskLevel, ToolDefinition, ToolResult

logger = get_logger(__name__)

_SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SkillParameter(BaseModel):
    name: str
    type: str = "string"
    required: bool = False
    default: Any = None
    description: str = ""


class SkillStep(BaseModel):
    tool: str
    params: dict[str, Any] = {}
    description: str = ""
    on_failure: str = "abort"  # abort | continue | retry


class SkillManifest(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str
    triggers: list[str] = []
    parameters: list[SkillParameter] = []
    steps: list[SkillStep] = []
    hitl_checkpoints: list[dict[str, Any]] = []
    api_equivalents: list[str] = []  # Descriptions for router mapping

    @classmethod
    def from_yaml(cls, path: Path) -> SkillManifest:
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)

    def to_tool_definition(self) -> ToolDefinition:
        """
        Convert this skill into a ToolDefinition so it appears in the LLM
        tool schema as a callable macro.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return ToolDefinition(
            name=self.name,
            description=self.description,
            func=self._make_executor(),
            risk_level=RiskLevel.MEDIUM,
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            category="skill",
        )

    def _make_executor(self):
        """Create a callable that executes this skill's steps."""
        manifest = self

        async def skill_executor(**kwargs: Any) -> ToolResult:
            from computer_agent.skills.executor import SkillExecutor
            executor = SkillExecutor(manifest)
            return await executor.run(kwargs)

        return skill_executor


# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Discovers and loads skill manifests from the plugins directory."""

    def __init__(self, skills_dir: Path = _SKILLS_DIR) -> None:
        self._skills_dir = skills_dir
        self._skills: dict[str, SkillManifest] = {}

    def discover(self) -> None:
        """
        Scan for skills from:
          1. The built-in skills/ directory (YAML manifests)
          2. External packages via entry-points (group: "computer_agent.skills")

        External packages declare extra skill dirs in their pyproject.toml:
            [project.entry-points."computer_agent.skills"]
            my_skills = "my_package:SKILLS_DIR"
        where SKILLS_DIR is a pathlib.Path or str pointing to a directory
        containing skill.yaml files.
        """
        loaded = 0

        # 1. Built-in skills directory
        if not self._skills_dir.exists():
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            logger.info("skills_dir_created", path=str(self._skills_dir))
        else:
            for yaml_file in self._skills_dir.rglob("skill.yaml"):
                try:
                    manifest = SkillManifest.from_yaml(yaml_file)
                    self._skills[manifest.name] = manifest
                    loaded += 1
                    logger.debug("skill_loaded", name=manifest.name, version=manifest.version)
                except Exception as e:
                    logger.warning("skill_load_failed", file=str(yaml_file), error=str(e))

        # 2. Entry-point registered skill directories
        loaded += self._load_entry_point_skills()

        logger.info("skills_discovered", count=loaded)

    def _load_entry_point_skills(self) -> int:
        """Load skills from external packages registered via Python entry-points."""
        loaded = 0
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="computer_agent.skills")
            for ep in eps:
                try:
                    skills_dir_attr = ep.load()
                    skills_dir = Path(str(skills_dir_attr))
                    for yaml_file in skills_dir.rglob("skill.yaml"):
                        try:
                            manifest = SkillManifest.from_yaml(yaml_file)
                            self._skills[manifest.name] = manifest
                            loaded += 1
                            logger.debug(
                                "ep_skill_loaded",
                                name=manifest.name,
                                source=ep.name,
                            )
                        except Exception as e:
                            logger.warning("ep_skill_load_failed", file=str(yaml_file), error=str(e))
                except Exception as e:
                    logger.warning("entry_point_skill_dir_failed", name=ep.name, error=str(e))
        except Exception as e:
            logger.debug("entry_point_scan_failed", group="computer_agent.skills", error=str(e))
        return loaded

    def register_with_tool_registry(self) -> None:
        """Register all loaded skills as tools in the global BrickRegistry."""
        from computer_agent.tools.registry import registry
        for skill in self._skills.values():
            tool_def = skill.to_tool_definition()
            registry.register_tool(tool_def)

    def get_skill(self, name: str) -> SkillManifest | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())

    def skills_summary(self) -> str:
        """Return a human-readable summary for the LLM system prompt."""
        if not self._skills:
            return "No skills loaded."
        lines = ["Available skills:"]
        for skill in self._skills.values():
            lines.append(f"  - {skill.name} v{skill.version}: {skill.description}")
        return "\n".join(lines)


# Module-level singleton
skill_registry = SkillRegistry()
