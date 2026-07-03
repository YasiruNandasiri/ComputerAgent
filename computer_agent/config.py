"""
Computer Agent — Centralized configuration via pydantic-settings.
All settings are loaded from environment variables and .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")

    # LLM_PROVIDER: "auto" (detect from model name) | "anthropic" | "openai" | "google" | "litellm"
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")

    primary_model: str = Field(default="claude-sonnet-4-20250514", alias="PRIMARY_MODEL")
    planner_model: str = Field(default="claude-sonnet-4-20250514", alias="PLANNER_MODEL")
    vision_model: str = Field(default="claude-sonnet-4-20250514", alias="VISION_MODEL")
    router_model: str = Field(default="claude-haiku-4-20250514", alias="ROUTER_MODEL")

    # Optional: override API base URL (for local endpoints, proxies, LM Studio, Ollama)
    llm_api_base: str = Field(default="", alias="LLM_API_BASE")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # --- Database ---
    database_url: str = Field(
        default="postgresql://agent:agentpass@localhost:5432/computer_agent",
        alias="DATABASE_URL",
    )

    # --- Memory ---
    memory_similarity_threshold: float = Field(default=0.85, alias="MEMORY_SIMILARITY_THRESHOLD")
    memory_top_k: int = Field(default=5, alias="MEMORY_TOP_K")

    # --- Security ---
    allowed_read_paths_raw: str = Field(
        default="~/Documents:~/Downloads:~/Desktop",
        alias="ALLOWED_READ_PATHS",
    )
    allowed_write_paths_raw: str = Field(
        default="~/Documents:~/Downloads:~/Desktop",
        alias="ALLOWED_WRITE_PATHS",
    )
    allowed_shell_commands_raw: str = Field(default="", alias="ALLOWED_SHELL_COMMANDS")
    allowed_api_domains_raw: str = Field(default="", alias="ALLOWED_API_DOMAINS")

    # --- Browser ---
    browser_type: str = Field(default="chromium", alias="BROWSER_TYPE")
    browser_headless: bool = Field(default=False, alias="BROWSER_HEADLESS")
    browser_user_data_dir: str = Field(
        default="~/.computer_agent/browser_profile",
        alias="BROWSER_USER_DATA_DIR",
    )

    # --- HITL ---
    hitl_approval_timeout: int = Field(default=300, alias="HITL_APPROVAL_TIMEOUT")
    # "pause": wait indefinitely for the user's decision (task shows awaiting_approval)
    # "expire": auto-expire (deny) after hitl_approval_timeout seconds
    hitl_timeout_action: str = Field(default="pause", alias="HITL_TIMEOUT_ACTION")

    # --- Daemon ---
    daemon_host: str = Field(default="127.0.0.1", alias="DAEMON_HOST")
    daemon_port: int = Field(default=8765, alias="DAEMON_PORT")

    # --- Task manager / scheduler ---
    max_concurrent_tasks: int = Field(default=1, alias="MAX_CONCURRENT_TASKS")
    scheduler_enabled: bool = Field(default=True, alias="SCHEDULER_ENABLED")

    # --- Autonomy ---
    # low: ask before anything with side effects
    # medium: handle simple reversible actions, ask for major ones
    # high: handle most things, ask only for critical/always_hitl actions
    autonomy_level: str = Field(default="medium", alias="AUTONOMY_LEVEL")

    # --- Observability ---
    otel_enabled: bool = Field(default=True, alias="OTEL_ENABLED")
    otel_exporter: str = Field(default="console", alias="OTEL_EXPORTER")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # --- Agent Behavior ---
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    step_timeout_seconds: int = Field(default=120, alias="STEP_TIMEOUT_SECONDS")
    max_conversation_turns: int = Field(default=50, alias="MAX_CONVERSATION_TURNS")

    # --- Derived properties ---

    @property
    def allowed_read_paths(self) -> list[Path]:
        return [
            Path(p.strip()).expanduser()
            for p in self.allowed_read_paths_raw.split(":")
            if p.strip()
        ]

    @property
    def allowed_write_paths(self) -> list[Path]:
        return [
            Path(p.strip()).expanduser()
            for p in self.allowed_write_paths_raw.split(":")
            if p.strip()
        ]

    @property
    def allowed_shell_commands(self) -> list[str]:
        if not self.allowed_shell_commands_raw.strip():
            return _DEFAULT_ALLOWED_COMMANDS
        return [c.strip() for c in self.allowed_shell_commands_raw.split(",") if c.strip()]

    @property
    def allowed_api_domains(self) -> list[str]:
        if not self.allowed_api_domains_raw.strip():
            return _DEFAULT_ALLOWED_DOMAINS
        return [d.strip() for d in self.allowed_api_domains_raw.split(",") if d.strip()]

    @property
    def browser_user_data_path(self) -> Path:
        return Path(self.browser_user_data_dir).expanduser()

    @property
    def agent_data_dir(self) -> Path:
        path = Path("~/.computer_agent").expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


# Safe defaults: basic read-only shell commands
_DEFAULT_ALLOWED_COMMANDS: list[str] = [
    "ls", "pwd", "cat", "head", "tail", "grep", "find", "echo",
    "date", "whoami", "which", "env", "printenv", "wc", "sort",
    "uniq", "cut", "awk", "sed", "tr", "diff", "stat",
    "python3", "python", "node", "npm", "pip", "uv",
    "git", "curl", "open", "pbcopy", "pbpaste",
]

# Safe API domains: common productivity services
_DEFAULT_ALLOWED_DOMAINS: list[str] = [
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.openai.com",
    "graph.microsoft.com",
    "www.googleapis.com",
    "api.github.com",
    "slack.com",
    "api.notion.com",
    "api.linear.app",
]


# Module-level singleton — imported by everything else
settings = Settings()
