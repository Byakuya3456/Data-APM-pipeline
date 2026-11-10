"""Application settings and configuration management via Pydantic BaseSettings."""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def discover_project_root() -> Path:
    """Dynamically discover project root directory.

    Order of precedence:
    1. Explicit `PROJECT_ROOT` env var if directory exists on disk.
    2. Parent directory search starting from this file looking for project markers.
    3. Parent directory search starting from current working directory.
    4. Fallback to `Path.cwd().resolve()`.
    """
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        candidate = Path(str(env_root).strip('"\'')).expanduser()
        try:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved
        except Exception:
            pass

    # Search upwards from config/ directory
    anchor = Path(__file__).resolve().parent
    for candidate in [anchor] + list(anchor.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "PROJECT.md").exists():
            return candidate

    # Search upwards from current working directory
    cwd = Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists() or (candidate / "PROJECT.md").exists():
            return candidate

    return cwd


_DISCOVERED_ROOT = discover_project_root()


class Settings(BaseSettings):
    """Platform-wide centralized configuration schema."""

    model_config = SettingsConfigDict(
        env_file=(str(_DISCOVERED_ROOT / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core paths
    project_root: Path = Field(
        default_factory=discover_project_root,
        validation_alias=AliasChoices("PROJECT_ROOT", "project_root"),
    )
    data_input_dir: Path = Field(
        default=Path("data/input"),
        validation_alias=AliasChoices("DATA_INPUT_DIR", "data_input_dir"),
    )
    workspace_dir: Path = Field(
        default=Path("workspace"),
        validation_alias=AliasChoices("WORKSPACE_DIR", "workspace_dir"),
    )
    temp_dir: Path = Field(
        default=Path("temp"),
        validation_alias=AliasChoices("TEMP_DIR", "temp_dir"),
    )

    # OpenAI-compatible LLM configuration (accepts both AI_* and OPENAI_* keys)
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("AI_API_KEY", "OPENAI_API_KEY", "openai_api_key"),
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("AI_BASE_URL", "OPENAI_BASE_URL", "openai_base_url"),
    )
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("AI_MODEL", "OPENAI_MODEL", "openai_model"),
    )
    openai_temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices("AI_TEMPERATURE", "OPENAI_TEMPERATURE", "openai_temperature"),
    )
    openai_max_tokens: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("AI_MAX_TOKENS", "OPENAI_MAX_TOKENS", "openai_max_tokens"),
    )
    openai_timeout: float = Field(
        default=60.0,
        validation_alias=AliasChoices("AI_REQUEST_TIMEOUT", "OPENAI_REQUEST_TIMEOUT", "openai_timeout"),
    )
    openai_max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("AI_MAX_RETRIES", "OPENAI_MAX_RETRIES", "openai_max_retries"),
    )

    # Out-of-core and resource limits
    duckdb_memory_limit: str = Field(
        default="8GB",
        validation_alias=AliasChoices("DUCKDB_MEMORY_LIMIT", "duckdb_memory_limit"),
    )
    duckdb_threads: int = Field(
        default=4,
        validation_alias=AliasChoices("DUCKDB_THREADS", "duckdb_threads"),
    )
    max_repair_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices("MAX_REPAIR_ATTEMPTS", "max_repair_attempts"),
    )
    max_script_seconds: int = Field(
        default=3600,
        validation_alias=AliasChoices("MAX_SCRIPT_SECONDS", "max_script_seconds"),
    )
    max_worker_memory_gb: float = Field(
        default=12.0,
        validation_alias=AliasChoices("MAX_WORKER_MEMORY_GB", "max_worker_memory_gb"),
    )
    max_data_sample_rows: int = Field(
        default=100,
        validation_alias=AliasChoices("MAX_DATA_SAMPLE_ROWS", "max_data_sample_rows"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> Settings:
        """Resolve all relative paths against dynamic project_root and register root in sys.path."""
        if not self.project_root.is_absolute():
            self.project_root = self.project_root.resolve()

        if not self.data_input_dir.is_absolute():
            self.data_input_dir = (self.project_root / self.data_input_dir).resolve()

        if not self.workspace_dir.is_absolute():
            self.workspace_dir = (self.project_root / self.workspace_dir).resolve()

        if not self.temp_dir.is_absolute():
            self.temp_dir = (self.project_root / self.temp_dir).resolve()

        # Ensure project root is available on sys.path for child module imports
        root_str = str(self.project_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        return self

    # Derived workspace directory properties
    @property
    def workspace_generated_dir(self) -> Path:
        return self.workspace_dir / "generated"

    @property
    def workspace_intermediate_dir(self) -> Path:
        return self.workspace_dir / "intermediate"

    @property
    def workspace_artifacts_dir(self) -> Path:
        return self.workspace_dir / "artifacts"

    @property
    def workspace_checkpoints_dir(self) -> Path:
        return self.workspace_dir / "checkpoints"

    @property
    def workspace_reports_dir(self) -> Path:
        return self.workspace_dir / "reports"

    @property
    def workspace_logs_dir(self) -> Path:
        return self.workspace_dir / "logs"

    @property
    def data_samples_dir(self) -> Path:
        return self.project_root / "data" / "samples"

    def ensure_directories(self) -> None:
        """Create all required directory hierarchies if they do not exist."""
        directories = [
            self.data_input_dir,
            self.data_samples_dir,
            self.workspace_dir,
            self.workspace_generated_dir,
            self.workspace_intermediate_dir,
            self.workspace_artifacts_dir,
            self.workspace_checkpoints_dir,
            self.workspace_reports_dir,
            self.workspace_logs_dir,
            self.temp_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
