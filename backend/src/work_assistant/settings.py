from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated server configuration.

    No dotenv file is loaded implicitly. Production and live-smoke callers must
    deliberately provide environment variables or an explicit dotenv path.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    app_env: Literal["development", "test", "production"] = "development"
    database_url: str = (
        "postgresql+psycopg://work_assistant:work_assistant@localhost:55432/work_assistant"
    )
    checkpoint_database_url: str = (
        "postgresql://work_assistant:work_assistant@localhost:55432/work_assistant"
    )
    model_mode: Literal["fake", "deepseek"] = "fake"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_model_steps: int = Field(default=8, ge=2, le=32)
    max_tool_calls: int = Field(default=4, ge=1, le=16)
    run_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    model_concurrency: int = Field(default=5, ge=1, le=50)
    fake_step_delay_seconds: float = Field(default=0.02, ge=0, le=10)
    sse_poll_interval_seconds: float = Field(default=0.05, gt=0, le=2)
    sse_keepalive_seconds: float = Field(default=15.0, gt=0, le=60)

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith(("postgresql+psycopg://", "sqlite+aiosqlite://")):
            raise ValueError("unsupported product database driver")
        return value

    @field_validator("checkpoint_database_url")
    @classmethod
    def validate_checkpoint_database_url(cls, value: str) -> str:
        if not value.startswith(("postgresql://", "postgres://")):
            raise ValueError("checkpoint database must be PostgreSQL")
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def validate_deepseek_base_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("DeepSeek base URL must use HTTPS")
        return value.rstrip("/")

    @field_validator("deepseek_model")
    @classmethod
    def validate_deepseek_model(cls, value: str) -> str:
        if value not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
            raise ValueError("DeepSeek model is not in the public allowlist")
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
