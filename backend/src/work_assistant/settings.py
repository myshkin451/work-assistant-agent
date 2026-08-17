from __future__ import annotations

from functools import lru_cache
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
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
    identity_provider_mode: Literal["external", "anonymous", "development_header"] = "external"
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
    max_identical_tool_calls: int = Field(default=1, ge=1, le=5)
    max_no_progress_steps: int = Field(default=2, ge=1, le=8)
    run_timeout_seconds: float = Field(default=120.0, gt=0, le=900)
    database_operation_timeout_seconds: float = Field(default=2.0, ge=0.1, le=30)
    repository_cleanup_grace_seconds: float = Field(default=3.0, ge=0.2, le=60)
    model_concurrency: int = Field(default=5, ge=1, le=50)
    fake_step_delay_seconds: float = Field(default=0.02, ge=0, le=10)
    sse_poll_interval_seconds: float = Field(default=0.05, gt=0, le=2)
    sse_keepalive_seconds: float = Field(default=15.0, gt=0, le=60)

    @model_validator(mode="after")
    def validate_identity_mode(self) -> Settings:
        if self.app_env == "production" and self.identity_provider_mode != "external":
            raise ValueError("production requires an external identity provider")
        if self.app_env == "production" and self.model_mode == "fake":
            raise ValueError("production requires a non-fake model profile")
        if self.repository_cleanup_grace_seconds <= self.database_operation_timeout_seconds:
            raise ValueError("repository cleanup grace must exceed the database timeout")
        if (
            self.database_url.startswith("postgresql+psycopg://")
            and self.database_operation_timeout_seconds < 2
        ):
            raise ValueError("PostgreSQL database timeout must be at least two seconds")
        if self.app_env == "production":
            for origin in self.cors_origins:
                hostname = urlsplit(origin).hostname
                try:
                    is_loopback = hostname == "localhost" or (
                        hostname is not None and ip_address(hostname).is_loopback
                    )
                except ValueError:
                    is_loopback = False
                if is_loopback:
                    raise ValueError("production requires explicit non-loopback allowed origins")
        return self

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

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, value: str) -> str:
        origins = [item.strip() for item in value.split(",") if item.strip()]
        if not origins or len(set(origins)) != len(origins):
            raise ValueError("allowed origins must be a non-empty unique list")
        for origin in origins:
            if origin in {"*", "null"}:
                raise ValueError("credentialed CORS requires exact origins")
            parsed = urlsplit(origin)
            try:
                parsed_port = parsed.port
            except ValueError as exc:
                raise ValueError("allowed origin has an invalid port") from exc
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or (parsed_port is not None and not 1 <= parsed_port <= 65535)
            ):
                raise ValueError("allowed origin must be an exact HTTP origin")
        return ",".join(origins)

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
