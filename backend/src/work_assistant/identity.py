from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Protocol

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .settings import Settings

INTERNAL_SUBJECT_PREFIX = "urn:work-assistant:internal:"
LEGACY_UNOWNED_SUBJECT = f"{INTERNAL_SUBJECT_PREFIX}legacy-unowned:v0.2"
ANONYMOUS_DEVELOPMENT_SUBJECT = "urn:work-assistant:principal:anonymous-development"
DEVELOPMENT_PRINCIPAL_HEADER = "X-Work-Assistant-Dev-Subject"


class PrincipalDisplayExtensions(BaseModel):
    """Optional, pre-sanitized account facts supplied by an identity adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_expires_at: datetime | None = None
    permission_summary: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("session_expires_at")
    @classmethod
    def require_aware_expiry(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("session expiry must include a timezone")
        return value

    @field_validator("permission_summary")
    @classmethod
    def clean_permission_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value or any(
            unicodedata.category(character).startswith("C") for character in value
        ):
            raise ValueError("permission summary is not display safe")
        return value


class Principal(BaseModel):
    """Authenticated application identity without credential material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1, max_length=255)
    display_name: str | None = Field(default=None, max_length=200)
    organization: str | None = Field(default=None, max_length=200)
    roles: tuple[str, ...] = ()
    session_id: str | None = Field(default=None, max_length=255)
    display_extensions: PrincipalDisplayExtensions = Field(
        default_factory=PrincipalDisplayExtensions
    )

    @field_validator("subject")
    @classmethod
    def validate_subject(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("principal subject must not contain surrounding whitespace")
        if value.startswith(INTERNAL_SUBJECT_PREFIX):
            raise ValueError("principal subject uses a reserved internal namespace")
        if any(unicodedata.category(character).startswith("C") for character in value):
            raise ValueError("principal subject must not contain control characters")
        return value

    @field_validator("display_name", "organization")
    @classmethod
    def clean_display_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value or any(
            unicodedata.category(character).startswith("C") for character in value
        ):
            raise ValueError("principal display text is invalid")
        return value

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 64:
            raise ValueError("principal has too many roles")
        if len(set(value)) != len(value):
            raise ValueError("principal roles must be unique")
        for role in value:
            if not role or role != role.strip() or len(role) > 128:
                raise ValueError("principal role is invalid")
            if any(unicodedata.category(character).startswith("C") for character in role):
                raise ValueError("principal role must not contain control characters")
        return value


class IdentityProvider(Protocol):
    async def authenticate(self, request: Request) -> Principal | None: ...


class IdentityConfigurationError(RuntimeError):
    """The configured deployment cannot provide a trustworthy current identity."""


class AnonymousIdentityProvider:
    """Explicit local-only compatibility identity for development and tests."""

    _principal = Principal(
        subject=ANONYMOUS_DEVELOPMENT_SUBJECT,
        display_name="当前用户",
    )

    async def authenticate(self, request: Request) -> Principal:
        del request
        return self._principal


class DevelopmentHeaderIdentityProvider:
    """Local-only deterministic identity used by isolation tests."""

    async def authenticate(self, request: Request) -> Principal | None:
        values = request.headers.getlist(DEVELOPMENT_PRINCIPAL_HEADER)
        if len(values) != 1:
            return None
        try:
            return Principal(subject=values[0], display_name=values[0])
        except ValueError:
            return None


async def authenticate_request(
    provider: IdentityProvider,
    request: Request,
) -> Principal | None:
    """Validate an adapter assertion at the Host boundary."""

    try:
        assertion = await provider.authenticate(request)
        if not isinstance(assertion, Principal):
            return None
        # Rebuild from plain data so model_construct and subclass adapters cannot
        # bypass the Host's subject validators.
        return Principal.model_validate(assertion.model_dump())
    except Exception:
        # Provider exceptions and assertion details may contain credential data.
        # They are deliberately collapsed without exception chaining.
        return None


def resolve_identity_provider(
    settings: Settings,
    external_provider: IdentityProvider | None,
) -> IdentityProvider:
    """Resolve one explicit provider without ever falling back to anonymous."""

    if settings.app_env == "production" and isinstance(
        external_provider,
        (AnonymousIdentityProvider, DevelopmentHeaderIdentityProvider),
    ):
        raise IdentityConfigurationError("development_identity_provider_in_production")
    if settings.identity_provider_mode == "external":
        if external_provider is None:
            raise IdentityConfigurationError("external_identity_provider_missing")
        return external_provider
    if external_provider is not None:
        raise IdentityConfigurationError("external_identity_provider_unexpected")
    if settings.identity_provider_mode == "anonymous":
        return AnonymousIdentityProvider()
    if settings.identity_provider_mode == "development_header":
        return DevelopmentHeaderIdentityProvider()
    raise IdentityConfigurationError("unsupported_identity_provider_mode")
