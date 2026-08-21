from __future__ import annotations

import unicodedata
from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .identity import PrincipalDisplayExtensions
from .usage import RunUsage, UsageMetric, unavailable_run_usage

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
MessageRole = Literal["user", "assistant"]
RunFailureCode = Literal[
    "run_timeout",
    "agent_execution_failed",
    "service_restarted",
    "model_step_limit",
    "tool_call_limit",
    "repeated_tool_call",
    "no_progress",
    "tool_not_allowed",
    "result_schema_invalid",
    "source_validation_failed",
]
ProductEventType = Literal[
    "run.started",
    "tool.started",
    "tool.finished",
    "message.delta",
    "source.added",
    "message.completed",
    "run.completed",
    "run.failed",
    "run.cancelled",
]
RuntimeEventType = Literal[
    "tool.started",
    "tool.finished",
    "message.delta",
    "source.added",
]

PRODUCT_EVENT_TYPES = frozenset(
    {
        "run.started",
        "tool.started",
        "tool.finished",
        "message.delta",
        "source.added",
        "message.completed",
        "run.completed",
        "run.failed",
        "run.cancelled",
    }
)
RUNTIME_EVENT_TYPES = frozenset({"tool.started", "tool.finished", "message.delta", "source.added"})


class ProductEventValidationError(ValueError):
    """A Runtime or Host event is outside the stable public product contract."""


class ThreadCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class ThreadUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if any(unicodedata.category(character).startswith("C") for character in value):
            raise ValueError("title must not contain control characters")
        value = " ".join(value.split())
        if not value:
            raise ValueError("title must not be blank")
        return value


class RunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8_000)
    idempotency_key: str = Field(min_length=1, max_length=128)

    @field_validator("message", "idempotency_key")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class ThreadSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class Message(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    message_id: str
    role: MessageRole
    content: str
    created_at: datetime
    run_id: str | None


class RunView(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    thread_id: str
    status: RunStatus
    last_seq: int
    created_at: datetime
    completed_at: datetime | None
    usage: RunUsage = Field(
        default_factory=lambda: unavailable_run_usage(state="unknown")
    )


class InitialRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread: ThreadSummary
    run: RunView


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStartedPayload(_StrictPayload):
    status: Literal["running"]


class ToolStartedPayload(_StrictPayload):
    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=200)
    input_summary: str | None = Field(default=None, max_length=500)


class ToolFinishedPayload(_StrictPayload):
    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=200)
    output_summary: str = Field(min_length=1, max_length=1_000)


class MessageDeltaPayload(_StrictPayload):
    delta: str = Field(min_length=1, max_length=8_000)


class SourceAddedPayload(_StrictPayload):
    source_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1_000)


class MessageCompletedPayload(_StrictPayload):
    message: Message


class RunCompletedPayload(_StrictPayload):
    status: Literal["completed"]
    usage: RunUsage | None = None


class RunFailedPayload(_StrictPayload):
    status: Literal["failed"]
    error_code: RunFailureCode
    usage: RunUsage | None = None


class RunCancelledPayload(_StrictPayload):
    status: Literal["cancelled"]
    usage: RunUsage | None = None


EVENT_PAYLOAD_MODELS: dict[str, type[_StrictPayload]] = {
    "run.started": RunStartedPayload,
    "tool.started": ToolStartedPayload,
    "tool.finished": ToolFinishedPayload,
    "message.delta": MessageDeltaPayload,
    "source.added": SourceAddedPayload,
    "message.completed": MessageCompletedPayload,
    "run.completed": RunCompletedPayload,
    "run.failed": RunFailedPayload,
    "run.cancelled": RunCancelledPayload,
}


def validate_product_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    payload_model = EVENT_PAYLOAD_MODELS.get(event_type)
    if payload_model is None:
        raise ProductEventValidationError("unsupported_product_event")
    try:
        payload = payload_model.model_validate(data)
    except ValidationError as exc:
        raise ProductEventValidationError("invalid_product_event_payload") from exc
    if event_type in {"run.completed", "run.failed", "run.cancelled"}:
        # Current terminal events preserve the exact REST usage shape, including
        # explicit nulls for fields the provider did not return. Legacy terminal
        # events remain valid and simply omit the optional usage object.
        terminal_payload = payload.model_dump(mode="json")
        if terminal_payload.get("usage") is None:
            terminal_payload.pop("usage", None)
        return terminal_payload
    return payload.model_dump(mode="json", exclude_none=True)


def validate_runtime_event(
    event_type: str, data: dict[str, Any]
) -> tuple[RuntimeEventType, dict[str, Any]]:
    if event_type not in RUNTIME_EVENT_TYPES:
        raise ProductEventValidationError("runtime_event_not_allowed")
    return cast(RuntimeEventType, event_type), validate_product_event(event_type, data)


def normalize_stored_product_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Map the one v0.1 failure code to the frozen v0.2 public contract on read."""
    normalized = dict(data)
    if event_type == "run.failed" and normalized.get("error_code") == "agent_result_missing":
        normalized["error_code"] = "agent_execution_failed"
    return normalized


class ThreadList(BaseModel):
    items: list[ThreadSummary]


UsageRange = Literal["7d", "30d", "all"]


class AccountView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=200)
    organization: str | None = Field(default=None, max_length=200)
    extensions: PrincipalDisplayExtensions = Field(
        default_factory=PrincipalDisplayExtensions
    )


class AccountUsageScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    range: UsageRange
    from_at: datetime | None
    to_at: datetime
    thread_id: str | None


class AccountRunCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    active: int = Field(ge=0)


class AccountUsageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: AccountView
    scope: AccountUsageScope
    runs: AccountRunCounts
    model_calls: UsageMetric
    retries: UsageMetric
    input_tokens: UsageMetric
    output_tokens: UsageMetric
    cached_tokens: UsageMetric
    reasoning_tokens: UsageMetric
    total_tokens: UsageMetric


class EventEnvelope(BaseModel):
    event_id: str
    run_id: str
    thread_id: str
    seq: int
    type: ProductEventType
    occurred_at: datetime
    data: dict[str, Any]

    @model_validator(mode="after")
    def validate_public_payload(self) -> EventEnvelope:
        self.data = validate_product_event(self.type, self.data)
        return self


class RunSnapshot(RunView):
    events: list[EventEnvelope]


class ThreadSnapshot(ThreadSummary):
    messages: list[Message]
    runs: list[RunSnapshot]
    active_run: RunView | None
