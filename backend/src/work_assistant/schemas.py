from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
MessageRole = Literal["user", "assistant"]
RunFailureCode = Literal["run_timeout", "agent_execution_failed", "service_restarted"]
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
RUNTIME_EVENT_TYPES = frozenset(
    {"tool.started", "tool.finished", "message.delta", "source.added"}
)


class ProductEventValidationError(ValueError):
    """A Runtime or Host event is outside the stable public product contract."""


class ThreadCreate(BaseModel):
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


class RunCreate(BaseModel):
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


class RunFailedPayload(_StrictPayload):
    status: Literal["failed"]
    error_code: RunFailureCode


class RunCancelledPayload(_StrictPayload):
    status: Literal["cancelled"]


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
    return payload.model_dump(mode="json", exclude_none=True)


def validate_runtime_event(
    event_type: str, data: dict[str, Any]
) -> tuple[RuntimeEventType, dict[str, Any]]:
    if event_type not in RUNTIME_EVENT_TYPES:
        raise ProductEventValidationError("runtime_event_not_allowed")
    return cast(RuntimeEventType, event_type), validate_product_event(event_type, data)


def normalize_stored_product_event(
    event_type: str, data: dict[str, Any]
) -> dict[str, Any]:
    """Map the one v0.1 failure code to the frozen v0.2 public contract on read."""
    normalized = dict(data)
    if event_type == "run.failed" and normalized.get("error_code") == "agent_result_missing":
        normalized["error_code"] = "agent_execution_failed"
    return normalized


class ThreadList(BaseModel):
    items: list[ThreadSummary]


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
