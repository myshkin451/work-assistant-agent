from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunStatus = Literal["created", "running", "completed", "failed", "cancelled"]
MessageRole = Literal["user", "assistant"]


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
    model_config = ConfigDict(from_attributes=True)

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


class ThreadSnapshot(ThreadSummary):
    messages: list[Message]
    active_run: RunView | None


class ThreadList(BaseModel):
    items: list[ThreadSummary]


class EventEnvelope(BaseModel):
    event_id: str
    run_id: str
    thread_id: str
    seq: int
    type: str
    occurred_at: datetime
    data: dict[str, Any]
