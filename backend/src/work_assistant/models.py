from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ThreadRecord(Base):
    __tablename__ = "product_threads"
    __table_args__ = (
        Index("ix_product_threads_owner_updated", "owner_subject", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RunRecord(Base):
    __tablename__ = "product_runs"
    __table_args__ = (
        UniqueConstraint("thread_id", "idempotency_key", name="uq_product_run_idempotency"),
        UniqueConstraint("id", "thread_id", name="uq_product_run_id_thread"),
        CheckConstraint(
            "status IN ('created', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_product_run_status",
        ),
        Index(
            "uq_product_run_one_active_thread",
            "thread_id",
            unique=True,
            postgresql_where=text("status IN ('created', 'running')"),
            sqlite_where=text("status IN ('created', 'running')"),
        ),
        Index("ix_product_runs_created_at", "created_at"),
        Index("ix_product_runs_actor_created", "actor_subject", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("product_threads.id", ondelete="CASCADE"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_subject: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="created")
    last_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # T-006+ Runs freeze their evaluated policy at creation and write an outcome
    # only in the winning terminal transaction. Pre-T-006 rows remain explicitly
    # null rather than being assigned invented modern policy evidence.
    execution_plan: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    execution_outcome: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    # T-010 does not backfill legacy Runs. A non-null version means the Run was
    # admitted with provider-attempt metering enabled; terminal usage is written
    # only by the winning terminal transaction.
    metering_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    first_visible_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    usage_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelAttemptRecord(Base):
    __tablename__ = "product_model_attempts"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "call_index",
            "attempt_index",
            name="uq_product_model_attempt_position",
        ),
        ForeignKeyConstraint(
            ["run_id", "thread_id"],
            ["product_runs.id", "product_runs.thread_id"],
            name="fk_product_model_attempts_run_thread",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "call_kind IN ('direct', 'decision', 'finalizer')",
            name="ck_product_model_attempt_kind",
        ),
        CheckConstraint(
            "status IN ('started', 'succeeded', 'failed', 'cancelled', 'interrupted')",
            name="ck_product_model_attempt_status",
        ),
        CheckConstraint("call_index >= 1", name="ck_product_model_attempt_call_index"),
        CheckConstraint("attempt_index >= 1", name="ck_product_model_attempt_retry_index"),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_product_model_attempt_input_tokens",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_product_model_attempt_output_tokens",
        ),
        CheckConstraint(
            "cached_tokens IS NULL OR cached_tokens >= 0",
            name="ck_product_model_attempt_cached_tokens",
        ),
        CheckConstraint(
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
            name="ck_product_model_attempt_reasoning_tokens",
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_product_model_attempt_total_tokens",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="ck_product_model_attempt_completed_at",
        ),
        Index("ix_product_model_attempts_run", "run_id", "call_index", "attempt_index"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False)
    call_index: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_index: Mapped[int] = mapped_column(Integer, nullable=False)
    call_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="started")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)


class MessageRecord(Base):
    __tablename__ = "product_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_product_message_role"),
        ForeignKeyConstraint(
            ["run_id", "thread_id"],
            ["product_runs.id", "product_runs.thread_id"],
            name="fk_product_messages_run_thread",
        ),
        Index("ix_product_messages_thread_created", "thread_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("product_threads.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("product_runs.id", ondelete="SET NULL"), nullable=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class EventRecord(Base):
    __tablename__ = "product_events"
    __table_args__ = (
        UniqueConstraint("run_id", "seq", name="uq_product_event_run_seq"),
        ForeignKeyConstraint(
            ["run_id", "thread_id"],
            ["product_runs.id", "product_runs.thread_id"],
            name="fk_product_events_run_thread",
        ),
        Index("ix_product_events_run_seq", "run_id", "seq"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("product_runs.id", ondelete="CASCADE"), nullable=False
    )
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("product_threads.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(48), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
