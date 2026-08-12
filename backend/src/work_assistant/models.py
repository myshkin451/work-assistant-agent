from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
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

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RunRecord(Base):
    __tablename__ = "product_runs"
    __table_args__ = (
        UniqueConstraint("thread_id", "idempotency_key", name="uq_product_run_idempotency"),
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
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("product_threads.id", ondelete="CASCADE"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="created")
    last_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MessageRecord(Base):
    __tablename__ = "product_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_product_message_role"),
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
