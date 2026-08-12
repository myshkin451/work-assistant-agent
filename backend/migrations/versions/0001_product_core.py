"""Create the product-owned Thread, Run, Message, and Event tables.

Revision ID: 0001_product_core
Revises: None
Create Date: 2026-08-12

LangGraph checkpoint tables are deliberately outside this migration history.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_product_core"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "product_threads",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "product_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("last_seq", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('created', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_product_run_status",
        ),
        sa.ForeignKeyConstraint(["thread_id"], ["product_threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("thread_id", "idempotency_key", name="uq_product_run_idempotency"),
    )
    op.create_index("ix_product_runs_created_at", "product_runs", ["created_at"])
    op.create_index(
        "uq_product_run_one_active_thread",
        "product_runs",
        ["thread_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('created', 'running')"),
        sqlite_where=sa.text("status IN ('created', 'running')"),
    )
    op.create_table(
        "product_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_product_message_role"),
        sa.ForeignKeyConstraint(["run_id"], ["product_runs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["thread_id"], ["product_threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_product_messages_thread_created",
        "product_messages",
        ["thread_id", "created_at"],
    )
    op.create_table(
        "product_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=48), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["product_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["thread_id"], ["product_threads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "seq", name="uq_product_event_run_seq"),
    )
    op.create_index("ix_product_events_run_seq", "product_events", ["run_id", "seq"])


def downgrade() -> None:
    op.drop_index("ix_product_events_run_seq", table_name="product_events")
    op.drop_table("product_events")
    op.drop_index("ix_product_messages_thread_created", table_name="product_messages")
    op.drop_table("product_messages")
    op.drop_index("uq_product_run_one_active_thread", table_name="product_runs")
    op.drop_index("ix_product_runs_created_at", table_name="product_runs")
    op.drop_table("product_runs")
    op.drop_table("product_threads")
