"""Persist auditable provider attempts and terminal Run usage.

Revision ID: 0004_run_usage_metering
Revises: 0003_agent_policy_evidence
Create Date: 2026-08-20

Existing Runs deliberately remain NULL. Their timestamps, model steps, and text
lengths predate this contract and cannot be reinterpreted as provider evidence.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0004_run_usage_metering"
down_revision: str | None = "0003_agent_policy_evidence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("product_runs") as batch:
        batch.add_column(sa.Column("metering_version", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("first_visible_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(
            sa.Column("usage_metrics", sa.JSON(none_as_null=True), nullable=True)
        )
        batch.create_index(
            "ix_product_runs_actor_created",
            ["actor_subject", "created_at"],
            unique=False,
        )

    op.create_table(
        "product_model_attempts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("thread_id", sa.String(length=36), nullable=False),
        sa.Column("call_index", sa.Integer(), nullable=False),
        sa.Column("attempt_index", sa.Integer(), nullable=False),
        sa.Column("call_kind", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cached_tokens", sa.Integer(), nullable=True),
        sa.Column("reasoning_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "call_kind IN ('direct', 'decision', 'finalizer')",
            name="ck_product_model_attempt_kind",
        ),
        sa.CheckConstraint(
            "status IN ('started', 'succeeded', 'failed', 'cancelled', 'interrupted')",
            name="ck_product_model_attempt_status",
        ),
        sa.CheckConstraint("call_index >= 1", name="ck_product_model_attempt_call_index"),
        sa.CheckConstraint(
            "attempt_index >= 1", name="ck_product_model_attempt_retry_index"
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_product_model_attempt_input_tokens",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_product_model_attempt_output_tokens",
        ),
        sa.CheckConstraint(
            "cached_tokens IS NULL OR cached_tokens >= 0",
            name="ck_product_model_attempt_cached_tokens",
        ),
        sa.CheckConstraint(
            "reasoning_tokens IS NULL OR reasoning_tokens >= 0",
            name="ck_product_model_attempt_reasoning_tokens",
        ),
        sa.CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_product_model_attempt_total_tokens",
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="ck_product_model_attempt_completed_at",
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "thread_id"],
            ["product_runs.id", "product_runs.thread_id"],
            name="fk_product_model_attempts_run_thread",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "call_index",
            "attempt_index",
            name="uq_product_model_attempt_position",
        ),
    )
    op.create_index(
        "ix_product_model_attempts_run",
        "product_model_attempts",
        ["run_id", "call_index", "attempt_index"],
    )


def downgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("Run usage metering downgrade requires an online connection")
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        connection.execute(
            sa.text(
                "LOCK TABLE product_runs, product_model_attempts IN ACCESS EXCLUSIVE MODE"
            )
        )
    evidence_count = connection.execute(
        sa.text(
            "SELECT "
            "(SELECT COUNT(*) FROM product_model_attempts) + "
            "(SELECT COUNT(*) FROM product_runs "
            "WHERE metering_version IS NOT NULL OR first_visible_at IS NOT NULL "
            "OR usage_metrics IS NOT NULL)"
        )
    ).scalar_one()
    if evidence_count:
        raise RuntimeError("Run usage metering migration cannot discard recorded evidence")

    op.drop_index("ix_product_model_attempts_run", table_name="product_model_attempts")
    op.drop_table("product_model_attempts")
    with op.batch_alter_table("product_runs") as batch:
        batch.drop_index("ix_product_runs_actor_created")
        batch.drop_column("usage_metrics")
        batch.drop_column("first_visible_at")
        batch.drop_column("metering_version")
