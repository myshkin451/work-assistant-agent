"""Persist immutable Agent policy plans and terminal execution outcomes.

Revision ID: 0003_agent_policy_evidence
Revises: 0002_principal_ownership
Create Date: 2026-08-17

Existing Runs deliberately remain NULL: assigning them a T-006 Agent, Prompt,
Tool policy, or budget would manufacture audit evidence that did not exist.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0003_agent_policy_evidence"
down_revision: str | None = "0002_principal_ownership"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("product_runs") as batch:
        batch.add_column(
            sa.Column("execution_plan", sa.JSON(none_as_null=True), nullable=True)
        )
        batch.add_column(
            sa.Column("execution_outcome", sa.JSON(none_as_null=True), nullable=True)
        )


def downgrade() -> None:
    if context.is_offline_mode():
        raise RuntimeError("Agent policy evidence downgrade requires an online connection")
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        # Hold the same table lock needed by DROP COLUMN before inspecting the
        # guard. Without it, an application transaction could write the first
        # policy evidence after COUNT(*) returned zero and before ALTER TABLE,
        # causing the downgrade to discard evidence it promised to preserve.
        connection.execute(sa.text("LOCK TABLE product_runs IN ACCESS EXCLUSIVE MODE"))
    evidence_count = connection.execute(
        sa.text(
            "SELECT COUNT(*) FROM product_runs "
            "WHERE execution_plan IS NOT NULL OR execution_outcome IS NOT NULL"
        )
    ).scalar_one()
    if evidence_count:
        raise RuntimeError("Agent policy evidence migration cannot discard recorded evidence")
    with op.batch_alter_table("product_runs") as batch:
        batch.drop_column("execution_outcome")
        batch.drop_column("execution_plan")
