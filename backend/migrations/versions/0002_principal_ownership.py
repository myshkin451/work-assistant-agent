"""Add Principal ownership and quarantine unattributed v0.2 data.

Revision ID: 0002_principal_ownership
Revises: 0001_product_core
Create Date: 2026-08-13

Pre-authentication rows are preserved under an internal subject that no
IdentityProvider may emit. They require an explicit future offline reassignment;
they are never claimed by the first authenticated user.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

revision: str = "0002_principal_ownership"
down_revision: str | None = "0001_product_core"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_UNOWNED_SUBJECT = "urn:work-assistant:internal:legacy-unowned:v0.2"


def _require_online_migration(direction: str) -> None:
    if context.is_offline_mode():
        raise RuntimeError(f"ownership {direction} requires an online connection")


def _validate_legacy_child_links() -> None:
    connection = op.get_bind()
    for child_table in ("product_messages", "product_events"):
        mismatch_count = connection.execute(
            sa.text(
                f"SELECT COUNT(*) FROM {child_table} AS child "
                "LEFT JOIN product_runs AS run ON run.id = child.run_id "
                "WHERE child.run_id IS NOT NULL "
                "AND (run.id IS NULL OR run.thread_id <> child.thread_id)"
            )
        ).scalar_one()
        if mismatch_count:
            raise RuntimeError("legacy product child has inconsistent run and thread")


def upgrade() -> None:
    _require_online_migration("upgrade")
    # SQLite table rebuilds do not validate pre-existing rows while Alembic has
    # foreign-key checks disabled. Refuse unsafe legacy data before any DDL so a
    # failed migration remains entirely at v0.2.
    _validate_legacy_child_links()
    with op.batch_alter_table("product_threads") as batch:
        batch.add_column(sa.Column("owner_subject", sa.String(length=255), nullable=True))
    with op.batch_alter_table("product_runs") as batch:
        batch.add_column(sa.Column("actor_subject", sa.String(length=255), nullable=True))

    op.execute(
        sa.text("UPDATE product_threads SET owner_subject = :subject").bindparams(
            subject=LEGACY_UNOWNED_SUBJECT
        )
    )
    op.execute(
        sa.text("UPDATE product_runs SET actor_subject = :subject").bindparams(
            subject=LEGACY_UNOWNED_SUBJECT
        )
    )

    with op.batch_alter_table("product_threads") as batch:
        batch.alter_column(
            "owner_subject",
            existing_type=sa.String(length=255),
            nullable=False,
        )
        batch.create_index(
            "ix_product_threads_owner_updated",
            ["owner_subject", "updated_at"],
            unique=False,
        )
    with op.batch_alter_table("product_runs") as batch:
        batch.alter_column(
            "actor_subject",
            existing_type=sa.String(length=255),
            nullable=False,
        )
        batch.create_unique_constraint(
            "uq_product_run_id_thread",
            ["id", "thread_id"],
        )
    with op.batch_alter_table("product_messages") as batch:
        batch.create_foreign_key(
            "fk_product_messages_run_thread",
            "product_runs",
            ["run_id", "thread_id"],
            ["id", "thread_id"],
        )
    with op.batch_alter_table("product_events") as batch:
        batch.create_foreign_key(
            "fk_product_events_run_thread",
            "product_runs",
            ["run_id", "thread_id"],
            ["id", "thread_id"],
        )


def downgrade() -> None:
    _require_online_migration("downgrade")
    connection = op.get_bind()
    for table_name in (
        "product_threads",
        "product_runs",
        "product_messages",
        "product_events",
    ):
        count = connection.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
        if count:
            raise RuntimeError("ownership migration cannot downgrade non-empty product data")

    with op.batch_alter_table("product_events") as batch:
        batch.drop_constraint("fk_product_events_run_thread", type_="foreignkey")
    with op.batch_alter_table("product_messages") as batch:
        batch.drop_constraint("fk_product_messages_run_thread", type_="foreignkey")
    with op.batch_alter_table("product_runs") as batch:
        batch.drop_constraint("uq_product_run_id_thread", type_="unique")
        batch.drop_column("actor_subject")
    with op.batch_alter_table("product_threads") as batch:
        batch.drop_index("ix_product_threads_owner_updated")
        batch.drop_column("owner_subject")
