from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def test_product_migration_is_reversible_and_owns_no_checkpoint_tables(tmp_path: Path) -> None:
    backend = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "migration.db"
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")

    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
    assert {
        "product_threads",
        "product_runs",
        "product_messages",
        "product_events",
    }.issubset(tables)
    assert not any("checkpoint" in table for table in tables)

    command.downgrade(config, "base")
    with sqlite3.connect(database_path) as connection:
        remaining = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
    assert not any(table.startswith("product_") for table in remaining)
