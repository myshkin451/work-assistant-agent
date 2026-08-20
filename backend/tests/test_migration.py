from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient

from work_assistant.identity import DEVELOPMENT_PRINCIPAL_HEADER, LEGACY_UNOWNED_SUBJECT
from work_assistant.main import create_app
from work_assistant.settings import Settings

BACKEND = Path(__file__).resolve().parents[1]


def migration_config(backend: Path, database_path: Path) -> Config:
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
    return config


def test_migration_setup_preserves_existing_application_loggers(tmp_path: Path) -> None:
    application_logger = logging.getLogger("work_assistant.service")
    application_logger.disabled = False

    command.upgrade(migration_config(BACKEND, tmp_path / "logging.db"), "head")

    assert application_logger.disabled is False


def test_product_migration_is_reversible_and_owns_no_checkpoint_tables(tmp_path: Path) -> None:
    backend = BACKEND
    database_path = tmp_path / "migration.db"
    config = migration_config(backend, database_path)

    command.upgrade(config, "0002_principal_ownership")
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            "0002_principal_ownership"
        )
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


async def test_v02_rows_are_quarantined_and_nonempty_downgrade_is_blocked(
    tmp_path: Path,
) -> None:
    backend = BACKEND
    database_path = tmp_path / "legacy-v02.db"
    config = migration_config(backend, database_path)
    command.upgrade(config, "0001_product_core")

    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO product_threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("legacy-thread", "Legacy private conversation", now, now),
        )
        connection.execute(
            "INSERT INTO product_runs "
            "(id, thread_id, idempotency_key, status, last_seq, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("legacy-run", "legacy-thread", "legacy-key", "completed", 1, now, now),
        )
        connection.execute(
            "INSERT INTO product_messages "
            "(id, thread_id, run_id, role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy-message",
                "legacy-thread",
                "legacy-run",
                "user",
                "Legacy content must never be auto-claimed",
                now,
            ),
        )
        connection.execute(
            "INSERT INTO product_events "
            "(id, run_id, thread_id, seq, type, occurred_at, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-event",
                "legacy-run",
                "legacy-thread",
                1,
                "run.completed",
                now,
                '{"status":"completed"}',
            ),
        )
        connection.commit()

    command.upgrade(config, "0002_principal_ownership")
    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        thread = connection.execute(
            "SELECT id, owner_subject, title FROM product_threads"
        ).fetchone()
        run = connection.execute(
            "SELECT id, actor_subject, status, last_seq, execution_plan, execution_outcome "
            "FROM product_runs"
        ).fetchone()
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "product_threads",
                "product_runs",
                "product_messages",
                "product_events",
            )
        }
        thread_columns = {
            row[1]: (row[3], row[4])
            for row in connection.execute("PRAGMA table_info(product_threads)")
        }
        run_columns = {
            row[1]: (row[3], row[4])
            for row in connection.execute("PRAGMA table_info(product_runs)")
        }
        thread_indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(product_threads)")
        }
        run_indexes = {row[1] for row in connection.execute("PRAGMA index_list(product_runs)")}
        run_index_columns = {
            index_name: tuple(
                row[2] for row in connection.execute(f'PRAGMA index_info("{index_name}")')
            )
            for index_name in run_indexes
        }
        message_foreign_keys = list(connection.execute("PRAGMA foreign_key_list(product_messages)"))
        event_foreign_keys = list(connection.execute("PRAGMA foreign_key_list(product_events)"))
        foreign_key_check = list(connection.execute("PRAGMA foreign_key_check"))
        message = connection.execute(
            "SELECT id, thread_id, run_id, role, content FROM product_messages"
        ).fetchone()
        event = connection.execute(
            "SELECT id, run_id, thread_id, seq, type, payload FROM product_events"
        ).fetchone()
    assert thread == (
        "legacy-thread",
        LEGACY_UNOWNED_SUBJECT,
        "Legacy private conversation",
    )
    assert run == ("legacy-run", LEGACY_UNOWNED_SUBJECT, "completed", 1, None, None)
    assert counts == {
        "product_threads": 1,
        "product_runs": 1,
        "product_messages": 1,
        "product_events": 1,
    }
    assert thread_columns["owner_subject"] == (1, None)
    assert run_columns["actor_subject"] == (1, None)
    assert "ix_product_threads_owner_updated" in thread_indexes
    assert ("id", "thread_id") in run_index_columns.values()

    def grouped_run_foreign_keys(
        rows: list[tuple[object, ...]],
    ) -> set[tuple[tuple[str, str], ...]]:
        grouped: dict[int, list[tuple[int, str, str]]] = {}
        for row in rows:
            if row[2] == "product_runs":
                grouped.setdefault(int(row[0]), []).append((int(row[1]), str(row[3]), str(row[4])))
        return {
            tuple((source, target) for _, source, target in sorted(items))
            for items in grouped.values()
        }

    expected_composite = (("run_id", "id"), ("thread_id", "thread_id"))
    assert expected_composite in grouped_run_foreign_keys(message_foreign_keys)
    assert expected_composite in grouped_run_foreign_keys(event_foreign_keys)
    assert foreign_key_check == []
    assert message == (
        "legacy-message",
        "legacy-thread",
        "legacy-run",
        "user",
        "Legacy content must never be auto-claimed",
    )
    assert event == (
        "legacy-event",
        "legacy-run",
        "legacy-thread",
        1,
        "run.completed",
        '{"status":"completed"}',
    )

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "INSERT INTO product_threads "
            "(id, owner_subject, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("other-thread", "neutral-owner", "Other", now, now),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO product_messages "
                "(id, thread_id, run_id, role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("mismatch-message", "other-thread", "legacy-run", "user", "blocked", now),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO product_events "
                "(id, run_id, thread_id, seq, type, occurred_at, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "mismatch-event",
                    "legacy-run",
                    "other-thread",
                    2,
                    "run.completed",
                    now,
                    '{"status":"completed"}',
                ),
            )

    settings = Settings(
        app_env="test",
        identity_provider_mode="development_header",
        database_url=f"sqlite+aiosqlite:///{database_path}",
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
        model_mode="fake",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        for subject in ("neutral-principal-a", "neutral-principal-b"):
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={DEVELOPMENT_PRINCIPAL_HEADER: subject},
            ) as client:
                listed = await client.get("/api/threads")
                assert listed.status_code == 200
                assert listed.json() == {"items": []}
                direct = await client.get("/api/threads/legacy-thread")
                assert direct.status_code == 403
                assert direct.json() == {"detail": {"code": "thread_forbidden"}}
                events = await client.get("/api/runs/legacy-run/events")
                assert events.status_code == 403
                assert events.json() == {"detail": {"code": "run_forbidden"}}

    with pytest.raises(
        RuntimeError,
        match="ownership migration cannot downgrade non-empty product data",
    ):
        command.downgrade(config, "0001_product_core")
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            "0002_principal_ownership"
        )
        assert (
            connection.execute(
                "SELECT owner_subject FROM product_threads WHERE id = 'legacy-thread'"
            ).fetchone()[0]
            == LEGACY_UNOWNED_SUBJECT
        )


def test_v03_downgrade_refuses_to_discard_agent_policy_evidence(tmp_path: Path) -> None:
    database_path = tmp_path / "agent-policy-evidence.db"
    config = migration_config(BACKEND, database_path)
    command.upgrade(config, "head")
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO product_threads "
            "(id, owner_subject, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("thread-v03", "neutral-owner", "Evidence", now, now),
        )
        connection.execute(
            "INSERT INTO product_runs "
            "(id, thread_id, actor_subject, idempotency_key, status, last_seq, "
            "execution_plan, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "run-v03",
                "thread-v03",
                "neutral-owner",
                "evidence-key",
                "created",
                0,
                '{"schema_version":"1.0.0"}',
                now,
                None,
            ),
        )
        connection.commit()

    with pytest.raises(
        RuntimeError,
        match="Agent policy evidence migration cannot discard recorded evidence",
    ):
        command.downgrade(config, "0002_principal_ownership")

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            "0003_agent_policy_evidence"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(product_runs)")}
        assert {"execution_plan", "execution_outcome"}.issubset(columns)
        assert (
            connection.execute(
                "SELECT execution_plan FROM product_runs WHERE id = 'run-v03'"
            ).fetchone()[0]
            == '{"schema_version":"1.0.0"}'
        )


def test_v02_migration_rejects_inconsistent_child_links_before_schema_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-inconsistent.db"
    config = migration_config(BACKEND, database_path)
    command.upgrade(config, "0001_product_core")
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO product_threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("thread-a", "A", now, now),
        )
        connection.execute(
            "INSERT INTO product_threads (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("thread-b", "B", now, now),
        )
        connection.execute(
            "INSERT INTO product_runs "
            "(id, thread_id, idempotency_key, status, last_seq, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("run-a", "thread-a", "key", "completed", 0, now, now),
        )
        connection.execute(
            "INSERT INTO product_messages "
            "(id, thread_id, run_id, role, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("bad-message", "thread-b", "run-a", "user", "must be rejected", now),
        )
        connection.commit()

    with pytest.raises(
        RuntimeError,
        match="legacy product child has inconsistent run and thread",
    ):
        command.upgrade(config, "head")

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == (
            "0001_product_core"
        )
        thread_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(product_threads)")
        }
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(product_runs)")}
        assert "owner_subject" not in thread_columns
        assert "actor_subject" not in run_columns
        assert connection.execute("SELECT COUNT(*) FROM product_messages").fetchone()[0] == 1
