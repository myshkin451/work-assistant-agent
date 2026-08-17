from __future__ import annotations

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from .models import Base


class Database:
    def __init__(self, url: str, *, operation_timeout_seconds: float = 2.0) -> None:
        is_sqlite = url.startswith("sqlite+")
        timeout_ms = max(1, int(operation_timeout_seconds * 1_000))
        connect_args: dict[str, object]
        if is_sqlite:
            connect_args = {
                "check_same_thread": False,
                "timeout": operation_timeout_seconds,
            }
        else:
            lock_timeout_ms = max(1, timeout_ms - min(250, timeout_ms // 4))
            connect_args = {
                # libpq represents this timeout in whole seconds and treats
                # values below two seconds as two. Settings therefore reject a
                # smaller PostgreSQL operation budget at startup.
                "connect_timeout": max(2, int(operation_timeout_seconds)),
                "options": (f"-c statement_timeout={timeout_ms} -c lock_timeout={lock_timeout_ms}"),
            }
        engine_options: dict[str, object] = {
            "pool_pre_ping": not is_sqlite,
            "connect_args": connect_args,
        }
        if not is_sqlite:
            engine_options["pool_timeout"] = operation_timeout_seconds
        self.engine: AsyncEngine = create_async_engine(
            url,
            **engine_options,  # type: ignore[arg-type]
        )
        if is_sqlite:

            @event.listens_for(self.engine.sync_engine, "connect")
            def configure_sqlite(dbapi_connection: object, _: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute(f"PRAGMA busy_timeout={timeout_ms}")
                cursor.close()

        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def healthcheck(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def create_schema_for_tests(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()
