from __future__ import annotations

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from .models import Base


class Database:
    def __init__(self, url: str) -> None:
        is_sqlite = url.startswith("sqlite+")
        connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
        self.engine: AsyncEngine = create_async_engine(
            url,
            pool_pre_ping=not is_sqlite,
            connect_args=connect_args,
        )
        if is_sqlite:

            @event.listens_for(self.engine.sync_engine, "connect")
            def configure_sqlite(dbapi_connection: object, _: object) -> None:
                cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
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
