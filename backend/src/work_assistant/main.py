from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent_runtime import runtime_for_settings
from .api import router
from .db import Database
from .repository import ProductRepository
from .service import RunService
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(resolved.database_url)
        repository = ProductRepository(database.session_factory)
        # The current deployment contract has one executing backend process. Any
        # active Run present before this process accepts traffic belonged to a lost
        # executor and must become an immutable, retryable product failure.
        await repository.fail_orphaned_runs()
        async with runtime_for_settings(resolved) as runner:
            service = RunService(repository=repository, runner=runner, settings=resolved)
            app.state.settings = resolved
            app.state.database = database
            app.state.repository = repository
            app.state.run_service = service
            try:
                yield
            finally:
                await service.shutdown()
                await database.dispose()

    app = FastAPI(
        title="Work Assistant Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )
    app.include_router(router)
    return app


app = create_app()
