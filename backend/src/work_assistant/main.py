from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .agent_runtime import runtime_for_settings
from .api import router
from .db import Database
from .identity import IdentityProvider, authenticate_request, resolve_identity_provider
from .repository import ProductRepository
from .service import RunService
from .settings import Settings, get_settings


class ProductNoStoreMiddleware:
    """Prevent a shared browser cache from retaining subject-scoped resources."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        async def send_no_store(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if headers.get("content-type", "").startswith("text/event-stream"):
                    headers["Cache-Control"] = "private, no-store, no-transform"
                else:
                    headers["Cache-Control"] = "private, no-store"
                headers["Pragma"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_no_store)


class PrincipalAuthenticationMiddleware:
    """Authenticate product requests before FastAPI reads or validates a body."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not scope.get("path", "").startswith("/api/")
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        provider = request.app.state.identity_provider
        principal = await authenticate_request(provider, request)
        if principal is None:
            response = JSONResponse(
                status_code=401,
                content={"detail": {"code": "authentication_required"}},
            )
            await response(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["principal"] = principal
        await self.app(scope, receive, send)


def create_app(
    settings: Settings | None = None,
    *,
    identity_provider: IdentityProvider | None = None,
) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        provider = resolve_identity_provider(resolved, identity_provider)
        database = Database(resolved.database_url)
        repository = ProductRepository(database.session_factory)
        # The current deployment contract has one executing backend process. Any
        # active Run present before this process accepts traffic belonged to a lost
        # executor and must become an immutable, retryable product failure.
        await repository.fail_orphaned_runs()
        async with runtime_for_settings(resolved) as runner:
            service = RunService(repository=repository, runner=runner, settings=resolved)
            app.state.settings = resolved
            app.state.identity_provider = provider
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
    app.add_middleware(PrincipalAuthenticationMiddleware)
    app.add_middleware(ProductNoStoreMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Last-Event-ID",
            "X-Work-Assistant-Dev-Subject",
        ],
    )
    app.include_router(router)
    return app


app = create_app()
