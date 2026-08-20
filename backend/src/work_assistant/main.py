from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .agent_definition import AgentDefinition
from .agent_runtime import AgentRunner, runtime_for_settings
from .api import router
from .bootstrap import build_policy_kernel
from .capabilities import (
    PrincipalCapabilityPolicy,
    ToolRegistry,
)
from .db import Database
from .identity import IdentityProvider, authenticate_request, resolve_identity_provider
from .repository import ProductRepository, RepositoryUnavailableError
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
        service = getattr(request.app.state, "run_service", None)
        if service is not None and not service.is_healthy:
            response = JSONResponse(
                status_code=503,
                content={"detail": {"code": "service_unavailable"}},
            )
            await response(scope, receive, send)
            return
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
    agent_definitions: Sequence[AgentDefinition] | None = None,
    tool_registry: ToolRegistry | None = None,
    capability_policy: PrincipalCapabilityPolicy | None = None,
    runner_decorator: Callable[[AgentRunner], AgentRunner] | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    policy_kernel = build_policy_kernel(
        resolved,
        agent_definitions=agent_definitions,
        tool_registry=tool_registry,
        capability_policy=capability_policy,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        provider = resolve_identity_provider(resolved, identity_provider)
        database = Database(
            resolved.database_url,
            operation_timeout_seconds=resolved.database_operation_timeout_seconds,
        )
        repository = ProductRepository(database.session_factory)
        async with runtime_for_settings(resolved, policy_kernel=policy_kernel) as runner:
            active_runner = runner_decorator(runner) if runner_decorator is not None else runner
            # The current deployment contract has one executing backend process. Any
            # active Run present before this process accepts traffic belonged to a lost
            # executor and must become an immutable, retryable product failure.
            await repository.fail_orphaned_runs()
            service = RunService(
                repository=repository,
                runner=active_runner,
                policy_kernel=policy_kernel,
                settings=resolved,
            )
            app.state.settings = resolved
            app.state.identity_provider = provider
            app.state.database = database
            app.state.repository = repository
            app.state.run_service = service
            app.state.policy_kernel = policy_kernel
            try:
                yield
            finally:
                await service.shutdown()
                await database.dispose()

    app = FastAPI(
        title="Work Assistant Agent API",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RepositoryUnavailableError)
    async def repository_unavailable(
        request: Request,
        exc: RepositoryUnavailableError,
    ) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=503,
            content={"detail": {"code": "service_unavailable"}},
        )

    app.add_middleware(PrincipalAuthenticationMiddleware)
    app.add_middleware(ProductNoStoreMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH"],
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
