from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from policy_fixtures import make_execution_plan, terminal_outcome
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from work_assistant.authorization import ResourceForbiddenError
from work_assistant.db import Database
from work_assistant.identity import (
    DEVELOPMENT_PRINCIPAL_HEADER,
    LEGACY_UNOWNED_SUBJECT,
    AnonymousIdentityProvider,
    DevelopmentHeaderIdentityProvider,
    IdentityConfigurationError,
    IdentityProvider,
    Principal,
)
from work_assistant.main import create_app
from work_assistant.models import EventRecord, MessageRecord, RunRecord, ThreadRecord, utc_now
from work_assistant.repository import ResourceNotFoundError
from work_assistant.settings import Settings

PRINCIPAL_A = Principal(subject="neutral-principal-a", display_name="Neutral A")
PRINCIPAL_B = Principal(subject="neutral-principal-b", display_name="Neutral B")


@pytest.fixture
async def identity_clients(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, AsyncClient, AsyncClient, Any]]:
    database_path = tmp_path / "identity.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    settings = Settings(
        app_env="test",
        identity_provider_mode="development_header",
        database_url=database_url,
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
        model_mode="fake",
        fake_step_delay_seconds=0.15,
        sse_poll_interval_seconds=0.005,
        sse_keepalive_seconds=0.02,
        run_timeout_seconds=5,
    )
    database = Database(database_url)
    await database.create_schema_for_tests()
    await database.dispose()
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with (
            AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={DEVELOPMENT_PRINCIPAL_HEADER: PRINCIPAL_A.subject},
            ) as principal_a,
            AsyncClient(
                transport=transport,
                base_url="http://test",
                headers={DEVELOPMENT_PRINCIPAL_HEADER: PRINCIPAL_B.subject},
            ) as principal_b,
            AsyncClient(transport=transport, base_url="http://test") as unauthenticated,
        ):
            yield principal_a, principal_b, unauthenticated, app


async def create_thread(client: AsyncClient, title: str) -> str:
    response = await client.post("/api/threads", json={"title": title})
    assert response.status_code == 201
    return str(response.json()["thread_id"])


def parse_sse(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in body.split("\n\n"):
        data = next((line[6:] for line in frame.splitlines() if line.startswith("data: ")), None)
        if data is not None:
            events.append(json.loads(data))
    return events


def test_identity_contract_rejects_reserved_and_invalid_subjects() -> None:
    with pytest.raises(ValidationError, match="reserved internal namespace"):
        Principal(subject=LEGACY_UNOWNED_SUBJECT)
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        Principal(subject=" neutral-a")
    with pytest.raises(ValidationError, match="control characters"):
        Principal(subject="neutral\nprincipal")


@pytest.mark.parametrize("mode", ["anonymous", "development_header"])
def test_production_rejects_development_identity_modes(mode: str) -> None:
    with pytest.raises(ValidationError, match="production requires an external"):
        Settings(
            app_env="production",
            identity_provider_mode=mode,  # type: ignore[arg-type]
            model_mode="deepseek",
            deepseek_api_key="test-only-key",
            allowed_origins="https://neutral.example",
        )


@pytest.mark.parametrize(
    "provider",
    [AnonymousIdentityProvider(), DevelopmentHeaderIdentityProvider()],
)
async def test_production_rejects_injected_development_providers(
    tmp_path: Path,
    provider: IdentityProvider,
) -> None:
    settings = Settings(
        app_env="production",
        identity_provider_mode="external",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'must-not-open.db'}",
        allowed_origins="https://neutral.example",
        model_mode="deepseek",
        deepseek_api_key="test-only-key",
    )
    app = create_app(settings, identity_provider=provider)
    with pytest.raises(
        IdentityConfigurationError,
        match="development_identity_provider_in_production",
    ):
        async with app.router.lifespan_context(app):
            pass
    assert not (tmp_path / "must-not-open.db").exists()


@pytest.mark.parametrize(
    "origin",
    ["*", "null", "https://example.com/path", "https://user@example.com"],
)
def test_credentialed_cors_rejects_non_origin_values(origin: str) -> None:
    with pytest.raises(ValidationError, match="allowed origin|exact origins"):
        Settings(allowed_origins=origin)


def test_production_rejects_loopback_allowed_origins() -> None:
    with pytest.raises(ValidationError, match="non-loopback allowed origins"):
        Settings(
            app_env="production",
            identity_provider_mode="external",
            model_mode="deepseek",
            deepseek_api_key="test-only-key",
        )


async def test_missing_external_provider_fails_before_database_work(tmp_path: Path) -> None:
    settings = Settings(
        app_env="production",
        identity_provider_mode="external",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'must-not-open.db'}",
        allowed_origins="https://neutral.example",
        model_mode="deepseek",
        deepseek_api_key="test-only-key",
    )
    app = create_app(settings)
    with pytest.raises(IdentityConfigurationError, match="external_identity_provider_missing"):
        async with app.router.lifespan_context(app):
            pass
    assert not (tmp_path / "must-not-open.db").exists()


async def test_external_provider_output_is_revalidated_at_host_boundary(
    tmp_path: Path,
) -> None:
    class MutableProvider:
        result: object = PRINCIPAL_A

        async def authenticate(self, request: object) -> object:
            del request
            return self.result

    database_path = tmp_path / "external-provider.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    database = Database(database_url)
    await database.create_schema_for_tests()
    await database.dispose()
    provider = MutableProvider()
    settings = Settings(
        app_env="test",
        identity_provider_mode="external",
        database_url=database_url,
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
    )
    app = create_app(settings, identity_provider=provider)  # type: ignore[arg-type]
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/threads")).status_code == 200
            provider.result = {"subject": PRINCIPAL_A.subject}
            invalid_type = await client.get("/api/threads")
            assert invalid_type.status_code == 401
            provider.result = Principal.model_construct(subject=LEGACY_UNOWNED_SUBJECT)
            reserved = await client.post("/api/threads", json={})
            assert reserved.status_code == 401
            assert reserved.json() == {"detail": {"code": "authentication_required"}}


async def test_health_is_public_but_every_product_entry_requires_identity(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    _, _, client, _ = identity_clients
    assert (await client.get("/health")).status_code == 200

    requests = [
        client.get("/api/threads"),
        client.post("/api/threads", json={}),
        client.post(
            "/api/threads",
            content="{",
            headers={"Content-Type": "application/json"},
        ),
        client.get("/api/threads/unknown"),
        client.patch("/api/threads/unknown", json={"title": "Rename"}),
        client.post(
            f"/api/threads/{uuid4()}/initial-run",
            json={"message": "question", "idempotency_key": "initial-key"},
        ),
        client.post(
            "/api/threads/unknown/runs",
            json={"message": "question", "idempotency_key": "key"},
        ),
        client.post("/api/runs/unknown/cancel"),
        client.get("/api/runs/unknown/events"),
        # Authentication takes precedence over request validation.
        client.post("/api/threads/unknown/runs", json={"message": " ", "idempotency_key": " "}),
        client.post(
            "/api/threads/unknown/runs",
            content="{",
            headers={"Content-Type": "application/json"},
        ),
    ]
    for response in await asyncio.gather(*requests):
        assert response.status_code == 401
        assert response.json() == {"detail": {"code": "authentication_required"}}

    duplicate = await client.get(
        "/api/threads",
        headers=[
            (DEVELOPMENT_PRINCIPAL_HEADER, "neutral-a"),
            (DEVELOPMENT_PRINCIPAL_HEADER, "neutral-b"),
        ],
    )
    assert duplicate.status_code == 401
    reserved = await client.get(
        "/api/threads", headers={DEVELOPMENT_PRINCIPAL_HEADER: LEGACY_UNOWNED_SUBJECT}
    )
    assert reserved.status_code == 401


async def test_thread_listing_detail_and_creation_are_owner_scoped(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, _, app = identity_clients
    thread_a = await create_thread(client_a, "A private thread")
    thread_b = await create_thread(client_b, "B private thread")

    list_a = (await client_a.get("/api/threads")).json()["items"]
    list_b = (await client_b.get("/api/threads")).json()["items"]
    assert [item["thread_id"] for item in list_a] == [thread_a]
    assert [item["thread_id"] for item in list_b] == [thread_b]
    assert (await client_a.get(f"/api/threads/{thread_a}")).status_code == 200

    forbidden = await client_b.get(f"/api/threads/{thread_a}")
    assert forbidden.status_code == 403
    assert forbidden.json() == {"detail": {"code": "thread_forbidden"}}
    assert "private" not in forbidden.text.casefold()
    missing = await client_b.get("/api/threads/not-a-thread")
    assert missing.status_code == 404
    assert missing.json() == {"detail": {"code": "thread_not_found"}}

    async with app.state.repository._sessions() as session:  # noqa: SLF001
        stored_a = await session.get(ThreadRecord, thread_a)
        stored_b = await session.get(ThreadRecord, thread_b)
    assert stored_a is not None and stored_a.owner_subject == PRINCIPAL_A.subject
    assert stored_b is not None and stored_b.owner_subject == PRINCIPAL_B.subject


async def test_initial_run_conflicts_and_foreign_replays_fail_closed(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, _, app = identity_clients
    thread_id = str(uuid4())
    body = {"message": "Owner A first question", "idempotency_key": "initial-owner-a"}

    created = await client_a.post(f"/api/threads/{thread_id}/initial-run", json=body)
    replayed = await client_a.post(f"/api/threads/{thread_id}/initial-run", json=body)
    assert created.status_code == replayed.status_code == 201
    assert created.json()["run"]["run_id"] == replayed.json()["run"]["run_id"]

    mismatch = await client_a.post(
        f"/api/threads/{thread_id}/initial-run",
        json={**body, "message": "A different first question"},
    )
    assert mismatch.status_code == 409
    assert mismatch.json() == {"detail": {"code": "idempotency_mismatch"}}

    different_key = await client_a.post(
        f"/api/threads/{thread_id}/initial-run",
        json={**body, "idempotency_key": "another-initial-key"},
    )
    assert different_key.status_code == 409
    assert different_key.json() == {"detail": {"code": "thread_already_exists"}}

    rejected_origin_thread_id = str(uuid4())
    rejected_origin = await client_a.post(
        f"/api/threads/{rejected_origin_thread_id}/initial-run",
        json={"message": "Must not persist", "idempotency_key": "bad-origin"},
        headers={"Origin": "https://untrusted.example"},
    )
    assert rejected_origin.status_code == 403
    assert rejected_origin.json() == {"detail": {"code": "origin_forbidden"}}
    assert (await client_a.get(f"/api/threads/{rejected_origin_thread_id}")).status_code == 404

    forbidden = await client_b.post(f"/api/threads/{thread_id}/initial-run", json=body)
    forbidden_malformed = await client_b.post(
        f"/api/threads/{thread_id}/initial-run",
        json={"message": " ", "idempotency_key": " "},
    )
    assert forbidden.status_code == forbidden_malformed.status_code == 403
    assert (
        forbidden.json() == forbidden_malformed.json() == {"detail": {"code": "thread_forbidden"}}
    )
    forbidden_text = forbidden.text + forbidden_malformed.text
    assert created.json()["run"]["run_id"] not in forbidden_text
    assert "Owner A" not in forbidden_text

    async with app.state.repository._sessions() as session:  # noqa: SLF001
        threads = (await session.scalars(select(ThreadRecord))).all()
        runs = (await session.scalars(select(RunRecord))).all()
        messages = (
            await session.scalars(select(MessageRecord).where(MessageRecord.role == "user"))
        ).all()
    assert len(threads) == len(runs) == len(messages) == 1
    assert threads[0].owner_subject == runs[0].actor_subject == PRINCIPAL_A.subject


async def test_thread_rename_is_strict_owner_scoped_and_cors_allowed(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, unauthenticated, _ = identity_clients
    thread_id = await create_thread(client_a, "Original title")

    renamed = await client_a.patch(
        f"/api/threads/{thread_id}",
        json={"title": "  Renamed   conversation  "},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed conversation"
    normalized_long_input = await client_a.patch(
        f"/api/threads/{thread_id}",
        json={"title": f"Concise{' ' * 220}title"},
    )
    assert normalized_long_input.status_code == 200
    assert normalized_long_input.json()["title"] == "Concise title"
    assert (await client_a.get(f"/api/threads/{thread_id}")).json()["title"] == ("Concise title")
    assert (await client_a.get("/api/threads")).json()["items"][0]["title"] == ("Concise title")

    blank = await client_a.patch(f"/api/threads/{thread_id}", json={"title": " "})
    control = await client_a.patch(
        f"/api/threads/{thread_id}", json={"title": "line one\nline two"}
    )
    extra = await client_a.patch(
        f"/api/threads/{thread_id}", json={"title": "Valid", "owner": "injected"}
    )
    assert blank.status_code == control.status_code == extra.status_code == 422

    forbidden = await client_b.patch(f"/api/threads/{thread_id}", json={"title": "Foreign rename"})
    forbidden_malformed = await client_b.patch(f"/api/threads/{thread_id}", json={"title": " "})
    assert forbidden.status_code == forbidden_malformed.status_code == 403
    assert (
        forbidden.json() == forbidden_malformed.json() == {"detail": {"code": "thread_forbidden"}}
    )
    assert "Renamed conversation" not in forbidden.text + forbidden_malformed.text

    missing = await client_a.patch(
        f"/api/threads/{uuid4()}", json={"title": "Missing conversation"}
    )
    assert missing.status_code == 404
    unauthenticated_rename = await unauthenticated.patch(
        f"/api/threads/{thread_id}", json={"title": " "}
    )
    assert unauthenticated_rename.status_code == 401

    bad_origin = await client_a.patch(
        f"/api/threads/{thread_id}",
        json={"title": "Origin must not mutate"},
        headers={"Origin": "https://untrusted.example"},
    )
    assert bad_origin.status_code == 403
    assert bad_origin.json() == {"detail": {"code": "origin_forbidden"}}
    assert (await client_a.get(f"/api/threads/{thread_id}")).json()["title"] == ("Concise title")

    preflight = await client_a.options(
        f"/api/threads/{thread_id}",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PATCH",
        },
    )
    assert preflight.status_code == 200
    assert "PATCH" in preflight.headers["access-control-allow-methods"]


async def test_run_idempotency_actor_and_cross_owner_rejection(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, _, app = identity_clients
    thread_a = await create_thread(client_a, "A run thread")
    thread_b = await create_thread(client_b, "B run thread")
    body = {"message": "Current time UTC", "idempotency_key": "shared-key"}

    first_a = await client_a.post(f"/api/threads/{thread_a}/runs", json=body)
    replay_a = await client_a.post(f"/api/threads/{thread_a}/runs", json=body)
    first_b = await client_b.post(f"/api/threads/{thread_b}/runs", json=body)
    assert first_a.status_code == replay_a.status_code == first_b.status_code == 201
    assert first_a.json()["run_id"] == replay_a.json()["run_id"]
    assert first_a.json()["run_id"] != first_b.json()["run_id"]

    forbidden = await client_b.post(f"/api/threads/{thread_a}/runs", json=body)
    assert forbidden.status_code == 403
    assert forbidden.json() == {"detail": {"code": "thread_forbidden"}}
    assert first_a.json()["run_id"] not in forbidden.text

    forbidden_malformed = await client_b.post(
        f"/api/threads/{thread_a}/runs",
        json={"message": " ", "idempotency_key": " "},
    )
    assert forbidden_malformed.status_code == 403

    privileged_b = Principal(
        subject=PRINCIPAL_B.subject,
        display_name="Neutral B",
        roles=("admin", "allow-all-tools"),
    )
    allowed_plan_for_b = make_execution_plan(privileged_b)
    assert tuple(tool.tool_id for tool in allowed_plan_for_b.visible_tools) == ("get_current_time",)
    with pytest.raises(ResourceForbiddenError, match="thread"):
        await app.state.repository.create_run(
            principal=privileged_b,
            thread_id=thread_a,
            message="Policy cannot grant ownership",
            idempotency_key="policy-owner-bypass",
            execution_plan=allowed_plan_for_b,
        )

    owner_plan, _ = await app.state.repository.get_run_evidence(
        first_a.json()["run_id"], principal=PRINCIPAL_A
    )
    assert owner_plan is not None
    assert [tool["tool_id"] for tool in owner_plan["visible_tools"]] == ["get_current_time"]
    with pytest.raises(ResourceForbiddenError, match="run"):
        await app.state.repository.get_run_evidence(
            first_a.json()["run_id"], principal=privileged_b
        )

    missing = await client_a.post("/api/threads/unknown/runs", json=body)
    assert missing.status_code == 404
    missing_malformed = await client_a.post(
        "/api/threads/unknown/runs",
        json={"message": " ", "idempotency_key": " "},
    )
    assert missing_malformed.status_code == 404

    async with app.state.repository._sessions() as session:  # noqa: SLF001
        run_a = await session.get(RunRecord, first_a.json()["run_id"])
        run_b = await session.get(RunRecord, first_b.json()["run_id"])
        run_count = len((await session.scalars(select(RunRecord))).all())
    assert run_a is not None and run_a.actor_subject == PRINCIPAL_A.subject
    assert run_b is not None and run_b.actor_subject == PRINCIPAL_B.subject
    assert run_count == 2

    await client_a.post(f"/api/runs/{first_a.json()['run_id']}/cancel")
    await client_b.post(f"/api/runs/{first_b.json()['run_id']}/cancel")


async def test_cancel_and_retry_equivalent_post_cannot_cross_owner(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, _, app = identity_clients
    repository = app.state.repository
    thread = await repository.create_thread(principal=PRINCIPAL_A, title="Controlled active run")
    run, created = await repository.create_run(
        principal=PRINCIPAL_A,
        thread_id=thread.thread_id,
        message="Original question",
        idempotency_key="original",
        execution_plan=make_execution_plan(PRINCIPAL_A),
    )
    assert created is True

    forged_origin = await client_a.post(
        f"/api/runs/{run.run_id}/cancel",
        headers={"Origin": "https://evil.example"},
    )
    assert forged_origin.status_code == 403
    assert forged_origin.json() == {"detail": {"code": "origin_forbidden"}}
    after_forged_origin = await repository.get_run(run.run_id, principal=PRINCIPAL_A)
    assert after_forged_origin.status == "created"
    assert after_forged_origin.last_seq == 0

    forbidden_cancel = await client_b.post(f"/api/runs/{run.run_id}/cancel")
    assert forbidden_cancel.status_code == 403
    assert forbidden_cancel.json() == {"detail": {"code": "run_forbidden"}}
    unchanged = await repository.get_run(run.run_id, principal=PRINCIPAL_A)
    assert unchanged.status == "created" and unchanged.last_seq == 0
    assert await repository.get_events(run.run_id, 0, principal=PRINCIPAL_A) == []

    forbidden_retry = await client_b.post(
        f"/api/threads/{thread.thread_id}/runs",
        json={"message": "Retry another user's run", "idempotency_key": "retry"},
    )
    assert forbidden_retry.status_code == 403
    assert (await repository.get_run(run.run_id, principal=PRINCIPAL_A)).status == "created"

    cancelled = await client_a.post(
        f"/api/runs/{run.run_id}/cancel",
        headers={"Origin": "http://localhost:5173"},
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    owner_retry = await client_a.post(
        f"/api/threads/{thread.thread_id}/runs",
        json={"message": "Owner retry", "idempotency_key": "retry"},
    )
    assert owner_retry.status_code == 201
    assert owner_retry.json()["run_id"] != run.run_id
    await client_a.post(f"/api/runs/{owner_retry.json()['run_id']}/cancel")


async def test_sse_replay_and_reconnect_reauthorize_before_streaming(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, unauthenticated, _ = identity_clients
    thread_a = await create_thread(client_a, "A SSE thread")
    created = await client_a.post(
        f"/api/threads/{thread_a}/runs",
        json={"message": "Current time UTC", "idempotency_key": "sse-a"},
    )
    run_id = created.json()["run_id"]

    full = await client_a.get(f"/api/runs/{run_id}/events")
    assert full.status_code == 200
    events = parse_sse(full.text)
    assert events[-1]["type"] == "run.completed"
    cursor = events[-3]["seq"]
    replay = await client_a.get(
        f"/api/runs/{run_id}/events",
        params={"after_seq": cursor},
        headers={"Last-Event-ID": "not-an-integer"},
    )
    assert [event["seq"] for event in parse_sse(replay.text)] == [
        event["seq"] for event in events if event["seq"] > cursor
    ]

    forbidden = await client_b.get(
        f"/api/runs/{run_id}/events",
        params={"after_seq": 1},
    )
    assert forbidden.status_code == 403
    assert forbidden.headers["content-type"].startswith("application/json")
    assert forbidden.json() == {"detail": {"code": "run_forbidden"}}
    assert "data:" not in forbidden.text
    forbidden_bad_cursor = await client_b.get(
        f"/api/runs/{run_id}/events", params={"after_seq": -1}
    )
    assert forbidden_bad_cursor.status_code == 403

    unauthenticated_response = await unauthenticated.get(f"/api/runs/{run_id}/events")
    assert unauthenticated_response.status_code == 401
    missing = await client_a.get("/api/runs/not-a-run/events")
    assert missing.status_code == 404
    missing_bad_cursor = await client_a.get("/api/runs/not-a-run/events", params={"after_seq": -1})
    assert missing_bad_cursor.status_code == 404
    owned_bad_cursor = await client_a.get(f"/api/runs/{run_id}/events", params={"after_seq": -1})
    assert owned_bad_cursor.status_code == 422
    assert owned_bad_cursor.json() == {"detail": {"code": "invalid_after_seq"}}
    assert full.headers["cache-control"] == "private, no-store, no-transform"

    snapshot = await client_a.get(f"/api/threads/{thread_a}")
    assert snapshot.headers["cache-control"] == "private, no-store"
    assert snapshot.headers["pragma"] == "no-cache"


async def test_corrupt_actor_and_cross_thread_children_fail_closed(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, client_b, _, app = identity_clients
    repository = app.state.repository
    thread_a = await repository.create_thread(principal=PRINCIPAL_A, title="Owner A")
    thread_b = await repository.create_thread(principal=PRINCIPAL_A, title="Owner A second")
    run, _ = await repository.create_run(
        principal=PRINCIPAL_A,
        thread_id=thread_a.thread_id,
        message="Keep ownership coherent",
        idempotency_key="coherent",
        execution_plan=make_execution_plan(PRINCIPAL_A),
    )

    now = utc_now()
    async with repository._sessions() as session:  # noqa: SLF001
        session.add(
            MessageRecord(
                id="cross-thread-message",
                thread_id=thread_b.thread_id,
                run_id=run.run_id,
                role="assistant",
                content="must not persist",
                created_at=now,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
        session.add(
            EventRecord(
                id="cross-thread-event",
                run_id=run.run_id,
                thread_id=thread_b.thread_id,
                seq=1,
                type="run.started",
                occurred_at=now,
                payload={"status": "running"},
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()
        stored = await session.get(RunRecord, run.run_id)
        assert stored is not None
        stored.actor_subject = PRINCIPAL_B.subject
        await session.commit()

    assert (await client_a.get(f"/api/threads/{thread_a.thread_id}")).status_code == 403
    assert (await client_b.get(f"/api/threads/{thread_a.thread_id}")).status_code == 403
    assert (await client_a.get(f"/api/runs/{run.run_id}/events")).status_code == 403
    assert (await client_b.get(f"/api/runs/{run.run_id}/events")).status_code == 403


async def test_internal_execution_never_reads_messages_from_an_inconsistent_actor(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    _, _, _, app = identity_clients
    repository = app.state.repository
    thread = await repository.create_thread(principal=PRINCIPAL_A, title="Context owner A")
    old_run, _ = await repository.create_run(
        principal=PRINCIPAL_A,
        thread_id=thread.thread_id,
        message="Old private question",
        idempotency_key="old-context",
        execution_plan=make_execution_plan(PRINCIPAL_A),
    )
    assert await repository.start_run(old_run.run_id) is True
    await repository.complete_run(
        old_run.run_id,
        "Old private answer",
        execution_outcome=terminal_outcome(PRINCIPAL_A, status="completed"),
    )
    current_run, _ = await repository.create_run(
        principal=PRINCIPAL_A,
        thread_id=thread.thread_id,
        message="New question",
        idempotency_key="new-context",
        execution_plan=make_execution_plan(PRINCIPAL_A),
    )

    async with repository._sessions() as session:  # noqa: SLF001
        stored_old = await session.get(RunRecord, old_run.run_id)
        assert stored_old is not None
        stored_old.actor_subject = PRINCIPAL_B.subject
        await session.commit()

    assert await repository.start_run(current_run.run_id) is True
    with pytest.raises(ResourceNotFoundError, match="run_ownership"):
        await repository.get_run_context(current_run.run_id)
    current = await repository.get_run(current_run.run_id, principal=PRINCIPAL_A)
    assert current.status == "running"
    assert current.last_seq == 1


async def test_development_identity_header_cors_is_explicit_and_credentialed(
    identity_clients: tuple[AsyncClient, AsyncClient, AsyncClient, Any],
) -> None:
    client_a, _, _, _ = identity_clients
    response = await client_a.options(
        "/api/threads",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": DEVELOPMENT_PRINCIPAL_HEADER,
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-credentials"] == "true"
    assert (
        DEVELOPMENT_PRINCIPAL_HEADER.casefold()
        in response.headers["access-control-allow-headers"].casefold()
    )
