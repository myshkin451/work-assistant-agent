from __future__ import annotations

import asyncio
import json
from typing import Any

from httpx import AsyncClient

from work_assistant.identity import ANONYMOUS_DEVELOPMENT_SUBJECT, Principal
from work_assistant.repository import ActiveRunConflictError

TEST_PRINCIPAL = Principal(subject=ANONYMOUS_DEVELOPMENT_SUBJECT)


def parse_sse(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in body.split("\n\n"):
        data = next((line[6:] for line in frame.splitlines() if line.startswith("data: ")), None)
        if data is not None:
            events.append(json.loads(data))
    return events


async def create_thread(client: AsyncClient, title: str = "Timezone check") -> str:
    response = await client.post("/api/threads", json={"title": title})
    assert response.status_code == 201
    return str(response.json()["thread_id"])


async def read_events(
    client: AsyncClient,
    run_id: str,
    *,
    after_seq: int | None = None,
    last_event_id: str | None = None,
) -> list[dict[str, Any]]:
    params = {} if after_seq is None else {"after_seq": after_seq}
    headers = {} if last_event_id is None else {"Last-Event-ID": last_event_id}
    response = await client.get(f"/api/runs/{run_id}/events", params=params, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


async def test_fake_vertical_persists_contract_and_replays(app_client: tuple[Any, Any]) -> None:
    client, _ = app_client
    thread_id = await create_thread(client)
    created = await client.post(
        f"/api/threads/{thread_id}/runs",
        json={
            "message": "What is the current time in Asia/Shanghai?",
            "idempotency_key": "request-1",
        },
    )
    assert created.status_code == 201
    run = created.json()
    assert run["thread_id"] == thread_id
    assert run["status"] == "created"

    events = await read_events(client, run["run_id"])
    event_types = [event["type"] for event in events]
    assert event_types == [
        "run.started",
        "tool.started",
        "tool.finished",
        "source.added",
        "message.delta",
        "message.delta",
        "message.completed",
        "run.completed",
    ]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["run_id"] == run["run_id"] for event in events)
    assert all(event["thread_id"] == thread_id for event in events)

    completed_message = events[-2]["data"]["message"]
    assert completed_message["role"] == "assistant"
    assert completed_message["run_id"] == run["run_id"]
    assert "Asia/Shanghai" in completed_message["content"]

    snapshot = (await client.get(f"/api/threads/{thread_id}")).json()
    assert snapshot["active_run"] is None
    assert [message["role"] for message in snapshot["messages"]] == ["user", "assistant"]

    replay = await read_events(client, run["run_id"], after_seq=4)
    assert [event["seq"] for event in replay] == [5, 6, 7, 8]
    header_replay = await read_events(client, run["run_id"], last_event_id="6")
    assert [event["seq"] for event in header_replay] == [7, 8]
    explicit_wins = await read_events(
        client, run["run_id"], after_seq=7, last_event_id="not-an-integer"
    )
    assert [event["seq"] for event in explicit_wins] == [8]


async def test_idempotency_and_single_active_run_hold_under_concurrency(
    app_client: tuple[Any, Any],
) -> None:
    client, app = app_client
    thread_id = await create_thread(client)
    body = {
        "message": "What time is it in Europe/London?",
        "idempotency_key": "same-request",
    }
    responses = await asyncio.gather(
        *(client.post(f"/api/threads/{thread_id}/runs", json=body) for _ in range(20))
    )
    assert {response.status_code for response in responses} == {201}
    run_ids = {response.json()["run_id"] for response in responses}
    assert len(run_ids) == 1

    conflict = await client.post(
        f"/api/threads/{thread_id}/runs",
        json={"message": "Try again", "idempotency_key": "different-request"},
    )
    assert conflict.status_code == 409
    await client.post(f"/api/runs/{run_ids.pop()}/cancel")

    second_thread = await create_thread(client, "Concurrent conflict")
    # Keep the winning Run in `created` while every contender is admitted. The API
    # launches successful Runs immediately, so using it here made the assertion
    # depend on whether the fake Run completed before slower CI requests arrived.
    # The API's 409 mapping is covered above; this phase isolates the repository's
    # atomic one-active-Run invariant under genuinely overlapping creation calls.
    contenders = await asyncio.gather(
        *(
            app.state.repository.create_run(
                principal=TEST_PRINCIPAL,
                thread_id=second_thread,
                message="Current time in UTC",
                idempotency_key=f"key-{index}",
            )
            for index in range(10)
        ),
        return_exceptions=True,
    )
    created = [result for result in contenders if isinstance(result, tuple)]
    conflicts = [result for result in contenders if isinstance(result, ActiveRunConflictError)]
    assert len(created) == 1
    assert created[0][1] is True
    assert len(conflicts) == 9
    await app.state.repository.cancel_run(created[0][0].run_id, principal=TEST_PRINCIPAL)


async def test_cancel_is_terminal_and_late_agent_results_are_discarded(
    app_client: tuple[Any, Any],
) -> None:
    client, app = app_client
    thread_id = await create_thread(client)
    response = await client.post(
        f"/api/threads/{thread_id}/runs",
        json={"message": "Current time in America/New_York", "idempotency_key": "cancel-me"},
    )
    run_id = response.json()["run_id"]
    await asyncio.sleep(0.02)
    cancelled = await client.post(f"/api/runs/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    await asyncio.sleep(0.25)
    await app.state.run_service.wait_for_idle()

    events = await read_events(client, run_id)
    types = [event["type"] for event in events]
    assert types[-1] == "run.cancelled"
    assert "message.completed" not in types
    assert "run.completed" not in types
    assert "run.failed" not in types

    second_cancel = await client.post(f"/api/runs/{run_id}/cancel")
    assert second_cancel.json()["status"] == "cancelled"
    assert len(await read_events(client, run_id)) == len(events)
    snapshot = (await client.get(f"/api/threads/{thread_id}")).json()
    assert [message["role"] for message in snapshot["messages"]] == ["user"]


async def test_cancel_and_event_append_reserve_distinct_sequences(
    app_client: tuple[Any, Any],
) -> None:
    _, app = app_client
    repository = app.state.repository
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title="Atomic event sequence"
    )
    run, created = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Current time UTC",
        idempotency_key="cancel-event-race",
    )
    assert created is True
    assert await repository.start_run(run.run_id) is True

    append_result, cancelled = await asyncio.gather(
        repository.append_active_event(
            run.run_id,
            "tool.started",
            {
                "tool_call_id": "race-1",
                "name": "get_current_time",
                "label": "Read current time",
            },
        ),
        repository.cancel_run(run.run_id, principal=TEST_PRINCIPAL),
    )

    assert cancelled.status == "cancelled"
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert events[-1].type == "run.cancelled"
    if append_result is not None:
        assert append_result.seq < events[-1].seq


async def test_errors_and_public_events_are_bounded(app_client: tuple[Any, Any]) -> None:
    client, _ = app_client
    missing = await client.get("/api/threads/not-found")
    assert missing.status_code == 404
    assert missing.json() == {"detail": {"code": "thread_not_found"}}

    invalid = await client.post(
        "/api/threads/not-found/runs",
        json={"message": " ", "idempotency_key": " "},
    )
    assert invalid.status_code == 404
    assert invalid.json() == {"detail": {"code": "thread_not_found"}}
    assert "traceback" not in invalid.text.casefold()

    thread_id = await create_thread(client)
    run = (
        await client.post(
            f"/api/threads/{thread_id}/runs",
            json={"message": "Current time UTC", "idempotency_key": "public-check"},
        )
    ).json()
    events = await read_events(client, run["run_id"])
    serialized = json.dumps(events).casefold()
    for forbidden in (
        "api_key",
        "checkpoint",
        "reasoning_content",
        "system_prompt",
        "provider_response",
    ):
        assert forbidden not in serialized


async def test_invalid_last_event_id_is_safe_422(app_client: tuple[Any, Any]) -> None:
    client, _ = app_client
    thread_id = await create_thread(client)
    run = (
        await client.post(
            f"/api/threads/{thread_id}/runs",
            json={"message": "Current time UTC", "idempotency_key": "bad-header"},
        )
    ).json()
    response = await client.get(
        f"/api/runs/{run['run_id']}/events", headers={"Last-Event-ID": "bad"}
    )
    assert response.status_code == 422
    await client.post(f"/api/runs/{run['run_id']}/cancel")
