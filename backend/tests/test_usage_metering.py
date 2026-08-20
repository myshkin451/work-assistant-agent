from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from work_assistant.agent_runtime import AgentResult, ProductEvent, _auditable_provider_usage
from work_assistant.context_builder import BuiltContext
from work_assistant.db import Database
from work_assistant.execution_policy import RunExecution
from work_assistant.identity import DEVELOPMENT_PRINCIPAL_HEADER
from work_assistant.main import create_app
from work_assistant.models import ModelAttemptRecord
from work_assistant.schemas import Message
from work_assistant.settings import Settings
from work_assistant.usage import (
    ProviderTokenUsage,
    RunUsageLedger,
    UsageMeteringError,
    UsageMetric,
    aggregate_usage_metrics,
)


async def _wait_for_run(client: AsyncClient, run_id: str) -> None:
    response = await client.get(f"/api/runs/{run_id}/events")
    assert response.status_code == 200


def _parse_sse(body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in body.split("\n\n"):
        data = next(
            (line[6:] for line in frame.splitlines() if line.startswith("data: ")),
            None,
        )
        if data is not None:
            events.append(json.loads(data))
    return events


async def test_attempt_ledger_aggregates_exact_usage_once_and_keeps_missing_fields_null() -> None:
    now = 0.0

    def clock() -> float:
        return now

    ledger = RunUsageLedger(clock=clock)
    direct = await ledger.begin_attempt(call_kind="direct")
    usage = ProviderTokenUsage(
        input_tokens=10,
        output_tokens=4,
        cached_tokens=0,
        total_tokens=14,
    )
    ledger.observe_usage(direct, usage)
    ledger.observe_usage(direct, usage)
    now = 0.25
    await ledger.finish_attempt(direct, status="succeeded")
    now = 0.3
    ledger.record_first_visible()
    now = 0.5

    evidence = ledger.snapshot(error_category=None)
    assert evidence.model_call_count == 1
    assert evidence.retry_count == 0
    assert evidence.input_tokens.model_dump() == {"value": 10, "availability": "complete"}
    assert evidence.output_tokens.model_dump() == {"value": 4, "availability": "complete"}
    assert evidence.cached_tokens.model_dump() == {"value": 0, "availability": "complete"}
    assert evidence.reasoning_tokens.model_dump() == {
        "value": None,
        "availability": "unavailable",
    }
    assert evidence.total_tokens.model_dump() == {"value": 14, "availability": "complete"}
    assert evidence.generation_duration_ms == 250
    assert evidence.time_to_first_visible_ms == 300
    assert evidence.run_duration_ms == 500

    with pytest.raises(UsageMeteringError, match="provider_attempt_not_active"):
        ledger.observe_usage(direct, usage)


async def test_retry_is_a_second_attempt_and_unknown_first_usage_makes_totals_partial() -> None:
    ledger = RunUsageLedger()
    first = await ledger.begin_attempt(call_kind="decision")
    await ledger.finish_attempt(first, status="failed")
    retry = await ledger.begin_attempt(call_kind="decision", retry_of=first)
    ledger.observe_usage(
        retry,
        ProviderTokenUsage(
            input_tokens=295,
            output_tokens=38,
            cached_tokens=0,
            total_tokens=333,
        ),
    )
    await ledger.finish_attempt(retry, status="succeeded")

    evidence = ledger.snapshot(error_category=None)
    assert evidence.model_call_count == 2
    assert evidence.retry_count == 1
    for metric in (
        evidence.input_tokens,
        evidence.output_tokens,
        evidence.cached_tokens,
        evidence.total_tokens,
    ):
        assert metric.value is None
        assert metric.availability == "partial"
    assert evidence.reasoning_tokens.value is None
    assert evidence.reasoning_tokens.availability == "unavailable"


async def test_failed_attempt_finish_can_retry_the_same_durable_write() -> None:
    ledger = RunUsageLedger()
    finish_calls = 0

    async def persist_start(_: Any) -> None:
        return None

    async def persist_usage(_: Any) -> None:
        return None

    async def persist_finish(_: Any) -> None:
        nonlocal finish_calls
        finish_calls += 1
        if finish_calls == 1:
            raise RuntimeError("one durable write failure")

    ledger.bind_persistence(
        start_writer=persist_start,
        finish_writer=persist_finish,
        usage_writer=persist_usage,
    )
    attempt_id = await ledger.begin_attempt(call_kind="direct")
    ledger.observe_usage(
        attempt_id,
        ProviderTokenUsage(input_tokens=7, output_tokens=3, total_tokens=10),
    )
    with pytest.raises(RuntimeError, match="one durable write failure"):
        await ledger.finish_attempt(attempt_id, status="succeeded")
    await ledger.finish_attempt(attempt_id, status="succeeded")

    assert finish_calls == 2
    assert ledger.snapshot(error_category=None).generation_duration_ms is not None


def test_legacy_unknown_dominates_active_pending_in_account_aggregation() -> None:
    metric = aggregate_usage_metrics(
        (
            UsageMetric(value=None, availability="pending"),
            UsageMetric(value=None, availability="unknown"),
        )
    )
    assert metric == UsageMetric(value=None, availability="unknown")


def test_raw_provider_usage_presence_is_preserved_without_langchain_inference() -> None:
    partial = _auditable_provider_usage({"prompt_tokens": 7})
    assert partial is not None
    assert partial.input_tokens == 7
    assert partial.output_tokens is None
    assert partial.total_tokens is None
    assert partial.cached_tokens is None
    assert partial.reasoning_tokens is None

    deepseek = _auditable_provider_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": None,
        }
    )
    assert deepseek == ProviderTokenUsage(
        input_tokens=10,
        output_tokens=2,
        cached_tokens=0,
        total_tokens=12,
    )


async def test_fake_runs_persist_exact_attempts_and_owner_safe_account_aggregate(
    app_client: tuple[Any, Any],
) -> None:
    client, app = app_client
    direct_thread = str(uuid4())
    direct = await client.post(
        f"/api/threads/{direct_thread}/initial-run",
        json={"message": "Explain this briefly", "idempotency_key": "direct-usage"},
    )
    assert direct.status_code == 201
    direct_run = direct.json()["run"]
    await _wait_for_run(client, direct_run["run_id"])

    tool_thread = str(uuid4())
    tool = await client.post(
        f"/api/threads/{tool_thread}/initial-run",
        json={
            "message": "What time is it in Asia/Shanghai?",
            "idempotency_key": "tool-usage",
        },
    )
    assert tool.status_code == 201
    tool_run_id = tool.json()["run"]["run_id"]
    tool_events_response = await client.get(f"/api/runs/{tool_run_id}/events")
    assert tool_events_response.status_code == 200
    tool_events = _parse_sse(tool_events_response.text)

    direct_snapshot = (await client.get(f"/api/threads/{direct_thread}")).json()
    direct_usage = direct_snapshot["runs"][0]["usage"]
    assert direct_usage["state"] == "final"
    assert direct_usage["model_call_count"] == 1
    assert direct_usage["retry_count"] == 0
    assert direct_usage["input_tokens"] == {"value": 80, "availability": "complete"}
    assert direct_usage["output_tokens"] == {"value": 24, "availability": "complete"}
    assert direct_usage["cached_tokens"] == {"value": 0, "availability": "complete"}
    assert direct_usage["reasoning_tokens"] == {
        "value": None,
        "availability": "unavailable",
    }
    assert direct_usage["total_tokens"] == {"value": 104, "availability": "complete"}
    assert direct_usage["time_to_first_visible_ms"] is not None
    assert direct_usage["generation_duration_ms"] is not None
    assert direct_usage["run_duration_ms"] is not None

    tool_snapshot = (await client.get(f"/api/threads/{tool_thread}")).json()
    tool_usage = tool_snapshot["runs"][0]["usage"]
    assert tool_usage["model_call_count"] == 2
    assert tool_usage["input_tokens"] == {"value": 240, "availability": "complete"}
    assert tool_usage["output_tokens"] == {"value": 48, "availability": "complete"}
    assert tool_usage["total_tokens"] == {"value": 288, "availability": "complete"}
    terminal_event = next(event for event in tool_events if event["type"] == "run.completed")
    assert terminal_event["data"]["usage"] == tool_usage

    account = await client.get("/api/account/usage", params={"range": "30d"})
    assert account.status_code == 200
    payload = account.json()
    assert set(payload["account"]) == {"display_name", "organization", "extensions"}
    serialized = str(payload)
    assert "session_id" not in serialized
    assert "roles" not in serialized
    assert payload["runs"] == {
        "total": 2,
        "completed": 2,
        "failed": 0,
        "cancelled": 0,
        "active": 0,
    }
    assert payload["model_calls"] == {"value": 3, "availability": "complete"}
    assert payload["retries"] == {"value": 0, "availability": "complete"}
    assert payload["input_tokens"] == {"value": 320, "availability": "complete"}
    assert payload["output_tokens"] == {"value": 72, "availability": "complete"}
    assert payload["cached_tokens"] == {"value": 0, "availability": "complete"}
    assert payload["reasoning_tokens"] == {
        "value": None,
        "availability": "unavailable",
    }
    assert payload["total_tokens"] == {"value": 392, "availability": "complete"}

    async with app.state.repository._sessions() as session:  # noqa: SLF001
        before_replay = await session.scalar(select(func.count(ModelAttemptRecord.id)))
    await _wait_for_run(client, tool_run_id)
    replay = await client.post(
        f"/api/threads/{tool_thread}/runs",
        json={
            "message": "What time is it in Asia/Shanghai?",
            "idempotency_key": "tool-usage",
        },
    )
    assert replay.status_code == 201
    assert replay.json()["run_id"] == tool_run_id
    async with app.state.repository._sessions() as session:  # noqa: SLF001
        after_replay = await session.scalar(select(func.count(ModelAttemptRecord.id)))
    assert before_replay == after_replay == 3


async def test_account_scope_hides_foreign_and_absent_threads_identically(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'owners.db'}"
    settings = Settings(
        app_env="test",
        identity_provider_mode="development_header",
        database_url=database_url,
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
        model_mode="fake",
        fake_step_delay_seconds=0,
        sse_poll_interval_seconds=0.001,
        sse_keepalive_seconds=0.01,
        run_timeout_seconds=5,
    )
    database = Database(database_url)
    await database.create_schema_for_tests()
    await database.dispose()
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        headers_a = {DEVELOPMENT_PRINCIPAL_HEADER: "principal-a"}
        headers_b = {DEVELOPMENT_PRINCIPAL_HEADER: "principal-b"}
        async with (
            AsyncClient(transport=transport, base_url="http://test", headers=headers_a) as a,
            AsyncClient(transport=transport, base_url="http://test", headers=headers_b) as b,
        ):
            thread_id = str(uuid4())
            created = await a.post(
                f"/api/threads/{thread_id}/initial-run",
                json={"message": "A private question", "idempotency_key": "owner-a"},
            )
            await _wait_for_run(a, created.json()["run"]["run_id"])

            own = await a.get("/api/account/usage", params={"thread_id": thread_id})
            assert own.status_code == 200
            assert own.json()["runs"]["total"] == 1
            other = await b.get("/api/account/usage")
            assert other.status_code == 200
            assert other.json()["runs"]["total"] == 0

            foreign = await b.get("/api/account/usage", params={"thread_id": thread_id})
            absent = await b.get(
                "/api/account/usage",
                params={"thread_id": str(uuid4())},
            )
            assert foreign.status_code == absent.status_code == 404
            assert foreign.json() == absent.json() == {
                "detail": {"code": "usage_scope_not_found"}
            }


async def test_failed_run_keeps_provider_usage_and_exposes_stable_safe_category(
    app_client: tuple[Any, Any],
) -> None:
    client, _ = app_client
    thread_id = str(uuid4())
    response = await client.post(
        f"/api/threads/{thread_id}/initial-run",
        json={
            "message": "[policy:tool-denied]",
            "idempotency_key": "failed-usage",
        },
    )
    assert response.status_code == 201
    run_id = response.json()["run"]["run_id"]
    events_response = await client.get(f"/api/runs/{run_id}/events")
    assert events_response.status_code == 200
    events = _parse_sse(events_response.text)
    terminal = events[-1]
    assert terminal["type"] == "run.failed"
    assert terminal["data"]["error_code"] == "tool_not_allowed"
    usage = terminal["data"]["usage"]
    assert usage["state"] == "final"
    assert usage["model_call_count"] == 1
    assert usage["input_tokens"] == {"value": 96, "availability": "complete"}
    assert usage["output_tokens"] == {"value": 12, "availability": "complete"}
    assert usage["reasoning_tokens"] == {
        "value": None,
        "availability": "unavailable",
    }
    assert usage["error_category"] == "access_or_input"
    serialized = json.dumps(terminal).casefold()
    assert "traceback" not in serialized
    assert "prompt" not in serialized
    assert "tool_call" not in serialized


class _ObservedUsageCancellationRunner:
    def __init__(self) -> None:
        self.usage_observed = asyncio.Event()
        self.cancel_seen = asyncio.Event()
        self.allow_finish = asyncio.Event()

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, built_context
        execution.before_model_call()
        attempt_id = await execution.begin_provider_attempt(call_kind="direct")
        execution.observe_provider_usage(
            attempt_id,
            ProviderTokenUsage(
                input_tokens=11,
                output_tokens=5,
                cached_tokens=0,
                total_tokens=16,
            ),
        )
        self.usage_observed.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_seen.set()
            await self.allow_finish.wait()
            await execution.finish_provider_attempt(attempt_id, status="cancelled")
            raise
        if False:  # pragma: no cover - retain the AgentRunner async-generator shape.
            yield AgentResult(text="unreachable", source_ids=())


async def test_cancel_flushes_observed_usage_before_freezing_terminal_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'cancel-usage-race.db'}"
    settings = Settings(
        app_env="test",
        identity_provider_mode="anonymous",
        database_url=database_url,
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
        model_mode="fake",
        fake_step_delay_seconds=0,
        sse_poll_interval_seconds=0.001,
        sse_keepalive_seconds=0.01,
        run_timeout_seconds=5,
    )
    database = Database(database_url)
    await database.create_schema_for_tests()
    await database.dispose()
    runner = _ObservedUsageCancellationRunner()
    app = create_app(settings, runner_decorator=lambda _: runner)

    async with app.router.lifespan_context(app):
        usage_write_entered = asyncio.Event()
        allow_usage_write = asyncio.Event()
        original = app.state.repository.record_model_attempt_usage

        async def gated_usage_write(run_id: str, observed: Any) -> bool:
            usage_write_entered.set()
            await allow_usage_write.wait()
            return await original(run_id, observed)

        monkeypatch.setattr(
            app.state.repository,
            "record_model_attempt_usage",
            gated_usage_write,
        )
        transport = ASGITransport(app=app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            thread_id = str(uuid4())
            created = await client.post(
                f"/api/threads/{thread_id}/initial-run",
                json={"message": "Cancel after usage", "idempotency_key": "cancel-race"},
            )
            assert created.status_code == 201
            run_id = created.json()["run"]["run_id"]
            await asyncio.wait_for(runner.usage_observed.wait(), timeout=1)

            cancel_task = asyncio.create_task(client.post(f"/api/runs/{run_id}/cancel"))
            await asyncio.wait_for(runner.cancel_seen.wait(), timeout=1)
            await asyncio.wait_for(usage_write_entered.wait(), timeout=1)
            assert cancel_task.done() is False
            allow_usage_write.set()
            cancelled = await asyncio.wait_for(cancel_task, timeout=1)
            runner.allow_finish.set()
            await app.state.run_service.wait_for_idle()

            assert cancelled.status_code == 200
            usage = cancelled.json()["usage"]
            assert usage["state"] == "final"
            assert usage["model_call_count"] == 1
            assert usage["input_tokens"] == {
                "value": 11,
                "availability": "complete",
            }
            assert usage["output_tokens"] == {
                "value": 5,
                "availability": "complete",
            }
            assert usage["cached_tokens"] == {
                "value": 0,
                "availability": "complete",
            }
            assert usage["total_tokens"] == {
                "value": 16,
                "availability": "complete",
            }
            assert usage["reasoning_tokens"] == {
                "value": None,
                "availability": "unavailable",
            }
            assert usage["error_category"] == "cancelled"
