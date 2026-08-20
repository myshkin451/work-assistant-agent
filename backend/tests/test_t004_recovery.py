from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast, get_args

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from openai import APIConnectionError
from policy_fixtures import (
    make_execution,
    make_execution_plan,
    make_settings,
    terminal_outcome,
)

import work_assistant.repository as repository_module
from work_assistant.agent_runtime import (
    AgentResult,
    AgentRunner,
    ProductEvent,
    RuntimeCleanupTimeout,
)
from work_assistant.bootstrap import build_policy_kernel
from work_assistant.context_builder import BuiltContext
from work_assistant.db import Database
from work_assistant.execution_policy import (
    ExecutionOutcomeEvidence,
    PolicyKernelConfigurationError,
    RunExecution,
    execute_tool_call,
)
from work_assistant.identity import Principal
from work_assistant.models import EventRecord, RunRecord, utc_now
from work_assistant.repository import ProductRepository, RepositoryUnavailableError
from work_assistant.schemas import (
    Message,
    ProductEventValidationError,
    RunFailureCode,
    RuntimeEventType,
    RunView,
    validate_product_event,
)
from work_assistant.service import RunService

TEST_PRINCIPAL = Principal(subject="neutral-test-principal")


@pytest.fixture
async def recovery_repository(tmp_path: Path) -> AsyncIterator[ProductRepository]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 't004-recovery.db'}")
    await database.create_schema_for_tests()
    try:
        yield ProductRepository(database.session_factory)
    finally:
        await database.dispose()


def service_for(
    repository: ProductRepository,
    runner: AgentRunner,
    *,
    run_timeout_seconds: float = 2,
) -> RunService:
    settings = make_settings(
        run_timeout_seconds=run_timeout_seconds,
    )
    return RunService(
        repository=repository,
        runner=runner,
        policy_kernel=build_policy_kernel(settings),
        settings=settings,
    )


class CapturingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[tuple[str, str, str | None]]]] = []

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        assert built_context.conversation == tuple(messages)
        self.calls.append(
            (
                thread_id,
                run_id,
                [(message.role, message.content, message.run_id) for message in messages],
            )
        )
        prompt = messages[-1].content
        timezone = {
            "请查询当前上海时间。": "Asia/Shanghai",
            "那伦敦呢？": "Europe/London",
            "再看看纽约。": "America/New_York",
        }.get(prompt, "UTC")
        tool_call_id = f"time-{run_id}"
        tool_call = {
            "id": tool_call_id,
            "name": "get_current_time",
            "args": {"timezone": timezone},
            "type": "tool_call",
        }
        execution.before_model_call()
        execution.after_model_response([AIMessage(content="", tool_calls=[tool_call])])

        emitted: list[ProductEvent] = []

        async def emit(event: ProductEvent) -> None:
            emitted.append(event)

        async def invoke(validated: dict[str, Any]) -> ToolMessage:
            implementation = execution.tool_registry.require("get_current_time").implementation
            output = await implementation.ainvoke(validated)
            assert isinstance(output, str)
            return ToolMessage(
                content=output,
                tool_call_id=tool_call_id,
                name="get_current_time",
            )

        await execute_tool_call(
            execution=execution,
            tool_call_id=tool_call_id,
            tool_id="get_current_time",
            arguments={"timezone": timezone},
            handler=invoke,
            emit=emit,
        )
        for event in emitted:
            yield event

        execution.before_model_call()
        execution.after_model_response([AIMessage(content=f"answer:{prompt}")])
        yield ProductEvent("message.delta", {"delta": f"answer:{prompt}"})
        yield AgentResult(
            text=f"answer:{prompt}",
            source_ids=execution.generated_source_ids,
        )


class InvalidEventRunner:
    def __init__(self, event_type: str, data: dict[str, Any]) -> None:
        self._event_type = event_type
        self._data = data

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        yield ProductEvent(self._event_type, self._data)
        yield AgentResult(text="must not complete", source_ids=())


class SlowRunner:
    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        await asyncio.Event().wait()
        yield AgentResult(text="must time out", source_ids=())


class FatalRuntimeRunner:
    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        raise RuntimeCleanupTimeout("runtime_cleanup_timeout")
        if False:  # pragma: no cover - make this an async generator.
            yield AgentResult(text="unreachable", source_ids=())


class CancellationSensitiveRepository:
    """Minimal cancel repository for repeated-cancellation cleanup probes."""

    def __init__(self) -> None:
        self.run = RunView(
            run_id="cancellation-probe-run",
            thread_id="cancellation-probe-thread",
            status="cancelled",
            last_seq=0,
            created_at=utc_now(),
            completed_at=utc_now(),
        )

    async def cancel_run(
        self,
        run_id: str,
        *,
        principal: Principal,
        execution_outcome: ExecutionOutcomeEvidence,
    ) -> RunView:
        del principal
        assert run_id == self.run.run_id
        assert execution_outcome.status == "cancelled"
        assert execution_outcome.stop_reason == "user_cancelled"
        return self.run


class RepositoryCallGate(CancellationSensitiveRepository):
    """Expose whether Run cancellation reaches an in-flight repository call."""

    def __init__(self) -> None:
        super().__init__()
        self.start_entered = asyncio.Event()
        self.start_release = asyncio.Event()
        self.start_cancelled = asyncio.Event()

    async def start_run(self, run_id: str) -> bool:
        assert run_id == self.run.run_id
        self.start_entered.set()
        try:
            await self.start_release.wait()
        except asyncio.CancelledError:
            self.start_cancelled.set()
            raise
        return False


async def test_cancel_shields_inflight_repository_cleanup() -> None:
    repository = RepositoryCallGate()
    service = service_for(cast(ProductRepository, repository), CapturingRunner())
    service._launch(  # noqa: SLF001
        repository.run.run_id,
        make_execution(TEST_PRINCIPAL),
    )
    await asyncio.wait_for(repository.start_entered.wait(), timeout=1)

    cancelled = await service.cancel_run(repository.run.run_id, principal=TEST_PRINCIPAL)
    shutdown = asyncio.create_task(service.shutdown())
    await asyncio.sleep(0)

    worker = service._tasks[repository.run.run_id]  # noqa: SLF001
    assert cancelled.status == "cancelled"
    assert worker.cancelling() == 1
    assert repository.start_cancelled.is_set() is False
    assert shutdown.done() is False

    repository.start_release.set()
    await asyncio.wait_for(shutdown, timeout=1)
    await asyncio.sleep(0)
    assert repository.start_cancelled.is_set() is False
    assert service._tasks == {}  # noqa: SLF001


@pytest.mark.parametrize("shutdown_first", [False, True])
async def test_cancel_and_shutdown_request_worker_cancellation_once(
    shutdown_first: bool,
) -> None:
    repository = CancellationSensitiveRepository()
    service = service_for(
        cast(ProductRepository, repository),
        CapturingRunner(),
    )
    worker_started = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def cancellation_sensitive_worker() -> None:
        try:
            worker_started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_entered.set()
            await cleanup_release.wait()

    worker = asyncio.create_task(cancellation_sensitive_worker())
    service._tasks[repository.run.run_id] = worker  # noqa: SLF001
    await asyncio.wait_for(worker_started.wait(), timeout=1)

    if shutdown_first:
        shutdown = asyncio.create_task(service.shutdown())
        await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
        cancelled = await service.cancel_run(repository.run.run_id, principal=TEST_PRINCIPAL)
    else:
        cancelled = await service.cancel_run(repository.run.run_id, principal=TEST_PRINCIPAL)
        await asyncio.wait_for(cleanup_entered.wait(), timeout=1)
        shutdown = asyncio.create_task(service.shutdown())
        await asyncio.sleep(0)

    assert cancelled.status == "cancelled"
    assert worker.cancelling() == 1
    assert shutdown.done() is False

    cleanup_release.set()
    await asyncio.wait_for(shutdown, timeout=1)


class RaisingRunner:
    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        yield ProductEvent("message.delta", {"delta": "safe prefix"})
        raise APIConnectionError(
            message="provider-secret-sentinel",
            request=httpx.Request("POST", "https://provider.invalid"),
        )


class CumulativeDeltaOverflowRunner:
    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        yield ProductEvent("message.delta", {"delta": "safe prefix"})
        yield ProductEvent("message.delta", {"delta": "x" * 8_000})
        yield AgentResult(text="must not complete", source_ids=())


class StreamResultMismatchRunner:
    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        yield ProductEvent("message.delta", {"delta": "safe prefix"})
        yield AgentResult(text="different terminal result", source_ids=())


class TerminalOnlyRunner:
    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[ProductEvent | AgentResult]:
        del thread_id, run_id, messages, execution, built_context
        yield AgentResult(text="terminal text must not be post-hoc chunked", source_ids=())


async def test_three_runs_keep_messages_events_and_context_isolated(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    runner = CapturingRunner()
    service = service_for(repository, runner)
    thread = await repository.create_thread(principal=TEST_PRINCIPAL, title="Three-turn context")
    prompts = ("请查询当前上海时间。", "那伦敦呢？", "再看看纽约。")
    run_ids: list[str] = []

    for index, prompt in enumerate(prompts, start=1):
        run = await service.create_run(
            principal=TEST_PRINCIPAL,
            thread_id=thread.thread_id,
            message=prompt,
            idempotency_key=f"turn-{index}",
        )
        run_ids.append(run.run_id)
        await service.wait_for_idle()
        assert (
            await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
        ).status == "completed"

    snapshot = await repository.get_thread(thread.thread_id, principal=TEST_PRINCIPAL)
    assert snapshot.active_run is None
    assert [run.run_id for run in snapshot.runs] == run_ids
    assert [message.role for message in snapshot.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message.run_id for message in snapshot.messages] == [
        run_ids[0],
        run_ids[0],
        run_ids[1],
        run_ids[1],
        run_ids[2],
        run_ids[2],
    ]

    for index, run in enumerate(snapshot.runs):
        assert run.status == "completed"
        assert [event.seq for event in run.events] == list(range(1, run.last_seq + 1))
        assert all(event.run_id == run.run_id for event in run.events)
        assert all(event.thread_id == thread.thread_id for event in run.events)
        assert [event.type for event in run.events] == [
            "run.started",
            "tool.started",
            "tool.finished",
            "source.added",
            "message.delta",
            "message.completed",
            "run.completed",
        ]
        tool_started = next(event for event in run.events if event.type == "tool.started")
        assert (
            tool_started.data["input_summary"]
            == (
                "Asia/Shanghai",
                "Europe/London",
                "America/New_York",
            )[index]
        )
        plan, outcome = await repository.get_run_evidence(
            run.run_id,
            principal=TEST_PRINCIPAL,
        )
        assert plan is not None
        assert plan["agent_id"] == "default-work-assistant"
        assert plan["visible_tools"] == [
            {
                "tool_id": "get_current_time",
                "version": "1.1.0",
            }
        ]
        assert outcome is not None
        assert outcome["status"] == "completed"
        assert outcome["result_validation"] == "passed"
        assert outcome["accepted_source_ids"] == ["system-clock-iana-tzdb"]
        usage = outcome["usage"]
        assert usage is not None
        assert usage["model_steps"] == 2
        assert usage["tool_calls_attempted"] == 1
        assert usage["tool_calls_succeeded"] == 1
        assert usage["repeated_tool_calls"] == 0
        assert usage["no_progress_steps"] == 0
        assert usage["elapsed_ms"] >= 0

    assert len(runner.calls) == 3
    for index, (thread_id, run_id, captured) in enumerate(runner.calls):
        expected_messages: list[tuple[str, str, str | None]] = []
        for previous in range(index):
            expected_messages.extend(
                [
                    ("user", prompts[previous], run_ids[previous]),
                    ("assistant", f"answer:{prompts[previous]}", run_ids[previous]),
                ]
            )
        expected_messages.append(("user", prompts[index], run_ids[index]))
        assert thread_id == thread.thread_id
        assert run_id == run_ids[index]
        assert captured == expected_messages

    await service.shutdown()


@pytest.mark.parametrize(
    ("event_type", "data", "expected_code"),
    [
        pytest.param(
            "future.runtime.event",
            {"marker": "private-runtime-payload"},
            "agent_execution_failed",
            id="unknown",
        ),
        pytest.param(
            "provider.reasoning",
            {"reasoning": "private-runtime-payload"},
            "agent_execution_failed",
            id="private",
        ),
        pytest.param(
            "run.completed",
            {"status": "completed", "marker": "private-runtime-payload"},
            "agent_execution_failed",
            id="host-owned",
        ),
        pytest.param(
            "tool.started",
            {
                "tool_call_id": "bad-tool",
                "name": "get_current_time",
                "input_summary": "private-runtime-payload",
            },
            "tool_not_allowed",
            id="bad-payload",
        ),
        pytest.param(
            "source.added",
            {
                "source_id": "system-clock-iana-tzdb",
                "label": "System clock with IANA timezone data",
                "description": ("Current server clock converted with the requested IANA timezone."),
            },
            "source_validation_failed",
            id="source-without-successful-tool-ledger",
        ),
    ],
)
async def test_invalid_runtime_events_fail_without_persisting_or_skipping_sequence(
    recovery_repository: ProductRepository,
    event_type: str,
    data: dict[str, Any],
    expected_code: RunFailureCode,
) -> None:
    repository = recovery_repository
    service = service_for(repository, InvalidEventRunner(event_type, data))
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title=f"Invalid event: {event_type}"
    )

    run = await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Trigger invalid Runtime output",
        idempotency_key=f"invalid-{event_type}",
    )
    await service.wait_for_idle()

    failed = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert failed.status == "failed"
    assert failed.last_seq == 2
    assert [event.seq for event in events] == [1, 2]
    assert [event.type for event in events] == ["run.started", "run.failed"]
    assert events[-1].data == {
        "status": "failed",
        "error_code": expected_code,
    }
    plan, outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert plan is not None
    assert outcome is not None
    assert outcome["status"] == "failed"
    assert outcome["failure_code"] == expected_code
    assert outcome["result_validation"] == "not_run"
    assert outcome["accepted_source_ids"] == []
    serialized = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "plan": plan,
            "outcome": outcome,
        }
    )
    assert event_type not in serialized
    assert "private-runtime-payload" not in serialized

    await service.shutdown()


async def test_repository_rejects_invalid_events_before_reserving_sequence(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    thread = await repository.create_thread(principal=TEST_PRINCIPAL, title="Repository validation")
    run, created = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Keep the sequence contiguous",
        idempotency_key="repository-invalid-event",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert created is True
    assert await repository.start_run(run.run_id) is True

    invalid_events = [
        (cast(RuntimeEventType, "provider.reasoning"), {"reasoning": "private"}),
        (
            cast(RuntimeEventType, "run.completed"),
            {"status": "completed"},
        ),
        (
            cast(RuntimeEventType, "run.failed"),
            {"status": "failed", "error_code": "agent_execution_failed"},
        ),
        (
            cast(RuntimeEventType, "message.completed"),
            {
                "message": {
                    "message_id": "forged-assistant",
                    "role": "assistant",
                    "content": "forged",
                    "created_at": "2026-08-13T00:00:00Z",
                    "run_id": run.run_id,
                }
            },
        ),
        (
            cast(RuntimeEventType, "tool.started"),
            {"tool_call_id": "missing-fields"},
        ),
    ]
    for event_type, data in invalid_events:
        with pytest.raises(ProductEventValidationError):
            await repository.append_active_event(run.run_id, event_type, data)
        current = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
        assert current.status == "running"
        assert current.last_seq == 1
        assert [
            event.seq
            for event in await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
        ] == [1]

    appended = await repository.append_active_event(
        run.run_id,
        "message.delta",
        {"delta": "valid after rejected events"},
    )
    assert appended is not None
    assert appended.seq == 2
    await repository.cancel_run(
        run.run_id,
        principal=TEST_PRINCIPAL,
        execution_outcome=terminal_outcome(TEST_PRINCIPAL, status="cancelled"),
    )
    assert [
        event.seq for event in await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    ] == [1, 2, 3]


@pytest.mark.parametrize(
    ("operation", "execution_outcome", "expected_error"),
    [
        pytest.param(
            "complete",
            terminal_outcome(TEST_PRINCIPAL, status="cancelled"),
            "execution_outcome_status_mismatch",
            id="complete-status",
        ),
        pytest.param(
            "fail",
            terminal_outcome(TEST_PRINCIPAL, status="completed"),
            "execution_outcome_status_mismatch",
            id="fail-status",
        ),
        pytest.param(
            "fail",
            terminal_outcome(
                TEST_PRINCIPAL,
                status="failed",
                failure_code="agent_execution_failed",
            ),
            "execution_outcome_failure_mismatch",
            id="fail-code",
        ),
        pytest.param(
            "cancel",
            terminal_outcome(TEST_PRINCIPAL, status="completed"),
            "execution_outcome_status_mismatch",
            id="cancel-status",
        ),
    ],
)
async def test_repository_rejects_terminal_outcome_mismatches_without_mutation(
    recovery_repository: ProductRepository,
    operation: str,
    execution_outcome: ExecutionOutcomeEvidence,
    expected_error: str,
) -> None:
    repository = recovery_repository
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL,
        title=f"Reject {operation} evidence mismatch",
    )
    run, created = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Keep this Run active after rejected terminal evidence",
        idempotency_key=f"terminal-mismatch-{operation}-{expected_error}",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert created is True
    assert await repository.start_run(run.run_id) is True

    with pytest.raises(PolicyKernelConfigurationError, match=expected_error):
        if operation == "complete":
            await repository.complete_run(
                run.run_id,
                "must not persist",
                execution_outcome=execution_outcome,
            )
        elif operation == "fail":
            await repository.fail_run(
                run.run_id,
                "run_timeout",
                execution_outcome=execution_outcome,
            )
        else:
            await repository.cancel_run(
                run.run_id,
                principal=TEST_PRINCIPAL,
                execution_outcome=execution_outcome,
            )

    current = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    assert current.status == "running"
    assert current.last_seq == 1
    assert current.completed_at is None
    plan, persisted_outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert plan is not None
    assert persisted_outcome is None
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert [event.type for event in events] == ["run.started"]
    snapshot = await repository.get_thread(thread.thread_id, principal=TEST_PRINCIPAL)
    assert [message.role for message in snapshot.messages] == ["user"]


@pytest.mark.parametrize(
    ("runner", "timeout_seconds", "expected_code", "expected_types"),
    [
        pytest.param(
            SlowRunner(),
            0.25,
            "run_timeout",
            ["run.started", "run.failed"],
            id="timeout",
        ),
        pytest.param(
            RaisingRunner(),
            2,
            "agent_execution_failed",
            ["run.started", "message.delta", "run.failed"],
            id="provider-exception",
        ),
    ],
)
async def test_service_failures_are_bounded_contiguous_and_do_not_persist_exceptions(
    recovery_repository: ProductRepository,
    runner: AgentRunner,
    timeout_seconds: float,
    expected_code: str,
    expected_types: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="work_assistant.service")
    repository = recovery_repository
    service = service_for(
        repository,
        runner,
        run_timeout_seconds=timeout_seconds,
    )
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title=f"Service failure: {expected_code}"
    )
    run = await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Exercise bounded failure",
        idempotency_key=f"bounded-{expected_code}",
    )
    await service.wait_for_idle()

    failed = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert failed.status == "failed"
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert [event.type for event in events] == expected_types
    assert events[-1].data == {"status": "failed", "error_code": expected_code}
    plan, outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert plan is not None
    assert outcome is not None
    assert outcome["status"] == "failed"
    assert outcome["failure_code"] == expected_code
    assert outcome["stop_reason"] == expected_code
    assert outcome["result_validation"] == "not_run"
    serialized = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "plan": plan,
            "outcome": outcome,
        }
    )
    assert "provider-secret-sentinel" not in serialized
    assert "provider-secret-sentinel" not in caplog.text
    if expected_code == "agent_execution_failed":
        record = next(
            record for record in caplog.records if record.message == "run_execution_failed"
        )
        assert record.run_id == run.run_id
        assert record.error_category == "provider_connection"
        assert record.exception_type == "APIConnectionError"
        assert record.exc_info is None

    await service.shutdown()


async def test_runtime_cleanup_failure_stops_service_without_reusing_repository(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    service = service_for(repository, FatalRuntimeRunner())
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL,
        title="Runtime fail-stop",
    )
    run = await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Trip the Runtime cleanup boundary",
        idempotency_key="runtime-cleanup-fatal",
    )
    await service.wait_for_idle()

    assert service.is_healthy is False
    with pytest.raises(RepositoryUnavailableError, match="runtime_cleanup_timeout"):
        await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)

    fresh_repository = ProductRepository(repository._sessions)  # noqa: SLF001
    stranded = await fresh_repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    events = await fresh_repository.get_events(
        run.run_id,
        0,
        principal=TEST_PRINCIPAL,
    )
    assert stranded.status == "running"
    assert [event.type for event in events] == ["run.started"]

    await service.shutdown()


@pytest.mark.parametrize(
    ("runner", "expected_result_validation"),
    [
        pytest.param(CumulativeDeltaOverflowRunner(), "failed", id="cumulative-overflow"),
        pytest.param(StreamResultMismatchRunner(), "failed", id="stream-result-mismatch"),
    ],
)
async def test_invalid_stream_results_fail_stably_without_assistant_commit(
    recovery_repository: ProductRepository,
    runner: AgentRunner,
    expected_result_validation: str,
) -> None:
    repository = recovery_repository
    service = service_for(repository, runner)
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL,
        title="Invalid streamed result",
    )
    run = await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Reject an invalid streamed result",
        idempotency_key=f"invalid-stream-{expected_result_validation}",
    )
    await service.wait_for_idle()

    failed = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    assert failed.status == "failed"
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert [event.seq for event in events] == [1, 2, 3]
    assert [event.type for event in events] == [
        "run.started",
        "message.delta",
        "run.failed",
    ]
    assert events[1].data == {"delta": "safe prefix"}
    assert events[-1].data == {
        "status": "failed",
        "error_code": "result_schema_invalid",
    }
    assert "message.completed" not in {event.type for event in events}
    assert "run.completed" not in {event.type for event in events}

    snapshot = await repository.get_thread(thread.thread_id, principal=TEST_PRINCIPAL)
    assert [message.role for message in snapshot.messages] == ["user"]
    _, outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert outcome is not None
    assert outcome["status"] == "failed"
    assert outcome["failure_code"] == "result_schema_invalid"
    assert outcome["stop_reason"] == "result_schema_invalid"
    assert outcome["result_validation"] == expected_result_validation
    assert outcome["result_schema_version"] == "1.0.0"
    assert outcome["result_source_ids"] == []

    await service.shutdown()


async def test_terminal_result_without_live_deltas_fails_instead_of_being_chunked(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    service = service_for(repository, TerminalOnlyRunner())
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL,
        title="Reject terminal chunk fallback",
    )
    run = await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Require live Runtime deltas",
        idempotency_key="no-terminal-chunk-fallback",
    )
    await service.wait_for_idle()

    failed = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    assert failed.status == "failed"
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert [event.type for event in events] == ["run.started", "run.failed"]
    assert events[-1].data == {
        "status": "failed",
        "error_code": "result_schema_invalid",
    }
    snapshot = await repository.get_thread(thread.thread_id, principal=TEST_PRINCIPAL)
    assert [message.role for message in snapshot.messages] == ["user"]

    await service.shutdown()


async def test_orphan_sweep_is_idempotent_preserves_terminals_and_allows_new_run(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository

    created_thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title="Created orphan"
    )
    created_orphan, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=created_thread.thread_id,
        message="Created before restart",
        idempotency_key="created-orphan",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )

    running_thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title="Running orphan"
    )
    running_orphan, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=running_thread.thread_id,
        message="Running before restart",
        idempotency_key="running-orphan",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(running_orphan.run_id) is True
    await repository.append_active_event(
        running_orphan.run_id,
        "message.delta",
        {"delta": "partial result"},
    )
    await repository.append_active_event(
        running_orphan.run_id,
        "tool.started",
        {
            "tool_call_id": "orphan-time",
            "name": "get_current_time",
            "label": "Read current time",
            "input_summary": "UTC",
        },
    )
    await repository.append_active_event(
        running_orphan.run_id,
        "tool.finished",
        {
            "tool_call_id": "orphan-time",
            "name": "get_current_time",
            "label": "Read current time",
            "output_summary": "UTC: 2026-08-17T00:00:00+00:00",
        },
    )
    await repository.append_active_event(
        running_orphan.run_id,
        "source.added",
        {
            "source_id": "system-clock-iana-tzdb",
            "label": "System clock with IANA timezone data",
            "description": ("Current server clock converted with the requested IANA timezone."),
        },
    )

    legacy_thread = await repository.create_thread(
        principal=TEST_PRINCIPAL,
        title="Legacy orphan without policy evidence",
    )
    legacy_orphan, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=legacy_thread.thread_id,
        message="Created before policy evidence existed",
        idempotency_key="legacy-created-orphan",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    async with repository._sessions() as session, session.begin():  # noqa: SLF001
        stored_legacy = await session.get(RunRecord, legacy_orphan.run_id)
        assert stored_legacy is not None
        stored_legacy.execution_plan = None

    completed_thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title="Completed terminal"
    )
    completed, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=completed_thread.thread_id,
        message="Already complete",
        idempotency_key="completed-terminal",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(completed.run_id) is True
    await repository.complete_run(
        completed.run_id,
        "completed answer",
        execution_outcome=terminal_outcome(TEST_PRINCIPAL, status="completed"),
    )

    cancelled_thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title="Cancelled terminal"
    )
    cancelled, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=cancelled_thread.thread_id,
        message="Already cancelled",
        idempotency_key="cancelled-terminal",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(cancelled.run_id) is True
    await repository.cancel_run(
        cancelled.run_id,
        principal=TEST_PRINCIPAL,
        execution_outcome=terminal_outcome(TEST_PRINCIPAL, status="cancelled"),
    )

    terminal_event_counts = {
        completed.run_id: len(
            await repository.get_events(completed.run_id, 0, principal=TEST_PRINCIPAL)
        ),
        cancelled.run_id: len(
            await repository.get_events(cancelled.run_id, 0, principal=TEST_PRINCIPAL)
        ),
    }
    terminal_evidence = {
        completed.run_id: await repository.get_run_evidence(
            completed.run_id,
            principal=TEST_PRINCIPAL,
        ),
        cancelled.run_id: await repository.get_run_evidence(
            cancelled.run_id,
            principal=TEST_PRINCIPAL,
        ),
    }
    swept = await repository.fail_orphaned_runs()
    assert {run.run_id for run in swept} == {
        created_orphan.run_id,
        running_orphan.run_id,
        legacy_orphan.run_id,
    }

    expected_sources = {
        created_orphan.run_id: [],
        running_orphan.run_id: ["system-clock-iana-tzdb"],
    }
    for orphan_id in (created_orphan.run_id, running_orphan.run_id):
        orphan = await repository.get_run(orphan_id, principal=TEST_PRINCIPAL)
        events = await repository.get_events(orphan_id, 0, principal=TEST_PRINCIPAL)
        assert orphan.status == "failed"
        assert [event.seq for event in events] == list(range(1, orphan.last_seq + 1))
        assert events[-1].type == "run.failed"
        assert events[-1].data == {
            "status": "failed",
            "error_code": "service_restarted",
        }
        plan, outcome = await repository.get_run_evidence(
            orphan_id,
            principal=TEST_PRINCIPAL,
        )
        assert plan is not None
        assert outcome is not None
        assert outcome == {
            "schema_version": "1.0.0",
            "status": "failed",
            "stop_reason": "service_restarted",
            "failure_code": "service_restarted",
            "usage": None,
            "accepted_source_ids": expected_sources[orphan_id],
            "result_source_ids": [],
            "result_schema_version": None,
            "result_validation": "not_run",
        }

    legacy = await repository.get_run(legacy_orphan.run_id, principal=TEST_PRINCIPAL)
    legacy_events = await repository.get_events(
        legacy_orphan.run_id,
        0,
        principal=TEST_PRINCIPAL,
    )
    assert legacy.status == "failed"
    assert [event.type for event in legacy_events] == ["run.failed"]
    assert legacy_events[0].data == {
        "status": "failed",
        "error_code": "service_restarted",
    }
    assert await repository.get_run_evidence(
        legacy_orphan.run_id,
        principal=TEST_PRINCIPAL,
    ) == (None, None)

    orphan_event_counts = {
        orphan_id: len(await repository.get_events(orphan_id, 0, principal=TEST_PRINCIPAL))
        for orphan_id in (
            created_orphan.run_id,
            running_orphan.run_id,
            legacy_orphan.run_id,
        )
    }
    assert await repository.fail_orphaned_runs() == []
    assert {
        orphan_id: len(await repository.get_events(orphan_id, 0, principal=TEST_PRINCIPAL))
        for orphan_id in orphan_event_counts
    } == orphan_event_counts

    assert (
        await repository.get_run(completed.run_id, principal=TEST_PRINCIPAL)
    ).status == "completed"
    assert (
        await repository.get_run(cancelled.run_id, principal=TEST_PRINCIPAL)
    ).status == "cancelled"
    assert {
        completed.run_id: len(
            await repository.get_events(completed.run_id, 0, principal=TEST_PRINCIPAL)
        ),
        cancelled.run_id: len(
            await repository.get_events(cancelled.run_id, 0, principal=TEST_PRINCIPAL)
        ),
    } == terminal_event_counts
    assert {
        completed.run_id: await repository.get_run_evidence(
            completed.run_id,
            principal=TEST_PRINCIPAL,
        ),
        cancelled.run_id: await repository.get_run_evidence(
            cancelled.run_id,
            principal=TEST_PRINCIPAL,
        ),
    } == terminal_evidence

    same_run, created = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=created_thread.thread_id,
        message="Must return immutable old Run",
        idempotency_key="created-orphan",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert created is False
    assert same_run.run_id == created_orphan.run_id
    assert same_run.status == "failed"
    same_plan, same_outcome = await repository.get_run_evidence(
        same_run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert same_plan is not None
    assert same_outcome is not None
    assert same_outcome["failure_code"] == "service_restarted"

    retry_runner = CapturingRunner()
    retry_service = service_for(repository, retry_runner)
    retry = await retry_service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=created_thread.thread_id,
        message="Retry after restart",
        idempotency_key="created-orphan-retry",
    )
    await retry_service.wait_for_idle()
    assert retry.run_id != created_orphan.run_id
    assert (await repository.get_run(retry.run_id, principal=TEST_PRINCIPAL)).status == "completed"
    assert (
        await repository.get_run(created_orphan.run_id, principal=TEST_PRINCIPAL)
    ).status == "failed"
    await retry_service.shutdown()


async def test_orphan_sweep_rejects_active_run_with_terminal_outcome(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL,
        title="Contradictory active audit evidence",
    )
    run, created = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Do not overwrite a terminal outcome on an active Run",
        idempotency_key="active-with-terminal-outcome",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert created is True
    assert await repository.start_run(run.run_id) is True
    stored_outcome = terminal_outcome(
        TEST_PRINCIPAL,
        status="failed",
        failure_code="agent_execution_failed",
    ).model_dump(mode="json")
    async with repository._sessions() as session, session.begin():  # noqa: SLF001
        stored = await session.get(RunRecord, run.run_id)
        assert stored is not None
        stored.execution_outcome = stored_outcome

    with pytest.raises(
        PolicyKernelConfigurationError,
        match="active_run_execution_outcome_present",
    ):
        await repository.fail_orphaned_runs()

    current = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    assert current.status == "running"
    assert current.last_seq == 1
    plan, outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert plan is not None
    assert outcome == stored_outcome
    assert [
        event.type
        for event in await repository.get_events(
            run.run_id,
            0,
            principal=TEST_PRINCIPAL,
        )
    ] == ["run.started"]


async def test_failure_codes_are_frozen_and_stream_unavailable_cannot_be_terminal(
    recovery_repository: ProductRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = recovery_repository
    frozen_codes = {
        "run_timeout",
        "agent_execution_failed",
        "service_restarted",
        "model_step_limit",
        "tool_call_limit",
        "repeated_tool_call",
        "no_progress",
        "tool_not_allowed",
        "result_schema_invalid",
        "source_validation_failed",
    }
    assert set(get_args(RunFailureCode)) == frozen_codes
    for code in frozen_codes:
        assert (
            validate_product_event(
                "run.failed",
                {"status": "failed", "error_code": code},
            )["error_code"]
            == code
        )
    with pytest.raises(ProductEventValidationError):
        validate_product_event(
            "run.failed",
            {"status": "failed", "error_code": "stream_unavailable"},
        )

    thread = await repository.create_thread(principal=TEST_PRINCIPAL, title="Failure code rollback")
    run, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Reject connection-only state",
        idempotency_key="stream-unavailable",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(run.run_id) is True
    original_validate = repository_module.validate_product_event

    def reject_terminal_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        if event_type == "run.failed":
            raise ProductEventValidationError("synthetic terminal schema failure")
        return original_validate(event_type, data)

    # Use a valid failure/outcome pair so this reaches _event_record after the
    # status/outcome UPDATE and proves the whole terminal transaction rolls back.
    with monkeypatch.context() as patch_context:
        patch_context.setattr(
            repository_module,
            "validate_product_event",
            reject_terminal_event,
        )
        with pytest.raises(ProductEventValidationError):
            await repository.fail_run(
                run.run_id,
                "run_timeout",
                execution_outcome=terminal_outcome(
                    TEST_PRINCIPAL,
                    status="failed",
                    failure_code="run_timeout",
                ),
            )
    still_running = await repository.get_run(run.run_id, principal=TEST_PRINCIPAL)
    assert still_running.status == "running"
    assert still_running.last_seq == 1
    plan, outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert plan is not None
    assert outcome is None
    assert [
        event.type for event in await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    ] == ["run.started"]

    await repository.fail_run(
        run.run_id,
        "run_timeout",
        execution_outcome=terminal_outcome(
            TEST_PRINCIPAL,
            status="failed",
            failure_code="run_timeout",
        ),
    )
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    assert [event.seq for event in events] == [1, 2]
    assert events[-1].data["error_code"] == "run_timeout"
    _, outcome = await repository.get_run_evidence(
        run.run_id,
        principal=TEST_PRINCIPAL,
    )
    assert outcome is not None
    assert outcome["failure_code"] == "run_timeout"


async def test_v01_result_missing_is_normalized_only_for_stored_event_reads(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    legacy = {"status": "failed", "error_code": "agent_result_missing"}
    with pytest.raises(ProductEventValidationError):
        validate_product_event("run.failed", legacy)

    thread = await repository.create_thread(principal=TEST_PRINCIPAL, title="v0.1 compatibility")
    run, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="Read the legacy terminal event",
        idempotency_key="legacy-result-missing",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(run.run_id) is True
    now = utc_now()
    async with repository._sessions() as session, session.begin():  # noqa: SLF001
        stored_run = await session.get(RunRecord, run.run_id)
        assert stored_run is not None
        stored_run.status = "failed"
        stored_run.last_seq = 2
        stored_run.completed_at = now
        session.add(
            EventRecord(
                id="legacy-agent-result-missing",
                run_id=run.run_id,
                thread_id=thread.thread_id,
                seq=2,
                type="run.failed",
                occurred_at=now,
                payload=legacy,
            )
        )

    expected = {"status": "failed", "error_code": "agent_execution_failed"}
    events = await repository.get_events(run.run_id, 0, principal=TEST_PRINCIPAL)
    snapshot = await repository.get_thread(thread.thread_id, principal=TEST_PRINCIPAL)
    assert events[-1].data == expected
    assert snapshot.runs[0].events[-1].data == expected
    assert legacy["error_code"] == "agent_result_missing"


async def test_failed_and_cancelled_user_messages_do_not_enter_later_context(
    recovery_repository: ProductRepository,
) -> None:
    repository = recovery_repository
    thread = await repository.create_thread(
        principal=TEST_PRINCIPAL, title="Committed context only"
    )

    completed, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="completed question",
        idempotency_key="completed",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(completed.run_id) is True
    await repository.complete_run(
        completed.run_id,
        "completed answer",
        execution_outcome=terminal_outcome(TEST_PRINCIPAL, status="completed"),
    )

    failed, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="failed question must stay out",
        idempotency_key="failed",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(failed.run_id) is True
    await repository.fail_run(
        failed.run_id,
        "agent_execution_failed",
        execution_outcome=terminal_outcome(
            TEST_PRINCIPAL,
            status="failed",
            failure_code="agent_execution_failed",
        ),
    )

    cancelled, _ = await repository.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="cancelled question must stay out",
        idempotency_key="cancelled",
        execution_plan=make_execution_plan(TEST_PRINCIPAL),
    )
    assert await repository.start_run(cancelled.run_id) is True
    await repository.cancel_run(
        cancelled.run_id,
        principal=TEST_PRINCIPAL,
        execution_outcome=terminal_outcome(TEST_PRINCIPAL, status="cancelled"),
    )

    runner = CapturingRunner()
    service = service_for(repository, runner)
    current = await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=thread.thread_id,
        message="current question",
        idempotency_key="current",
    )
    await service.wait_for_idle()

    assert len(runner.calls) == 1
    thread_id, run_id, captured = runner.calls[0]
    assert thread_id == thread.thread_id
    assert run_id == current.run_id
    assert captured == [
        ("user", "completed question", completed.run_id),
        ("assistant", "completed answer", completed.run_id),
        ("user", "current question", current.run_id),
    ]
    assert "failed question must stay out" not in {content for _, content, _ in captured}
    assert "cancelled question must stay out" not in {content for _, content, _ in captured}

    await service.shutdown()
