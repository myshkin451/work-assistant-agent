from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import pytest
from policy_fixtures import make_settings

from work_assistant.agent_runtime import AgentResult, AgentRunner, ProductEvent
from work_assistant.bootstrap import build_policy_kernel
from work_assistant.context_builder import BuiltContext
from work_assistant.execution_policy import RunExecution
from work_assistant.identity import Principal
from work_assistant.models import utc_now
from work_assistant.repository import ProductRepository
from work_assistant.schemas import Message, RunView
from work_assistant.service import (
    RepositoryCleanupTimeout,
    RunAdmissionTimeout,
    RunService,
    RunServiceUnavailable,
)

TEST_PRINCIPAL = Principal(subject="repository-deadline-principal")
THREAD_ID = "repository-deadline-thread"
RUN_ID = "repository-deadline-run"


class UnusedRunner:
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
        if False:  # pragma: no cover - this runner must never be entered.
            yield AgentResult(text="unused", source_ids=())
        raise AssertionError("admission test unexpectedly entered the Runtime")


class LaunchCapturingRunService(RunService):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.launches: list[tuple[str, RunExecution, float]] = []

    def _launch(self, run_id: str, execution: RunExecution) -> None:
        # Keep these tests on the admission boundary. Recording remaining time
        # here also proves that admission and worker execution share one budget.
        self.launches.append((run_id, execution, execution.remaining_seconds))


def service_for(
    repository: object,
    *,
    run_timeout_seconds: float,
    database_operation_timeout_seconds: float = 0.1,
    repository_cleanup_grace_seconds: float = 0.2,
) -> LaunchCapturingRunService:
    settings = make_settings(
        run_timeout_seconds=run_timeout_seconds,
        database_operation_timeout_seconds=database_operation_timeout_seconds,
        repository_cleanup_grace_seconds=repository_cleanup_grace_seconds,
    )
    return LaunchCapturingRunService(
        repository=cast(ProductRepository, repository),
        runner=cast(AgentRunner, UnusedRunner()),
        policy_kernel=build_policy_kernel(settings),
        settings=settings,
    )


def test_postgres_timeout_smaller_than_libpq_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two seconds"):
        make_settings(
            database_url=(
                "postgresql+psycopg://work_assistant:work_assistant@localhost/work_assistant"
            ),
            database_operation_timeout_seconds=0.5,
            repository_cleanup_grace_seconds=1,
        )


def created_run() -> RunView:
    return RunView(
        run_id=RUN_ID,
        thread_id=THREAD_ID,
        status="created",
        last_seq=0,
        created_at=utc_now(),
        completed_at=None,
    )


async def request_run(service: RunService, *, key: str = "admission-key") -> RunView:
    return await service.create_run(
        principal=TEST_PRINCIPAL,
        thread_id=THREAD_ID,
        message="bounded admission",
        idempotency_key=key,
    )


class DeadlineAdmissionRepository:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancel_count = 0

    async def create_run(self, **_: Any) -> tuple[RunView, bool]:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_count += 1
            raise
        raise AssertionError("unreachable")


async def test_admission_deadline_cancels_operation_once_without_launch() -> None:
    repository = DeadlineAdmissionRepository()
    service = service_for(repository, run_timeout_seconds=0.02)

    with pytest.raises(RunAdmissionTimeout):
        await request_run(service)

    assert repository.entered.is_set()
    assert repository.cancel_count == 1
    assert service.launches == []
    assert service.is_healthy is True
    await service.shutdown()


class CommittingAdmissionRepository:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_count = 0
        self.calls = 0

    async def create_run(self, **_: Any) -> tuple[RunView, bool]:
        self.calls += 1
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancel_count += 1
            raise
        return created_run(), True


async def test_committed_admission_launches_once_despite_request_cancellation() -> None:
    repository = CommittingAdmissionRepository()
    service = service_for(
        repository,
        run_timeout_seconds=1,
        database_operation_timeout_seconds=0.1,
        repository_cleanup_grace_seconds=0.2,
    )
    request = asyncio.create_task(request_run(service))
    await asyncio.wait_for(repository.entered.wait(), timeout=1)

    request.cancel()
    await asyncio.sleep(0)
    repository.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, timeout=1)
    assert repository.cancel_count == 0
    assert repository.calls == 1
    assert [run_id for run_id, _, _ in service.launches] == [RUN_ID]
    await service.shutdown()


class StuckCleanupAdmissionRepository:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cleanup_entered = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancel_count = 0
        self.calls = 0
        self.operation_task: asyncio.Task[object] | None = None

    async def create_run(self, **_: Any) -> tuple[RunView, bool]:
        self.calls += 1
        self.operation_task = cast(asyncio.Task[object], asyncio.current_task())
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_count += 1
            self.cleanup_entered.set()
            await self.cleanup_release.wait()
            raise RuntimeError("late-cleanup-sentinel") from None
        finally:
            self.finished.set()
        raise AssertionError("unreachable")


async def test_cleanup_grace_trips_fatal_without_second_cancel() -> None:
    repository = StuckCleanupAdmissionRepository()
    service = service_for(
        repository,
        run_timeout_seconds=0.01,
        database_operation_timeout_seconds=0.1,
        repository_cleanup_grace_seconds=0.2,
    )

    with pytest.raises(RepositoryCleanupTimeout):
        await request_run(service)
    await asyncio.wait_for(repository.cleanup_entered.wait(), timeout=1)

    assert repository.cancel_count == 1
    assert service.is_healthy is False
    with pytest.raises(RunServiceUnavailable):
        await request_run(service, key="must-be-rejected")
    assert repository.calls == 1

    shutdown = asyncio.create_task(service.shutdown())
    await asyncio.sleep(0)
    assert repository.cancel_count == 1

    repository.cleanup_release.set()
    await asyncio.wait_for(repository.finished.wait(), timeout=1)
    await asyncio.wait_for(shutdown, timeout=1)
    await asyncio.sleep(0)

    assert repository.cancel_count == 1
    assert repository.operation_task is not None
    assert repository.operation_task.done()
    # The quarantine callback must consume a late child exception; otherwise
    # asyncio would report "Task exception was never retrieved" at teardown.
    assert getattr(repository.operation_task, "_log_traceback", True) is False


async def test_worker_inherits_deadline_started_before_admission() -> None:
    repository = CommittingAdmissionRepository()
    run_timeout_seconds = 0.30
    service = service_for(
        repository,
        run_timeout_seconds=run_timeout_seconds,
        database_operation_timeout_seconds=0.1,
        repository_cleanup_grace_seconds=0.2,
    )
    request = asyncio.create_task(request_run(service))
    await asyncio.wait_for(repository.entered.wait(), timeout=1)

    await asyncio.sleep(0.04)
    repository.release.set()
    assert (await asyncio.wait_for(request, timeout=1)).run_id == RUN_ID

    assert len(service.launches) == 1
    _, execution, remaining_at_launch = service.launches[0]
    assert execution.plan_evidence.budget.deadline_seconds == run_timeout_seconds
    assert remaining_at_launch < run_timeout_seconds - 0.02
    assert remaining_at_launch > 0
    await service.shutdown()
