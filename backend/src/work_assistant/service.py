from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from openai import APIConnectionError, APIError

from .agent_runtime import AgentResult, AgentRunner, ProductEvent, RuntimeFatalError
from .execution_policy import (
    AgentExecutionFailed,
    AgentPolicyKernel,
    PolicyViolation,
    ResultSchemaInvalid,
    RunExecution,
)
from .identity import Principal
from .repository import ProductRepository
from .schemas import (
    EventEnvelope,
    InitialRunResponse,
    ProductEventValidationError,
    RuntimeEventType,
    RunView,
    validate_runtime_event,
)
from .settings import Settings
from .usage import ModelAttemptFinish, ModelAttemptStart, ModelAttemptUsage, RunErrorCategory

logger = logging.getLogger(__name__)


def _safe_exception_log_fields(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, APIConnectionError):
        return "provider_connection", "APIConnectionError"
    if isinstance(exc, RuntimeError):
        return "runtime_error", "RuntimeError"
    return "unexpected", "UnclassifiedError"


def _policy_error_category(exc: PolicyViolation) -> RunErrorCategory:
    if exc.failure_code == "run_timeout":
        return "timeout"
    if exc.failure_code in {
        "model_step_limit",
        "tool_call_limit",
        "repeated_tool_call",
        "no_progress",
    }:
        return "limit"
    if exc.failure_code == "tool_not_allowed":
        return "access_or_input"
    if exc.failure_code in {"result_schema_invalid", "source_validation_failed"}:
        return "validation"
    if exc.stop_reason.startswith("tool_"):
        return "tool"
    return "internal"


def _exception_error_category(exc: Exception) -> RunErrorCategory:
    if isinstance(exc, APIError):
        return "provider"
    return "internal"


class RepositoryCleanupTimeout(RuntimeError):
    """A transaction did not settle after its one deadline cancellation."""


class RunAdmissionTimeout(RuntimeError):
    """Run persistence did not finish within the Run's original deadline."""


class RunServiceUnavailable(RuntimeError):
    """The executor failed closed after a repository cleanup invariant failed."""


async def _finish_repository_call[T](
    operation_factory: Callable[[], Awaitable[T]],
    *,
    deadline_at: float,
    cleanup_grace_seconds: float,
    quarantine: set[asyncio.Task[Any]],
) -> T:
    """Run one transaction with one cancellation and bounded cleanup observation.

    The child owns the absolute timeout, so SQLAlchemy receives exactly one
    cancellation and may finish rollback/connection cleanup. Parent cancellation
    records intent but never injects a second cancellation into that child.
    """

    async def execute() -> T:
        async with asyncio.timeout_at(deadline_at):
            return await operation_factory()

    operation = asyncio.create_task(execute())
    parent_cancelled = False

    def observe_quarantined(completed: asyncio.Task[Any]) -> None:
        quarantine.discard(completed)
        try:
            completed.exception()
        except BaseException:
            pass

    while True:
        if operation.done():
            try:
                result = operation.result()
            except BaseException:
                if parent_cancelled:
                    raise asyncio.CancelledError from None
                raise
            if parent_cancelled:
                raise asyncio.CancelledError
            return result

        remaining = deadline_at + cleanup_grace_seconds - time.monotonic()
        if remaining <= 0:
            quarantine.add(operation)
            operation.add_done_callback(observe_quarantined)
            raise RepositoryCleanupTimeout("repository_cleanup_timeout")
        try:
            await asyncio.wait({operation}, timeout=remaining)
        except asyncio.CancelledError:
            parent_cancelled = True


class DisconnectAware(Protocol):
    async def is_disconnected(self) -> bool: ...


class RunService:
    def __init__(
        self,
        *,
        repository: ProductRepository,
        runner: AgentRunner,
        policy_kernel: AgentPolicyKernel,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self._runner = runner
        self._policy_kernel = policy_kernel
        self._settings = settings
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._executions: dict[str, RunExecution] = {}
        self._terminal_locks: dict[str, asyncio.Lock] = {}
        self._quarantined_repository_tasks: set[asyncio.Task[Any]] = set()
        self._fatal_error: str | None = None
        self._closed = False

    @property
    def is_healthy(self) -> bool:
        return not self._closed and self._fatal_error is None

    def _require_healthy(self) -> None:
        if self._closed:
            raise RunServiceUnavailable("run_service_closed")
        self._require_repository_safe()

    def _require_repository_safe(self) -> None:
        if self._fatal_error is not None:
            raise RunServiceUnavailable(self._fatal_error)

    def _mark_fatal(self, reason: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = reason
        fail_closed = getattr(self.repository, "fail_closed", None)
        if callable(fail_closed):
            fail_closed(reason)
        current = asyncio.current_task()
        for task in tuple(self._tasks.values()):
            if task is not current:
                self._request_cancel(task)

    def _terminal_deadline(self) -> float:
        return time.monotonic() + self._settings.database_operation_timeout_seconds

    async def _repository_call[T](
        self,
        operation_factory: Callable[[], Awaitable[T]],
        *,
        deadline_at: float,
    ) -> T:
        self._require_repository_safe()

        async def guarded_operation() -> T:
            # Recheck inside the child immediately before the repository
            # coroutine begins. There is no await gap between this check and a
            # production repository's own synchronous admission gate.
            self._require_repository_safe()
            return await operation_factory()

        try:
            return await _finish_repository_call(
                guarded_operation,
                deadline_at=deadline_at,
                cleanup_grace_seconds=self._settings.repository_cleanup_grace_seconds,
                quarantine=self._quarantined_repository_tasks,
            )
        except RepositoryCleanupTimeout:
            self._mark_fatal("repository_cleanup_timeout")
            raise

    async def create_run(
        self,
        *,
        principal: Principal,
        thread_id: str,
        message: str,
        idempotency_key: str,
    ) -> RunView:
        self._require_healthy()
        execution = self._policy_kernel.prepare_run(principal=principal)

        async def admit_and_launch() -> RunView:
            view, created = await self.repository.create_run(
                principal=principal,
                thread_id=thread_id,
                message=message,
                idempotency_key=idempotency_key,
                execution_plan=execution.plan_evidence,
            )
            # No await may separate a committed new Run from worker ownership.
            if created:
                self._launch(view.run_id, execution)
            return view

        try:
            return await self._repository_call(
                admit_and_launch,
                deadline_at=execution.deadline_at,
            )
        except TimeoutError as exc:
            raise RunAdmissionTimeout("run_admission_timeout") from exc

    async def create_initial_run(
        self,
        *,
        principal: Principal,
        thread_id: str,
        message: str,
        idempotency_key: str,
    ) -> InitialRunResponse:
        self._require_healthy()
        execution = self._policy_kernel.prepare_run(principal=principal)

        async def admit_and_launch() -> InitialRunResponse:
            thread, view, created = await self.repository.create_initial_run(
                principal=principal,
                thread_id=thread_id,
                message=message,
                idempotency_key=idempotency_key,
                execution_plan=execution.plan_evidence,
            )
            # Match ordinary Run admission: a committed new Run cannot yield
            # ownership before its in-process worker has been registered.
            if created:
                self._launch(view.run_id, execution)
            return InitialRunResponse(thread=thread, run=view)

        try:
            return await self._repository_call(
                admit_and_launch,
                deadline_at=execution.deadline_at,
            )
        except TimeoutError as exc:
            raise RunAdmissionTimeout("run_admission_timeout") from exc

    async def cancel_run(self, run_id: str, *, principal: Principal) -> RunView:
        # Existing Runs remain cancellable during graceful shutdown. A fatal
        # repository cleanup failure is different: no transaction may be
        # attempted again until the process has restarted with a clean pool.
        self._require_repository_safe()
        execution = self._executions.get(run_id)
        usage_known = execution is not None
        if execution is None:
            # A terminal idempotent cancel will ignore this fresh snapshot. An
            # active Run without an in-process execution is not expected in the
            # supported single-executor topology because startup closes orphans.
            execution = self._policy_kernel.prepare_run(principal=principal)
        terminal_lock = self._terminal_locks.get(run_id)
        task = self._tasks.get(run_id)
        if task is not None:
            # Stop new provider progress first. The usage ledger then flushes any
            # terminal usage already observed before cancellation can freeze the
            # Run, without waiting for unrelated Runtime cleanup.
            self._request_cancel(task)
        if usage_known:
            await execution.settle_provider_usage_for_terminal()

        async def commit_cancel() -> RunView:
            outcome = execution.outcome(
                status="cancelled",
                stop_reason="user_cancelled",
                usage_known=usage_known,
            )
            return await self._repository_call(
                lambda: self.repository.cancel_run(
                    run_id,
                    principal=principal,
                    execution_outcome=outcome,
                ),
                deadline_at=self._terminal_deadline(),
            )

        if terminal_lock is None:
            view = await commit_cancel()
        else:
            # Serialize the terminal snapshot with any in-flight event append. If
            # the append wins, its ledger confirmation is included; if cancel
            # wins, the append observes the immutable terminal Run and is ignored.
            async with terminal_lock:
                view = await commit_cancel()
        return view

    async def stream_events(
        self,
        *,
        principal: Principal,
        run_id: str,
        after_seq: int,
        is_disconnected: DisconnectAware,
    ) -> AsyncIterator[EventEnvelope | None]:
        self._require_repository_safe()
        cursor = after_seq
        elapsed_since_keepalive = 0.0
        while True:
            self._require_repository_safe()
            events = await self.repository.get_events(run_id, cursor, principal=principal)
            for event in events:
                cursor = event.seq
                elapsed_since_keepalive = 0.0
                yield event
            run = await self.repository.get_run(run_id, principal=principal)
            if run.status in {"completed", "failed", "cancelled"} and cursor >= run.last_seq:
                return
            if await is_disconnected.is_disconnected():
                return
            await asyncio.sleep(self._settings.sse_poll_interval_seconds)
            elapsed_since_keepalive += self._settings.sse_poll_interval_seconds
            if elapsed_since_keepalive >= self._settings.sse_keepalive_seconds:
                elapsed_since_keepalive = 0.0
                yield None

    async def wait_for_idle(self) -> None:
        tasks = list(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        for task in tasks:
            self._request_cancel(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # A quarantined DB task has already received its one deadline
        # cancellation. Observe it for a bounded period, but never inject a
        # second cancellation during shutdown.
        if self._quarantined_repository_tasks:
            await asyncio.wait(
                tuple(self._quarantined_repository_tasks),
                timeout=self._settings.repository_cleanup_grace_seconds,
            )

    def _launch(self, run_id: str, execution: RunExecution) -> None:
        self._require_healthy()

        async def start_attempt(start: ModelAttemptStart) -> None:
            await self._repository_call(
                lambda: self.repository.start_model_attempt(run_id, start),
                deadline_at=execution.deadline_at,
            )

        async def finish_attempt(finish: ModelAttemptFinish) -> None:
            await self._repository_call(
                lambda: self.repository.finish_model_attempt(run_id, finish),
                deadline_at=self._terminal_deadline(),
            )

        async def persist_usage(observed: ModelAttemptUsage) -> None:
            await self._repository_call(
                lambda: self.repository.record_model_attempt_usage(run_id, observed),
                deadline_at=self._terminal_deadline(),
            )

        execution.bind_usage_persistence(
            start_writer=start_attempt,
            finish_writer=finish_attempt,
            usage_writer=persist_usage,
        )
        task = asyncio.create_task(
            self._execute(run_id, execution),
            name=f"product-run-{run_id}",
        )
        self._tasks[run_id] = task
        self._executions[run_id] = execution
        self._terminal_locks[run_id] = asyncio.Lock()

        def discard(completed: asyncio.Task[None]) -> None:
            self._tasks.pop(run_id, None)
            self._executions.pop(run_id, None)
            self._terminal_locks.pop(run_id, None)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(discard)

    @staticmethod
    def _request_cancel(task: asyncio.Task[None]) -> None:
        # A second cancel can interrupt SQLAlchemy's shielded transaction cleanup
        # and strand a pooled connection. Cancellation is a level-triggered Run
        # intent here, so one outstanding request is sufficient.
        if not task.done() and task.cancelling() == 0:
            task.cancel()

    async def _execute(self, run_id: str, execution: RunExecution) -> None:
        try:
            async with asyncio.timeout_at(execution.deadline_at):
                if not await self._repository_call(
                    lambda: self.repository.start_run(run_id),
                    deadline_at=execution.deadline_at,
                ):
                    return
                thread_id, messages = await self._repository_call(
                    lambda: self.repository.get_run_context(run_id),
                    deadline_at=execution.deadline_at,
                )
                built_context = execution.build_context(messages)
                result: AgentResult | None = None
                runtime_text_parts: list[str] = []
                runtime_text_chars = 0

                async def persist_controlled_event(
                    event: ProductEvent,
                    event_type: RuntimeEventType,
                    data: dict[str, Any],
                ) -> EventEnvelope | None:
                    async def append_and_confirm() -> EventEnvelope | None:
                        persisted_event = await self.repository.append_active_event(
                            run_id, event_type, data
                        )
                        if persisted_event is not None:
                            # This synchronous confirmation is part of the
                            # shielded child operation, so cancellation cannot
                            # split the database fact from its evidence ledger.
                            execution.accept_runtime_event(event)
                        return persisted_event

                    terminal_lock = self._terminal_locks[run_id]
                    async with terminal_lock:
                        return await self._repository_call(
                            append_and_confirm,
                            deadline_at=execution.deadline_at,
                        )

                async for item in self._runner.stream(
                    thread_id=thread_id,
                    run_id=run_id,
                    messages=messages,
                    execution=execution,
                    built_context=built_context,
                ):
                    if result is not None:
                        raise AgentExecutionFailed("runtime_item_after_result")
                    if isinstance(item, ProductEvent):
                        execution.validate_runtime_event(item)
                        try:
                            event_type, data = validate_runtime_event(item.type, item.data)
                        except ProductEventValidationError as exc:
                            if item.type == "message.delta":
                                raise ResultSchemaInvalid from exc
                            raise
                        if event_type == "message.delta":
                            runtime_text_chars += len(data["delta"])
                            if (
                                runtime_text_chars
                                > execution.agent.result_contract.max_answer_chars
                            ):
                                raise ResultSchemaInvalid
                            runtime_text_parts.append(data["delta"])
                        persisted = await persist_controlled_event(item, event_type, data)
                        if persisted is None:
                            return
                    else:
                        result = item
                if result is None:
                    raise AgentExecutionFailed("agent_result_missing")
                runtime_text = "".join(runtime_text_parts)
                validated_result = execution.validate_result(
                    result,
                    runtime_text=runtime_text,
                )
            outcome = execution.outcome(
                status="completed",
                stop_reason="completed",
            )
            await self._commit_terminal(
                run_id,
                lambda: self.repository.complete_run(
                    run_id,
                    validated_result.text,
                    execution_outcome=outcome,
                    error_category=None,
                ),
            )
        except asyncio.CancelledError:
            # The cancel endpoint has already committed the immutable terminal event.
            return
        except RuntimeFatalError:
            # The provider/checkpointer cleanup boundary is no longer reusable.
            # Stop all product work and let startup recovery close active Runs
            # after the process supervisor replaces this instance.
            self._mark_fatal("runtime_cleanup_timeout")
            return
        except (RepositoryCleanupTimeout, RunServiceUnavailable):
            # _repository_call has already failed the service closed. Do not
            # recursively attempt another transaction on the uncertain pool.
            return
        except TimeoutError:
            outcome = execution.outcome(
                status="failed",
                stop_reason="run_timeout",
                failure_code="run_timeout",
            )
            await self._commit_terminal(
                run_id,
                lambda: self.repository.fail_run(
                    run_id,
                    "run_timeout",
                    execution_outcome=outcome,
                    error_category="timeout",
                ),
            )
        except PolicyViolation as exc:
            if isinstance(exc, ResultSchemaInvalid):
                execution.record_result_validation_failure()
            failure_code = exc.failure_code
            error_category = _policy_error_category(exc)
            outcome = execution.outcome(
                status="failed",
                stop_reason=exc.stop_reason,
                failure_code=failure_code,
            )
            await self._commit_terminal(
                run_id,
                lambda: self.repository.fail_run(
                    run_id,
                    failure_code,
                    execution_outcome=outcome,
                    error_category=error_category,
                ),
            )
        except Exception as exc:
            # Provider responses and exception text are intentionally not persisted.
            log_category, exception_type = _safe_exception_log_fields(exc)
            persisted_error_category = _exception_error_category(exc)
            logger.warning(
                "run_execution_failed",
                extra={
                    "run_id": run_id,
                    "error_category": log_category,
                    "exception_type": exception_type,
                },
            )
            outcome = execution.outcome(
                status="failed",
                stop_reason="agent_execution_failed",
                failure_code="agent_execution_failed",
            )
            await self._commit_terminal(
                run_id,
                lambda: self.repository.fail_run(
                    run_id,
                    "agent_execution_failed",
                    execution_outcome=outcome,
                    error_category=persisted_error_category,
                ),
            )

    async def _commit_terminal[T](
        self,
        run_id: str,
        operation_factory: Callable[[], Awaitable[T]],
    ) -> T | None:
        """Persist one terminal fact without recursively retrying a bad pool."""

        try:
            async with self._terminal_locks[run_id]:
                return await self._repository_call(
                    operation_factory,
                    deadline_at=self._terminal_deadline(),
                )
        except asyncio.CancelledError:
            raise
        except RepositoryCleanupTimeout:
            # _repository_call already recorded the stronger fatal reason.
            return None
        except Exception:
            # A terminal write that cannot be confirmed leaves an active Run.
            # Stop admitting work so startup recovery can deterministically
            # close it after the supervisor replaces this process.
            self._mark_fatal("repository_finalization_failed")
            return None
