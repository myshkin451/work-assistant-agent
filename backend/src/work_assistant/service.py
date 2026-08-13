from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Protocol

from .agent_runtime import AgentResult, AgentRunner, ProductEvent
from .identity import Principal
from .repository import ProductRepository
from .schemas import EventEnvelope, RunView, validate_runtime_event
from .settings import Settings


async def _finish_repository_call[T](awaitable: Awaitable[T]) -> T:
    """Let a database call reach a clean boundary before propagating cancellation.

    SQLAlchemy shields parts of async transaction cleanup. Cancelling the parent
    Run task during that cleanup can strand a driver connection, so repository
    calls execute in a child task. The parent still records cancellation at once,
    waits only for the in-flight database operation to finish, and then exits.
    """

    async def execute() -> T:
        return await awaitable

    operation = asyncio.create_task(execute())
    cancellation_requested = False
    while True:
        try:
            result = await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancellation_requested = True
            if operation.done():
                break
        except BaseException:
            if cancellation_requested:
                break
            raise
        else:
            if cancellation_requested:
                raise asyncio.CancelledError
            return result

    # Retrieve any child exception so cancellation never leaves an unobserved
    # database-operation Task behind.
    try:
        operation.result()
    except BaseException:
        pass
    raise asyncio.CancelledError


class DisconnectAware(Protocol):
    async def is_disconnected(self) -> bool: ...


class RunService:
    def __init__(
        self,
        *,
        repository: ProductRepository,
        runner: AgentRunner,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self._runner = runner
        self._settings = settings
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    async def create_run(
        self,
        *,
        principal: Principal,
        thread_id: str,
        message: str,
        idempotency_key: str,
    ) -> RunView:
        view, created = await self.repository.create_run(
            principal=principal,
            thread_id=thread_id,
            message=message,
            idempotency_key=idempotency_key,
        )
        if created:
            self._launch(view.run_id)
        return view

    async def cancel_run(self, run_id: str, *, principal: Principal) -> RunView:
        view = await self.repository.cancel_run(run_id, principal=principal)
        task = self._tasks.get(run_id)
        if task is not None:
            self._request_cancel(task)
        return view

    async def stream_events(
        self,
        *,
        principal: Principal,
        run_id: str,
        after_seq: int,
        is_disconnected: DisconnectAware,
    ) -> AsyncIterator[EventEnvelope | None]:
        cursor = after_seq
        elapsed_since_keepalive = 0.0
        while True:
            events = await self.repository.get_events(
                run_id, cursor, principal=principal
            )
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

    def _launch(self, run_id: str) -> None:
        if self._closed:
            raise RuntimeError("run service is closed")
        task = asyncio.create_task(self._execute(run_id), name=f"product-run-{run_id}")
        self._tasks[run_id] = task

        def discard(completed: asyncio.Task[None]) -> None:
            self._tasks.pop(run_id, None)
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

    async def _execute(self, run_id: str) -> None:
        try:
            if not await _finish_repository_call(self.repository.start_run(run_id)):
                return
            thread_id, messages = await _finish_repository_call(
                self.repository.get_run_context(run_id)
            )
            result: AgentResult | None = None
            async with asyncio.timeout(self._settings.run_timeout_seconds):
                async for item in self._runner.stream(
                    thread_id=thread_id,
                    run_id=run_id,
                    messages=messages,
                ):
                    if isinstance(item, ProductEvent):
                        event_type, data = validate_runtime_event(item.type, item.data)
                        persisted = await _finish_repository_call(
                            self.repository.append_active_event(run_id, event_type, data)
                        )
                        if persisted is None:
                            return
                    else:
                        result = item
            if result is None:
                await _finish_repository_call(
                    self.repository.fail_run(run_id, "agent_execution_failed")
                )
                return
            await _finish_repository_call(self.repository.complete_run(run_id, result.text))
        except asyncio.CancelledError:
            # The cancel endpoint has already committed the immutable terminal event.
            return
        except TimeoutError:
            await _finish_repository_call(self.repository.fail_run(run_id, "run_timeout"))
        except Exception:
            # Provider responses and exception text are intentionally not persisted.
            await _finish_repository_call(
                self.repository.fail_run(run_id, "agent_execution_failed")
            )
