from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from .agent_runtime import AgentResult, AgentRunner, ProductEvent
from .repository import ProductRepository
from .schemas import EventEnvelope, RunView
from .settings import Settings


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

    async def create_run(self, *, thread_id: str, message: str, idempotency_key: str) -> RunView:
        view, created = await self.repository.create_run(
            thread_id=thread_id,
            message=message,
            idempotency_key=idempotency_key,
        )
        if created:
            self._launch(view.run_id)
        return view

    async def cancel_run(self, run_id: str) -> RunView:
        view = await self.repository.cancel_run(run_id)
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        return view

    async def stream_events(
        self,
        *,
        run_id: str,
        after_seq: int,
        is_disconnected: DisconnectAware,
    ) -> AsyncIterator[EventEnvelope | None]:
        cursor = after_seq
        elapsed_since_keepalive = 0.0
        while True:
            events = await self.repository.get_events(run_id, cursor)
            for event in events:
                cursor = event.seq
                elapsed_since_keepalive = 0.0
                yield event
            run = await self.repository.get_run(run_id)
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
            if not task.done():
                task.cancel()
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

    async def _execute(self, run_id: str) -> None:
        try:
            if not await self.repository.start_run(run_id):
                return
            thread_id, user_message = await self.repository.get_run_input(run_id)
            result: AgentResult | None = None
            async with asyncio.timeout(self._settings.run_timeout_seconds):
                async for item in self._runner.stream(
                    thread_id=thread_id,
                    run_id=run_id,
                    message=user_message,
                ):
                    if isinstance(item, ProductEvent):
                        persisted = await self.repository.append_active_event(
                            run_id, item.type, item.data
                        )
                        if persisted is None:
                            return
                    else:
                        result = item
            if result is None:
                await self.repository.fail_run(run_id, "agent_result_missing")
                return
            await self.repository.complete_run(run_id, result.text)
        except asyncio.CancelledError:
            # The cancel endpoint has already committed the immutable terminal event.
            return
        except TimeoutError:
            await self.repository.fail_run(run_id, "run_timeout")
        except Exception:
            # Provider responses and exception text are intentionally not persisted.
            await self.repository.fail_run(run_id, "agent_execution_failed")
