from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from weakref import WeakSet

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError

from .agent_definition import ModelProfile
from .capabilities import TIME_SOURCE, get_current_time, read_current_time
from .context_builder import BuiltContext
from .execution_policy import (
    AgentExecutionFailed,
    AgentPolicyKernel,
    AgentResult,
    ProductEvent,
    ResultSchemaInvalid,
    RunExecution,
    execute_tool_call,
)
from .schemas import Message
from .settings import Settings


class RuntimeConfigurationError(Exception):
    pass


class RuntimeFatalError(RuntimeError):
    """The current Runtime instance cannot safely execute another Run."""


class RuntimeCleanupTimeout(RuntimeFatalError):
    """A cancelled Runtime producer did not settle within its cleanup grace."""


class RuntimeUnavailable(RuntimeFatalError):
    """A previous fatal cleanup failure made this Runtime instance unusable."""


RuntimeItem = ProductEvent | AgentResult


class AgentRunner(Protocol):
    def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[RuntimeItem]: ...


def _timezone_from_message(message: str) -> str:
    match = re.search(r"\b[A-Za-z]+(?:[_+-][A-Za-z]+)*/[A-Za-z0-9_+\-]+\b", message)
    if match:
        return match.group(0)
    lowered = message.casefold()
    if "shanghai" in lowered or "china" in lowered or "上海" in message or "北京" in message:
        return "Asia/Shanghai"
    if "new york" in lowered or "纽约" in message:
        return "America/New_York"
    if "london" in lowered or "伦敦" in message:
        return "Europe/London"
    return "UTC"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _terminal_assistant_text(messages: Sequence[BaseMessage]) -> str:
    if not messages:
        return ""
    message = messages[-1]
    if not isinstance(message, AIMessage) or message.tool_calls:
        return ""
    return _content_text(message.content)


def _terminal_state_text(state: dict[str, Any] | None) -> str:
    if not state:
        return ""
    messages = state.get("messages", [])
    if not isinstance(messages, Sequence):
        return ""
    return _terminal_assistant_text(cast(Sequence[BaseMessage], messages))


def _tool_call(*, call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "args": arguments,
        "type": "tool_call",
    }


class FakeAgentRunner:
    """Replaceable deterministic adapter that exercises the real Host policy ledger."""

    def __init__(self, *, step_delay_seconds: float = 0.02) -> None:
        self._delay = step_delay_seconds

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[RuntimeItem]:
        del thread_id, run_id, built_context
        if not messages:
            raise RuntimeError("conversation_context_missing")
        message = messages[-1].content

        if "[policy:deadline]" in message:
            execution.before_model_call()
            await asyncio.sleep(execution.agent.budget.deadline_seconds + 0.05)
            raise AssertionError("deadline guard did not cancel the fake adapter")

        if "[policy:model-step-limit]" in message:
            for index in range(execution.agent.budget.max_model_steps + 1):
                execution.before_model_call()
                execution.after_model_response([AIMessage(content=f"bounded-progress-{index}")])
            raise AssertionError("model-step guard did not stop the fake adapter")

        if "[policy:no-progress]" in message:
            for _ in range(execution.agent.budget.max_no_progress_steps + 2):
                execution.before_model_call()
                execution.after_model_response([AIMessage(content="")])
            raise AssertionError("no-progress guard did not stop the fake adapter")

        if "[policy:tool-call-limit]" in message:
            execution.before_model_call()
            calls = [
                _tool_call(
                    call_id=f"time-limit-{index}",
                    name="get_current_time",
                    arguments={"timezone": f"Etc/GMT+{index}"},
                )
                for index in range(execution.agent.budget.max_tool_calls + 1)
            ]
            execution.after_model_response([AIMessage(content="", tool_calls=calls)])
            raise AssertionError("Tool-call guard did not stop the fake adapter")

        if "[policy:tool-denied]" in message:
            execution.before_model_call()
            execution.after_model_response(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call(
                                call_id="denied-1",
                                name="unregistered_tool",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
            raise AssertionError("capability guard did not stop the fake adapter")

        is_direct = "[policy:direct]" in message or not any(
            token in message.casefold()
            for token in (
                "time",
                "时",
                "london",
                "shanghai",
                "new york",
                "上海",
                "北京",
                "伦敦",
                "纽约",
            )
        )
        if is_direct:
            execution.before_model_call()
            text = "This request can be answered without an external Tool."
            execution.after_model_response([AIMessage(content=text)])
            for delta in _split_text(text):
                await asyncio.sleep(self._delay)
                yield ProductEvent("message.delta", {"delta": delta})
            yield AgentResult(text=text, source_ids=())
            return

        timezone_name = _timezone_from_message(message)
        call_id = "time-1"
        first_call = _tool_call(
            call_id=call_id,
            name="get_current_time",
            arguments={"timezone": timezone_name},
        )
        execution.before_model_call()
        execution.after_model_response([AIMessage(content="", tool_calls=[first_call])])
        emitted: list[ProductEvent] = []

        async def emit(event: ProductEvent) -> None:
            emitted.append(event)

        async def invoke(validated: dict[str, Any]) -> ToolMessage:
            await asyncio.sleep(self._delay)
            record = execution.tool_registry.require("get_current_time")
            output = await record.implementation.ainvoke(validated)
            if not isinstance(output, str):
                raise AgentExecutionFailed("tool_result_type_invalid")
            return ToolMessage(
                content=output,
                tool_call_id=call_id,
                name="get_current_time",
            )

        tool_message = await execute_tool_call(
            execution=execution,
            tool_call_id=call_id,
            tool_id="get_current_time",
            arguments={"timezone": timezone_name},
            handler=invoke,
            emit=emit,
        )
        for event in emitted:
            yield event

        if "[policy:repeat-tool]" in message:
            execution.before_model_call()
            execution.after_model_response(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call(
                                call_id="time-2",
                                name="get_current_time",
                                arguments={"timezone": timezone_name},
                            )
                        ],
                    )
                ]
            )
            raise AssertionError("repeat-call guard did not stop the fake adapter")

        try:
            result = json.loads(_content_text(tool_message.content))
        except (TypeError, json.JSONDecodeError) as exc:
            raise AgentExecutionFailed("tool_output_invalid") from exc
        if not isinstance(result, dict):
            raise AgentExecutionFailed("tool_output_invalid")
        text = (
            f"The current time in {result['timezone']} is {result['local_time']} "
            f"(UTC offset {result['utc_offset']})."
        )
        execution.before_model_call()
        execution.after_model_response([AIMessage(content=text)])
        for delta in _split_text(text):
            await asyncio.sleep(self._delay)
            yield ProductEvent("message.delta", {"delta": delta})

        if "[policy:source-missing]" in message:
            source_ids: tuple[str, ...] = ()
        elif "[policy:source-invalid]" in message:
            source_ids = ("forged-source",)
        else:
            source_ids = execution.generated_source_ids
        yield AgentResult(text=text, source_ids=source_ids)


def _split_text(text: str) -> tuple[str, str]:
    split_at = max(1, len(text) // 2)
    return text[:split_at], text[split_at:]


def _message_delta_events(text: str) -> list[tuple[str, dict[str, str]]]:
    chunk_size = 8_000
    return [
        ("message.delta", {"delta": text[offset : offset + chunk_size]})
        for offset in range(0, len(text), chunk_size)
    ]


class _RuntimeLifecycle:
    """Own one Runtime's cancellation, quarantine, and permanent fail-stop state."""

    def __init__(self, *, cleanup_grace_seconds: float) -> None:
        self._cleanup_grace_seconds = cleanup_grace_seconds
        self._quarantined: set[asyncio.Future[Any]] = set()
        self._observed: WeakSet[asyncio.Future[Any]] = WeakSet()
        self._fatal_error: str | None = None

    @property
    def is_healthy(self) -> bool:
        return self._fatal_error is None

    def require_healthy(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeUnavailable(self._fatal_error)

    def _mark_fatal(self) -> None:
        if self._fatal_error is None:
            self._fatal_error = "runtime_cleanup_timeout"

    def _observe(self, future: asyncio.Future[Any]) -> tuple[asyncio.Future[Any], ...]:
        """Consume one terminal result and discover LangGraph cleanup futures."""

        if future in self._observed:
            return ()
        self._observed.add(future)
        nested: tuple[asyncio.Future[Any], ...] = ()
        try:
            future.result()
        except asyncio.CancelledError as exc:
            nested = tuple(
                value
                for value in exc.args
                if isinstance(value, asyncio.Future) and value is not future
            )
        except Exception:
            # Runtime exceptions are delivered through the bounded queue.
            pass
        return nested

    def _quarantine(self, futures: Sequence[asyncio.Future[Any]]) -> None:
        for future in futures:
            if future.done():
                nested = self._observe(future)
                if nested:
                    self._quarantine(nested)
                continue
            if future in self._quarantined:
                continue
            self._quarantined.add(future)
            future.add_done_callback(self._observe_quarantined)

    def _observe_quarantined(self, future: asyncio.Future[Any]) -> None:
        self._quarantined.discard(future)
        nested = self._observe(future)
        if nested:
            self._quarantine(nested)

    async def finish_producer(
        self,
        producer: asyncio.Task[None],
        *,
        cancel: bool = True,
    ) -> None:
        """Cancel once, then observe producer and nested cleanup without recancelling."""

        if cancel and not producer.done() and producer.cancelling() == 0:
            producer.cancel()
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + self._cleanup_grace_seconds
        pending: set[asyncio.Future[Any]] = {producer}
        while pending:
            remaining = deadline_at - loop.time()
            if remaining <= 0:
                self._mark_fatal()
                self._quarantine(tuple(pending))
                raise RuntimeCleanupTimeout("runtime_cleanup_timeout")
            try:
                done, still_pending = await asyncio.wait(pending, timeout=remaining)
            except asyncio.CancelledError:
                # A second parent cancellation must not reach the Runtime child.
                # Quarantine it and force process-level fail-stop instead.
                self._mark_fatal()
                self._quarantine(tuple(pending))
                raise RuntimeCleanupTimeout("runtime_cleanup_timeout") from None
            pending = set(still_pending)
            for future in done:
                pending.update(self._observe(future))
            if pending and loop.time() >= deadline_at:
                self._mark_fatal()
                self._quarantine(tuple(pending))
                raise RuntimeCleanupTimeout("runtime_cleanup_timeout")

    async def shutdown(self) -> None:
        """Observe quarantined cleanup for one grace without another cancellation."""

        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + self._cleanup_grace_seconds
        while self._quarantined:
            remaining = deadline_at - loop.time()
            if remaining <= 0:
                return
            await asyncio.wait(tuple(self._quarantined), timeout=remaining)


@dataclass
class DeepSeekRuntimeContext:
    execution: RunExecution = field(repr=False)
    built_context: BuiltContext = field(repr=False)
    emit: Callable[[ProductEvent], Awaitable[None]] = field(repr=False)
    final_response_text: str | None = field(default=None, repr=False)


@wrap_model_call
async def apply_model_policy(
    request: ModelRequest[DeepSeekRuntimeContext],
    handler: Callable[[ModelRequest[DeepSeekRuntimeContext]], Awaitable[ModelResponse[Any]]],
) -> ModelResponse[Any]:
    context = request.runtime.context
    context.execution.before_model_call()
    visible = set(context.execution.visible_tool_ids)
    tools = [tool for tool in request.tools if getattr(tool, "name", None) in visible]
    response = await handler(
        request.override(
            tools=tools,
            tool_choice="auto" if tools else "none",
            system_message=SystemMessage(content=context.built_context.system_prompt),
        )
    )
    context.execution.after_model_response(response.result)
    terminal_message = response.result[-1] if response.result else None
    if isinstance(terminal_message, AIMessage) and not terminal_message.tool_calls:
        final_text = _content_text(terminal_message.content)
        if not final_text.strip():
            raise ResultSchemaInvalid
        if len(final_text) > context.execution.agent.result_contract.max_answer_chars:
            raise ResultSchemaInvalid
        if context.final_response_text is not None:
            raise AgentExecutionFailed("multiple_terminal_model_responses")
        context.final_response_text = final_text
        for event_type, data in _message_delta_events(final_text):
            await context.emit(ProductEvent(event_type, data))
    return response


@wrap_tool_call
async def apply_tool_policy(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Any]],
) -> ToolMessage | Any:
    context = request.runtime.context
    if not isinstance(context, DeepSeekRuntimeContext):
        raise AgentExecutionFailed("runtime_context_invalid")
    tool_call_id = request.tool_call.get("id")
    tool_id = request.tool_call.get("name")
    arguments = request.tool_call.get("args")
    if (
        not isinstance(tool_call_id, str)
        or not isinstance(tool_id, str)
        or not isinstance(arguments, dict)
    ):
        raise AgentExecutionFailed("tool_call_invalid")

    async def invoke(validated: dict[str, Any]) -> ToolMessage:
        modified = {**request.tool_call, "args": validated}
        result = await handler(request.override(tool_call=modified))
        if not isinstance(result, ToolMessage):
            raise AgentExecutionFailed("tool_result_type_invalid")
        return result

    return await execute_tool_call(
        execution=context.execution,
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        arguments=arguments,
        handler=invoke,
        emit=context.emit,
    )


class DeepSeekAgentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        checkpointer: BaseCheckpointSaver[Any],
        policy_kernel: AgentPolicyKernel,
    ) -> None:
        validate_runtime_configuration(settings)
        model = ChatDeepSeek(  # type: ignore[call-arg]  # upstream stub uses model_name
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            timeout=120,
            max_retries=0,
            streaming=True,
            stream_usage=True,
            extra_body={"thinking": {"type": "disabled"}},
        )
        self._agent = create_agent(
            model=model,
            tools=policy_kernel.tool_registry.enabled_implementations,
            middleware=[apply_model_policy, cast(Any, apply_tool_policy)],
            context_schema=DeepSeekRuntimeContext,
            checkpointer=checkpointer,
            name="work_assistant_agent_runtime",
        )
        self._semaphore = asyncio.Semaphore(settings.model_concurrency)
        self._recursion_limit = policy_kernel.framework_recursion_limit
        self._lifecycle = _RuntimeLifecycle(
            cleanup_grace_seconds=settings.repository_cleanup_grace_seconds
        )

    @property
    def is_healthy(self) -> bool:
        return self._lifecycle.is_healthy

    async def shutdown(self) -> None:
        await self._lifecycle.shutdown()

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[RuntimeItem]:
        del thread_id
        self._lifecycle.require_healthy()
        if not messages:
            raise RuntimeError("conversation_context_missing")
        # Every model response and Tool call is already business-bounded. This
        # capacity holds the maximum 3 events per reserved Tool, four answer
        # chunks, one exception, and the sentinel without unbounded buffering.
        queue_capacity = execution.agent.budget.max_tool_calls * 3 + 6
        queue: asyncio.Queue[RuntimeItem | Exception | None] = asyncio.Queue(maxsize=queue_capacity)

        async def emit(event: ProductEvent) -> None:
            await queue.put(event)

        context = DeepSeekRuntimeContext(
            execution=execution,
            built_context=built_context,
            emit=emit,
        )

        async def produce() -> None:
            latest_state: dict[str, Any] | None = None
            try:
                async with self._semaphore:
                    async for item in self._agent.astream(
                        {
                            "messages": [
                                {"role": message.role, "content": message.content}
                                for message in built_context.conversation
                            ]
                        },
                        config={
                            # Runtime checkpoints remain isolated per product Run.
                            "configurable": {"thread_id": run_id},
                            "recursion_limit": self._recursion_limit,
                        },
                        context=context,
                        stream_mode="values",
                    ):
                        if isinstance(item, dict):
                            latest_state = item
                final_text = _terminal_state_text(latest_state)
                if not final_text or context.final_response_text != final_text:
                    raise AgentExecutionFailed("assistant_message_missing")
                await queue.put(
                    AgentResult(
                        text=final_text,
                        source_ids=execution.generated_source_ids,
                    )
                )
            except GraphRecursionError:
                await queue.put(AgentExecutionFailed("runtime_safety_limit"))
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce(), name=f"deepseek-agent-{run_id}")
        producer_reached_sentinel = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    producer_reached_sentinel = True
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # A normal sentinel means producer work reached its own final
            # boundary; observe it without injecting an unnecessary cancel.
            # Early consumer exit or Run cancellation still cancels exactly once.
            await self._lifecycle.finish_producer(
                producer,
                cancel=not producer_reached_sentinel,
            )


def model_profile_for_settings(settings: Settings) -> ModelProfile:
    if settings.model_mode == "fake":
        return ModelProfile(
            profile_id="default",
            version="1.0.0",
            provider="local-fake",
            model_id="deterministic-fake-v1",
        )
    return ModelProfile(
        profile_id="default",
        version="1.0.0",
        provider="deepseek",
        model_id=settings.deepseek_model,
    )


def validate_runtime_configuration(settings: Settings) -> None:
    if settings.model_mode == "deepseek" and (
        settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value()
    ):
        raise RuntimeConfigurationError("deepseek_api_key_missing")


@asynccontextmanager
async def runtime_for_settings(
    settings: Settings,
    *,
    policy_kernel: AgentPolicyKernel,
) -> AsyncIterator[AgentRunner]:
    validate_runtime_configuration(settings)
    if settings.model_mode == "fake":
        yield FakeAgentRunner(step_delay_seconds=settings.fake_step_delay_seconds)
        return
    serde = JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=None,
    )
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url,
        serde=serde,
    ) as checkpointer:
        # LangGraph owns these tables. Product Alembic migrations never modify them.
        await checkpointer.setup()
        runner = DeepSeekAgentRunner(
            settings=settings,
            checkpointer=checkpointer,
            policy_kernel=policy_kernel,
        )
        try:
            yield runner
        finally:
            await runner.shutdown()


__all__ = [
    "AgentResult",
    "AgentRunner",
    "DeepSeekAgentRunner",
    "FakeAgentRunner",
    "ProductEvent",
    "RuntimeConfigurationError",
    "RuntimeCleanupTimeout",
    "RuntimeFatalError",
    "RuntimeUnavailable",
    "TIME_SOURCE",
    "apply_model_policy",
    "apply_tool_policy",
    "get_current_time",
    "model_profile_for_settings",
    "read_current_time",
    "runtime_for_settings",
    "validate_runtime_configuration",
]
