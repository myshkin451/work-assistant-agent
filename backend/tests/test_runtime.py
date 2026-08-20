from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langgraph.runtime import Runtime
from policy_fixtures import make_execution, make_settings

from work_assistant.agent_runtime import (
    AgentResult,
    DeepSeekAgentRunner,
    DeepSeekRuntimeContext,
    FakeAgentRunner,
    ProductEvent,
    RuntimeCleanupTimeout,
    RuntimeConfigurationError,
    RuntimeUnavailable,
    _RuntimeLifecycle,
    apply_model_policy,
    get_current_time,
    read_current_time,
    runtime_for_settings,
    validate_runtime_configuration,
)
from work_assistant.bootstrap import build_policy_kernel
from work_assistant.context_builder import BuiltContext
from work_assistant.identity import Principal
from work_assistant.schemas import Message

TEST_PRINCIPAL = Principal(subject="neutral-runtime-principal")


class ToolFreeStreamingModel:
    def __init__(self) -> None:
        self.received_messages: list[tuple[BaseMessage, ...]] = []
        self.model_kwargs: list[dict[str, Any]] = []

    async def astream(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        self.received_messages.append(tuple(messages))
        self.model_kwargs.append(dict(kwargs))
        yield AIMessageChunk(content="safe ")
        yield AIMessageChunk(content="direct answer")


class BurstAgentGraph:
    async def astream(
        self,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any],
        context: DeepSeekRuntimeContext,
        stream_mode: str,
    ) -> AsyncIterator[dict[str, Any]]:
        del inputs, config, stream_mode
        for _ in range(64):
            await context.emit(ProductEvent("message.delta", {"delta": "x"}))
        context.final_response_text = "x" * 64
        yield {"messages": [AIMessage(content=context.final_response_text)]}


def message(content: str, run_id: str = "run") -> Message:
    return Message(
        message_id=f"message-{run_id}",
        role="user",
        content=content,
        created_at=datetime.now(UTC),
        run_id=run_id,
    )


async def collect_fake(
    runner: FakeAgentRunner,
    *,
    messages: list[Message],
    run_id: str,
) -> tuple[list[ProductEvent], AgentResult]:
    execution = make_execution(TEST_PRINCIPAL)
    built = execution.build_context(messages)
    events: list[ProductEvent] = []
    result: AgentResult | None = None
    async for item in runner.stream(
        thread_id="thread",
        run_id=run_id,
        messages=messages,
        execution=execution,
        built_context=built,
    ):
        if isinstance(item, ProductEvent):
            execution.accept_runtime_event(item)
            events.append(item)
        else:
            result = execution.validate_result(item)
    assert result is not None
    return events, result


def test_time_tool_validates_iana_timezone() -> None:
    result = read_current_time("Asia/Shanghai")
    assert result["timezone"] == "Asia/Shanghai"
    assert result["source_id"] == "system-clock-iana-tzdb"
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        read_current_time("Not/A_Real_Zone")


async def test_fake_runner_uses_policy_guard_and_public_tool_lifecycle() -> None:
    events, result = await collect_fake(
        FakeAgentRunner(step_delay_seconds=0),
        messages=[message("What time is it in Europe/London?")],
        run_id="run",
    )
    assert [event.type for event in events][:3] == [
        "tool.started",
        "tool.finished",
        "source.added",
    ]
    assert "Europe/London" in result.text
    assert result.source_ids == ("system-clock-iana-tzdb",)


async def test_fake_runner_supports_the_chinese_multiturn_acceptance_prompts() -> None:
    runner = FakeAgentRunner(step_delay_seconds=0)
    messages: list[Message] = []
    expected = ["Asia/Shanghai", "Europe/London", "America/New_York"]
    for index, prompt in enumerate(("请查询当前上海时间。", "那伦敦呢？", "再看看纽约。")):
        messages.append(message(prompt, f"run-{index}"))
        events, _ = await collect_fake(
            runner,
            messages=messages,
            run_id=f"run-{index}",
        )
        started = next(event for event in events if event.type == "tool.started")
        assert started.data["input_summary"] == expected[index]


async def test_model_hook_filters_tools_and_replaces_the_full_system_message() -> None:
    @tool
    def forbidden_probe() -> str:
        """A Tool that must not be visible for this Run."""

        return "forbidden"

    execution = make_execution(TEST_PRINCIPAL)
    built = execution.build_context([message("Answer directly")])
    emitted: list[ProductEvent] = []

    async def emit(event: ProductEvent) -> None:
        emitted.append(event)

    context = DeepSeekRuntimeContext(
        execution=execution,
        built_context=built,
        emit=emit,
    )
    observed: dict[str, Any] = {}
    model = ToolFreeStreamingModel()

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        observed["tools"] = [getattr(item, "name", None) for item in request.tools]
        observed["system"] = request.system_message.text if request.system_message else None
        observed["tool_choice"] = request.tool_choice
        return ModelResponse(result=[AIMessage(content="safe direct answer")])

    request = ModelRequest(
        model=cast(Any, model),
        messages=[],
        tools=[get_current_time, forbidden_probe],
        runtime=Runtime(context=context),
    )
    await apply_model_policy.awrap_model_call(request, handler)
    assert observed == {
        "tools": ["get_current_time"],
        "system": built.system_prompt,
        "tool_choice": "auto",
    }
    assert model.model_kwargs == [{}]
    assert len(model.received_messages) == 1
    assert isinstance(model.received_messages[0][0], SystemMessage)
    assert model.received_messages[0][0].content == built.system_prompt
    assert [event.data for event in emitted] == [
        {"delta": "safe "},
        {"delta": "direct answer"},
    ]
    assert "".join(cast(str, event.data["delta"]) for event in emitted) == "safe direct answer"
    assert execution.usage().model_steps == 2


async def test_runtime_configuration_fails_closed_before_checkpoint_work() -> None:
    settings = make_settings(model_mode="deepseek", deepseek_api_key=None)
    with pytest.raises(RuntimeConfigurationError, match="deepseek_api_key_missing"):
        validate_runtime_configuration(settings)

    fake = make_settings(model_mode="fake", deepseek_api_key=None)
    kernel = build_policy_kernel(fake)
    async with runtime_for_settings(fake, policy_kernel=kernel) as runner:
        assert isinstance(runner, FakeAgentRunner)


def test_context_type_remains_strict_and_does_not_need_principal_metadata() -> None:
    execution = make_execution(TEST_PRINCIPAL)
    built = execution.build_context([message("hello")])
    assert isinstance(built, BuiltContext)
    assert TEST_PRINCIPAL.subject not in built.system_prompt


async def test_runtime_cleanup_timeout_cancels_once_and_permanently_fails_closed() -> None:
    lifecycle = _RuntimeLifecycle(cleanup_grace_seconds=0.01)
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    cleanup_release = asyncio.Event()
    cancel_count = 0

    async def stuck_producer() -> None:
        nonlocal cancel_count
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_count += 1
            cleanup_entered.set()
            await cleanup_release.wait()
            raise RuntimeError("late-runtime-cleanup") from None

    producer = asyncio.create_task(stuck_producer())
    await entered.wait()

    with pytest.raises(RuntimeCleanupTimeout, match="runtime_cleanup_timeout"):
        await lifecycle.finish_producer(producer)
    await cleanup_entered.wait()

    assert cancel_count == 1
    assert lifecycle.is_healthy is False
    with pytest.raises(RuntimeUnavailable, match="runtime_cleanup_timeout"):
        lifecycle.require_healthy()

    # Shutdown observes only for the same finite grace and returns while the
    # already-cancelled producer is still quarantined.
    await asyncio.wait_for(lifecycle.shutdown(), timeout=0.2)
    assert cancel_count == 1
    assert producer.done() is False
    cleanup_release.set()
    done, pending = await asyncio.wait({producer}, timeout=1)
    assert done == {producer}
    assert pending == set()
    await asyncio.sleep(0)

    assert cancel_count == 1
    assert producer.done()
    assert getattr(producer, "_log_traceback", True) is False


async def test_runtime_normal_completion_is_observed_without_cancellation() -> None:
    lifecycle = _RuntimeLifecycle(cleanup_grace_seconds=0.1)

    async def completed_producer() -> None:
        await asyncio.sleep(0)

    producer = asyncio.create_task(completed_producer())
    await lifecycle.finish_producer(producer, cancel=False)

    assert producer.done()
    assert producer.cancelling() == 0
    assert lifecycle.is_healthy is True


async def test_stream_consumer_exit_does_not_deadlock_a_full_runtime_queue() -> None:
    execution = make_execution(TEST_PRINCIPAL)
    messages = [message("Answer directly")]
    runner = object.__new__(DeepSeekAgentRunner)
    mutable_runner = cast(Any, runner)
    mutable_runner._agent = BurstAgentGraph()
    mutable_runner._semaphore = asyncio.Semaphore(1)
    mutable_runner._recursion_limit = 8
    mutable_runner._lifecycle = _RuntimeLifecycle(cleanup_grace_seconds=0.05)

    stream = runner.stream(
        thread_id="thread",
        run_id="queue-cancel-run",
        messages=messages,
        execution=execution,
        built_context=execution.build_context(messages),
    )
    first = await anext(stream)
    assert first == ProductEvent("message.delta", {"delta": "x"})
    await asyncio.sleep(0.01)

    await asyncio.wait_for(stream.aclose(), timeout=0.2)
    assert runner.is_healthy is True


async def test_runtime_cleanup_bounds_langgraph_nested_cleanup_without_cancelling_it() -> None:
    lifecycle = _RuntimeLifecycle(cleanup_grace_seconds=0.01)
    entered = asyncio.Event()
    nested_cleanup: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def producer_with_nested_cleanup() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise asyncio.CancelledError(nested_cleanup) from None

    producer = asyncio.create_task(producer_with_nested_cleanup())
    await entered.wait()

    with pytest.raises(RuntimeCleanupTimeout, match="runtime_cleanup_timeout"):
        await lifecycle.finish_producer(producer)

    assert producer.cancelling() == 1
    assert nested_cleanup.cancelled() is False
    nested_cleanup.set_result(None)
    await lifecycle.shutdown()
    await asyncio.sleep(0)
    assert nested_cleanup.cancelled() is False
