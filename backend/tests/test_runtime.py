from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from work_assistant.agent_runtime import (
    AgentResult,
    FakeAgentRunner,
    ProductEvent,
    RuntimeConfigurationError,
    enforce_single_time_tool_call,
    read_current_time,
    runtime_for_settings,
)
from work_assistant.schemas import Message
from work_assistant.settings import Settings


def test_time_tool_validates_iana_timezone() -> None:
    result = read_current_time("Asia/Shanghai")
    assert result["timezone"] == "Asia/Shanghai"
    assert result["source_id"] == "system-clock-iana-tzdb"
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        read_current_time("Not/A_Real_Zone")


async def test_fake_runner_uses_public_tool_lifecycle() -> None:
    runner = FakeAgentRunner(step_delay_seconds=0)
    items: list[Any] = [
        item
        async for item in runner.stream(
            thread_id="thread",
            run_id="run",
            messages=[
                Message(
                    message_id="message",
                    role="user",
                    content="What time is it in Europe/London?",
                    created_at=datetime.now(UTC),
                    run_id="run",
                )
            ],
        )
    ]
    assert [item.type for item in items if isinstance(item, ProductEvent)][:3] == [
        "tool.started",
        "tool.finished",
        "source.added",
    ]
    result = next(item for item in items if isinstance(item, AgentResult))
    assert "Europe/London" in result.text


async def test_fake_runner_supports_the_chinese_multiturn_acceptance_prompts() -> None:
    runner = FakeAgentRunner(step_delay_seconds=0)
    messages: list[Message] = []
    expected = ["Asia/Shanghai", "Europe/London", "America/New_York"]
    for index, prompt in enumerate(("请查询当前上海时间。", "那伦敦呢？", "再看看纽约。")):
        messages.append(
            Message(
                message_id=f"message-{index}",
                role="user",
                content=prompt,
                created_at=datetime.now(UTC),
                run_id=f"run-{index}",
            )
        )
        items = [
            item
            async for item in runner.stream(
                thread_id="thread",
                run_id=f"run-{index}",
                messages=messages,
            )
        ]
        started = next(
            item
            for item in items
            if isinstance(item, ProductEvent) and item.type == "tool.started"
        )
        assert started.data["input_summary"] == expected[index]


@pytest.mark.parametrize(
    ("messages", "expected_choice"),
    [
        ([HumanMessage(content="What time is it?")], "required"),
        (
            [
                HumanMessage(content="What time is it?"),
                ToolMessage(content="{}", tool_call_id="time-1"),
            ],
            "none",
        ),
    ],
)
async def test_time_agent_enforces_exactly_one_tool_choice(
    messages: list[HumanMessage | ToolMessage], expected_choice: str
) -> None:
    choices: list[Any] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        choices.append(request.tool_choice)
        return ModelResponse(result=[AIMessage(content="done")])

    request = ModelRequest(model=cast(Any, object()), messages=messages)
    await enforce_single_time_tool_call.awrap_model_call(request, handler)
    assert choices == [expected_choice]


async def test_deepseek_mode_fails_closed_without_key() -> None:
    settings = Settings(
        model_mode="deepseek",
        deepseek_api_key=None,
        checkpoint_database_url="postgresql://unused:unused@localhost/unused",
    )
    with pytest.raises(RuntimeConfigurationError, match="deepseek_api_key_missing"):
        # Validate before any checkpoint connection by constructing the runner path directly.
        from work_assistant.agent_runtime import DeepSeekAgentRunner

        DeepSeekAgentRunner(settings=settings, checkpointer=None)  # type: ignore[arg-type]

    # The default fake lane never needs or reads a model credential.
    fake = Settings(model_mode="fake", deepseek_api_key=None)
    async with runtime_for_settings(fake) as runner:
        assert isinstance(runner, FakeAgentRunner)
