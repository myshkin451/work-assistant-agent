from __future__ import annotations

from typing import Any

import pytest

from work_assistant.agent_runtime import (
    AgentResult,
    FakeAgentRunner,
    ProductEvent,
    RuntimeConfigurationError,
    read_current_time,
    runtime_for_settings,
)
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
            message="What time is it in Europe/London?",
        )
    ]
    assert [item.type for item in items if isinstance(item, ProductEvent)][:3] == [
        "tool.started",
        "tool.finished",
        "source.added",
    ]
    result = next(item for item in items if isinstance(item, AgentResult))
    assert "Europe/London" in result.text


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
