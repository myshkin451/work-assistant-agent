from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .agent_runtime import (
    AgentResult,
    ProductEvent,
    RuntimeConfigurationError,
    runtime_for_settings,
)
from .bootstrap import build_policy_kernel
from .identity import Principal
from .schemas import Message
from .settings import Settings


async def _run(env_file: Path | None, timezone: str) -> int:
    started = time.perf_counter()
    try:
        settings_input: Any = {"model_mode": "deepseek"}
        if env_file is not None:
            settings_input["_env_file"] = env_file
        settings = Settings(**settings_input)
        event_types: list[str] = []
        result: AgentResult | None = None
        kernel = build_policy_kernel(settings)
        execution = kernel.prepare_run(
            principal=Principal(subject="urn:work-assistant:principal:live-smoke")
        )
        async with runtime_for_settings(settings, policy_kernel=kernel) as runner:
            run_id = f"live-smoke-{uuid4()}"
            messages = [
                Message(
                    message_id=f"live-smoke-{uuid4()}",
                    role="user",
                    content=f"What is the current time in {timezone}?",
                    created_at=datetime.now(UTC),
                    run_id=run_id,
                )
            ]
            built_context = execution.build_context(messages)
            async for item in runner.stream(
                thread_id=f"live-smoke-{uuid4()}",
                run_id=run_id,
                messages=messages,
                execution=execution,
                built_context=built_context,
            ):
                if isinstance(item, ProductEvent):
                    execution.accept_runtime_event(item)
                    event_types.append(item.type)
                else:
                    result = execution.validate_result(item)
        if result is None:
            raise RuntimeError("result_missing")
        summary = {
            "lane": "deepseek_live_smoke",
            "status": "passed",
            "model": settings.deepseek_model,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "event_types": sorted(set(event_types)),
            "message_delta_events": event_types.count("message.delta"),
            "tool_calls": event_types.count("tool.started"),
            "answer_chars": len(result.text),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except (ValidationError, RuntimeConfigurationError):
        print(
            json.dumps(
                {
                    "lane": "deepseek_live_smoke",
                    "status": "blocked",
                    "error_code": "configuration_invalid",
                }
            )
        )
        return 2
    except Exception:
        print(
            json.dumps(
                {
                    "lane": "deepseek_live_smoke",
                    "status": "failed",
                    "error_code": "live_smoke_failed",
                }
            )
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an explicit, redacted DeepSeek smoke test")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    args = parser.parse_args()
    return asyncio.run(_run(args.env_file, args.timezone))


if __name__ == "__main__":
    raise SystemExit(main())
