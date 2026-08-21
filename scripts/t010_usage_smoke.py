#!/usr/bin/env python3
"""Prove one redacted, owner-scoped provider usage Run through REST and SSE."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


BASE_URL = os.environ.get("WORK_ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEV_PRINCIPAL = os.environ.get(
    "WORK_ASSISTANT_DEV_PRINCIPAL",
    "urn:work-assistant:neutral:t010-usage-smoke",
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled"}
PROMPT = "请用一句话说明安排今日工作重点时最重要的原则。"


def validate_base_url() -> None:
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("smoke_base_url_must_be_loopback")


def request_json(
    method: str,
    path: str,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Work-Assistant-Dev-Subject": DEV_PRINCIPAL,
        },
    )
    try:
        with OPENER.open(request, timeout=20) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def wait_until_ready() -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            status, _ = request_json("GET", "/health")
            if status == 200:
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise RuntimeError("backend_not_ready")


def consume_events(run_id: str, *, after_seq: int = 0) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        (
            f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events"
            f"?after_seq={after_seq}"
        ),
        headers={
            "Accept": "text/event-stream",
            "X-Work-Assistant-Dev-Subject": DEV_PRINCIPAL,
        },
    )
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []
    with OPENER.open(request, timeout=240) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
                continue
            if line or not data_lines:
                continue
            event = json.loads("\n".join(data_lines))
            data_lines = []
            events.append(event)
            if event.get("type") in TERMINAL_EVENTS:
                return events
    return events


def require_complete_metric(container: dict[str, Any], name: str) -> int:
    metric = container.get(name)
    if not isinstance(metric, dict) or metric.get("availability") != "complete":
        raise RuntimeError(f"{name}_not_complete")
    value = metric.get("value")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{name}_value_invalid")
    return value


def require_unavailable_metric(container: dict[str, Any], name: str) -> None:
    metric = container.get(name)
    if metric != {"value": None, "availability": "unavailable"}:
        raise RuntimeError(f"{name}_must_remain_unavailable")


def require_duration(usage: dict[str, Any], name: str) -> int:
    value = usage.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{name}_invalid")
    return value


def validate_usage(usage: dict[str, Any]) -> dict[str, int]:
    if usage.get("schema_version") != "1.0.0" or usage.get("state") != "final":
        raise RuntimeError("run_usage_state_invalid")
    model_calls = usage.get("model_call_count")
    retries = usage.get("retry_count")
    if not isinstance(model_calls, int) or isinstance(model_calls, bool) or model_calls < 1:
        raise RuntimeError("model_call_count_invalid")
    if retries != 0:
        raise RuntimeError("unexpected_provider_retry")
    input_tokens = require_complete_metric(usage, "input_tokens")
    output_tokens = require_complete_metric(usage, "output_tokens")
    cached_tokens = require_complete_metric(usage, "cached_tokens")
    total_tokens = require_complete_metric(usage, "total_tokens")
    require_unavailable_metric(usage, "reasoning_tokens")
    first_visible_ms = require_duration(usage, "time_to_first_visible_ms")
    generation_ms = require_duration(usage, "generation_duration_ms")
    run_ms = require_duration(usage, "run_duration_ms")
    if first_visible_ms > run_ms or generation_ms > run_ms:
        raise RuntimeError("run_timing_order_invalid")
    if usage.get("error_category") is not None:
        raise RuntimeError("completed_run_error_category_invalid")
    return {
        "model_calls": model_calls,
        "retries": retries,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "first_visible_ms": first_visible_ms,
        "generation_ms": generation_ms,
        "run_ms": run_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", default="t010_deepseek_e4")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        validate_base_url()
        wait_until_ready()
        thread_id = str(uuid.uuid4())
        idempotency_key = f"t010-{uuid.uuid4().hex}"
        body = {"message": PROMPT, "idempotency_key": idempotency_key}
        status, created = request_json(
            "POST",
            f"/api/threads/{urllib.parse.quote(thread_id)}/initial-run",
            body,
        )
        if status != 201:
            raise RuntimeError("initial_run_request_failed")
        run_id = str(created["run"]["run_id"])

        events = consume_events(run_id)
        seqs = [int(event["seq"]) for event in events]
        if seqs != list(range(1, len(seqs) + 1)):
            raise RuntimeError("event_sequence_invalid")
        if not events or events[-1].get("type") != "run.completed":
            raise RuntimeError("run_not_completed")
        deltas = [
            event["data"]["delta"]
            for event in events
            if event.get("type") == "message.delta"
        ]
        completed_messages = [
            event["data"]["message"]["content"]
            for event in events
            if event.get("type") == "message.completed"
        ]
        if not deltas or len(completed_messages) != 1:
            raise RuntimeError("visible_answer_contract_invalid")
        if "".join(deltas) != completed_messages[0]:
            raise RuntimeError("delta_terminal_mismatch")

        terminal_usage = events[-1]["data"].get("usage")
        if not isinstance(terminal_usage, dict):
            raise RuntimeError("terminal_usage_missing")
        summary = validate_usage(terminal_usage)

        detail_status, snapshot = request_json(
            "GET", f"/api/threads/{urllib.parse.quote(thread_id)}"
        )
        if detail_status != 200 or len(snapshot.get("runs", [])) != 1:
            raise RuntimeError("thread_snapshot_invalid")
        rest_usage = snapshot["runs"][0].get("usage")
        if rest_usage != terminal_usage:
            raise RuntimeError("rest_sse_usage_mismatch")

        replay_status, replay = request_json(
            "POST",
            f"/api/threads/{urllib.parse.quote(thread_id)}/initial-run",
            body,
        )
        if replay_status != 201 or replay["run"]["run_id"] != run_id:
            raise RuntimeError("idempotent_replay_invalid")
        replay_events = consume_events(run_id, after_seq=seqs[-1] - 1)
        if len(replay_events) != 1 or replay_events[0]["data"].get("usage") != terminal_usage:
            raise RuntimeError("terminal_usage_replay_invalid")

        account_status, account = request_json(
            "GET",
            "/api/account/usage?range=all&thread_id="
            + urllib.parse.quote(thread_id),
        )
        if account_status != 200:
            raise RuntimeError("account_usage_request_failed")
        if account.get("runs") != {
            "total": 1,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "active": 0,
        }:
            raise RuntimeError("account_run_counts_invalid")
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            if account.get(name) != terminal_usage.get(name):
                raise RuntimeError(f"account_{name}_mismatch")
        if require_complete_metric(account, "model_calls") != summary["model_calls"]:
            raise RuntimeError("account_model_calls_mismatch")
        if require_complete_metric(account, "retries") != summary["retries"]:
            raise RuntimeError("account_retries_mismatch")
        serialized_account = json.dumps(account, ensure_ascii=False).casefold()
        for forbidden in ("subject", "roles", "session_id", "prompt", "tool_call"):
            if forbidden in serialized_account:
                raise RuntimeError("account_response_contains_private_field")

        print(
            json.dumps(
                {
                    "lane": args.lane,
                    "status": "passed",
                    "model": args.model,
                    "event_count": len(events),
                    "delta_count": len(deltas),
                    "rest_sse_replay_exact": True,
                    "owner_scoped_account_exact": True,
                    "reasoning_tokens": "unavailable",
                    **summary,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - emit only a bounded classification.
        print(json.dumps({"status": "failed", "classification": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
