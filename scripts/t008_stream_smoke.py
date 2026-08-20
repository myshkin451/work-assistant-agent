#!/usr/bin/env python3
"""Validate first-turn atomicity and redacted live delta timing through Compose."""

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
from concurrent.futures import ThreadPoolExecutor
from typing import Any


BASE_URL = os.environ.get("WORK_ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEV_PRINCIPAL = os.environ.get(
    "WORK_ASSISTANT_DEV_PRINCIPAL",
    "urn:work-assistant:neutral:t008-stream-smoke",
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled"}
PROMPT = (
    "请查询 Asia/Shanghai 当前时间。最终回答请使用安全 Markdown，"
    "包含二级标题、三项列表、一行两列表格、引用、行内代码和 "
    "https://www.iana.org/time-zones 链接，内容控制在 300 至 500 字。"
)


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


def create_initial_run(
    thread_id: str,
    idempotency_key: str,
) -> tuple[int, dict[str, Any]]:
    return request_json(
        "POST",
        f"/api/threads/{urllib.parse.quote(thread_id)}/initial-run",
        {"message": PROMPT, "idempotency_key": idempotency_key},
    )


def consume_events(
    run_id: str,
    *,
    after_seq: int,
    started_at: float,
    stop_after_first_delta: bool,
) -> list[tuple[dict[str, Any], int]]:
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
    received: list[tuple[dict[str, Any], int]] = []
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
            received.append((event, int((time.perf_counter() - started_at) * 1000)))
            if stop_after_first_delta and event.get("type") == "message.delta":
                return received
            if event.get("type") in TERMINAL_EVENTS:
                return received
    return received


def validate_events(
    received: list[tuple[dict[str, Any], int]],
    *,
    minimum_deltas: int,
) -> dict[str, Any]:
    events = [event for event, _ in received]
    seqs = [int(event["seq"]) for event in events]
    if seqs != list(range(1, len(seqs) + 1)):
        raise RuntimeError("event_sequence_invalid")
    if len({(event["run_id"], event["seq"]) for event in events}) != len(events):
        raise RuntimeError("event_deduplication_invalid")

    event_types = [str(event["type"]) for event in events]
    required = {
        "run.started",
        "tool.started",
        "tool.finished",
        "source.added",
        "message.delta",
        "message.completed",
        "run.completed",
    }
    if not required.issubset(event_types) or event_types[-1] != "run.completed":
        raise RuntimeError("event_contract_invalid")

    deltas = [event["data"]["delta"] for event in events if event["type"] == "message.delta"]
    if len(deltas) < minimum_deltas or any(not isinstance(delta, str) or not delta for delta in deltas):
        raise RuntimeError("delta_count_invalid")
    completed = [event for event in events if event["type"] == "message.completed"]
    if len(completed) != 1:
        raise RuntimeError("message_completed_invalid")
    answer = completed[0]["data"]["message"]["content"]
    if "".join(deltas) != answer:
        raise RuntimeError("delta_terminal_mismatch")

    first_delta_ms = next(
        received_ms for event, received_ms in received if event["type"] == "message.delta"
    )
    terminal_ms = next(
        received_ms for event, received_ms in received if event["type"] == "run.completed"
    )
    if first_delta_ms >= terminal_ms:
        raise RuntimeError("delta_not_observed_before_terminal")

    return {
        "event_count": len(events),
        "delta_count": len(deltas),
        "delta_chars": sum(len(delta) for delta in deltas),
        "first_delta_ms": first_delta_ms,
        "terminal_ms": terminal_ms,
        "disconnect_after_first_delta": True,
        "resume_after_seq": True,
        "delta_terminal_exact": True,
        "source_count": event_types.count("source.added"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", default="t008_compose_e3")
    parser.add_argument("--model", default="fake")
    parser.add_argument("--initial-requests", type=int, default=1)
    parser.add_argument("--minimum-deltas", type=int, default=1)
    args = parser.parse_args()

    try:
        validate_base_url()
        if not 1 <= args.initial_requests <= 20:
            raise RuntimeError("initial_request_count_invalid")
        if not 1 <= args.minimum_deltas <= 100:
            raise RuntimeError("minimum_delta_count_invalid")
        wait_until_ready()

        thread_id = str(uuid.uuid4())
        idempotency_key = f"t008-{uuid.uuid4().hex}"
        started_at = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.initial_requests) as executor:
            responses = list(
                executor.map(
                    lambda _: create_initial_run(thread_id, idempotency_key),
                    range(args.initial_requests),
                )
            )
        if {status for status, _ in responses} != {201}:
            raise RuntimeError("initial_run_request_failed")
        run_ids = {str(payload["run"]["run_id"]) for _, payload in responses}
        if len(run_ids) != 1:
            raise RuntimeError("initial_run_idempotency_failed")
        if {str(payload["thread"]["thread_id"]) for _, payload in responses} != {thread_id}:
            raise RuntimeError("initial_thread_id_mismatch")
        run_id = run_ids.pop()

        prefix = consume_events(
            run_id,
            after_seq=0,
            started_at=started_at,
            stop_after_first_delta=True,
        )
        cursor = int(prefix[-1][0]["seq"])
        suffix = consume_events(
            run_id,
            after_seq=cursor,
            started_at=started_at,
            stop_after_first_delta=False,
        )
        summary = validate_events(
            [*prefix, *suffix],
            minimum_deltas=args.minimum_deltas,
        )

        replay_status, replay = create_initial_run(thread_id, idempotency_key)
        if replay_status != 201 or replay["run"]["run_id"] != run_id:
            raise RuntimeError("initial_run_replay_failed")
        rename_status, renamed = request_json(
            "PATCH",
            f"/api/threads/{urllib.parse.quote(thread_id)}",
            {"title": "T-008 streaming evidence"},
        )
        if rename_status != 200 or renamed.get("title") != "T-008 streaming evidence":
            raise RuntimeError("thread_rename_failed")
        detail_status, snapshot = request_json(
            "GET", f"/api/threads/{urllib.parse.quote(thread_id)}"
        )
        if detail_status != 200:
            raise RuntimeError("thread_snapshot_failed")
        roles = [message.get("role") for message in snapshot.get("messages", [])]
        if roles.count("user") != 1 or roles.count("assistant") != 1:
            raise RuntimeError("persisted_message_count_invalid")

        print(
            json.dumps(
                {
                    "lane": args.lane,
                    "status": "passed",
                    "model": args.model,
                    "initial_requests": args.initial_requests,
                    "unique_threads": 1,
                    "unique_runs": 1,
                    "persisted_message_count": len(roles),
                    "rename_persisted": True,
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
