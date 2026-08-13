#!/usr/bin/env python3
"""Exercise phase-2 persistence, reconnect, restart, and multi-turn contracts."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    "urn:work-assistant:neutral:phase2-smoke",
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled"}
PROMPTS = ("请查询当前上海时间。", "那伦敦呢？", "再看看纽约。")
EXPECTED_TIMEZONES = ("Asia/Shanghai", "Europe/London", "America/New_York")
TIME_SOURCE_ID = "system-clock-iana-tzdb"


def validate_base_url() -> None:
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("smoke_base_url_must_be_loopback")


def request_json(
    method: str, path: str, body: dict[str, object] | None = None
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
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


def consume_events(
    run_id: str, *, after_seq: int = 0, stop_after_events: int | None = None
) -> list[dict[str, Any]]:
    path = (
        f"/api/runs/{urllib.parse.quote(run_id)}/events"
        f"?after_seq={after_seq}"
    )
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
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
            events.append(event)
            data_lines = []
            if stop_after_events is not None and len(events) >= stop_after_events:
                return events
            if event.get("type") in TERMINAL_EVENTS:
                return events
    return events


def create_thread(title: str) -> str:
    status, payload = request_json("POST", "/api/threads", {"title": title})
    if status != 201:
        raise RuntimeError("thread_create_failed")
    return str(payload["thread_id"])


def create_run(thread_id: str, prompt: str, idempotency_key: str) -> str:
    status, payload = request_json(
        "POST",
        f"/api/threads/{urllib.parse.quote(thread_id)}/runs",
        {"message": prompt, "idempotency_key": idempotency_key},
    )
    if status != 201:
        raise RuntimeError("run_create_failed")
    return str(payload["run_id"])


def get_thread(thread_id: str) -> dict[str, Any]:
    status, payload = request_json(
        "GET", f"/api/threads/{urllib.parse.quote(thread_id)}"
    )
    if status != 200:
        raise RuntimeError("thread_read_failed")
    return payload


def validate_completed_turn(
    events: list[dict[str, Any]], expected_timezone: str
) -> dict[str, Any]:
    seqs = [int(event["seq"]) for event in events]
    if seqs != list(range(1, len(events) + 1)):
        raise RuntimeError("event_sequence_invalid")
    event_types = [str(event["type"]) for event in events]
    if not event_types or event_types[-1] != "run.completed":
        raise RuntimeError("run_not_completed")
    started = [event for event in events if event["type"] == "tool.started"]
    if len(started) != 1:
        raise RuntimeError("tool_call_count_invalid")
    timezone_match = started[0]["data"].get("input_summary") == expected_timezone
    if not timezone_match:
        raise RuntimeError("tool_timezone_invalid")
    sources = [event for event in events if event["type"] == "source.added"]
    source_match = (
        len(sources) == 1 and sources[0]["data"].get("source_id") == TIME_SOURCE_ID
    )
    if not source_match:
        raise RuntimeError("source_invalid")
    finished = [event for event in events if event["type"] == "tool.finished"]
    if len(finished) != 1:
        raise RuntimeError("tool_result_count_invalid")
    if started[0]["data"].get("tool_call_id") != finished[0]["data"].get(
        "tool_call_id"
    ):
        raise RuntimeError("tool_call_id_mismatch")
    completed = [event for event in events if event["type"] == "message.completed"]
    answer = (
        completed[0]["data"].get("message", {}).get("content", "")
        if len(completed) == 1
        else ""
    )
    output_summary = str(finished[0]["data"].get("output_summary", ""))
    offset_match = re.search(r"([+-])(\d{2}):(\d{2})$", output_summary)
    local_time_match = re.search(r"\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2})", output_summary)
    offset_forms: set[str] = set()
    if offset_match:
        sign, hours, minutes = offset_match.groups()
        offset_forms = {
            f"{sign}{hours}:{minutes}",
            f"{sign}{hours}{minutes}",
            f"UTC{sign}{int(hours)}",
            f"GMT{sign}{int(hours)}",
        }
    local_time_forms: set[str] = set()
    if local_time_match:
        hours, minutes = local_time_match.groups()
        local_time_forms = {f"{hours}:{minutes}", f"{int(hours)}:{minutes}"}
    normalized_answer = answer.upper().replace(" ", "")
    tool_result_match = bool(local_time_forms) and any(
        form in answer for form in local_time_forms
    )
    answer_contract = (
        expected_timezone in answer
        and tool_result_match
        and any(form.upper() in normalized_answer for form in offset_forms)
    )
    if not answer_contract:
        raise RuntimeError("answer_contract_invalid")
    return {
        "event_count": len(events),
        "event_types": event_types,
        "tool_calls": len(started),
        "timezone_match": timezone_match,
        "source_match": source_match,
        "tool_result_match": tool_result_match,
        "answer_contract": answer_contract,
        "terminal": event_types[-1],
    }


def conversation(lane: str, model: str) -> dict[str, Any]:
    thread_id = create_thread("Phase 2 conversation smoke")
    run_ids: list[str] = []
    summaries: list[dict[str, Any]] = []
    first_key = f"phase2-{uuid.uuid4().hex}"
    disconnect_resume = False

    for index, (prompt, timezone) in enumerate(zip(PROMPTS, EXPECTED_TIMEZONES, strict=True)):
        key = first_key if index == 0 else f"phase2-{uuid.uuid4().hex}"
        run_id = create_run(thread_id, prompt, key)
        run_ids.append(run_id)
        started = time.perf_counter()
        if index == 0:
            prefix = consume_events(run_id, stop_after_events=2)
            cursor = int(prefix[-1]["seq"])
            suffix = consume_events(run_id, after_seq=cursor)
            events = [*prefix, *suffix]
            disconnect_resume = (
                cursor == 2
                and len({(event["run_id"], event["seq"]) for event in events}) == len(events)
            )
            if not disconnect_resume:
                raise RuntimeError("disconnect_resume_invalid")
        else:
            events = consume_events(run_id)
        summary = validate_completed_turn(events, timezone)
        summary["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        summaries.append(summary)

    repeated_run_id = create_run(thread_id, PROMPTS[0], first_key)
    if repeated_run_id != run_ids[0]:
        raise RuntimeError("idempotent_replay_created_second_run")

    snapshot = get_thread(thread_id)
    if snapshot.get("active_run") is not None:
        raise RuntimeError("thread_remained_active")
    if len(snapshot.get("runs", [])) != 3:
        raise RuntimeError("persisted_run_count_invalid")
    roles = [message.get("role") for message in snapshot.get("messages", [])]
    if roles.count("user") != 3 or roles.count("assistant") != 3:
        raise RuntimeError("persisted_message_count_invalid")
    for index, run in enumerate(snapshot["runs"]):
        if run.get("run_id") != run_ids[index] or run.get("status") != "completed":
            raise RuntimeError("snapshot_run_order_invalid")
        validate_completed_turn(run.get("events", []), EXPECTED_TIMEZONES[index])

    return {
        "lane": lane,
        "status": "passed",
        "model": model,
        "turns": summaries,
        "disconnect_resume": disconnect_resume,
        "idempotent_replay_same_run": True,
        "persisted_message_count": len(snapshot["messages"]),
        "persisted_run_count": len(snapshot["runs"]),
    }


def restart_create() -> dict[str, Any]:
    thread_id = create_thread("Phase 2 restart smoke")
    run_id = create_run(
        thread_id,
        PROMPTS[0],
        f"phase2-restart-{uuid.uuid4().hex}",
    )
    prefix = consume_events(run_id, stop_after_events=2)
    if [event["type"] for event in prefix] != ["run.started", "tool.started"]:
        raise RuntimeError("restart_probe_not_active")
    return {
        "lane": "phase2_restart_create",
        "status": "ready_for_restart",
        "thread_id": thread_id,
        "run_id": run_id,
        "last_seq": int(prefix[-1]["seq"]),
    }


def restart_verify(thread_id: str, run_id: str) -> dict[str, Any]:
    snapshot = get_thread(thread_id)
    old = next((run for run in snapshot.get("runs", []) if run.get("run_id") == run_id), None)
    if old is None or old.get("status") != "failed":
        raise RuntimeError("orphan_not_failed")
    old_events = old.get("events", [])
    if not old_events:
        raise RuntimeError("orphan_events_missing")
    terminal = old_events[-1]
    if terminal.get("type") != "run.failed" or terminal.get("data") != {
        "status": "failed",
        "error_code": "service_restarted",
    }:
        raise RuntimeError("orphan_failure_contract_invalid")
    old_event_count = len(old_events)

    retry_run_id = create_run(
        thread_id,
        PROMPTS[0],
        f"phase2-retry-{uuid.uuid4().hex}",
    )
    if retry_run_id == run_id:
        raise RuntimeError("retry_reused_failed_run")
    retry_events = consume_events(retry_run_id)
    retry_summary = validate_completed_turn(retry_events, EXPECTED_TIMEZONES[0])

    final_snapshot = get_thread(thread_id)
    final_old = next(
        run for run in final_snapshot["runs"] if run.get("run_id") == run_id
    )
    if final_old.get("status") != "failed" or len(final_old.get("events", [])) != old_event_count:
        raise RuntimeError("failed_run_was_mutated")
    return {
        "lane": "phase2_restart_verify",
        "status": "passed",
        "old_terminal": "run.failed",
        "old_error_code": "service_restarted",
        "old_event_count_unchanged": True,
        "retry_created_new_run": True,
        "retry": retry_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    conversation_parser = subparsers.add_parser("conversation")
    conversation_parser.add_argument("--lane", default="phase2_compose_e3")
    conversation_parser.add_argument("--model", default="deterministic-fake")
    subparsers.add_parser("restart-create")
    verify_parser = subparsers.add_parser("restart-verify")
    verify_parser.add_argument("--thread-id", required=True)
    verify_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    try:
        validate_base_url()
        wait_until_ready()
        if args.command == "conversation":
            result = conversation(args.lane, args.model)
        elif args.command == "restart-create":
            result = restart_create()
        else:
            result = restart_verify(args.thread_id, args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI emits only bounded classifications.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "classification": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
