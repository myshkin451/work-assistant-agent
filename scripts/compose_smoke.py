#!/usr/bin/env python3
"""Exercise the public REST/SSE contract without printing message content."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


BASE_URL = os.environ.get("WORK_ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def validate_base_url() -> None:
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("smoke_base_url_must_be_loopback")


def request_json(method: str, path: str, body: dict[str, object] | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with OPENER.open(request, timeout=15) as response:
        return json.load(response)


def wait_until_ready() -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            request_json("GET", "/health")
            return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError("backend_not_ready")


def consume_events(run_id: str) -> list[dict]:
    url = f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events?after_seq=0"
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    events: list[dict] = []
    data_lines: list[str] = []
    with OPENER.open(request, timeout=150) as response:
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
            if event.get("type") in TERMINAL_EVENTS:
                break
    return events


def main() -> int:
    try:
        validate_base_url()
        wait_until_ready()
        thread = request_json("POST", "/api/threads", {"title": "Contract smoke"})
        thread_id = str(thread["thread_id"])
        run = request_json(
            "POST",
            f"/api/threads/{urllib.parse.quote(thread_id)}/runs",
            {
                "message": "Please check the current time in Asia/Shanghai and cite the source.",
                "idempotency_key": f"smoke-{uuid.uuid4().hex}",
            },
        )
        run_id = str(run["run_id"])
        events = consume_events(run_id)
        event_types = [str(event.get("type")) for event in events]
        seqs = [int(event["seq"]) for event in events]
        required = {
            "run.started",
            "tool.started",
            "tool.finished",
            "source.added",
            "message.completed",
            "run.completed",
        }
        if not required.issubset(event_types):
            raise RuntimeError(f"missing_event_types:{sorted(required.difference(event_types))}")
        if seqs != sorted(set(seqs)):
            raise RuntimeError("event_sequence_not_strictly_increasing")
        snapshot = request_json("GET", f"/api/threads/{urllib.parse.quote(thread_id)}")
        roles = [message.get("role") for message in snapshot.get("messages", [])]
        if roles.count("user") != 1 or roles.count("assistant") != 1:
            raise RuntimeError("persisted_message_roles_invalid")
        replay = consume_replay(run_id, seqs[-2])
        if [int(event["seq"]) for event in replay] != [seqs[-1]]:
            raise RuntimeError("after_seq_replay_invalid")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "event_count": len(events),
                    "terminal_event": event_types[-1],
                    "persisted_message_count": len(snapshot.get("messages", [])),
                    "replay_event_count": len(replay),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must emit a bounded classification.
        print(json.dumps({"status": "failed", "classification": type(exc).__name__}))
        return 1


def consume_replay(run_id: str, after_seq: int) -> list[dict]:
    url = (
        f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events"
        f"?after_seq={after_seq}"
    )
    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    events: list[dict] = []
    data_lines: list[str] = []
    with OPENER.open(request, timeout=15) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="strict").rstrip("\r\n")
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
                continue
            if line or not data_lines:
                continue
            events.append(json.loads("\n".join(data_lines)))
            data_lines = []
    return events


if __name__ == "__main__":
    sys.exit(main())
