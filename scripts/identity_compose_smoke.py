#!/usr/bin/env python3
"""Exercise two-Principal REST/SSE ownership without printing private content."""

from __future__ import annotations

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
DEV_HEADER = "X-Work-Assistant-Dev-Subject"
PRINCIPAL_A = "urn:work-assistant:neutral:test-principal-a"
PRINCIPAL_B = "urn:work-assistant:neutral:test-principal-b"
LEGACY_THREAD_ID = os.environ.get("WORK_ASSISTANT_LEGACY_THREAD_ID")
LEGACY_RUN_ID = os.environ.get("WORK_ASSISTANT_LEGACY_RUN_ID")
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def validate_base_url() -> None:
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("smoke_base_url_must_be_loopback")


def headers(*, principal: str | None, accept: str) -> dict[str, str]:
    result = {"Accept": accept}
    if principal is not None:
        result[DEV_HEADER] = principal
    return result


def request_json(
    method: str,
    path: str,
    *,
    principal: str | None,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = headers(principal=principal, accept="application/json")
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, method=method, headers=request_headers
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
            status, _ = request_json("GET", "/health", principal=None)
            if status == 200:
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise RuntimeError("backend_not_ready")


def error_code(payload: dict[str, Any]) -> str | None:
    detail = payload.get("detail")
    return str(detail.get("code")) if isinstance(detail, dict) and detail.get("code") else None


def require_error(
    result: tuple[int, dict[str, Any]], *, status: int, code: str
) -> None:
    actual_status, payload = result
    if actual_status != status or error_code(payload) != code:
        raise RuntimeError("error_contract_invalid")


def create_thread(principal: str, title: str) -> str:
    status, payload = request_json(
        "POST", "/api/threads", principal=principal, body={"title": title}
    )
    if status != 201:
        raise RuntimeError("thread_create_failed")
    return str(payload["thread_id"])


def create_run(principal: str, thread_id: str, key: str) -> tuple[int, dict[str, Any]]:
    return request_json(
        "POST",
        f"/api/threads/{urllib.parse.quote(thread_id)}/runs",
        principal=principal,
        body={
            "message": "Check the current time in Asia/Shanghai.",
            "idempotency_key": key,
        },
    )


def consume_events(
    principal: str,
    run_id: str,
    *,
    after_seq: int | None = 0,
    last_event_id: str | None = None,
    stop_after_events: int | None = None,
) -> list[dict[str, Any]]:
    query = "" if after_seq is None else f"?after_seq={after_seq}"
    request_headers = headers(principal=principal, accept="text/event-stream")
    if last_event_id is not None:
        request_headers["Last-Event-ID"] = last_event_id
    request = urllib.request.Request(
        f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events{query}",
        headers=request_headers,
    )
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []
    with OPENER.open(request, timeout=180) as response:
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


def request_sse_error(
    principal: str | None,
    run_id: str,
    *,
    expected_status: int,
    expected_code: str,
    last_event_id: str | None = None,
) -> None:
    request_headers = headers(principal=principal, accept="text/event-stream")
    if last_event_id is not None:
        request_headers["Last-Event-ID"] = last_event_id
    request = urllib.request.Request(
        f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events",
        headers=request_headers,
    )
    try:
        OPENER.open(request, timeout=15)
    except urllib.error.HTTPError as exc:
        with exc:
            content_type = exc.headers.get_content_type()
            payload = json.load(exc)
        if (
            exc.code != expected_status
            or content_type != "application/json"
            or error_code(payload) != expected_code
        ):
            raise RuntimeError("sse_error_contract_invalid") from None
        return
    raise RuntimeError("sse_authorization_unexpectedly_succeeded")


def require_terminal(events: list[dict[str, Any]], expected: str) -> None:
    seqs = [int(event["seq"]) for event in events]
    if not events or str(events[-1].get("type")) != expected:
        raise RuntimeError("terminal_event_invalid")
    if seqs != list(range(1, len(events) + 1)):
        raise RuntimeError("event_sequence_invalid")


def main() -> int:
    stage = "startup"
    try:
        validate_base_url()
        wait_until_ready()
        stage = "authentication_and_legacy"
        require_error(
            request_json("GET", "/api/threads", principal=None),
            status=401,
            code="authentication_required",
        )
        initial_status_a, initial_list_a = request_json(
            "GET", "/api/threads", principal=PRINCIPAL_A
        )
        initial_status_b, initial_list_b = request_json(
            "GET", "/api/threads", principal=PRINCIPAL_B
        )
        if initial_status_a != 200 or initial_status_b != 200:
            raise RuntimeError("initial_owned_list_failed")
        initial_ids_a = {
            str(item["thread_id"]) for item in initial_list_a.get("items", [])
        }
        initial_ids_b = {
            str(item["thread_id"]) for item in initial_list_b.get("items", [])
        }
        legacy_quarantine_checked = LEGACY_THREAD_ID is not None and LEGACY_RUN_ID is not None
        if legacy_quarantine_checked:
            assert LEGACY_THREAD_ID is not None and LEGACY_RUN_ID is not None
            if LEGACY_THREAD_ID in initial_ids_a or LEGACY_THREAD_ID in initial_ids_b:
                raise RuntimeError("legacy_thread_visible_in_owned_list")
            for principal in (PRINCIPAL_A, PRINCIPAL_B):
                require_error(
                    request_json(
                        "GET", f"/api/threads/{LEGACY_THREAD_ID}", principal=principal
                    ),
                    status=403,
                    code="thread_forbidden",
                )
                request_sse_error(
                    principal,
                    LEGACY_RUN_ID,
                    expected_status=403,
                    expected_code="run_forbidden",
                )

        stage = "owned_run_and_idempotency"
        thread_a = create_thread(PRINCIPAL_A, "Neutral ownership smoke one")
        thread_b = create_thread(PRINCIPAL_B, "Neutral ownership smoke two")
        same_key = f"identity-same-{uuid.uuid4().hex}"
        status_a, run_a_payload = create_run(PRINCIPAL_A, thread_a, same_key)
        status_b, run_b_payload = create_run(PRINCIPAL_B, thread_b, same_key)
        if status_a != 201 or status_b != 201:
            raise RuntimeError("owned_run_create_failed")
        run_a = str(run_a_payload["run_id"])
        run_b = str(run_b_payload["run_id"])
        if run_a == run_b:
            raise RuntimeError("cross_principal_idempotency_collision")
        repeated_status, repeated_payload = create_run(PRINCIPAL_A, thread_a, same_key)
        if repeated_status != 201 or str(repeated_payload.get("run_id")) != run_a:
            raise RuntimeError("owner_idempotency_replay_invalid")
        require_error(
            create_run(PRINCIPAL_B, thread_a, same_key),
            status=403,
            code="thread_forbidden",
        )

        stage = "sse_replay"
        events_a = consume_events(PRINCIPAL_A, run_a)
        events_b = consume_events(PRINCIPAL_B, run_b)
        require_terminal(events_a, "run.completed")
        require_terminal(events_b, "run.completed")
        replay_a = consume_events(PRINCIPAL_A, run_a, after_seq=int(events_a[-2]["seq"]))
        if [int(event["seq"]) for event in replay_a] != [int(events_a[-1]["seq"])]:
            raise RuntimeError("after_seq_replay_invalid")
        header_replay_a = consume_events(
            PRINCIPAL_A,
            run_a,
            after_seq=None,
            last_event_id=str(events_a[-2]["seq"]),
        )
        if [int(event["seq"]) for event in header_replay_a] != [int(events_a[-1]["seq"])]:
            raise RuntimeError("last_event_id_replay_invalid")
        explicit_cursor_a = consume_events(
            PRINCIPAL_A,
            run_a,
            after_seq=int(events_a[-2]["seq"]),
            last_event_id="invalid-but-ignored",
        )
        if [int(event["seq"]) for event in explicit_cursor_a] != [int(events_a[-1]["seq"])]:
            raise RuntimeError("explicit_cursor_precedence_invalid")

        request_sse_error(
            PRINCIPAL_B,
            run_a,
            expected_status=403,
            expected_code="run_forbidden",
            last_event_id="1",
        )
        request_sse_error(
            None, run_a, expected_status=401, expected_code="authentication_required"
        )
        missing = str(uuid.uuid4())
        request_sse_error(
            PRINCIPAL_A, missing, expected_status=404, expected_code="run_not_found"
        )

        stage = "sse_reconnect"
        reconnect_thread = create_thread(PRINCIPAL_A, "Neutral reconnect smoke")
        reconnect_status, reconnect_payload = create_run(
            PRINCIPAL_A, reconnect_thread, f"identity-reconnect-{uuid.uuid4().hex}"
        )
        if reconnect_status != 201:
            raise RuntimeError("reconnect_run_create_failed")
        reconnect_run = str(reconnect_payload["run_id"])
        prefix = consume_events(PRINCIPAL_A, reconnect_run, stop_after_events=2)
        cursor = int(prefix[-1]["seq"])
        request_sse_error(
            PRINCIPAL_B,
            reconnect_run,
            expected_status=403,
            expected_code="run_forbidden",
        )
        suffix = consume_events(PRINCIPAL_A, reconnect_run, after_seq=cursor)
        reconnected = [*prefix, *suffix]
        require_terminal(reconnected, "run.completed")

        stage = "cancel_and_retry"
        cancel_thread = create_thread(PRINCIPAL_A, "Neutral cancel and retry smoke")
        cancel_status, cancel_payload = create_run(
            PRINCIPAL_A, cancel_thread, f"identity-cancel-{uuid.uuid4().hex}"
        )
        if cancel_status != 201:
            raise RuntimeError("cancel_run_create_failed")
        cancel_run = str(cancel_payload["run_id"])
        before_forbidden_cancel_status, before_forbidden_cancel = request_json(
            "GET", f"/api/threads/{cancel_thread}", principal=PRINCIPAL_A
        )
        if before_forbidden_cancel_status != 200:
            raise RuntimeError("owner_snapshot_before_cancel_failed")
        require_error(
            request_json(
                "POST",
                f"/api/runs/{urllib.parse.quote(cancel_run)}/cancel",
                principal=PRINCIPAL_B,
            ),
            status=403,
            code="run_forbidden",
        )
        after_forbidden_cancel_status, after_forbidden_cancel = request_json(
            "GET", f"/api/threads/{cancel_thread}", principal=PRINCIPAL_A
        )
        before_run = next(
            (
                run
                for run in before_forbidden_cancel.get("runs", [])
                if str(run.get("run_id")) == cancel_run
            ),
            None,
        )
        after_run = next(
            (
                run
                for run in after_forbidden_cancel.get("runs", [])
                if str(run.get("run_id")) == cancel_run
            ),
            None,
        )
        after_events = after_run.get("events", []) if isinstance(after_run, dict) else []
        if (
            after_forbidden_cancel_status != 200
            or not isinstance(before_run, dict)
            or not isinstance(after_run, dict)
            or after_run.get("run_id") != before_run.get("run_id")
            or after_run.get("status") == "cancelled"
            or any(event.get("type") == "run.cancelled" for event in after_events)
        ):
            raise RuntimeError("forbidden_cancel_mutated_owner_state")
        owner_cancel_status, owner_cancel_payload = request_json(
            "POST",
            f"/api/runs/{urllib.parse.quote(cancel_run)}/cancel",
            principal=PRINCIPAL_A,
        )
        if owner_cancel_status != 200 or owner_cancel_payload.get("status") != "cancelled":
            raise RuntimeError("owner_cancel_failed")
        cancelled_events = consume_events(PRINCIPAL_A, cancel_run)
        require_terminal(cancelled_events, "run.cancelled")
        stage = "owned_detail_and_list"
        require_error(
            create_run(PRINCIPAL_B, cancel_thread, f"identity-forbidden-{uuid.uuid4().hex}"),
            status=403,
            code="thread_forbidden",
        )
        retry_status, retry_payload = create_run(
            PRINCIPAL_A, cancel_thread, f"identity-retry-{uuid.uuid4().hex}"
        )
        if retry_status != 201:
            raise RuntimeError("owner_retry_failed")
        retry_events = consume_events(PRINCIPAL_A, str(retry_payload["run_id"]))
        require_terminal(retry_events, "run.completed")

        require_error(
            request_json("GET", f"/api/threads/{thread_a}", principal=PRINCIPAL_B),
            status=403,
            code="thread_forbidden",
        )
        require_error(
            request_json("GET", f"/api/threads/{missing}", principal=PRINCIPAL_A),
            status=404,
            code="thread_not_found",
        )
        require_error(
            request_json("POST", f"/api/runs/{missing}/cancel", principal=PRINCIPAL_A),
            status=404,
            code="run_not_found",
        )
        detail_status_a, detail_a = request_json(
            "GET", f"/api/threads/{thread_a}", principal=PRINCIPAL_A
        )
        detail_status_b, detail_b = request_json(
            "GET", f"/api/threads/{thread_b}", principal=PRINCIPAL_B
        )
        if detail_status_a != 200 or detail_status_b != 200:
            raise RuntimeError("owner_detail_failed")
        detail_events_a = detail_a.get("runs", [{}])[0].get("events", [])
        detail_events_b = detail_b.get("runs", [{}])[0].get("events", [])
        if (
            {str(run["run_id"]) for run in detail_a.get("runs", [])} != {run_a}
            or {str(run["run_id"]) for run in detail_b.get("runs", [])} != {run_b}
            or len(detail_a.get("messages", [])) != 2
            or len(detail_b.get("messages", [])) != 2
            or [int(event["seq"]) for event in detail_events_a]
            != [int(event["seq"]) for event in events_a]
            or [int(event["seq"]) for event in detail_events_b]
            != [int(event["seq"]) for event in events_b]
        ):
            raise RuntimeError("owner_detail_isolation_invalid")
        status_list_a, list_a = request_json("GET", "/api/threads", principal=PRINCIPAL_A)
        status_list_b, list_b = request_json("GET", "/api/threads", principal=PRINCIPAL_B)
        ids_a = {str(item["thread_id"]) for item in list_a.get("items", [])}
        ids_b = {str(item["thread_id"]) for item in list_b.get("items", [])}
        if status_list_a != 200 or status_list_b != 200:
            raise RuntimeError("owned_list_failed")
        if ids_a != initial_ids_a | {thread_a, reconnect_thread, cancel_thread}:
            raise RuntimeError("owned_list_isolation_invalid")
        if ids_b != initial_ids_b | {thread_b}:
            raise RuntimeError("owned_list_isolation_invalid")

        exposed = json.dumps(
            [
                run_a_payload,
                run_b_payload,
                events_a,
                events_b,
                list_a,
                list_b,
                detail_a,
                detail_b,
            ],
            sort_keys=True,
        )
        if PRINCIPAL_A in exposed or PRINCIPAL_B in exposed:
            raise RuntimeError("principal_exposed_in_public_contract")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "principal_count": 2,
                    "unauthenticated_status": 401,
                    "cross_thread_status": 403,
                    "missing_resource_status": 404,
                    "created_owned_thread_counts": [3, 1],
                    "same_key_cross_principal_distinct": True,
                    "owner_idempotent_replay": True,
                    "sse_after_seq": True,
                    "sse_last_event_id": True,
                    "sse_reauthenticated_on_reconnect": True,
                    "owner_cancel_and_retry": True,
                    "forbidden_cancel_unchanged": True,
                    "owner_snapshots_isolated": True,
                    "legacy_quarantine_checked": legacy_quarantine_checked,
                    "public_contract_hides_principal": True,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI emits a bounded classification only.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": stage,
                    "classification": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
