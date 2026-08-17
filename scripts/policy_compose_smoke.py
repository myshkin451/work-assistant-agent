#!/usr/bin/env python3
"""Exercise the T-006 policy kernel without printing prompts or private payloads."""

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

import psycopg

BASE_URL = os.environ.get("WORK_ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
DATABASE_URL = os.environ.get(
    "WORK_ASSISTANT_DATABASE_URL",
    "postgresql://work_assistant:work_assistant@127.0.0.1:55432/work_assistant",
)
DEV_HEADER = "X-Work-Assistant-Dev-Subject"
PRINCIPAL_A = "urn:work-assistant:neutral:t006-principal-a"
PRINCIPAL_B = "urn:work-assistant:neutral:t006-principal-b"
TERMINAL_EVENTS = {"run.completed", "run.failed", "run.cancelled"}
TIME_SOURCE_ID = "system-clock-iana-tzdb"
PROMPT_SHA256 = "cfe388ab19e7609fa7f66f6813f66cf0579daf8365aa92f6a1bbefa2c73dbacb"
LEGACY_THREAD_ID = "t006-v02-legacy-thread"
LEGACY_RUN_ID = "t006-v02-legacy-run"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def validate_local_targets() -> None:
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError("smoke_base_url_must_be_loopback")
    database = urllib.parse.urlparse(DATABASE_URL)
    if database.scheme not in {"postgresql", "postgres"} or database.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError("smoke_database_url_must_be_loopback")


def request_json(
    method: str,
    path: str,
    *,
    principal: str,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = (
        None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    )
    headers = {"Accept": "application/json", DEV_HEADER: principal}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, method=method, headers=headers
    )
    try:
        with OPENER.open(request, timeout=20) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def request_sse_error(principal: str, run_id: str) -> str:
    request = urllib.request.Request(
        f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events",
        headers={"Accept": "text/event-stream", DEV_HEADER: principal},
    )
    try:
        OPENER.open(request, timeout=10)
    except urllib.error.HTTPError as exc:
        with exc:
            payload = json.load(exc)
        detail = payload.get("detail")
        code = detail.get("code") if isinstance(detail, dict) else None
        if exc.code == 403 and code == "run_forbidden":
            return str(code)
        raise RuntimeError("cross_owner_sse_contract_invalid") from None
    raise RuntimeError("cross_owner_sse_unexpectedly_succeeded")


def wait_until_ready() -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            status, _ = request_json("GET", "/health", principal=PRINCIPAL_A)
            if status == 200:
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise RuntimeError("backend_not_ready")


def create_thread(principal: str) -> str:
    status, payload = request_json(
        "POST",
        "/api/threads",
        principal=principal,
        body={"title": "Neutral T-006 policy smoke"},
    )
    if status != 201:
        raise RuntimeError("thread_create_failed")
    return str(payload["thread_id"])


def create_run(
    principal: str,
    thread_id: str,
    *,
    message: str,
    idempotency_key: str,
) -> str:
    status, payload = request_json(
        "POST",
        f"/api/threads/{urllib.parse.quote(thread_id)}/runs",
        principal=principal,
        body={"message": message, "idempotency_key": idempotency_key},
    )
    if status != 201:
        raise RuntimeError("run_create_failed")
    return str(payload["run_id"])


def consume_events(
    principal: str,
    run_id: str,
    *,
    after_seq: int = 0,
    stop_after_events: int | None = None,
) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        f"{BASE_URL}/api/runs/{urllib.parse.quote(run_id)}/events?after_seq={after_seq}",
        headers={"Accept": "text/event-stream", DEV_HEADER: principal},
    )
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []
    with OPENER.open(request, timeout=30) as response:
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


def require_terminal(
    events: list[dict[str, Any]],
    *,
    terminal_type: str,
    failure_code: str | None = None,
) -> None:
    if not events or events[-1].get("type") != terminal_type:
        raise RuntimeError("terminal_event_invalid")
    sequences = [int(event["seq"]) for event in events]
    if sequences != list(range(sequences[0], sequences[-1] + 1)):
        raise RuntimeError("event_sequence_invalid")
    if (
        failure_code is not None
        and events[-1].get("data", {}).get("error_code") != failure_code
    ):
        raise RuntimeError("failure_code_invalid")


def fetch_evidence(run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with psycopg.connect(
        DATABASE_URL,
        connect_timeout=5,
        options="-c statement_timeout=5000 -c lock_timeout=3000",
    ) as connection:
        row = connection.execute(
            "SELECT execution_plan, execution_outcome FROM product_runs WHERE id = %s",
            (run_id,),
        ).fetchone()
    if row is None or not isinstance(row[0], dict) or not isinstance(row[1], dict):
        raise RuntimeError("run_evidence_missing")
    return row[0], row[1]


def validate_plan(plan: dict[str, Any], *, forbidden: tuple[str, ...]) -> None:
    expected_budget = {
        "max_model_steps": int(os.environ.get("MAX_MODEL_STEPS", "8")),
        "max_tool_calls": int(os.environ.get("MAX_TOOL_CALLS", "4")),
        "deadline_seconds": float(os.environ.get("RUN_TIMEOUT_SECONDS", "120")),
        "max_identical_tool_calls": int(
            os.environ.get("MAX_IDENTICAL_TOOL_CALLS", "1")
        ),
        "max_no_progress_steps": int(os.environ.get("MAX_NO_PROGRESS_STEPS", "2")),
    }
    if (
        plan.get("schema_version") != "1.0.0"
        or plan.get("agent_schema_version") != "1.0.0"
        or plan.get("agent_id") != "default-work-assistant"
        or plan.get("agent_version") != "1.0.0"
        or plan.get("model_profile_id") != "default"
        or plan.get("model_profile_version") != "1.0.0"
        or plan.get("model_provider") != "local-fake"
        or plan.get("model_id") != "deterministic-fake-v1"
        or plan.get("prompt_id") != "neutral-work-assistant"
        or plan.get("prompt_version") != "1.0.0"
        or plan.get("prompt_sha256") != PROMPT_SHA256
        or plan.get("context_builder_version") != "1.0.0"
        or plan.get("context_layer_versions")
        != {
            "host": "1.0.0",
            "agent": "1.0.0",
            "run": "1.0.0",
            "conversation": "product-message-v0.3",
            "tool_data": "tool-message-v1",
        }
        or plan.get("capability_policy_id") != "neutral-authenticated-tools"
        or plan.get("capability_policy_version") != "1.0.0"
        or plan.get("tool_registry_id") != "builtin-tool-registry"
        or plan.get("tool_registry_version") != "1.0.0"
        or plan.get("agent_allowed_tools") != ["get_current_time"]
        or plan.get("base_tools") != ["get_current_time"]
        or plan.get("visible_tools")
        != [{"tool_id": "get_current_time", "version": "1.0.0"}]
        or plan.get("budget") != expected_budget
        or plan.get("result_contract")
        != {
            "schema_version": "1.0.0",
            "max_answer_chars": 8_000,
            "source_policy": "required_if_tool_used",
        }
    ):
        raise RuntimeError("execution_plan_contract_invalid")
    serialized = json.dumps(plan, ensure_ascii=False, sort_keys=True)
    if any(value and value in serialized for value in forbidden):
        raise RuntimeError("execution_plan_contains_forbidden_data")


def run_case(
    *,
    message: str,
    expected_terminal: str,
    expected_failure: str | None,
) -> tuple[
    str,
    str,
    str,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    thread_id = create_thread(PRINCIPAL_A)
    idempotency_key = f"t006-{uuid.uuid4().hex}"
    run_id = create_run(
        PRINCIPAL_A,
        thread_id,
        message=message,
        idempotency_key=idempotency_key,
    )
    events = consume_events(PRINCIPAL_A, run_id)
    require_terminal(
        events,
        terminal_type=expected_terminal,
        failure_code=expected_failure,
    )
    plan, outcome = fetch_evidence(run_id)
    validate_plan(plan, forbidden=(PRINCIPAL_A, PRINCIPAL_B, message, "api_key"))
    expected_status = "completed" if expected_terminal == "run.completed" else "failed"
    if (
        outcome.get("status") != expected_status
        or outcome.get("failure_code") != expected_failure
    ):
        raise RuntimeError("execution_outcome_contract_invalid")
    return thread_id, run_id, idempotency_key, events, plan, outcome


def prepare_v02_legacy() -> dict[str, object]:
    validate_local_targets()
    with psycopg.connect(
        DATABASE_URL,
        connect_timeout=5,
        options="-c statement_timeout=5000 -c lock_timeout=3000",
    ) as connection:
        revision = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
        if revision != ("0002_principal_ownership",):
            raise RuntimeError("migration_not_at_v02")
        now = connection.execute("SELECT CURRENT_TIMESTAMP").fetchone()[0]
        connection.execute(
            "INSERT INTO product_threads "
            "(id, owner_subject, title, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
            (
                LEGACY_THREAD_ID,
                PRINCIPAL_A,
                "Neutral migration probe",
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO product_runs "
            "(id, thread_id, actor_subject, idempotency_key, status, last_seq, "
            "created_at, completed_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                LEGACY_RUN_ID,
                LEGACY_THREAD_ID,
                PRINCIPAL_A,
                "t006-v02-legacy-key",
                "completed",
                0,
                now,
                now,
            ),
        )
    return {"lane": "v02_seed", "status": "passed", "legacy_row_created": True}


def verify_v03_legacy() -> dict[str, object]:
    validate_local_targets()
    with psycopg.connect(
        DATABASE_URL,
        connect_timeout=5,
        options="-c statement_timeout=5000 -c lock_timeout=3000",
    ) as connection:
        revision = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
        row = connection.execute(
            "SELECT actor_subject, execution_plan, execution_outcome "
            "FROM product_runs WHERE id = %s",
            (LEGACY_RUN_ID,),
        ).fetchone()
    if revision != ("0003_agent_policy_evidence",) or row != (PRINCIPAL_A, None, None):
        raise RuntimeError("v02_to_v03_preservation_invalid")
    return {
        "lane": "v02_to_v03",
        "status": "passed",
        "legacy_evidence_remained_unknown": True,
    }


def exercise_policy() -> dict[str, object]:
    validate_local_targets()
    wait_until_ready()
    direct_message = "[policy:direct] neutral direct response"
    (
        direct_thread,
        direct_id,
        direct_key,
        direct_events,
        direct_plan,
        direct_outcome,
    ) = run_case(
        message=direct_message,
        expected_terminal="run.completed",
        expected_failure=None,
    )
    if any(event["type"].startswith("tool.") for event in direct_events):
        raise RuntimeError("direct_answer_used_tool")
    direct_usage = direct_outcome.get("usage", {})
    if (
        direct_usage.get("model_steps") != 1
        or direct_usage.get("tool_calls_attempted") != 0
    ):
        raise RuntimeError("direct_budget_evidence_invalid")

    _, _, _, time_events, _, time_outcome = run_case(
        message="Current time in Europe/London",
        expected_terminal="run.completed",
        expected_failure=None,
    )
    time_types = [event["type"] for event in time_events]
    if time_types.count("tool.started") != 1 or time_types.count("source.added") != 1:
        raise RuntimeError("allowed_tool_path_invalid")
    time_usage = time_outcome.get("usage", {})
    if (
        time_usage.get("model_steps") != 2
        or time_usage.get("tool_calls_attempted") != 1
        or time_usage.get("tool_calls_succeeded") != 1
        or time_outcome.get("accepted_source_ids") != [TIME_SOURCE_ID]
        or time_outcome.get("result_source_ids") != [TIME_SOURCE_ID]
    ):
        raise RuntimeError("allowed_tool_evidence_invalid")

    failures = {
        "tool_not_allowed": "[policy:tool-denied]",
        "model_step_limit": "[policy:model-step-limit]",
        "tool_call_limit": "[policy:tool-call-limit]",
        "run_timeout": "[policy:deadline]",
        "repeated_tool_call": "Current time UTC [policy:repeat-tool]",
        "no_progress": "[policy:no-progress]",
        "source_validation_failed": "Current time UTC [policy:source-invalid]",
    }
    observed: dict[str, str] = {}
    for expected_code, message in failures.items():
        _, _, _, _, _, outcome = run_case(
            message=message,
            expected_terminal="run.failed",
            expected_failure=expected_code,
        )
        if outcome.get("result_validation") == "passed":
            raise RuntimeError("failed_result_marked_valid")
        observed[expected_code] = str(outcome["failure_code"])

    direct_plan_before = json.dumps(direct_plan, sort_keys=True)
    direct_outcome_before = json.dumps(direct_outcome, sort_keys=True)
    replayed_id = create_run(
        PRINCIPAL_A,
        direct_thread,
        message=direct_message,
        idempotency_key=direct_key,
    )
    replay_plan, replay_outcome = fetch_evidence(replayed_id)
    if (
        replayed_id != direct_id
        or json.dumps(replay_plan, sort_keys=True) != direct_plan_before
        or json.dumps(replay_outcome, sort_keys=True) != direct_outcome_before
    ):
        raise RuntimeError("idempotent_evidence_changed")

    cancel_thread = create_thread(PRINCIPAL_A)
    cancel_id = create_run(
        PRINCIPAL_A,
        cancel_thread,
        message="[policy:deadline] cancellation race",
        idempotency_key=f"t006-cancel-{uuid.uuid4().hex}",
    )
    prefix = consume_events(PRINCIPAL_A, cancel_id, stop_after_events=1)
    if [event["type"] for event in prefix] != ["run.started"]:
        raise RuntimeError("cancel_probe_not_running")
    cancel_status, cancel_payload = request_json(
        "POST",
        f"/api/runs/{urllib.parse.quote(cancel_id)}/cancel",
        principal=PRINCIPAL_A,
    )
    if cancel_status != 200 or cancel_payload.get("status") != "cancelled":
        raise RuntimeError("cancel_failed")
    cancel_suffix = consume_events(PRINCIPAL_A, cancel_id, after_seq=1)
    cancel_events = [*prefix, *cancel_suffix]
    require_terminal(cancel_events, terminal_type="run.cancelled")
    _, cancel_outcome = fetch_evidence(cancel_id)
    time.sleep(0.5)
    later_status, later_snapshot = request_json(
        "GET",
        f"/api/threads/{urllib.parse.quote(cancel_thread)}",
        principal=PRINCIPAL_A,
    )
    if later_status != 200:
        raise RuntimeError("cancel_snapshot_failed")
    later_run = next(
        run for run in later_snapshot["runs"] if run.get("run_id") == cancel_id
    )
    if (
        later_run.get("status") != "cancelled"
        or len(later_run.get("events", [])) != len(cancel_events)
        or cancel_outcome.get("status") != "cancelled"
    ):
        raise RuntimeError("cancel_late_result_mutated_run")

    status, payload = request_json(
        "GET",
        f"/api/threads/{urllib.parse.quote(direct_thread)}",
        principal=PRINCIPAL_B,
    )
    detail = payload.get("detail")
    if (
        status != 403
        or not isinstance(detail, dict)
        or detail.get("code") != "thread_forbidden"
    ):
        raise RuntimeError("cross_owner_thread_contract_invalid")
    cross_owner_sse = request_sse_error(PRINCIPAL_B, direct_id)

    return {
        "lane": "t006_policy_kernel",
        "status": "passed",
        "direct_answer_without_tool": True,
        "allowed_tool_and_source": True,
        "failure_codes": sorted(observed),
        "idempotent_plan_and_outcome_unchanged": True,
        "cancel_late_result_discarded": True,
        "cross_owner_thread": "thread_forbidden",
        "cross_owner_sse": cross_owner_sse,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("exercise", "prepare-v02-legacy", "verify-v03-legacy"),
        default="exercise",
    )
    arguments = parser.parse_args()
    try:
        if arguments.mode == "prepare-v02-legacy":
            result = prepare_v02_legacy()
        elif arguments.mode == "verify-v03-legacy":
            result = verify_v03_legacy()
        else:
            result = exercise_policy()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "lane": arguments.mode,
                    "status": "failed",
                    "error": type(exc).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
