#!/usr/bin/env python3
"""Validate product concurrency constraints through the loopback Compose API."""

from __future__ import annotations

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


BASE_URL = os.environ.get("WORK_ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def validate_base_url() -> None:
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("smoke_base_url_must_be_loopback")


def request_json(
    method: str, path: str, body: dict[str, object] | None = None
) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with OPENER.open(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, json.load(exc)


def create_thread(title: str) -> str:
    status, payload = request_json("POST", "/api/threads", {"title": title})
    if status != 201:
        raise RuntimeError("thread_create_failed")
    return str(payload["thread_id"])


def wait_until_ready() -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            status, _ = request_json("GET", "/health")
            if status == 200:
                return
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    raise RuntimeError("backend_not_ready")


def create_run(thread_id: str, key: str) -> tuple[int, dict]:
    return request_json(
        "POST",
        f"/api/threads/{urllib.parse.quote(thread_id)}/runs",
        {
            "message": "Check the current time in Asia/Shanghai.",
            "idempotency_key": key,
        },
    )


def main() -> int:
    try:
        validate_base_url()
        wait_until_ready()
        idempotent_thread = create_thread("Idempotency smoke")
        with ThreadPoolExecutor(max_workers=20) as executor:
            idempotent = list(
                executor.map(
                    lambda _: create_run(idempotent_thread, "same-key"),
                    range(20),
                )
            )
        idempotent_ids = {
            str(payload.get("run_id"))
            for status, payload in idempotent
            if status == 201
        }
        if {status for status, _ in idempotent} != {201} or len(idempotent_ids) != 1:
            raise RuntimeError("idempotency_constraint_failed")
        request_json("POST", f"/api/runs/{idempotent_ids.pop()}/cancel")

        active_thread = create_thread("Single active Run smoke")
        with ThreadPoolExecutor(max_workers=10) as executor:
            contenders = list(
                executor.map(
                    lambda index: create_run(active_thread, f"different-{index}"),
                    range(10),
                )
            )
        accepted = [payload for status, payload in contenders if status == 201]
        conflicts = [payload for status, payload in contenders if status == 409]
        if len(accepted) != 1 or len(conflicts) != 9:
            raise RuntimeError("single_active_run_constraint_failed")
        request_json("POST", f"/api/runs/{accepted[0]['run_id']}/cancel")

        print(
            json.dumps(
                {
                    "status": "passed",
                    "same_key_requests": 20,
                    "same_key_unique_runs": 1,
                    "different_key_requests": 10,
                    "accepted_active_runs": 1,
                    "conflicts": 9,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - print only a bounded classification.
        print(json.dumps({"status": "failed", "classification": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
