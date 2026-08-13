from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryFile
from typing import Literal

import anyio
import pytest
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import McpError

import neutral_mcp_server

FIXTURE_ROOT = Path(__file__).resolve().parents[1]
Mode = Literal["normal", "slow", "error", "oversized"]
EXPECTED_SERVER_NAME = "neutral-mcp-server"
EXPECTED_SERVER_VERSION = "0.1.0"
EXPECTED_TOOL_NAME = "read_neutral_record"
EXPECTED_OVERSIZED_PAYLOAD_BYTES = 262_144
EXPECTED_RECORD = {
    "id": "neutral-record-001",
    "label": "Controlled extension fixture",
    "category": "neutral-reference",
    "status": "available",
}
EXPECTED_NORMAL_PAYLOAD = "deterministic-read-only-result"
EXPECTED_ERROR_OUTPUT = {
    "error": {
        "code": "neutral_fixture_forced_error",
        "message": "The controlled fixture was started in error mode.",
        "retryable": False,
    }
}
EXPECTED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "record": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "label": {"type": "string"},
                "category": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["id", "label", "category", "status"],
            "additionalProperties": False,
        },
        "payload": {"type": "string"},
    },
    "required": ["record", "payload"],
    "additionalProperties": False,
}


@dataclass
class SessionProbe:
    session: ClientSession | None = None
    stdout_protocol_errors: list[str] = field(default_factory=list)
    stderr_text: str = ""


@asynccontextmanager
async def connected_client(mode: Mode) -> AsyncIterator[SessionProbe]:
    probe = SessionProbe()

    async def collect_protocol_message(message: object) -> None:
        if isinstance(message, Exception):
            probe.stdout_protocol_errors.append(repr(message))

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "neutral_mcp_server", "--mode", mode],
        cwd=FIXTURE_ROOT,
    )

    with TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        try:
            async with stdio_client(parameters, errlog=stderr_file) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=2),
                    message_handler=collect_protocol_message,
                ) as session:
                    probe.session = session
                    try:
                        yield probe
                    finally:
                        probe.session = None
        finally:
            stderr_file.flush()
            stderr_file.seek(0)
            probe.stderr_text = stderr_file.read()


def active_session(probe: SessionProbe) -> ClientSession:
    assert probe.session is not None
    return probe.session


def assert_text_matches_structured(result: types.CallToolResult) -> None:
    assert len(result.content) == 1
    text = result.content[0]
    assert isinstance(text, types.TextContent)
    assert json.loads(text.text) == result.structuredContent


async def normal_lifecycle() -> None:
    async with connected_client("normal") as probe:
        session = active_session(probe)

        initialized = await session.initialize()
        assert initialized.serverInfo.name == EXPECTED_SERVER_NAME
        assert initialized.serverInfo.version == EXPECTED_SERVER_VERSION
        assert initialized.capabilities.tools is not None

        listed = await session.list_tools()
        assert [tool.name for tool in listed.tools] == [EXPECTED_TOOL_NAME]
        tool = listed.tools[0]
        assert tool.inputSchema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        assert "mode" not in json.dumps(tool.inputSchema)
        assert tool.outputSchema == EXPECTED_OUTPUT_SCHEMA
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False

        result = await session.call_tool(EXPECTED_TOOL_NAME, {})
        assert result.isError is False
        assert result.structuredContent == {
            "record": EXPECTED_RECORD,
            "payload": EXPECTED_NORMAL_PAYLOAD,
        }
        assert_text_matches_structured(result)

    assert probe.stdout_protocol_errors == []
    expected_start_log = (
        f"starting server={EXPECTED_SERVER_NAME} "
        f"version={EXPECTED_SERVER_VERSION} mode=normal"
    )
    assert expected_start_log in probe.stderr_text
    assert f"serving tool={EXPECTED_TOOL_NAME} mode=normal" in probe.stderr_text


def test_real_stdio_initialize_list_and_call() -> None:
    anyio.run(normal_lifecycle)


async def mode_cannot_be_selected_by_tool_arguments() -> None:
    async with connected_client("normal") as probe:
        session = active_session(probe)
        await session.initialize()
        await session.list_tools()

        rejected = await session.call_tool(EXPECTED_TOOL_NAME, {"mode": "error"})
        assert rejected.isError is True
        assert rejected.structuredContent is None
        assert len(rejected.content) == 1
        assert isinstance(rejected.content[0], types.TextContent)
        assert "Input validation error" in rejected.content[0].text

        still_normal = await session.call_tool(EXPECTED_TOOL_NAME, {})
        assert still_normal.isError is False
        assert still_normal.structuredContent == {
            "record": EXPECTED_RECORD,
            "payload": EXPECTED_NORMAL_PAYLOAD,
        }

    assert probe.stdout_protocol_errors == []


def test_tool_arguments_cannot_switch_server_mode() -> None:
    anyio.run(mode_cannot_be_selected_by_tool_arguments)


async def slow_mode_times_out() -> None:
    async with connected_client("slow") as probe:
        session = active_session(probe)
        await session.initialize()
        await session.list_tools()

        with pytest.raises(McpError) as raised, anyio.fail_after(1):
            await session.call_tool(
                EXPECTED_TOOL_NAME,
                {},
                read_timeout_seconds=timedelta(milliseconds=50),
            )

        assert raised.value.error.code == 408
        assert "Timed out while waiting" in raised.value.error.message

    assert probe.stdout_protocol_errors == []


def test_slow_mode_exposes_real_client_timeout() -> None:
    anyio.run(slow_mode_times_out)


async def error_mode_is_structured() -> None:
    async with connected_client("error") as probe:
        session = active_session(probe)
        await session.initialize()
        tools = await session.list_tools()
        assert [tool.name for tool in tools.tools] == [EXPECTED_TOOL_NAME]

        result = await session.call_tool(EXPECTED_TOOL_NAME, {})
        assert result.isError is True
        assert result.structuredContent == EXPECTED_ERROR_OUTPUT
        assert_text_matches_structured(result)

    assert probe.stdout_protocol_errors == []


def test_error_mode_returns_structured_tool_error() -> None:
    anyio.run(error_mode_is_structured)


async def oversized_mode_is_exact_and_complete() -> None:
    async with connected_client("oversized") as probe:
        session = active_session(probe)
        await session.initialize()
        await session.list_tools()

        result = await session.call_tool(EXPECTED_TOOL_NAME, {})
        assert result.isError is False
        assert result.structuredContent is not None
        payload = result.structuredContent["payload"]
        assert isinstance(payload, str)
        encoded = payload.encode("ascii")
        assert len(encoded) == EXPECTED_OVERSIZED_PAYLOAD_BYTES
        assert hashlib.sha256(encoded).hexdigest() == (
            "81d0141af58569f91edf2d036b67a6b82c062d1829fd93aec50ed96b4773d225"
        )
        assert_text_matches_structured(result)

    assert probe.stdout_protocol_errors == []


def test_oversized_mode_transfers_fixed_256_kib_payload() -> None:
    anyio.run(oversized_mode_is_exact_and_complete)


def test_runtime_uses_the_exact_sdk_pin() -> None:
    assert version("mcp") == "1.27.1"
    assert version("neutral-mcp-server-fixture") == EXPECTED_SERVER_VERSION
    assert neutral_mcp_server.__version__ == EXPECTED_SERVER_VERSION
