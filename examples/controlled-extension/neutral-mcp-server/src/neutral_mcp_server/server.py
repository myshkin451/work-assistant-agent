from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any, Literal, cast

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

SERVER_NAME = "neutral-mcp-server"
SERVER_VERSION = "0.1.0"
TOOL_NAME = "read_neutral_record"
OVERSIZED_PAYLOAD_BYTES = 262_144
SLOW_DELAY_SECONDS = 0.5

Mode = Literal["normal", "slow", "error", "oversized"]
MODES: tuple[Mode, ...] = ("normal", "slow", "error", "oversized")

FIXED_RECORD = {
    "id": "neutral-record-001",
    "label": "Controlled extension fixture",
    "category": "neutral-reference",
    "status": "available",
}
NORMAL_PAYLOAD = "deterministic-read-only-result"

SUCCESS_OUTPUT_SCHEMA: dict[str, Any] = {
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

ERROR_OUTPUT = {
    "error": {
        "code": "neutral_fixture_forced_error",
        "message": "The controlled fixture was started in error mode.",
        "retryable": False,
    }
}

LOGGER = logging.getLogger(SERVER_NAME)


def _json_text(value: dict[str, Any]) -> types.TextContent:
    return types.TextContent(
        type="text",
        text=json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    )


def _success_result(mode: Mode) -> types.CallToolResult:
    payload = "Z" * OVERSIZED_PAYLOAD_BYTES if mode == "oversized" else NORMAL_PAYLOAD
    output = {"record": dict(FIXED_RECORD), "payload": payload}
    return types.CallToolResult(
        content=[_json_text(output)],
        structuredContent=output,
        isError=False,
    )


def _error_result() -> types.CallToolResult:
    return types.CallToolResult(
        content=[_json_text(ERROR_OUTPUT)],
        structuredContent=ERROR_OUTPUT,
        isError=True,
    )


def create_server(mode: Mode) -> Server:
    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "A deterministic, read-only local fixture. It exposes one tool backed only by "
            "constants compiled into this package."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=TOOL_NAME,
                title="Read neutral fixture record",
                description="Return one fixed, neutral record without external access.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                outputSchema=SUCCESS_OUTPUT_SCHEMA,
                annotations=types.ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if name != TOOL_NAME:
            unknown_tool = {
                "error": {
                    "code": "neutral_fixture_unknown_tool",
                    "message": "The requested tool is not registered by this fixture.",
                    "retryable": False,
                }
            }
            return types.CallToolResult(
                content=[_json_text(unknown_tool)],
                structuredContent=unknown_tool,
                isError=True,
            )

        if arguments:
            invalid_arguments = {
                "error": {
                    "code": "neutral_fixture_invalid_arguments",
                    "message": "This tool accepts no arguments.",
                    "retryable": False,
                }
            }
            return types.CallToolResult(
                content=[_json_text(invalid_arguments)],
                structuredContent=invalid_arguments,
                isError=True,
            )

        LOGGER.info("serving tool=%s mode=%s", TOOL_NAME, mode)

        if mode == "slow":
            await anyio.sleep(SLOW_DELAY_SECONDS)
            return _success_result(mode)
        if mode == "error":
            return _error_result()
        return _success_result(mode)

    return server


async def _serve(mode: Mode) -> None:
    server = create_server(mode)
    LOGGER.info("starting server=%s version=%s mode=%s", SERVER_NAME, SERVER_VERSION, mode)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _parse_args(argv: Sequence[str] | None) -> Mode:
    parser = argparse.ArgumentParser(description="Run the controlled neutral MCP stdio fixture.")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="normal",
        help="Server-wide fixture behavior selected before the MCP session starts.",
    )
    return cast(Mode, parser.parse_args(argv).mode)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    anyio.run(_serve, _parse_args(argv))
