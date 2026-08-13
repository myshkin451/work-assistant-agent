from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any, Literal, cast

import anyio
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)

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


def _json_text(value: dict[str, Any]) -> TextContent:
    return TextContent(
        type="text",
        text=json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    )


def _success_result(mode: Mode) -> CallToolResult:
    payload = "Z" * OVERSIZED_PAYLOAD_BYTES if mode == "oversized" else NORMAL_PAYLOAD
    output = {"record": dict(FIXED_RECORD), "payload": payload}
    return CallToolResult(
        content=[_json_text(output)],
        structured_content=output,
        is_error=False,
    )


def _error_result() -> CallToolResult:
    return CallToolResult(
        content=[_json_text(ERROR_OUTPUT)],
        structured_content=ERROR_OUTPUT,
        is_error=True,
    )


def create_server(mode: Mode) -> Server:
    async def list_tools(
        _ctx: ServerRequestContext[Any],
        _params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=TOOL_NAME,
                    title="Read neutral fixture record",
                    description="Return one fixed, neutral record without external access.",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    output_schema=SUCCESS_OUTPUT_SCHEMA,
                    annotations=ToolAnnotations(
                        read_only_hint=True,
                        destructive_hint=False,
                        idempotent_hint=True,
                        open_world_hint=False,
                    ),
                )
            ]
        )

    async def call_tool(
        _ctx: ServerRequestContext[Any],
        params: CallToolRequestParams,
    ) -> CallToolResult:
        if params.name != TOOL_NAME:
            unknown_tool = {
                "error": {
                    "code": "neutral_fixture_unknown_tool",
                    "message": "The requested tool is not registered by this fixture.",
                    "retryable": False,
                }
            }
            return CallToolResult(
                content=[_json_text(unknown_tool)],
                structured_content=unknown_tool,
                is_error=True,
            )

        arguments = params.arguments or {}
        if arguments:
            invalid_arguments = {
                "error": {
                    "code": "neutral_fixture_invalid_arguments",
                    "message": "This tool accepts no arguments.",
                    "retryable": False,
                }
            }
            return CallToolResult(
                content=[_json_text(invalid_arguments)],
                structured_content=invalid_arguments,
                is_error=True,
            )

        LOGGER.info("serving tool=%s mode=%s", TOOL_NAME, mode)

        if mode == "slow":
            await anyio.sleep(SLOW_DELAY_SECONDS)
            return _success_result(mode)
        if mode == "error":
            return _error_result()
        return _success_result(mode)

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "A deterministic, read-only local fixture. It exposes one tool backed only by "
            "constants compiled into this package."
        ),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


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
