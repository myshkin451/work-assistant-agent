from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.tools import BaseTool, tool
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .agent_definition import IDENTIFIER_PATTERN, VERSION_PATTERN, AgentDefinition
from .identity import Principal

TIME_SOURCE = {
    "source_id": "system-clock-iana-tzdb",
    "label": "System clock with IANA timezone data",
    "description": "Current server clock converted with the requested IANA timezone.",
}
_TERMINAL_TOOL_DESCRIPTION_SUFFIX = (
    " Host routing: this Tool is terminal-eligible. Include the Host finalization "
    "control in the same Tool-call batch only when that exact batch supplies every "
    "external fact needed for the user's complete request; otherwise continue planning."
)


class CapabilityConfigurationError(RuntimeError):
    """A registered capability or policy cannot be trusted at startup."""


class ToolInvocationContractError(RuntimeError):
    """A model Tool request or Tool result is outside its registered contract."""


class _FrozenCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSource(_FrozenCapability):
    source_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1_000)


class ParsedToolOutcome(_FrozenCapability):
    output_summary: str = Field(min_length=1, max_length=1_000)
    fact_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    sources: tuple[ToolSource, ...] = ()

    @field_validator("sources")
    @classmethod
    def validate_unique_sources(cls, value: tuple[ToolSource, ...]) -> tuple[ToolSource, ...]:
        source_ids = [source.source_id for source in value]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Tool sources must be unique")
        return value


InputSummarizer = Callable[[dict[str, Any]], str | None]
OutputParser = Callable[[ToolMessage], ParsedToolOutcome]


@dataclass(frozen=True)
class RegisteredTool:
    tool_id: str
    version: str
    enabled: bool
    label: str
    implementation: BaseTool = field(repr=False)
    summarize_input: InputSummarizer = field(repr=False)
    parse_output: OutputParser = field(repr=False)
    terminal_after_success: bool = False

    def __post_init__(self) -> None:
        if len(self.tool_id) > 128 or re.fullmatch(IDENTIFIER_PATTERN, self.tool_id) is None:
            raise CapabilityConfigurationError("registered_tool_invalid")
        if len(self.version) > 32 or re.fullmatch(VERSION_PATTERN, self.version) is None:
            raise CapabilityConfigurationError("registered_tool_invalid")
        if not isinstance(self.implementation, BaseTool):
            raise CapabilityConfigurationError("registered_tool_invalid")
        if self.implementation.name != self.tool_id:
            raise CapabilityConfigurationError("registered_tool_name_mismatch")
        if not self.label.strip() or len(self.label) > 200:
            raise CapabilityConfigurationError("registered_tool_invalid")
        if not callable(self.summarize_input) or not callable(self.parse_output):
            raise CapabilityConfigurationError("registered_tool_invalid")
        if not isinstance(self.terminal_after_success, bool):
            raise CapabilityConfigurationError("registered_tool_invalid")


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[RegisteredTool],
        *,
        registry_id: str = "builtin-tool-registry",
        version: str = "1.0.0",
    ) -> None:
        if (
            len(registry_id) > 128
            or re.fullmatch(IDENTIFIER_PATTERN, registry_id) is None
            or len(version) > 32
            or re.fullmatch(VERSION_PATTERN, version) is None
        ):
            raise CapabilityConfigurationError("tool_registry_invalid")
        records: dict[str, RegisteredTool] = {}
        for record in tools:
            if record.tool_id in records:
                raise CapabilityConfigurationError("duplicate_registered_tool")
            records[record.tool_id] = record
        if not records:
            raise CapabilityConfigurationError("empty_tool_registry")
        self.registry_id = registry_id
        self.version = version
        self._records = MappingProxyType(records)

    @property
    def enabled_tool_ids(self) -> frozenset[str]:
        return frozenset(record.tool_id for record in self._records.values() if record.enabled)

    @property
    def enabled_implementations(self) -> list[BaseTool]:
        implementations: list[BaseTool] = []
        for record in self._records.values():
            if not record.enabled:
                continue
            update: dict[str, Any] = {"return_direct": record.terminal_after_success}
            if record.terminal_after_success:
                update["description"] = (
                    record.implementation.description.rstrip() + _TERMINAL_TOOL_DESCRIPTION_SUFFIX
                )
            implementations.append(record.implementation.model_copy(update=update))
        return implementations

    def require(self, tool_id: str) -> RegisteredTool:
        record = self._records.get(tool_id)
        if record is None or not record.enabled:
            raise ToolInvocationContractError("tool_unavailable")
        return record

    def version_for(self, tool_id: str) -> str:
        return self.require(tool_id).version

    def canonicalize_call(
        self,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        record = self.require(tool_id)
        try:
            schema = record.implementation.get_input_schema()
            validated_model = cast(Any, schema).model_validate(arguments)
            validated = cast(Any, validated_model).model_dump(mode="json")
            encoded = json.dumps(
                validated,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception as exc:
            raise ToolInvocationContractError("tool_arguments_invalid") from exc
        fingerprint = sha256(f"{record.tool_id}@{record.version}:{encoded}".encode()).hexdigest()
        return validated, fingerprint

    def parse_tool_message(self, tool_id: str, message: ToolMessage) -> ParsedToolOutcome:
        if message.status != "success":
            raise ToolInvocationContractError("tool_execution_failed")
        record = self.require(tool_id)
        try:
            outcome = record.parse_output(message)
            return ParsedToolOutcome.model_validate(outcome.model_dump())
        except ToolInvocationContractError:
            raise
        except Exception as exc:
            raise ToolInvocationContractError("tool_output_invalid") from exc


class CapabilityDecision(_FrozenCapability):
    policy_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    policy_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    allowed_tools: frozenset[str]

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tool_ids(cls, value: frozenset[str]) -> frozenset[str]:
        if any(
            len(tool_id) > 128 or re.fullmatch(IDENTIFIER_PATTERN, tool_id) is None
            for tool_id in value
        ):
            raise ValueError("capability decision Tool IDs are invalid")
        return value


class PrincipalCapabilityPolicy(Protocol):
    policy_id: str
    version: str

    def decide(
        self,
        *,
        principal: Principal,
        agent: AgentDefinition,
        registered_enabled_tools: frozenset[str],
    ) -> CapabilityDecision: ...


class NeutralAuthenticatedToolPolicy:
    """Neutral default: authentication permits registered read-only public Tools.

    Agent and base-tool scopes are intersected separately by the Host, so this
    policy cannot expand either boundary. Downstream policies may only narrow it.
    """

    policy_id = "neutral-authenticated-tools"
    version = "1.0.0"

    def decide(
        self,
        *,
        principal: Principal,
        agent: AgentDefinition,
        registered_enabled_tools: frozenset[str],
    ) -> CapabilityDecision:
        del principal, agent
        return CapabilityDecision(
            policy_id=self.policy_id,
            policy_version=self.version,
            allowed_tools=registered_enabled_tools,
        )


def read_current_time(timezone_name: str) -> dict[str, str]:
    from datetime import UTC, datetime

    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("unknown IANA timezone") from exc
    now = datetime.now(UTC).astimezone(timezone)
    return {
        "timezone": timezone_name,
        "local_time": now.isoformat(timespec="seconds"),
        "utc_offset": now.strftime("%z"),
        "source_id": TIME_SOURCE["source_id"],
    }


@tool
async def get_current_time(timezone: str) -> str:
    """Return the current time for one valid IANA timezone, such as Asia/Shanghai."""

    return json.dumps(read_current_time(timezone), ensure_ascii=False)


def _tool_message_text(message: ToolMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _time_input_summary(arguments: dict[str, Any]) -> str:
    timezone = arguments.get("timezone")
    return timezone if isinstance(timezone, str) else ""


def _parse_time_output(message: ToolMessage) -> ParsedToolOutcome:
    content = _tool_message_text(message)
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ToolInvocationContractError("tool_output_invalid") from exc
    if not isinstance(data, dict) or set(data) != {
        "timezone",
        "local_time",
        "utc_offset",
        "source_id",
    }:
        raise ToolInvocationContractError("tool_output_invalid")
    if data.get("source_id") != TIME_SOURCE["source_id"] or not all(
        isinstance(data.get(key), str) and data[key]
        for key in ("timezone", "local_time", "utc_offset")
    ):
        raise ToolInvocationContractError("tool_output_invalid")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return ParsedToolOutcome(
        output_summary=f"{data['timezone']}: {data['local_time']}",
        fact_fingerprint=sha256(canonical.encode("utf-8")).hexdigest(),
        sources=(ToolSource(**TIME_SOURCE),),
    )


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            RegisteredTool(
                tool_id="get_current_time",
                version="1.1.0",
                enabled=True,
                label="Read current time",
                implementation=get_current_time,
                summarize_input=_time_input_summary,
                parse_output=_parse_time_output,
                terminal_after_success=True,
            )
        ]
    )
