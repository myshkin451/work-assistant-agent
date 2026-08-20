from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .agent_definition import (
    AGENT_DEFINITION_SCHEMA_VERSION,
    AGENT_RESULT_SCHEMA_VERSION,
    IDENTIFIER_PATTERN,
    VERSION_PATTERN,
    AgentDefinition,
    ModelProfile,
)
from .capabilities import (
    CapabilityDecision,
    PrincipalCapabilityPolicy,
    ToolInvocationContractError,
    ToolRegistry,
)
from .context_builder import BuiltContext, ContextBuilder
from .identity import Principal
from .schemas import Message, RunFailureCode

EXECUTION_PLAN_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
EXECUTION_OUTCOME_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
RESULT_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


class PolicyKernelConfigurationError(RuntimeError):
    """Agent policy configuration failed closed before startup side effects."""


class PolicyViolation(RuntimeError):
    failure_code: RunFailureCode
    stop_reason: str

    def __init__(self, *, failure_code: RunFailureCode, stop_reason: str) -> None:
        super().__init__(stop_reason)
        self.failure_code = failure_code
        self.stop_reason = stop_reason


class ModelStepLimitExceeded(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(failure_code="model_step_limit", stop_reason="model_step_limit")


class ToolCallLimitExceeded(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(failure_code="tool_call_limit", stop_reason="tool_call_limit")


class RepeatedToolCall(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(failure_code="repeated_tool_call", stop_reason="repeated_tool_call")


class NoProgress(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(failure_code="no_progress", stop_reason="no_progress")


class ToolNotAllowed(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(failure_code="tool_not_allowed", stop_reason="tool_not_allowed")


class ResultSchemaInvalid(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(
            failure_code="result_schema_invalid",
            stop_reason="result_schema_invalid",
        )


class SourceValidationFailed(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(
            failure_code="source_validation_failed",
            stop_reason="source_validation_failed",
        )


class AgentExecutionFailed(PolicyViolation):
    def __init__(self, stop_reason: str = "agent_execution_failed") -> None:
        super().__init__(
            failure_code="agent_execution_failed",
            stop_reason=stop_reason,
        )


class RunDeadlineExceeded(PolicyViolation):
    def __init__(self) -> None:
        super().__init__(failure_code="run_timeout", stop_reason="run_timeout")


class _FrozenEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionedToolEvidence(_FrozenEvidence):
    tool_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)


class BudgetLimitEvidence(_FrozenEvidence):
    max_model_steps: int = Field(ge=1, le=32)
    max_tool_calls: int = Field(ge=0, le=16)
    deadline_seconds: float = Field(gt=0, le=900)
    max_identical_tool_calls: int = Field(ge=1, le=5)
    max_no_progress_steps: int = Field(ge=1, le=8)


class ResultContractEvidence(_FrozenEvidence):
    schema_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    max_answer_chars: int = Field(ge=1, le=32_000)
    source_policy: Literal["none", "required_if_tool_used", "required"]


class ContextLayerVersionsEvidence(_FrozenEvidence):
    host: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    agent: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    run: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    conversation: str = Field(min_length=1, max_length=64)
    tool_data: str = Field(min_length=1, max_length=64)


class ExecutionPlanEvidence(_FrozenEvidence):
    schema_version: Literal["1.0.0"] = EXECUTION_PLAN_SCHEMA_VERSION
    agent_schema_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    agent_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    agent_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    model_profile_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    model_profile_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    model_provider: str = Field(min_length=1, max_length=64, pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    prompt_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    prompt_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    context_builder_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    context_layer_versions: ContextLayerVersionsEvidence
    capability_policy_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    capability_policy_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    tool_registry_id: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    tool_registry_version: str = Field(min_length=5, max_length=32, pattern=VERSION_PATTERN)
    agent_allowed_tools: tuple[str, ...]
    base_tools: tuple[str, ...]
    visible_tools: tuple[VersionedToolEvidence, ...]
    budget: BudgetLimitEvidence
    result_contract: ResultContractEvidence

    @field_validator("agent_allowed_tools", "base_tools")
    @classmethod
    def validate_tool_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            len(tool_id) > 128 or re.fullmatch(IDENTIFIER_PATTERN, tool_id) is None
            for tool_id in value
        ):
            raise ValueError("execution plan Tool IDs are invalid")
        return value

    @model_validator(mode="after")
    def validate_capability_intersection(self) -> ExecutionPlanEvidence:
        allowed = set(self.agent_allowed_tools)
        base = set(self.base_tools)
        visible = [tool.tool_id for tool in self.visible_tools]
        if not base.issubset(allowed):
            raise ValueError("execution plan base Tools exceed Agent scope")
        if len(visible) != len(set(visible)) or not set(visible).issubset(base & allowed):
            raise ValueError("execution plan visible Tools exceed deterministic scope")
        return self


class BudgetUsageEvidence(_FrozenEvidence):
    model_steps: int = Field(ge=0)
    tool_calls_attempted: int = Field(ge=0)
    tool_calls_succeeded: int = Field(ge=0)
    repeated_tool_calls: int = Field(ge=0)
    no_progress_steps: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)


class ExecutionOutcomeEvidence(_FrozenEvidence):
    schema_version: Literal["1.0.0"] = EXECUTION_OUTCOME_SCHEMA_VERSION
    status: Literal["completed", "failed", "cancelled"]
    stop_reason: str = Field(min_length=1, max_length=128, pattern=IDENTIFIER_PATTERN)
    failure_code: RunFailureCode | None = None
    usage: BudgetUsageEvidence | None
    accepted_source_ids: tuple[str, ...] = ()
    result_source_ids: tuple[str, ...] = ()
    result_schema_version: str | None = Field(
        default=None, min_length=5, max_length=32, pattern=VERSION_PATTERN
    )
    result_validation: Literal["passed", "failed", "not_run"]

    @field_validator("accepted_source_ids", "result_source_ids")
    @classmethod
    def validate_accepted_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            not source_id or len(source_id) > 128 for source_id in value
        ):
            raise ValueError("accepted source IDs are invalid")
        return value

    @model_validator(mode="after")
    def validate_terminal_semantics(self) -> ExecutionOutcomeEvidence:
        if self.status == "failed" and self.failure_code is None:
            raise ValueError("a failed outcome requires a failure code")
        if self.status != "failed" and self.failure_code is not None:
            raise ValueError("only a failed outcome may have a failure code")
        if self.status == "completed":
            if self.result_validation != "passed" or self.usage is None:
                raise ValueError("a completed outcome requires a validated result and usage")
        if self.usage is None and self.failure_code != "service_restarted":
            raise ValueError("only a restart orphan may have unknown usage")
        if (self.result_validation == "not_run") != (self.result_schema_version is None):
            raise ValueError("result validation and schema evidence disagree")
        if not set(self.result_source_ids).issubset(self.accepted_source_ids):
            raise ValueError("result sources exceed the accepted source ledger")
        if self.result_validation != "passed" and self.result_source_ids:
            raise ValueError("an unvalidated result cannot record cited sources")
        return self


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = RESULT_SCHEMA_VERSION
    text: str = Field(min_length=1, max_length=32_000)
    source_ids: tuple[str, ...] = ()

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("result text must not be blank")
        return value

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("result source IDs must be unique")
        if any(not source_id or len(source_id) > 128 for source_id in value):
            raise ValueError("result source ID is invalid")
        return value


@dataclass(frozen=True)
class ProductEvent:
    type: str
    data: dict[str, Any]


@dataclass(frozen=True)
class _ToolReservation:
    tool_call_id: str
    tool_id: str
    validated_arguments: dict[str, Any]
    fingerprint: str


Clock = Callable[[], float]


class RunExecution:
    """One evaluated, Principal-scoped policy decision and its bounded ledger."""

    def __init__(
        self,
        *,
        principal: Principal,
        agent: AgentDefinition,
        model_profile: ModelProfile,
        capability_decision: CapabilityDecision,
        visible_tool_ids: tuple[str, ...],
        plan_evidence: ExecutionPlanEvidence,
        tool_registry: ToolRegistry,
        capability_policy: PrincipalCapabilityPolicy,
        context_builder: ContextBuilder,
        clock: Clock = time.monotonic,
    ) -> None:
        self.principal = principal
        self.agent = agent
        self.model_profile = model_profile
        self.capability_decision = capability_decision
        self.visible_tool_ids = visible_tool_ids
        self.plan_evidence = plan_evidence
        self.tool_registry = tool_registry
        self.capability_policy = capability_policy
        self.context_builder = context_builder
        self._clock = clock
        self._started_at = clock()
        self.deadline_at = self._started_at + agent.budget.deadline_seconds
        self._model_steps = 0
        self._tool_calls_attempted = 0
        self._tool_calls_succeeded = 0
        self._repeated_tool_calls = 0
        self._consecutive_no_progress = 0
        self._progress_revision = 0
        self._last_model_progress_revision = 0
        self._last_tool_fingerprint: str | None = None
        self._identical_tool_calls = 0
        self._seen_progress_fingerprints: set[str] = set()
        self._reservations: dict[str, _ToolReservation] = {}
        self._started_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self._finished_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        self._source_events: dict[str, tuple[set[str], dict[str, Any]]] = {}
        self._persisted_started: set[str] = set()
        self._persisted_finished: set[str] = set()
        self._persisted_sources: set[str] = set()
        self._persisted_source_order: list[str] = []
        # One stable source may be corroborated by more than one successful Tool
        # call (for example, two timezone reads share the same system-clock
        # provenance). Keep every contributing call while emitting the public
        # source fact once.
        self._accepted_sources: dict[str, set[str]] = {}
        self._result_validation: Literal["passed", "failed", "not_run"] = "not_run"
        self._result_source_ids: tuple[str, ...] = ()

    def build_context(self, messages: Sequence[Message]) -> BuiltContext:
        self.ensure_deadline()
        return self.context_builder.build(
            agent=self.agent,
            visible_tools=self.visible_tool_ids,
            conversation=messages,
        )

    def ensure_deadline(self) -> None:
        if self._clock() >= self.deadline_at:
            raise RunDeadlineExceeded

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - self._clock())

    def before_model_call(self) -> None:
        self.ensure_deadline()
        if self._model_steps > 0:
            if self._progress_revision == self._last_model_progress_revision:
                self._consecutive_no_progress += 1
                if self._consecutive_no_progress >= self.agent.budget.max_no_progress_steps:
                    raise NoProgress
            else:
                self._consecutive_no_progress = 0
        self._last_model_progress_revision = self._progress_revision
        self._model_steps += 1
        if self._model_steps > self.agent.budget.max_model_steps:
            raise ModelStepLimitExceeded

    def after_model_response(self, messages: Sequence[BaseMessage]) -> None:
        tool_calls: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            text = _content_text(message.content)
            if text:
                self._record_progress(f"assistant:{sha256(text.encode('utf-8')).hexdigest()}")
            tool_calls.extend(cast(list[dict[str, Any]], message.tool_calls))
        if tool_calls:
            self.reserve_tool_calls(tool_calls)

    def record_finalizer_signal(self) -> None:
        """Record one Host-private finalization decision without consuming Tool budget."""

        self.ensure_deadline()
        self._record_progress("host-control:finalize-answer")

    def reserve_tool_calls(self, tool_calls: Sequence[dict[str, Any]]) -> None:
        self.ensure_deadline()
        if not tool_calls:
            return
        self._tool_calls_attempted += len(tool_calls)
        if self._tool_calls_attempted > self.agent.budget.max_tool_calls:
            raise ToolCallLimitExceeded

        pending: list[_ToolReservation] = []
        last_fingerprint = self._last_tool_fingerprint
        identical_count = self._identical_tool_calls
        repeated_count = self._repeated_tool_calls
        known_call_ids = set(self._reservations) | set(self._started_calls)
        for call in tool_calls:
            tool_call_id = call.get("id")
            tool_id = call.get("name")
            arguments = call.get("args")
            if (
                not isinstance(tool_call_id, str)
                or not tool_call_id
                or len(tool_call_id) > 128
                or tool_call_id in known_call_ids
                or not isinstance(tool_id, str)
                or not isinstance(arguments, dict)
            ):
                raise AgentExecutionFailed("tool_call_invalid")
            known_call_ids.add(tool_call_id)
            if tool_id not in self.visible_tool_ids:
                raise ToolNotAllowed
            try:
                validated, fingerprint = self.tool_registry.canonicalize_call(tool_id, arguments)
            except ToolInvocationContractError as exc:
                raise AgentExecutionFailed("tool_arguments_invalid") from exc
            if fingerprint == last_fingerprint:
                identical_count += 1
                repeated_count += 1
            else:
                identical_count = 1
            last_fingerprint = fingerprint
            if identical_count > self.agent.budget.max_identical_tool_calls:
                self._last_tool_fingerprint = last_fingerprint
                self._identical_tool_calls = identical_count
                self._repeated_tool_calls = repeated_count
                raise RepeatedToolCall
            pending.append(
                _ToolReservation(
                    tool_call_id=tool_call_id,
                    tool_id=tool_id,
                    validated_arguments=validated,
                    fingerprint=fingerprint,
                )
            )

        self._last_tool_fingerprint = last_fingerprint
        self._identical_tool_calls = identical_count
        self._repeated_tool_calls = repeated_count
        self._reservations.update({item.tool_call_id: item for item in pending})

    def begin_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], ProductEvent]:
        self.ensure_deadline()
        reservation = self._reservations.get(tool_call_id)
        if (
            reservation is None
            or reservation.tool_id != tool_id
            or tool_call_id in self._started_calls
        ):
            raise ToolNotAllowed
        try:
            validated, fingerprint = self.tool_registry.canonicalize_call(tool_id, arguments)
        except ToolInvocationContractError as exc:
            raise AgentExecutionFailed("tool_arguments_invalid") from exc
        if fingerprint != reservation.fingerprint or validated != reservation.validated_arguments:
            raise ToolNotAllowed
        self._require_current_tool_permission(tool_id)
        record = self.tool_registry.require(tool_id)
        input_summary = record.summarize_input(validated)
        data: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "name": tool_id,
            "label": record.label,
        }
        if input_summary:
            data["input_summary"] = input_summary[:500]
        self._started_calls[tool_call_id] = (tool_id, data)
        return validated, ProductEvent("tool.started", data)

    def finish_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_id: str,
        message: ToolMessage,
    ) -> list[ProductEvent]:
        self.ensure_deadline()
        started = self._started_calls.get(tool_call_id)
        if (
            started is None
            or started[0] != tool_id
            or tool_call_id in self._finished_calls
            or message.tool_call_id != tool_call_id
            or (message.name is not None and message.name != tool_id)
        ):
            raise AgentExecutionFailed("tool_lifecycle_invalid")
        try:
            outcome = self.tool_registry.parse_tool_message(tool_id, message)
        except ToolInvocationContractError as exc:
            raise AgentExecutionFailed("tool_execution_failed") from exc
        record = self.tool_registry.require(tool_id)
        source_payloads = [source.model_dump(mode="json") for source in outcome.sources]
        for source, source_data in zip(outcome.sources, source_payloads, strict=True):
            existing = self._source_events.get(source.source_id)
            if existing is not None and existing[1] != source_data:
                raise SourceValidationFailed
        finished_data = {
            "tool_call_id": tool_call_id,
            "name": tool_id,
            "label": record.label,
            "output_summary": outcome.output_summary,
        }
        self._finished_calls[tool_call_id] = (tool_id, finished_data)
        self._tool_calls_succeeded += 1
        self._record_progress(f"tool-fact:{outcome.fact_fingerprint}")
        events = [ProductEvent("tool.finished", finished_data)]
        for source, source_data in zip(outcome.sources, source_payloads, strict=True):
            contributing_calls = self._accepted_sources.setdefault(source.source_id, set())
            contributing_calls.add(tool_call_id)
            existing = self._source_events.get(source.source_id)
            if existing is not None:
                existing[0].add(tool_call_id)
                continue
            self._source_events[source.source_id] = ({tool_call_id}, source_data)
            self._record_progress(f"source:{source.source_id}")
            events.append(ProductEvent("source.added", source_data))
        return events

    def validate_runtime_event(self, event: ProductEvent) -> None:
        """Validate a Runtime item without claiming that it reached durable storage."""

        if event.type == "message.delta":
            return
        if event.type == "tool.started":
            tool_call_id = event.data.get("tool_call_id")
            expected_started = self._started_calls.get(cast(str, tool_call_id))
            if (
                expected_started is None
                or expected_started[1] != event.data
                or tool_call_id in self._persisted_started
            ):
                raise ToolNotAllowed
            return
        if event.type == "tool.finished":
            tool_call_id = event.data.get("tool_call_id")
            expected_finished = self._finished_calls.get(cast(str, tool_call_id))
            if (
                expected_finished is None
                or expected_finished[1] != event.data
                or tool_call_id not in self._persisted_started
                or tool_call_id in self._persisted_finished
            ):
                raise AgentExecutionFailed("tool_lifecycle_invalid")
            return
        if event.type == "source.added":
            source_id = event.data.get("source_id")
            expected_source = self._source_events.get(cast(str, source_id))
            if (
                expected_source is None
                or expected_source[1] != event.data
                or not expected_source[0].intersection(self._persisted_finished)
                or source_id in self._persisted_sources
            ):
                raise SourceValidationFailed
            return
        raise AgentExecutionFailed("runtime_event_not_allowed")

    def accept_runtime_event(self, event: ProductEvent) -> None:
        """Confirm that a previously validated Runtime item was durably appended."""

        self.validate_runtime_event(event)
        if event.type == "message.delta":
            return
        if event.type == "tool.started":
            self._persisted_started.add(cast(str, event.data.get("tool_call_id")))
            return
        if event.type == "tool.finished":
            self._persisted_finished.add(cast(str, event.data.get("tool_call_id")))
            return
        if event.type == "source.added":
            source_id = cast(str, event.data.get("source_id"))
            self._persisted_sources.add(source_id)
            self._persisted_source_order.append(source_id)
            return
        raise AgentExecutionFailed("runtime_event_not_allowed")

    def validate_result(
        self,
        value: Any,
        *,
        runtime_text: str | None = None,
    ) -> AgentResult:
        try:
            if isinstance(value, AgentResult):
                result = AgentResult.model_validate(value.model_dump())
            else:
                result = AgentResult.model_validate(value)
        except (ValidationError, TypeError, ValueError) as exc:
            self._result_validation = "failed"
            raise ResultSchemaInvalid from exc
        if result.schema_version != self.agent.result_contract.schema_version:
            self._result_validation = "failed"
            raise ResultSchemaInvalid
        if len(result.text) > self.agent.result_contract.max_answer_chars:
            self._result_validation = "failed"
            raise ResultSchemaInvalid
        if runtime_text is not None and runtime_text != result.text:
            self._result_validation = "failed"
            raise ResultSchemaInvalid
        if any(source_id not in self._persisted_sources for source_id in result.source_ids):
            self._result_validation = "failed"
            raise SourceValidationFailed
        if self._tool_calls_succeeded == 0 and result.source_ids:
            self._result_validation = "failed"
            raise SourceValidationFailed
        source_policy = self.agent.result_contract.source_policy
        if source_policy == "required" and not result.source_ids:
            self._result_validation = "failed"
            raise SourceValidationFailed
        if (
            source_policy == "required_if_tool_used"
            and self._tool_calls_succeeded > 0
            and not result.source_ids
        ):
            self._result_validation = "failed"
            raise SourceValidationFailed
        self._result_validation = "passed"
        self._result_source_ids = result.source_ids
        return result

    def record_result_validation_failure(self) -> None:
        """Freeze a schema rejection raised before the final object validator."""

        self._result_validation = "failed"
        self._result_source_ids = ()

    def outcome(
        self,
        *,
        status: Literal["completed", "failed", "cancelled"],
        stop_reason: str,
        failure_code: RunFailureCode | None = None,
        usage_known: bool = True,
    ) -> ExecutionOutcomeEvidence:
        usage = self.usage() if usage_known else None
        return ExecutionOutcomeEvidence(
            status=status,
            stop_reason=stop_reason,
            failure_code=failure_code,
            usage=usage,
            accepted_source_ids=self.accepted_source_ids,
            result_source_ids=self._result_source_ids,
            result_schema_version=(
                self.agent.result_contract.schema_version
                if self._result_validation != "not_run"
                else None
            ),
            result_validation=self._result_validation,
        )

    def usage(self) -> BudgetUsageEvidence:
        return BudgetUsageEvidence(
            model_steps=self._model_steps,
            tool_calls_attempted=self._tool_calls_attempted,
            tool_calls_succeeded=self._tool_calls_succeeded,
            repeated_tool_calls=self._repeated_tool_calls,
            no_progress_steps=self._consecutive_no_progress,
            elapsed_ms=max(0, int((self._clock() - self._started_at) * 1_000)),
        )

    @property
    def accepted_source_ids(self) -> tuple[str, ...]:
        return tuple(self._persisted_source_order)

    @property
    def generated_source_ids(self) -> tuple[str, ...]:
        return tuple(self._accepted_sources)

    def _require_current_tool_permission(self, tool_id: str) -> None:
        try:
            raw_decision = self.capability_policy.decide(
                principal=self.principal,
                agent=self.agent,
                registered_enabled_tools=self.tool_registry.enabled_tool_ids,
            )
            decision = CapabilityDecision.model_validate(raw_decision.model_dump())
        except Exception as exc:
            raise ToolNotAllowed from exc
        if (
            decision.policy_id != self.capability_decision.policy_id
            or decision.policy_version != self.capability_decision.policy_version
            or tool_id not in decision.allowed_tools
            or tool_id not in self.visible_tool_ids
            or tool_id not in self.agent.allowed_tools
            or tool_id not in self.agent.base_tools
        ):
            raise ToolNotAllowed
        try:
            self.tool_registry.require(tool_id)
        except ToolInvocationContractError as exc:
            raise ToolNotAllowed from exc

    def _record_progress(self, fingerprint: str) -> None:
        if fingerprint in self._seen_progress_fingerprints:
            return
        self._seen_progress_fingerprints.add(fingerprint)
        self._progress_revision += 1


class AgentPolicyKernel:
    def __init__(
        self,
        *,
        definitions: Iterable[AgentDefinition],
        default_agent_id: str,
        model_profile: ModelProfile,
        tool_registry: ToolRegistry,
        capability_policy: PrincipalCapabilityPolicy,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        definition_map: dict[str, AgentDefinition] = {}
        for raw_definition in definitions:
            if (
                raw_definition.schema_version != AGENT_DEFINITION_SCHEMA_VERSION
                or raw_definition.result_contract.schema_version != AGENT_RESULT_SCHEMA_VERSION
            ):
                raise PolicyKernelConfigurationError("agent_schema_version_unsupported")
            try:
                definition = AgentDefinition.model_validate(raw_definition.model_dump())
            except (AttributeError, ValidationError) as exc:
                raise PolicyKernelConfigurationError("agent_definition_invalid") from exc
            if definition.agent_id in definition_map:
                raise PolicyKernelConfigurationError("duplicate_agent_definition")
            definition_map[definition.agent_id] = definition
        if not definition_map:
            raise PolicyKernelConfigurationError("agent_definition_missing")
        try:
            model_profile = ModelProfile.model_validate(model_profile.model_dump())
        except (AttributeError, ValidationError) as exc:
            raise PolicyKernelConfigurationError("model_profile_invalid") from exc
        default = definition_map.get(default_agent_id)
        if default is None:
            raise PolicyKernelConfigurationError("default_agent_unknown")
        if not default.enabled:
            raise PolicyKernelConfigurationError("default_agent_disabled")
        for definition in definition_map.values():
            if definition.model_profile != model_profile.profile_id:
                raise PolicyKernelConfigurationError("agent_model_profile_unknown")
            for tool_id in (*definition.allowed_tools, *definition.base_tools):
                try:
                    tool_registry.require(tool_id)
                except ToolInvocationContractError as exc:
                    raise PolicyKernelConfigurationError(
                        "agent_tool_reference_unavailable"
                    ) from exc
        try:
            CapabilityDecision(
                policy_id=capability_policy.policy_id,
                policy_version=capability_policy.version,
                allowed_tools=frozenset(),
            )
        except (AttributeError, ValidationError) as exc:
            raise PolicyKernelConfigurationError("capability_policy_invalid") from exc
        self._definitions = definition_map
        self.default_agent_id = default_agent_id
        self.model_profile = model_profile
        self.tool_registry = tool_registry
        self.capability_policy = capability_policy
        self.context_builder = context_builder or ContextBuilder()

    def resolve(self, agent_id: str | None = None) -> AgentDefinition:
        selected_id = agent_id or self.default_agent_id
        definition = self._definitions.get(selected_id)
        if definition is None:
            raise PolicyKernelConfigurationError("agent_definition_unknown")
        if not definition.enabled:
            raise PolicyKernelConfigurationError("agent_definition_disabled")
        return definition

    def prepare_run(
        self,
        *,
        principal: Principal,
        agent_id: str | None = None,
        clock: Clock = time.monotonic,
    ) -> RunExecution:
        agent = self.resolve(agent_id)
        try:
            raw_decision = self.capability_policy.decide(
                principal=principal,
                agent=agent,
                registered_enabled_tools=self.tool_registry.enabled_tool_ids,
            )
            decision = CapabilityDecision.model_validate(raw_decision.model_dump())
            if (
                decision.policy_id != self.capability_policy.policy_id
                or decision.policy_version != self.capability_policy.version
            ):
                raise ValueError("capability policy identity changed")
        except Exception:
            decision = CapabilityDecision(
                policy_id=self.capability_policy.policy_id,
                policy_version=self.capability_policy.version,
                allowed_tools=frozenset(),
            )
        visible = tuple(
            tool_id
            for tool_id in agent.base_tools
            if tool_id in self.tool_registry.enabled_tool_ids
            and tool_id in agent.allowed_tools
            and tool_id in decision.allowed_tools
        )
        context_probe = self.context_builder.build(
            agent=agent,
            visible_tools=visible,
            conversation=(),
        )
        plan = ExecutionPlanEvidence(
            agent_schema_version=agent.schema_version,
            agent_id=agent.agent_id,
            agent_version=agent.version,
            model_profile_id=self.model_profile.profile_id,
            model_profile_version=self.model_profile.version,
            model_provider=self.model_profile.provider,
            model_id=self.model_profile.model_id,
            prompt_id=agent.prompt.prompt_id,
            prompt_version=agent.prompt.version,
            prompt_sha256=agent.prompt.sha256,
            context_builder_version=self.context_builder.version,
            context_layer_versions=ContextLayerVersionsEvidence(**context_probe.layer_versions),
            capability_policy_id=decision.policy_id,
            capability_policy_version=decision.policy_version,
            tool_registry_id=self.tool_registry.registry_id,
            tool_registry_version=self.tool_registry.version,
            agent_allowed_tools=agent.allowed_tools,
            base_tools=agent.base_tools,
            visible_tools=tuple(
                VersionedToolEvidence(
                    tool_id=tool_id,
                    version=self.tool_registry.version_for(tool_id),
                )
                for tool_id in visible
            ),
            budget=BudgetLimitEvidence(**agent.budget.model_dump()),
            result_contract=ResultContractEvidence(**agent.result_contract.model_dump()),
        )
        return RunExecution(
            principal=principal,
            agent=agent,
            model_profile=self.model_profile,
            capability_decision=decision,
            visible_tool_ids=visible,
            plan_evidence=plan,
            tool_registry=self.tool_registry,
            capability_policy=self.capability_policy,
            context_builder=self.context_builder,
            clock=clock,
        )

    @property
    def framework_recursion_limit(self) -> int:
        maximum_model_steps = max(
            definition.budget.max_model_steps
            for definition in self._definitions.values()
            if definition.enabled
        )
        maximum_tool_calls = max(
            definition.budget.max_tool_calls
            for definition in self._definitions.values()
            if definition.enabled
        )
        # This is deliberately wider than every business budget. LangGraph counts
        # super-steps, including middleware nodes, so it is only a secondary fuse.
        return (maximum_model_steps + maximum_tool_calls + 4) * 3


EmitProductEvent = Callable[[ProductEvent], Awaitable[None]]
ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolMessage]]


async def execute_tool_call(
    *,
    execution: RunExecution,
    tool_call_id: str,
    tool_id: str,
    arguments: dict[str, Any],
    handler: ToolHandler,
    emit: EmitProductEvent,
) -> ToolMessage:
    validated, started = execution.begin_tool_call(
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        arguments=arguments,
    )
    await emit(started)
    try:
        message = await handler(validated)
    except PolicyViolation:
        raise
    except Exception as exc:
        raise AgentExecutionFailed("tool_execution_failed") from exc
    if not isinstance(message, ToolMessage):
        raise AgentExecutionFailed("tool_result_type_invalid")
    for event in execution.finish_tool_call(
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        message=message,
    ):
        await emit(event)
    return message


def orphaned_run_outcome(
    *,
    accepted_source_ids: tuple[str, ...] = (),
) -> ExecutionOutcomeEvidence:
    return ExecutionOutcomeEvidence(
        status="failed",
        stop_reason="service_restarted",
        failure_code="service_restarted",
        usage=None,
        accepted_source_ids=accepted_source_ids,
        result_source_ids=(),
        result_schema_version=None,
        result_validation="not_run",
    )


def validate_execution_plan(value: Any) -> dict[str, Any]:
    try:
        return ExecutionPlanEvidence.model_validate(value).model_dump(mode="json")
    except ValidationError as exc:
        raise PolicyKernelConfigurationError("execution_plan_invalid") from exc


def validate_execution_outcome(value: Any) -> dict[str, Any]:
    try:
        return ExecutionOutcomeEvidence.model_validate(value).model_dump(mode="json")
    except ValidationError as exc:
        raise PolicyKernelConfigurationError("execution_outcome_invalid") from exc


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
