from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

METERING_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

UsageAvailability = Literal["complete", "partial", "unavailable", "unknown", "pending"]
UsageState = Literal["final", "unknown", "pending"]
ModelCallKind = Literal["direct", "decision", "finalizer"]
ModelAttemptStatus = Literal["started", "succeeded", "failed", "cancelled", "interrupted"]
RunErrorCategory = Literal[
    "provider",
    "tool",
    "access_or_input",
    "limit",
    "validation",
    "timeout",
    "cancelled",
    "service",
    "internal",
]


class UsageMeteringError(RuntimeError):
    """Provider metering evidence is contradictory or cannot be attributed."""


class _FrozenUsageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderTokenUsage(_FrozenUsageModel):
    """Only fields explicitly present in one provider response.

    Missing fields stay ``None``. Callers must never derive one field from another.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_reported_field(self) -> ProviderTokenUsage:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("provider usage contains no reported field")
        return self


class UsageMetric(_FrozenUsageModel):
    value: int | None = Field(default=None, ge=0)
    availability: UsageAvailability

    @model_validator(mode="after")
    def validate_availability(self) -> UsageMetric:
        if self.availability == "complete" and self.value is None:
            raise ValueError("complete usage metric requires a value")
        if self.availability != "complete" and self.value is not None:
            raise ValueError("incomplete usage metric cannot expose a total")
        return self


class RunUsage(_FrozenUsageModel):
    schema_version: Literal["1.0.0"] | None
    state: UsageState
    model_call_count: int | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    input_tokens: UsageMetric
    output_tokens: UsageMetric
    cached_tokens: UsageMetric
    reasoning_tokens: UsageMetric
    total_tokens: UsageMetric
    time_to_first_visible_ms: int | None = Field(default=None, ge=0)
    generation_duration_ms: int | None = Field(default=None, ge=0)
    run_duration_ms: int | None = Field(default=None, ge=0)
    error_category: RunErrorCategory | None = None

    @model_validator(mode="after")
    def validate_state(self) -> RunUsage:
        metrics = (
            self.input_tokens,
            self.output_tokens,
            self.cached_tokens,
            self.reasoning_tokens,
            self.total_tokens,
        )
        if self.state == "final":
            if self.schema_version != METERING_SCHEMA_VERSION:
                raise ValueError("final usage requires the current schema version")
            if self.model_call_count is None or self.retry_count is None:
                raise ValueError("final usage requires exact call counts")
            if self.retry_count > self.model_call_count:
                raise ValueError("retry count exceeds model call count")
            if any(metric.availability in {"unknown", "pending"} for metric in metrics):
                raise ValueError("final usage cannot contain pending or legacy metrics")
        else:
            expected = self.state
            if self.schema_version is not None:
                raise ValueError("non-final usage cannot claim a schema version")
            if self.model_call_count is not None or self.retry_count is not None:
                raise ValueError("non-final usage cannot expose final call counts")
            if any(metric.availability != expected for metric in metrics):
                raise ValueError("non-final usage metrics must match state")
            if any(
                value is not None
                for value in (
                    self.time_to_first_visible_ms,
                    self.generation_duration_ms,
                    self.run_duration_ms,
                    self.error_category,
                )
            ):
                raise ValueError("non-final usage cannot expose terminal evidence")
        return self


class ModelAttemptStart(_FrozenUsageModel):
    attempt_id: str = Field(min_length=36, max_length=36)
    call_index: int = Field(ge=1)
    attempt_index: int = Field(ge=1)
    call_kind: ModelCallKind


class ModelAttemptFinish(_FrozenUsageModel):
    attempt_id: str = Field(min_length=36, max_length=36)
    status: Literal["succeeded", "failed", "cancelled"]
    usage: ProviderTokenUsage | None = None


class ModelAttemptUsage(_FrozenUsageModel):
    attempt_id: str = Field(min_length=36, max_length=36)
    usage: ProviderTokenUsage


AttemptStartWriter = Callable[[ModelAttemptStart], Awaitable[None]]
AttemptFinishWriter = Callable[[ModelAttemptFinish], Awaitable[None]]
AttemptUsageWriter = Callable[[ModelAttemptUsage], Awaitable[None]]
Clock = Callable[[], float]


@dataclass
class _Attempt:
    start: ModelAttemptStart
    started_at: float
    status: ModelAttemptStatus = "started"
    completed_at: float | None = None
    usage: ProviderTokenUsage | None = None
    usage_persisted: bool = False
    pending_finish: ModelAttemptFinish | None = None
    pending_completed_at: float | None = None


def unavailable_run_usage(*, state: Literal["unknown", "pending"]) -> RunUsage:
    metric = UsageMetric(value=None, availability=state)
    return RunUsage(
        schema_version=None,
        state=state,
        input_tokens=metric,
        output_tokens=metric,
        cached_tokens=metric,
        reasoning_tokens=metric,
        total_tokens=metric,
    )


def validate_run_usage(value: object) -> dict[str, object]:
    try:
        return RunUsage.model_validate(value).model_dump(mode="json")
    except ValueError as exc:
        raise UsageMeteringError("run_usage_invalid") from exc


def aggregate_usage_metrics(metrics: Iterable[UsageMetric]) -> UsageMetric:
    items = tuple(metrics)
    if not items:
        return UsageMetric(value=0, availability="complete")
    states = {item.availability for item in items}
    if "unknown" in states:
        return UsageMetric(value=None, availability="unknown")
    if "pending" in states:
        return UsageMetric(value=None, availability="pending")
    if states == {"complete"}:
        return UsageMetric(
            value=sum(item.value or 0 for item in items),
            availability="complete",
        )
    if "complete" in states or "partial" in states:
        return UsageMetric(value=None, availability="partial")
    return UsageMetric(value=None, availability="unavailable")


class RunUsageLedger:
    """Run-local provider-attempt ledger with optional durable writers."""

    def __init__(self, *, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._attempts: dict[str, _Attempt] = {}
        self._next_call_index = 1
        self._first_visible_at: float | None = None
        self._start_writer: AttemptStartWriter | None = None
        self._finish_writer: AttemptFinishWriter | None = None
        self._usage_writer: AttemptUsageWriter | None = None
        self._persistence_lock = asyncio.Lock()

    def bind_persistence(
        self,
        *,
        start_writer: AttemptStartWriter,
        finish_writer: AttemptFinishWriter,
        usage_writer: AttemptUsageWriter,
    ) -> None:
        if (
            self._start_writer is not None
            or self._finish_writer is not None
            or self._usage_writer is not None
        ):
            raise UsageMeteringError("attempt_persistence_already_bound")
        if self._attempts:
            raise UsageMeteringError("attempt_persistence_bound_after_start")
        self._start_writer = start_writer
        self._finish_writer = finish_writer
        self._usage_writer = usage_writer

    async def begin_attempt(
        self,
        *,
        call_kind: ModelCallKind,
        retry_of: str | None = None,
    ) -> str:
        if retry_of is None:
            call_index = self._next_call_index
            attempt_index = 1
            self._next_call_index += 1
        else:
            previous = self._attempts.get(retry_of)
            if previous is None or previous.status == "started":
                raise UsageMeteringError("retry_parent_invalid")
            call_index = previous.start.call_index
            attempt_index = previous.start.attempt_index + 1
            if any(
                attempt.start.call_index == call_index
                and attempt.start.attempt_index == attempt_index
                for attempt in self._attempts.values()
            ):
                raise UsageMeteringError("retry_attempt_duplicate")
        start = ModelAttemptStart(
            attempt_id=str(uuid4()),
            call_index=call_index,
            attempt_index=attempt_index,
            call_kind=call_kind,
        )
        if self._start_writer is not None:
            await self._start_writer(start)
        self._attempts[start.attempt_id] = _Attempt(start=start, started_at=self._clock())
        return start.attempt_id

    def observe_usage(self, attempt_id: str, usage: ProviderTokenUsage) -> None:
        attempt = self._require_started(attempt_id)
        if attempt.usage is None:
            attempt.usage = usage
            return
        if attempt.usage != usage:
            raise UsageMeteringError("provider_usage_conflict")

    async def finish_attempt(
        self,
        attempt_id: str,
        *,
        status: Literal["succeeded", "failed", "cancelled"],
    ) -> None:
        attempt = self._attempts.get(attempt_id)
        if attempt is None:
            raise UsageMeteringError("provider_attempt_unknown")
        if attempt.status != "started":
            if attempt.status == status:
                return
            raise UsageMeteringError("provider_attempt_terminal_conflict")
        finish = ModelAttemptFinish(
            attempt_id=attempt_id,
            status=status,
            usage=attempt.usage,
        )
        if attempt.pending_finish is None:
            attempt.pending_finish = finish
            attempt.pending_completed_at = self._clock()
        elif attempt.pending_finish != finish:
            raise UsageMeteringError("provider_attempt_terminal_conflict")
        async with self._persistence_lock:
            if attempt.status != "started":
                if attempt.status == status:
                    return
                raise UsageMeteringError("provider_attempt_terminal_conflict")
            await self._persist_pending_finish(attempt)

    async def settle_for_terminal(self) -> None:
        """Persist every usage fact observed before a Run terminal can win."""

        async with self._persistence_lock:
            for attempt in self._attempts.values():
                if attempt.status != "started":
                    continue
                if attempt.pending_finish is not None:
                    await self._persist_pending_finish(attempt)
                    continue
                if (
                    attempt.usage is not None
                    and not attempt.usage_persisted
                    and self._usage_writer is not None
                ):
                    await self._usage_writer(
                        ModelAttemptUsage(
                            attempt_id=attempt.start.attempt_id,
                            usage=attempt.usage,
                        )
                    )
                    attempt.usage_persisted = True

    async def _persist_pending_finish(self, attempt: _Attempt) -> None:
        finish = attempt.pending_finish
        completed_at = attempt.pending_completed_at
        if finish is None or completed_at is None:
            raise UsageMeteringError("provider_attempt_pending_finish_invalid")
        if self._finish_writer is not None:
            await self._finish_writer(finish)
        attempt.status = finish.status
        attempt.completed_at = completed_at
        attempt.usage_persisted = finish.usage is not None
        attempt.pending_finish = None
        attempt.pending_completed_at = None

    def record_first_visible(self) -> None:
        if self._first_visible_at is None:
            self._first_visible_at = self._clock()

    def snapshot(
        self,
        *,
        error_category: RunErrorCategory | None,
    ) -> RunUsage:
        attempts = tuple(self._attempts.values())
        return RunUsage(
            schema_version=METERING_SCHEMA_VERSION,
            state="final",
            model_call_count=len(attempts),
            retry_count=sum(item.start.attempt_index > 1 for item in attempts),
            input_tokens=self._token_metric(attempts, "input_tokens"),
            output_tokens=self._token_metric(attempts, "output_tokens"),
            cached_tokens=self._token_metric(attempts, "cached_tokens"),
            reasoning_tokens=self._token_metric(attempts, "reasoning_tokens"),
            total_tokens=self._token_metric(attempts, "total_tokens"),
            time_to_first_visible_ms=self._duration_ms(self._first_visible_at),
            generation_duration_ms=self._generation_duration_ms(attempts),
            run_duration_ms=self._duration_ms(self._clock()),
            error_category=error_category,
        )

    def _require_started(self, attempt_id: str) -> _Attempt:
        attempt = self._attempts.get(attempt_id)
        if attempt is None or attempt.status != "started":
            raise UsageMeteringError("provider_attempt_not_active")
        return attempt

    @staticmethod
    def _token_metric(attempts: tuple[_Attempt, ...], field_name: str) -> UsageMetric:
        if not attempts:
            return UsageMetric(value=0, availability="complete")
        values = [
            getattr(attempt.usage, field_name) if attempt.usage is not None else None
            for attempt in attempts
        ]
        known = [value for value in values if value is not None]
        if len(known) == len(values):
            return UsageMetric(value=sum(known), availability="complete")
        if known:
            return UsageMetric(value=None, availability="partial")
        return UsageMetric(value=None, availability="unavailable")

    def _duration_ms(self, end: float | None) -> int | None:
        if end is None:
            return None
        return max(0, int((end - self._started_at) * 1_000))

    @staticmethod
    def _generation_duration_ms(attempts: tuple[_Attempt, ...]) -> int | None:
        answer_attempts = [
            attempt for attempt in attempts if attempt.start.call_kind in {"direct", "finalizer"}
        ]
        if len(answer_attempts) != 1:
            return None
        attempt = answer_attempts[0]
        if attempt.status != "succeeded" or attempt.completed_at is None:
            return None
        return max(0, int((attempt.completed_at - attempt.started_at) * 1_000))
