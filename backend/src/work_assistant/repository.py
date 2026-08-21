from __future__ import annotations

import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import Select, case, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .authorization import ExactOwnershipAuthorizer, ResourceKind
from .execution_policy import (
    ExecutionOutcomeEvidence,
    ExecutionPlanEvidence,
    PolicyKernelConfigurationError,
    orphaned_run_outcome,
    validate_execution_outcome,
    validate_execution_plan,
)
from .identity import INTERNAL_SUBJECT_PREFIX, Principal
from .models import (
    EventRecord,
    MessageRecord,
    ModelAttemptRecord,
    RunRecord,
    ThreadRecord,
    utc_now,
)
from .schemas import (
    AccountRunCounts,
    AccountUsageResponse,
    AccountUsageScope,
    AccountView,
    EventEnvelope,
    Message,
    MessageRole,
    ProductEventType,
    RunFailureCode,
    RunSnapshot,
    RunStatus,
    RuntimeEventType,
    RunView,
    ThreadSnapshot,
    ThreadSummary,
    UsageRange,
    normalize_stored_product_event,
    validate_product_event,
    validate_runtime_event,
)
from .usage import (
    METERING_SCHEMA_VERSION,
    ModelAttemptFinish,
    ModelAttemptStart,
    ModelAttemptUsage,
    RunErrorCategory,
    RunUsage,
    UsageMeteringError,
    UsageMetric,
    aggregate_usage_metrics,
    unavailable_run_usage,
)

ACTIVE_STATUSES = ("created", "running")


class ResourceNotFoundError(Exception):
    pass


class ActiveRunConflictError(Exception):
    pass


class InitialThreadExistsError(Exception):
    pass


class IdempotencyMismatchError(Exception):
    pass


class RepositoryUnavailableError(RuntimeError):
    """No new transaction may start after the product pool fails closed."""


def _validated_terminal_outcome(
    value: ExecutionOutcomeEvidence,
    *,
    status: str,
    failure_code: RunFailureCode | None = None,
) -> dict[str, Any]:
    validated = validate_execution_outcome(value)
    if validated["status"] != status:
        raise PolicyKernelConfigurationError("execution_outcome_status_mismatch")
    if status == "failed" and validated["failure_code"] != failure_code:
        raise PolicyKernelConfigurationError("execution_outcome_failure_mismatch")
    return validated


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _initial_thread_title(message: str) -> str:
    visible = "".join(
        character
        for character in message
        if not unicodedata.category(character).startswith("C") or character.isspace()
    )
    collapsed = " ".join(visible.split()) or "New conversation"
    return f"{collapsed[:28]}…" if len(collapsed) > 28 else collapsed


def _thread_summary(record: ThreadRecord) -> ThreadSummary:
    return ThreadSummary(
        thread_id=record.id,
        title=record.title,
        created_at=_aware(record.created_at),
        updated_at=_aware(record.updated_at),
    )


def _message_view(record: MessageRecord) -> Message:
    return Message(
        message_id=record.id,
        role=cast(MessageRole, record.role),
        content=record.content,
        created_at=_aware(record.created_at),
        run_id=record.run_id,
    )


def _run_view(record: RunRecord) -> RunView:
    if record.metering_version is None:
        usage = unavailable_run_usage(state="unknown")
    elif record.usage_metrics is None:
        if record.status not in ACTIVE_STATUSES:
            raise UsageMeteringError("terminal_run_usage_missing")
        usage = unavailable_run_usage(state="pending")
    else:
        try:
            usage = RunUsage.model_validate(record.usage_metrics)
        except ValueError as exc:
            raise UsageMeteringError("run_usage_invalid") from exc
    return RunView(
        run_id=record.id,
        thread_id=record.thread_id,
        status=cast(RunStatus, record.status),
        last_seq=record.last_seq,
        created_at=_aware(record.created_at),
        completed_at=_aware(record.completed_at) if record.completed_at else None,
        usage=usage,
    )


def _event_view(record: EventRecord) -> EventEnvelope:
    return EventEnvelope(
        event_id=record.id,
        run_id=record.run_id,
        thread_id=record.thread_id,
        seq=record.seq,
        type=cast(ProductEventType, record.type),
        occurred_at=_aware(record.occurred_at),
        data=normalize_stored_product_event(record.type, record.payload),
    )


def _run_snapshot(record: RunRecord, events: list[EventEnvelope]) -> RunSnapshot:
    return RunSnapshot(**_run_view(record).model_dump(), events=events)


def _elapsed_ms(start: datetime, end: datetime | None) -> int | None:
    if end is None:
        return None
    return max(0, int((_aware(end) - _aware(start)).total_seconds() * 1_000))


def _attempt_token_metric(
    attempts: list[ModelAttemptRecord], field_name: str
) -> UsageMetric:
    if not attempts:
        return UsageMetric(value=0, availability="complete")
    values = [cast(int | None, getattr(attempt, field_name)) for attempt in attempts]
    known = [value for value in values if value is not None]
    if len(known) == len(values):
        return UsageMetric(value=sum(known), availability="complete")
    if known:
        return UsageMetric(value=None, availability="partial")
    return UsageMetric(value=None, availability="unavailable")


def _terminal_run_usage(
    run: RunRecord,
    attempts: list[ModelAttemptRecord],
    *,
    completed_at: datetime,
    error_category: RunErrorCategory | None,
) -> RunUsage:
    answer_attempts = [
        attempt for attempt in attempts if attempt.call_kind in {"direct", "finalizer"}
    ]
    generation_duration_ms: int | None = None
    if (
        len(answer_attempts) == 1
        and answer_attempts[0].status == "succeeded"
        and answer_attempts[0].completed_at is not None
    ):
        generation_duration_ms = _elapsed_ms(
            answer_attempts[0].started_at,
            answer_attempts[0].completed_at,
        )
    return RunUsage(
        schema_version=METERING_SCHEMA_VERSION,
        state="final",
        model_call_count=len(attempts),
        retry_count=sum(attempt.attempt_index > 1 for attempt in attempts),
        input_tokens=_attempt_token_metric(attempts, "input_tokens"),
        output_tokens=_attempt_token_metric(attempts, "output_tokens"),
        cached_tokens=_attempt_token_metric(attempts, "cached_tokens"),
        reasoning_tokens=_attempt_token_metric(attempts, "reasoning_tokens"),
        total_tokens=_attempt_token_metric(attempts, "total_tokens"),
        time_to_first_visible_ms=_elapsed_ms(run.created_at, run.first_visible_at),
        generation_duration_ms=generation_duration_ms,
        run_duration_ms=_elapsed_ms(run.created_at, completed_at),
        error_category=error_category,
    )


def _usage_count_metric(usage: RunUsage, field_name: str) -> UsageMetric:
    if usage.state == "final":
        value = cast(int | None, getattr(usage, field_name))
        if value is None:
            raise UsageMeteringError("final_usage_count_missing")
        return UsageMetric(value=value, availability="complete")
    return UsageMetric(value=None, availability=usage.state)


class ProductRepository:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sessions = session_factory
        self._fatal_error: str | None = None
        # Exact ownership is a non-replaceable security baseline. A future policy
        # hook may add denials, but it must never broaden this check.
        self._authorizer = ExactOwnershipAuthorizer()

    def fail_closed(self, reason: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = reason

    def _require_available(self) -> None:
        if self._fatal_error is not None:
            raise RepositoryUnavailableError(self._fatal_error)

    async def create_thread(self, *, principal: Principal, title: str | None) -> ThreadSnapshot:
        self._require_available()
        now = utc_now()
        record = ThreadRecord(
            id=str(uuid4()),
            owner_subject=principal.subject,
            title=title or "New conversation",
            created_at=now,
            updated_at=now,
        )
        async with self._sessions() as session, session.begin():
            session.add(record)
        return ThreadSnapshot(
            **_thread_summary(record).model_dump(),
            messages=[],
            runs=[],
            active_run=None,
        )

    async def create_initial_run(
        self,
        *,
        principal: Principal,
        thread_id: str,
        message: str,
        idempotency_key: str,
        execution_plan: ExecutionPlanEvidence,
    ) -> tuple[ThreadSummary, RunView, bool]:
        """Atomically persist a client-addressable Thread and its first Run."""

        self._require_available()
        validated_plan = validate_execution_plan(execution_plan)
        now = utc_now()
        thread = ThreadRecord(
            id=thread_id,
            owner_subject=principal.subject,
            title=_initial_thread_title(message),
            created_at=now,
            updated_at=now,
        )
        run = RunRecord(
            id=str(uuid4()),
            thread_id=thread_id,
            idempotency_key=idempotency_key,
            actor_subject=principal.subject,
            status="created",
            last_seq=0,
            execution_plan=validated_plan,
            execution_outcome=None,
            metering_version=METERING_SCHEMA_VERSION,
            first_visible_at=None,
            usage_metrics=None,
            created_at=now,
        )
        user_message = MessageRecord(
            id=str(uuid4()),
            thread_id=thread_id,
            run_id=run.id,
            role="user",
            content=message,
            created_at=now,
        )
        try:
            async with self._sessions() as session, session.begin():
                existing_thread = await session.scalar(
                    select(ThreadRecord).where(ThreadRecord.id == thread_id).with_for_update()
                )
                if existing_thread is not None:
                    existing_run = await self._resolve_initial_run_replay(
                        session,
                        thread=existing_thread,
                        principal=principal,
                        message=message,
                        idempotency_key=idempotency_key,
                    )
                    return _thread_summary(existing_thread), _run_view(existing_run), False

                # Flush each parent before its child while retaining one transaction.
                # PostgreSQL and FK-enabled SQLite therefore either commit all three
                # product records or leave no empty Thread behind.
                session.add(thread)
                await session.flush()
                session.add(run)
                await session.flush()
                session.add(user_message)
                await session.flush()
            return _thread_summary(thread), _run_view(run), True
        except IntegrityError:
            # A concurrent creator may have won the Thread primary-key race after
            # our initial lookup. Re-read the committed winner, re-authorize, and
            # return it only when this is an exact idempotent replay.
            async with self._sessions() as session:
                existing_thread = await session.get(ThreadRecord, thread_id)
                if existing_thread is None:
                    raise
                existing_run = await self._resolve_initial_run_replay(
                    session,
                    thread=existing_thread,
                    principal=principal,
                    message=message,
                    idempotency_key=idempotency_key,
                )
                return _thread_summary(existing_thread), _run_view(existing_run), False

    async def rename_thread(
        self,
        thread_id: str,
        *,
        principal: Principal,
        title: str,
    ) -> ThreadSummary:
        self._require_available()
        async with self._sessions() as session, session.begin():
            thread = await session.scalar(
                select(ThreadRecord).where(ThreadRecord.id == thread_id).with_for_update()
            )
            if thread is None:
                raise ResourceNotFoundError("thread")
            self._require_thread_owner(thread, principal)
            await self._require_thread_run_consistency(session, thread)
            thread.title = title
            thread.updated_at = utc_now()
            await session.flush()
        return _thread_summary(thread)

    async def list_threads(self, *, principal: Principal) -> list[ThreadSummary]:
        self._require_available()
        async with self._sessions() as session:
            records = (
                await session.scalars(
                    select(ThreadRecord)
                    .where(ThreadRecord.owner_subject == principal.subject)
                    .order_by(ThreadRecord.updated_at.desc(), ThreadRecord.id.desc())
                )
            ).all()
        return [_thread_summary(record) for record in records]

    async def get_account_usage(
        self,
        *,
        principal: Principal,
        usage_range: UsageRange,
        thread_id: str | None,
    ) -> AccountUsageResponse:
        """Read only the current Principal's aggregate product facts."""

        self._require_available()
        now = utc_now()
        cutoff = {
            "7d": now - timedelta(days=7),
            "30d": now - timedelta(days=30),
            "all": None,
        }[usage_range]
        async with self._sessions() as session:
            if thread_id is not None:
                owned_thread = await session.scalar(
                    select(ThreadRecord.id).where(
                        ThreadRecord.id == thread_id,
                        ThreadRecord.owner_subject == principal.subject,
                    )
                )
                if owned_thread is None:
                    # Foreign and absent scopes deliberately share one response.
                    raise ResourceNotFoundError("usage_scope")
            conditions = [
                ThreadRecord.owner_subject == principal.subject,
                RunRecord.actor_subject == principal.subject,
                RunRecord.created_at <= now,
            ]
            if cutoff is not None:
                conditions.append(RunRecord.created_at >= cutoff)
            if thread_id is not None:
                conditions.append(RunRecord.thread_id == thread_id)
            records = list(
                (
                    await session.scalars(
                        select(RunRecord)
                        .join(ThreadRecord, ThreadRecord.id == RunRecord.thread_id)
                        .where(*conditions)
                        .order_by(RunRecord.created_at, RunRecord.id)
                    )
                ).all()
            )

        usages = [_run_view(record).usage for record in records]
        statuses = [record.status for record in records]
        return AccountUsageResponse(
            account=AccountView(
                display_name=(principal.display_name or "当前用户"),
                organization=principal.organization,
                extensions=principal.display_extensions,
            ),
            scope=AccountUsageScope(
                range=usage_range,
                from_at=cutoff,
                to_at=now,
                thread_id=thread_id,
            ),
            runs=AccountRunCounts(
                total=len(records),
                completed=statuses.count("completed"),
                failed=statuses.count("failed"),
                cancelled=statuses.count("cancelled"),
                active=sum(status in ACTIVE_STATUSES for status in statuses),
            ),
            model_calls=aggregate_usage_metrics(
                _usage_count_metric(usage, "model_call_count") for usage in usages
            ),
            retries=aggregate_usage_metrics(
                _usage_count_metric(usage, "retry_count") for usage in usages
            ),
            input_tokens=aggregate_usage_metrics(usage.input_tokens for usage in usages),
            output_tokens=aggregate_usage_metrics(usage.output_tokens for usage in usages),
            cached_tokens=aggregate_usage_metrics(usage.cached_tokens for usage in usages),
            reasoning_tokens=aggregate_usage_metrics(
                usage.reasoning_tokens for usage in usages
            ),
            total_tokens=aggregate_usage_metrics(usage.total_tokens for usage in usages),
        )

    async def _resolve_initial_run_replay(
        self,
        session: AsyncSession,
        *,
        thread: ThreadRecord,
        principal: Principal,
        message: str,
        idempotency_key: str,
    ) -> RunRecord:
        self._require_thread_owner(thread, principal)
        await self._require_thread_run_consistency(session, thread)
        existing = await session.scalar(
            select(RunRecord).where(
                RunRecord.thread_id == thread.id,
                RunRecord.idempotency_key == idempotency_key,
            )
        )
        if existing is None:
            raise InitialThreadExistsError
        self._require_run_actor(existing, principal, resource_kind="thread")
        user_messages = (
            await session.scalars(
                select(MessageRecord).where(
                    MessageRecord.run_id == existing.id,
                    MessageRecord.thread_id == thread.id,
                    MessageRecord.role == "user",
                )
            )
        ).all()
        if len(user_messages) != 1:
            raise ResourceNotFoundError("run_ownership")
        if user_messages[0].content != message:
            raise IdempotencyMismatchError
        return existing

    async def get_thread(self, thread_id: str, *, principal: Principal) -> ThreadSnapshot:
        self._require_available()
        async with self._sessions() as session:
            thread = await session.get(ThreadRecord, thread_id)
            if thread is None:
                raise ResourceNotFoundError("thread")
            self._require_thread_owner(thread, principal)
            runs = (
                await session.scalars(
                    select(RunRecord)
                    .where(RunRecord.thread_id == thread_id)
                    .order_by(RunRecord.created_at, RunRecord.id)
                )
            ).all()
            for run in runs:
                self._require_run_actor(run, principal, resource_kind="thread")
            messages = (
                await session.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.thread_id == thread_id)
                    .order_by(MessageRecord.created_at, MessageRecord.id)
                )
            ).all()
            events = (
                await session.scalars(
                    select(EventRecord)
                    .where(EventRecord.thread_id == thread_id)
                    .order_by(EventRecord.run_id, EventRecord.seq)
                )
            ).all()
        events_by_run: dict[str, list[EventEnvelope]] = {}
        for event in events:
            events_by_run.setdefault(event.run_id, []).append(_event_view(event))
        active = next((run for run in runs if run.status in ACTIVE_STATUSES), None)
        return ThreadSnapshot(
            **_thread_summary(thread).model_dump(),
            messages=[_message_view(message) for message in messages],
            runs=[_run_snapshot(run, events_by_run.get(run.id, [])) for run in runs],
            active_run=_run_view(active) if active else None,
        )

    async def require_thread_access(self, thread_id: str, *, principal: Principal) -> None:
        """Authorize a Thread without reading its subject-scoped content."""

        self._require_available()
        async with self._sessions() as session:
            thread = await session.get(ThreadRecord, thread_id)
            if thread is None:
                raise ResourceNotFoundError("thread")
            self._require_thread_owner(thread, principal)

    async def create_run(
        self,
        *,
        principal: Principal,
        thread_id: str,
        message: str,
        idempotency_key: str,
        execution_plan: ExecutionPlanEvidence,
    ) -> tuple[RunView, bool]:
        self._require_available()
        validated_plan = validate_execution_plan(execution_plan)
        run = RunRecord(
            id=str(uuid4()),
            thread_id=thread_id,
            idempotency_key=idempotency_key,
            actor_subject=principal.subject,
            status="created",
            last_seq=0,
            execution_plan=validated_plan,
            execution_outcome=None,
            metering_version=METERING_SCHEMA_VERSION,
            first_visible_at=None,
            usage_metrics=None,
            created_at=utc_now(),
        )
        user_message = MessageRecord(
            id=str(uuid4()),
            thread_id=thread_id,
            run_id=run.id,
            role="user",
            content=message,
            created_at=run.created_at,
        )
        try:
            async with self._sessions() as session, session.begin():
                thread = await session.scalar(
                    select(ThreadRecord).where(ThreadRecord.id == thread_id).with_for_update()
                )
                if thread is None:
                    raise ResourceNotFoundError("thread")
                self._require_thread_owner(thread, principal)
                await self._require_thread_run_consistency(session, thread)
                existing = await session.scalar(
                    select(RunRecord).where(
                        RunRecord.thread_id == thread_id,
                        RunRecord.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    self._require_run_actor(existing, principal, resource_kind="thread")
                    return _run_view(existing), False
                active = await session.scalar(self._active_run_query(thread_id))
                if active is not None and active.idempotency_key == idempotency_key:
                    self._require_run_actor(active, principal, resource_kind="thread")
                    return _run_view(active), False
                if active is not None:
                    self._require_run_actor(active, principal, resource_kind="thread")
                    raise ActiveRunConflictError
                # These tables intentionally have no ORM ownership cascade. Flush the
                # referenced Run first so both PostgreSQL and FK-enabled SQLite see it.
                session.add(run)
                await session.flush()
                session.add(user_message)
                thread.updated_at = run.created_at
                await session.flush()
            return _run_view(run), True
        except IntegrityError:
            # Database constraints are the final arbiter for concurrent callers.
            async with self._sessions() as session:
                thread = await session.get(ThreadRecord, thread_id)
                if thread is None:
                    raise ResourceNotFoundError("thread") from None
                self._require_thread_owner(thread, principal)
                await self._require_thread_run_consistency(session, thread)
                existing = await session.scalar(
                    select(RunRecord).where(
                        RunRecord.thread_id == thread_id,
                        RunRecord.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    self._require_run_actor(existing, principal, resource_kind="thread")
                    return _run_view(existing), False
                active = await session.scalar(self._active_run_query(thread_id))
                if active is not None:
                    self._require_run_actor(active, principal, resource_kind="thread")
                    raise ActiveRunConflictError from None
            raise

    async def get_run(self, run_id: str, *, principal: Principal) -> RunView:
        self._require_available()
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_authorized_run(session, run, principal)
            return _run_view(run)

    async def get_run_evidence(
        self,
        run_id: str,
        *,
        principal: Principal,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return server-side audit evidence without exposing it in the product API."""

        self._require_available()
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_authorized_run(session, run, principal)
            plan = (
                validate_execution_plan(run.execution_plan)
                if run.execution_plan is not None
                else None
            )
            outcome = (
                validate_execution_outcome(run.execution_outcome)
                if run.execution_outcome is not None
                else None
            )
            return plan, outcome

    async def get_run_context(self, run_id: str) -> tuple[str, list[Message]]:
        self._require_available()
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            thread = await session.get(ThreadRecord, run.thread_id)
            if thread is None:
                raise ResourceNotFoundError("run_ownership")
            await self._require_internal_run_consistency(session, run)
            await self._require_thread_run_consistency(session, thread)
            records = (
                await session.scalars(
                    select(MessageRecord)
                    .join(RunRecord, MessageRecord.run_id == RunRecord.id)
                    .where(
                        MessageRecord.thread_id == run.thread_id,
                        or_(RunRecord.status == "completed", RunRecord.id == run_id),
                    )
                    .order_by(MessageRecord.created_at, MessageRecord.id)
                )
            ).all()
            messages = [_message_view(record) for record in records]
            if not any(message.run_id == run_id and message.role == "user" for message in messages):
                raise ResourceNotFoundError("message")
            return run.thread_id, messages

    async def start_run(self, run_id: str) -> bool:
        self._require_available()
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "created")
                .values(status="running", last_seq=RunRecord.last_seq + 1)
                .returning(RunRecord)
            )
            if run is None:
                if await session.get(RunRecord, run_id) is None:
                    raise ResourceNotFoundError("run")
                return False
            await self._require_internal_run_consistency(session, run)
            session.add(self._event_record(run, "run.started", {"status": "running"}))
        return True

    async def start_model_attempt(self, run_id: str, start: ModelAttemptStart) -> None:
        """Persist the auditable attempt before crossing the provider boundary."""

        self._require_available()
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(RunRecord).where(RunRecord.id == run_id).with_for_update()
            )
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_internal_run_consistency(session, run)
            if run.status != "running" or run.metering_version != METERING_SCHEMA_VERSION:
                raise UsageMeteringError("model_attempt_run_not_meterable")
            existing = await session.scalar(
                select(ModelAttemptRecord).where(
                    or_(
                        ModelAttemptRecord.id == start.attempt_id,
                        (
                            (ModelAttemptRecord.run_id == run_id)
                            & (ModelAttemptRecord.call_index == start.call_index)
                            & (ModelAttemptRecord.attempt_index == start.attempt_index)
                        ),
                    )
                )
            )
            if existing is not None:
                if (
                    existing.id == start.attempt_id
                    and existing.run_id == run_id
                    and existing.call_index == start.call_index
                    and existing.attempt_index == start.attempt_index
                    and existing.call_kind == start.call_kind
                    and existing.status == "started"
                ):
                    return
                raise UsageMeteringError("model_attempt_start_conflict")
            session.add(
                ModelAttemptRecord(
                    id=start.attempt_id,
                    run_id=run_id,
                    thread_id=run.thread_id,
                    call_index=start.call_index,
                    attempt_index=start.attempt_index,
                    call_kind=start.call_kind,
                    status="started",
                    started_at=utc_now(),
                )
            )

    async def finish_model_attempt(self, run_id: str, finish: ModelAttemptFinish) -> bool:
        """Close one attempt once; a winning Run terminal prevents late mutation."""

        self._require_available()
        usage = finish.usage
        token_values = {
            "input_tokens": usage.input_tokens if usage is not None else None,
            "output_tokens": usage.output_tokens if usage is not None else None,
            "cached_tokens": usage.cached_tokens if usage is not None else None,
            "reasoning_tokens": usage.reasoning_tokens if usage is not None else None,
            "total_tokens": usage.total_tokens if usage is not None else None,
        }
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(RunRecord).where(RunRecord.id == run_id).with_for_update()
            )
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_internal_run_consistency(session, run)
            attempt = await session.get(ModelAttemptRecord, finish.attempt_id)
            if attempt is None or attempt.run_id != run_id:
                raise UsageMeteringError("model_attempt_finish_unknown")
            if run.status not in ACTIVE_STATUSES:
                return False
            if attempt.status != "started":
                if attempt.status == finish.status and all(
                    getattr(attempt, field_name) == value
                    for field_name, value in token_values.items()
                ):
                    return True
                raise UsageMeteringError("model_attempt_finish_conflict")
            attempt.status = finish.status
            attempt.completed_at = utc_now()
            for field_name, value in token_values.items():
                setattr(attempt, field_name, value)
        return True

    async def record_model_attempt_usage(
        self,
        run_id: str,
        observed: ModelAttemptUsage,
    ) -> bool:
        """Persist an observed terminal usage object before Run cancellation."""

        self._require_available()
        usage = observed.usage
        token_values = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_tokens": usage.total_tokens,
        }
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(RunRecord).where(RunRecord.id == run_id).with_for_update()
            )
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_internal_run_consistency(session, run)
            attempt = await session.get(ModelAttemptRecord, observed.attempt_id)
            if attempt is None or attempt.run_id != run_id:
                raise UsageMeteringError("model_attempt_usage_unknown")
            if run.status not in ACTIVE_STATUSES:
                return False
            existing = {
                field_name: getattr(attempt, field_name)
                for field_name in token_values
            }
            if any(value is not None for value in existing.values()):
                if existing == token_values:
                    return True
                raise UsageMeteringError("model_attempt_usage_conflict")
            if attempt.status != "started":
                raise UsageMeteringError("model_attempt_usage_terminal")
            for field_name, value in token_values.items():
                setattr(attempt, field_name, value)
        return True

    async def _write_terminal_usage(
        self,
        session: AsyncSession,
        run: RunRecord,
        *,
        completed_at: datetime,
        open_attempt_status: str,
        error_category: RunErrorCategory | None,
    ) -> None:
        if run.metering_version is None:
            return
        if run.metering_version != METERING_SCHEMA_VERSION:
            raise UsageMeteringError("run_metering_version_unsupported")
        await session.execute(
            update(ModelAttemptRecord)
            .where(
                ModelAttemptRecord.run_id == run.id,
                ModelAttemptRecord.status == "started",
            )
            .values(status=open_attempt_status, completed_at=completed_at)
        )
        attempts = list(
            (
                await session.scalars(
                    select(ModelAttemptRecord)
                    .where(ModelAttemptRecord.run_id == run.id)
                    .order_by(
                        ModelAttemptRecord.call_index,
                        ModelAttemptRecord.attempt_index,
                    )
                )
            ).all()
        )
        run.usage_metrics = _terminal_run_usage(
            run,
            attempts,
            completed_at=completed_at,
            error_category=error_category,
        ).model_dump(mode="json")

    async def append_active_event(
        self, run_id: str, event_type: RuntimeEventType, data: dict[str, Any]
    ) -> EventEnvelope | None:
        self._require_available()
        event_type, validated = validate_runtime_event(event_type, data)
        now = utc_now()
        values: dict[str, Any] = {"last_seq": RunRecord.last_seq + 1}
        if event_type == "message.delta":
            # The atomic CASE makes the first persisted public delta the only
            # clock writer. SSE replay never touches this product-visible fact.
            values["first_visible_at"] = case(
                (RunRecord.first_visible_at.is_(None), now),
                else_=RunRecord.first_visible_at,
            )
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "running")
                .values(**values)
                .returning(RunRecord)
            )
            if run is None:
                if await session.get(RunRecord, run_id) is None:
                    raise ResourceNotFoundError("run")
                return None
            await self._require_internal_run_consistency(session, run)
            event = self._event_record(run, event_type, validated, occurred_at=now)
            session.add(event)
        return _event_view(event)

    async def complete_run(
        self,
        run_id: str,
        content: str,
        *,
        execution_outcome: ExecutionOutcomeEvidence,
        error_category: RunErrorCategory | None = None,
    ) -> RunView:
        self._require_available()
        validated_outcome = _validated_terminal_outcome(
            execution_outcome,
            status="completed",
        )
        now = utc_now()
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "running")
                .values(
                    status="completed",
                    completed_at=now,
                    last_seq=RunRecord.last_seq + 2,
                    execution_outcome=validated_outcome,
                )
                .returning(RunRecord)
            )
            if run is None:
                existing = await session.get(RunRecord, run_id)
                if existing is None:
                    raise ResourceNotFoundError("run")
                return _run_view(existing)
            await self._write_terminal_usage(
                session,
                run,
                completed_at=now,
                open_attempt_status="interrupted",
                error_category=error_category,
            )
            terminal_usage = _run_view(run).usage.model_dump(mode="json")
            thread = await self._require_internal_run_consistency(session, run)
            message = MessageRecord(
                id=str(uuid4()),
                thread_id=run.thread_id,
                run_id=run.id,
                role="assistant",
                content=content,
                created_at=now,
            )
            session.add(message)
            message_view = _message_view(message)
            session.add(
                self._event_record(
                    run,
                    "message.completed",
                    {"message": message_view.model_dump(mode="json")},
                    seq=run.last_seq - 1,
                    occurred_at=now,
                )
            )
            session.add(
                self._event_record(
                    run,
                    "run.completed",
                    {"status": "completed", "usage": terminal_usage},
                    occurred_at=now,
                )
            )
            thread.updated_at = now
        return _run_view(run)

    async def fail_run(
        self,
        run_id: str,
        error_code: RunFailureCode,
        *,
        execution_outcome: ExecutionOutcomeEvidence,
        error_category: RunErrorCategory = "internal",
    ) -> RunView:
        self._require_available()
        validated_outcome = _validated_terminal_outcome(
            execution_outcome,
            status="failed",
            failure_code=error_code,
        )
        now = utc_now()
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status.in_(ACTIVE_STATUSES))
                .values(
                    status="failed",
                    completed_at=now,
                    last_seq=RunRecord.last_seq + 1,
                    execution_outcome=validated_outcome,
                )
                .returning(RunRecord)
            )
            if run is None:
                existing = await session.get(RunRecord, run_id)
                if existing is None:
                    raise ResourceNotFoundError("run")
                return _run_view(existing)
            await self._write_terminal_usage(
                session,
                run,
                completed_at=now,
                open_attempt_status="failed",
                error_category=error_category,
            )
            terminal_usage = _run_view(run).usage.model_dump(mode="json")
            thread = await self._require_internal_run_consistency(session, run)
            session.add(
                self._event_record(
                    run,
                    "run.failed",
                    {
                        "status": "failed",
                        "error_code": error_code,
                        "usage": terminal_usage,
                    },
                    occurred_at=now,
                )
            )
            thread.updated_at = now
        return _run_view(run)

    async def cancel_run(
        self,
        run_id: str,
        *,
        principal: Principal,
        execution_outcome: ExecutionOutcomeEvidence,
        error_category: RunErrorCategory = "cancelled",
    ) -> RunView:
        self._require_available()
        validated_outcome = _validated_terminal_outcome(
            execution_outcome,
            status="cancelled",
        )
        now = utc_now()
        async with self._sessions() as session, session.begin():
            ownership = (
                await session.execute(
                    select(RunRecord.thread_id, RunRecord.actor_subject).where(
                        RunRecord.id == run_id
                    )
                )
            ).one_or_none()
            if ownership is None:
                raise ResourceNotFoundError("run")
            thread = await session.get(ThreadRecord, ownership.thread_id)
            if thread is None:
                raise ResourceNotFoundError("run")
            self._authorizer.require_owner(
                principal=principal,
                owner_subject=thread.owner_subject,
                resource_kind="run",
            )
            self._authorizer.require_owner(
                principal=principal,
                owner_subject=ownership.actor_subject,
                resource_kind="run",
            )
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status.in_(ACTIVE_STATUSES))
                .values(
                    status="cancelled",
                    completed_at=now,
                    last_seq=RunRecord.last_seq + 1,
                    execution_outcome=validated_outcome,
                )
                .returning(RunRecord)
            )
            if run is None:
                current = await session.get(RunRecord, run_id, populate_existing=True)
                if current is None:
                    raise ResourceNotFoundError("run")
                return _run_view(current)
            await self._write_terminal_usage(
                session,
                run,
                completed_at=now,
                open_attempt_status="cancelled",
                error_category=error_category,
            )
            terminal_usage = _run_view(run).usage.model_dump(mode="json")
            session.add(
                self._event_record(
                    run,
                    "run.cancelled",
                    {"status": "cancelled", "usage": terminal_usage},
                    occurred_at=now,
                )
            )
            thread.updated_at = now
        return _run_view(run)

    async def fail_orphaned_runs(self) -> list[RunView]:
        """Close work owned by a previous process before this executor accepts traffic.

        This sweep is intentionally scoped to the single-executor deployment supported by
        the current release. A multi-replica deployment requires ownership leases first.
        """

        self._require_available()
        now = utc_now()
        async with self._sessions() as session, session.begin():
            candidates = (
                await session.scalars(
                    select(RunRecord)
                    .where(RunRecord.status.in_(ACTIVE_STATUSES))
                    .order_by(RunRecord.created_at, RunRecord.id)
                )
            ).all()
            runs: list[RunRecord] = []
            for candidate in candidates:
                if candidate.execution_outcome is not None:
                    # A terminal outcome and an active status can never be
                    # produced by the atomic repository transitions. Refuse to
                    # overwrite that contradictory audit fact during recovery.
                    validate_execution_outcome(candidate.execution_outcome)
                    raise PolicyKernelConfigurationError("active_run_execution_outcome_present")
                execution_outcome: dict[str, Any] | None = None
                if candidate.execution_plan is not None:
                    # Corrupt modern evidence must stop startup rather than be
                    # reinterpreted. Legacy rows deliberately retain NULL facts.
                    validate_execution_plan(candidate.execution_plan)
                    source_events = (
                        await session.scalars(
                            select(EventRecord)
                            .where(
                                EventRecord.run_id == candidate.id,
                                EventRecord.type == "source.added",
                            )
                            .order_by(EventRecord.seq)
                        )
                    ).all()
                    source_ids: list[str] = []
                    for event in source_events:
                        source_data = validate_product_event(event.type, event.payload)
                        source_id = cast(str, source_data["source_id"])
                        if source_id not in source_ids:
                            source_ids.append(source_id)
                    execution_outcome = validate_execution_outcome(
                        orphaned_run_outcome(accepted_source_ids=tuple(source_ids))
                    )
                run = await session.scalar(
                    update(RunRecord)
                    .where(
                        RunRecord.id == candidate.id,
                        RunRecord.status.in_(ACTIVE_STATUSES),
                    )
                    .values(
                        status="failed",
                        completed_at=now,
                        last_seq=RunRecord.last_seq + 1,
                        execution_outcome=execution_outcome,
                    )
                    .returning(RunRecord)
                )
                if run is None:
                    continue
                await self._write_terminal_usage(
                    session,
                    run,
                    completed_at=now,
                    open_attempt_status="interrupted",
                    error_category="service",
                )
                terminal_usage = _run_view(run).usage.model_dump(mode="json")
                thread = await self._require_internal_run_consistency(
                    session, run, allow_internal=True
                )
                session.add(
                    self._event_record(
                        run,
                        "run.failed",
                        {
                            "status": "failed",
                            "error_code": "service_restarted",
                            "usage": terminal_usage,
                        },
                        occurred_at=now,
                    )
                )
                thread.updated_at = now
                runs.append(run)
        return [_run_view(run) for run in runs]

    async def get_events(
        self, run_id: str, after_seq: int, *, principal: Principal
    ) -> list[EventEnvelope]:
        self._require_available()
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_authorized_run(session, run, principal)
            records = (
                await session.scalars(
                    select(EventRecord)
                    .where(EventRecord.run_id == run_id, EventRecord.seq > after_seq)
                    .order_by(EventRecord.seq)
                )
            ).all()
        return [_event_view(record) for record in records]

    def _require_thread_owner(self, thread: ThreadRecord, principal: Principal) -> None:
        self._authorizer.require_owner(
            principal=principal,
            owner_subject=thread.owner_subject,
            resource_kind="thread",
        )

    def _require_run_actor(
        self,
        run: RunRecord,
        principal: Principal,
        *,
        resource_kind: ResourceKind,
    ) -> None:
        self._authorizer.require_owner(
            principal=principal,
            owner_subject=run.actor_subject,
            resource_kind=resource_kind,
        )

    async def _require_authorized_run(
        self,
        session: AsyncSession,
        run: RunRecord,
        principal: Principal,
    ) -> ThreadRecord:
        thread = await session.get(ThreadRecord, run.thread_id)
        if thread is None:
            raise ResourceNotFoundError("run")
        self._authorizer.require_owner(
            principal=principal,
            owner_subject=thread.owner_subject,
            resource_kind="run",
        )
        self._require_run_actor(run, principal, resource_kind="run")
        return thread

    async def _require_thread_run_consistency(
        self,
        session: AsyncSession,
        thread: ThreadRecord,
    ) -> None:
        inconsistent_run = await session.scalar(
            select(RunRecord.id)
            .where(
                RunRecord.thread_id == thread.id,
                RunRecord.actor_subject != thread.owner_subject,
            )
            .limit(1)
        )
        if inconsistent_run is not None:
            raise ResourceNotFoundError("run_ownership")

    async def _require_internal_run_consistency(
        self,
        session: AsyncSession,
        run: RunRecord,
        *,
        allow_internal: bool = False,
    ) -> ThreadRecord:
        thread = await session.get(ThreadRecord, run.thread_id)
        if thread is None or run.actor_subject != thread.owner_subject:
            raise ResourceNotFoundError("run_ownership")
        if not allow_internal and thread.owner_subject.startswith(INTERNAL_SUBJECT_PREFIX):
            raise ResourceNotFoundError("run_ownership")
        return thread

    @staticmethod
    def _active_run_query(thread_id: str) -> Select[tuple[RunRecord]]:
        return (
            select(RunRecord)
            .where(RunRecord.thread_id == thread_id, RunRecord.status.in_(ACTIVE_STATUSES))
            .order_by(RunRecord.created_at)
        )

    @staticmethod
    def _event_record(
        run: RunRecord,
        event_type: str,
        data: dict[str, Any],
        *,
        seq: int | None = None,
        occurred_at: datetime | None = None,
    ) -> EventRecord:
        validated = validate_product_event(event_type, data)
        return EventRecord(
            id=str(uuid4()),
            run_id=run.id,
            thread_id=run.thread_id,
            seq=run.last_seq if seq is None else seq,
            type=event_type,
            occurred_at=occurred_at or utc_now(),
            payload=validated,
        )
