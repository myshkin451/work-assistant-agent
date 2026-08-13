from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import Select, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .authorization import ExactOwnershipAuthorizer, ResourceKind
from .identity import INTERNAL_SUBJECT_PREFIX, Principal
from .models import EventRecord, MessageRecord, RunRecord, ThreadRecord, utc_now
from .schemas import (
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
    normalize_stored_product_event,
    validate_product_event,
    validate_runtime_event,
)

ACTIVE_STATUSES = ("created", "running")


class ResourceNotFoundError(Exception):
    pass


class ActiveRunConflictError(Exception):
    pass


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


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
    return RunView(
        run_id=record.id,
        thread_id=record.thread_id,
        status=cast(RunStatus, record.status),
        last_seq=record.last_seq,
        created_at=_aware(record.created_at),
        completed_at=_aware(record.completed_at) if record.completed_at else None,
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


class ProductRepository:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sessions = session_factory
        # Exact ownership is a non-replaceable security baseline. A future policy
        # hook may add denials, but it must never broaden this check.
        self._authorizer = ExactOwnershipAuthorizer()

    async def create_thread(
        self, *, principal: Principal, title: str | None
    ) -> ThreadSnapshot:
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

    async def list_threads(self, *, principal: Principal) -> list[ThreadSummary]:
        async with self._sessions() as session:
            records = (
                await session.scalars(
                    select(ThreadRecord)
                    .where(ThreadRecord.owner_subject == principal.subject)
                    .order_by(
                        ThreadRecord.updated_at.desc(), ThreadRecord.id.desc()
                    )
                )
            ).all()
        return [_thread_summary(record) for record in records]

    async def get_thread(self, thread_id: str, *, principal: Principal) -> ThreadSnapshot:
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

    async def require_thread_access(
        self, thread_id: str, *, principal: Principal
    ) -> None:
        """Authorize a Thread without reading its subject-scoped content."""

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
    ) -> tuple[RunView, bool]:
        run = RunRecord(
            id=str(uuid4()),
            thread_id=thread_id,
            idempotency_key=idempotency_key,
            actor_subject=principal.subject,
            status="created",
            last_seq=0,
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
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            await self._require_authorized_run(session, run, principal)
            return _run_view(run)

    async def get_run_context(self, run_id: str) -> tuple[str, list[Message]]:
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
            session.add(
                self._event_record(run, "run.started", {"status": "running"})
            )
        return True

    async def append_active_event(
        self, run_id: str, event_type: RuntimeEventType, data: dict[str, Any]
    ) -> EventEnvelope | None:
        event_type, validated = validate_runtime_event(event_type, data)
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "running")
                .values(last_seq=RunRecord.last_seq + 1)
                .returning(RunRecord)
            )
            if run is None:
                if await session.get(RunRecord, run_id) is None:
                    raise ResourceNotFoundError("run")
                return None
            await self._require_internal_run_consistency(session, run)
            event = self._event_record(run, event_type, validated)
            session.add(event)
        return _event_view(event)

    async def complete_run(self, run_id: str, content: str) -> RunView:
        now = utc_now()
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status == "running")
                .values(
                    status="completed",
                    completed_at=now,
                    last_seq=RunRecord.last_seq + 2,
                )
                .returning(RunRecord)
            )
            if run is None:
                existing = await session.get(RunRecord, run_id)
                if existing is None:
                    raise ResourceNotFoundError("run")
                return _run_view(existing)
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
                    {"status": "completed"},
                    occurred_at=now,
                )
            )
            thread.updated_at = now
        return _run_view(run)

    async def fail_run(self, run_id: str, error_code: RunFailureCode) -> RunView:
        now = utc_now()
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                update(RunRecord)
                .where(RunRecord.id == run_id, RunRecord.status.in_(ACTIVE_STATUSES))
                .values(
                    status="failed",
                    completed_at=now,
                    last_seq=RunRecord.last_seq + 1,
                )
                .returning(RunRecord)
            )
            if run is None:
                existing = await session.get(RunRecord, run_id)
                if existing is None:
                    raise ResourceNotFoundError("run")
                return _run_view(existing)
            thread = await self._require_internal_run_consistency(session, run)
            session.add(
                self._event_record(
                    run,
                    "run.failed",
                    {"status": "failed", "error_code": error_code},
                    occurred_at=now,
                )
            )
            thread.updated_at = now
        return _run_view(run)

    async def cancel_run(self, run_id: str, *, principal: Principal) -> RunView:
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
                )
                .returning(RunRecord)
            )
            if run is None:
                current = await session.get(RunRecord, run_id, populate_existing=True)
                if current is None:
                    raise ResourceNotFoundError("run")
                return _run_view(current)
            session.add(
                self._event_record(
                    run,
                    "run.cancelled",
                    {"status": "cancelled"},
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

        now = utc_now()
        async with self._sessions() as session, session.begin():
            runs = (
                await session.scalars(
                    update(RunRecord)
                    .where(RunRecord.status.in_(ACTIVE_STATUSES))
                    .values(
                        status="failed",
                        completed_at=now,
                        last_seq=RunRecord.last_seq + 1,
                    )
                    .returning(RunRecord)
                )
            ).all()
            for run in runs:
                thread = await self._require_internal_run_consistency(
                    session, run, allow_internal=True
                )
                session.add(
                    self._event_record(
                        run,
                        "run.failed",
                        {"status": "failed", "error_code": "service_restarted"},
                        occurred_at=now,
                    )
                )
                thread.updated_at = now
        return [_run_view(run) for run in runs]

    async def get_events(
        self, run_id: str, after_seq: int, *, principal: Principal
    ) -> list[EventEnvelope]:
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
