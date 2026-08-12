from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import Select, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import EventRecord, MessageRecord, RunRecord, ThreadRecord, utc_now
from .schemas import (
    EventEnvelope,
    Message,
    MessageRole,
    RunStatus,
    RunView,
    ThreadSnapshot,
    ThreadSummary,
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
        type=record.type,
        occurred_at=_aware(record.occurred_at),
        data=record.payload,
    )


class ProductRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def create_thread(self, title: str | None) -> ThreadSnapshot:
        now = utc_now()
        record = ThreadRecord(
            id=str(uuid4()),
            title=title or "New conversation",
            created_at=now,
            updated_at=now,
        )
        async with self._sessions() as session, session.begin():
            session.add(record)
        return ThreadSnapshot(**_thread_summary(record).model_dump(), messages=[], active_run=None)

    async def list_threads(self) -> list[ThreadSummary]:
        async with self._sessions() as session:
            records = (
                await session.scalars(
                    select(ThreadRecord).order_by(
                        ThreadRecord.updated_at.desc(), ThreadRecord.id.desc()
                    )
                )
            ).all()
        return [_thread_summary(record) for record in records]

    async def get_thread(self, thread_id: str) -> ThreadSnapshot:
        async with self._sessions() as session:
            thread = await session.get(ThreadRecord, thread_id)
            if thread is None:
                raise ResourceNotFoundError("thread")
            messages = (
                await session.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.thread_id == thread_id)
                    .order_by(MessageRecord.created_at, MessageRecord.id)
                )
            ).all()
            active = await session.scalar(self._active_run_query(thread_id))
        return ThreadSnapshot(
            **_thread_summary(thread).model_dump(),
            messages=[_message_view(message) for message in messages],
            active_run=_run_view(active) if active else None,
        )

    async def create_run(
        self, *, thread_id: str, message: str, idempotency_key: str
    ) -> tuple[RunView, bool]:
        run = RunRecord(
            id=str(uuid4()),
            thread_id=thread_id,
            idempotency_key=idempotency_key,
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
                existing = await session.scalar(
                    select(RunRecord).where(
                        RunRecord.thread_id == thread_id,
                        RunRecord.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    return _run_view(existing), False
                active = await session.scalar(self._active_run_query(thread_id))
                if active is not None and active.idempotency_key == idempotency_key:
                    return _run_view(active), False
                if active is not None:
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
                existing = await session.scalar(
                    select(RunRecord).where(
                        RunRecord.thread_id == thread_id,
                        RunRecord.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    return _run_view(existing), False
                if await session.scalar(self._active_run_query(thread_id)) is not None:
                    raise ActiveRunConflictError from None
            raise

    async def get_run(self, run_id: str) -> RunView:
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            return _run_view(run)

    async def get_run_input(self, run_id: str) -> tuple[str, str]:
        async with self._sessions() as session:
            run = await session.get(RunRecord, run_id)
            if run is None:
                raise ResourceNotFoundError("run")
            message = await session.scalar(
                select(MessageRecord).where(
                    MessageRecord.run_id == run_id,
                    MessageRecord.role == "user",
                )
            )
            if message is None:
                raise ResourceNotFoundError("message")
            return run.thread_id, message.content

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
            session.add(
                self._event_record(run, "run.started", {"status": "running"})
            )
        return True

    async def append_active_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> EventEnvelope | None:
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
            event = self._event_record(run, event_type, data)
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
            thread = await session.get(ThreadRecord, run.thread_id)
            if thread is not None:
                thread.updated_at = now
        return _run_view(run)

    async def fail_run(self, run_id: str, error_code: str) -> RunView:
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
            session.add(
                self._event_record(
                    run,
                    "run.failed",
                    {"status": "failed", "error_code": error_code},
                    occurred_at=now,
                )
            )
        return _run_view(run)

    async def cancel_run(self, run_id: str) -> RunView:
        now = utc_now()
        async with self._sessions() as session, session.begin():
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
                existing = await session.get(RunRecord, run_id)
                if existing is None:
                    raise ResourceNotFoundError("run")
                return _run_view(existing)
            session.add(
                self._event_record(
                    run,
                    "run.cancelled",
                    {"status": "cancelled"},
                    occurred_at=now,
                )
            )
        return _run_view(run)

    async def get_events(self, run_id: str, after_seq: int) -> list[EventEnvelope]:
        async with self._sessions() as session:
            if await session.get(RunRecord, run_id) is None:
                raise ResourceNotFoundError("run")
            records = (
                await session.scalars(
                    select(EventRecord)
                    .where(EventRecord.run_id == run_id, EventRecord.seq > after_seq)
                    .order_by(EventRecord.seq)
                )
            ).all()
        return [_event_view(record) for record in records]

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
        return EventRecord(
            id=str(uuid4()),
            run_id=run.id,
            thread_id=run.thread_id,
            seq=run.last_seq if seq is None else seq,
            type=event_type,
            occurred_at=occurred_at or utc_now(),
            payload=data,
        )
