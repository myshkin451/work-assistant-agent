from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from .repository import ActiveRunConflictError, ProductRepository, ResourceNotFoundError
from .schemas import (
    EventEnvelope,
    RunCreate,
    RunView,
    ThreadCreate,
    ThreadList,
    ThreadSnapshot,
)
from .service import RunService

router = APIRouter()


def _service(request: Request) -> RunService:
    return request.app.state.run_service  # type: ignore[no-any-return]


def _repository(request: Request) -> ProductRepository:
    return request.app.state.repository  # type: ignore[no-any-return]


def _not_found(resource: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": f"{resource}_not_found"})


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    await request.app.state.database.healthcheck()
    return {"status": "ok"}


@router.post("/api/threads", response_model=ThreadSnapshot, status_code=status.HTTP_201_CREATED)
async def create_thread(body: ThreadCreate, request: Request) -> ThreadSnapshot:
    return await _repository(request).create_thread(body.title)


@router.get("/api/threads", response_model=ThreadList)
async def list_threads(request: Request) -> ThreadList:
    return ThreadList(items=await _repository(request).list_threads())


@router.get("/api/threads/{thread_id}", response_model=ThreadSnapshot)
async def get_thread(thread_id: str, request: Request) -> ThreadSnapshot:
    try:
        return await _repository(request).get_thread(thread_id)
    except ResourceNotFoundError as exc:
        raise _not_found("thread") from exc


@router.post(
    "/api/threads/{thread_id}/runs",
    response_model=RunView,
    status_code=status.HTTP_201_CREATED,
)
async def create_run(thread_id: str, body: RunCreate, request: Request) -> RunView:
    try:
        return await _service(request).create_run(
            thread_id=thread_id,
            message=body.message,
            idempotency_key=body.idempotency_key,
        )
    except ResourceNotFoundError as exc:
        raise _not_found("thread") from exc
    except ActiveRunConflictError as exc:
        raise HTTPException(status_code=409, detail={"code": "thread_has_active_run"}) from exc


@router.post("/api/runs/{run_id}/cancel", response_model=RunView)
async def cancel_run(run_id: str, request: Request) -> RunView:
    try:
        return await _service(request).cancel_run(run_id)
    except ResourceNotFoundError as exc:
        raise _not_found("run") from exc


def _sse(event: EventEnvelope) -> str:
    payload = event.model_dump(mode="json")
    return (
        f"id: {event.seq}\n"
        f"event: {event.type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


@router.get("/api/runs/{run_id}/events")
async def events(
    run_id: str,
    request: Request,
    after_seq: int | None = Query(default=None, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    try:
        await _repository(request).get_run(run_id)
    except ResourceNotFoundError as exc:
        raise _not_found("run") from exc

    cursor = 0
    if after_seq is not None:
        cursor = after_seq
    elif last_event_id is not None:
        try:
            cursor = int(last_event_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_last_event_id"}) from exc
        if cursor < 0:
            raise HTTPException(status_code=422, detail={"code": "invalid_last_event_id"})

    async def generate() -> AsyncIterator[str]:
        async for event in _service(request).stream_events(
            run_id=run_id,
            after_seq=cursor,
            is_disconnected=request,
        ):
            if event is None:
                yield ": keep-alive\n\n"
            else:
                yield _sse(event)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
