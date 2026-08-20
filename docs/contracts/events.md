# Product API and event contract v0.5

The frontend uses the product contract below. It does not consume raw model,
LangChain, LangGraph, or provider events.

All product endpoints below require a current neutral Principal and enforce the
ownership rules in
[`identity-and-ownership.md`](identity-and-ownership.md). `/health` is public.

## Endpoints

- `POST /api/threads`
- `POST /api/threads/{client_thread_id}/initial-run`
- `GET /api/threads`
- `GET /api/threads/{thread_id}`
- `PATCH /api/threads/{thread_id}`
- `POST /api/threads/{thread_id}/runs`
- `GET /api/runs/{run_id}/events?after_seq={last_seen_seq}`
- `POST /api/runs/{run_id}/cancel`

Run creation accepts a user message and an idempotency key. Reusing the same
key in one thread returns the existing Run. A different key while that thread
has an active Run returns `409`.

## JSON resources

All identifiers are opaque strings and all timestamps are UTC ISO 8601 values.

```ts
type ThreadSummary = {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

type Message = {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  run_id: string | null;
};

type RunView = {
  run_id: string;
  thread_id: string;
  status: "created" | "running" | "completed" | "failed" | "cancelled";
  last_seq: number;
  created_at: string;
  completed_at: string | null;
};

type InitialRunResponse = {
  thread: ThreadSummary;
  run: RunView;
};

type ProductEventType =
  | "run.started"
  | "tool.started"
  | "tool.finished"
  | "message.delta"
  | "source.added"
  | "message.completed"
  | "run.completed"
  | "run.failed"
  | "run.cancelled";

type ProductEvent = {
  event_id: string;
  run_id: string;
  thread_id: string;
  seq: number;
  type: ProductEventType;
  occurred_at: string;
  data: Record<string, unknown>;
};

type RunSnapshot = RunView & {
  events: ProductEvent[];
};

type ThreadSnapshot = ThreadSummary & {
  messages: Message[];
  runs: RunSnapshot[];
  active_run: RunView | null;
};
```

`runs` is ordered by creation time. Each entry includes all persisted product
events for that Run in ascending `seq` order, including its terminal event.
Clients rebuild historical Tool, source, message, failure, and cancellation
state from this snapshot without starting Runtime work. An active Run is first
rebuilt from its snapshot events and then streamed from its last accepted
`seq`.

Request and response bodies:

- `POST /api/threads` accepts `{ "title"?: string }` and returns `ThreadSnapshot`.
  It remains a compatibility endpoint; the conversation workspace does not call
  it for an empty local draft.
- `POST /api/threads/{client_thread_id}/initial-run` accepts
  `{ "message": string, "idempotency_key": string }` and returns
  `InitialRunResponse`. `client_thread_id` must parse as a UUID and the Host
  canonicalizes it before persistence. The Host atomically creates the Thread,
  first Run, and user Message, derives a bounded title from the normalized first
  question, and then launches that Run.
- `GET /api/threads` returns `{ "items": ThreadSummary[] }`, newest first.
- `GET /api/threads/{thread_id}` returns `ThreadSnapshot`.
- `PATCH /api/threads/{thread_id}` accepts `{ "title": string }` and returns
  `ThreadSummary`. The title is a non-empty, single-line value of at most 200
  characters after deterministic whitespace normalization.
- `POST /api/threads/{thread_id}/runs` accepts
  `{ "message": string, "idempotency_key": string }` and returns `RunView`.
- `POST /api/runs/{run_id}/cancel` returns the current `RunView` after recording
  the cancellation request or terminal cancellation.

Thread creation, initial Run, rename, and ordinary Run bodies are strict. Unknown
fields, including a client-supplied Agent ID, budget, Principal role, or Tool
policy, return `422` and cannot alter the server-evaluated execution plan.

Authentication and resource errors are stable:

- `401 {"detail":{"code":"authentication_required"}}` when no valid Principal exists;
- `403 {"detail":{"code":"thread_forbidden"}}` for an existing foreign Thread;
- `403 {"detail":{"code":"run_forbidden"}}` for an existing foreign Run;
- `403 {"detail":{"code":"origin_forbidden"}}` for a browser mutation from
  an Origin outside the exact configured allowlist;
- `404` with `thread_not_found` or `run_not_found` for an unknown identifier.

Initial creation additionally uses two bounded conflicts after exact ownership
authorization: `409 idempotency_mismatch` when one key is replayed with different
message text, and `409 thread_already_exists` when an existing owned Thread ID is
present without that key. An exact same-ID, same-key, same-message replay returns
the original Run and never relaunches Runtime work.

Authentication occurs before product body or cursor validation. For an
authenticated caller, ownership is checked before idempotency replay, active-Run
conflict, cancellation, replay, or retry behavior. Therefore a forbidden caller
never receives another Principal's Run ID or a `409` hint. A different
idempotency key while the caller's own Thread has an active Run uses `409`.
Validation errors for an owned resource use `422`. Error bodies never include
resource content, owner/actor subjects, provider responses, prompts, stack
traces, or credentials.

Every browser `POST` or `PATCH` with an `Origin` header also passes the
exact-origin guard before mutation; service-to-service clients that omit
`Origin` remain supported.

## SSE envelope

Every data event is a JSON object with:

```json
{
  "event_id": "uuid",
  "run_id": "uuid",
  "thread_id": "uuid",
  "seq": 1,
  "type": "run.started",
  "occurred_at": "2026-08-12T12:00:00Z",
  "data": { "status": "running" }
}
```

The server also sets SSE `id` to `seq` and `event` to `type`. `seq` is strictly
increasing within a Run. Reconnecting with `after_seq` replays only persisted
events and never starts or resumes Agent execution by itself. The endpoint also
accepts the standard `Last-Event-ID` header; an explicit `after_seq` query value
takes precedence, otherwise the header is used, otherwise replay starts at `0`.
Every initial connection and reconnect authenticates and authorizes again. A
`401`, `403`, or `404` is returned as JSON before the `StreamingResponse` starts,
so it contains no SSE frame or heartbeat.
While a Run is active the server may send SSE comment heartbeats so proxies and
clients can detect liveness. Once a Run is terminal and all persisted events
after the requested sequence have been sent, the server closes the response.

Initial event types:

- `run.started`
- `tool.started`
- `tool.finished`
- `message.delta`
- `source.added`
- `message.completed`
- `run.completed`
- `run.failed`
- `run.cancelled`

Initial public event data:

- `run.started`: `{ "status": "running" }`
- `tool.started`: `{ "tool_call_id", "name", "label", "input_summary"? }`
- `tool.finished`: `{ "tool_call_id", "name", "label", "output_summary" }`
- `message.delta`: `{ "delta": string }`
- `source.added`: `{ "source_id", "label", "description" }`
- `message.completed`: `{ "message": Message }`
- `run.completed`: `{ "status": "completed" }`
- `run.failed`: `{ "status": "failed", "error_code": RunFailureCode }`
- `run.cancelled`: `{ "status": "cancelled" }`

The public failure codes are:

- `run_timeout`
- `agent_execution_failed`
- `service_restarted`
- `model_step_limit`
- `tool_call_limit`
- `repeated_tool_call`
- `no_progress`
- `tool_not_allowed`
- `result_schema_invalid`
- `source_validation_failed`

`stream_unavailable` is a client connection state, not a Run failure code or
terminal event. A failed Run remains immutable; retry creates a new Run with a
new idempotency key.

For upgrade compatibility only, a stored v0.1 `agent_result_missing` failure is
normalized to `agent_execution_failed` while reading. New writes still reject
that legacy code.

All event types and payloads are validated by the Host before sequence
allocation and again when read into the public response model. Unknown types,
extra payload fields, malformed values, and Runtime-private events are rejected.
The Runtime adapter may emit only `tool.started`, `tool.finished`,
`message.delta`, and `source.added`. Run lifecycle and `message.completed`
events are Host-owned and committed with product state. Tool and source events
are also checked against the current Run's reserved-call and successful-Tool
ledger before persistence; a structurally valid event cannot invent a Tool call
or source.

Agent and Tool-decision model rounds never produce public text deltas. Once the
Agent reaches a terminal draft, the Host makes one separately budgeted,
deadline-bounded presentation call against the same model without binding any
Tool or function schema. Only allowlisted answer-text chunks from that provider
stream may become `message.delta`; reasoning, Tool arguments, unknown blocks,
and raw provider metadata are never serialized. The first safe non-empty text
block is emitted while the provider stream is still open, including for a short
answer; subsequent text is coalesced into bounded blocks to reduce persistence
write amplification. A terminal-only adapter is rejected: the Host does not
split the completed answer after generation to simulate streaming. The ordered
delta concatenation must equal the committed `message.completed` content
byte-for-byte. The presentation call consumes one additional model step, so a
custom Agent budget must reserve that step; the Host never bypasses the declared
budget to obtain a terminal answer.

The public payload contains user-facing Tool purpose, bounded result summaries,
source references, text deltas, and errors. It never contains model reasoning,
credentials, raw provider events, hidden prompts, or internal checkpoint state.

## Conversation and restart boundaries

Every new Run receives prior product-committed messages from completed Runs in
the same Thread plus its current user message. Runtime checkpoints are scoped to
the product `run_id`; cancelled, failed, or crashed partial Runtime state is not
used as conversation history by a later Run.

On process startup, the current single-executor deployment closes persisted
`created` or `running` Runs as `failed / service_restarted` before accepting
traffic. This releases the Thread for an explicit new Run retry. It is not a
multi-replica ownership protocol; deployments with concurrent executors require
an ownership lease before using this startup sweep.
