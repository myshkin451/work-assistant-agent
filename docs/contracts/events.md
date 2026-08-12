# Product API and event contract v0.1

The frontend uses the product contract below. It does not consume raw model,
LangChain, LangGraph, or provider events.

## Endpoints

- `POST /api/threads`
- `GET /api/threads`
- `GET /api/threads/{thread_id}`
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

type ThreadSnapshot = ThreadSummary & {
  messages: Message[];
  active_run: RunView | null;
};
```

Request and response bodies:

- `POST /api/threads` accepts `{ "title"?: string }` and returns `ThreadSnapshot`.
- `GET /api/threads` returns `{ "items": ThreadSummary[] }`, newest first.
- `GET /api/threads/{thread_id}` returns `ThreadSnapshot`.
- `POST /api/threads/{thread_id}/runs` accepts
  `{ "message": string, "idempotency_key": string }` and returns `RunView`.
- `POST /api/runs/{run_id}/cancel` returns the current `RunView` after recording
  the cancellation request or terminal cancellation.

Missing resources use `404`. A different idempotency key while the thread has
an active Run uses `409`. Validation errors use `422`. Error bodies never include
provider responses, prompts, stack traces, or credentials.

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
  "data": {}
}
```

The server also sets SSE `id` to `seq` and `event` to `type`. `seq` is strictly
increasing within a Run. Reconnecting with `after_seq` replays only persisted
events and never starts or resumes Agent execution by itself. The endpoint also
accepts the standard `Last-Event-ID` header; an explicit `after_seq` query value
takes precedence, otherwise the header is used, otherwise replay starts at `0`.
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
- terminal Run events: `{ "status", "error_code"? }`

The public payload contains user-facing Tool purpose, bounded result summaries,
source references, text deltas, and errors. It never contains model reasoning,
credentials, raw provider events, hidden prompts, or internal checkpoint state.
