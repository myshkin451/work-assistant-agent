# Current account and provider usage contract v1.0

This contract exposes only the current Principal's safe display projection and
provider-reported usage. It is a product-read boundary, not an administrator
console, billing ledger, cost estimator, or general metrics platform.

## Provider attempt facts

A new metered Run has `metering_version = "1.0.0"`. Before each request crosses
the model-provider boundary, the Host durably inserts one attempt identified by
its Run-local logical call and attempt index. A retry is a new attempt for the
same logical call; it is never folded into the previous attempt. The public
model-call count is the number of provider attempts actually started, and the
retry count is the number whose attempt index is greater than one. Neither value
is derived from Agent model steps.

An attempt accepts at most one canonical terminal usage object. Repeating the
same object is idempotent; a conflicting object is rejected. SSE replay,
idempotent Run creation, Thread refresh, and account reads only read these
facts, so they cannot increment usage. Cancellation or a winning Run terminal
closes any still-open attempt and rejects later mutation. Before cancellation
freezes the Run, it stops new provider progress and durably flushes any terminal
usage already observed in memory; this bounded settlement cannot be skipped in
favor of a faster but incomplete terminal snapshot.

The attempt record contains only its Run/Thread relationship, call kind,
indices, status, timestamps, and the five nullable Token values. It never stores
the raw provider response, provider request ID, Prompt, answer, Tool arguments or
result, credential, identity assertion, or internal Trace.

## Token truth and availability

The five public metrics are:

- `input_tokens`;
- `output_tokens`;
- `cached_tokens`, meaning provider-reported prompt-cache reads;
- `reasoning_tokens`;
- `total_tokens`.

Every value comes from a field explicitly present in the provider response. The
Host does not estimate Tokens from characters, infer a missing total from input
and output, treat model steps as Tokens, invent cache or reasoning values, or
derive a monetary cost. Provider adapters must preserve raw field presence even
when a framework normalizer would synthesize zeroes or totals.

Each metric has this shape:

```ts
type UsageMetric = {
  value: number | null;
  availability: "complete" | "partial" | "unavailable" | "unknown" | "pending";
};
```

- `complete`: every included attempt explicitly reported the field; `value` is
  their exact sum. An exact reported zero remains zero.
- `partial`: at least one included attempt reported the field and at least one
  did not; `value` stays `null` because the known subtotal is not the total.
- `unavailable`: no included attempt reported the field; `value` is `null`.
- `unknown`: the Run predates this metering contract; `value` is `null`.
- `pending`: the Run is not terminal; `value` is `null`.

A terminal metered Run with no provider attempt has exact call/retry counts of
zero and Token metrics of `complete / 0`: no provider call existed from which a
Token could be missing. For any started attempt that ends without terminal
usage, its Token fields remain unavailable and propagate to the Run and account
aggregate. A successful retry therefore cannot make an earlier uncertain
attempt look like zero usage.

## Run usage view

Every `RunView` includes:

```ts
type RunUsage = {
  schema_version: "1.0.0" | null;
  state: "final" | "unknown" | "pending";
  model_call_count: number | null;
  retry_count: number | null;
  input_tokens: UsageMetric;
  output_tokens: UsageMetric;
  cached_tokens: UsageMetric;
  reasoning_tokens: UsageMetric;
  total_tokens: UsageMetric;
  time_to_first_visible_ms: number | null;
  generation_duration_ms: number | null;
  run_duration_ms: number | null;
  error_category:
    | "provider"
    | "tool"
    | "access_or_input"
    | "limit"
    | "validation"
    | "timeout"
    | "cancelled"
    | "service"
    | "internal"
    | null;
};
```

Timing semantics are fixed for metered Runs only:

- time to first visible answer is from Run creation to the first successfully
  persisted public `message.delta`;
- generation duration is the one direct/finalizer provider stream from request
  start through accepted close; a failed/cancelled or ambiguous answer attempt
  has no generation duration;
- Run duration is from Run creation through the winning terminal transaction.

The stable error category is a bounded product fact, separate from the more
specific public failure code. It does not contain exception text. A completed
Run has no error category. Historical Runs keep the entire usage view
`unknown`; migration does not reinterpret old timestamps or fill Token zeroes.

The normalized final usage object is included in `run.completed`, `run.failed`,
and `run.cancelled` SSE event data. Its explicit `null` values are preserved and
match the REST `RunView`. Stored terminal events from older versions may omit
the optional `usage` object.

## Current-account read

`GET /api/account/usage?range={7d|30d|all}&thread_id={optional}` returns:

```ts
type AccountUsageResponse = {
  account: {
    display_name: string;
    organization: string | null;
    extensions: {
      session_expires_at: string | null;
      permission_summary: string | null;
    };
  };
  scope: {
    range: "7d" | "30d" | "all";
    from_at: string | null;
    to_at: string;
    thread_id: string | null;
  };
  runs: {
    total: number;
    completed: number;
    failed: number;
    cancelled: number;
    active: number;
  };
  model_calls: UsageMetric;
  retries: UsageMetric;
  input_tokens: UsageMetric;
  output_tokens: UsageMetric;
  cached_tokens: UsageMetric;
  reasoning_tokens: UsageMetric;
  total_tokens: UsageMetric;
};
```

The database query requires both the Thread owner and Run actor to equal the
current Principal. A supplied Thread scope must also be owned by that Principal;
foreign and absent Thread IDs both return `404 usage_scope_not_found`, preventing
a statistics-based existence signal. The response contains no Run IDs, message
or Prompt content, Tool content, source, owner/actor subject, role list, session
ID, provider payload, failure code, policy evidence, credential, or Trace.

The Host builds the account object by explicit projection rather than
serializing `Principal`. `session_expires_at` and `permission_summary` are strict,
pre-sanitized optional display extensions for neutral downstream identity
adapters. Built-in public development providers leave them empty; this public
repository contains no company session or permission implementation.
