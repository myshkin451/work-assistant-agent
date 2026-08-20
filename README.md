# Work Assistant Agent

Work Assistant Agent is a neutral, open-source Agent Host with a real employee
chat experience. It owns stable Thread, Run, Message, Event, Tool, cancellation,
replay, persistence, and subject-scoped ownership semantics while reusing
LangChain and LangGraph for the model/Tool loop.

Core 1.0 now also includes one versioned, startup-validated Agent definition, a
fixed-precedence Context Builder, deterministic Principal/Agent/base-Tool
capability intersection, bounded execution, validated result/source provenance,
and immutable per-Run policy evidence. See the
[`Agent policy kernel contract`](docs/contracts/agent-policy.md).

The first vertical slice is deliberately small but complete: every conversation
has a stable `/threads/{thread_id}` URL, a local blank draft is persisted only
with its first valid question, and the owner can switch, rename, refresh, or use
browser back/forward navigation. The model uses a read-only time Tool, emits
validated answer deltas while the final response is being generated, and renders
safe CommonMark/GFM. Live text is coalesced into short provider-received phrases
instead of repainting isolated Chinese characters; it is never reconstructed
from a completed answer or a client-side typing animation. The browser preserves
every received sequence while grouping rapid adjacent SSE frames into at most
one visible render per 60 milliseconds; terminal events flush immediately.
PostgreSQL preserves every turn and
monotonic SSE sequence for reconnect, replay, cancellation, source validation,
and failure recovery.

![T-008 live DeepSeek conversation workspace](output/playwright/t008/t008-deepseek-desktop.png)

The screenshot above is a loopback-only browser E4 capture from the locked
DeepSeek provider. Additional desktop scrolling and 390×844 mobile evidence is
catalogued in [`output/playwright/README.md`](output/playwright/README.md).

## Stack

- React, TypeScript, and Vite
- FastAPI and Python 3.12
- LangChain `create_agent` and LangGraph
- DeepSeek as the first live provider, with a deterministic fake mode for tests
- PostgreSQL 17 for product state and Runtime checkpoints
- REST plus a stable server-sent event contract
- Docker Compose for local integration

## Quick start

The Compose path only requires Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:5173`. The checked-in default uses deterministic fake
mode, no model credential, and the explicitly configured single-Principal
`anonymous` development identity. To use DeepSeek, set `MODEL_MODE=deepseek`
and provide `DEEPSEEK_API_KEY` only in a local ignored secret source.

`/` is an unsaved new-conversation view. A persisted conversation uses
`/threads/{thread_id}`; opening that URL directly or refreshing it reconstructs
the owner-scoped snapshot and resumes an active SSE stream from its last
accepted sequence.

Compose binds the database, API, and web UI to `127.0.0.1` only. Anonymous mode
is not a production identity system and must not be exposed to a LAN or the
public internet. The default server identity mode is `external`: a production
deployment must inject a real neutral `IdentityProvider`, and startup fails
closed if it does not. The public core defines no company login or token format.
See the
[`Principal and ownership contract`](docs/contracts/identity-and-ownership.md).

## Verification

Host-side verification additionally requires Python 3.12 with `uv`, Node 24,
and pnpm 11:

```bash
./scripts/verify.sh
```

The default verification does not read `.env` and does not call a model or
business service. A first dependency installation may access package registries;
use the committed lock files for reproducibility. Live DeepSeek and browser
acceptance are explicit lanes; neither is implied by a successful unit test.

With Compose running, exercise the persisted REST/SSE contract without printing
message content:

```bash
python3 scripts/compose_smoke.py
```

The T-008 lane additionally exercises atomic first-turn creation, concurrent
idempotent replay, a deliberate live SSE disconnect/resume, byte-exact delta /
terminal equality, rename persistence, and redacted first-delta timing:

```bash
python3 scripts/t008_stream_smoke.py \
  --lane t008_compose_e3 --model fake --initial-requests 20
```

For a live provider, add `--minimum-deltas 3` and use a redacted model label.
The output contains only event counts, character counts, timings, booleans, and
bounded identifiers; it never prints the prompt, answer, Tool result, or Secret.

For the two-Principal ownership lane, start an isolated Compose project with the
development header provider and a deliberately slow fake Run so cancellation is
observable, then run the bounded identity smoke:

```bash
COMPOSE_DISABLE_ENV_FILE=1 APP_ENV=development \
  IDENTITY_PROVIDER_MODE=development_header MODEL_MODE=fake \
  FAKE_STEP_DELAY_SECONDS=1 \
  docker compose --env-file /dev/null -p work-assistant-t005-e3 up --build
python3 scripts/identity_compose_smoke.py
```

The smoke covers owner-filtered list/detail, independent same-key idempotency,
cross-subject denial, cancellation, retry, SSE `after_seq`, `Last-Event-ID`, and
re-authentication on reconnect. The development subject header is confined to
loopback tests; it is not placed in frontend configuration, URLs, browser
storage, product payloads, or smoke output.

The bounded T-006 Compose lane uses the same Fake and Runtime policy hooks,
checks stable stop codes and source provenance, and inspects only redacted
server-side policy evidence:

```bash
set -euo pipefail
export COMPOSE_DISABLE_ENV_FILE=1 APP_ENV=development
export IDENTITY_PROVIDER_MODE=development_header MODEL_MODE=fake
export MAX_MODEL_STEPS=3 MAX_TOOL_CALLS=2 MAX_IDENTICAL_TOOL_CALLS=1
export MAX_NO_PROGRESS_STEPS=2 RUN_TIMEOUT_SECONDS=2
export DATABASE_OPERATION_TIMEOUT_SECONDS=2
export REPOSITORY_CLEANUP_GRACE_SECONDS=3 FAKE_STEP_DELAY_SECONDS=0.02
export WORK_ASSISTANT_API_URL=http://127.0.0.1:8000
export WORK_ASSISTANT_DATABASE_URL=postgresql://work_assistant:work_assistant@127.0.0.1:55432/work_assistant
T006_COMPOSE=(docker compose --env-file /dev/null -p work-assistant-t006-e3)

"${T006_COMPOSE[@]}" down --volumes --remove-orphans
"${T006_COMPOSE[@]}" build backend frontend
"${T006_COMPOSE[@]}" up -d --wait --wait-timeout 120 postgres
"${T006_COMPOSE[@]}" run --rm --no-deps backend \
  uv run --no-sync alembic upgrade 0002_principal_ownership
backend/.venv/bin/python scripts/policy_compose_smoke.py prepare-v02-legacy
"${T006_COMPOSE[@]}" run --rm --no-deps backend \
  uv run --no-sync alembic upgrade head
backend/.venv/bin/python scripts/policy_compose_smoke.py verify-v03-legacy
"${T006_COMPOSE[@]}" up -d --no-build --wait --wait-timeout 120 backend frontend
backend/.venv/bin/python scripts/policy_compose_smoke.py exercise
```

The fixed two-second deadline keeps the deliberate timeout case bounded without
turning container scheduling into a 300 ms performance gate. The explicit
database timeout and cleanup grace prevent ambient shell values from weakening
the fail-stop contract. Development-header identity is required for the two
neutral Principals; none of these settings is a production default.

For a real PostgreSQL `0002 → 0003` upgrade proof, run the script's
`prepare-v02-legacy` mode at revision `0002_principal_ownership`, upgrade to
head, and then run `verify-v03-legacy`. The probe confirms that historical Runs
remain honestly unversioned. The backend image normally upgrades to head before
serving, so the migration proof must use the one-off commands above before the
backend service starts. Migration `0003_agent_policy_evidence` takes an
exclusive PostgreSQL table lock before its downgrade guard and refuses to
discard any recorded T-006 evidence. Run it only in an application downtime
window even though the database guard is the final safety boundary.

Migration `0002_principal_ownership` requires a live database connection but an
exclusive application downtime window. It preserves v0.2 rows under an
unclaimable internal quarantine subject; it does not support offline `--sql`
generation or a non-empty downgrade to the unauthenticated schema.

The phase-2 smoke uses one Thread for Shanghai, London, and New York, deliberately
disconnects and resumes the first SSE stream, checks idempotent replay, and then
rebuilds all three Runs from the persisted Thread snapshot:

```bash
python3 scripts/phase2_compose_smoke.py conversation
```

For the redacted live-model lane, explicitly start Compose with a local secret,
independently confirm the effective provider/model inside the backend, and use
distinct evidence labels:

```bash
cp .env.example .env
# Edit the ignored .env locally: set MODEL_MODE=deepseek,
# DEEPSEEK_API_KEY, and DEEPSEEK_MODEL=deepseek-v4-flash.
docker compose up -d --force-recreate backend frontend
docker compose exec -T backend python -c \
  'from work_assistant.settings import get_settings; s=get_settings(); print(s.model_mode, s.deepseek_model)'
python3 scripts/phase2_compose_smoke.py conversation \
  --lane deepseek_multiturn_e4 --model deepseek-v4-flash
```

The CLI `--model` value is an operator-supplied redacted evidence label; the
container check above is the independent configuration proof. The smoke output
contains only that label, timings, event types, counts, and boolean contract
checks, never the credential, answers, Tool output, or exception details.

To exercise hard-restart recovery, start a long fake Run, record the bounded
IDs printed by `restart-create`, kill only the backend, restart it without
deleting the PostgreSQL volume, and then verify the old and retry Runs:

```bash
FAKE_STEP_DELAY_SECONDS=10 docker compose up -d --force-recreate backend
python3 scripts/phase2_compose_smoke.py restart-create
docker compose kill -s SIGKILL backend
FAKE_STEP_DELAY_SECONDS=0.02 docker compose up -d --no-deps backend
python3 scripts/phase2_compose_smoke.py restart-verify \
  --thread-id '<thread_id>' --run-id '<run_id>'
```

With `FAKE_STEP_DELAY_SECONDS=2` on the backend, the PostgreSQL concurrency
smoke verifies 20 duplicate idempotency requests and 10 competing active Runs:

```bash
python3 scripts/postgres_concurrency_smoke.py
```

The included Compose topology is a single execution instance. On startup it
closes orphaned active Runs as `service_restarted`; a multi-replica deployment
must add explicit Run ownership before using that startup sweep.

The public-core Stage 3 milestone is intentionally split into four independently
accepted slices: principal and ownership safety, the Agent policy kernel,
controlled neutral Skill/MCP extension, and one downstream assembly and release
path. This repository does not treat a standalone neutral MCP fixture as Host
integration or Stage 3 acceptance evidence.

## Public/private boundary

This repository contains no company adapters or private deployment material.
Downstream private integrations may depend on a released public version. This
repository never depends on those private integrations.

Third-party packages remain under their own licenses. The lockfile-based
inventory and distribution notes are recorded in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Licensed under the MIT License.
