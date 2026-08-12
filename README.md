# Work Assistant Agent

Work Assistant Agent is a neutral, open-source Agent Host with a real employee
chat experience. It owns stable Thread, Run, Message, Event, Tool, cancellation,
replay, and persistence semantics while reusing LangChain and LangGraph for the
model/Tool loop.

The first vertical slice is deliberately small but complete: a user asks for
the current time in a named timezone, the model selects a read-only time Tool,
the UI streams product events and sources, and PostgreSQL preserves the result
for refresh and replay.

![Completed time Tool run](output/playwright/t003-normal-flow.png)

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
mode and no model credential. To use DeepSeek, set `MODEL_MODE=deepseek` and
provide `DEEPSEEK_API_KEY` only in a local ignored secret source.

Compose binds the database, API, and web UI to `127.0.0.1` only. This v0.1 has
no production authentication and must not be exposed directly to a LAN or the
public internet.

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

With `FAKE_STEP_DELAY_SECONDS=2` on the backend, the PostgreSQL concurrency
smoke verifies 20 duplicate idempotency requests and 10 competing active Runs:

```bash
python3 scripts/postgres_concurrency_smoke.py
```

## Public/private boundary

This repository contains no company adapters or private deployment material.
Downstream private integrations may depend on a released public version. This
repository never depends on those private integrations.

Third-party packages remain under their own licenses. The lockfile-based
inventory and distribution notes are recorded in
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

Licensed under the MIT License.
