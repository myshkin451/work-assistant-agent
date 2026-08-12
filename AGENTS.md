# AGENTS

This repository is the neutral, public Agent core. Keep every change runnable,
testable, and free of company-specific material.

## Before editing

1. Read `README.md`.
2. Read `docs/contracts/events.md` when changing API, events, runs, or frontend state.
3. Inspect `git status --short`; preserve unrelated work.
4. Run the narrowest relevant test while developing and `./scripts/verify.sh` before closing a task.

## Hard boundaries

- Never add company names, internal endpoints, identity mappings, private prompts,
  customer data, credentials, production logs, or copied private adapters.
- Secrets belong only in local ignored sources. `.env.example` contains names and safe defaults only.
- The frontend consumes this repository's REST and SSE contract, never provider or LangGraph private events.
- Product Thread, Run, Message, and Event state is distinct from Runtime checkpoints.
- Keep one active Run per thread, idempotent Run creation, monotonic event sequence,
  terminal-state immutability, replay without re-execution, and cancellation isolation.
- Do not add Redis, queues, Kubernetes, a second Runtime, multi-agent orchestration,
  long-term memory, arbitrary code execution, or admin platforms without a separate accepted task.
- Use migrations for product schema changes. LangGraph owns its checkpoint tables.

## Repository shape

- `backend/`: FastAPI Host, Agent Runtime adapter, storage, migrations, and tests.
- `frontend/`: React employee chat UI and tests.
- `docs/contracts/`: public product contracts.
- `scripts/verify.sh`: default offline verification.
- `compose.yaml`: reproducible local integration environment.
