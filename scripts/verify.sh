#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$repo_dir/scripts/scan_public.py"

UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/work-assistant-agent-uv-cache}" \
  uv run --locked --project "$repo_dir/backend" ruff check "$repo_dir/backend"
UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/work-assistant-agent-uv-cache}" \
  uv run --locked --project "$repo_dir/backend" mypy "$repo_dir/backend/src"
UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/work-assistant-agent-uv-cache}" \
  uv run --locked --project "$repo_dir/backend" pytest -q "$repo_dir/backend/tests"

pnpm --dir "$repo_dir/frontend" install --frozen-lockfile
pnpm --dir "$repo_dir/frontend" lint
pnpm --dir "$repo_dir/frontend" typecheck
pnpm --dir "$repo_dir/frontend" test -- --run
pnpm --dir "$repo_dir/frontend" build

COMPOSE_DISABLE_ENV_FILE=1 docker compose --env-file /dev/null -f "$repo_dir/compose.yaml" config --quiet
