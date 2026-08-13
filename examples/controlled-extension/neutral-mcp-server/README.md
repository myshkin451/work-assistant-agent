# Controlled neutral MCP server fixture

This directory is an isolated preparation Spike for a future controlled-extension task. It is
not wired into the Host, does not change the current T-004 entry point, and is not Stage 3
acceptance evidence.

## Boundary

The fixture is intentionally small:

- local `stdio` transport only;
- one zero-argument Tool, `read_neutral_record`;
- deterministic constants compiled into the package;
- no network calls, file access, environment/Secret reads, model calls, or company data;
- read-only, idempotent, closed-world annotations;
- behavior selected once at server startup, never through Tool arguments.

The implementation uses the official SDK's low-level `Server` and `stdio_server` APIs. It does
not use the separate third-party `fastmcp` package. The low-level API also avoids FastMCP's
settings layer and its `.env` lookup behavior, keeping configuration limited to the explicit
`--mode` process argument.

## SDK pin and evidence limit

Execution-date selection: `mcp==1.27.1`, locked transitively in `uv.lock`.

The task prohibited external network access. On 2026-08-13, `1.27.1` was the only official MCP
Python SDK release available in the local package cache. Its package metadata identifies the
`modelcontextprotocol/python-sdk` repository, and its PEP 440 version is a final release without
an alpha, beta, release-candidate, development, or local-version suffix. The cached wheel digest
is `sha256:1af3c4203b329430fde7a87b4fcb6392a041f5cb851fd68fc674016ab4e7c06f`.

This offline evidence proves the selected and locked artifact is an official, non-prerelease
release. It cannot prove that the public package index had no newer final release on 2026-08-13.
The upstream package metadata also retains the classifier `Development Status :: 4 - Beta`, so
this fixture does not claim that the SDK project as a whole has left beta maturity.

## Modes

Each process exposes the same Tool and schema. Restart the server with a different startup mode
to select a failure fixture:

| Mode | Deterministic behavior |
| --- | --- |
| `normal` | Returns the fixed neutral record and short payload. |
| `slow` | Waits 500 ms, then returns the normal result. |
| `error` | Returns `isError=true` with a fixed structured error object. |
| `oversized` | Returns a fixed 256 KiB ASCII payload. |

Examples:

```bash
uv run --offline --locked neutral-mcp-server --mode normal
uv run --offline --locked neutral-mcp-server --mode slow
uv run --offline --locked neutral-mcp-server --mode error
uv run --offline --locked neutral-mcp-server --mode oversized
```

The Tool input schema has no properties and rejects additional properties. In particular, a
caller cannot pass `mode` to alter server behavior.

## Verification

All protocol tests spawn the server as a real child process and use the official
`StdioServerParameters`, `stdio_client`, and `ClientSession` APIs. They perform
`initialize -> list_tools -> call_tool` over actual stdio; there is no mock or in-memory
transport.

From this directory:

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync ruff check .
uv run --locked --no-sync pytest
```

In an air-gapped environment, use the same commands with `--offline` only after an artifact
mirror or cache contains every wheel in the lock. This Spike itself stayed offline: the lock was
checked with `uv lock --offline --check`, the isolated environment was installed only from
locally cached artifacts, and the final `ruff` and `pytest` runs used `--offline --locked
--no-sync`.

The tests cover:

- initialization, the one-Tool list, annotations, schemas, and the normal result;
- rejection of a Tool argument that attempts to switch modes;
- a real client-side 408 timeout against a slow server;
- the fixed structured Tool error;
- exact transfer and SHA-256 verification of the 256 KiB payload;
- stdout protocol cleanliness by collecting every non-JSON message seen by the MCP client;
- server logging on stderr, separate from JSON-RPC stdout;
- the exact runtime MCP SDK version.

## Future Adapter use

After T-004 is complete and a dedicated Stage 3 MCP Adapter Task is active, the Host should run
this fixture as an external process; it should not import the fixture package into Host code.
Register a fixed Server ID and an administrator-controlled command, then start a fresh process
for each mode. The Adapter Task can use the fixture to verify allowlisting, discovery/schema
mapping, Tool filtering, structured error mapping, Host-enforced deadlines, output byte limits,
process cleanup, source/version evidence, and stdout protocol integrity.

Passing this fixture's tests proves only a local deterministic stdio MCP server/client contract.
It does not prove that a Host MCP Adapter, identity/policy intersection, runtime budget, source
recording, deployment, or any Stage 3 acceptance result exists.
