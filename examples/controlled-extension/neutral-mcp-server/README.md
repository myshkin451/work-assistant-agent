# Controlled neutral MCP server fixture

This directory is an isolated preparation Spike for a future controlled-extension task. It is
not wired into the Host, does not change the current T-005 entry point, and is not Stage 3
acceptance evidence.

## Boundary

The fixture is intentionally small:

- local `stdio` transport only;
- one zero-argument Tool, `read_neutral_record`;
- deterministic constants compiled into the package;
- no network calls, file access, environment/Secret reads, model calls, or company data;
- read-only, idempotent, closed-world annotations;
- behavior selected once at server startup, never through Tool arguments.

The implementation uses the official SDK v2 low-level `Server` and `stdio_server` APIs. It does
not use the separate third-party `fastmcp` package. Low-level v2 handlers are registered through
the Server constructor, return full protocol result objects, and validate the zero-argument
contract explicitly. Configuration remains limited to the explicit `--mode` process argument.

## SDK pin and evidence limit

Execution-date selection: `mcp==2.0.0`, locked with its exact `mcp-types==2.0.0` dependency in
`uv.lock`.

The official [PyPI release](https://pypi.org/project/mcp/2.0.0/) identifies v2 as the current
stable release line, records a 2026-07-28 release date, and classifies it as
`Development Status :: 5 - Production/Stable`. PyPI's verified provenance links the artifacts to
the official `modelcontextprotocol/python-sdk` tag `v2.0.0` at commit
`6f69a3758ebf2ee55ce050f58b470ce11af71133`. The locked universal wheel digest is
`sha256:1cb4c75d2d2c7b8c1d756355e5d82a39f2822cc7f13e22a2051d7ca3592349d6`.

The migration follows the official [v1 to v2 guide](https://py.sdk.modelcontextprotocol.io/migration/):
constructor-based low-level handlers, snake-case Python attributes, explicit low-level input
validation, float timeouts, and `REQUEST_TIMEOUT` (`-32001`). MCP v2's `stdio_server` moves the
protocol stream to private descriptors while serving, so ordinary stdout writes cannot corrupt
the wire. Local tests still assert that the client observes no protocol parse errors.

These facts establish the selected official artifact and this fixture's local contract. They do
not establish the reliability of every SDK feature or prove any Host Adapter, deployment, or
Stage 3 acceptance result.

## Modes

Each process exposes the same Tool and schema. Restart the server with a different startup mode
to select a failure fixture:

| Mode | Deterministic behavior |
| --- | --- |
| `normal` | Returns the fixed neutral record and short payload. |
| `slow` | Waits 500 ms, then returns the normal result. |
| `error` | Returns `is_error=true` with a fixed structured error object. |
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
mirror or cache contains every wheel in the lock. The committed lock records the exact official
v2 artifacts; registry access used to resolve or install them is not a claim about runtime
network access by the fixture itself.

The tests cover:

- initialization, the one-Tool list, annotations, schemas, and the normal result;
- rejection of a Tool argument that attempts to switch modes;
- a real client-side `REQUEST_TIMEOUT` (`-32001`) against a slow server;
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
