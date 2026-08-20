# Agent policy kernel contract 1.0

This contract defines the public Host's smallest durable Agent policy boundary.
It has one default Agent and one Runtime adapter at a time. It deliberately does
not define Skills, MCP discovery, downstream assembly, company policy, or a
multi-Agent selector.

## Startup definition

`AgentDefinition` is a frozen, strict, versioned object. Unknown fields are
rejected. Its public facts are:

- schema, Agent, and Prompt semantic versions;
- enabled state and one referenced model Profile;
- Agent-allowed Tool IDs and explicit `base_tools`;
- model-step, Tool-call, total-deadline, identical-call, and no-progress limits;
- result schema version, answer-size bound, and source requirement.

Prompt instructions are non-empty bounded text. Persistent evidence stores the
Prompt ID, version, and SHA-256 digest, never the Prompt body. Tool and Agent IDs
use stable lowercase identifiers; versions use `major.minor.patch`.

The Host validates the complete definition, model Profile, Tool Registry, and
policy metadata while `create_app` is constructed. An unknown, disabled,
duplicate, malformed, or unavailable reference aborts before a product database
connection, orphan-Run sweep, Runtime checkpoint setup, or request handling.
Production additionally rejects the deterministic Fake model Profile.

## Context precedence

The pure `ContextBuilder` always emits these layers in this order:

| priority | layer | trust | model representation |
| ---: | --- | --- | --- |
| 100 | Host rules | trusted | System message |
| 90 | Agent Prompt | trusted | System message |
| 80 | Run limits and visible capability summary | trusted | System message |
| 20 | committed conversation | user | ordinary user/assistant messages |
| 10 | Tool data | external | Tool messages |

Conversation text and Tool output are never concatenated into a trusted layer.
Tool data can supply facts but cannot grant permission or replace Host/Agent
instructions. Principal subject, roles, display/session metadata, credentials,
and provider assertions are not model context.

## Deterministic capability intersection

For T-006, no Skill is loaded, so the only model-visible capabilities are:

```text
registered and enabled Tools
intersection AgentDefinition.allowed_tools
intersection PrincipalCapabilityPolicy.allowed_tools
intersection AgentDefinition.base_tools
```

The resulting ordered Tool IDs and versions are frozen in the Run's execution
plan. `terminal_after_success` is Runtime behavior of that versioned RegisteredTool,
not an unversioned execution-plan field: changing it requires a RegisteredTool
version bump so persisted plan evidence and its digest reveal the semantic
change. The flag defaults to false. An extension owner may enable it only for the
declared request range whose same-turn batch can be known to supply every
external fact needed for a complete answer; the owner remains responsible for
that support-range completeness contract.

The model middleware filters its Tool list to the per-Run capability decision.
Immediately before a Tool implementation is invoked, the Host validates the
reserved call, validated arguments, current registry state, original Agent/base
scope, and Principal policy again. A policy exception denies all capabilities;
a later policy denial cannot reach the Tool implementation.

A terminal-eligible Tool alone never proves that the user's requested fact set
is complete. Locked `create_agent` routing reaches `after_agent` after it runs,
then the Host jumps back to the model for an explicit completion decision. The
Host skips that additional decision round only when one model response contains
both (a) one or more terminal-eligible Tools and (b) exactly one no-argument,
no-text finalization control declaring that exact batch complete, and every Tool
handler succeeds. A batch that mixes the control with a non-terminal Tool fails
closed; a mixed Tool batch without the control returns to the model. Tool result
parsing, source registration, public Tool events, budget usage, and result
validation are never skipped. Raw Tool text is finalizer input and never becomes
answer text directly.

For a direct answer while external Tools are visible, the preferred model path
selects a Host-private, no-argument finalization signal instead of generating a
complete hidden draft. That control signal is not an external capability,
consumes no Tool-call budget, emits no Tool event, and carries no answer text.
If a provider ignores required Tool choice and returns one non-empty answer
draft with no Tool signal, the Host treats it only as a private no-Tool decision,
discards its text, and starts the same structurally Tool-free live finalizer at
graph exit. Blank, repeated, argument-bearing, text-bearing, or invalidly mixed
signals fail closed. If no external Tool is visible, the Host starts the
structurally Tool-free answer stream immediately and does not perform a separate
decision call.

This capability decision can only add denials. The exact Thread owner and Run
actor checks from the Principal/ownership contract remain a separate,
non-replaceable authorization boundary. Roles, including an admin-shaped role,
never grant cross-owner access.

A downstream assembly may install one narrow `AgentRunner` decorator at
application construction time for domain-specific result admission. The Host
still owns execution, persistence, event validation, cancellation, ownership,
and terminal commits; a decorator may only further reject or constrain Runtime
items and cannot weaken the public policy ledger.

## Bounded execution

All adapters, including the deterministic Fake, use the same per-Run
`RunExecution` ledger.

- A model step is counted before the provider handler. The attempt beyond the
  limit is rejected before provider execution.
- A first decision that fails with a provider connection error may be retried
  exactly once after a 150 ms bounded backoff, but only before any Tool attempt,
  public Runtime event, or final answer exists. The retry consumes another model
  step and remains inside the original Run deadline; later or ambiguous failures
  are never replayed.
- Tool calls in one model response reserve their attempt budget as one atomic
  batch before any Tool side effect. The batch is rejected if it exceeds the
  remaining limit.
- Total deadline begins before Run persistence. Repository admission, startup,
  Context construction, model calls, Tool calls, and retries consume the same
  monotonic deadline; a new graph node cannot reset it.
- Each repository transaction runs in one child task that owns the absolute
  timeout. Request cancellation is observed without cancelling that child a
  second time, so transaction rollback and connection return can settle.
  Product-database connect, pool checkout, statement/lock, and SQLite busy
  waits also have fixed validated timeouts.
- Repetition uses `Tool ID@version` plus canonical JSON of schema-validated
  arguments. A new call ID or different JSON key order does not evade it.
- Progress is a new assistant fact, successful Tool fact fingerprint, or new
  source ID. Repeated deltas, call IDs, facts, or sources do not reset the
  no-progress counter.

LangGraph's recursion limit remains a wider framework safety fuse. It is not
reported as a model-step, Tool-call, deadline, repeat, or no-progress budget.
Cancellation wins through the existing atomic terminal transition; later model,
Tool, source, or result writes cannot change that terminal Run.

Runtime producer cleanup follows the same bounded ownership rule. The Host
issues at most one cancellation, observes the producer and any framework cleanup
futures only until the absolute cleanup grace, and never cancels them again.
Failure to settle quarantines the Runtime, permanently refuses its reuse, marks
the service unhealthy, and leaves the active Run for fresh-process startup
recovery. Shutdown performs only the same bounded observation, so a provider or
checkpointer that swallows cancellation cannot hang service termination.

Terminal persistence receives one separate, short database-only finalization
window after the business deadline; no model or Tool work may run in it. If a
database child does not settle within its configured cleanup grace, or a
terminal fact cannot be confirmed, the service fails closed: health and product
traffic return `503`, new Runs are refused, and startup recovery closes any
active Run after process replacement. The unsafe pool is never reused and the
quarantined operation is not cancelled a second time.

## Result and source ledger

The adapter must return `AgentResult` schema `1.0.0` with bounded non-blank text
and unique source IDs. A Tool source becomes eligible only after this sequence:

1. the model call was reserved within budget;
2. the execution gate authorized the exact Tool and canonical arguments;
3. `tool.started` was emitted from that reservation;
4. a successful typed Tool result passed its registered output parser;
5. `tool.finished` was accepted and persisted;
6. the corresponding `source.added` was accepted and persisted.

Multiple successful calls may contribute the same stable source ID only when
their validated source metadata is identical. The public source event is then
deduplicated while all contributing successful call IDs remain in the private
ledger; conflicting reuse of an ID fails source validation.

An Agent result may cite only source IDs in that Run's successful persisted
ledger. A failed, denied, unfinished, forged, duplicate, or other-Run source
cannot complete the Run. When the result contract is
`required_if_tool_used`, a direct answer can complete without a source, while
any successful Tool path must cite at least one eligible source. Result or
source validation failure creates no assistant Message and no completed event.
Every result-schema rejection records validation state `failed` with the schema
version attempted, including malformed or over-limit stream output before a
final `AgentResult` is available. This keeps the terminal failure code and audit
outcome consistent.
Natural-language truthfulness remains an evaluation concern; the deterministic
validator does not claim to prove it.

## Persistent evidence

Migration `0003_agent_policy_evidence` adds two versioned JSON facts to each
product Run:

- `execution_plan`, written once with Run creation: actual Agent, Prompt digest,
  model Profile/provider/model, Context versions, capability-policy and Tool
  Registry versions, final visible Tools, effective budgets, and result contract;
- `execution_outcome`, written by the winning terminal transaction: status,
  stable stop reason/failure code, bounded usage, accepted sources, result
  validation state, and the validated result's cited-source subset.

New Runs always receive a validated plan. Idempotent replay returns the original
Run and never recalculates or replaces either evidence object. Terminal writes
store the outcome atomically with Run status and the terminal event. Legacy Runs
remain `null` rather than being falsely attributed to the current Agent. A
downgrade refuses to discard any recorded T-006 evidence.

Evidence excludes Prompt bodies, messages, Principal metadata, credentials,
Tool arguments, raw Tool output, provider responses, exception text, and model
reasoning. It is currently a server-side audit boundary, not part of the public
Thread/Run REST representation.

## Stable policy failures

The Host maps deterministic stops to these product failure codes:

- `model_step_limit`
- `tool_call_limit`
- `repeated_tool_call`
- `no_progress`
- `tool_not_allowed`
- `result_schema_invalid`
- `source_validation_failed`

`run_timeout`, `agent_execution_failed`, and `service_restarted` retain their
existing meanings. Run status remains `failed`; the failure code and execution
outcome explain why. `stream_unavailable` remains a client connection state.

## Runtime interface evidence

The locked implementation uses LangChain `1.3.15`, LangChain Core `1.5.4`, and
LangGraph `1.2.11`. T-006 relies only on the documented `create_agent`
`context_schema`, `wrap_model_call`, `ModelRequest.override`, and
`wrap_tool_call` surfaces. A scripted `BaseChatModel` regression crosses the
real `create_agent` graph and proves model-visible filtering plus execution-time
denial; no live model call is required for this deterministic boundary.

Official references checked for this increment:

- [LangChain agents and dynamic Tool filtering](https://docs.langchain.com/oss/python/langchain/agents)
- [Custom middleware hooks](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [Runtime context](https://docs.langchain.com/oss/python/langchain/runtime)
- [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [LangGraph recursion limit](https://docs.langchain.com/oss/python/langgraph/graph-api#recursion-limit)
