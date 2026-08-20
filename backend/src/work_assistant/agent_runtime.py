from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from weakref import WeakSet

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    after_agent,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError

from .agent_definition import ModelProfile
from .capabilities import TIME_SOURCE, get_current_time, read_current_time
from .context_builder import BuiltContext
from .execution_policy import (
    AgentExecutionFailed,
    AgentPolicyKernel,
    AgentResult,
    ProductEvent,
    ResultSchemaInvalid,
    RunExecution,
    execute_tool_call,
)
from .schemas import Message
from .settings import Settings


class RuntimeConfigurationError(Exception):
    pass


class RuntimeFatalError(RuntimeError):
    """The current Runtime instance cannot safely execute another Run."""


class RuntimeCleanupTimeout(RuntimeFatalError):
    """A cancelled Runtime producer did not settle within its cleanup grace."""


class RuntimeUnavailable(RuntimeFatalError):
    """A previous fatal cleanup failure made this Runtime instance unusable."""


RuntimeItem = ProductEvent | AgentResult


class AgentRunner(Protocol):
    def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[RuntimeItem]: ...


def _timezone_from_message(message: str) -> str:
    match = re.search(r"\b[A-Za-z]+(?:[_+-][A-Za-z]+)*/[A-Za-z0-9_+\-]+\b", message)
    if match:
        return match.group(0)
    lowered = message.casefold()
    if "shanghai" in lowered or "china" in lowered or "上海" in message or "北京" in message:
        return "Asia/Shanghai"
    if "new york" in lowered or "纽约" in message:
        return "America/New_York"
    if "london" in lowered or "伦敦" in message:
        return "Europe/London"
    return "UTC"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _terminal_assistant_text(messages: Sequence[BaseMessage]) -> str:
    if not messages:
        return ""
    message = messages[-1]
    if not isinstance(message, AIMessage) or message.tool_calls:
        return ""
    return _content_text(message.content)


def _terminal_state_text(state: dict[str, Any] | None) -> str:
    if not state:
        return ""
    messages = state.get("messages", [])
    if not isinstance(messages, Sequence):
        return ""
    return _terminal_assistant_text(cast(Sequence[BaseMessage], messages))


def _tool_call(*, call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "name": name,
        "args": arguments,
        "type": "tool_call",
    }


class FakeAgentRunner:
    """Replaceable deterministic adapter that exercises the real Host policy ledger."""

    def __init__(self, *, step_delay_seconds: float = 0.02) -> None:
        self._delay = step_delay_seconds

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[RuntimeItem]:
        del thread_id, run_id, built_context
        if not messages:
            raise RuntimeError("conversation_context_missing")
        message = messages[-1].content

        if "[policy:deadline]" in message:
            execution.before_model_call()
            await asyncio.sleep(execution.agent.budget.deadline_seconds + 0.05)
            raise AssertionError("deadline guard did not cancel the fake adapter")

        if "[policy:model-step-limit]" in message:
            for index in range(execution.agent.budget.max_model_steps + 1):
                execution.before_model_call()
                execution.after_model_response([AIMessage(content=f"bounded-progress-{index}")])
            raise AssertionError("model-step guard did not stop the fake adapter")

        if "[policy:no-progress]" in message:
            for _ in range(execution.agent.budget.max_no_progress_steps + 2):
                execution.before_model_call()
                execution.after_model_response([AIMessage(content="")])
            raise AssertionError("no-progress guard did not stop the fake adapter")

        if "[policy:tool-call-limit]" in message:
            execution.before_model_call()
            calls = [
                _tool_call(
                    call_id=f"time-limit-{index}",
                    name="get_current_time",
                    arguments={"timezone": f"Etc/GMT+{index}"},
                )
                for index in range(execution.agent.budget.max_tool_calls + 1)
            ]
            execution.after_model_response([AIMessage(content="", tool_calls=calls)])
            raise AssertionError("Tool-call guard did not stop the fake adapter")

        if "[policy:tool-denied]" in message:
            execution.before_model_call()
            execution.after_model_response(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call(
                                call_id="denied-1",
                                name="unregistered_tool",
                                arguments={},
                            )
                        ],
                    )
                ]
            )
            raise AssertionError("capability guard did not stop the fake adapter")

        is_direct = "[policy:direct]" in message or not any(
            token in message.casefold()
            for token in (
                "time",
                "时",
                "london",
                "shanghai",
                "new york",
                "上海",
                "北京",
                "伦敦",
                "纽约",
            )
        )
        if is_direct:
            execution.before_model_call()
            text = "This request can be answered without an external Tool."
            execution.after_model_response([AIMessage(content=text)])
            for delta in _split_text(text):
                await asyncio.sleep(self._delay)
                yield ProductEvent("message.delta", {"delta": delta})
            yield AgentResult(text=text, source_ids=())
            return

        timezone_name = _timezone_from_message(message)
        call_id = "time-1"
        first_call = _tool_call(
            call_id=call_id,
            name="get_current_time",
            arguments={"timezone": timezone_name},
        )
        execution.before_model_call()
        execution.after_model_response([AIMessage(content="", tool_calls=[first_call])])
        emitted: list[ProductEvent] = []

        async def emit(event: ProductEvent) -> None:
            emitted.append(event)

        async def invoke(validated: dict[str, Any]) -> ToolMessage:
            await asyncio.sleep(self._delay)
            record = execution.tool_registry.require("get_current_time")
            output = await record.implementation.ainvoke(validated)
            if not isinstance(output, str):
                raise AgentExecutionFailed("tool_result_type_invalid")
            return ToolMessage(
                content=output,
                tool_call_id=call_id,
                name="get_current_time",
            )

        tool_message = await execute_tool_call(
            execution=execution,
            tool_call_id=call_id,
            tool_id="get_current_time",
            arguments={"timezone": timezone_name},
            handler=invoke,
            emit=emit,
        )
        for event in emitted:
            yield event

        if "[policy:repeat-tool]" in message:
            execution.before_model_call()
            execution.after_model_response(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call(
                                call_id="time-2",
                                name="get_current_time",
                                arguments={"timezone": timezone_name},
                            )
                        ],
                    )
                ]
            )
            raise AssertionError("repeat-call guard did not stop the fake adapter")

        try:
            result = json.loads(_content_text(tool_message.content))
        except (TypeError, json.JSONDecodeError) as exc:
            raise AgentExecutionFailed("tool_output_invalid") from exc
        if not isinstance(result, dict):
            raise AgentExecutionFailed("tool_output_invalid")
        text = (
            f"The current time in {result['timezone']} is {result['local_time']} "
            f"(UTC offset {result['utc_offset']})."
        )
        execution.before_model_call()
        execution.after_model_response([AIMessage(content=text)])
        for delta in _split_text(text):
            await asyncio.sleep(self._delay)
            yield ProductEvent("message.delta", {"delta": delta})

        if "[policy:source-missing]" in message:
            source_ids: tuple[str, ...] = ()
        elif "[policy:source-invalid]" in message:
            source_ids = ("forged-source",)
        else:
            source_ids = execution.generated_source_ids
        yield AgentResult(text=text, source_ids=source_ids)


def _split_text(text: str) -> tuple[str, str]:
    split_at = max(1, len(text) // 2)
    return text[:split_at], text[split_at:]


_FINALIZER_FORBIDDEN_MODEL_SETTINGS = frozenset(
    {
        "function_call",
        "functions",
        "parallel_tool_calls",
        "tool_choice",
        "tools",
    }
)
_FINALIZER_TOOL_SIGNAL_KEYS = frozenset({"function_call", "tool_calls"})
_FINALIZER_TOOL_BLOCK_TYPES = frozenset(
    {
        "function_call",
        "input_json_delta",
        "tool_call",
        "tool_call_chunk",
        "tool_use",
    }
)
_PUBLIC_TEXT_BLOCK_TYPES = frozenset({"output_text", "text"})
_PUBLIC_DELTA_MAX_CHARS = 8_000
_STREAM_DELTA_TARGET_CHARS = 64
_STREAM_DELTA_MAX_WAIT_SECONDS = 0.08
# Deliberately outside the public Tool ID grammar (double separator), so a
# registered/downstream capability can never collide with this Host protocol.
_FINALIZE_CONTROL_TOOL_ID = "host__finalize_answer"
_FINALIZE_CONTROL_RESULT = "finalizer_ready"
_FINALIZE_DECISION_PROTOCOL = """
<host_finalization_protocol>
`host__finalize_answer` is a Host-private completion declaration, never an answer
or an external capability. Call it exactly once with `{}` and no assistant text.
If no external Tool is needed, call it alone. You may include it in the same
Tool-call batch as terminal-eligible Tools only when that exact batch contains
every external lookup needed to answer the user's complete request. Otherwise,
omit it, run the required Tools, continue planning, and declare completion in a
later model turn. Never combine it with a non-terminal Tool.
</host_finalization_protocol>
""".strip()


@tool(_FINALIZE_CONTROL_TOOL_ID, return_direct=True)
async def _request_final_answer() -> str:
    """Declare a complete Tool decision; call exactly once with no arguments.

    Call alone when no external Tool is needed. It may share one batch only with
    terminal-eligible Tools whose complete result set will supply every external
    fact needed for the user's full request. Never include answer text.
    """

    return _FINALIZE_CONTROL_RESULT


def _has_tool_signal(message: AIMessage | AIMessageChunk) -> bool:
    if message.tool_calls or message.invalid_tool_calls:
        return True
    if isinstance(message, AIMessageChunk) and message.tool_call_chunks:
        return True
    if _FINALIZER_TOOL_SIGNAL_KEYS.intersection(message.additional_kwargs):
        return True
    if message.response_metadata.get("finish_reason") in {"function_call", "tool_calls"}:
        return True
    if isinstance(message.content, list):
        return any(
            isinstance(block, dict) and block.get("type") in _FINALIZER_TOOL_BLOCK_TYPES
            for block in message.content
        )
    return False


def _public_stream_chunk_text(chunk: AIMessageChunk) -> str:
    """Extract only public answer text from one provider stream chunk.

    DeepSeek reasoning lives outside ``content`` in ``additional_kwargs``. Tool
    arguments and all unknown provider blocks are intentionally never serialized
    into a product event.
    """

    if _has_tool_signal(chunk):
        raise AgentExecutionFailed("finalizer_tool_call_forbidden")
    if isinstance(chunk.content, str):
        return chunk.content
    if not isinstance(chunk.content, list):
        raise AgentExecutionFailed("finalizer_chunk_invalid")
    parts: list[str] = []
    for block in chunk.content:
        if not isinstance(block, dict) or block.get("type") not in _PUBLIC_TEXT_BLOCK_TYPES:
            continue
        text = block.get("text")
        if not isinstance(text, str):
            raise AgentExecutionFailed("finalizer_chunk_invalid")
        parts.append(text)
    return "".join(parts)


async def _emit_stream_chunk(context: DeepSeekRuntimeContext, text: str) -> None:
    """Frame one live buffered block to the public payload limit without post-hoc splitting."""

    for offset in range(0, len(text), _PUBLIC_DELTA_MAX_CHARS):
        await context.emit(
            ProductEvent(
                "message.delta",
                {"delta": text[offset : offset + _PUBLIC_DELTA_MAX_CHARS]},
            )
        )


async def _stream_final_response(
    request: ModelRequest[DeepSeekRuntimeContext],
    context: DeepSeekRuntimeContext,
) -> str:
    forbidden_settings = _FINALIZER_FORBIDDEN_MODEL_SETTINGS.intersection(request.model_settings)
    if forbidden_settings or request.tools or request.tool_choice not in {None, "none"}:
        raise AgentExecutionFailed("finalizer_tool_configuration_forbidden")

    messages: list[BaseMessage] = []
    if request.system_message is not None:
        messages.append(request.system_message)
    messages.extend(request.messages)

    text_parts: list[str] = []
    public_buffer = ""
    has_non_whitespace = False
    emitted_public_delta = False
    text_chars = 0
    max_answer_chars = context.execution.agent.result_contract.max_answer_chars
    loop = asyncio.get_running_loop()
    flush_at: float | None = None

    # This call is structurally Tool-free: it uses the unbound BaseChatModel
    # directly and passes neither request.tools nor a Tool choice. That provider
    # protocol invariant removes the single-call ambiguity where text can precede
    # a later Tool call. Any contrary provider signal fails the Run closed; text
    # already emitted before such a violation can only remain on that failed Run.
    async with asyncio.timeout(context.execution.remaining_seconds):
        provider_stream = request.model.astream(messages, **request.model_settings)
        provider_iterator = provider_stream.__aiter__()
        next_chunk: asyncio.Future[AIMessageChunk] = asyncio.ensure_future(anext(provider_iterator))
        try:
            while True:
                if flush_at is not None and loop.time() >= flush_at:
                    if not public_buffer or not emitted_public_delta:
                        raise AgentExecutionFailed("finalizer_flush_state_invalid")
                    await _emit_stream_chunk(context, public_buffer)
                    public_buffer = ""
                    flush_at = None
                    continue
                wait_seconds = None if flush_at is None else max(0.0, flush_at - loop.time())
                done, _ = await asyncio.wait({next_chunk}, timeout=wait_seconds)
                if not done:
                    if not public_buffer or not emitted_public_delta:
                        raise AgentExecutionFailed("finalizer_flush_state_invalid")
                    await _emit_stream_chunk(context, public_buffer)
                    public_buffer = ""
                    flush_at = None
                    continue

                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break
                next_chunk = asyncio.ensure_future(anext(provider_iterator))

                if not isinstance(chunk, AIMessageChunk):
                    raise AgentExecutionFailed("finalizer_chunk_invalid")
                delta = _public_stream_chunk_text(chunk)
                if not delta:
                    continue
                text_chars += len(delta)
                if text_chars > max_answer_chars:
                    raise ResultSchemaInvalid
                text_parts.append(delta)
                buffer_was_empty = not public_buffer
                public_buffer += delta
                if not has_non_whitespace and delta.strip():
                    has_non_whitespace = True
                if not has_non_whitespace:
                    continue
                if not emitted_public_delta:
                    # Make the first safe provider text observable before asking
                    # for another chunk. Later text uses a character-or-time
                    # threshold to bound both write amplification and UI stalls.
                    await _emit_stream_chunk(context, public_buffer)
                    public_buffer = ""
                    emitted_public_delta = True
                    flush_at = None
                    continue
                if buffer_was_empty:
                    flush_at = loop.time() + _STREAM_DELTA_MAX_WAIT_SECONDS
                while len(public_buffer) >= _STREAM_DELTA_TARGET_CHARS:
                    public_delta = public_buffer[:_STREAM_DELTA_TARGET_CHARS]
                    public_buffer = public_buffer[_STREAM_DELTA_TARGET_CHARS:]
                    await _emit_stream_chunk(context, public_delta)
                    flush_at = (
                        loop.time() + _STREAM_DELTA_MAX_WAIT_SECONDS if public_buffer else None
                    )
        finally:
            if not next_chunk.done():
                next_chunk.cancel()
            await asyncio.gather(next_chunk, return_exceptions=True)

    final_text = "".join(text_parts)
    if not final_text.strip():
        raise ResultSchemaInvalid
    if public_buffer:
        await _emit_stream_chunk(context, public_buffer)
    return final_text


class _RuntimeLifecycle:
    """Own one Runtime's cancellation, quarantine, and permanent fail-stop state."""

    def __init__(self, *, cleanup_grace_seconds: float) -> None:
        self._cleanup_grace_seconds = cleanup_grace_seconds
        self._quarantined: set[asyncio.Future[Any]] = set()
        self._observed: WeakSet[asyncio.Future[Any]] = WeakSet()
        self._fatal_error: str | None = None

    @property
    def is_healthy(self) -> bool:
        return self._fatal_error is None

    def require_healthy(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeUnavailable(self._fatal_error)

    def _mark_fatal(self) -> None:
        if self._fatal_error is None:
            self._fatal_error = "runtime_cleanup_timeout"

    def _observe(self, future: asyncio.Future[Any]) -> tuple[asyncio.Future[Any], ...]:
        """Consume one terminal result and discover LangGraph cleanup futures."""

        if future in self._observed:
            return ()
        self._observed.add(future)
        nested: tuple[asyncio.Future[Any], ...] = ()
        try:
            future.result()
        except asyncio.CancelledError as exc:
            nested = tuple(
                value
                for value in exc.args
                if isinstance(value, asyncio.Future) and value is not future
            )
        except Exception:
            # Runtime exceptions are delivered through the bounded queue.
            pass
        return nested

    def _quarantine(self, futures: Sequence[asyncio.Future[Any]]) -> None:
        for future in futures:
            if future.done():
                nested = self._observe(future)
                if nested:
                    self._quarantine(nested)
                continue
            if future in self._quarantined:
                continue
            self._quarantined.add(future)
            future.add_done_callback(self._observe_quarantined)

    def _observe_quarantined(self, future: asyncio.Future[Any]) -> None:
        self._quarantined.discard(future)
        nested = self._observe(future)
        if nested:
            self._quarantine(nested)

    async def finish_producer(
        self,
        producer: asyncio.Task[None],
        *,
        cancel: bool = True,
    ) -> None:
        """Cancel once, then observe producer and nested cleanup without recancelling."""

        if cancel and not producer.done() and producer.cancelling() == 0:
            producer.cancel()
        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + self._cleanup_grace_seconds
        pending: set[asyncio.Future[Any]] = {producer}
        while pending:
            remaining = deadline_at - loop.time()
            if remaining <= 0:
                self._mark_fatal()
                self._quarantine(tuple(pending))
                raise RuntimeCleanupTimeout("runtime_cleanup_timeout")
            try:
                done, still_pending = await asyncio.wait(pending, timeout=remaining)
            except asyncio.CancelledError:
                # A second parent cancellation must not reach the Runtime child.
                # Quarantine it and force process-level fail-stop instead.
                self._mark_fatal()
                self._quarantine(tuple(pending))
                raise RuntimeCleanupTimeout("runtime_cleanup_timeout") from None
            pending = set(still_pending)
            for future in done:
                pending.update(self._observe(future))
            if pending and loop.time() >= deadline_at:
                self._mark_fatal()
                self._quarantine(tuple(pending))
                raise RuntimeCleanupTimeout("runtime_cleanup_timeout")

    async def shutdown(self) -> None:
        """Observe quarantined cleanup for one grace without another cancellation."""

        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + self._cleanup_grace_seconds
        while self._quarantined:
            remaining = deadline_at - loop.time()
            if remaining <= 0:
                return
            await asyncio.wait(tuple(self._quarantined), timeout=remaining)


@dataclass
class DeepSeekRuntimeContext:
    execution: RunExecution = field(repr=False)
    built_context: BuiltContext = field(repr=False)
    emit: Callable[[ProductEvent], Awaitable[None]] = field(repr=False)
    final_response_text: str | None = field(default=None, repr=False)
    finalizer_request: ModelRequest[DeepSeekRuntimeContext] | None = field(default=None, repr=False)
    pending_finalize_call_id: str | None = field(default=None, repr=False)
    completed_finalize_call_id: str | None = field(default=None, repr=False)


@wrap_model_call
async def apply_model_policy(
    request: ModelRequest[DeepSeekRuntimeContext],
    handler: Callable[[ModelRequest[DeepSeekRuntimeContext]], Awaitable[ModelResponse[Any]]],
) -> ModelResponse[Any]:
    context = request.runtime.context
    context.execution.before_model_call()
    visible = set(context.execution.visible_tool_ids)
    tools = [
        item
        for item in request.tools
        if getattr(item, "name", None) in visible | {_FINALIZE_CONTROL_TOOL_ID}
    ]
    decision_prompt = (
        f"{context.built_context.system_prompt}\n\n{_FINALIZE_DECISION_PROTOCOL}"
        if visible
        else context.built_context.system_prompt
    )
    controlled_request = request.override(
        tools=tools,
        tool_choice="required" if visible else "none",
        system_message=SystemMessage(content=decision_prompt),
    )

    # With no visible external Tool there is no decision ambiguity: make the
    # single provider call itself the structurally Tool-free public stream.
    if not visible:
        if context.final_response_text is not None:
            raise AgentExecutionFailed("multiple_terminal_model_responses")
        direct_request = controlled_request.override(tools=[], tool_choice="none")
        final_text = await _stream_final_response(direct_request, context)
        final_message = AIMessage(content=final_text)
        context.execution.after_model_response([final_message])
        context.final_response_text = final_text
        return ModelResponse(result=[final_message], structured_response=None)

    context.finalizer_request = request.override(
        tools=[],
        tool_choice="none",
        system_message=SystemMessage(content=context.built_context.system_prompt),
    )
    response = await handler(controlled_request)
    terminal_message = response.result[-1] if response.result else None
    if len(response.result) != 1 or not isinstance(terminal_message, AIMessage):
        raise AgentExecutionFailed("model_response_invalid")
    if controlled_request.response_format is not None or response.structured_response is not None:
        raise AgentExecutionFailed("finalizer_structured_response_unsupported")

    control_calls = [
        call
        for call in terminal_message.tool_calls
        if call.get("name") == _FINALIZE_CONTROL_TOOL_ID
    ]
    if control_calls:
        control_call = control_calls[0]
        call_id = control_call.get("id")
        call_ids = [call.get("id") for call in terminal_message.tool_calls]
        external_calls = [
            call
            for call in terminal_message.tool_calls
            if call.get("name") != _FINALIZE_CONTROL_TOOL_ID
        ]
        if (
            len(control_calls) != 1
            or not isinstance(call_id, str)
            or not call_id
            or any(not isinstance(item, str) or not item for item in call_ids)
            or len(call_ids) != len(set(call_ids))
            or control_call.get("args") != {}
            or _content_text(terminal_message.content).strip()
            or terminal_message.invalid_tool_calls
            or context.pending_finalize_call_id is not None
            or context.completed_finalize_call_id is not None
        ):
            raise AgentExecutionFailed("finalizer_signal_invalid")
        for external_call in external_calls:
            tool_id = external_call.get("name")
            if (
                not isinstance(tool_id, str)
                or tool_id not in visible
                or not context.execution.tool_registry.require(tool_id).terminal_after_success
            ):
                raise AgentExecutionFailed("finalizer_signal_invalid")
        if external_calls:
            # The private control is deliberately absent from the public Tool
            # ledger. Reserve and validate the external subset exactly once.
            external_decision = terminal_message.model_copy(update={"tool_calls": external_calls})
            context.execution.after_model_response([external_decision])
        context.pending_finalize_call_id = call_id
        context.execution.record_finalizer_signal()
        return response

    if not terminal_message.tool_calls:
        if _has_tool_signal(terminal_message):
            raise AgentExecutionFailed("terminal_tool_signal_invalid")
        # Rewriting a complete hidden draft would add latency and create two
        # conflicting answer authorities. Visible-Tool runs must choose either
        # an external Tool or the Host-private no-argument finalizer signal.
        raise AgentExecutionFailed("finalizer_signal_required")

    context.execution.after_model_response(response.result)
    return response


def _last_ai_message(messages: Sequence[BaseMessage]) -> tuple[int, AIMessage] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, AIMessage):
            return index, message
    return None


@after_agent(can_jump_to=["model"])
async def finalize_after_agent(state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
    """Stream the one public answer after a verified terminal graph exit."""

    context = runtime.context
    if not isinstance(context, DeepSeekRuntimeContext):
        raise AgentExecutionFailed("runtime_context_invalid")
    if context.final_response_text is not None:
        return None
    if context.pending_finalize_call_id is not None or context.finalizer_request is None:
        raise AgentExecutionFailed("finalizer_exit_invalid")

    messages = cast(list[BaseMessage], state["messages"])
    last_ai = _last_ai_message(messages)
    if last_ai is None:
        raise AgentExecutionFailed("finalizer_exit_invalid")
    ai_index, decision_message = last_ai
    calls = cast(list[dict[str, Any]], decision_message.tool_calls)
    tool_messages = messages[ai_index + 1 :]
    call_id_list = [call.get("id") for call in calls]
    if (
        not calls
        or any(not isinstance(call_id, str) or not call_id for call_id in call_id_list)
        or len(call_id_list) != len(set(call_id_list))
        or len(tool_messages) != len(calls)
        or any(not isinstance(message, ToolMessage) for message in tool_messages)
    ):
        raise AgentExecutionFailed("finalizer_exit_invalid")
    call_ids = cast(set[str], set(call_id_list))
    completed_tool_messages = cast(list[ToolMessage], tool_messages)
    if {message.tool_call_id for message in completed_tool_messages} != call_ids or any(
        message.status != "success" for message in completed_tool_messages
    ):
        raise AgentExecutionFailed("finalizer_exit_invalid")

    control_calls = [call for call in calls if call.get("name") == _FINALIZE_CONTROL_TOOL_ID]
    external_calls = [call for call in calls if call.get("name") != _FINALIZE_CONTROL_TOOL_ID]
    terminal_external_batch = bool(external_calls) and all(
        isinstance((tool_id := call.get("name")), str)
        and tool_id in context.execution.visible_tool_ids
        and context.execution.tool_registry.require(tool_id).terminal_after_success
        for call in external_calls
    )
    if not control_calls:
        if (
            not terminal_external_batch
            or context.completed_finalize_call_id is not None
            or context.pending_finalize_call_id is not None
        ):
            raise AgentExecutionFailed("finalizer_exit_invalid")
        # A terminal-eligible Tool is only a routing opportunity, never proof
        # that the user's requested fact set is complete. Ask the model to make
        # an explicit Host completion declaration in a later decision round.
        return {"jump_to": "model"}

    if (
        len(control_calls) != 1
        or (external_calls and not terminal_external_batch)
        or context.completed_finalize_call_id != control_calls[0].get("id")
        or context.pending_finalize_call_id is not None
    ):
        raise AgentExecutionFailed("finalizer_exit_invalid")

    # The Host-private control exchange carries no business fact. Strip its
    # provider metadata and Tool result. Same-batch external Tool calls/results
    # remain ordinary untrusted finalizer input, preserving their role boundary.
    if external_calls:
        external_ids = cast(set[str], {call.get("id") for call in external_calls})
        sanitized_decision = AIMessage(
            content="",
            name=decision_message.name,
            tool_calls=external_calls,
        )
        external_results = [
            message for message in completed_tool_messages if message.tool_call_id in external_ids
        ]
        finalizer_messages = [*messages[:ai_index], sanitized_decision, *external_results]
    else:
        finalizer_messages = messages[:ai_index]
    context.execution.before_model_call()
    finalizer_request = context.finalizer_request.override(
        messages=cast(Any, finalizer_messages),
        tools=[],
        tool_choice="none",
        system_message=SystemMessage(content=context.built_context.system_prompt),
    )
    final_text = await _stream_final_response(finalizer_request, context)
    final_message = AIMessage(content=final_text, name=decision_message.name)
    context.execution.after_model_response([final_message])
    context.final_response_text = final_text
    return {"messages": [final_message]}


@wrap_tool_call
async def apply_tool_policy(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Any]],
) -> ToolMessage | Any:
    context = request.runtime.context
    if not isinstance(context, DeepSeekRuntimeContext):
        raise AgentExecutionFailed("runtime_context_invalid")
    tool_call_id = request.tool_call.get("id")
    tool_id = request.tool_call.get("name")
    arguments = request.tool_call.get("args")
    if (
        not isinstance(tool_call_id, str)
        or not isinstance(tool_id, str)
        or not isinstance(arguments, dict)
    ):
        raise AgentExecutionFailed("tool_call_invalid")

    if tool_id == _FINALIZE_CONTROL_TOOL_ID:
        if (
            arguments
            or context.pending_finalize_call_id != tool_call_id
            or context.completed_finalize_call_id is not None
        ):
            raise AgentExecutionFailed("finalizer_signal_invalid")
        context.execution.ensure_deadline()
        result = await handler(request)
        if (
            not isinstance(result, ToolMessage)
            or result.tool_call_id != tool_call_id
            or result.name not in {None, _FINALIZE_CONTROL_TOOL_ID}
            or result.status != "success"
            or _content_text(result.content) != _FINALIZE_CONTROL_RESULT
        ):
            raise AgentExecutionFailed("finalizer_signal_invalid")
        context.pending_finalize_call_id = None
        context.completed_finalize_call_id = tool_call_id
        return result

    async def invoke(validated: dict[str, Any]) -> ToolMessage:
        modified = {**request.tool_call, "args": validated}
        result = await handler(request.override(tool_call=modified))
        if not isinstance(result, ToolMessage):
            raise AgentExecutionFailed("tool_result_type_invalid")
        return result

    return await execute_tool_call(
        execution=context.execution,
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        arguments=arguments,
        handler=invoke,
        emit=context.emit,
    )


class DeepSeekAgentRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        checkpointer: BaseCheckpointSaver[Any],
        policy_kernel: AgentPolicyKernel,
    ) -> None:
        validate_runtime_configuration(settings)
        model = ChatDeepSeek(  # type: ignore[call-arg]  # upstream stub uses model_name
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            timeout=120,
            max_retries=0,
            streaming=True,
            stream_usage=True,
            extra_body={"thinking": {"type": "disabled"}},
        )
        self._agent = create_agent(
            model=model,
            tools=[
                *policy_kernel.tool_registry.enabled_implementations,
                _request_final_answer,
            ],
            middleware=[
                apply_model_policy,
                cast(Any, apply_tool_policy),
                finalize_after_agent,
            ],
            context_schema=DeepSeekRuntimeContext,
            checkpointer=checkpointer,
            name="work_assistant_agent_runtime",
        )
        self._semaphore = asyncio.Semaphore(settings.model_concurrency)
        self._recursion_limit = policy_kernel.framework_recursion_limit
        self._lifecycle = _RuntimeLifecycle(
            cleanup_grace_seconds=settings.repository_cleanup_grace_seconds
        )

    @property
    def is_healthy(self) -> bool:
        return self._lifecycle.is_healthy

    async def shutdown(self) -> None:
        await self._lifecycle.shutdown()

    async def stream(
        self,
        *,
        thread_id: str,
        run_id: str,
        messages: Sequence[Message],
        execution: RunExecution,
        built_context: BuiltContext,
    ) -> AsyncIterator[RuntimeItem]:
        del thread_id
        self._lifecycle.require_healthy()
        if not messages:
            raise RuntimeError("conversation_context_missing")
        # Keep a small bounded lead between the provider and durable persistence.
        # Completion signaling separately observes consumer exit, so a full queue
        # cannot strand a cancelled producer in its finally block.
        queue_capacity = execution.agent.budget.max_tool_calls * 3 + 6
        queue: asyncio.Queue[RuntimeItem | Exception | None] = asyncio.Queue(maxsize=queue_capacity)
        consumer_stopped = asyncio.Event()

        async def emit(event: ProductEvent) -> None:
            await queue.put(event)

        context = DeepSeekRuntimeContext(
            execution=execution,
            built_context=built_context,
            emit=emit,
        )

        async def produce() -> None:
            latest_state: dict[str, Any] | None = None
            try:
                async with self._semaphore:
                    async for item in self._agent.astream(
                        {
                            "messages": [
                                {"role": message.role, "content": message.content}
                                for message in built_context.conversation
                            ]
                        },
                        config={
                            # Runtime checkpoints remain isolated per product Run.
                            "configurable": {"thread_id": run_id},
                            "recursion_limit": self._recursion_limit,
                        },
                        context=context,
                        stream_mode="values",
                    ):
                        if isinstance(item, dict):
                            latest_state = item
                final_text = _terminal_state_text(latest_state)
                if not final_text or context.final_response_text != final_text:
                    raise AgentExecutionFailed("assistant_message_missing")
                await queue.put(
                    AgentResult(
                        text=final_text,
                        source_ids=execution.generated_source_ids,
                    )
                )
            except GraphRecursionError:
                await queue.put(AgentExecutionFailed("runtime_safety_limit"))
            except Exception as exc:
                await queue.put(exc)
            finally:
                if not consumer_stopped.is_set():
                    put_sentinel = asyncio.create_task(queue.put(None))
                    wait_for_consumer = asyncio.create_task(consumer_stopped.wait())
                    completion_tasks = {put_sentinel, wait_for_consumer}
                    try:
                        await asyncio.wait(
                            completion_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        for task in completion_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*completion_tasks, return_exceptions=True)

        producer = asyncio.create_task(produce(), name=f"deepseek-agent-{run_id}")
        producer_reached_sentinel = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    producer_reached_sentinel = True
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            consumer_stopped.set()
            # A normal sentinel means producer work reached its own final
            # boundary; observe it without injecting an unnecessary cancel.
            # Early consumer exit or Run cancellation still cancels exactly once.
            await self._lifecycle.finish_producer(
                producer,
                cancel=not producer_reached_sentinel,
            )


def model_profile_for_settings(settings: Settings) -> ModelProfile:
    if settings.model_mode == "fake":
        return ModelProfile(
            profile_id="default",
            version="1.0.0",
            provider="local-fake",
            model_id="deterministic-fake-v1",
        )
    return ModelProfile(
        profile_id="default",
        version="1.0.0",
        provider="deepseek",
        model_id=settings.deepseek_model,
    )


def validate_runtime_configuration(settings: Settings) -> None:
    if settings.model_mode == "deepseek" and (
        settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value()
    ):
        raise RuntimeConfigurationError("deepseek_api_key_missing")


@asynccontextmanager
async def runtime_for_settings(
    settings: Settings,
    *,
    policy_kernel: AgentPolicyKernel,
) -> AsyncIterator[AgentRunner]:
    validate_runtime_configuration(settings)
    if settings.model_mode == "fake":
        yield FakeAgentRunner(step_delay_seconds=settings.fake_step_delay_seconds)
        return
    serde = JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=None,
    )
    async with AsyncPostgresSaver.from_conn_string(
        settings.checkpoint_database_url,
        serde=serde,
    ) as checkpointer:
        # LangGraph owns these tables. Product Alembic migrations never modify them.
        await checkpointer.setup()
        runner = DeepSeekAgentRunner(
            settings=settings,
            checkpointer=checkpointer,
            policy_kernel=policy_kernel,
        )
        try:
            yield runner
        finally:
            await runner.shutdown()


__all__ = [
    "AgentResult",
    "AgentRunner",
    "DeepSeekAgentRunner",
    "FakeAgentRunner",
    "ProductEvent",
    "RuntimeConfigurationError",
    "RuntimeCleanupTimeout",
    "RuntimeFatalError",
    "RuntimeUnavailable",
    "TIME_SOURCE",
    "apply_model_policy",
    "apply_tool_policy",
    "get_current_time",
    "model_profile_for_settings",
    "read_current_time",
    "runtime_for_settings",
    "validate_runtime_configuration",
]
