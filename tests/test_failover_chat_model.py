"""Tests for the priority-ordered failover chat model and its circuit breaker."""

import asyncio
from typing import Any, AsyncIterator, Iterator

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, PrivateAttr

from meeplemate.failover_chat_model import (
    CircuitBreaker,
    FailoverChatModel,
    _afailover,
    _FailoverRunnable,
    _sfailover,
)


class ScriptedChatModel(BaseChatModel):
    """A fake chat model whose calls are driven by a script.

    Each entry of ``script`` is consumed per call: an ``Exception`` is raised,
    anything else is returned as the ``AIMessage`` content. Once exhausted the
    last entry repeats. ``calls`` counts how many times it was invoked.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str = "scripted"
    script: list[Any] = ["ok"]
    _calls: int = PrivateAttr(default=0)

    @property
    def calls(self) -> int:
        return self._calls

    def _next(self) -> str:
        item = self.script[min(self._calls, len(self.script) - 1)]
        self._calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def _result(self, text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._result(self._next())

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._result(self._next())

    def bind_tools(self, tools, **kwargs):
        # Enough to exercise the transformed failover path: ignores tools and
        # still runs the script via _agenerate.
        return self.bind(tools=tools, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        # Return a runnable that succeeds/fails according to the script.
        model = self

        async def _run(_input):
            return {"parsed": model._next()}

        return RunnableLambda(_run)

    @property
    def _llm_type(self) -> str:
        return "scripted"


def _msgs():
    return [HumanMessage(content="hi")]


# --- CircuitBreaker ---------------------------------------------------------


def test_breaker_opens_after_consecutive_failures():
    clock = {"t": 0.0}
    b = CircuitBreaker(2, max_failures=3, cooldown_seconds=100, clock=lambda: clock["t"])
    assert b.record_failure(0) is False
    assert b.record_failure(0) is False
    assert b.record_failure(0) is True  # third consecutive → opens
    assert b.is_open(0) is True
    assert b.ordered_candidates() == [1, 0]  # open model goes last


def test_breaker_success_resets_consecutive_count():
    clock = {"t": 0.0}
    b = CircuitBreaker(1, max_failures=3, cooldown_seconds=100, clock=lambda: clock["t"])
    b.record_failure(0)
    b.record_failure(0)
    b.record_success(0)  # reset
    assert b.record_failure(0) is False
    assert b.record_failure(0) is False
    assert b.record_failure(0) is True  # needs 3 fresh consecutive failures again
    assert b.is_open(0) is True


def test_breaker_reopens_after_cooldown():
    clock = {"t": 0.0}
    b = CircuitBreaker(1, max_failures=1, cooldown_seconds=100, clock=lambda: clock["t"])
    b.record_failure(0)
    assert b.is_open(0) is True
    clock["t"] = 101.0  # cooldown elapsed
    assert b.is_open(0) is False
    assert b.ordered_candidates() == [0]


# --- FailoverChatModel generation ------------------------------------------


@pytest.mark.asyncio
async def test_agenerate_uses_first_healthy_in_priority_order():
    m0 = ScriptedChatModel(label="m0", script=[RuntimeError("down")])
    m1 = ScriptedChatModel(label="m1", script=["from-m1"])
    fom = FailoverChatModel(models=[m0, m1])
    out = await fom.ainvoke(_msgs())
    assert out.content == "from-m1"
    assert m0.calls == 1 and m1.calls == 1


@pytest.mark.asyncio
async def test_response_metadata_records_responding_model_on_failover():
    m0 = ScriptedChatModel(label="primary", script=[RuntimeError("down")])
    m1 = ScriptedChatModel(label="secondary", script=["from-m1"])
    fom = FailoverChatModel(models=[m0, m1])
    out = await fom.ainvoke(_msgs())
    failover = out.response_metadata["failover"]
    assert failover["index"] == 1
    assert failover["attempts"] == 2
    assert failover["fell_back"] is True


@pytest.mark.asyncio
async def test_response_metadata_records_primary_when_no_failover():
    m0 = ScriptedChatModel(label="primary", script=["clean"])
    m1 = ScriptedChatModel(label="secondary", script=["unused"])
    fom = FailoverChatModel(models=[m0, m1])
    out = await fom.ainvoke(_msgs())
    failover = out.response_metadata["failover"]
    assert failover["index"] == 0
    assert failover["attempts"] == 1
    assert failover["fell_back"] is False
    assert m1.calls == 0


@pytest.mark.asyncio
async def test_all_models_fail_raises_last_error():
    m0 = ScriptedChatModel(label="m0", script=[RuntimeError("a")])
    m1 = ScriptedChatModel(label="m1", script=[ValueError("b")])
    fom = FailoverChatModel(models=[m0, m1])
    with pytest.raises(ValueError, match="b"):
        await fom.ainvoke(_msgs())


@pytest.mark.asyncio
async def test_open_breaker_skips_model_on_subsequent_calls():
    # m0 always fails; after max_failures it should be skipped entirely while
    # the healthy m1 is available.
    m0 = ScriptedChatModel(label="m0", script=[RuntimeError("down")])
    m1 = ScriptedChatModel(label="m1", script=["ok"])
    fom = FailoverChatModel(models=[m0, m1], max_failures=3, cooldown_seconds=10_000)

    for _ in range(3):
        await fom.ainvoke(_msgs())
    assert m0.calls == 3  # tried three times, then breaker opened
    assert fom._breaker.is_open(0)

    # Further calls skip m0 (it is closed-list empty of 0, only tried as last
    # resort, but m1 succeeds first).
    for _ in range(5):
        await fom.ainvoke(_msgs())
    assert m0.calls == 3  # never retried while m1 is healthy


@pytest.mark.asyncio
async def test_success_resets_before_breaker_opens():
    # Fail, fail, then succeed → breaker must never open.
    m0 = ScriptedChatModel(
        label="m0", script=[RuntimeError("x"), RuntimeError("x"), "recovered"]
    )
    m1 = ScriptedChatModel(label="m1", script=["fallback"])
    fom = FailoverChatModel(models=[m0, m1], max_failures=3, cooldown_seconds=10_000)

    assert (await fom.ainvoke(_msgs())).content == "fallback"  # m0 fail #1
    assert (await fom.ainvoke(_msgs())).content == "fallback"  # m0 fail #2
    out = await fom.ainvoke(_msgs())  # m0 recovers
    assert out.content == "recovered"
    assert not fom._breaker.is_open(0)
    assert fom._breaker._failures[0] == 0


# --- transformed paths ------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_tools_fails_over():
    m0 = ScriptedChatModel(label="m0", script=[RuntimeError("down")])
    m1 = ScriptedChatModel(label="m1", script=["tooled"])
    fom = FailoverChatModel(models=[m0, m1])
    runnable = fom.bind_tools([], tool_choice="any")
    assert isinstance(runnable, _FailoverRunnable)
    out = await runnable.ainvoke(_msgs())
    assert out.content == "tooled"


@pytest.mark.asyncio
async def test_with_structured_output_fails_over():
    m0 = ScriptedChatModel(label="m0", script=[RuntimeError("down")])
    m1 = ScriptedChatModel(label="m1", script=["structured"])
    fom = FailoverChatModel(models=[m0, m1])
    runnable = fom.with_structured_output(dict)
    assert isinstance(runnable, _FailoverRunnable)
    out = await runnable.ainvoke(_msgs())
    assert out == {"parsed": "structured"}


@pytest.mark.asyncio
async def test_failover_runnable_ainvoke_direct():
    def boom(_):
        raise RuntimeError("down")

    breaker = CircuitBreaker(2, max_failures=3, cooldown_seconds=100)
    fr = _FailoverRunnable(
        [RunnableLambda(boom), RunnableLambda(lambda x: "second")], breaker, "test"
    )
    assert await fr.ainvoke("in") == "second"


def test_transform_methods_return_failover_runnable_matching_children():
    m0 = ScriptedChatModel(label="m0")
    m1 = ScriptedChatModel(label="m1")
    fom = FailoverChatModel(models=[m0, m1])
    for r in (fom.bind_tools([]), fom.with_structured_output(dict), fom.bind(max_tokens=8)):
        assert isinstance(r, _FailoverRunnable)
        assert len(r.runnables) == 2
        assert r._breaker is fom._breaker  # shared breaker


def test_requires_at_least_one_model():
    with pytest.raises(ValueError):
        FailoverChatModel(models=[])


# --- cancellation must never be treated as a provider failure --------------
#
# asyncio.CancelledError derives from BaseException (not Exception) since
# Python 3.8, so the deliberately-broad `except Exception` failover guards do
# not catch it: a cancelled request/task propagates immediately, is never
# counted as a provider failure, and never trips the circuit breaker or falls
# over to the next model. These tests pin that behavior on each failover path.


async def _cancel_attempt(i: int, target: Any) -> Any:
    if i == 0:
        raise asyncio.CancelledError()
    return "second"  # pragma: no cover - must never be reached


def _sync_cancel_attempt(i: int, target: Any) -> Any:
    if i == 0:
        raise asyncio.CancelledError()
    return "second"  # pragma: no cover - must never be reached


@pytest.mark.asyncio
async def test_cancellederror_propagates_on_async_failover():
    breaker = CircuitBreaker(2, max_failures=3, cooldown_seconds=100)
    with pytest.raises(asyncio.CancelledError):
        await _afailover([object(), object()], breaker, _cancel_attempt, "generate")
    assert breaker._failures[0] == 0  # not recorded as a failure
    assert not breaker.is_open(0)


def test_cancellederror_propagates_on_sync_failover():
    breaker = CircuitBreaker(2, max_failures=3, cooldown_seconds=100)
    with pytest.raises(asyncio.CancelledError):
        _sfailover([object(), object()], breaker, _sync_cancel_attempt, "generate")
    assert breaker._failures[0] == 0
    assert not breaker.is_open(0)


@pytest.mark.asyncio
async def test_cancellederror_propagates_on_stream_failover():
    def boom(_):
        raise asyncio.CancelledError()

    breaker = CircuitBreaker(2, max_failures=3, cooldown_seconds=100)
    fr = _FailoverRunnable(
        [RunnableLambda(boom), RunnableLambda(lambda x: "second")], breaker, "test"
    )
    with pytest.raises(asyncio.CancelledError):
        async for _ in fr.astream("in"):
            pass
    assert breaker._failures[0] == 0
    assert not breaker.is_open(0)


class StreamingOnlyChatModel(BaseChatModel):
    """Mirrors a ``streaming=True`` ChatOpenAI: generation must be routed through
    ``_astream``. Calling ``_generate``/``_agenerate`` directly is invalid (as
    ChatOpenAI's are when a stream payload is parsed non-streaming), so they
    raise here. ``_(a)generate_with_cache`` routes to ``_astream`` via
    ``_should_stream`` and must succeed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str = "streamed-answer"
    # _should_stream() only routes to _astream when `streaming` is in
    # model_fields_set — this is the exact trigger ChatOpenAI(streaming=True)
    # relies on, so construct instances with streaming=True.
    streaming: bool = True

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise AssertionError("must not call _generate directly on a streaming model")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise AssertionError("must not call _agenerate directly on a streaming model")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        for word in self.text.split():
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        for word in self.text.split():
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))

    @property
    def _llm_type(self) -> str:
        return "streaming-only"


@pytest.mark.asyncio
async def test_ainvoke_routes_streaming_child_through_cache_not_raw_agenerate():
    # Regression: FailoverChatModel._agenerate must delegate to the child's
    # _agenerate_with_cache so a streaming-only child is served via _astream.
    fom = FailoverChatModel(models=[StreamingOnlyChatModel(text="hello world", streaming=True)])
    out = await fom.ainvoke(_msgs())
    assert out.content.strip() == "hello world"


def test_invoke_routes_streaming_child_through_cache_not_raw_generate():
    fom = FailoverChatModel(models=[StreamingOnlyChatModel(text="hi there", streaming=True)])
    out = fom.invoke(_msgs())
    assert out.content.strip() == "hi there"


def test_tracing_metadata_delegates_to_primary_model():
    # _identifying_params / _get_ls_params should report the primary child's real
    # model identity + params, not a generic "failoverchatmodel".
    from langchain_openai import ChatOpenAI

    primary = ChatOpenAI(
        model="gpt-4o-mini", api_key="x", base_url="http://x/v1",
        temperature=0.3, max_tokens=1234,
    )
    secondary = ChatOpenAI(model="other", api_key="y", base_url="http://y/v1")
    fom = FailoverChatModel(models=[primary, secondary])

    assert fom._identifying_params == primary._identifying_params
    ls = fom._get_ls_params(stop=None)
    assert ls["ls_model_name"] == "gpt-4o-mini"
    assert ls["ls_provider"] == "openai"
    assert ls["ls_temperature"] == 0.3
    assert ls["ls_max_tokens"] == 1234
