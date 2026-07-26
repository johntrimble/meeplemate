"""Priority-ordered chat model failover with a per-model circuit breaker.

Production inference providers are unreliable, and the same model is often
available from several providers that accept different sampling params and
different conventions (e.g. how "thinking" is disabled). This module lets us
configure an ordered list of chat models and, at call time, try them in strict
priority order, skipping any whose circuit breaker is open.

Design (see also the module-level docstring in ``config.py``'s builder):

* ``FailoverChatModel`` is a real ``BaseChatModel`` so it drops straight into the
  existing injection point and the raw call sites (``prompt | chat_model``).
  Its ``_generate``/``_agenerate`` (and streaming variants) iterate the child
  models under a shared circuit breaker.
* ``bind_tools`` / ``with_structured_output`` / ``bind`` cannot return a
  ``BaseChatModel`` because the per-child transformations return plain
  ``Runnable``s (``RunnableBinding`` / ``RunnableSequence``). Each transformation
  is therefore applied to every child *using the child's own native
  implementation* (preserving per-provider tool/structured formatting) and the
  results are wrapped in a ``_FailoverRunnable`` that shares the *same* breaker.

Streaming note: failover only triggers on errors raised before the first chunk.
Once a chunk has been emitted, a mid-stream failure propagates (there is no way
to un-emit tokens). Current API-reachable paths all use ``.ainvoke()``, so this
is not exercised in practice, but the streaming paths honor it for safety.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence, TypeVar

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import ConfigDict, PrivateAttr
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitBreaker:
    """Consecutive-failure circuit breaker over a fixed list of models.

    State is per-model index and lives in-process (per worker). Async races are
    benign: at worst a model is attempted a couple extra times before it opens.
    """

    def __init__(
        self,
        n: int,
        max_failures: int,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._failures = [0] * n
        self._open_until = [0.0] * n

    def is_open(self, i: int) -> bool:
        return self._clock() < self._open_until[i]

    def ordered_candidates(self) -> list[int]:
        """Closed-breaker models first (in priority order), then open ones.

        Open models are still returned, last, as a half-open last resort so the
        system degrades rather than hard-failing when everything is tripped.
        """
        now = self._clock()
        n = len(self._failures)
        closed = [i for i in range(n) if now >= self._open_until[i]]
        open_ = [i for i in range(n) if now < self._open_until[i]]
        return closed + open_

    def record_success(self, i: int) -> None:
        self._failures[i] = 0
        self._open_until[i] = 0.0

    def record_failure(self, i: int) -> bool:
        """Increment the consecutive-failure count; open the breaker at the
        threshold. Returns True if this failure opened the breaker."""
        self._failures[i] += 1
        if self._failures[i] >= self.max_failures:
            self._open_until[i] = self._clock() + self.cooldown_seconds
            self._failures[i] = 0
            return True
        return False


def _label(target: Any, index: int) -> str:
    """Best-effort human label for logging (ChatOpenAI exposes ``model_name``)."""
    for attr in ("model_name", "model"):
        name = getattr(target, attr, None)
        if isinstance(name, str):
            return name
    return f"model[{index}]"


def _transform_children(children: Sequence[Any], method: str, *args: Any, **kwargs: Any) -> list[Any]:
    """Apply a transformation (bind_tools/with_structured_output/bind) to each
    child using the child's own native implementation."""
    return [getattr(child, method)(*args, **kwargs) for child in children]


async def _afailover(
    targets: Sequence[Any],
    breaker: CircuitBreaker,
    attempt: Callable[[int, Any], Awaitable[T]],
    what: str,
) -> tuple[T, int, int]:
    """Try targets in priority order. Returns ``(result, index, attempts)`` where
    ``index`` is the winning target's position and ``attempts`` is how many
    targets were tried this call (``>1`` means failover happened)."""
    last_exc: BaseException | None = None
    attempts = 0
    for i in breaker.ordered_candidates():
        target = targets[i]
        attempts += 1
        try:
            result = await attempt(i, target)
        except Exception as exc:  # noqa: BLE001 - deliberate broad failover
            last_exc = exc
            opened = breaker.record_failure(i)
            logger.warning(
                "chat_model_failover",
                what=what,
                model=_label(target, i),
                index=i,
                error=repr(exc),
                breaker_opened=bool(opened),
            )
            continue
        breaker.record_success(i)
        logger.debug(
            "chat_model_selected",
            what=what,
            model=_label(target, i),
            index=i,
            attempts=attempts,
        )
        return result, i, attempts
    assert last_exc is not None  # ordered_candidates is never empty (>=1 model)
    raise last_exc


def _sfailover(
    targets: Sequence[Any],
    breaker: CircuitBreaker,
    attempt: Callable[[int, Any], T],
    what: str,
) -> tuple[T, int, int]:
    last_exc: BaseException | None = None
    attempts = 0
    for i in breaker.ordered_candidates():
        target = targets[i]
        attempts += 1
        try:
            result = attempt(i, target)
        except Exception as exc:  # noqa: BLE001 - deliberate broad failover
            last_exc = exc
            opened = breaker.record_failure(i)
            logger.warning(
                "chat_model_failover",
                what=what,
                model=_label(target, i),
                index=i,
                error=repr(exc),
                breaker_opened=bool(opened),
            )
            continue
        breaker.record_success(i)
        logger.debug(
            "chat_model_selected",
            what=what,
            model=_label(target, i),
            index=i,
            attempts=attempts,
        )
        return result, i, attempts
    assert last_exc is not None
    raise last_exc


def _stamp_failover(result: Any, targets: Sequence[Any], index: int, attempts: int) -> Any:
    """Record which model actually answered onto the result's ``response_metadata``
    so it lands in the trace output (authoritatively, from the config entry we
    invoked — not dependent on the provider echoing its model name).

    Applies to message-bearing results (``ChatResult`` on the generate path,
    ``AIMessage`` on the tool path). Structured-output results are plain
    objects with no metadata slot; those paths already emit real per-child
    spans, so nothing is lost.
    """
    info = {
        "model": _label(targets[index], index),
        "index": index,
        "attempts": attempts,
        "fell_back": index != 0,
    }
    if isinstance(result, ChatResult):
        for gen in result.generations:
            message = getattr(gen, "message", None)
            if message is not None:
                message.response_metadata["failover"] = info
    elif isinstance(result, BaseMessage):
        result.response_metadata["failover"] = info
    return result


class _FailoverRunnable(Runnable[LanguageModelInput, Any]):
    """Failover over a list of already-transformed child runnables.

    Produced by ``FailoverChatModel.bind_tools`` / ``with_structured_output`` /
    ``bind``. Shares the parent's circuit breaker so a provider that fails on one
    call type is skipped for all of them. Children are 1:1 with the parent's
    models, so breaker indices line up.
    """

    def __init__(self, runnables: Sequence[Runnable], breaker: CircuitBreaker, what: str):
        self.runnables = list(runnables)
        self._breaker = breaker
        self._what = what

    def invoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        result, index, attempts = _sfailover(
            self.runnables,
            self._breaker,
            lambda i, r: r.invoke(input, config=config, **kwargs),
            self._what,
        )
        return _stamp_failover(result, self.runnables, index, attempts)

    async def ainvoke(self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        result, index, attempts = await _afailover(
            self.runnables,
            self._breaker,
            lambda i, r: r.ainvoke(input, config=config, **kwargs),
            self._what,
        )
        return _stamp_failover(result, self.runnables, index, attempts)

    async def astream(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        last_exc: BaseException | None = None
        for i in self._breaker.ordered_candidates():
            runnable = self.runnables[i]
            yielded = False
            try:
                async for chunk in runnable.astream(input, config=config, **kwargs):
                    if not yielded:
                        self._breaker.record_success(i)
                        yielded = True
                    yield chunk
                if not yielded:
                    self._breaker.record_success(i)
                return
            except Exception as exc:  # noqa: BLE001
                if yielded:
                    raise
                last_exc = exc
                self._breaker.record_failure(i)
                continue
        if last_exc is not None:
            raise last_exc

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_FailoverRunnable":
        return _FailoverRunnable(
            _transform_children(self.runnables, "bind_tools", tools, **kwargs),
            self._breaker,
            self._what,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "_FailoverRunnable":
        return _FailoverRunnable(
            _transform_children(self.runnables, "with_structured_output", schema, **kwargs),
            self._breaker,
            self._what,
        )

    def bind(self, **kwargs: Any) -> "_FailoverRunnable":
        return _FailoverRunnable(
            _transform_children(self.runnables, "bind", **kwargs),
            self._breaker,
            self._what,
        )


class FailoverChatModel(BaseChatModel):
    """A ``BaseChatModel`` that fails over across an ordered list of chat models.

    Tries ``models`` in strict priority order, skipping any whose breaker is
    open, opening a model's breaker after ``max_failures`` consecutive failures
    and leaving it open for ``cooldown_seconds``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[BaseChatModel]
    max_failures: int = 3
    cooldown_seconds: float = 300.0

    _breaker: CircuitBreaker = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        if not self.models:
            raise ValueError("FailoverChatModel requires at least one model")
        self._breaker = CircuitBreaker(
            len(self.models), self.max_failures, self.cooldown_seconds
        )

    @property
    def _llm_type(self) -> str:
        return "failover"

    # --- generation --------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Delegate to the child's _generate_with_cache (not _generate directly)
        # so the child's own streaming routing runs: a child with streaming=True
        # must go through _stream, otherwise ChatOpenAI builds a stream=True
        # payload and then fails parsing it as a non-streaming response. Passing
        # the shared run_manager keeps this on the parent's LLM run (no nested
        # run, no double token counting).
        result, index, attempts = _sfailover(
            self.models,
            self._breaker,
            lambda i, m: m._generate_with_cache(messages, stop=stop, run_manager=run_manager, **kwargs),
            "generate",
        )
        return _stamp_failover(result, self.models, index, attempts)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # See _generate: delegate to _agenerate_with_cache so a streaming=True
        # child is routed through _astream rather than a broken non-streaming
        # parse of a stream payload.
        result, index, attempts = await _afailover(
            self.models,
            self._breaker,
            lambda i, m: m._agenerate_with_cache(messages, stop=stop, run_manager=run_manager, **kwargs),
            "generate",
        )
        return _stamp_failover(result, self.models, index, attempts)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        last_exc: BaseException | None = None
        for i in self._breaker.ordered_candidates():
            model = self.models[i]
            yielded = False
            try:
                for chunk in model._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    if not yielded:
                        self._breaker.record_success(i)
                        yielded = True
                    yield chunk
                if not yielded:
                    self._breaker.record_success(i)
                return
            except Exception as exc:  # noqa: BLE001
                if yielded:
                    raise
                last_exc = exc
                self._breaker.record_failure(i)
                continue
        if last_exc is not None:
            raise last_exc

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        last_exc: BaseException | None = None
        for i in self._breaker.ordered_candidates():
            model = self.models[i]
            yielded = False
            try:
                async for chunk in model._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    if not yielded:
                        self._breaker.record_success(i)
                        yielded = True
                    yield chunk
                if not yielded:
                    self._breaker.record_success(i)
                return
            except Exception as exc:  # noqa: BLE001
                if yielded:
                    raise
                last_exc = exc
                self._breaker.record_failure(i)
                continue
        if last_exc is not None:
            raise last_exc

    # --- transformations (pushed down per-child) ---------------------------

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _FailoverRunnable:
        return _FailoverRunnable(
            _transform_children(self.models, "bind_tools", tools, **kwargs),
            self._breaker,
            "bind_tools",
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _FailoverRunnable:
        return _FailoverRunnable(
            _transform_children(self.models, "with_structured_output", schema, **kwargs),
            self._breaker,
            "with_structured_output",
        )

    def bind(self, **kwargs: Any) -> _FailoverRunnable:
        return _FailoverRunnable(
            _transform_children(self.models, "bind", **kwargs),
            self._breaker,
            "bind",
        )

    # --- tracing metadata --------------------------------------------------
    # Report the primary (highest-priority) model's identity/params so persisted
    # traces and LangSmith-style callbacks show the real model name and sampling
    # params instead of a generic "failoverchatmodel". The primary is the
    # canonical model under strict-priority selection; a lower-priority model
    # that actually served a request will still be reported as the primary here,
    # which is acceptable for trace metadata.

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self.models[0]._identifying_params

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return self.models[0]._get_ls_params(stop=stop, **kwargs)
