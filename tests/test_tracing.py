"""Unit tests for agent trace persistence (meeplemate.tracing)."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.tracers import Run

from meeplemate.config import TraceConfig
from meeplemate.tracing import (
    LocalFileTraceSink,
    NoopTraceSink,
    PersistingTracer,
    build_trace_sink,
)


def _make_run_tree() -> Run:
    """A small root run with one meaningful child and one plumbing child."""
    now = datetime.now(timezone.utc)
    meaningful = Run(
        id=uuid4(), name="provide_response", run_type="chain",
        start_time=now, end_time=now, inputs={}, outputs={"response": "hi"},
    )
    plumbing = Run(
        id=uuid4(), name="RunnableSequence", run_type="chain",
        start_time=now, end_time=now, inputs={}, outputs={},
    )
    return Run(
        id=uuid4(), name="qa_graph", run_type="chain",
        start_time=now, end_time=now, inputs={"q": "x"}, outputs={},
        child_runs=[meaningful, plumbing],
    )


class _RaisingSink:
    async def write(self, key: str, data: bytes) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_noop_sink_writes_nothing(tmp_path):
    sink = NoopTraceSink()
    # No exception, no file, nothing observable.
    assert await sink.write("a/b.json.gz", b"data") is None


@pytest.mark.asyncio
async def test_local_sink_roundtrips(tmp_path):
    sink = LocalFileTraceSink(tmp_path)
    root = _make_run_tree()
    tracer = PersistingTracer(sink, chat_id="chat1", message_id="msg1")

    await tracer._persist_run(root)

    out = tmp_path / "chat1" / "msg1.json.gz"
    assert out.exists()
    restored = Run.model_validate(json.loads(gzip.decompress(out.read_bytes())))
    assert restored.name == "qa_graph"


@pytest.mark.asyncio
async def test_default_transformer_prunes_plumbing(tmp_path):
    sink = LocalFileTraceSink(tmp_path)
    tracer = PersistingTracer(sink, chat_id="c", message_id="m")

    await tracer._persist_run(_make_run_tree())

    restored = Run.model_validate(
        json.loads(gzip.decompress((tmp_path / "c" / "m.json.gz").read_bytes()))
    )
    names = [c.name for c in restored.child_runs]
    assert "provide_response" in names
    assert "RunnableSequence" not in names  # pruned by DEFAULT_SKIP_RUN_TYPES


@pytest.mark.asyncio
async def test_persist_writes_expected_key(tmp_path):
    written: dict[str, bytes] = {}

    class _CapturingSink:
        async def write(self, key: str, data: bytes) -> None:
            written[key] = data

    tracer = PersistingTracer(_CapturingSink(), chat_id="abc", message_id="def")
    await tracer._persist_run(_make_run_tree())

    assert list(written.keys()) == ["abc/def.json.gz"]


@pytest.mark.asyncio
async def test_sink_failure_is_swallowed():
    tracer = PersistingTracer(_RaisingSink(), chat_id="c", message_id="m")
    # Must not raise — trace persistence can never break the response.
    await tracer._persist_run(_make_run_tree())


def test_build_trace_sink_backends(tmp_path):
    assert isinstance(build_trace_sink(TraceConfig(backend="noop")), NoopTraceSink)
    assert isinstance(
        build_trace_sink(TraceConfig(backend="local", local_dir=str(tmp_path))),
        LocalFileTraceSink,
    )
    with pytest.raises(ValueError):
        build_trace_sink(TraceConfig(backend="gcs"))  # bucket required


@pytest.mark.asyncio
async def test_tracer_fires_via_callbacks(tmp_path):
    """Attached as a callback, the tracer persists when the root run completes.

    This is the mechanism the streaming handler relies on: the write happens
    during the invocation (before the response closes), not after.
    """
    from langchain_core.runnables import RunnableLambda

    sink = LocalFileTraceSink(tmp_path)
    tracer = PersistingTracer(sink, chat_id="C", message_id="M")
    # Name the root so the default Runnable* pruning keeps it.
    chain = RunnableLambda(lambda x: x + 1).with_config(run_name="qa_graph")

    result = await chain.ainvoke(3, config={"callbacks": [tracer]})

    assert result == 4
    out = tmp_path / "C" / "M.json.gz"
    assert out.exists()
    restored = Run.model_validate(json.loads(gzip.decompress(out.read_bytes())))
    assert restored.name == "qa_graph"


def test_eval_reexports_resolve():
    """Existing eval call sites must keep importing the moved helpers."""
    import meeplemate.eval as e

    assert e.RunEncoder is not None
    assert callable(e._run_to_dict)
    assert callable(e.skip_run_types)
    assert callable(e.transform_run_tree)
