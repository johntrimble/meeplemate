"""PersistingTracer: a LangChain tracer that persists completed run trees to a sink.

Attached per-request to the graph's callbacks. Because it subclasses
``AsyncBaseTracer``, its :meth:`_persist_run` fires exactly once — when the root
(parentless) run completes — at which point the whole child-run tree is assembled.
That completion happens *inside* the streaming request handler (as the graph
finishes), so the awaited upload lands before the HTTP response closes. This matters
on Cloud Run, where CPU is throttled to near-zero once the response completes.

Any failure while serializing or writing is swallowed and logged: persisting a
trace must never break the user-facing response.
"""

from __future__ import annotations

import gzip
import json

from langchain_core.tracers import Run
from langchain_core.tracers.base import AsyncBaseTracer
from structlog import get_logger

from meeplemate.tracing.serialization import (
    RunEncoder,
    RunTransformer,
    run_to_dict,
    skip_run_types,
    transform_run_tree,
)
from meeplemate.tracing.sinks import TraceSink

logger = get_logger(__name__)

# LangChain plumbing nodes that add noise without trace value. Pruned by default
# (their children are promoted, preserving the meaningful execution structure).
DEFAULT_SKIP_RUN_TYPES = {
    "RunnableLambda*",
    "RunnableSequence*",
    "RunnableAssign*",
    "RunnableParallel*",
    "RunnableWithFallbacks*",
}


class PersistingTracer(AsyncBaseTracer):
    """Serializes the completed run tree and writes it to a :class:`TraceSink`."""

    def __init__(
        self,
        sink: TraceSink,
        chat_id: str,
        message_id: str,
        run_transformer: RunTransformer | None = None,
    ):
        super().__init__()
        self.sink = sink
        self.chat_id = chat_id
        self.message_id = message_id
        self.run_transformer = (
            run_transformer
            if run_transformer is not None
            else skip_run_types(DEFAULT_SKIP_RUN_TYPES)
        )

    @property
    def key(self) -> str:
        return f"{self.chat_id}/{self.message_id}.json.gz"

    async def _persist_run(self, run: Run) -> None:
        try:
            if self.run_transformer:
                transformed = transform_run_tree(run, self.run_transformer)
                if transformed is None:
                    logger.warning(
                        "trace_root_filtered_out",
                        chat_id=self.chat_id,
                        message_id=self.message_id,
                    )
                    return
                run = transformed

            payload = json.dumps(run_to_dict(run), cls=RunEncoder).encode("utf-8")
            data = gzip.compress(payload)
            await self.sink.write(self.key, data)
        except Exception:
            # Trace persistence must never break the response.
            logger.error(
                "trace_persist_failed",
                chat_id=self.chat_id,
                message_id=self.message_id,
                exc_info=True,
            )
