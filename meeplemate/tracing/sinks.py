"""Trace sinks: pluggable backends for persisting serialized agent traces.

A ``TraceSink`` receives an already-serialized (gzipped JSON) trace blob and a key,
and is responsible for storing it. Three backends are provided:

- ``NoopTraceSink`` — discards everything (the off switch; the default).
- ``LocalFileTraceSink`` — writes to the local filesystem (dev/testing).
- ``GCSTraceSink`` — uploads to a Google Cloud Storage bucket (production).

The backend is selected by config via :func:`build_trace_sink`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from structlog import get_logger

if TYPE_CHECKING:
    from meeplemate.config import TraceConfig

logger = get_logger(__name__)


@runtime_checkable
class TraceSink(Protocol):
    """Stores a serialized trace blob under a key."""

    async def write(self, key: str, data: bytes) -> None:
        ...


class NoopTraceSink:
    """A trace sink that discards everything. Used when tracing is disabled."""

    async def write(self, key: str, data: bytes) -> None:  # noqa: D102
        return None


class LocalFileTraceSink:
    """Writes trace blobs to the local filesystem under ``root/<key>``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    async def write(self, key: str, data: bytes) -> None:
        path = self.root / key
        # Filesystem I/O is blocking; keep the event loop free.
        await asyncio.to_thread(self._write_sync, path, data)

    @staticmethod
    def _write_sync(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class GCSTraceSink:
    """Uploads trace blobs to a Google Cloud Storage bucket.

    Uses the synchronous ``google-cloud-storage`` client (authenticated via
    Application Default Credentials, matching the Firebase setup) wrapped in
    ``asyncio.to_thread`` so the upload can be awaited from async code without
    blocking the event loop. The upload is awaited before the request's
    streaming response closes, which is required on Cloud Run where CPU is
    throttled once the response completes.

    The bucket is expected to already exist (provisioned out-of-band with its
    retention/lifecycle rule); this sink never creates it.
    """

    def __init__(self, bucket: str, prefix: str = ""):
        # Imported lazily so the dependency is only required when GCS is selected.
        from google.cloud import storage

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self.prefix = prefix

    async def write(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._write_sync, key, data)

    def _write_sync(self, key: str, data: bytes) -> None:
        blob = self._bucket.blob(f"{self.prefix}{key}")
        blob.upload_from_string(data, content_type="application/gzip")


def build_trace_sink(cfg: "TraceConfig") -> TraceSink:
    """Construct the configured trace sink backend."""
    backend = cfg.backend
    if backend == "noop":
        return NoopTraceSink()
    if backend == "local":
        logger.info("trace_sink_local", local_dir=cfg.local_dir)
        return LocalFileTraceSink(cfg.local_dir)
    if backend == "gcs":
        if not cfg.bucket:
            raise ValueError("MM_TRACE__BUCKET must be set when MM_TRACE__BACKEND=gcs")
        logger.info("trace_sink_gcs", bucket=cfg.bucket, prefix=cfg.prefix)
        return GCSTraceSink(bucket=cfg.bucket, prefix=cfg.prefix)
    raise ValueError(f"Unknown trace backend: {backend!r}")
