"""Structured logging for the API, shaped for Google Cloud Run.

Cloud Run reads logs straight from the container's stdout/stderr — there is no
logging agent and no OpenTelemetry sidecar. Cloud Logging parses each line as JSON
when it is valid JSON, and promotes a handful of *special fields* out of
``jsonPayload`` onto the LogEntry itself (``severity``, ``message``, ``time``,
``httpRequest``, ``logging.googleapis.com/trace``, ...). Lines that are not valid
JSON land in ``textPayload`` and are only substring-searchable.

So the two hard requirements are: one line per record, and the special fields named
exactly right. Everything here exists to satisfy those.

Both logging stacks are routed through stdlib logging::

    structlog loggers ─┐
                       ├→ stdlib logging → ProcessorFormatter → StreamHandler(stdout)
    stdlib loggers ────┘

Going through stdlib is what makes uvicorn, langchain, httpx and sqlalchemy records
come out in the same JSON shape as ours, instead of a second, differently-formatted
stream. (:mod:`meeplemate.eval_logging` uses the same bridging pattern for the eval
CLI; the two are deliberately kept separate, since that one renders to a file plus a
console.)

This module must not import ``meeplemate.config`` — that module pulls in the whole
component graph (``qa_graph``, ``chatloop``, ...), and logging has to be configured
*before* any of that is imported. Configuration is therefore driven from the
environment, with :func:`configure_logging` accepting explicit overrides so
``Config``-resolved values (which may come from a ``.env`` or YAML file that
``os.environ`` never sees) can be re-applied later.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import Any, Literal, Optional

import structlog

LogFormat = Literal["json", "console", "auto"]

# Loggers that are chatty at INFO and tell us nothing we want per-request.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "sqlalchemy.engine",
    "google.auth",
    "google.api_core",
)

# Set by Cloud Run on every instance; the most reliable "am I on Cloud Run" signal.
_CLOUD_RUN_ENV_VAR = "K_SERVICE"

_METADATA_PROJECT_URL = (
    "http://metadata.google.internal/computeMetadata/v1/project/project-id"
)


# ---------------------------------------------------------------------------
# GCP project id
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _resolve_gcp_project_id(explicit: Optional[str] = None) -> Optional[str]:
    """Best-effort GCP project id, for the ``logging.googleapis.com/trace`` field.

    Cloud Run does *not* set ``GOOGLE_CLOUD_PROJECT``, so the metadata server is the
    fallback that makes trace correlation work with no deploy-time configuration.
    Cached: this runs once at startup, never per request.

    Returns None if the project id can't be determined, in which case the trace
    fields are omitted and everything else still works.
    """
    if explicit:
        return explicit

    for var in ("MM_LOG__GCP_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"):
        value = os.environ.get(var)
        if value:
            return value

    # Only worth asking the metadata server when we look like we're on GCP;
    # elsewhere the hostname won't resolve and we'd just eat the timeout.
    if not os.environ.get(_CLOUD_RUN_ENV_VAR):
        return None

    try:
        import urllib.request

        request = urllib.request.Request(
            _METADATA_PROJECT_URL, headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(request, timeout=0.5) as response:
            return response.read().decode("utf-8").strip() or None
    except Exception:
        # Never let log configuration break startup.
        return None


# ---------------------------------------------------------------------------
# Cloud Logging processors
# ---------------------------------------------------------------------------

# structlog's add_log_level emits lowercase names, and uses "warn"/"exception",
# neither of which Cloud Logging recognises as a severity.
_LEVEL_TO_SEVERITY = {
    "debug": "DEBUG",
    "info": "INFO",
    "warn": "WARNING",
    "warning": "WARNING",
    "error": "ERROR",
    "exception": "ERROR",
    "critical": "CRITICAL",
    "fatal": "CRITICAL",
}


def _level_to_severity(logger, method_name, event_dict):
    """Rename ``level`` -> ``severity`` with a value Cloud Logging understands."""
    level = event_dict.pop("level", None)
    if level is not None:
        event_dict["severity"] = _LEVEL_TO_SEVERITY.get(str(level).lower(), "DEFAULT")
    return event_dict


def _event_to_message(logger, method_name, event_dict):
    """Rename ``event`` -> ``message``.

    ``message`` is the field Logs Explorer shows as the entry summary. There is no
    ``msg`` alias — it has to be spelled exactly this way.
    """
    if "event" in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def _exception_to_message(logger, method_name, event_dict):
    """Fold a rendered traceback into ``message``.

    Error Reporting only scans the ``message`` field for stack traces, so a traceback
    parked in its own key is invisible to it. ``format_exc_info`` puts the rendered
    text in ``exception``; append it to the message and drop the separate key.
    """
    exception = event_dict.pop("exception", None)
    if exception:
        message = event_dict.get("message", "")
        event_dict["message"] = f"{message}\n{exception}" if message else exception
    return event_dict


def _make_trace_processor(project_id: Optional[str]):
    """Build the processor that emits Cloud Logging's trace-correlation fields.

    The middleware binds ``trace_id``/``span_id``/``trace_sampled`` into contextvars;
    ``merge_contextvars`` has already copied them onto the event dict by the time this
    runs. Converting them into the ``logging.googleapis.com/*`` keys is what makes
    container logs nest under the Cloud Run request log in Logs Explorer.
    """

    def _add_trace_context(logger, method_name, event_dict):
        trace_id = event_dict.pop("trace_id", None)
        span_id = event_dict.pop("span_id", None)
        sampled = event_dict.pop("trace_sampled", None)

        # All three keys are promoted or none of them are. A spanId or a sampled
        # flag without a `logging.googleapis.com/trace` to anchor it populates the
        # LogEntry with a value nothing can be joined to, so when the project id is
        # unknown they stay plain, searchable fields instead.
        if trace_id and project_id:
            event_dict["logging.googleapis.com/trace"] = (
                f"projects/{project_id}/traces/{trace_id}"
            )
            if span_id:
                event_dict["logging.googleapis.com/spanId"] = span_id
            if sampled is not None:
                event_dict["logging.googleapis.com/trace_sampled"] = bool(sampled)
        else:
            if trace_id:
                event_dict["trace_id"] = trace_id
            if span_id:
                event_dict["span_id"] = span_id
            if sampled is not None:
                event_dict["trace_sampled"] = bool(sampled)

        return event_dict

    return _add_trace_context


def _drop_color_message(logger, method_name, event_dict):
    """Discard uvicorn's ``color_message`` duplicate of the log message."""
    event_dict.pop("color_message", None)
    return event_dict


def _timestamp_to_time(logger, method_name, event_dict):
    """Rename ``timestamp`` -> ``time``, Cloud Logging's special timestamp field.

    The shared chain stamps it as ``timestamp`` because that is the key
    ``ConsoleRenderer`` knows how to render nicely in dev; only the JSON chain needs
    the Cloud Logging spelling.
    """
    if "timestamp" in event_dict:
        event_dict["time"] = event_dict.pop("timestamp")
    return event_dict


# ---------------------------------------------------------------------------
# Processor chains
# ---------------------------------------------------------------------------


def _shared_processors() -> list[structlog.types.Processor]:
    """Processors applied to structlog and stdlib records alike."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _drop_color_message,
        # ISO-8601 UTC. Stamped as "timestamp" here and renamed to Cloud Logging's
        # "time" in the JSON chain — see _timestamp_to_time.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]


def _json_renderer_processors(project_id: Optional[str]) -> list[structlog.types.Processor]:
    """Final chain for Cloud Logging JSON output."""
    return [
        structlog.processors.format_exc_info,
        _event_to_message,
        _level_to_severity,
        _timestamp_to_time,
        _make_trace_processor(project_id),
        _exception_to_message,
        # Must be last, and must not emit newlines: one line == one log entry.
        structlog.processors.JSONRenderer(),
    ]


def _console_renderer_processors() -> list[structlog.types.Processor]:
    """Final chain for human-readable local development output."""
    return [
        structlog.dev.ConsoleRenderer(),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_format(fmt: LogFormat) -> Literal["json", "console"]:
    """Resolve ``auto`` to json on Cloud Run, console everywhere else."""
    if fmt != "auto":
        return fmt
    return "json" if os.environ.get(_CLOUD_RUN_ENV_VAR) else "console"


def configure_logging(
    level: Optional[str] = None,
    fmt: Optional[LogFormat] = None,
    gcp_project_id: Optional[str] = None,
) -> None:
    """Configure structlog and stdlib logging for the API process.

    Idempotent — safe to call more than once. It is called at import time in
    ``meeplemate.server.api`` (so records emitted while the package is still being
    imported are captured) and again from ``create_app`` once ``Config`` has resolved
    settings from a ``.env`` or YAML file.

    Arguments override the corresponding ``MM_LOG__*`` environment variables.
    """
    level = level or os.environ.get("MM_LOG__LEVEL") or "INFO"
    fmt = fmt or os.environ.get("MM_LOG__FORMAT") or "auto"  # type: ignore[assignment]
    resolved_format = resolve_format(fmt)  # type: ignore[arg-type]
    project_id = _resolve_gcp_project_id(gcp_project_id)

    numeric_level = logging.getLevelName(str(level).upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    shared = _shared_processors()
    renderer = (
        _json_renderer_processors(project_id)
        if resolved_format == "json"
        else _console_renderer_processors()
    )

    structlog.configure(
        processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # False so repeated configure_logging() calls actually take effect on
        # loggers that were already obtained at import time.
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *renderer,
        ],
        # Applied to records from stdlib loggers (uvicorn, langchain, ...) so they
        # come out in the same shape as structlog's own.
        foreign_pre_chain=shared,
    )

    # Cloud Run captures both streams identically, and stdout keeps ordering
    # predictable relative to anything else the process prints.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # uvicorn installs its own handlers via dictConfig when it boots, which happens
    # before it imports the app factory — so we run second and win. Strip those
    # handlers and let the records propagate to our root handler instead, otherwise
    # every uvicorn line would be emitted twice, once unformatted.
    for name in ("uvicorn", "uvicorn.error", "gunicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # uvicorn.access needs care in both directions. uvicorn decides whether to log a
    # request at all via `uvicorn.access`.hasHandlers(), and `--no-access-log` turns
    # it off by clearing that logger's handlers *and* setting propagate=False. We run
    # after that, so:
    #   - forcing propagate=True would silently re-enable the access log we asked to
    #     turn off (the production Dockerfile passes --no-access-log);
    #   - but clearing handlers while leaving the default propagate=False would make
    #     hasHandlers() false and disable access logging even when it was wanted.
    # So only re-point it at the root handler when it is currently enabled.
    access_logger = logging.getLogger("uvicorn.access")
    if access_logger.handlers:
        access_logger.propagate = True
        access_logger.handlers.clear()

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(numeric_level, logging.WARNING))


def get_logger(name: Optional[str] = None) -> Any:
    """Return a structlog logger. Thin wrapper, for a single import site."""
    return structlog.get_logger(name)
