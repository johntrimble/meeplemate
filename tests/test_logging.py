"""Tests for Cloud Run structured logging.

The contract these protect is Cloud Logging's, not ours: it only promotes fields out
of ``jsonPayload`` when they are named *exactly* right, and it treats every newline as
an entry boundary. Both are easy to break by accident and neither fails loudly — the
logs just quietly stop being queryable in production.
"""

from __future__ import annotations

import io
import json
import logging

import pytest
import structlog

from meeplemate.logging_config import (
    _resolve_gcp_project_id,
    configure_logging,
    resolve_format,
)
from meeplemate.server.logging_middleware import (
    RequestContextMiddleware,
    parse_cloud_trace_context,
)


@pytest.fixture
def json_logs(monkeypatch):
    """Configure real JSON logging into a buffer and yield the parsed records."""
    monkeypatch.setenv("K_SERVICE", "api-test")
    _resolve_gcp_project_id.cache_clear()

    configure_logging(level="DEBUG", fmt="json", gcp_project_id="test-project")
    buffer = io.StringIO()
    logging.getLogger().handlers[0].stream = buffer
    structlog.contextvars.clear_contextvars()

    def read():
        return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]

    yield read

    structlog.contextvars.clear_contextvars()
    _resolve_gcp_project_id.cache_clear()


# ---------------------------------------------------------------------------
# Cloud Logging special fields
# ---------------------------------------------------------------------------


def test_special_fields_use_cloud_logging_names(json_logs):
    structlog.get_logger("meeplemate.test").info("chat.stream_started", chat_id="c1")

    (record,) = json_logs()
    # `message` and `severity` are the exact spellings Cloud Logging promotes;
    # `event`/`level` (structlog's defaults) would be left buried in jsonPayload.
    assert record["message"] == "chat.stream_started"
    assert record["severity"] == "INFO"
    assert "event" not in record
    assert "level" not in record
    assert record["time"].endswith("Z")
    assert record["chat_id"] == "c1"
    assert record["logger"] == "meeplemate.test"


@pytest.mark.parametrize(
    "method,expected",
    [("debug", "DEBUG"), ("info", "INFO"), ("warning", "WARNING"), ("error", "ERROR"), ("critical", "CRITICAL")],
)
def test_severity_is_uppercase_per_level(json_logs, method, expected):
    getattr(structlog.get_logger("t"), method)("evt")
    assert json_logs()[0]["severity"] == expected


def test_exception_severity_is_error_and_traceback_lands_in_message(json_logs):
    try:
        raise ValueError("boom")
    except ValueError:
        structlog.get_logger("t").exception("qa.failed")

    (record,) = json_logs()
    # Error Reporting only scans `message` for stack traces, so a traceback in its
    # own field would never surface there.
    assert record["severity"] == "ERROR"
    assert record["message"].startswith("qa.failed")
    assert "ValueError: boom" in record["message"]
    assert "Traceback" in record["message"]
    assert "exception" not in record


def test_trace_context_becomes_cloud_logging_fields(json_logs):
    structlog.contextvars.bind_contextvars(
        trace_id="abc123", span_id="span9", trace_sampled=True
    )
    structlog.get_logger("t").info("evt")

    (record,) = json_logs()
    assert record["logging.googleapis.com/trace"] == "projects/test-project/traces/abc123"
    assert record["logging.googleapis.com/spanId"] == "span9"
    assert record["logging.googleapis.com/trace_sampled"] is True
    # The raw keys must be consumed, not left as duplicates.
    assert "trace_id" not in record
    assert "span_id" not in record


def test_trace_fields_kept_plain_when_project_id_unknown(monkeypatch):
    """Without a project id, *no* logging.googleapis.com/* trace key may appear.

    The three keys are a unit. A spanId or trace_sampled flag promoted without a
    `logging.googleapis.com/trace` to anchor it puts a value on the LogEntry that
    nothing can be joined to.
    """
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("MM_LOG__GCP_PROJECT_ID", raising=False)
    _resolve_gcp_project_id.cache_clear()

    configure_logging(level="INFO", fmt="json")
    buffer = io.StringIO()
    logging.getLogger().handlers[0].stream = buffer

    structlog.contextvars.clear_contextvars()
    # Exactly what the middleware binds from a real X-Cloud-Trace-Context header.
    structlog.contextvars.bind_contextvars(
        trace_id="abc123", span_id="span9", trace_sampled=True
    )
    structlog.get_logger("t").info("evt")
    structlog.contextvars.clear_contextvars()
    _resolve_gcp_project_id.cache_clear()

    record = json.loads(buffer.getvalue().strip())
    assert not [k for k in record if k.startswith("logging.googleapis.com/")]
    assert record["trace_id"] == "abc123"
    assert record["span_id"] == "span9"
    assert record["trace_sampled"] is True


def test_every_record_is_exactly_one_line(json_logs):
    """A newline inside a record splits it into two entries and breaks the parse."""
    try:
        raise ValueError("multi\nline\nerror")
    except ValueError:
        structlog.get_logger("t").exception("evt", blob="a\nb\nc")

    structlog.get_logger("t").info("evt2", text="x\ny")

    # Two log calls must produce exactly two physical lines despite the newlines.
    assert len(json_logs()) == 2


def test_stdlib_records_get_the_same_shape(json_logs):
    """uvicorn/langchain use stdlib logging; they must not bypass the JSON chain."""
    logging.getLogger("uvicorn.error").warning("Application startup complete.")

    (record,) = json_logs()
    assert record["message"] == "Application startup complete."
    assert record["severity"] == "WARNING"
    assert record["logger"] == "uvicorn.error"


def test_percent_style_stdlib_args_are_interpolated(json_logs):
    logging.getLogger("meeplemate.legacy").warning("quota %s exceeds %s", "hour", "app")
    assert json_logs()[0]["message"] == "quota hour exceeds app"


def test_configure_logging_is_idempotent(json_logs):
    configure_logging(level="DEBUG", fmt="json", gcp_project_id="test-project")
    assert len(logging.getLogger().handlers) == 1


def test_uvicorn_access_log_disabled_state_is_respected():
    """`--no-access-log` must survive our reconfiguration.

    uvicorn turns the access log off by clearing that logger's handlers and setting
    propagate=False, then decides per connection via hasHandlers(). Forcing
    propagate=True here would silently re-enable it in production.
    """
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False  # what --no-access-log leaves behind

    configure_logging(level="INFO", fmt="json")

    assert access.propagate is False
    assert not access.hasHandlers()


def test_uvicorn_access_log_enabled_state_is_routed_to_root():
    """When access logging is on, its records must still render as JSON."""
    access = logging.getLogger("uvicorn.access")
    access.handlers = [logging.NullHandler()]  # what uvicorn's dictConfig leaves
    access.propagate = False

    configure_logging(level="INFO", fmt="json")

    # Re-pointed at the root handler rather than dropped, so hasHandlers() stays
    # true and uvicorn keeps emitting the line.
    assert access.propagate is True
    assert access.handlers == []
    assert access.hasHandlers()


def test_resolve_format_auto_follows_cloud_run(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "api")
    assert resolve_format("auto") == "json"
    monkeypatch.delenv("K_SERVICE")
    assert resolve_format("auto") == "console"
    # An explicit choice always wins.
    monkeypatch.setenv("K_SERVICE", "api")
    assert resolve_format("console") == "console"


# ---------------------------------------------------------------------------
# X-Cloud-Trace-Context parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        ("abc123/456;o=1", ("abc123", "456", True)),
        ("abc123/456;o=0", ("abc123", "456", False)),
        ("abc123/456", ("abc123", "456", None)),
        ("abc123", ("abc123", None, None)),
        ("", (None, None, None)),
        ("/456", (None, "456", None)),
        (";o=1", (None, None, True)),
    ],
)
def test_parse_cloud_trace_context(header, expected):
    assert parse_cloud_trace_context(header) == expected


def test_parse_cloud_trace_context_never_raises():
    """A malformed header must degrade to None, never break the request."""
    for junk in ("///", ";;;", "o=1", "a/b/c;o=x", "   "):
        assert isinstance(parse_cloud_trace_context(junk), tuple)


# ---------------------------------------------------------------------------
# Request context middleware, end to end through the real app
# ---------------------------------------------------------------------------


# Note on capture_logs below: it clears the configured processor chain for the
# duration of the block, which drops merge_contextvars and with it every
# request-scoped field we care about here. Hence the explicit `processors=` arg.

# The shared api_client fixture stubs the chat data layer but not list_games,
# so use a chat endpoint for the 200 path.
CHAT_URL = "/api/chats/00000000-0000-0000-0000-000000000001/messages"


def test_response_carries_generated_request_id(api_client):
    response = api_client.get(CHAT_URL)
    assert response.status_code == 200
    assert response.headers["x-request-id"]


def test_supplied_request_id_is_echoed_back(api_client):
    response = api_client.get(CHAT_URL, headers={"X-Request-Id": "client-supplied-id"})
    assert response.headers["x-request-id"] == "client-supplied-id"


async def _drive_middleware(downstream_headers: list[tuple[bytes, bytes]],
                            request_headers: list[tuple[bytes, bytes]]):
    """Run RequestContextMiddleware over a stub app and return the sent headers."""
    async def downstream_app(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": list(downstream_headers),
        })
        await send({"type": "http.response.body", "body": b"{}"})

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/x",
        "scheme": "http",
        "http_version": "1.1",
        "query_string": b"",
        "headers": request_headers,
        "client": ("1.2.3.4", 1234),
    }
    await RequestContextMiddleware(downstream_app)(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["headers"]


@pytest.mark.asyncio
async def test_existing_request_id_response_header_is_replaced_not_duplicated():
    """A downstream x-request-id must be replaced, never appended alongside ours.

    ASGI headers are a list, so appending blindly emits the header twice, and
    intermediaries disagree on whether first or last wins for a singleton header.
    """
    headers = await _drive_middleware(
        # A handler further down the stack setting its own id.
        downstream_headers=[(b"content-type", b"application/json"),
                            (b"x-request-id", b"downstream-value")],
        request_headers=[(b"x-request-id", b"ours")],
    )

    ids = [v for k, v in headers if k.lower() == b"x-request-id"]
    assert ids == [b"ours"], f"expected exactly our id, got {ids}"
    # Unrelated headers must survive the rebuild.
    assert (b"content-type", b"application/json") in headers


@pytest.mark.asyncio
async def test_other_response_headers_are_untouched():
    headers = await _drive_middleware(
        downstream_headers=[(b"x-vercel-ai-ui-message-stream", b"v1"),
                            (b"ratelimit-limit", b"100")],
        request_headers=[],
    )
    assert (b"x-vercel-ai-ui-message-stream", b"v1") in headers
    assert (b"ratelimit-limit", b"100") in headers
    assert len([1 for k, _ in headers if k.lower() == b"x-request-id"]) == 1


def test_request_context_is_bound_for_the_whole_request(api_client):
    """Every log line during a request must be attributable to it."""
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as logs:
        api_client.get(
            CHAT_URL,
            headers={
                "X-Request-Id": "req-42",
                "X-Cloud-Trace-Context": "trace-abc/span-1;o=1",
            },
        )

    completion = [e for e in logs if e["event"] == "http.request"]
    assert len(completion) == 1
    entry = completion[0]
    assert entry["request_id"] == "req-42"
    assert entry["trace_id"] == "trace-abc"
    assert entry["span_id"] == "span-1"
    assert entry["http_method"] == "GET"
    assert entry["http_path"] == CHAT_URL
    assert entry["httpRequest"]["status"] == 200
    assert entry["httpRequest"]["latency"].endswith("s")
    assert entry["log_level"] == "info"


def test_client_errors_log_at_warning(api_client):
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as logs:
        api_client.get("/api/chats/not-a-uuid/messages")

    (entry,) = [e for e in logs if e["event"] == "http.request"]
    assert entry["httpRequest"]["status"] == 400
    assert entry["log_level"] == "warning"


def test_context_does_not_leak_between_requests(api_client):
    """Requests are served on separate tasks; ids must not bleed across them."""
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as logs:
        api_client.get(CHAT_URL, headers={"X-Request-Id": "first"})
        api_client.get(CHAT_URL, headers={"X-Request-Id": "second"})

    ids = [e["request_id"] for e in logs if e["event"] == "http.request"]
    assert ids == ["first", "second"]


# ---------------------------------------------------------------------------
# The SSE generator
# ---------------------------------------------------------------------------
#
# This is the case the middleware exists in its current form for. The generator
# body runs *after* stream_chat returns its response object, so if request context
# did not survive into it, none of the agent's logging would be attributable — and
# an exception in there could not be surfaced as an HTTP error either.

STREAM_URL = "/api/chats/00000000-0000-0000-0000-000000000001/stream"
STREAM_BODY = {"message": "How do I win?", "game_id": "test-game"}


def test_request_context_survives_into_the_sse_generator(api_client):
    """A log emitted from inside the streaming generator keeps the request id."""

    async def _logging_astream(*args, **kwargs):
        structlog.get_logger("meeplemate.fake_agent").info("agent.working")
        return
        yield  # make it a generator

    api_client.app.state.deps.chatloop_service.astream = _logging_astream

    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as logs:
        response = api_client.post(
            STREAM_URL, json=STREAM_BODY, headers={"X-Request-Id": "stream-req"}
        )
    assert response.status_code == 200

    agent_logs = [e for e in logs if e["event"] == "agent.working"]
    assert len(agent_logs) == 1
    assert agent_logs[0]["request_id"] == "stream-req"
    # Bound by stream_chat, so the agent's own logging is attributable to the chat.
    assert agent_logs[0]["chat_id"] == "00000000-0000-0000-0000-000000000001"
    assert agent_logs[0]["game_id"] == "test-game"

    # And the stream's own lifecycle events carry it too.
    assert {"chat.stream_started", "chat.stream_finished"} <= {e["event"] for e in logs}


def test_generator_failure_is_logged(api_client):
    """An exception in the generator cannot become an HTTP error; log it or lose it."""

    async def _failing_astream(*args, **kwargs):
        raise RuntimeError("agent exploded")
        yield  # make it a generator

    api_client.app.state.deps.chatloop_service.astream = _failing_astream

    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as logs:
        # The response has already been handed back by the time the generator runs,
        # so the failure cannot change the status code — the client just sees a
        # truncated stream. That is precisely why the log line is the only signal.
        api_client.post(
            STREAM_URL, json=STREAM_BODY, headers={"X-Request-Id": "doomed-req"}
        )

    (failure,) = [e for e in logs if e["event"] == "chat.stream_failed"]
    assert failure["request_id"] == "doomed-req"
    assert failure["log_level"] == "error"
    assert "chat.stream_finished" not in {e["event"] for e in logs}
