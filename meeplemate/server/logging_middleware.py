"""Per-request logging context for the API.

Binds a request id — and, on Cloud Run, the inbound trace id — into structlog's
contextvars so that *every* log line emitted while serving a request carries them.
Without this there is no way to tell which of several concurrent chats produced a
given ``qa_graph`` line.

Also emits the one request-completion record we control. Cloud Run already logs every
request automatically, but that log cannot know our ``request_id``, the authenticated
user, or the chat being streamed — those only exist inside the application.
"""

from __future__ import annotations

import time
from typing import Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

REQUEST_ID_HEADER = b"x-request-id"
CLOUD_TRACE_HEADER = b"x-cloud-trace-context"


def parse_cloud_trace_context(value: str) -> tuple[Optional[str], Optional[str], Optional[bool]]:
    """Parse Cloud Run's ``X-Cloud-Trace-Context`` header.

    Format is ``TRACE_ID/SPAN_ID;o=TRACE_TRUE``, where the span and the ``;o=`` flag
    are both optional. Returns ``(trace_id, span_id, sampled)`` with None for any
    part that is absent or unparseable — a malformed header must never break a
    request.
    """
    if not value:
        return None, None, None

    sampled: Optional[bool] = None
    remainder = value.strip()

    if ";" in remainder:
        remainder, _, options = remainder.partition(";")
        for option in options.split(";"):
            if option.startswith("o="):
                sampled = option[2:].strip() == "1"

    trace_id, _, span_id = remainder.partition("/")
    trace_id = trace_id.strip() or None
    span_id = span_id.strip() or None

    return trace_id, span_id, sampled


def _header(scope, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _request_url(scope) -> str:
    scheme = scope.get("scheme", "http")
    path = scope.get("path", "")
    query = scope.get("query_string", b"").decode("latin-1")
    host = _header(scope, b"host") or ""
    url = f"{scheme}://{host}{path}" if host else path
    return f"{url}?{query}" if query else url


class RequestContextMiddleware:
    """Pure-ASGI middleware binding request context and logging completion.

    Deliberately *not* a ``BaseHTTPMiddleware`` subclass. That class runs the
    downstream app in a separate anyio task and pumps the response body through a
    queue, which breaks contextvar propagation into a ``StreamingResponse`` body
    generator. The SSE chat endpoint is exactly where correlated logging matters
    most — its generator runs after the response object is returned — so we call the
    downstream app in the *same* task, where contextvars bound here stay visible for
    the whole request, generator included.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Each request is its own task with its own contextvars copy, so clearing
        # here cannot disturb a concurrent request.
        structlog.contextvars.clear_contextvars()

        trace_id, span_id, sampled = parse_cloud_trace_context(
            _header(scope, CLOUD_TRACE_HEADER) or ""
        )
        # Honour a client-supplied request id so a trace can be followed across the
        # frontend boundary; otherwise mint one. Always present, which means local
        # dev and tests get correlation even with no Cloud Run trace header.
        request_id = _header(scope, REQUEST_ID_HEADER) or uuid4().hex

        context: dict[str, object] = {
            "request_id": request_id,
            "http_method": scope.get("method"),
            "http_path": scope.get("path"),
        }
        if trace_id:
            context["trace_id"] = trace_id
        if span_id:
            context["span_id"] = span_id
        if sampled is not None:
            context["trace_sampled"] = sampled
        structlog.contextvars.bind_contextvars(**context)

        status_code = 500
        response_size = 0
        started = time.perf_counter()

        async def send_wrapper(message):
            nonlocal status_code, response_size
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Echo the request id so a user-reported failure can be looked up
                # directly in Cloud Logging. Drop any existing value first: ASGI
                # headers are a list, so appending blindly would emit the header
                # twice, and intermediaries disagree on whether first or last wins
                # for a singleton header. This middleware is the authority here —
                # our id is the one that was actually logged.
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != REQUEST_ID_HEADER
                ]
                headers.append((REQUEST_ID_HEADER, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            elif message["type"] == "http.response.body":
                response_size += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "http.request_failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise

        # No unbind: uvicorn runs each request in its own task, so the contextvars
        # copy dies with it, and clear_contextvars() above covers any reuse. Unbinding
        # in a `finally` would strip request_id off the completion line below.
        duration_s = time.perf_counter() - started
        # A 5xx is our problem, a 4xx is usually the client's; neither should be
        # buried at INFO alongside successful traffic.
        if status_code >= 500:
            level = "error"
        elif status_code >= 400:
            level = "warning"
        else:
            level = "info"

        getattr(logger, level)(
            "http.request",
            duration_ms=round(duration_s * 1000, 2),
            httpRequest={
                "requestMethod": scope.get("method"),
                "requestUrl": _request_url(scope),
                "status": status_code,
                "responseSize": str(response_size),
                "userAgent": _header(scope, b"user-agent"),
                "remoteIp": (scope.get("client") or (None, None))[0],
                "protocol": f"HTTP/{scope.get('http_version', '1.1')}",
                # Cloud Logging wants a duration string, e.g. "1.234s".
                "latency": f"{duration_s:.3f}s",
            },
        )
