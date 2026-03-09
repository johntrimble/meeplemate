"""API integration tests for rate limiting on the stream endpoint."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from meeplemate.db.datalayer import UserRecord, WindowStats
from meeplemate.server.rate_limit import RateLimitState, RateLimitConfig, WINDOWS, WindowState

STREAM_URL = "/api/chats/00000000-0000-0000-0000-000000000001/stream"
STREAM_BODY = {"message": "How do I win?", "game_id": "test-game"}


def _zero_state(config: RateLimitConfig) -> RateLimitState:
    return RateLimitState(windows=[
        WindowState(
            name=name,
            duration_seconds=secs,
            limit=config.user_limit(name),
            used=0,
            oldest_record_at=None,
        )
        for name, secs in WINDOWS
    ])


# ---------------------------------------------------------------------------
# 429 behaviour
# ---------------------------------------------------------------------------

def test_stream_returns_429_when_rate_limited(api_client, rate_limit_config) -> None:
    """POST to stream returns 429 with correct JSON schema when limit exceeded."""
    from meeplemate.server.rate_limit import check_rate_limit
    from meeplemate.server.api import app

    def _raise_429():
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "You have used your token quota for the 8H window.",
                "window": "8H",
                "limit": 10_000,
                "used": 9_500,
                "resets_at": "2026-03-09T18:00:00+00:00",
            },
            headers=_zero_state(rate_limit_config).headers(),
        )

    app.dependency_overrides[check_rate_limit] = _raise_429
    try:
        response = api_client.post(STREAM_URL, json=STREAM_BODY)
    finally:
        app.dependency_overrides.pop(check_rate_limit, None)

    assert response.status_code == 429
    body = response.json()
    assert body["detail"]["error"] == "rate_limit_exceeded"
    assert body["detail"]["window"] == "8H"
    assert "resets_at" in body["detail"]


def test_429_includes_rate_limit_headers(api_client, rate_limit_config) -> None:
    """429 response includes IETF ratelimit headers."""
    from meeplemate.server.rate_limit import check_rate_limit
    from meeplemate.server.api import app

    headers = _zero_state(rate_limit_config).headers()

    def _raise_429():
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "window": "8H",
                    "limit": 10_000, "used": 9_500, "message": "limit hit",
                    "resets_at": "2026-03-09T18:00:00+00:00"},
            headers=headers,
        )

    app.dependency_overrides[check_rate_limit] = _raise_429
    try:
        response = api_client.post(STREAM_URL, json=STREAM_BODY)
    finally:
        app.dependency_overrides.pop(check_rate_limit, None)

    assert response.status_code == 429
    for header in ("RateLimit-Limit", "RateLimit-Remaining", "RateLimit-Reset", "RateLimit-Policy"):
        assert header in response.headers, f"Missing header: {header}"


# ---------------------------------------------------------------------------
# Success behaviour
# ---------------------------------------------------------------------------

def test_success_includes_rate_limit_headers(api_client, rate_limit_config) -> None:
    """Successful stream response includes all four IETF ratelimit headers."""
    from meeplemate.server.rate_limit import check_rate_limit
    from meeplemate.server.api import app

    state = _zero_state(rate_limit_config)

    async def mock_astream(*args, **kwargs):
        yield None, None, {"type": "mm_user_query_answered", "answer": "42"}

    app.dependency_overrides[check_rate_limit] = lambda: state
    api_client.app.state.deps.chatloop_service.astream = mock_astream

    try:
        response = api_client.post(STREAM_URL, json=STREAM_BODY)
    finally:
        app.dependency_overrides.pop(check_rate_limit, None)

    assert response.status_code == 200
    for header in ("RateLimit-Limit", "RateLimit-Remaining", "RateLimit-Reset", "RateLimit-Policy"):
        assert header in response.headers, f"Missing header: {header}"


def test_token_usage_recorded_after_stream(api_client, mock_data_layer, rate_limit_config) -> None:
    """record_token_usage is called once after a successful stream completes."""
    from meeplemate.server.rate_limit import check_rate_limit
    from meeplemate.server.api import app

    state = _zero_state(rate_limit_config)

    async def mock_astream(*args, **kwargs):
        yield None, None, {"type": "mm_user_query_answered", "answer": "test answer"}

    app.dependency_overrides[check_rate_limit] = lambda: state
    api_client.app.state.deps.chatloop_service.astream = mock_astream

    mock_data_layer.record_token_usage.reset_mock()

    try:
        api_client.post(STREAM_URL, json=STREAM_BODY)
    finally:
        app.dependency_overrides.pop(check_rate_limit, None)

    mock_data_layer.record_token_usage.assert_called_once()
    call_args = mock_data_layer.record_token_usage.call_args
    assert call_args.args[0] == "test-uid"  # user_id
    # No real LLM called → token_callback.tokens == estimated → delta == 0
    assert call_args.args[1] == 0


def test_token_usage_not_recorded_on_429(api_client, mock_data_layer, rate_limit_config) -> None:
    """record_token_usage is NOT called when the request is rejected pre-stream."""
    from meeplemate.server.rate_limit import check_rate_limit
    from meeplemate.server.api import app

    def _raise_429():
        raise HTTPException(status_code=429, detail={"error": "rate_limit_exceeded",
                                                      "window": "8H", "limit": 1,
                                                      "used": 2, "message": "x",
                                                      "resets_at": "2026-01-01T00:00:00+00:00"})

    app.dependency_overrides[check_rate_limit] = _raise_429
    mock_data_layer.record_token_usage.reset_mock()

    try:
        api_client.post(STREAM_URL, json=STREAM_BODY)
    finally:
        app.dependency_overrides.pop(check_rate_limit, None)

    mock_data_layer.record_token_usage.assert_not_called()
