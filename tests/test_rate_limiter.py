"""Unit tests for RateLimiter.check()."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from meeplemate.db.datalayer import UserRecord, WindowStats
from meeplemate.server.rate_limit import RateLimitConfig, RateLimiter, WINDOWS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_with_usage(user_tokens: int, app_tokens: int = 0) -> AsyncMock:
    """Build a mock data layer.

    ``check_and_reserve_user`` returns uniform stats across all three windows
    (same total_tokens for each). ``get_app_window_stats`` returns uniform app
    stats across all windows.
    """
    layer = AsyncMock()
    user_stats = WindowStats(
        total_tokens=user_tokens,
        oldest_recorded_at=datetime.now(UTC) - timedelta(hours=1),
    )
    layer.check_and_reserve_user.return_value = [user_stats] * 3
    layer.get_app_window_stats.return_value = WindowStats(
        total_tokens=app_tokens,
        oldest_recorded_at=datetime.now(UTC) - timedelta(hours=1) if app_tokens else None,
    )
    return layer


def _user(metadata: dict | None = None) -> UserRecord:
    return UserRecord(uid="u1", email=None, name=None, metadata=metadata or {})


@pytest.fixture
def config(rate_limit_config: RateLimitConfig) -> RateLimitConfig:
    """Shared with the API tests — see the fixture in conftest.py."""
    return rate_limit_config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_fixture_derives_expected_quotas(config: RateLimitConfig) -> None:
    """Pin the derived values the rest of these tests reason about."""
    assert (config.user_8h, config.user_7d, config.user_30d) == (10_000, 50_000, 100_000)
    assert (config.app_8h, config.app_7d, config.app_30d) == (1_000_000, 5_000_000, 10_000_000)
    assert config.estimated_tokens_per_request == 1_000


@pytest.mark.asyncio
async def test_allows_request_under_limit(config: RateLimitConfig) -> None:
    """No exception raised when all windows have headroom."""
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(0))
    state = await limiter.check(_user())
    assert state is not None
    assert len(state.windows) == 3


@pytest.mark.asyncio
async def test_allows_request_calls_check_and_reserve(config: RateLimitConfig) -> None:
    """check_and_reserve_user is called exactly once on a successful check."""
    layer = _layer_with_usage(0)
    limiter = RateLimiter(config=config, data_layer=layer)
    await limiter.check(_user())
    layer.check_and_reserve_user.assert_called_once()


@pytest.mark.asyncio
async def test_blocks_on_8h_window(config: RateLimitConfig) -> None:
    """HTTPException(429) raised when 8H user window would be exceeded."""
    # 9100 used + 1000 estimated = 10100 > 10000
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(9_100))
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(_user())
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["window"] == "8H"
    assert exc_info.value.detail["error"] == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_blocks_on_most_restrictive_window(config: RateLimitConfig) -> None:
    """The shortest violated window is reported, not a longer one."""
    now = datetime.now(UTC)

    async def _reserve(user_id, window_params, estimated, user_limits):
        # 8H window: fine (0 tokens)
        # 7D window: violated (49500 used + 1000 est > 50000)
        # 30D window: fine
        result = []
        for name, since in window_params:
            age_hours = (now - since).total_seconds() / 3600
            if age_hours <= 9:
                result.append(WindowStats(total_tokens=0, oldest_recorded_at=None))
            elif age_hours <= 7 * 24 + 1:
                result.append(WindowStats(
                    total_tokens=49_500,
                    oldest_recorded_at=now - timedelta(days=1),
                ))
            else:
                result.append(WindowStats(total_tokens=0, oldest_recorded_at=None))
        return result

    layer = AsyncMock()
    layer.check_and_reserve_user.side_effect = _reserve
    layer.get_app_window_stats.return_value = WindowStats(total_tokens=0, oldest_recorded_at=None)

    limiter = RateLimiter(config=config, data_layer=layer)
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(_user())
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["window"] == "7D"


@pytest.mark.asyncio
async def test_app_wide_limit_blocks_all_users(config: RateLimitConfig) -> None:
    """App-level limit fires even when the user is well under their personal limit."""
    # User: 0 tokens; App: 999_500 + 1000 est > 1_000_000 (app 8H limit)
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(0, app_tokens=999_500))
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(_user())
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["window"] == "8H"
    assert "service has reached" in exc_info.value.detail["message"]


@pytest.mark.asyncio
async def test_per_user_override_used(config: RateLimitConfig) -> None:
    """metadata['rate_limits']['8H'] overrides the global per-user default."""
    # Global 8H = 10_000; override = 5_000; usage = 4_500 + 1000 est = 5500 > 5000
    user = _user(metadata={"rate_limits": {"8H": 5_000}})
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(4_500))
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(user)
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["limit"] == 5_000


@pytest.mark.asyncio
async def test_per_user_partial_override(config: RateLimitConfig) -> None:
    """Only overridden windows use the override; others use global defaults."""
    # Override 8H to 20_000 (higher than default); usage 9_100 + 1000 est = 10_100
    # With default 10_000 this would be blocked; with 20_000 it passes.
    user = _user(metadata={"rate_limits": {"8H": 20_000}})
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(9_100))
    state = await limiter.check(user)  # should not raise
    # 8H window shows the override limit
    eight_h = next(w for w in state.windows if w.name == "8H")
    assert eight_h.limit == 20_000
    # 7D window still uses global default
    seven_d = next(w for w in state.windows if w.name == "7D")
    assert seven_d.limit == config.user_7d


@pytest.mark.asyncio
async def test_resets_at_calculated_correctly(config: RateLimitConfig) -> None:
    """resets_at = oldest record timestamp + window duration."""
    oldest = datetime(2026, 3, 9, 10, 0, 0, tzinfo=UTC)

    async def _reserve(user_id, window_params, estimated, user_limits):
        return [WindowStats(total_tokens=9_500, oldest_recorded_at=oldest)] * 3

    layer = AsyncMock()
    layer.check_and_reserve_user.side_effect = _reserve
    layer.get_app_window_stats.return_value = WindowStats(total_tokens=0, oldest_recorded_at=None)

    limiter = RateLimiter(config=config, data_layer=layer)
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(_user())

    expected_reset = oldest + timedelta(hours=8)
    assert exc_info.value.detail["resets_at"] == expected_reset.isoformat()


@pytest.mark.asyncio
async def test_response_headers_present_on_success(config: RateLimitConfig) -> None:
    """Successful check returns a state that produces valid IETF ratelimit headers."""
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(0))
    state = await limiter.check(_user())
    headers = state.headers()
    assert "RateLimit-Limit" in headers
    assert "RateLimit-Remaining" in headers
    assert "RateLimit-Reset" in headers
    assert "RateLimit-Policy" in headers
    # Policy should mention all three windows
    for _, secs in WINDOWS:
        assert f"w={secs}" in headers["RateLimit-Policy"]


@pytest.mark.asyncio
async def test_response_headers_present_on_429(config: RateLimitConfig) -> None:
    """429 HTTPException includes IETF ratelimit headers."""
    limiter = RateLimiter(config=config, data_layer=_layer_with_usage(9_500))
    with pytest.raises(HTTPException) as exc_info:
        await limiter.check(_user())
    assert exc_info.value.headers is not None
    assert "RateLimit-Policy" in exc_info.value.headers
