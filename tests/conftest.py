"""Shared pytest fixtures for the MeepleMate test suite."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from meeplemate.db.datalayer import UserRecord, WindowStats
from meeplemate.server.auth import AuthUser
from meeplemate.server.deps import ApiDeps, CorsConfig
from meeplemate.server.rate_limit import RateLimitConfig, RateLimiter, RateLimitState, WINDOWS, WindowState


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rate_limit_config() -> RateLimitConfig:
    return RateLimitConfig(
        user_8h=10_000,
        user_7d=50_000,
        user_30d=100_000,
        app_8h=1_000_000,
        app_7d=5_000_000,
        app_30d=10_000_000,
        estimated_tokens_per_request=1_000,
    )


@pytest.fixture
def mock_user() -> UserRecord:
    return UserRecord(uid="test-uid", email="test@example.com", name="Test User", metadata={})


@pytest.fixture
def mock_data_layer() -> AsyncMock:
    layer = AsyncMock()
    layer.upsert_user.return_value = UserRecord(
        uid="test-uid", email="test@example.com", name="Test User", metadata={}
    )
    layer.get_window_stats.return_value = WindowStats(total_tokens=0, oldest_recorded_at=None)
    layer.get_app_window_stats.return_value = WindowStats(total_tokens=0, oldest_recorded_at=None)
    layer.check_and_reserve_user.return_value = [WindowStats(total_tokens=0, oldest_recorded_at=None)] * 3
    layer.record_token_usage.return_value = None
    return layer


@pytest.fixture
def mock_rate_limiter(rate_limit_config: RateLimitConfig, mock_data_layer: AsyncMock) -> RateLimiter:
    return RateLimiter(config=rate_limit_config, data_layer=mock_data_layer)


def _zero_rate_limit_state(config: RateLimitConfig) -> RateLimitState:
    """A RateLimitState with zero usage for all windows."""
    now = datetime.now(UTC)
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
# API test client
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(mock_data_layer: AsyncMock, rate_limit_config: RateLimitConfig):
    """TestClient with a fresh app instance and auth/rate-limit dependencies overridden."""
    from meeplemate.server.api import create_app
    from meeplemate.server.auth import get_current_user
    from meeplemate.server.rate_limit import get_db_user

    rate_limiter = RateLimiter(config=rate_limit_config, data_layer=mock_data_layer)

    mock_chatloop = MagicMock()
    mock_game_service = AsyncMock()
    mock_game_service.get_manifest.return_value = {
        "game_id": "test-game",
        "name": "Test Game",
        "rulebooks": [],
    }
    mock_data_layer.get_chat.return_value = {
        "chat_id": "00000000-0000-0000-0000-000000000001",
        "game_id": "test-game",
        "user_id": "test-uid",
        "created_at": "2026-01-01T00:00:00",
    }
    mock_data_layer.get_messages.return_value = []
    mock_data_layer.save_message.return_value = None
    mock_data_layer.get_message_owner.return_value = "test-uid"
    mock_data_layer.get_message.return_value = {
        "message_id": "00000000-0000-0000-0000-000000000002",
        "chat_id": "00000000-0000-0000-0000-000000000001",
        "role": "assistant",
    }
    mock_data_layer.deactivate_messages_from.return_value = None
    mock_data_layer.upsert_feedback.return_value = None
    mock_data_layer.delete_feedback.return_value = None

    mock_deps = ApiDeps(
        chatloop_service=mock_chatloop,
        game_service=mock_game_service,
        data_layer=mock_data_layer,
        rate_limiter=rate_limiter,
        cors_config=CorsConfig(),
    )

    app = create_app(api_deps=mock_deps)
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        uid="test-uid", email="test@example.com", name="Test User"
    )
    app.dependency_overrides[get_db_user] = lambda: UserRecord(
        uid="test-uid", email="test@example.com", name="Test User", metadata={}
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
