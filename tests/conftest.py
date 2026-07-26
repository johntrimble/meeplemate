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
    """Quotas and the pre-flight estimate are derived; these inputs yield user
    limits of 10k / 50k / 100k, app limits of 1M / 5M / 10M, and a 1,000-token
    estimate."""
    return RateLimitConfig(
        user_budget_30d_usd=0.01,               # 100_000 tokens at $0.10/M
        cost_per_m_input_usd=0.10,
        cost_per_m_output_usd=0.30,
        user_share_8h=0.10,
        user_share_7d=0.50,
        observed_input_tokens_per_request=1_000,
        observed_output_tokens_per_request=0,
        app_budget_30d_usd=1.00,                # 10_000_000 tokens at $0.10/M
        app_share_8h=0.10,
        app_share_7d=0.50,
    )


TEST_FIREBASE_UID = "test-uid"
TEST_EMAIL = "test@example.com"
# Rate limiting keys on the email, not the uid, so a user can't clear their
# budget by deleting and re-registering. Kept distinct in tests so a mix-up
# shows up as a failure rather than passing by coincidence.
TEST_QUOTA_KEY = TEST_EMAIL


def _test_user_record(**overrides) -> UserRecord:
    defaults = dict(
        uid=TEST_FIREBASE_UID,
        email=TEST_EMAIL,
        name="Test User",
        deleted_at=None,
        metadata={},
    )
    return UserRecord(**{**defaults, **overrides})


@pytest.fixture
def mock_user() -> UserRecord:
    return _test_user_record()


@pytest.fixture
def mock_data_layer() -> AsyncMock:
    layer = AsyncMock()
    layer.upsert_user.return_value = _test_user_record()
    layer.soft_delete_user.return_value = True
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
    from meeplemate.server.deps import get_db_user, get_db_user_allow_deleted

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
        "user_id": TEST_FIREBASE_UID,
        "created_at": "2026-01-01T00:00:00",
    }
    # The stream endpoint now create-on-first-message via ensure_chat (returns the
    # current row, owned by the test user for the test game) instead of get_chat.
    mock_data_layer.ensure_chat.return_value = {
        "chat_id": "00000000-0000-0000-0000-000000000001",
        "game_id": "test-game",
        "user_id": TEST_FIREBASE_UID,
        "created_at": "2026-01-01T00:00:00",
    }
    mock_data_layer.get_messages.return_value = []
    mock_data_layer.save_message.return_value = None
    mock_data_layer.get_message_owner.return_value = TEST_FIREBASE_UID
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
        uid=TEST_FIREBASE_UID, email=TEST_EMAIL, name="Test User"
    )
    app.dependency_overrides[get_db_user] = lambda: _test_user_record()
    app.dependency_overrides[get_db_user_allow_deleted] = lambda: _test_user_record()

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
