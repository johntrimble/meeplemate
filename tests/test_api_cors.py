"""Integration tests for CORS header behaviour."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from meeplemate.db.datalayer import UserRecord
from meeplemate.server.auth import AuthUser
from meeplemate.server.deps import ApiDeps, CorsConfig
from meeplemate.server.rate_limit import RateLimitConfig, RateLimiter

ALLOWED = "https://allowed.example.com"
BLOCKED = "https://blocked.example.com"


@pytest.fixture
def cors_client(mock_data_layer: AsyncMock, rate_limit_config: RateLimitConfig):
    """TestClient whose app is configured with a single known allowed origin."""
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
    mock_game_service.list_games.return_value = ([], False, None, None)
    mock_data_layer.get_chat.return_value = None
    mock_data_layer.get_messages.return_value = []

    mock_deps = ApiDeps(
        chatloop_service=mock_chatloop,
        game_service=mock_game_service,
        data_layer=mock_data_layer,
        rate_limiter=rate_limiter,
        cors_config=CorsConfig(allowed_origins=[ALLOWED]),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_allowed_origin_returns_cors_header(cors_client: TestClient):
    resp = cors_client.get("/api/games", headers={"Origin": ALLOWED})
    assert resp.headers.get("access-control-allow-origin") == ALLOWED


def test_disallowed_origin_omits_cors_header(cors_client: TestClient):
    resp = cors_client.get("/api/games", headers={"Origin": BLOCKED})
    assert "access-control-allow-origin" not in resp.headers


def test_preflight_allowed_origin(cors_client: TestClient):
    resp = cors_client.options(
        "/api/games",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == ALLOWED
    assert "access-control-allow-methods" in resp.headers


def test_preflight_disallowed_origin(cors_client: TestClient):
    resp = cors_client.options(
        "/api/games",
        headers={
            "Origin": BLOCKED,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in resp.headers


def test_cors_config_stored_on_api_deps(cors_client: TestClient):
    from meeplemate.server.api import create_app
    mock_deps = ApiDeps(
        chatloop_service=MagicMock(),
        game_service=AsyncMock(),
        data_layer=AsyncMock(),
        rate_limiter=MagicMock(),
        cors_config=CorsConfig(allowed_origins=[ALLOWED]),
    )
    app = create_app(api_deps=mock_deps)
    with TestClient(app) as client:
        deps: ApiDeps = app.state.deps
        assert ALLOWED in deps.cors_config.allowed_origins
