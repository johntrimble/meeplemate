"""Tests that the game-catalog endpoints require authentication.

`/api/games` and `/api/games/{game_id}` stay authed so anonymous traffic can't
reach the catalog DB query. The public first-paint copy is the static CDN
`games.json`, which the deploy generates by calling this endpoint with a
short-lived Firebase ID token minted from the service account — NOT by dropping
auth. These tests guard that invariant.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from meeplemate.db.datalayer import UserRecord
from meeplemate.server.auth import AuthUser, get_current_user
from meeplemate.server.deps import ApiDeps, CorsConfig
from meeplemate.server.rate_limit import RateLimitConfig, RateLimiter, get_db_user

CATAN = {"game_id": "catan", "name": "Catan", "emoji": "🏝️", "background_color": "#92400e"}


@pytest.fixture
def games_app(mock_data_layer: AsyncMock, rate_limit_config: RateLimitConfig, monkeypatch):
    """A fresh app with a stubbed game service and NO auth override by default."""
    from meeplemate.server.api import create_app

    # Ensure the auth-bypass env isn't leaking in, or the unauth cases would 200.
    monkeypatch.delenv("MM_AUTH_BYPASS", raising=False)

    mock_game_service = AsyncMock()
    mock_game_service.list_games.return_value = ([CATAN], False, "catan", "catan")
    mock_game_service.get_manifest.return_value = CATAN
    mock_game_service.get_example_questions.return_value = [None]

    mock_deps = ApiDeps(
        chatloop_service=MagicMock(),
        game_service=mock_game_service,
        data_layer=mock_data_layer,
        rate_limiter=RateLimiter(config=rate_limit_config, data_layer=mock_data_layer),
        cors_config=CorsConfig(),
    )
    return create_app(api_deps=mock_deps)


def test_get_games_requires_auth(games_app):
    with TestClient(games_app, raise_server_exceptions=False) as client:
        assert client.get("/api/games").status_code == 401  # no Authorization header


def test_get_game_requires_auth(games_app):
    with TestClient(games_app, raise_server_exceptions=False) as client:
        assert client.get("/api/games/catan").status_code == 401


def test_get_games_returns_catalog_when_authed(games_app):
    """With a valid principal (as the deploy's minted token provides) it returns the catalog."""
    games_app.dependency_overrides[get_current_user] = lambda: AuthUser(
        uid="deploy-bot", email=None, name="Deploy Bot"
    )
    games_app.dependency_overrides[get_db_user] = lambda: UserRecord(
        uid="deploy-bot", email=None, name="Deploy Bot", metadata={}
    )
    with TestClient(games_app) as client:
        resp = client.get("/api/games?first=100")
        assert resp.status_code == 200
        body = resp.json()
        assert [g["id"] for g in body["data"]] == ["catan"]
        assert body["pageInfo"]["hasNextPage"] is False
