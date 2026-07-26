"""Tests for DELETE /api/account."""
from unittest.mock import AsyncMock

import pytest
from conftest import TEST_FIREBASE_UID


@pytest.fixture
def fake_firebase_delete(monkeypatch) -> AsyncMock:
    """Stub out the Firebase Admin call, which would otherwise hit the network."""
    stub = AsyncMock()
    monkeypatch.setattr("meeplemate.server.api.delete_firebase_user", stub)
    return stub


def test_delete_account_flags_row_and_removes_firebase_user(
    api_client, mock_data_layer, fake_firebase_delete
):
    response = api_client.delete("/api/account")

    assert response.status_code == 204
    mock_data_layer.soft_delete_user.assert_awaited_once_with(TEST_FIREBASE_UID)
    fake_firebase_delete.assert_awaited_once_with(TEST_FIREBASE_UID)


def test_delete_account_flags_row_before_calling_firebase(
    api_client, mock_data_layer, fake_firebase_delete
):
    """Ordering matters: if Firebase fails, the account must already be locked
    out rather than left half-deleted with working credentials."""
    fake_firebase_delete.side_effect = RuntimeError("firebase unavailable")

    response = api_client.delete("/api/account")

    assert response.status_code == 500
    mock_data_layer.soft_delete_user.assert_awaited_once_with(TEST_FIREBASE_UID)


def test_delete_account_is_idempotent(api_client, mock_data_layer, fake_firebase_delete):
    """A retry after a partial failure must succeed, not 401 on the flag the
    first attempt set. soft_delete_user reports False when already flagged."""
    mock_data_layer.soft_delete_user.return_value = False

    response = api_client.delete("/api/account")

    assert response.status_code == 204
    fake_firebase_delete.assert_awaited_once_with(TEST_FIREBASE_UID)


def test_delete_account_requires_auth(mock_data_layer, rate_limit_config, fake_firebase_delete):
    """Without the dependency overrides the endpoint must reject an anonymous
    caller rather than deleting some default account."""
    from meeplemate.server.api import create_app
    from meeplemate.server.deps import ApiDeps, CorsConfig
    from meeplemate.server.rate_limit import RateLimiter
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock

    deps = ApiDeps(
        chatloop_service=MagicMock(),
        game_service=AsyncMock(),
        data_layer=mock_data_layer,
        rate_limiter=RateLimiter(config=rate_limit_config, data_layer=mock_data_layer),
        cors_config=CorsConfig(),
    )
    app = create_app(api_deps=deps)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.delete("/api/account")

    assert response.status_code == 401
    mock_data_layer.soft_delete_user.assert_not_awaited()
