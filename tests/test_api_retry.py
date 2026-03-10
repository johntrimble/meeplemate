"""API unit tests for the POST /api/messages/{id}/retry endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

MSG_ID = "00000000-0000-0000-0000-000000000002"
CHAT_ID = "00000000-0000-0000-0000-000000000001"
RETRY_URL = f"/api/messages/{MSG_ID}/retry"


async def _empty_astream(*args, **kwargs):
    """Async generator that yields nothing (no events)."""
    return
    yield  # make it a generator


async def _answer_astream(*args, **kwargs):
    """Async generator that yields one mm_user_query_answered event."""
    yield (None, None, {"type": "mm_user_query_answered", "answer": "Forty-two."})


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_retry_success(api_client: TestClient, mock_data_layer: AsyncMock):
    """A valid retry deactivates the old message and streams back a new answer."""
    from meeplemate.server import api

    api.app.state  # ensure app is set up via the TestClient fixture

    # Configure chatloop to yield one answer event
    mock_chatloop = api.app.state.deps.chatloop_service if hasattr(api.app.state, "deps") else None

    # Access the mock chatloop via the client fixture's internal mock_deps
    # The api_client fixture sets up mock_deps; we need to configure the chatloop mock.
    # We configure it via the fixture's mock_data_layer as a proxy — instead, we
    # look up the chatloop from the fixture's setup. Since we can't access it directly,
    # we patch the astream on the mock provided by the fixture.
    #
    # The conftest fixture creates mock_chatloop = MagicMock() inside api_client.
    # We need to reach it. The cleanest way: configure it via the app.state.deps
    # which is set by the mock_lifespan in conftest.
    # However that's set inside the TestClient context. We can access it here:
    deps = api.app.state.deps
    deps.chatloop_service.astream = _answer_astream

    resp = api_client.post(RETRY_URL)

    assert resp.status_code == 200
    mock_data_layer.get_message_owner.assert_called_once_with(UUID(MSG_ID))
    mock_data_layer.get_message.assert_called_once_with(UUID(MSG_ID))
    mock_data_layer.deactivate_messages_from.assert_called_once_with(
        UUID(CHAT_ID), UUID(MSG_ID)
    )
    mock_data_layer.save_message.assert_called()


# ---------------------------------------------------------------------------
# Auth / ownership errors
# ---------------------------------------------------------------------------


def test_retry_wrong_owner(api_client: TestClient, mock_data_layer: AsyncMock):
    """Returns 404 when the message belongs to a different user."""
    mock_data_layer.get_message_owner.return_value = "other-uid"
    resp = api_client.post(RETRY_URL)
    assert resp.status_code == 404
    mock_data_layer.deactivate_messages_from.assert_not_called()


def test_retry_message_not_found(api_client: TestClient, mock_data_layer: AsyncMock):
    """Returns 404 when the message does not exist."""
    mock_data_layer.get_message_owner.return_value = None
    resp = api_client.post(RETRY_URL)
    assert resp.status_code == 404
    mock_data_layer.deactivate_messages_from.assert_not_called()


# ---------------------------------------------------------------------------
# Role validation
# ---------------------------------------------------------------------------


def test_retry_user_message(api_client: TestClient, mock_data_layer: AsyncMock):
    """Returns 400 when attempting to retry a user message."""
    mock_data_layer.get_message.return_value = {
        "message_id": MSG_ID,
        "chat_id": CHAT_ID,
        "role": "user",
    }
    resp = api_client.post(RETRY_URL)
    assert resp.status_code == 400
    mock_data_layer.deactivate_messages_from.assert_not_called()


# ---------------------------------------------------------------------------
# Game not found
# ---------------------------------------------------------------------------


def test_retry_game_not_found(api_client: TestClient, mock_data_layer: AsyncMock):
    """Returns 404 when the game manifest is missing."""
    from meeplemate.server import api as api_module

    deps = api_module.app.state.deps
    deps.game_service.get_manifest.return_value = None
    try:
        resp = api_client.post(RETRY_URL)
    finally:
        deps.game_service.get_manifest.return_value = {
            "game_id": "test-game",
            "name": "Test Game",
            "rulebooks": [],
        }
    assert resp.status_code == 404
    mock_data_layer.deactivate_messages_from.assert_not_called()
