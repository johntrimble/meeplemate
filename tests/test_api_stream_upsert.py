"""Tests for create-on-first-message behaviour of POST /api/chats/{id}/stream.

The client mints the chat id and navigates optimistically, so the stream endpoint
creates the chat atomically (ensure_chat) on the first message and validates
ownership/game before writing anything.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

from fastapi.testclient import TestClient

CHAT_ID = "00000000-0000-0000-0000-000000000001"
STREAM_URL = f"/api/chats/{CHAT_ID}/stream"
BODY = {"message": "How do I win?", "game_id": "test-game"}


async def _empty_astream(*args, **kwargs):
    """Async generator that yields no events (the answer is empty)."""
    return
    yield  # make it a generator


def _owned_chat(user_id: str = "test-uid", game_id: str = "test-game") -> dict:
    return {
        "chat_id": CHAT_ID,
        "game_id": game_id,
        "user_id": user_id,
        "created_at": "2026-01-01T00:00:00",
    }


# ---------------------------------------------------------------------------
# Happy path: create + persist the user message
# ---------------------------------------------------------------------------


def test_stream_creates_chat_and_saves_user_message(api_client: TestClient, mock_data_layer: AsyncMock):
    api_client.app.state.deps.chatloop_service.astream = _empty_astream
    mock_data_layer.ensure_chat.return_value = _owned_chat()

    resp = api_client.post(STREAM_URL, json=BODY)

    assert resp.status_code == 200
    mock_data_layer.ensure_chat.assert_called_once_with(
        chat_id=UUID(CHAT_ID), game_id="test-game", user_id="test-uid"
    )
    # The new user message is persisted (plus the assistant message afterwards).
    user_saves = [c for c in mock_data_layer.save_message.call_args_list if c.kwargs.get("role") == "user"]
    assert len(user_saves) == 1
    assert user_saves[0].kwargs["chat_id"] == UUID(CHAT_ID)


def test_stream_forwards_client_message_id(api_client: TestClient, mock_data_layer: AsyncMock):
    """A client-supplied message_id is used verbatim so retries dedupe."""
    api_client.app.state.deps.chatloop_service.astream = _empty_astream
    mock_data_layer.ensure_chat.return_value = _owned_chat()
    client_msg_id = "00000000-0000-0000-0000-0000000000aa"

    resp = api_client.post(STREAM_URL, json={**BODY, "message_id": client_msg_id})

    assert resp.status_code == 200
    user_saves = [c for c in mock_data_layer.save_message.call_args_list if c.kwargs.get("role") == "user"]
    assert user_saves[0].kwargs["message_id"] == UUID(client_msg_id)


# ---------------------------------------------------------------------------
# Ownership / game validation (must reject BEFORE writing any message)
# ---------------------------------------------------------------------------


def test_stream_rejects_chat_owned_by_another_user(api_client: TestClient, mock_data_layer: AsyncMock):
    api_client.app.state.deps.chatloop_service.astream = _empty_astream
    mock_data_layer.ensure_chat.return_value = _owned_chat(user_id="other-uid")

    resp = api_client.post(STREAM_URL, json=BODY)

    assert resp.status_code == 404
    mock_data_layer.save_message.assert_not_called()


def test_stream_rejects_game_mismatch(api_client: TestClient, mock_data_layer: AsyncMock):
    api_client.app.state.deps.chatloop_service.astream = _empty_astream
    mock_data_layer.ensure_chat.return_value = _owned_chat(game_id="other-game")

    resp = api_client.post(STREAM_URL, json=BODY)

    assert resp.status_code == 404
    mock_data_layer.save_message.assert_not_called()


# ---------------------------------------------------------------------------
# Malformed ids -> clean 400 (not a 500)
# ---------------------------------------------------------------------------


def test_stream_malformed_chat_id_returns_400(api_client: TestClient, mock_data_layer: AsyncMock):
    resp = api_client.post("/api/chats/not-a-uuid/stream", json=BODY)
    assert resp.status_code == 400
    mock_data_layer.ensure_chat.assert_not_called()


def test_stream_coerces_non_uuid_client_message_id(api_client: TestClient, mock_data_layer: AsyncMock):
    """The AI SDK generates short non-UUID message ids; those must be accepted and
    deterministically coerced to a UUID (so retries still dedupe), not rejected."""
    from meeplemate.server.api import _coerce_message_uuid

    api_client.app.state.deps.chatloop_service.astream = _empty_astream
    mock_data_layer.ensure_chat.return_value = _owned_chat()
    client_id = "tUph1jxlmpniKcBi"  # AI SDK style id

    resp = api_client.post(STREAM_URL, json={**BODY, "message_id": client_id})

    assert resp.status_code == 200
    user_saves = [c for c in mock_data_layer.save_message.call_args_list if c.kwargs.get("role") == "user"]
    # Stored id is the deterministic derivation of the client id.
    assert user_saves[0].kwargs["message_id"] == _coerce_message_uuid(client_id)
