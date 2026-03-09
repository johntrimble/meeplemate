"""API unit tests for the feedback endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

MSG_ID = "00000000-0000-0000-0000-000000000002"
CHAT_ID = "00000000-0000-0000-0000-000000000001"

PUT_URL = f"/api/messages/{MSG_ID}/feedback"
DELETE_URL = f"/api/messages/{MSG_ID}/feedback"
MESSAGES_URL = f"/api/chats/{CHAT_ID}/messages"


# ---------------------------------------------------------------------------
# PUT /api/messages/{id}/feedback
# ---------------------------------------------------------------------------

def test_put_feedback_thumbs_up(api_client: TestClient, mock_data_layer: AsyncMock):
    resp = api_client.put(PUT_URL, json={"value": 1})
    assert resp.status_code == 204
    mock_data_layer.upsert_feedback.assert_called_once_with(UUID(MSG_ID), 1)


def test_put_feedback_thumbs_down(api_client: TestClient, mock_data_layer: AsyncMock):
    resp = api_client.put(PUT_URL, json={"value": 0})
    assert resp.status_code == 204
    mock_data_layer.upsert_feedback.assert_called_once_with(UUID(MSG_ID), 0)


def test_put_feedback_wrong_owner(api_client: TestClient, mock_data_layer: AsyncMock):
    mock_data_layer.get_message_owner.return_value = "other-uid"
    resp = api_client.put(PUT_URL, json={"value": 1})
    assert resp.status_code == 404
    mock_data_layer.upsert_feedback.assert_not_called()


def test_put_feedback_message_not_found(api_client: TestClient, mock_data_layer: AsyncMock):
    mock_data_layer.get_message_owner.return_value = None
    resp = api_client.put(PUT_URL, json={"value": 1})
    assert resp.status_code == 404
    mock_data_layer.upsert_feedback.assert_not_called()


def test_put_feedback_invalid_value(api_client: TestClient):
    resp = api_client.put(PUT_URL, json={"value": 2})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /api/messages/{id}/feedback
# ---------------------------------------------------------------------------

def test_delete_feedback(api_client: TestClient, mock_data_layer: AsyncMock):
    resp = api_client.delete(DELETE_URL)
    assert resp.status_code == 204
    mock_data_layer.delete_feedback.assert_called_once_with(UUID(MSG_ID))


def test_delete_feedback_wrong_owner(api_client: TestClient, mock_data_layer: AsyncMock):
    mock_data_layer.get_message_owner.return_value = "other-uid"
    resp = api_client.delete(DELETE_URL)
    assert resp.status_code == 404
    mock_data_layer.delete_feedback.assert_not_called()


# ---------------------------------------------------------------------------
# GET /api/chats/{id}/messages — feedback field included
# ---------------------------------------------------------------------------

def test_get_messages_includes_feedback(api_client: TestClient, mock_data_layer: AsyncMock):
    mock_data_layer.get_messages.return_value = [
        {"id": MSG_ID, "role": "assistant", "parts": [], "feedback": 1},
        {"id": "00000000-0000-0000-0000-000000000003", "role": "user", "parts": [], "feedback": None},
    ]
    resp = api_client.get(MESSAGES_URL)
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["feedback"] == 1
    assert data[1]["feedback"] is None
