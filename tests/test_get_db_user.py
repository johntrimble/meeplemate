"""Tests for the current-user dependencies in meeplemate.server.deps.

`get_db_user` is what makes account deletion take effect immediately. Firebase
ID tokens stay valid for up to an hour after the underlying user is destroyed
and aren't checked for revocation per request, so without this check a
just-deleted user could keep reading their own retained data until the token
expired.
"""
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from conftest import _test_user_record
from meeplemate.server.deps import (
    get_db_user,
    get_db_user_allow_deleted,
)


def _request_with(data_layer) -> MagicMock:
    request = MagicMock()
    request.app.state.deps.data_layer = data_layer
    return request


@pytest.mark.asyncio
async def test_allow_deleted_resolves_a_live_account():
    layer = AsyncMock()
    layer.upsert_user.return_value = _test_user_record()
    auth_user = MagicMock()

    record = await get_db_user_allow_deleted(_request_with(layer), auth_user)

    assert record.deleted_at is None
    layer.upsert_user.assert_awaited_once_with(
        auth_user.uid, auth_user.email, auth_user.name
    )


@pytest.mark.asyncio
async def test_allow_deleted_returns_deleted_accounts():
    """The deletion endpoint depends on this: it has to stay callable after the
    account is flagged so a client can retry a partial failure."""
    deleted = _test_user_record(deleted_at=datetime.now(UTC))
    layer = AsyncMock()
    layer.upsert_user.return_value = deleted

    record = await get_db_user_allow_deleted(_request_with(layer), MagicMock())

    assert record.deleted_at is not None


@pytest.mark.asyncio
async def test_get_db_user_passes_through_a_live_account():
    live = _test_user_record()
    assert await get_db_user(live) is live


@pytest.mark.asyncio
async def test_rejects_a_deleted_account():
    """The deleted check sits on `get_db_user`, which every account-scoped
    endpoint depends on - including the acceptance endpoint, so a deleted account
    cannot accept its way back in."""
    deleted = _test_user_record(deleted_at=datetime.now(UTC))

    with pytest.raises(HTTPException) as exc_info:
        await get_db_user(deleted)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Account deleted"
