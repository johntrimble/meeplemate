"""Tests for Terms/Privacy acceptance.

Two halves that have to agree: `has_accepted_current` decides whether a stored
acceptance covers the documents in force, and `get_db_user` turns that into a
403 that the client keys off. The endpoint is the only way to move a user from
the first state to the second.
"""
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from conftest import TEST_EMAIL, TEST_FIREBASE_UID, _test_user_record
from meeplemate.db.datalayer import LEGAL_METADATA_KEY
from meeplemate.server.deps import get_db_user, get_db_user_allow_unaccepted
from meeplemate.server.legal import (
    ACCEPTANCE_REQUIRED_CODE,
    PRIVACY_VERSION,
    TERMS_VERSION,
    has_accepted_current,
)

ENDPOINT = "/api/account/legal-acceptance"


def _accepted(terms: str = TERMS_VERSION, privacy: str = PRIVACY_VERSION) -> dict:
    return {
        LEGAL_METADATA_KEY: {
            "terms_version": terms,
            "privacy_version": privacy,
            "accepted_at": "2026-07-28T00:00:00+00:00",
        }
    }


# ---------------------------------------------------------------------------
# has_accepted_current
# ---------------------------------------------------------------------------

def test_accepts_when_both_versions_match():
    assert has_accepted_current(_test_user_record(metadata=_accepted()))


def test_rejects_a_user_who_has_never_accepted():
    assert not has_accepted_current(_test_user_record(metadata={}))


def test_rejects_a_stale_terms_version():
    user = _test_user_record(metadata=_accepted(terms="1999-01-01"))
    assert not has_accepted_current(user)


def test_rejects_a_stale_privacy_version():
    """Versioned separately so a privacy-only revision needn't claim the Terms
    changed - but consent is collected for the pair, so either being stale means
    they haven't agreed to what's being asked today."""
    user = _test_user_record(metadata=_accepted(privacy="1999-01-01"))
    assert not has_accepted_current(user)


def test_rejects_a_malformed_metadata_blob():
    """The blob is shared with the rate-limit overrides and is only as
    structured as whatever last wrote it, so a non-dict must not throw."""
    assert not has_accepted_current(_test_user_record(metadata={LEGAL_METADATA_KEY: "yes"}))


def test_ignores_unrelated_metadata_keys():
    metadata = {"rate_limits": {"8H": 100_000}, **_accepted()}
    assert has_accepted_current(_test_user_record(metadata=metadata))


# ---------------------------------------------------------------------------
# The dependency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_db_user_passes_an_accepting_account():
    user = _test_user_record(metadata=_accepted())
    assert await get_db_user(user) is user


@pytest.mark.asyncio
async def test_get_db_user_rejects_a_non_accepting_account():
    with pytest.raises(HTTPException) as exc_info:
        await get_db_user(_test_user_record(metadata={}))

    assert exc_info.value.status_code == 403
    # The client branches on the code, not the status.
    assert exc_info.value.detail["code"] == ACCEPTANCE_REQUIRED_CODE


@pytest.mark.asyncio
async def test_allow_unaccepted_passes_a_non_accepting_account():
    """Otherwise the acceptance endpoint would reject exactly the users who
    need to call it."""
    user = _test_user_record(metadata={})
    assert await get_db_user_allow_unaccepted(user) is user


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

def test_records_acceptance(api_client, mock_data_layer):
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": TERMS_VERSION, "privacy_version": PRIVACY_VERSION},
    )

    assert res.status_code == 204
    mock_data_layer.record_legal_acceptance.assert_awaited_once_with(
        TEST_FIREBASE_UID, TERMS_VERSION, PRIVACY_VERSION
    )


def test_rejects_a_stale_client_bundle(api_client, mock_data_layer):
    """A stale bundle showed the user the *old* documents, so recording that as
    acceptance of today's would falsify the one record that has to be true."""
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": "1999-01-01", "privacy_version": PRIVACY_VERSION},
    )

    assert res.status_code == 409
    body = res.json()["detail"]
    assert body["code"] == "legal_version_mismatch"
    # Tells the client what it should be showing.
    assert body["terms_version"] == TERMS_VERSION
    mock_data_layer.record_legal_acceptance.assert_not_awaited()


def test_rejects_a_stale_privacy_version_from_the_client(api_client, mock_data_layer):
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": TERMS_VERSION, "privacy_version": "1999-01-01"},
    )

    assert res.status_code == 409
    mock_data_layer.record_legal_acceptance.assert_not_awaited()


def test_ignores_a_client_claiming_a_future_version(api_client, mock_data_layer):
    """Versions come from the request, so the endpoint must not simply trust
    them - only the server's own constants are ever written."""
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": "2099-01-01", "privacy_version": "2099-01-01"},
    )

    assert res.status_code == 409
    mock_data_layer.record_legal_acceptance.assert_not_awaited()


def test_account_deletion_works_without_acceptance(api_client, mock_data_layer):
    """The decline path. `DELETE /api/account` depends on
    `get_db_user_allow_deleted`, which never runs the acceptance check, so a
    user who refuses the terms can still get their account removed."""
    res = api_client.delete("/api/account")

    assert res.status_code == 204
    mock_data_layer.soft_delete_user.assert_awaited_once_with(TEST_FIREBASE_UID)


# ---------------------------------------------------------------------------
# Interaction with deletion
# ---------------------------------------------------------------------------
#
# The deploy-bot invariant that forces this check to live in `get_db_user`
# rather than `get_current_user` is already covered by
# `tests/test_api_games.py`, which asserts `/api/games` resolves no account.

@pytest.mark.asyncio
async def test_a_deleted_account_cannot_accept_its_way_back_in():
    deleted = _test_user_record(metadata={}, deleted_at=datetime.now(UTC))

    with pytest.raises(HTTPException) as exc_info:
        await get_db_user_allow_unaccepted(deleted)

    assert exc_info.value.status_code == 401


def test_unused_email_fixture_is_still_the_quota_key():
    """Guards the assumption the other tests lean on: acceptance is recorded
    against the uid, while budgets stay keyed to the email."""
    user = _test_user_record(metadata=_accepted())
    assert user.quota_key == TEST_EMAIL
    assert user.uid == TEST_FIREBASE_UID
