"""Tests for Terms/Privacy acceptance.

The server *records* acceptance; it does not enforce it. `has_accepted_current`
still answers whether a stored acceptance covers the documents in force, but
nothing in the request path calls it - the frontend gate is the enforcement, and
these tests exist to keep the recording honest rather than to police access.

The property most of the endpoint tests defend: **no deploy order can cost a user
their acceptance record.** Anything that rejects a well-formed version - the old
409, or the version allow-list that looks like its natural replacement - breaks
that the moment the frontend ships ahead of the backend.
"""
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from conftest import TEST_EMAIL, TEST_FIREBASE_UID, _test_user_record
from meeplemate.db.datalayer import LEGAL_METADATA_KEY
from meeplemate.server.deps import get_db_user
from meeplemate.server.legal import (
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
# The dependency no longer gates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_db_user_passes_an_account_that_has_never_accepted():
    """The 403 this used to raise is what forced the client to await the
    acceptance POST, which put a full Cloud Run cold start in front of every new
    user's first screen. Enforcement is the frontend gate's job now."""
    user = _test_user_record(metadata={})
    assert await get_db_user(user) is user


@pytest.mark.asyncio
async def test_get_db_user_passes_an_account_on_a_stale_version():
    user = _test_user_record(metadata=_accepted(terms="1999-01-01"))
    assert await get_db_user(user) is user


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


def test_records_a_version_the_server_has_never_heard_of(api_client, mock_data_layer):
    """The frontend-deployed-first case, and the reason there is no allow-list of
    known versions.

    A newer bundle reaches users before the backend rolls out, so every client
    posts a version this process does not recognise. Rejecting those - with a 409
    as it once did, or a 400 against a `PUBLISHED_VERSIONS` set - throws away
    perfectly valid acceptances purely because of deploy ordering. Record it.
    """
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": "2099-01-01", "privacy_version": "2099-01-01"},
    )

    assert res.status_code == 204
    mock_data_layer.record_legal_acceptance.assert_awaited_once_with(
        TEST_FIREBASE_UID, "2099-01-01", "2099-01-01"
    )


def test_records_a_stale_client_bundle_verbatim(api_client, mock_data_layer):
    """A stale bundle showed the user *some* version and says which. Storing what
    it claims is the honest record; storing today's would be the lie. The client
    re-prompts itself on its next load, when its constants have moved on."""
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": "1999-01-01", "privacy_version": PRIVACY_VERSION},
    )

    assert res.status_code == 204
    mock_data_layer.record_legal_acceptance.assert_awaited_once_with(
        TEST_FIREBASE_UID, "1999-01-01", PRIVACY_VERSION
    )


@pytest.mark.parametrize(
    "bad",
    [
        "latest",
        "",
        "2026-7-28",  # not zero-padded: would break string ordering
        "2026-13-45",  # matches the pattern, is not a date
        "2026-07-28T00:00:00Z",
        "x" * 500,
    ],
)
def test_rejects_a_malformed_version(api_client, mock_data_layer, bad):
    """The one thing still refused. Versions are compared and ordered as strings
    - by `has_accepted_current` and by the repository's downgrade guard - so a
    value that isn't a zero-padded ISO date would quietly corrupt both."""
    res = api_client.post(
        ENDPOINT,
        json={"terms_version": bad, "privacy_version": PRIVACY_VERSION},
    )

    assert res.status_code == 422
    mock_data_layer.record_legal_acceptance.assert_not_awaited()


def test_account_deletion_works_without_acceptance(
    api_client, mock_data_layer, fake_firebase_delete
):
    """The decline path: a user who refuses the terms can still remove their
    account. Nothing gates on acceptance any more, but `DELETE /api/account`
    additionally has to survive the account already being flagged, which is why
    it keeps `get_db_user_allow_deleted`."""
    res = api_client.delete("/api/account")

    assert res.status_code == 204
    mock_data_layer.soft_delete_user.assert_awaited_once_with(TEST_FIREBASE_UID)


# ---------------------------------------------------------------------------
# Interaction with deletion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_deleted_account_cannot_accept():
    """Deletion is still enforced, and still by `get_db_user` - that check is
    what makes account deletion take effect before the ID token expires."""
    deleted = _test_user_record(metadata={}, deleted_at=datetime.now(UTC))

    with pytest.raises(HTTPException) as exc_info:
        await get_db_user(deleted)

    assert exc_info.value.status_code == 401


def test_unused_email_fixture_is_still_the_quota_key():
    """Guards the assumption the other tests lean on: acceptance is recorded
    against the uid, while budgets stay keyed to the email."""
    user = _test_user_record(metadata=_accepted())
    assert user.quota_key == TEST_EMAIL
    assert user.uid == TEST_FIREBASE_UID
