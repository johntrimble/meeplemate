"""Tests for the account-resurrection gate.

This is the security-critical decision in account deletion: it decides whether a
Firebase identity we've never seen may take over a soft-deleted account matched
only by email address. Written against the pure function so each rule is
exercised in isolation, without a database standing between the rule and the
assertion.
"""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional

import pytest

from meeplemate.db.account_recovery import (
    RESURRECTION_WINDOW,
    emails_match,
    should_claim,
    within_window,
)

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=UTC)


@dataclass
class Row:
    """Stands in for the candidate app_user row."""
    email: Optional[str] = "player@example.com"
    sign_in_provider: Optional[str] = "google.com"
    deleted_at: Optional[datetime] = NOW - timedelta(days=1)


@dataclass
class Claimant:
    """Stands in for the AuthUser presenting a token."""
    email: Optional[str] = "player@example.com"
    email_verified: bool = True
    sign_in_provider: Optional[str] = "google.com"


def test_allows_the_ordinary_case():
    """Same person, same provider, back inside the window."""
    assert should_claim(Row(), Claimant(), NOW) is True


# --- Adoption: a LIVE account whose external uid changed -------------------
#
# The bug this closes: upsert_user only looked up by firebase_uid, so a uid
# change forked one person into a second account — splitting their history and
# handing them a brand-new rate-limit budget. The dev emulator has no
# persistence, so it did this on every restart; 18 accounts had accumulated for
# a single address before it was noticed.


def test_allows_adopting_a_live_account_whose_uid_changed():
    assert should_claim(Row(deleted_at=None), Claimant(), NOW) is True


def test_adoption_still_requires_a_matching_provider():
    """Adoption displaces the account's current uid, so it gets no weaker a
    check than resurrection does."""
    live = Row(deleted_at=None, sign_in_provider="google.com")
    assert should_claim(live, Claimant(sign_in_provider="password"), NOW) is False


def test_adoption_still_requires_a_matching_email():
    assert should_claim(Row(deleted_at=None), Claimant(email="other@example.com"), NOW) is False


def test_adoption_ignores_the_window():
    """The grace period bounds *restoring a deleted* account. A live account is
    not deleted, so its age is irrelevant."""
    assert should_claim(Row(deleted_at=None), Claimant(), NOW + timedelta(days=3650)) is True


def test_denies_adopting_a_live_row_with_no_recorded_provider():
    """Rows predating provider tracking stay unclaimable until an ordinary
    sign-in records how they authenticate."""
    assert should_claim(Row(deleted_at=None, sign_in_provider=None), Claimant(), NOW) is False


def test_allows_google_despite_unverified_email():
    """Firebase reports email_verified=false for Google sign-ins in some cases
    (firebase-js-sdk#7702). Trusting the IdP here is what stops that bug from
    locking users out of restoring their own account."""
    assert should_claim(Row(), Claimant(email_verified=False), NOW) is True


def test_allows_email_differing_only_by_case_and_whitespace():
    row = Row(email="Player@Example.com")
    assert should_claim(row, Claimant(email=" player@example.com "), NOW) is True


# --- Denials ---------------------------------------------------------------


def test_denies_different_provider():
    """The core takeover defence: registering the victim's email under
    email/password must not claim a Google-provisioned account."""
    claimant = Claimant(sign_in_provider="password", email_verified=True)
    assert should_claim(Row(sign_in_provider="google.com"), claimant, NOW) is False


def test_denies_unverified_email_from_untrusted_provider():
    claimant = Claimant(sign_in_provider="password", email_verified=False)
    assert should_claim(Row(sign_in_provider="password"), claimant, NOW) is False


def test_denies_when_row_has_no_recorded_provider():
    """Accounts predating provider tracking can't prove how they were created,
    so they are never claimable."""
    assert should_claim(Row(sign_in_provider=None), Claimant(), NOW) is False


def test_denies_once_the_window_has_lapsed():
    stale = Row(deleted_at=NOW - RESURRECTION_WINDOW - timedelta(seconds=1))
    assert should_claim(stale, Claimant(), NOW) is False


def test_allows_at_the_window_boundary():
    edge = Row(deleted_at=NOW - RESURRECTION_WINDOW)
    assert should_claim(edge, Claimant(), NOW) is True


def test_denies_different_email():
    assert should_claim(Row(), Claimant(email="someone.else@example.com"), NOW) is False


@pytest.mark.parametrize("row_email,claim_email", [(None, "a@b.com"), ("a@b.com", None), (None, None)])
def test_denies_when_either_email_is_missing(row_email, claim_email):
    """A custom-token identity (the deploy bot) carries no email, and must never
    be treated as matching a row that also lacks one."""
    row = Row(email=row_email)
    assert should_claim(row, Claimant(email=claim_email), NOW) is False


# --- Helpers ---------------------------------------------------------------


@pytest.mark.parametrize("a,b,expected", [
    ("a@b.com", "a@b.com", True),
    ("A@B.com", "a@b.COM", True),
    (" a@b.com ", "a@b.com", True),
    ("a@b.com", "c@d.com", False),
    (None, "a@b.com", False),
    ("a@b.com", None, False),
    ("", "", False),
])
def test_emails_match(a, b, expected):
    assert emails_match(a, b) is expected


def test_within_window_rejects_none():
    assert within_window(None, NOW) is False
