"""Tests for how a user's rate-limit budget is keyed.

`quota_key` decides whose budget a request spends. Keying it on the email rather
than the Firebase uid is what stops someone clearing their 30-day budget by
deleting their account and signing up again — the uid changes, the person
doesn't.

These tests pin *how much* normalisation happens, in both directions: too little
and one person ends up with several budgets, too much and two people share one.
"""
import pytest

from meeplemate.db.datalayer import UserRecord


def _user(email, uid="uid-1") -> UserRecord:
    return UserRecord(uid=uid, email=email, name=None)


def test_keys_on_the_email_not_the_uid():
    """The whole point: same person, different uid, same budget."""
    before = _user("player@example.com", uid="uid-before-delete")
    after = _user("player@example.com", uid="uid-after-signup")
    assert before.quota_key == after.quota_key


@pytest.mark.parametrize("email", [
    "Player@Example.com",
    "PLAYER@EXAMPLE.COM",
    "  player@example.com  ",
])
def test_case_and_whitespace_are_normalised(email):
    """Must match the lower(trim(...)) the 0002 migration backfills with, or
    migrated rows would sit under a key nothing looks up again."""
    assert _user(email).quota_key == "player@example.com"


@pytest.mark.parametrize("email", [None, "", "   "])
def test_falls_back_to_uid_when_there_is_no_usable_email(email):
    """Custom-token identities (the deploy bot) carry no email; they pool with
    nobody rather than colliding on an empty key.

    Whitespace-only counts as no email. It is truthy, so normalising after the
    emptiness check would key every such identity to "" and silently merge
    their budgets.
    """
    assert _user(email, uid="deploy-bot").quota_key == "deploy-bot"


def test_blank_emails_do_not_collide_with_each_other():
    assert _user("   ", uid="uid-a").quota_key != _user("   ", uid="uid-b").quota_key


# --- Deliberate non-normalisation ------------------------------------------
#
# Gmail treats `a.b@gmail.com`, `ab@gmail.com` and `ab+tag@gmail.com` as one
# mailbox, so canonicalising them looks appealing. We don't, because that is
# only true for gmail.com: on Workspace custom domains dots are significant, so
# collapsing them could pool two different people onto one budget and let one
# throttle the other. That failure is worse than the double budget it avoids.
#
# It also isn't reachable while sign-in is Google-only — the user never types an
# address, so there is nowhere to inject a +tag, and Gmail rejects `+` in
# usernames anyway. Enabling email/password sign-up is the point to revisit.


def test_plus_suffixes_are_not_stripped():
    assert _user("player+games@example.com").quota_key != _user("player@example.com").quota_key


def test_dots_are_not_stripped():
    assert _user("p.layer@example.com").quota_key != _user("player@example.com").quota_key
