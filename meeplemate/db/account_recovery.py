"""Rules governing whether a known person arriving under an unknown uid may
take over an existing account.

Firebase uids are not stable identifiers for a person. Deleting a Firebase user
destroys the record, so signing in again mints a brand-new uid; the dev emulator
has no persistence, so it does the same on every restart. Either way the only
stable thing to match on is the email address — and matching on email is exactly
where an account takeover would come from. This module holds that decision, kept
free of database and Firebase dependencies so the rules can be read and tested
directly.

The threat: an attacker wants a Firebase ID token bearing the victim's email
under a uid we haven't seen. The realistic route is registering the victim's
email under a *different* provider — email/password against a project that also
allows Google. ``sign_in_provider`` matching is what closes it: a password
account can never claim a record provisioned through Google, whatever its
verification state.

``email_verified`` is deliberately the *secondary* control. Firebase has a
long-standing bug where the flag reads false even for Google sign-ins
(firebase-js-sdk#7702), so requiring it outright would lock legitimate users out
of restoring their own accounts. A trusted IdP satisfies it instead.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Protocol

# How long a soft-deleted account can still be restored.
#
# Enforced here rather than only in the purge job, so what a returning user
# experiences doesn't depend on when an operator last ran
# `mm-admin purge-deleted-accounts`.
#
# Do not lower this below the longest rate-limit window (30D, see
# server/rate_limit.py). Token usage follows the account, so once the window
# lapses a returning user starts a fresh one with an empty ledger; while that
# ledger would still be inside a rate-limit window, deleting and signing back in
# would clear a budget that is still being enforced.
RESURRECTION_WINDOW = timedelta(days=90)

# Providers that verify the email address themselves, so a false
# `email_verified` claim from them is a Firebase quirk rather than a signal.
TRUSTED_IDPS = frozenset({"google.com"})


class AccountRow(Protocol):
    """The fields of a candidate ``app_user`` row the gate needs."""
    email: Optional[str]
    sign_in_provider: Optional[str]
    deleted_at: Optional[datetime]


class Claimant(Protocol):
    """The fields of an authenticated user the gate needs (see ``AuthUser``)."""
    email: Optional[str]
    email_verified: bool
    sign_in_provider: Optional[str]


def emails_match(a: Optional[str], b: Optional[str]) -> bool:
    """Case-insensitive email comparison. A missing email never matches."""
    if not a or not b:
        return False
    return a.strip().lower() == b.strip().lower()


def within_window(deleted_at: Optional[datetime], now: datetime) -> bool:
    """Whether a soft delete is recent enough to still be reversible."""
    if deleted_at is None:
        return False
    return (now - deleted_at) <= RESURRECTION_WINDOW


def should_claim(row: AccountRow, claimant: Claimant, now: datetime) -> bool:
    """Whether ``claimant`` may take over the account in ``row``.

    Covers both reasons a known person can arrive under an unknown uid:

    * **Adoption** — the account is live but its external uid changed. Firebase
      shouldn't do this in production (a Google identity keeps its uid), but the
      dev emulator has no persistence, so every container restart hands the same
      person a new uid. Creating a second account there is not merely untidy: it
      splits their chat history and hands them a *fresh rate-limit budget*.
    * **Resurrection** — the account was soft-deleted and they're back inside the
      grace period.

    A False result is not an error — the caller creates a fresh account instead,
    so a rejected claimant gets a clean slate rather than a failed sign-in.
    """
    if not emails_match(row.email, claimant.email):
        return False

    # Primary control. A row with no recorded provider is unclaimable: we can't
    # prove the claimant is the same identity, so we don't guess. Such rows date
    # from before provider tracking and become claimable after one ordinary
    # sign-in records the provider.
    if not row.sign_in_provider or row.sign_in_provider != claimant.sign_in_provider:
        return False

    # Secondary control, with the trusted-IdP escape hatch described above.
    if not claimant.email_verified and claimant.sign_in_provider not in TRUSTED_IDPS:
        return False

    if row.deleted_at is None:
        # Adopting a live account displaces whoever currently holds its uid, so
        # it deserves a second look. It grants nothing a normal sign-in wouldn't:
        # reaching here means proving control of the same verified address at the
        # same provider, and anyone who can do that would simply be handed the
        # account's existing uid by the provider instead.
        return True

    return within_window(row.deleted_at, now)
