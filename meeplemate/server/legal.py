"""Terms of Use / Privacy Policy versions and the acceptance predicate.

The documents themselves live in ``frontend/src/content/`` — the backend never
renders them, it only needs to know which version is currently in force so it
can say whether a stored acceptance covers it.

The version string is the document's **effective date**, verbatim. Four places
carry it and must agree: the two markdown files, ``frontend/src/lib/legal.ts``,
and this module. ``tests/test_legal_versions.py`` enforces that; the procedure
for bumping them is in ``docs/legal.md``.

The metadata *key* lives in ``db.datalayer`` instead of here, because the
repository needs it to build the JSONB merge and the db layer must not import
the server layer.

Bump a version only when you want every user re-prompted. An edit that does not
warrant re-consent (a typo, a clarification) should leave the effective date —
and therefore the version — alone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from meeplemate.db.datalayer import LEGAL_METADATA_KEY

if TYPE_CHECKING:
    from meeplemate.db.datalayer import UserRecord

#: Effective date of ``frontend/src/content/terms.md``.
TERMS_VERSION = "2026-07-28"

#: Effective date of ``frontend/src/content/privacy.md``.
PRIVACY_VERSION = "2026-07-28"

def has_accepted_current(user: UserRecord) -> bool:
    """Whether ``user`` has accepted the versions currently in force.

    Both documents must match. They are versioned separately so that a
    privacy-only revision does not have to claim the Terms changed too, but
    acceptance is collected for the pair, so a mismatch in either one means the
    user has not agreed to what is being asked of them today.

    **This gates nothing.** It used to back a 403 in ``get_db_user``; that was
    removed (see the note there), so nothing in the request path calls this. It
    survives as a predicate to query the recorded state with — reporting, admin
    tooling, tests — and deliberately answers only "is this record current?",
    never "may this request proceed?".
    """
    accepted = user.metadata.get(LEGAL_METADATA_KEY)
    if not isinstance(accepted, dict):
        return False
    return (
        accepted.get("terms_version") == TERMS_VERSION
        and accepted.get("privacy_version") == PRIVACY_VERSION
    )
