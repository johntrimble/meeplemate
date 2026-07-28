"""The document versions must agree across all four places that carry them.

The version *is* the effective date printed at the top of each document, and it
is what the acceptance record stores. If the markdown, the frontend constant and
the backend constant drift apart, the failure is silent and ugly: users accept a
document whose date says one thing while the record says another, or the gate
prompts forever because the client keeps posting a version the server rejects.

Cheap to check, so check it. The bump procedure is in docs/legal.md.
"""
import re
from pathlib import Path

import pytest

from meeplemate.server.legal import PRIVACY_VERSION, TERMS_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = REPO_ROOT / "frontend" / "src" / "content"
FRONTEND_CONSTANTS = REPO_ROOT / "frontend" / "src" / "lib" / "legal.ts"

# "**Effective date: July 28, 2026**" -> "2026-07-28"
_EFFECTIVE_DATE = re.compile(r"\*\*Effective date:\s*(?P<date>[^*]+?)\s*\*\*")
_MONTHS = {
    "January": "01", "February": "02", "March": "03", "April": "04",
    "May": "05", "June": "06", "July": "07", "August": "08",
    "September": "09", "October": "10", "November": "11", "December": "12",
}

# The tests are skipped rather than failed when the frontend isn't on disk:
# the backend is installed as a package in CI images that carry no frontend,
# and a skip there is honest where a failure would be noise.
_needs_frontend = pytest.mark.skipif(
    not CONTENT_DIR.is_dir() or not FRONTEND_CONSTANTS.is_file(),
    reason="frontend sources not present",
)


def _effective_date(markdown_path: Path) -> str:
    match = _EFFECTIVE_DATE.search(markdown_path.read_text(encoding="utf-8"))
    assert match, f"no '**Effective date: ...**' line in {markdown_path.name}"
    month_name, day, year = match.group("date").replace(",", "").split()
    return f"{year}-{_MONTHS[month_name]}-{int(day):02d}"


def _frontend_constant(name: str) -> str:
    source = FRONTEND_CONSTANTS.read_text(encoding="utf-8")
    match = re.search(rf"export const {name} = '(?P<value>[^']+)'", source)
    assert match, f"no `export const {name}` in legal.ts"
    return match.group("value")


@_needs_frontend
def test_terms_version_matches_the_document():
    assert TERMS_VERSION == _effective_date(CONTENT_DIR / "terms.md")


@_needs_frontend
def test_privacy_version_matches_the_document():
    assert PRIVACY_VERSION == _effective_date(CONTENT_DIR / "privacy.md")


@_needs_frontend
def test_frontend_and_backend_terms_versions_agree():
    """A mismatch here is the nastiest failure mode: the client posts a version
    the server refuses with 409, so the gate can never be satisfied and the app
    is unusable for everyone."""
    assert _frontend_constant("TERMS_VERSION") == TERMS_VERSION


@_needs_frontend
def test_frontend_and_backend_privacy_versions_agree():
    assert _frontend_constant("PRIVACY_VERSION") == PRIVACY_VERSION
