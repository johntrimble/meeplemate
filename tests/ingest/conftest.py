"""Fixtures for ingest tests that need a real package directory on disk.

The fixture builds paths through the ingest path helpers rather than hardcoding
them, so it follows the package layout through the refactor instead of pinning
one particular shape.
"""
import hashlib
from pathlib import Path

import pytest

from meeplemate.ingest.gamepackage import (
    GamePackage,
    get_page,
    load_game_package,
    page_structured_path,
    write_manifest,
)
from meeplemate.ingest.layout import PackageLayout
from meeplemate.util import spit_json

# Two rulebooks, three pages each. Page 0 of d1 ends mid-sentence and page 1
# opens with its continuation, so the cross-page merge in build-text has
# something to do.
STRUCTURED: dict[str, list[list[dict]]] = {
    "d1": [
        [
            {"type": "title", "content": "# Setup"},
            {"type": "text", "content": "Place the board"},
        ],
        [
            {"type": "text", "content": "in the center."},
            {"type": "text", "content": "Shuffle the deck."},
        ],
        [
            {"type": "text", "content": "Deal three cards."},
        ],
    ],
    "d2": [
        [{"type": "title", "content": "# Scoring"}],
        [{"type": "text", "content": "Count your points."}],
        [{"type": "text", "content": "Highest total wins."}],
    ],
}


GAME_VERSION = "0199-test"


def build_manifest() -> dict:
    """Only what the source rulebooks imply — no version, no summaries."""
    return dict(
        name="Test Game",
        game_id="test-game",
        rulebooks=[
            {
                "name": f"Rulebook {key}",
                "path": f"{key}.pdf",
                "document_key": key,
                "page_count": len(pages),
            }
            for key, pages in STRUCTURED.items()
        ],
    )


@pytest.fixture
def package(tmp_path: Path) -> GamePackage:
    """A package with a manifest and per-page structured OCR output."""
    root = tmp_path / "test-game"
    root.mkdir()
    write_manifest(build_manifest(), root)
    PackageLayout(root).version().write_text(GAME_VERSION)

    # Load it back so tests see the same hydrated shape the pipeline sees.
    gp = load_game_package(root)
    for document_key, pages in STRUCTURED.items():
        for page_num, blocks in enumerate(pages):
            page = get_page(gp, document_key, page_num)
            spit_json(blocks, page_structured_path(page))

    return gp


class TreeWatcher:
    """Records which files a step creates or rewrites under a package root."""

    def __init__(self, root: Path):
        self.root = root
        self._before = self.snapshot()

    def snapshot(self) -> dict[str, str]:
        return _tree_snapshot(self.root)

    def touched(self) -> set[str]:
        after = self.snapshot()
        return {p for p, digest in after.items() if self._before.get(p) != digest}


@pytest.fixture
def watch(package) -> TreeWatcher:
    """Snapshots the package tree now; call .touched() after running a step."""
    return TreeWatcher(package["path"])


@pytest.fixture
def page_blocks() -> dict[str, list[list[dict]]]:
    return STRUCTURED


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Map every file under `root` to a digest of its contents."""
    return {
        str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def package_with_pdfs(package) -> GamePackage:
    """Package plus a stand-in source PDF per rulebook, as init-game-package leaves it."""
    layout = PackageLayout(package["path"])
    layout.raw_documents().mkdir(parents=True, exist_ok=True)
    for rulebook in package["rulebooks"]:
        layout.raw_document(rulebook["path"]).write_bytes(b"%PDF-1.4 stub")
    return package


@pytest.fixture
def package_with_images(package) -> GamePackage:
    """Package plus a rendered PNG per page, as render leaves it."""
    layout = PackageLayout(package["path"])
    for document_key, pages in STRUCTURED.items():
        for page_num in range(len(pages)):
            image_path = layout.page_image(document_key, page_num)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"\x89PNG stub")
    return package
