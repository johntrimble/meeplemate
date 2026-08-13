"""Migrating a package from the flat layout to the per-step layout.

The migration is a rename pass, so the test that matters is that every file
lands somewhere and nothing is silently dropped. The classifier refuses to run
when it meets a path it does not recognise, which is what makes it safe to
point at 19,000 files unattended.
"""
import pytest
import yaml

from meeplemate.ingest.gamepackage import load_game_package
from meeplemate.ingest.layout import PackageLayout
from meeplemate.ingest.migrate_layout import (
    apply_plan,
    plan_migration,
    rewrite_manifest,
    verify_package,
)

DOCUMENT_KEY = "ZG9jLnBkZg=="
PAGES = 2


@pytest.fixture
def old_layout_package(tmp_path):
    """A package in the pre-refactor shape, including a `.raw.md` alongside `.md`
    so the suffix ordering is exercised."""
    root = tmp_path / "oldgame"
    document_dir = root / DOCUMENT_KEY
    (document_dir / "chunks").mkdir(parents=True)

    (root / "rulebooks.yaml").write_text(yaml.safe_dump({
        "name": "Old Game",
        "game_id": "oldgame",
        "game_version": "0199-old",
        "summary": "Game level summary.",
        "rulebooks": [{
            "name": "Rules", "path": "doc.pdf", "document_key": DOCUMENT_KEY,
            "page_count": PAGES, "summary": "Short rulebook summary.",
        }],
    }))
    (root / "raw_documents").mkdir()
    (root / "raw_documents" / "doc.pdf").write_bytes(b"%PDF stub")
    (root / "summary.md").write_text("Game level summary.")
    (root / "game_setting_summary.md").write_text("Setting.")
    (root / "presentation.yaml").write_text("emoji: '*'\n")
    (root / "example_questions.yaml").write_text("questions: []\n")

    for page_num in range(PAGES):
        base = document_dir / f"{page_num:04d}"
        base.with_suffix(".png").write_bytes(b"\x89PNG")
        base.with_suffix(".md").write_text(f"page {page_num} text")
        base.with_suffix(".raw.md").write_text(f"page {page_num} raw")
        base.with_suffix(".structured.json").write_text("[]")
        base.with_suffix(".structured.fixed.json").write_text("[]")
        base.with_suffix(".page_number.txt").write_text(str(page_num + 1))
        base.with_suffix(".page_number.raw.txt").write_text(str(page_num + 1))
        base.with_suffix(".metadata.yaml").write_text(f"page_ordinal: {page_num}\n")
        (document_dir / "chunks" / f"{page_num:04d}_chunk_0000.json").write_text("{}")
        (document_dir / "chunks" / f"{page_num:04d}_chunk_0000_child_0000.json").write_text("{}")

    (document_dir / "document.md").write_text("whole document")
    (document_dir / "summary.md").write_text("Long rulebook summary.")
    (document_dir / "summary.short.md").write_text("Short rulebook summary.")
    (document_dir / "setting_summary.md").write_text("Rulebook setting.")
    return root


def migrate(root):
    plan = plan_migration(root)
    assert plan.ok, f"unclassified: {plan.unclassified}"
    apply_plan(plan)
    rewrite_manifest(root, apply=True)
    return plan


def test_every_file_is_classified(old_layout_package):
    assert plan_migration(old_layout_package).unclassified == []


def test_refuses_to_migrate_unrecognised_files(old_layout_package):
    (old_layout_package / DOCUMENT_KEY / "mystery.xyz").write_text("?")
    plan = plan_migration(old_layout_package)
    assert not plan.ok
    assert plan.unclassified[0].name == "mystery.xyz"


def test_lands_every_artifact_in_the_new_layout(old_layout_package):
    migrate(old_layout_package)
    layout = PackageLayout(old_layout_package)

    for page_num in range(PAGES):
        assert layout.page_image(DOCUMENT_KEY, page_num).exists()
        assert layout.page_md(DOCUMENT_KEY, page_num).read_text() == f"page {page_num} text"
        assert layout.page_raw_md(DOCUMENT_KEY, page_num).read_text() == f"page {page_num} raw"
        assert layout.page_structured(DOCUMENT_KEY, page_num).exists()
        assert layout.page_structured_fixed(DOCUMENT_KEY, page_num).exists()
        assert layout.page_number(DOCUMENT_KEY, page_num).exists()
        assert layout.page_number_raw(DOCUMENT_KEY, page_num).exists()
        assert layout.page_metadata(DOCUMENT_KEY, page_num).exists()
        assert layout.parent_chunk(DOCUMENT_KEY, page_num, 0).exists()
        assert layout.child_chunk(DOCUMENT_KEY, page_num, 0, 0).exists()

    assert layout.document_md(DOCUMENT_KEY).read_text() == "whole document"
    assert layout.game_reference().read_text() == "Game level summary."
    assert layout.rulebook_reference(DOCUMENT_KEY).read_text() == "Long rulebook summary."
    assert layout.rulebook_reference_short(DOCUMENT_KEY).read_text() == "Short rulebook summary."
    assert layout.game_setting().read_text() == "Setting."
    assert layout.rulebook_setting(DOCUMENT_KEY).read_text() == "Rulebook setting."
    # raw_documents and the presentation assets stay put.
    assert layout.raw_document("doc.pdf").exists()
    assert layout.presentation().exists()
    assert layout.example_questions().exists()


def test_raw_md_does_not_displace_md(old_layout_package):
    """Suffix ordering: a `.md` rule must not swallow `.raw.md`."""
    migrate(old_layout_package)
    layout = PackageLayout(old_layout_package)
    assert layout.page_md(DOCUMENT_KEY, 0).read_text() == "page 0 text"
    assert layout.page_raw_md(DOCUMENT_KEY, 0).read_text() == "page 0 raw"


def test_manifest_splits_into_manifest_and_version(old_layout_package):
    migrate(old_layout_package)
    layout = PackageLayout(old_layout_package)

    assert not (old_layout_package / "rulebooks.yaml").exists()
    manifest = yaml.safe_load(layout.manifest().read_text())
    assert "game_version" not in manifest
    assert "summary" not in manifest
    assert all("summary" not in rb for rb in manifest["rulebooks"])
    assert layout.version().read_text() == "0199-old"


def test_loaded_package_is_unchanged_by_the_migration(old_layout_package):
    """The whole point: Postgres must see exactly what it saw before."""
    before = yaml.safe_load((old_layout_package / "rulebooks.yaml").read_text())
    migrate(old_layout_package)
    after = dict(load_game_package(old_layout_package))
    after.pop("path")
    assert after == before


def test_migration_is_idempotent(old_layout_package):
    migrate(old_layout_package)
    assert plan_migration(old_layout_package).already_migrated
    assert verify_package(old_layout_package) == []
