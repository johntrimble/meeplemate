"""The manifest has exactly one writer, and the parts that moved out of it are
folded back in at load time.

`rulebooks.yaml` used to be written by three steps: init created it,
update-version stamped a uuid into it, and generate-reference merged summaries
into it. Those are now `manifest.yaml`, `version.txt` and `reference/`, each
owned by one step. Nothing outside ingest may notice: `import_game_data`
deep-copies the loaded package into Postgres, and chatloop, search and qa_graph
read `summary` and `game_version` from there.
"""
import pytest
from click.testing import CliRunner

from meeplemate.ingest.__main__ import update_version
from meeplemate.ingest.errors import MissingStepInput
from meeplemate.ingest.gamepackage import (
    get_pages_iter,
    load_game_package,
    load_manifest,
)
from meeplemate.ingest.layout import PackageLayout


def test_update_version_writes_version_txt_and_leaves_manifest_alone(package):
    root = package["path"]
    layout = PackageLayout(root)
    before = layout.manifest().read_bytes()

    result = CliRunner().invoke(update_version, [str(root)])
    assert result.exit_code == 0, result.output

    assert layout.manifest().read_bytes() == before, "update-version rewrote the manifest"
    assert layout.version().read_text().strip()


def test_manifest_never_carries_version_or_summaries(package):
    """What init writes is only what the source rulebooks imply."""
    CliRunner().invoke(update_version, [str(package["path"])])
    manifest = load_manifest(package["path"])

    assert "game_version" not in manifest
    assert "summary" not in manifest
    assert all("summary" not in rb for rb in manifest["rulebooks"])


def test_load_hydrates_version_and_summaries(package):
    root = package["path"]
    layout = PackageLayout(root)
    layout.version().write_text("0199-hydrated\n")
    layout.game_reference().parent.mkdir(parents=True, exist_ok=True)
    layout.game_reference().write_text("The whole game, briefly.")
    layout.rulebook_reference_short("d1").write_text("Rulebook d1, briefly.")

    gp = load_game_package(root)

    assert gp["game_version"] == "0199-hydrated"
    assert gp["summary"] == "The whole game, briefly."
    by_key = {rb["document_key"]: rb for rb in gp["rulebooks"]}
    assert by_key["d1"]["summary"] == "Rulebook d1, briefly."
    # A rulebook with no summary file stays absent rather than becoming "".
    assert "summary" not in by_key["d2"]


def test_require_version_names_the_step_to_run(package):
    PackageLayout(package["path"]).version().unlink()

    with pytest.raises(MissingStepInput) as excinfo:
        load_game_package(package["path"], require_version=True)
    assert "update-version" in str(excinfo.value)

    # Without the flag a missing version is not an error.
    assert "game_version" not in load_game_package(package["path"])


def test_page_enumeration_is_index_driven(package):
    """Stray files must not change the page count.

    Pages come from the manifest's page_count, not from globbing a directory, so
    debris in a text directory cannot inflate or reorder them.
    """
    import asyncio

    layout = PackageLayout(package["path"])
    text_dir = layout.page_md("d1", 0).parent
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "0000.raw.md").write_text("stray raw markdown")
    (text_dir / "0000.junk.md").write_text("junk")
    (text_dir / "notes.md").write_text("notes")

    async def count():
        return [p.page_num async for p in get_pages_iter(package, "d1")]

    assert asyncio.run(count()) == [0, 1, 2]
