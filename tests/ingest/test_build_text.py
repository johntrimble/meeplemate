"""Pins the text-building step: cross-page sentence merge, per-page markdown,
and the concatenated document.md.

This is the step most likely to break when `ocr` is split into
render / ocr / build-text, so it is pinned before the split rather than after.
It also asserts the step's *write set* — the files it touches and no others —
which is the executable form of "every output owned by exactly one step".
"""
import asyncio

from meeplemate.ingest.gamepackage import (
    get_page,
    layout_for,
    page_md_path,
    page_structured_fixed_path,
)
from meeplemate.ingest.ocr import write_fixed_structure_and_page_markdown



def run_build_text(gp, document_key: str) -> None:
    asyncio.run(write_fixed_structure_and_page_markdown(gp, document_key))


def test_merges_sentence_across_page_boundary(package):
    run_build_text(package, "d1")

    page_0 = page_md_path(get_page(package, "d1", 0)).read_text()
    page_1 = page_md_path(get_page(package, "d1", 1)).read_text()

    # "Place the board" (page 0) + "in the center." (page 1) become one sentence
    # on page 0, and page 1 no longer carries the fragment.
    assert "Place the board in the center." in page_0
    assert "in the center." not in page_1
    assert "Shuffle the deck." in page_1


def test_document_md_concatenates_pages_in_order(package):
    run_build_text(package, "d1")

    document = layout_for(package).document_md("d1").read_text()
    assert "Place the board in the center." in document
    assert document.index("Shuffle the deck.") < document.index("Deal three cards.")
    # The merged fragment must appear once, not once per page.
    assert document.count("in the center.") == 1


def test_writes_fixed_structure_for_every_page(package, page_blocks):
    run_build_text(package, "d1")

    for page_num in range(len(page_blocks["d1"])):
        page = get_page(package, "d1", page_num)
        assert page_structured_fixed_path(page).exists()


def test_writes_only_its_own_artifacts(package, page_blocks, watch):
    """The step must not touch the manifest or another document's directory."""
    root = package["path"]
    run_build_text(package, "d1")
    touched = watch.touched()

    expected = {str(layout_for(package).document_md("d1").relative_to(root))}
    for page_num in range(len(page_blocks["d1"])):
        page = get_page(package, "d1", page_num)
        expected.add(str(page_md_path(page).relative_to(root)))
        expected.add(str(page_structured_fixed_path(page).relative_to(root)))

    assert touched == expected
