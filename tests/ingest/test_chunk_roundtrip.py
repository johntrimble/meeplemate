"""Chunks must be readable by the code that wrote them.

build-chunks writes through PackageLayout while dataimport discovers chunks by
globbing a directory. Those two can drift — during this refactor they did, and
the result was an import that silently found zero chunks. These tests close the
loop: write via the layout, then discover via the iterators dataimport uses.
"""
from langchain_core.load import dumps
from langchain_core.documents import Document

from meeplemate.ingest.chunkbuild import (
    child_chunks_for_chunk_iter,
    chunks_for_page_iter,
    get_child_chunk_path,
    get_chunk_path,
)
from meeplemate.ingest.gamepackage import get_page, layout_for

PAGE = 0
PARENTS = 3
CHILDREN = 2


def write_chunks(package):
    layout = layout_for(package)
    for chunk_idx in range(PARENTS):
        path = layout.parent_chunk("d1", PAGE, chunk_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(Document(id=f"p{chunk_idx}", page_content="parent")))
        for child_idx in range(CHILDREN):
            child = layout.child_chunk("d1", PAGE, chunk_idx, child_idx)
            child.parent.mkdir(parents=True, exist_ok=True)
            child.write_text(dumps(Document(id=f"c{chunk_idx}-{child_idx}", page_content="child")))


def test_parent_chunks_are_discoverable(package):
    write_chunks(package)
    page = get_page(package, "d1", PAGE)
    found = sorted(c.chunk_idx for c in chunks_for_page_iter(page))
    assert found == list(range(PARENTS))


def test_child_chunks_are_discoverable(package):
    write_chunks(package)
    page = get_page(package, "d1", PAGE)
    for chunk in chunks_for_page_iter(page):
        found = sorted(c.child_idx for c in child_chunks_for_chunk_iter(chunk))
        assert found == list(range(CHILDREN)), f"chunk {chunk.chunk_idx}"


def test_children_are_not_mistaken_for_parents(package):
    """The old flat layout needed a regex to keep these apart; now they are in
    separate directories, so a parent listing cannot see children at all."""
    write_chunks(package)
    page = get_page(package, "d1", PAGE)
    assert len(list(chunks_for_page_iter(page))) == PARENTS


def test_discovered_paths_are_the_paths_written(package):
    write_chunks(package)
    page = get_page(package, "d1", PAGE)
    layout = layout_for(package)
    for chunk in chunks_for_page_iter(page):
        assert get_chunk_path(chunk) == layout.parent_chunk("d1", PAGE, chunk.chunk_idx)
        assert get_chunk_path(chunk).exists()
        for child in child_chunks_for_chunk_iter(chunk):
            assert get_child_chunk_path(child) == layout.child_chunk(
                "d1", PAGE, chunk.chunk_idx, child.child_idx
            )
            assert get_child_chunk_path(child).exists()
