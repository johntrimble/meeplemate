"""The package layout rule, as tests.

    Every step owns one or more top-level entries, no entry is owned by two
    steps, and no entry nests inside another.

This is what makes the layout expressible as DVC stages later: DVC rejects
overlapping outputs outright, and cannot expand globs in `outs:`, so a step's
output has to be one nameable path.
"""
from pathlib import Path

import pytest

from meeplemate.ingest.layout import PackageLayout

STEP_OUTPUTS = PackageLayout.STEP_OUTPUTS
ALL_ENTRIES = [(step, entry) for step, entries in STEP_OUTPUTS.items() for entry in entries]

# Every accessor that yields a path, with arguments for the fixture's shape.
ACCESSOR_CALLS = [
    ("manifest", ()),
    ("raw_documents", ()),
    ("raw_document", ("book.pdf",)),
    ("version", ()),
    ("page_image", ("dk", 0)),
    ("page_raw_md", ("dk", 0)),
    ("page_structured", ("dk", 0)),
    ("page_md", ("dk", 0)),
    ("page_structured_fixed", ("dk", 0)),
    ("document_md", ("dk",)),
    ("page_number_raw", ("dk", 0)),
    ("page_number", ("dk", 0)),
    ("page_metadata", ("dk", 0)),
    ("chunks_dir", ("dk",)),
    ("parent_chunks_dir", ("dk",)),
    ("child_chunks_dir", ("dk",)),
    ("parent_chunk", ("dk", 0, 1)),
    ("child_chunk", ("dk", 0, 1, 2)),
    ("game_reference", ()),
    ("rulebook_reference", ("dk",)),
    ("rulebook_reference_short", ("dk",)),
    ("game_setting", ()),
    ("rulebook_setting", ("dk",)),
    ("presentation", ()),
    ("example_questions", ()),
]


def test_every_entry_is_a_single_top_level_component():
    for step, entry in ALL_ENTRIES:
        assert "/" not in entry, f"{step} declares a nested output: {entry}"
        assert entry not in (".", ".."), f"{step} declares {entry}"


def test_no_entry_is_owned_by_two_steps():
    owners: dict[str, list[str]] = {}
    for step, entry in ALL_ENTRIES:
        owners.setdefault(entry, []).append(step)
    shared = {entry: steps for entry, steps in owners.items() if len(steps) > 1}
    assert not shared, f"outputs with more than one owner: {shared}"


@pytest.mark.parametrize("step_a,entry_a", ALL_ENTRIES)
def test_entries_do_not_nest(step_a, entry_a):
    """DVC rejects a stage output that contains another stage's output."""
    root = Path("/pkg")
    a = root / entry_a
    for step_b, entry_b in ALL_ENTRIES:
        if (step_a, entry_a) == (step_b, entry_b):
            continue
        b = root / entry_b
        assert a != b and b not in a.parents, (
            f"{step_a}:{entry_a} overlaps {step_b}:{entry_b}"
        )


def test_every_accessor_lives_under_exactly_one_step_entry():
    """An accessor added without an owning step is a layout hole."""
    root = Path("/pkg")
    layout = PackageLayout(root)
    entry_paths = {entry: root / entry for _, entry in ALL_ENTRIES}

    for name, args in ACCESSOR_CALLS:
        produced = getattr(layout, name)(*args)
        owners = [
            entry
            for entry, path in entry_paths.items()
            if produced == path or path in produced.parents
        ]
        assert len(owners) == 1, f"{name}{args} -> {produced} owned by {owners}"


def test_accessor_list_is_complete():
    """Guards the test above: a new accessor must be added to ACCESSOR_CALLS."""
    covered = {name for name, _ in ACCESSOR_CALLS}
    public = {
        name
        for name in vars(PackageLayout)
        if not name.startswith("_")
        and callable(getattr(PackageLayout, name))
        and name not in {"step_roots"}
    }
    assert public - covered == set(), f"accessors missing from ACCESSOR_CALLS: {public - covered}"
