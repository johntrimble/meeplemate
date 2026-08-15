"""Golden tests for the page and chunk ID format.

These IDs are persisted in Postgres as vectorstore and docstore keys, so the
format is a data contract, not an implementation detail. Changing it is a
migration. This pins it so a layout refactor cannot drift into it.
"""
from meeplemate.ingest.gamepackage import (
    get_game_key_for_id_version,
    get_page_chunk_id,
    get_page_id,
)


def test_game_key_format():
    assert get_game_key_for_id_version("catan", "0199-abc") == "catan#0199-abc"


def test_page_id_format():
    assert get_page_id("catan", "0199-abc", "ZG9jLnBkZg==", 3) == (
        "catan#0199-abc#ZG9jLnBkZg==#3"
    )


def test_page_chunk_id_format():
    assert get_page_chunk_id("catan", "0199-abc", "ZG9jLnBkZg==", 3, 7) == (
        "catan#0199-abc#ZG9jLnBkZg==#3#7"
    )


def test_version_occupies_the_second_field():
    """import-documents rewrites the version by index, so its position matters.

    add_game_metadata_to_document splits on '#' and assigns parts[1]. An empty
    version must still leave that slot addressable.
    """
    page_id = get_page_id("catan", "", "ZG9jLnBkZg==", 3)
    parts = page_id.split("#")
    assert len(parts) > 1
    parts[1] = "0199-abc"
    assert "#".join(parts) == get_page_id("catan", "0199-abc", "ZG9jLnBkZg==", 3)
