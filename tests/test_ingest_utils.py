import pytest

from meeplemate.ingest import (
    fix_dangling_sentences,
    markdown_text_to_structured_metadata,
    markdown_with_metadata_to_plain_markdown,
    merge_page_boundaries,
    parse_bounding_boxes,
    annotate_headers,
)


def test_parse_bounding_boxes_parses_lists():
    text = "[[1, 2, 3, 4], [5, 6, 7, 8]]"
    assert parse_bounding_boxes(text) == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_markdown_text_to_structured_metadata_splits_refs():
    text = (
        "Intro sentence.\n"
        "<|ref|>table<|/ref|><|det|>[ [1, 2, 3, 4] ]<|/det|>"
        "Table body\nMore text"
    )
    structured = markdown_text_to_structured_metadata(text)
    assert structured[0]["type"] == "text"
    assert "Intro sentence" in structured[0]["content"]

    ref_block = structured[1]
    assert ref_block["type"] == "table"
    assert ref_block["bboxes"] == [[1, 2, 3, 4]]
    assert "Table body" in ref_block["content"]


def test_fix_dangling_sentences_merges_text_blocks():
    structured = [
        {"type": "text", "content": "This is mid sentence"},
        {"type": "text", "content": "and continues here."},
    ]
    merged = fix_dangling_sentences(structured)
    assert len(merged) == 1
    assert merged[0]["content"] == "This is mid sentence and continues here."


def test_markdown_with_metadata_to_plain_markdown_merges_adjacent_text():
    structured = [
        {"type": "text", "content": "Hello"},
        {"type": "text", "content": "world."},
    ]
    plain = markdown_with_metadata_to_plain_markdown(structured)
    assert "Hello world." in plain
    assert plain.endswith("world.")


def test_merge_page_boundaries_rolls_text_to_previous_page():
    pages = [
        [{"type": "text", "content": "No ending"}],
        [{"type": "text", "content": "continues here."}],
    ]
    merged = merge_page_boundaries(pages)
    assert merged[0][0]["content"] == "No ending continues here."
    assert merged[1] == []


def test_annotate_headers_tracks_nesting_and_pops():
    blocks = [
        {"type": "title", "content": "# Setup"},
        {"type": "text", "content": "intro"},
        {"type": "title", "content": "## Board"},
        {"type": "text", "content": "board text"},
        {"type": "title", "content": "# Play"},
        {"type": "text", "content": "play text"},
    ]
    annotated, _ = annotate_headers(blocks, [])
    assert annotated[1]["headers"] == [(1, "Setup")]
    assert annotated[3]["headers"] == [(1, "Setup"), (2, "Board")]
    assert annotated[5]["headers"] == [(1, "Play")]
