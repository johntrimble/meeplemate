from pathlib import Path

import pytest

from meeplemate.ingest.dataimport import ImportStatusJob, inspect_import


class Store:
    def __init__(self, value):
        self.value = value

    async def amget(self, _keys):
        return [self.value]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "complete"),
    [
        # Publication is authoritative: the pointer is written only after the
        # complete import and BM25 build succeed.
        ("wingspan#v2", True),
        ("wingspan#v1", False),
        (None, False),
    ],
)
async def test_inspect_import_checks_publication(current, complete):
    result = await inspect_import(ImportStatusJob(
        gp={"game_id": "wingspan", "game_version": "v2", "path": Path("/nonexistent-package"), "name": "Wingspan", "rulebooks": []},
        game_version_store=Store(current),
    ))
    assert result["complete"] is complete
    assert result["desired_game_key"] == "wingspan#v2"


@pytest.mark.asyncio
async def test_status_detects_missing_questions_for_current_version(tmp_path):
    (tmp_path / "example_questions.yaml").write_text("questions:\n  - How do I play?\n")
    result = await inspect_import(ImportStatusJob(
        gp={"game_id": "wingspan", "game_version": "v2", "path": tmp_path, "name": "Wingspan", "rulebooks": []},
        game_version_store=Store("wingspan#v2"),
        game_questions_store=Store(None),
    ))
    assert result["complete"] is True
    assert result["questions_match"] is False
