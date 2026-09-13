from unittest.mock import AsyncMock

import pytest

from meeplemate.ingest.dataimport import ImportStatusJob, inspect_import


class Store:
    def __init__(self, value):
        self.value = value

    async def amget(self, _keys):
        return [self.value]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "existing", "bm25", "action", "complete"),
    [
        ("wingspan#v2", {"name": "Wingspan"}, {"doc_count": 2}, "unchanged", True),
        ("wingspan#v1", None, None, "import", False),
        ("wingspan#v1", {"name": "Wingspan"}, None, "resume", False),
        ("wingspan#v2", {"name": "Wingspan"}, None, "repair-current", False),
        ("wingspan#v2", {"name": "Wingspan"}, {"doc_count": 0}, "repair-current", False),
    ],
)
async def test_inspect_import_classifies_state(current, existing, bm25, action, complete):
    builder = AsyncMock()
    builder.astatus.return_value = bm25
    result = await inspect_import(ImportStatusJob(
        gp={"game_id": "wingspan", "game_version": "v2", "path": None, "name": "Wingspan", "rulebooks": []},
        game_data_store=Store(existing),
        game_version_store=Store(current),
        bm25_builder=builder,
    ))
    assert result["action"] == action
    assert result["complete"] is complete
    assert result["desired_game_key"] == "wingspan#v2"
