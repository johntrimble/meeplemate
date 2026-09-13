from unittest.mock import AsyncMock

import pytest

from meeplemate.ingest.cleardata import ClearOldDataJob


class Store:
    def __init__(self, values=(), current=None):
        self.values = list(values)
        self.current = current
        self.deleted = []

    async def amget(self, _keys):
        return [self.current]

    async def ayield_keys(self, prefix=None):
        for value in self.values:
            if prefix is None or value.startswith(prefix):
                yield value

    async def amdelete(self, keys):
        self.deleted.extend(keys)


@pytest.mark.asyncio
async def test_cleanup_dry_run_lists_without_deleting_exact_prefix_versions():
    game_data = Store(["game#1", "game#10"])
    versions = Store(current="game#1")
    job = ClearOldDataJob(
        game_data_store=game_data,
        game_version_store=versions,
        full_page_store=Store(),
        vector_store=AsyncMock(),
        docstore=Store(),
        bm25_index=AsyncMock(),
        dry_run=True,
    )

    await job.clear_old_game_data("game")

    assert game_data.deleted == []
    job.bm25_index.apurge.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_removes_only_noncurrent_exact_versions():
    game_data = Store(["game#1", "game#10"])
    versions = Store(current="game#1")
    job = ClearOldDataJob(
        game_data_store=game_data,
        game_version_store=versions,
        full_page_store=Store(),
        vector_store=AsyncMock(),
        docstore=Store(),
        bm25_index=AsyncMock(),
    )
    job.clear_old_version_data = AsyncMock()

    await job.clear_old_game_data("game")

    job.clear_old_version_data.assert_awaited_once_with("game#10")


@pytest.mark.asyncio
async def test_targeted_cleanup_refuses_current_version():
    job = ClearOldDataJob(
        game_data_store=Store(),
        game_version_store=Store(current="game#1"),
        full_page_store=Store(),
        vector_store=AsyncMock(),
        docstore=Store(),
        bm25_index=AsyncMock(),
    )
    job.clear_old_version_data = AsyncMock()

    with pytest.raises(ValueError, match="published game version"):
        await job.clear_unpublished_version("game", "game#1")

    job.clear_old_version_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_targeted_cleanup_removes_only_requested_unpublished_version():
    job = ClearOldDataJob(
        game_data_store=Store(),
        game_version_store=Store(current="game#1"),
        full_page_store=Store(),
        vector_store=AsyncMock(),
        docstore=Store(),
        bm25_index=AsyncMock(),
    )
    job.clear_old_version_data = AsyncMock()

    await job.clear_unpublished_version("game", "game#10")

    job.clear_old_version_data.assert_awaited_once_with("game#10")
