from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from meeplemate.ingest.dataimport import prepare_import_documents


def import_job(current_game_key: str | None):
    return SimpleNamespace(
        gp={
            "game_id": "wingspan",
            "game_version": "v2",
            "path": Path("/nonexistent-package"),
        },
        game_version_store=SimpleNamespace(
            amget=AsyncMock(return_value=[current_game_key])
        ),
        clear_data_job=SimpleNamespace(
            clear_unpublished_version=AsyncMock()
        ),
    )


@pytest.mark.asyncio
async def test_published_version_is_skipped():
    job = import_job("wingspan#v2")

    assert await prepare_import_documents(job) is False
    job.clear_data_job.clear_unpublished_version.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("current_game_key", [None, "wingspan#v1"])
async def test_unpublished_version_is_cleared_before_import(current_game_key):
    job = import_job(current_game_key)

    assert await prepare_import_documents(job) is True
    job.clear_data_job.clear_unpublished_version.assert_awaited_once_with(
        "wingspan", "wingspan#v2"
    )


@pytest.mark.asyncio
async def test_overwrite_does_not_clear_published_version():
    job = import_job("wingspan#v2")

    assert await prepare_import_documents(job, overwrite=True) is True
    job.clear_data_job.clear_unpublished_version.assert_not_awaited()
