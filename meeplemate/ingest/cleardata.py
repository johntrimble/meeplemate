from calendar import c
from dataclasses import dataclass, field

import asyncio
from sys import prefix
from typing import AsyncIterator, cast

from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from structlog import get_logger

from meeplemate.util import sem_guard

logger = get_logger(__name__)

@dataclass
class ClearOldDataJob:
    game_data_store: BaseStore
    game_version_store: BaseStore
    full_page_store: BaseStore
    vector_store: VectorStore
    docstore: BaseStore

    concurrency_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(10))


    async def adelete_vs(self, vs:VectorStore, ids: list[str]):
        async with self.concurrency_semaphore:
            # logger.info("Deleting vectors", num_ids=len(ids))
            return await vs.adelete(ids)


    async def amdelete_bs(self, bs:BaseStore, keys: list[str]):
        async with self.concurrency_semaphore:
            # logger.info("Deleting keys", num_keys=len(keys))
            return await bs.amdelete(keys)


    async def clear_old_chunk(self, chunk_key: str):
        # We need to clear data from the vector store. Unforunately, there isn't
        # a great way to recover all the IDs of the vectors given the
        # VectorStore interface. However, we do know that the IDs will be
        # prefixed by chunk_key followed by a # and 0-indexed suffix. We can
        # just iterate to find the relevant keys.

        prefix = f"{chunk_key}#"
        i = 0
        blocks_of = 10
        all_ids = []
        while True:
            keys = []
            for j in range(blocks_of):
                child_chunk_idx = i * blocks_of + j
                keys.append(f"{prefix}{child_chunk_idx}")
            items = await self.vector_store.aget_by_ids(keys)
            if not any(items):
                break
            all_ids.extend(item.id for item in items if item is not None)
            i += 1
        # The order here does matter. Because we stop searching for more keys
        # once we find a block with no items, if there are any missing keys in
        # the middle of the sequence, we might miss all subsequent keys. By
        # deleting in reverse order, we ensure we can still find the keys that
        # haven't been delete yet even if we encounter an error and have to
        # restart the process.
        all_ids = sorted(all_ids, reverse=True)
        for id in all_ids:
            await self.adelete_vs(self.vector_store, [id])
        await self.amdelete_bs(self.docstore, [chunk_key])
    

    async def clear_old_chunks(self, game_key_with_version: str):
        # All the chunks to delete will be prefixed by game_key_with_version
        tasks = []
        prefix = f"{game_key_with_version}#"
        async for chunk_id in cast(AsyncIterator[str], self.docstore.ayield_keys(prefix=prefix)):
            # First we clear the data for the chunk from the vector store
            tasks.append(asyncio.create_task(self.clear_old_chunk(chunk_id)))
        await asyncio.gather(*tasks)


    async def clear_data_with_prefix(self, data_store: BaseStore, prefix: str):
        async for key in cast(AsyncIterator[str], data_store.ayield_keys(prefix=prefix)):
            await self.amdelete_bs(data_store, [key])


    async def clear_old_version_data(self, game_key_with_version: str):
        # All the chunks to delete will be prefixed by game_key_with_version
        await self.clear_old_chunks(game_key_with_version)
        await self.clear_data_with_prefix(self.full_page_store, f"{game_key_with_version}#")
        await self.amdelete_bs(self.game_data_store, [game_key_with_version])

    async def clear_old_game_data(self, game_id: str):
        # Get the current version for the game
        (game_key, ) = await self.game_version_store.amget([game_id])

        # If a key is prefixed by <game_id>#, and not prefixed by game_key, then
        # it's an old version of game_id data and should be deleted
        prefix = f"{game_id}#"

        # Get all old version keys
        old_version_keys = []
        async for key in cast(AsyncIterator, self.game_data_store.ayield_keys(prefix=prefix)):
            if not key.startswith(game_key):
                old_version_keys.append(key)

        # For each old version key, clear the game data and the associated chunks
        for key in old_version_keys:
            logger.info("Clearing old version data for game", game_id=game_id, old_version_key=key)
            await self.clear_old_version_data(key)

    async def run(self):
        # Get all game IDs
        async for key in cast(AsyncIterator, self.game_version_store.ayield_keys()):
            await self.clear_old_game_data(key)
