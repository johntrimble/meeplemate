from dataclasses import dataclass, field

import asyncio
from typing import AsyncIterator, Optional, Sequence, cast

from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from structlog import get_logger

from meeplemate.postgres.bm25 import Bm25IndexBuilder
from meeplemate.util import sem_guard

logger = get_logger(__name__)

# Keys per DELETE. Large enough that the round trips stop mattering, small
# enough that the IN (...) list stays a sane statement to plan.
DELETE_BATCH = 500


def _batched(items: Sequence[str], size: int = DELETE_BATCH) -> list[Sequence[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class ClearOldDataJob:
    game_data_store: BaseStore
    game_version_store: BaseStore
    full_page_store: BaseStore
    vector_store: VectorStore
    docstore: BaseStore
    # Declared before concurrency_semaphore: dataclasses forbid a field without
    # a default after one that has one.
    bm25_index: Bm25IndexBuilder

    dry_run: bool = False
    # Guards individual statements, so that the fan-out below can never ask the
    # connection pool for more than it has.
    concurrency_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(10))
    # Bounds how many parent chunks are in flight at once. Deliberately a
    # *separate* semaphore: the work it guards acquires `concurrency_semaphore`
    # internally, and nesting a semaphore inside itself deadlocks as soon as the
    # outer tasks hold every permit.
    chunk_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(10))


    async def adelete_vs(self, vs: VectorStore, ids: list[str]):
        async with self.concurrency_semaphore:
            return await vs.adelete(ids)


    async def amdelete_bs(self, bs: BaseStore, keys: Sequence[str]):
        async with self.concurrency_semaphore:
            return await bs.amdelete(list(keys))


    async def acollect_keys(self, store: BaseStore, prefix: Optional[str] = None) -> list[str]:
        """Drain ``ayield_keys`` completely before any work is done with the result.

        langchain's ``SQLStore.ayield_keys`` holds its session — and with it a
        pooled connection, inside an open read transaction — for as long as the
        generator is alive. Iterating it while deleting therefore parks an
        idle-in-transaction connection for the whole clear: it occupies a pool
        slot, pins the xmin horizon so vacuum cannot reclaim the rows this job
        is deleting, and if anything closes the socket in the meantime the
        generator's own ``session.close()`` raises on a dead connection.

        Draining it first costs nothing: the query is not prefix-filtered in
        SQL and ``session.scalars()`` buffers every row up front, so the whole
        namespace is already in memory by the time the first key is yielded.
        """
        return [
            key
            async for key in cast(AsyncIterator[str], store.ayield_keys(prefix=prefix))
        ]


    async def child_vector_ids(self, chunk_key: str) -> list[str]:
        """Ids of the child vectors belonging to one parent chunk.

        There is no way to recover these from the VectorStore interface, but the
        ids are known to be ``{chunk_key}#{i}`` with ``i`` 0-indexed, so probe in
        blocks until a block comes back empty.
        """
        prefix = f"{chunk_key}#"
        i = 0
        blocks_of = 10
        all_ids: list[str] = []
        while True:
            keys = [f"{prefix}{i * blocks_of + j}" for j in range(blocks_of)]
            async with self.concurrency_semaphore:
                items = await self.vector_store.aget_by_ids(keys)
            if not any(items):
                break
            all_ids.extend(item.id for item in items if item is not None and item.id)
            i += 1
        return all_ids


    async def clear_old_chunk_vectors(self, chunk_key: str) -> None:
        """Delete one parent chunk's child vectors.

        Only used when the vector store cannot drop the version's partition
        wholesale — see ``clear_old_version_data``.

        Deleted in reverse id order. Because the probe above stops at the first
        empty block, a gap in the middle of the sequence would hide every id
        after it; deleting from the end means an interrupted run leaves a
        contiguous prefix that the next run can still walk.
        """
        all_ids = sorted(await self.child_vector_ids(chunk_key), reverse=True)
        for batch in _batched(all_ids):
            await self.adelete_vs(self.vector_store, list(batch))


    async def clear_old_chunks(self, game_key_with_version: str, *, drop_vectors: bool) -> None:
        """Delete every parent chunk for one game version, and optionally its vectors."""
        chunk_keys = await self.acollect_keys(self.docstore, f"{game_key_with_version}#")
        if not chunk_keys:
            return
        if drop_vectors:
            await asyncio.gather(
                *(
                    sem_guard(self.clear_old_chunk_vectors(key), self.chunk_semaphore)
                    for key in chunk_keys
                )
            )
        await asyncio.gather(
            *(self.amdelete_bs(self.docstore, batch) for batch in _batched(chunk_keys))
        )


    async def clear_data_with_prefix(self, data_store: BaseStore, prefix: str) -> None:
        keys = await self.acollect_keys(data_store, prefix)
        if not keys:
            return
        await asyncio.gather(
            *(self.amdelete_bs(data_store, batch) for batch in _batched(keys))
        )


    async def clear_old_version_data(self, game_key_with_version: str) -> None:
        game_version = game_key_with_version.split("#", 1)[1]

        # Drop the lexical index first. It is keyed only on game_version, so it
        # can always be removed cleanly; if the deletion below fails partway, a
        # version with no index degrades to vector-only, which is safe. The
        # reverse order would leave postings pointing at parents that no longer
        # exist.
        await self.bm25_index.apurge(game_version)

        # The vector store is partitioned by game_version, so dropping the
        # partition removes every child vector for this version in one DDL
        # statement. That makes the per-chunk probe-and-delete redundant —
        # it used to issue thousands of round trips against rows that were
        # about to be dropped anyway — so it now only runs as the fallback for
        # a store that cannot partition.
        #
        # Doing it up front is safe: these are superseded versions that nothing
        # queries, and the game data key is still deleted last, so an
        # interrupted run is picked up again on the next pass.
        delete_partition = getattr(self.vector_store, "delete_partition", None)
        can_drop_partition = callable(delete_partition)
        if can_drop_partition:
            await delete_partition(game_version)  # type: ignore[misc]

        await self.clear_old_chunks(game_key_with_version, drop_vectors=not can_drop_partition)
        await self.clear_data_with_prefix(self.full_page_store, f"{game_key_with_version}#")

        # Last: this key is what marks the version as still needing a clear, so
        # it must not disappear before the data it refers to.
        await self.amdelete_bs(self.game_data_store, [game_key_with_version])


    async def clear_unpublished_version(self, game_id: str, game_key: str) -> None:
        (current_game_key,) = await self.game_version_store.amget([game_id])
        if current_game_key == game_key:
            raise ValueError(f"Refusing to clear published game version: {game_key}")
        await self.clear_old_version_data(game_key)


    async def clear_old_game_data(self, game_id: str) -> None:
        # Get the current version for the game
        (game_key, ) = await self.game_version_store.amget([game_id])

        # game_data_store contains only whole-version records keyed exactly as <game_id>#<version>.
        # Descendant document keys live in separate stores, so exact inequality is required;
        # a version such as game#10 must not be mistaken for current game#1.
        prefix = f"{game_id}#"

        old_version_keys = [
            key
            for key in await self.acollect_keys(self.game_data_store, prefix)
            if key != game_key
        ]

        for key in old_version_keys:
            logger.info("Old version data for game", game_id=game_id, old_version_key=key, dry_run=self.dry_run)
            if not self.dry_run:
                await self.clear_old_version_data(key)

    async def run(self) -> None:
        for game_id in await self.acollect_keys(self.game_version_store):
            await self.clear_old_game_data(game_id)
