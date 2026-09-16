import asyncio
from typing import AsyncIterator, Optional, Sequence

import pytest

from meeplemate.ingest.cleardata import ClearOldDataJob


class FakeStore:
    """Mimics SQLStore's contract, including that ayield_keys holds a session open."""

    def __init__(self, data: Optional[dict] = None):
        self.data = dict(data or {})
        self.open_iterators = 0
        self.max_open_iterators = 0
        self.delete_calls: list[list[str]] = []

    async def amget(self, keys):
        return [self.data.get(k) for k in keys]

    async def amdelete(self, keys):
        self.delete_calls.append(list(keys))
        for k in keys:
            self.data.pop(k, None)

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        self.open_iterators += 1
        self.max_open_iterators = max(self.max_open_iterators, self.open_iterators)
        try:
            for k in list(self.data):
                if k.startswith(prefix or ""):
                    yield k
        finally:
            self.open_iterators -= 1


class FakeVectorStore:
    def __init__(self, ids: Sequence[str] = (), partitioned: bool = True):
        self.ids = set(ids)
        self.deleted: list[list[str]] = []
        self.dropped: list[str] = []
        self.get_calls = 0
        if not partitioned:
            del self.__class__.delete_partition  # never used; see partitioned fixture

    async def aget_by_ids(self, ids):
        self.get_calls += 1
        return [_Doc(i) if i in self.ids else None for i in ids]

    async def adelete(self, ids):
        self.deleted.append(list(ids))
        self.ids -= set(ids)

    async def delete_partition(self, version):
        self.dropped.append(version)
        self.ids = {i for i in self.ids if version not in i}


class _Doc:
    def __init__(self, id):
        self.id = id


class UnpartitionedVectorStore(FakeVectorStore):
    delete_partition = None  # not callable -> forces the fallback path


class FakeBm25:
    def __init__(self):
        self.purged: list[str] = []

    async def apurge(self, version):
        self.purged.append(version)


V_OLD, V_NEW = "v-old", "v-new"


def build(vector_store, n_chunks=40):
    chunk_keys = {f"catan#{V_OLD}#{i}": "x" for i in range(n_chunks)}
    chunk_keys.update({f"catan#{V_NEW}#{i}": "x" for i in range(n_chunks)})
    return ClearOldDataJob(
        game_data_store=FakeStore({f"catan#{V_OLD}": {}, f"catan#{V_NEW}": {}}),
        game_version_store=FakeStore({"catan": f"catan#{V_NEW}"}),
        full_page_store=FakeStore(
            {f"catan#{V_OLD}#p{i}": "x" for i in range(30)}
            | {f"catan#{V_NEW}#p{i}": "x" for i in range(30)}
        ),
        vector_store=vector_store,
        docstore=FakeStore(chunk_keys),
        bm25_index=FakeBm25(),
    )


@pytest.mark.asyncio
async def test_clears_only_the_old_version():
    vs = FakeVectorStore([f"catan#{V_OLD}#0#0", f"catan#{V_NEW}#0#0"])
    job = build(vs)
    await asyncio.wait_for(job.run(), timeout=10)

    assert job.bm25_index.purged == [V_OLD]
    assert vs.dropped == [V_OLD]
    assert list(job.game_data_store.data) == [f"catan#{V_NEW}"]
    assert all(V_OLD not in k for k in job.docstore.data)
    assert all(V_OLD not in k for k in job.full_page_store.data)
    # The live version is untouched.
    assert sum(1 for k in job.docstore.data if V_NEW in k) == 40
    assert sum(1 for k in job.full_page_store.data if V_NEW in k) == 30


@pytest.mark.asyncio
async def test_partitioned_store_skips_per_chunk_vector_probing():
    vs = FakeVectorStore([f"catan#{V_OLD}#{i}#0" for i in range(40)])
    job = build(vs)
    await asyncio.wait_for(job.run(), timeout=10)

    assert vs.dropped == [V_OLD]
    assert vs.get_calls == 0, "partition drop makes probing redundant"
    assert vs.deleted == []


@pytest.mark.asyncio
async def test_unpartitioned_store_falls_back_to_per_chunk_deletes():
    vs = UnpartitionedVectorStore([f"catan#{V_OLD}#{i}#0" for i in range(40)])
    job = build(vs)
    # A tight chunk_semaphore is what would deadlock if it were shared with
    # concurrency_semaphore; 10 parents in flight, 1 statement permit.
    job.chunk_semaphore = asyncio.Semaphore(10)
    job.concurrency_semaphore = asyncio.Semaphore(1)
    await asyncio.wait_for(job.run(), timeout=10)

    assert vs.dropped == []
    assert vs.deleted, "fallback must actually delete vectors"
    assert all(V_OLD in i for batch in vs.deleted for i in batch)


@pytest.mark.asyncio
async def test_key_iteration_is_drained_before_deleting():
    """No store is iterated while it is being mutated, and no generator is held open."""
    vs = FakeVectorStore([f"catan#{V_OLD}#0#0"])
    job = build(vs)

    docstore = job.docstore
    original = docstore.amdelete

    async def checked(keys):
        assert docstore.open_iterators == 0, "deleted while ayield_keys was still open"
        return await original(keys)

    docstore.amdelete = checked
    await asyncio.wait_for(job.run(), timeout=10)
    assert docstore.max_open_iterators == 1


@pytest.mark.asyncio
async def test_deletes_are_batched_not_per_key():
    vs = FakeVectorStore([f"catan#{V_OLD}#0#0"])
    job = build(vs, n_chunks=40)
    await asyncio.wait_for(job.run(), timeout=10)

    assert len(job.docstore.delete_calls) == 1
    assert len(job.docstore.delete_calls[0]) == 40
    assert len(job.full_page_store.delete_calls) == 1
    assert len(job.full_page_store.delete_calls[0]) == 30
