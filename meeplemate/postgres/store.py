import json
from typing import Any, AsyncGenerator, AsyncIterator, Iterator, Optional, Sequence

from langchain_community.storage.sql import SQLStore
from langchain_core.load import dumpd, load
from langchain_core.load.serializable import Serializable
from langchain_core.stores import BaseStore
from sqlalchemy.ext.asyncio import AsyncEngine


class PostgresJSONStore(BaseStore[str, Any]):
    def __init__(self, *, namespace: str, engine: AsyncEngine) -> None:
        self._store = SQLStore(namespace=namespace, engine=engine)

    def _encode(self, value: Any) -> bytes:
        return json.dumps(value).encode()

    def _decode(self, raw: Optional[bytes]) -> Optional[Any]:
        return json.loads(raw) if raw is not None else None

    def mget(self, keys: Sequence[str]) -> list[Optional[Any]]:
        return [self._decode(v) for v in self._store.mget(keys)]

    async def amget(self, keys: Sequence[str]) -> list[Optional[Any]]:
        return [self._decode(v) for v in await self._store.amget(keys)]

    def mset(self, key_value_pairs: Sequence[tuple[str, Any]]) -> None:
        self._store.mset([(k, self._encode(v)) for k, v in key_value_pairs])

    async def amset(self, key_value_pairs: Sequence[tuple[str, Any]]) -> None:
        await self._store.amset([(k, self._encode(v)) for k, v in key_value_pairs])

    def mdelete(self, keys: Sequence[str]) -> None:
        self._store.mdelete(keys)

    async def amdelete(self, keys: Sequence[str]) -> None:
        await self._store.amdelete(keys)

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        yield from self._store.yield_keys(prefix=prefix)

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:  # type: ignore[override]
        async for key in self._store.ayield_keys(prefix=prefix):
            yield key


class PostgresSerializableStore[T: Serializable](PostgresJSONStore):
    def _encode(self, value: T) -> bytes:
        return json.dumps(dumpd(value)).encode()

    def _decode(self, raw: Optional[bytes]) -> Optional[T]:
        return load(json.loads(raw)) if raw is not None else None
