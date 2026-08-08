import copy
import json
import uuid
from typing import Any, Iterable, Optional, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError


class PartitionedAsyncPGVectorStore(AsyncPGVectorStore):
    """AsyncPGVectorStore that uses ON CONFLICT (id_column, partition_key_column)
    to support upserts on PostgreSQL tables partitioned by a non-id column."""

    partition_key_column: str

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Identical to the parent implementation except the ON CONFLICT target
        includes partition_key_column alongside id_column."""
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]
        if not metadatas:
            metadatas = [{} for _ in texts]

        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        can_inline_embed = callable(inline_embed_func)

        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            metadata_col_names = (
                ", " + ", ".join(f'"{col}"' for col in self.metadata_columns)
                if len(self.metadata_columns) > 0
                else ""
            )
            # No tsv column: lexical retrieval moved to the BM25 index over
            # parent chunks (meeplemate/postgres/bm25.py), and migration 0003
            # dropped rules_vectors.content_tsv. Re-attaching a
            # HybridSearchConfig would not resurrect it — AsyncPGVectorStore.create
            # silently blanks tsv_column when the column is absent and then falls
            # back to an unindexed to_tsvector(content) scan of the whole
            # partition, which looks like a slowdown rather than an error.
            insert_stmt = f'INSERT INTO "{self.schema_name}"."{self.table_name}"("{self.id_column}", "{self.content_column}", "{self.embedding_column}"{metadata_col_names}'
            values: dict[str, Any] = {
                "langchain_id": id,
                "content": content,
                "embedding": str([float(d) for d in embedding]),
            }
            values_stmt = "VALUES (:langchain_id, :content, :embedding"

            if not embedding and can_inline_embed:
                values_stmt = f"VALUES (:langchain_id, :content, {self.embedding_service.embed_query_inline(content)}"  # type: ignore[union-attr]

            extra = copy.deepcopy(metadata)
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f", :{metadata_column}"
                    values[metadata_column] = (
                        json.dumps(metadata[metadata_column])
                        if isinstance(metadata[metadata_column], dict)
                        else metadata[metadata_column]
                    )
                    del extra[metadata_column]
                else:
                    values_stmt += ", null"

            insert_stmt += (
                f""", "{self.metadata_json_column}")"""
                if self.metadata_json_column
                else ")"
            )
            if self.metadata_json_column:
                values_stmt += ", :extra)"
                values["extra"] = json.dumps(extra)
            else:
                values_stmt += ")"

            # Use composite conflict target to match the partitioned table's PK
            upsert_stmt = (
                f' ON CONFLICT ("{self.id_column}", "{self.partition_key_column}")'
                f' DO UPDATE SET'
                f' "{self.content_column}" = EXCLUDED."{self.content_column}",'
                f' "{self.embedding_column}" = EXCLUDED."{self.embedding_column}"'
            )

            if self.metadata_json_column:
                upsert_stmt += f', "{self.metadata_json_column}" = EXCLUDED."{self.metadata_json_column}"'

            for column in self.metadata_columns:
                upsert_stmt += f', "{column}" = EXCLUDED."{column}"'

            upsert_stmt += ";"

            query = insert_stmt + values_stmt + upsert_stmt
            async with self.engine.connect() as conn:
                await conn.execute(text(query), values)
                await conn.commit()

        return ids  # type: ignore[return-value]

    async def ensure_partition(self, partition_key_value: str) -> None:
        safe_name = partition_key_value.replace("-", "_")
        partition_table = f"{self.table_name}_{safe_name}"
        sql = (
            f'CREATE TABLE IF NOT EXISTS "{self.schema_name}"."{partition_table}"'
            f' PARTITION OF "{self.schema_name}"."{self.table_name}"'
            f" FOR VALUES IN ('{partition_key_value}');"
        )
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text(sql))
                await conn.commit()
        except ProgrammingError as e:
            if "already exists" not in str(e):
                raise

    async def delete_partition(self, partition_key_value: str) -> None:
        safe_name = partition_key_value.replace("-", "_")
        partition_table = f"{self.table_name}_{safe_name}"
        sql = f'DROP TABLE IF EXISTS "{self.schema_name}"."{partition_table}";'
        async with self.engine.connect() as conn:
            await conn.execute(text(sql))
            await conn.commit()


class PartitionedPGVectorStore(PGVectorStore):
    """PGVectorStore for tables partitioned by a non-id column.

    Delegates to PartitionedAsyncPGVectorStore so that upserts use
    ON CONFLICT (id_column, partition_key_column) rather than id_column alone.
    """

    def __init__(self, engine: PGEngine, vs: PartitionedAsyncPGVectorStore) -> None:
        # Bypass PGVectorStore's key guard — instantiation is via our own create().
        # _PGVectorStore__vs is the name-mangled attribute PGVectorStore's methods use.
        self._engine = engine
        self._PGVectorStore__vs = vs  # type: ignore[attr-defined]
        self._vs = vs

    @classmethod
    async def create(  # type: ignore[override]
        cls,
        engine: PGEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
        partition_key_column: str = "game_version",
        **kwargs: Any,
    ) -> "PartitionedPGVectorStore":
        coro = PartitionedAsyncPGVectorStore.create(
            engine, embedding_service, table_name, **kwargs
        )
        vs = cast(PartitionedAsyncPGVectorStore, await engine._run_as_async(coro))  # type: ignore[attr-defined]
        vs.partition_key_column = partition_key_column
        return cls(engine, vs)

    async def ensure_partition(self, partition_key_value: str) -> None:
        await self._engine._run_as_async(self._vs.ensure_partition(partition_key_value))

    async def delete_partition(self, partition_key_value: str) -> None:
        await self._engine._run_as_async(self._vs.delete_partition(partition_key_value))
