import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from meeplemate.config import PGSettings
from meeplemate.db.base import Base
import meeplemate.db.models  # ensure models import so metadata is populated

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Tables that exist in the database but deliberately have no SQLAlchemy model.
# Without this filter autogenerate sees them as orphans and emits drop_table()
# for each one — which was already true of rules_vectors and the langchain KV
# store before the bm25_* tables were added. They live in raw SQL inside the
# migrations because they use features the ORM cannot model (LIST partitioning,
# vector/tsvector columns, HNSW and covering indexes).
EXTERNALLY_MANAGED_TABLES = {
    "rules_vectors",
    "langchain_key_value_stores",
    "bm25_doc",
    "bm25_term",
    "bm25_posting",
    "bm25_df",
    "bm25_rulebook",
    "bm25_index_meta",
}


def include_object(object_, name, type_, reflected, compare_to) -> bool:
    """Hide externally-managed tables (and their indexes) from autogenerate."""
    if type_ == "table":
        return name not in EXTERNALLY_MANAGED_TABLES
    if type_ == "index":
        table = getattr(object_, "table", None)
        return table is None or table.name not in EXTERNALLY_MANAGED_TABLES
    return True


def run_migrations_offline() -> None:
    url = str(PGSettings().pg.build_url())
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = create_async_engine(
        PGSettings().pg.build_url(),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
