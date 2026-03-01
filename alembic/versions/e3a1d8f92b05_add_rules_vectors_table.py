"""add rules_vectors table

Revision ID: e3a1d8f92b05
Revises: f2b6096cc14f
Create Date: 2026-03-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e3a1d8f92b05'
down_revision: Union[str, Sequence[str], None] = 'f2b6096cc14f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.rules_vectors (
          langchain_id       uuid    NOT NULL DEFAULT gen_random_uuid(),
          game_version       uuid    NOT NULL,
          game_id            text    NOT NULL,
          content            text    NOT NULL,
          embedding          vector(768) NOT NULL,
          langchain_metadata jsonb   NOT NULL DEFAULT '{}'::jsonb,
          PRIMARY KEY (langchain_id, game_version)
        ) PARTITION BY LIST (game_version);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS rules_vectors_embedding_hnsw_ip
          ON public.rules_vectors
          USING hnsw (embedding vector_ip_ops);
    """)


def downgrade() -> None:
    # vector extension is intentionally not dropped — other objects may depend on it
    op.execute("DROP TABLE IF EXISTS public.rules_vectors;")
