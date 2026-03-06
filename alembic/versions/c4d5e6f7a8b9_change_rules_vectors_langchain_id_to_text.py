"""change rules_vectors langchain_id to text

Revision ID: c4d5e6f7a8b9
Revises: b1c2d3e4f5a6
Create Date: 2026-03-01 00:02:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c4d5e6f7a8b9'
down_revision: Union[str, Sequence[str], None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS public.rules_vectors;")
    op.execute("""
        CREATE TABLE public.rules_vectors (
          langchain_id       text    NOT NULL,
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
    op.execute("DROP TABLE IF EXISTS public.rules_vectors;")
    op.execute("""
        CREATE TABLE public.rules_vectors (
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
