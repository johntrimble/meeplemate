"""add embedding_bge_small_15 column to rules_vectors

Revision ID: h5c6d7e8f9a0
Revises: g4b5c6d7e8f9
Create Date: 2026-03-12

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "h5c6d7e8f9a0"
down_revision: Union[str, Sequence[str], None] = "g4b5c6d7e8f9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Make existing embedding column nullable — only one embedding column will be
    # populated per row depending on which model was used for ingestion.
    op.execute(
        "ALTER TABLE public.rules_vectors ALTER COLUMN embedding DROP NOT NULL;"
    )
    # Add new 384-dim column for BAAI/bge-small-en-v1.5 embeddings.
    op.execute(
        "ALTER TABLE public.rules_vectors ADD COLUMN embedding_bge_small_15 vector(384);"
    )
    # HNSW cosine index — same setup as the existing embedding index.
    op.execute("""
        CREATE INDEX rules_vectors_embedding_bge_small_15_hnsw_cosine
          ON public.rules_vectors
          USING hnsw (embedding_bge_small_15 vector_cosine_ops);
    """)


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS public.rules_vectors_embedding_bge_small_15_hnsw_cosine;"
    )
    op.execute(
        "ALTER TABLE public.rules_vectors DROP COLUMN IF EXISTS embedding_bge_small_15;"
    )
    op.execute(
        "ALTER TABLE public.rules_vectors ALTER COLUMN embedding SET NOT NULL;"
    )
