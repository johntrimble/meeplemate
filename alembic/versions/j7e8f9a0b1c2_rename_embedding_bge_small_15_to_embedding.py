"""rename embedding_bge_small_15 to embedding (drop old embedding column)

Revision ID: j7e8f9a0b1c2
Revises: i6d7e8f9a0b1
Create Date: 2026-03-13

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "j7e8f9a0b1c2"
down_revision: Union[str, Sequence[str], None] = "i6d7e8f9a0b1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the old 768-dim embedding column and its index.
    op.execute(
        "DROP INDEX IF EXISTS public.rules_vectors_embedding_hnsw_cosine;"
    )
    op.execute(
        "ALTER TABLE public.rules_vectors DROP COLUMN IF EXISTS embedding;"
    )
    # Rename the bge-small-en-v1.5 column to the canonical name.
    op.execute(
        "ALTER TABLE public.rules_vectors RENAME COLUMN embedding_bge_small_15 TO embedding;"
    )
    # Rename the associated HNSW index to match.
    op.execute(
        "ALTER INDEX IF EXISTS public.rules_vectors_embedding_bge_small_15_hnsw_cosine RENAME TO rules_vectors_embedding_hnsw_cosine;"
    )


def downgrade() -> None:
    op.execute(
        "ALTER INDEX IF EXISTS public.rules_vectors_embedding_hnsw_cosine RENAME TO rules_vectors_embedding_bge_small_15_hnsw_cosine;"
    )
    op.execute(
        "ALTER TABLE public.rules_vectors RENAME COLUMN embedding TO embedding_bge_small_15;"
    )
    # Re-add the old column as nullable (original data is gone).
    op.execute(
        "ALTER TABLE public.rules_vectors ADD COLUMN embedding vector(768);"
    )
