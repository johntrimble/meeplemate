"""add content_tsv column to rules_vectors for full-text search

Revision ID: i6d7e8f9a0b1
Revises: h5c6d7e8f9a0
Create Date: 2026-03-12

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "i6d7e8f9a0b1"
down_revision: Union[str, Sequence[str], None] = "h5c6d7e8f9a0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE public.rules_vectors ADD COLUMN content_tsv tsvector;"
    )
    # Backfill existing rows with pre-computed tsvectors.
    op.execute(
        "UPDATE public.rules_vectors SET content_tsv = to_tsvector('pg_catalog.english', content);"
    )
    # GIN index for fast full-text search.
    op.execute("""
        CREATE INDEX rules_vectors_content_tsv_gin
          ON public.rules_vectors
          USING gin (content_tsv);
    """)


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS public.rules_vectors_content_tsv_gin;"
    )
    op.execute(
        "ALTER TABLE public.rules_vectors DROP COLUMN IF EXISTS content_tsv;"
    )
