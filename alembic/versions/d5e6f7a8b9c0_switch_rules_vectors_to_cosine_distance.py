"""switch rules_vectors to cosine distance

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-03-01 00:03:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd5e6f7a8b9c0'
down_revision: Union[str, Sequence[str], None] = 'c4d5e6f7a8b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.rules_vectors_embedding_hnsw_ip;")
    op.execute("""
        CREATE INDEX rules_vectors_embedding_hnsw_cosine
          ON public.rules_vectors
          USING hnsw (embedding vector_cosine_ops);
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.rules_vectors_embedding_hnsw_cosine;")
    op.execute("""
        CREATE INDEX rules_vectors_embedding_hnsw_ip
          ON public.rules_vectors
          USING hnsw (embedding vector_ip_ops);
    """)
