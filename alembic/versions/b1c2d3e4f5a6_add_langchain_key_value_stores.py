"""add langchain_key_value_stores table

Revision ID: b1c2d3e4f5a6
Revises: e3a1d8f92b05
Create Date: 2026-03-01 00:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = 'e3a1d8f92b05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "langchain_key_value_stores",
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("namespace", "key"),
    )
    op.create_index("ix_langchain_key_value_stores_namespace", "langchain_key_value_stores", ["namespace"])
    op.create_index("ix_langchain_key_value_stores_key", "langchain_key_value_stores", ["key"])


def downgrade() -> None:
    op.drop_index("ix_langchain_key_value_stores_key")
    op.drop_index("ix_langchain_key_value_stores_namespace")
    op.drop_table("langchain_key_value_stores")
