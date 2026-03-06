"""add user_id to chat

Revision ID: a1b2c3d4e5f6
Revises: d5e6f7a8b9c0
Create Date: 2026-03-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'd5e6f7a8b9c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add user_id with a server-side default so existing rows get a value.
    # The default 'system' is a placeholder — real rows will have a Firebase UID.
    op.add_column('chat', sa.Column('user_id', sa.Text(), nullable=False, server_default='system'))
    op.create_index(op.f('ix_chat_user_id'), 'chat', ['user_id'], unique=False)
    # Remove the server default after backfilling so future inserts must supply it.
    op.alter_column('chat', 'user_id', server_default=None)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_user_id'), table_name='chat')
    op.drop_column('chat', 'user_id')
