"""add chat table

Revision ID: c3f4a5b6c7d8
Revises: a85211bff985
Create Date: 2026-02-28 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'c3f4a5b6c7d8'
down_revision: Union[str, Sequence[str], None] = 'a85211bff985'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'chat',
        sa.Column('chat_id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('game_id', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('chat_id', name=op.f('pk_chat')),
    )
    op.create_index(op.f('ix_chat_game_id'), 'chat', ['game_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_game_id'), table_name='chat')
    op.drop_table('chat')
