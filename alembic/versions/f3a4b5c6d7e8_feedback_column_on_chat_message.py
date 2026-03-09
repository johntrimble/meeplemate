"""move feedback to chat_message column

Revision ID: f3a4b5c6d7e8
Revises: e1f2a3b4c5d6
Create Date: 2026-03-09 00:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f3a4b5c6d7e8'
down_revision: Union[str, Sequence[str], None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('message_feedback')
    op.add_column('chat_message', sa.Column('feedback', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('chat_message', 'feedback')
    op.create_table(
        'message_feedback',
        sa.Column('message_id', sa.UUID(), nullable=False),
        sa.Column('value', sa.Integer(), nullable=False),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['message_id'], ['chat_message.message_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('message_id'),
    )
