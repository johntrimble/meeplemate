"""add app_user and token_usage tables

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'app_user',
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('email', sa.Text(), nullable=True),
        sa.Column('name', sa.Text(), nullable=True),
        sa.Column('metadata', JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.PrimaryKeyConstraint('user_id', name='pk_app_user'),
    )

    op.create_table(
        'token_usage',
        sa.Column('id', UUID(as_uuid=True), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('tokens_used', sa.Integer(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id', name='pk_token_usage'),
    )
    op.create_index('ix_token_usage_user_recorded', 'token_usage', ['user_id', 'recorded_at'])
    op.create_index('ix_token_usage_recorded', 'token_usage', ['recorded_at'])


def downgrade() -> None:
    op.drop_index('ix_token_usage_recorded', table_name='token_usage')
    op.drop_index('ix_token_usage_user_recorded', table_name='token_usage')
    op.drop_table('token_usage')
    op.drop_table('app_user')
