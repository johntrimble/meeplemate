"""add is_active to chat_message

Revision ID: g4b5c6d7e8f9
Revises: f3a4b5c6d7e8
Create Date: 2026-03-10

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "g4b5c6d7e8f9"
down_revision: Union[str, Sequence[str], None] = "f3a4b5c6d7e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_message",
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    op.create_index(
        "ix_chat_message_chat_id_is_active",
        "chat_message",
        ["chat_id", "is_active"],
    )


def downgrade() -> None:
    op.drop_index("ix_chat_message_chat_id_is_active", table_name="chat_message")
    op.drop_column("chat_message", "is_active")
