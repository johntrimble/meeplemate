"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-18

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # chat_message
    op.create_table(
        "chat_message",
        sa.Column(
            "message_id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("feedback", sa.Integer(), nullable=True),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_chat_message")),
    )
    op.create_index(op.f("ix_chat_message_chat_id"), "chat_message", ["chat_id"])
    op.create_index(
        "ix_chat_message_chat_id_is_active", "chat_message", ["chat_id", "is_active"]
    )

    # chat_message_part
    op.create_table(
        "chat_message_part",
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("part_id", sa.Text(), nullable=False),
        sa.Column("part_type", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("payload", JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_message.message_id"],
            name=op.f("fk_chat_message_part_message_id_chat_message"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", "part_id", name=op.f("pk_chat_message_part")),
    )
    op.create_index(
        "ix_chat_message_part_message_id_ordinal",
        "chat_message_part",
        ["message_id", "ordinal"],
    )
    op.create_index(
        op.f("ix_chat_message_part_part_type"), "chat_message_part", ["part_type"]
    )

    # chat
    op.create_table(
        "chat",
        sa.Column(
            "chat_id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("game_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("chat_id", name=op.f("pk_chat")),
    )
    op.create_index(op.f("ix_chat_game_id"), "chat", ["game_id"])
    op.create_index(op.f("ix_chat_user_id"), "chat", ["user_id"])

    # app_user
    op.create_table(
        "app_user",
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.PrimaryKeyConstraint("user_id", name="pk_app_user"),
    )

    # token_usage
    op.create_table(
        "token_usage",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_token_usage"),
    )
    op.create_index("ix_token_usage_user_recorded", "token_usage", ["user_id", "recorded_at"])
    op.create_index("ix_token_usage_recorded", "token_usage", ["recorded_at"])

    # langchain_key_value_stores
    op.create_table(
        "langchain_key_value_stores",
        sa.Column("namespace", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("namespace", "key"),
    )
    op.create_index(
        "ix_langchain_key_value_stores_namespace", "langchain_key_value_stores", ["namespace"]
    )
    op.create_index(
        "ix_langchain_key_value_stores_key", "langchain_key_value_stores", ["key"]
    )

    # rules_vectors — partitioned table with pgvector column; created via raw SQL
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.rules_vectors (
            langchain_id       text    NOT NULL,
            game_version       uuid    NOT NULL,
            game_id            text    NOT NULL,
            content            text    NOT NULL,
            embedding          vector(384) NOT NULL,
            langchain_metadata jsonb   NOT NULL DEFAULT '{}'::jsonb,
            content_tsv        tsvector,
            PRIMARY KEY (langchain_id, game_version)
        ) PARTITION BY LIST (game_version);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS rules_vectors_embedding_hnsw_cosine
            ON public.rules_vectors
            USING hnsw (embedding vector_cosine_ops);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS rules_vectors_content_tsv_gin
            ON public.rules_vectors
            USING gin (content_tsv);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS public.rules_vectors;")
    op.drop_index("ix_langchain_key_value_stores_key", table_name="langchain_key_value_stores")
    op.drop_index("ix_langchain_key_value_stores_namespace", table_name="langchain_key_value_stores")
    op.drop_table("langchain_key_value_stores")
    op.drop_index("ix_token_usage_recorded", table_name="token_usage")
    op.drop_index("ix_token_usage_user_recorded", table_name="token_usage")
    op.drop_table("token_usage")
    op.drop_table("app_user")
    op.drop_index(op.f("ix_chat_user_id"), table_name="chat")
    op.drop_index(op.f("ix_chat_game_id"), table_name="chat")
    op.drop_table("chat")
    op.drop_index(op.f("ix_chat_message_part_part_type"), table_name="chat_message_part")
    op.drop_index("ix_chat_message_part_message_id_ordinal", table_name="chat_message_part")
    op.drop_table("chat_message_part")
    op.drop_index("ix_chat_message_chat_id_is_active", table_name="chat_message")
    op.drop_index(op.f("ix_chat_message_chat_id"), table_name="chat_message")
    op.drop_table("chat_message")
