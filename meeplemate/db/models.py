# db/models.py
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from meeplemate.db.base import Base


class AppUser(Base):
    """Firebase-authenticated user record with optional metadata overrides."""
    __tablename__ = "app_user"

    user_id = sa.Column(sa.Text, primary_key=True)  # Firebase UID
    email = sa.Column(sa.Text, nullable=True)
    name = sa.Column(sa.Text, nullable=True)
    # Stores per-user rate limit overrides under key "rate_limits":
    # {"rate_limits": {"8H": 100000, "7D": 400000, "30D": 1000000}}
    metadata_ = sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))


class TokenUsage(Base):
    """Token usage records for rolling-window rate limiting."""
    __tablename__ = "token_usage"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    user_id = sa.Column(sa.Text, nullable=False)
    tokens_used = sa.Column(sa.Integer, nullable=False)
    recorded_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        sa.Index("ix_token_usage_user_recorded", "user_id", "recorded_at"),
        sa.Index("ix_token_usage_recorded", "recorded_at"),
    )


class Chat(Base):
    __tablename__ = "chat"

    chat_id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    game_id = sa.Column(sa.Text, nullable=False, index=True)
    user_id = sa.Column(sa.Text, nullable=False, index=True)
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_message"

    message_id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    chat_id = sa.Column(UUID(as_uuid=True), nullable=False, index=True)
    role = sa.Column(sa.Text, nullable=False)  # optionally an Enum
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())
    feedback = sa.Column(sa.Integer, nullable=True)  # 0 = thumbs down, 1 = thumbs up, NULL = no feedback

    parts = orm.relationship("ChatMessagePart", back_populates="message", cascade="all, delete-orphan")

class ChatMessagePart(Base):
    __tablename__ = "chat_message_part"

    message_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("chat_message.message_id", ondelete="CASCADE"), primary_key=True)
    part_id = sa.Column(sa.Text, primary_key=True)
    part_type = sa.Column(sa.Text, nullable=False, index=True)
    ordinal = sa.Column(sa.Integer, nullable=False)
    payload = sa.Column(JSONB, nullable=False)
    updated_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    message = orm.relationship("ChatMessage", back_populates="parts")

    __table_args__ = (
        sa.Index("ix_chat_message_part_message_id_ordinal", "message_id", "ordinal"),
    )
