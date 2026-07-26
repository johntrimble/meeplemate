# db/models.py
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from meeplemate.db.base import Base


class AppUser(Base):
    """Application user record, authenticated by Firebase.

    ``id`` is the key everything else references: an internal UUID that never
    changes. ``firebase_uid`` is the *external* identity and is deliberately
    mutable — deleting a Firebase user and signing in again mints a brand-new
    uid, so restoring an account means re-pointing this column rather than
    rewriting every row the user owns.
    """
    __tablename__ = "app_user"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    firebase_uid = sa.Column(sa.Text, nullable=True, unique=True)
    email = sa.Column(sa.Text, nullable=True)
    name = sa.Column(sa.Text, nullable=True)
    # Which provider the token was obtained through ("google.com", "password",
    # "custom", ...). Recorded on every upsert; gates account resurrection.
    sign_in_provider = sa.Column(sa.Text, nullable=True)
    # NULL = live. Set on soft delete; cleared when an account is restored.
    deleted_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    # Stores per-user rate limit overrides under key "rate_limits":
    # {"rate_limits": {"8H": 100000, "7D": 400000, "30D": 1000000}}
    metadata_ = sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))


class TokenUsage(Base):
    """Token usage records for rolling-window rate limiting."""
    __tablename__ = "token_usage"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    user_id = sa.Column(
        UUID(as_uuid=True),
        sa.ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
    )
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
    user_id = sa.Column(
        UUID(as_uuid=True),
        sa.ForeignKey("app_user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_message"

    message_id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    chat_id = sa.Column(UUID(as_uuid=True), nullable=False, index=True)
    role = sa.Column(sa.Text, nullable=False)  # optionally an Enum
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())
    feedback = sa.Column(sa.Integer, nullable=True)  # 0 = thumbs down, 1 = thumbs up, NULL = no feedback
    is_active = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("true"))

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
