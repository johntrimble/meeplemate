# db/models.py
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from meeplemate.db.base import Base


class AppUser(Base):
    """Firebase-authenticated user record with optional metadata overrides.

    Keyed on the Firebase UID. That identifier is per-*credential*, not
    per-person: deleting the Firebase user and signing in again mints a new one,
    and the account here is a different account. That is deliberate — the price
    of deleting your account is losing your history. Rate limiting deliberately
    does *not* follow this key; see ``TokenUsage``.
    """
    __tablename__ = "app_user"

    user_id = sa.Column(sa.Text, primary_key=True)  # Firebase UID
    email = sa.Column(sa.Text, nullable=True)
    name = sa.Column(sa.Text, nullable=True)
    # NULL = live. Set on soft delete; the row and its chats survive until
    # `mm-admin purge-deleted-accounts` removes them.
    deleted_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    # Stores per-user rate limit overrides under key "rate_limits":
    # {"rate_limits": {"8H": 100000, "7D": 400000, "30D": 1000000}}
    metadata_ = sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"))


class TokenUsage(Base):
    """Token usage records for rolling-window rate limiting.

    Keyed on ``quota_key`` — the lowercased email address, falling back to the
    uid when a token carries no email — rather than on the user. A budget
    belongs to a *person*, and a person outlives any single Firebase uid, so
    keying this on the uid would let anyone clear their 30-day budget by
    deleting their account and signing up again.

    Pooling usage across an address is safe in a way that pooling *data* is not:
    inheriting someone's consumption can only ever cost you tokens, so there is
    nothing to gain by claiming an address you don't own. Enabling
    email/password sign-up alongside Google would change that — an unverified
    registration could then drain a real user's budget — and would be the point
    to require a verified email before pooling.

    Intentionally has no foreign key: rows outlive the accounts that produced
    them, which is what closes the loophole above.
    """
    __tablename__ = "token_usage"

    id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    quota_key = sa.Column(sa.Text, nullable=False)
    tokens_used = sa.Column(sa.Integer, nullable=False)
    recorded_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        sa.Index("ix_token_usage_quota_recorded", "quota_key", "recorded_at"),
        sa.Index("ix_token_usage_recorded", "recorded_at"),
    )


class Chat(Base):
    __tablename__ = "chat"

    chat_id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    game_id = sa.Column(sa.Text, nullable=False, index=True)
    # FK so purging an account takes its chats with it.
    user_id = sa.Column(
        sa.Text,
        sa.ForeignKey("app_user.user_id", ondelete="CASCADE"),
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
