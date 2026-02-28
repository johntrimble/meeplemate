# db/models.py
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from meeplemate.db.base import Base

class ChatMessage(Base):
    __tablename__ = "chat_message"

    message_id = sa.Column(UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
    chat_id = sa.Column(UUID(as_uuid=True), nullable=False, index=True)
    role = sa.Column(sa.Text, nullable=False)  # optionally an Enum
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now())

    parts = sa.orm.relationship("ChatMessagePart", back_populates="message", cascade="all, delete-orphan")

class ChatMessagePart(Base):
    __tablename__ = "chat_message_part"

    message_id = sa.Column(UUID(as_uuid=True), sa.ForeignKey("chat_message.message_id", ondelete="CASCADE"), primary_key=True)
    part_id = sa.Column(sa.Text, primary_key=True)
    part_type = sa.Column(sa.Text, nullable=False, index=True)
    ordinal = sa.Column(sa.Integer, nullable=False)
    payload = sa.Column(JSONB, nullable=False)
    updated_at = sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    message = sa.orm.relationship("ChatMessage", back_populates="parts")

    __table_args__ = (
        sa.Index("ix_chat_message_part_message_id_ordinal", "message_id", "ordinal"),
    )
