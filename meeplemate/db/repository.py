import base64
from datetime import datetime
from typing import Any, Optional, cast
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from meeplemate.db.datalayer import (
    BaseDataLayer,
    ChatDict,
    ChatSummary,
    MessageDict,
    MessagePart,
    PageInfo,
    Pagination,
    PaginatedResponse,
)
from meeplemate.db.models import Chat, ChatMessage, ChatMessagePart


def _encode_cursor(dt: Any) -> str:
    """Encode a datetime (from a SQLAlchemy Column) as a base64 cursor string."""
    return base64.b64encode(cast(datetime, dt).isoformat().encode()).decode()


def _decode_cursor(cursor: str) -> datetime:
    return datetime.fromisoformat(base64.b64decode(cursor).decode())


def _part_id(part: Any, ordinal: int) -> str:
    """Derive a stable part_id from the part's natural identifier."""
    part_type = part.get("type", "")
    if part_type in ("text", "reasoning") or part_type.startswith("data-"):
        return part.get("id", str(ordinal))
    elif part_type in ("source-url", "source-document"):
        return part.get("sourceId", str(ordinal))
    elif part_type == "dynamic-tool" or part_type.startswith("tool-"):
        return part.get("toolCallId", str(ordinal))
    else:
        return str(ordinal)


class PostgresDataLayer(BaseDataLayer):
    """PostgreSQL implementation of BaseDataLayer backed by SQLAlchemy async."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # --- Chats ---

    async def get_chat(self, chat_id: UUID) -> Optional[ChatDict]:
        async with self._session_factory() as session:
            result = await session.execute(select(Chat).where(Chat.chat_id == chat_id))
            chat = result.scalar_one_or_none()
            if chat is None:
                return None
            return ChatDict(
                chat_id=str(chat.chat_id),
                game_id=str(chat.game_id),
                created_at=_encode_cursor(chat.created_at),
            )

    async def list_chats(
        self, game_id: str, pagination: Pagination
    ) -> PaginatedResponse[ChatSummary]:
        async with self._session_factory() as session:
            q = (
                select(Chat)
                .where(Chat.game_id == game_id)
                .order_by(Chat.created_at.desc())
            )
            if pagination.cursor:
                q = q.where(Chat.created_at < _decode_cursor(pagination.cursor))
            q = q.limit(pagination.first + 1)

            chats = list((await session.execute(q)).scalars().all())
            has_next = len(chats) > pagination.first
            chats = chats[: pagination.first]

            summaries = []
            for chat in chats:
                title = await _derive_title(session, cast(UUID, chat.chat_id))
                summaries.append(ChatSummary(chat_id=str(chat.chat_id), title=title))

            start_cursor = _encode_cursor(chats[0].created_at) if chats else None
            end_cursor = _encode_cursor(chats[-1].created_at) if chats else None

            return PaginatedResponse(
                pageInfo=PageInfo(
                    hasNextPage=has_next,
                    startCursor=start_cursor,
                    endCursor=end_cursor,
                ),
                data=summaries,
            )

    async def create_chat(self, game_id: str) -> UUID:
        async with self._session_factory() as session:
            chat = Chat(game_id=game_id)
            session.add(chat)
            await session.commit()
            await session.refresh(chat)
            return cast(UUID, chat.chat_id)

    async def delete_chat(self, chat_id: UUID) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(select(Chat).where(Chat.chat_id == chat_id))
            if result.scalar_one_or_none() is None:
                return False
            # Parts cascade via DB FK (chat_message_part → chat_message).
            # ChatMessage has no FK to Chat in the model, so delete messages explicitly.
            await session.execute(
                delete(ChatMessage).where(ChatMessage.chat_id == chat_id)
            )
            await session.execute(delete(Chat).where(Chat.chat_id == chat_id))
            await session.commit()
            return True

    # --- Messages ---

    async def get_messages(self, chat_id: UUID) -> list[MessageDict]:
        async with self._session_factory() as session:
            messages = list(
                (
                    await session.execute(
                        select(ChatMessage)
                        .where(ChatMessage.chat_id == chat_id)
                        .order_by(ChatMessage.created_at)
                        .options(selectinload(ChatMessage.parts))
                    )
                )
                .scalars()
                .all()
            )
            return [
                MessageDict(
                    id=str(msg.message_id),
                    role=str(msg.role),
                    parts=[
                        p.payload
                        for p in sorted(msg.parts, key=lambda p: p.ordinal)
                    ],
                )
                for msg in messages
            ]

    async def save_message(
        self,
        *,
        message_id: UUID,
        chat_id: UUID,
        role: str,
        parts: list[MessagePart],
    ) -> None:
        async with self._session_factory() as session:
            session.add(ChatMessage(message_id=message_id, chat_id=chat_id, role=role))
            for ordinal, part in enumerate(parts):
                session.add(
                    ChatMessagePart(
                        message_id=message_id,
                        part_id=_part_id(part, ordinal),
                        part_type=part["type"],
                        ordinal=ordinal,
                        payload=part,
                    )
                )
            await session.commit()

    # --- Lifecycle ---

    async def close(self) -> None:
        pass  # Engine lifetime is managed by the component system


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _derive_title(session: AsyncSession, chat_id: UUID) -> str:
    """Return a title from the first user message's text, or 'New Chat'."""
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.chat_id == chat_id, ChatMessage.role == "user")
        .order_by(ChatMessage.created_at)
        .limit(1)
        .options(selectinload(ChatMessage.parts))
    )
    first_msg = result.scalar_one_or_none()
    if not first_msg:
        return "New Chat"
    return (
        "".join(
            p.payload.get("text", "")
            for p in sorted(first_msg.parts, key=lambda p: p.ordinal)
            if p.part_type == "text"
        )
        or "New Chat"
    )
