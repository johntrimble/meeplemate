import base64
import hashlib
from datetime import datetime
from typing import Any, Optional, cast
from uuid import UUID

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    UserRecord,
    WindowStats,
)
from meeplemate.db.models import AppUser, Chat, ChatMessage, ChatMessagePart, TokenUsage


def _user_lock_key(user_id: str) -> int:
    """Deterministic 63-bit positive integer for pg_advisory_xact_lock."""
    digest = hashlib.blake2b(user_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


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
                user_id=str(chat.user_id),
                created_at=_encode_cursor(chat.created_at),
            )

    async def list_chats(
        self, game_id: str, user_id: str, pagination: Pagination
    ) -> PaginatedResponse[ChatSummary]:
        async with self._session_factory() as session:
            q = (
                select(Chat)
                .where(Chat.game_id == game_id, Chat.user_id == user_id)
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

    async def create_chat(self, game_id: str, user_id: str) -> UUID:
        async with self._session_factory() as session:
            chat = Chat(game_id=game_id, user_id=user_id)
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

    async def list_recent_game_ids(
        self, user_id: str, pagination: Pagination
    ) -> PaginatedResponse[str]:
        async with self._session_factory() as session:
            latest_per_game = (
                select(Chat.game_id, func.max(Chat.created_at).label("last_active"))
                .where(Chat.user_id == user_id)
                .group_by(Chat.game_id)
                .subquery()
            )
            q = select(
                latest_per_game.c.game_id,
                latest_per_game.c.last_active,
            ).order_by(latest_per_game.c.last_active.desc())
            if pagination.cursor:
                q = q.where(latest_per_game.c.last_active < _decode_cursor(pagination.cursor))
            q = q.limit(pagination.first + 1)
            rows = list((await session.execute(q)).all())

            has_next = len(rows) > pagination.first
            rows = rows[: pagination.first]

            game_ids = [row.game_id for row in rows]
            start_cursor = _encode_cursor(rows[0].last_active) if rows else None
            end_cursor = _encode_cursor(rows[-1].last_active) if rows else None

            return PaginatedResponse(
                pageInfo=PageInfo(hasNextPage=has_next, startCursor=start_cursor, endCursor=end_cursor),
                data=game_ids,
            )

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
                    parts=[p.payload for p in sorted(msg.parts, key=lambda p: p.ordinal)],
                    feedback=msg.feedback,
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

    # --- Feedback ---

    async def get_message_owner(self, message_id: UUID) -> str | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Chat.user_id)
                .join(ChatMessage, ChatMessage.chat_id == Chat.chat_id)
                .where(ChatMessage.message_id == message_id)
            )
            row = result.scalar_one_or_none()
            return str(row) if row is not None else None

    async def upsert_feedback(self, message_id: UUID, value: int, comment: str | None = None) -> None:
        async with self._session_factory() as session:
            await session.execute(
                ChatMessage.__table__.update()
                .where(ChatMessage.message_id == message_id)
                .values(feedback=value)
            )
            await session.commit()

    async def delete_feedback(self, message_id: UUID) -> None:
        async with self._session_factory() as session:
            await session.execute(
                ChatMessage.__table__.update()
                .where(ChatMessage.message_id == message_id)
                .values(feedback=None)
            )
            await session.commit()

    # --- Users ---

    async def upsert_user(self, uid: str, email: Optional[str], name: Optional[str]) -> UserRecord:
        async with self._session_factory() as session:
            stmt = (
                pg_insert(AppUser)
                .values(user_id=uid, email=email, name=name)
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={"email": email, "name": name},
                )
                .returning(AppUser)
            )
            result = await session.execute(stmt)
            row = result.scalar_one()
            await session.commit()
            return UserRecord(
                uid=str(row.user_id),
                email=row.email,
                name=row.name,
                metadata=dict(row.metadata_ or {}),
            )

    # --- Token usage ---

    async def get_window_stats(self, user_id: str, since: datetime) -> WindowStats:
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.coalesce(func.sum(TokenUsage.tokens_used), 0),
                    func.min(TokenUsage.recorded_at),
                ).where(
                    TokenUsage.user_id == user_id,
                    TokenUsage.recorded_at >= since,
                )
            )
            total, oldest = result.one()
            return WindowStats(total_tokens=int(total), oldest_recorded_at=oldest)

    async def get_app_window_stats(self, since: datetime) -> WindowStats:
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.coalesce(func.sum(TokenUsage.tokens_used), 0),
                    func.min(TokenUsage.recorded_at),
                ).where(TokenUsage.recorded_at >= since)
            )
            total, oldest = result.one()
            return WindowStats(total_tokens=int(total), oldest_recorded_at=oldest)

    async def check_and_reserve_user(
        self,
        user_id: str,
        window_params: list[tuple[str, datetime]],
        estimated: int,
        user_limits: dict[str, int],
    ) -> list[WindowStats]:
        lock_key = _user_lock_key(user_id)
        async with self._session_factory() as session:
            await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})

            stats: list[WindowStats] = []
            violated = False
            for name, since in window_params:
                result = await session.execute(
                    select(
                        func.coalesce(func.sum(TokenUsage.tokens_used), 0),
                        func.min(TokenUsage.recorded_at),
                    ).where(
                        TokenUsage.user_id == user_id,
                        TokenUsage.recorded_at >= since,
                    )
                )
                total, oldest = result.one()
                ws = WindowStats(total_tokens=int(total), oldest_recorded_at=oldest)
                stats.append(ws)
                if int(total) + estimated > user_limits[name]:
                    violated = True

            if not violated:
                session.add(TokenUsage(user_id=user_id, tokens_used=estimated))

            await session.commit()
            return stats

    async def record_token_usage(self, user_id: str, tokens: int) -> None:
        async with self._session_factory() as session:
            session.add(TokenUsage(user_id=user_id, tokens_used=tokens))
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
