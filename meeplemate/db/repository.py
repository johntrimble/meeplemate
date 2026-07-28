import base64
import hashlib
from datetime import UTC, datetime
from typing import Any, Optional, cast
from uuid import UUID

import sqlalchemy as sa
import structlog
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from meeplemate.db.datalayer import (
    LEGAL_METADATA_KEY,
    BaseDataLayer,
    ChatDict,
    ChatSummary,
    MessageDict,
    MessagePart,
    PageInfo,
    Pagination,
    PaginatedResponse,
    PurgeSummary,
    UserRecord,
    WindowStats,
)
from meeplemate.db.models import AppUser, Chat, ChatMessage, ChatMessagePart, TokenUsage

logger = structlog.get_logger(__name__)


def _quota_lock_key(quota_key: str) -> int:
    """Deterministic 63-bit positive integer for pg_advisory_xact_lock."""
    digest = hashlib.blake2b(quota_key.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


def _to_user_record(row: AppUser) -> UserRecord:
    return UserRecord(
        uid=str(row.user_id),
        email=row.email,
        name=row.name,
        deleted_at=row.deleted_at,
        metadata=dict(row.metadata_ or {}),
    )


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

    async def ensure_chat(self, *, chat_id: UUID, game_id: str, user_id: str) -> ChatDict:
        async with self._session_factory() as session:
            # Atomic create-if-absent: on a conflicting chat_id (retry, collision, or
            # tamper) the existing row is left untouched — we never overwrite ownership.
            await session.execute(
                pg_insert(Chat)
                .values(chat_id=chat_id, game_id=game_id, user_id=user_id)
                .on_conflict_do_nothing(index_elements=[Chat.chat_id])
            )
            await session.commit()

            chat = (
                await session.execute(select(Chat).where(Chat.chat_id == chat_id))
            ).scalar_one()
            return ChatDict(
                chat_id=str(chat.chat_id),
                game_id=str(chat.game_id),
                user_id=str(chat.user_id),
                created_at=_encode_cursor(chat.created_at),
            )

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
                        .where(ChatMessage.chat_id == chat_id, ChatMessage.is_active == True)
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

    async def get_message(self, message_id: UUID) -> Optional[dict]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ChatMessage.message_id, ChatMessage.chat_id, ChatMessage.role)
                .where(ChatMessage.message_id == message_id)
            )
            row = result.one_or_none()
            if row is None:
                return None
            return {
                "message_id": str(row.message_id),
                "chat_id": str(row.chat_id),
                "role": str(row.role),
            }

    async def deactivate_messages_from(self, chat_id: UUID, from_message_id: UUID) -> None:
        async with self._session_factory() as session:
            # Find the created_at of the target message (scoped to this chat for safety)
            result = await session.execute(
                select(ChatMessage.created_at)
                .where(
                    ChatMessage.message_id == from_message_id,
                    ChatMessage.chat_id == chat_id,
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return
            threshold = row
            await session.execute(
                ChatMessage.__table__.update()
                .where(
                    ChatMessage.chat_id == chat_id,
                    ChatMessage.created_at >= threshold,
                    ChatMessage.is_active == True,
                )
                .values(is_active=False)
            )
            await session.commit()

    async def save_message(
        self,
        *,
        message_id: UUID,
        chat_id: UUID,
        role: str,
        parts: list[MessagePart],
    ) -> None:
        async with self._session_factory() as session:
            # Idempotent on message_id: a retried first stream (cold start) can re-send
            # the same client-supplied user message id, so re-saving must be a no-op
            # rather than a duplicate row / PK violation.
            await session.execute(
                pg_insert(ChatMessage)
                .values(message_id=message_id, chat_id=chat_id, role=role)
                .on_conflict_do_nothing(index_elements=[ChatMessage.message_id])
            )
            for ordinal, part in enumerate(parts):
                await session.execute(
                    pg_insert(ChatMessagePart)
                    .values(
                        message_id=message_id,
                        part_id=_part_id(part, ordinal),
                        part_type=part["type"],
                        ordinal=ordinal,
                        payload=part,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[ChatMessagePart.message_id, ChatMessagePart.part_id]
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
            row = (
                await session.execute(select(AppUser).where(AppUser.user_id == uid))
            ).scalar_one_or_none()

            if row is not None:
                # Read-before-write: this runs on every user-scoped request, so
                # an unchanged user must not cost a write. Soft-deleted rows are
                # returned untouched — the caller rejects them, and syncing
                # profile fields onto a deleted account would be pointless.
                if row.deleted_at is None:
                    changed = {}
                    if row.email != email:
                        changed["email"] = email
                    if row.name != name:
                        changed["name"] = name
                    if changed:
                        await session.execute(
                            AppUser.__table__.update()
                            .where(AppUser.user_id == uid)
                            .values(**changed)
                        )
                        await session.commit()
                        await session.refresh(row)
                return _to_user_record(row)

            # First time we've seen this uid. Insert-if-absent rather than a
            # plain INSERT: the frontend fires several account-scoped requests
            # in parallel right after sign-in, so two of them can both find no
            # row and race here, and a bare insert would lose one to a primary
            # key violation — a 500 on the user's very first request. Same
            # pattern as ensure_chat.
            await session.execute(
                pg_insert(AppUser)
                .values(user_id=uid, email=email, name=name)
                .on_conflict_do_nothing(index_elements=[AppUser.user_id])
            )
            await session.commit()
            row = (
                await session.execute(select(AppUser).where(AppUser.user_id == uid))
            ).scalar_one()
            return _to_user_record(row)

    async def record_legal_acceptance(
        self, uid: str, terms_version: str, privacy_version: str
    ) -> None:
        payload = {
            "terms_version": terms_version,
            "privacy_version": privacy_version,
            "accepted_at": datetime.now(UTC).isoformat(),
        }
        stored_terms = AppUser.metadata_[(LEGAL_METADATA_KEY, "terms_version")].astext
        stored_privacy = AppUser.metadata_[
            (LEGAL_METADATA_KEY, "privacy_version")
        ].astext
        async with self._session_factory() as session:
            # A JSONB *merge* (`||`) rather than read-modify-write. The blob is
            # shared with the per-user rate-limit overrides, so replacing it
            # wholesale would drop them, and a read-modify-write could lose a
            # concurrent override to a lost update. Merging one key is atomic.
            #
            # The WHERE clause decides whether this acceptance is worth storing.
            # Two things it has to get right:
            #
            # 1. Re-accepting what is already recorded must be a no-op, so the
            #    stored `accepted_at` keeps saying when they *first* agreed
            #    rather than when they last cleared a browser.
            # 2. An *older* acceptance must never overwrite a newer one. The
            #    endpoint records whatever version the client posted (so no
            #    deploy order can lose a record), which means a stale tab can
            #    post yesterday's version after another device already accepted
            #    today's. Without this guard that stale write would silently
            #    downgrade the record.
            #
            # So: write only when nothing is stored yet, or when a posted version
            # is strictly newer than what is stored. Versions are zero-padded ISO
            # dates — enforced by `LegalAcceptanceRequest` — so lexicographic
            # comparison is chronological. `IS NULL` covers the never-accepted
            # case, where the ->> projections are NULL rather than a string.
            await session.execute(
                AppUser.__table__.update()
                .where(
                    AppUser.user_id == uid,
                    sa.or_(
                        stored_terms.is_(None),
                        stored_privacy.is_(None),
                        stored_terms < terms_version,
                        stored_privacy < privacy_version,
                    ),
                )
                # Keyed on the Column object, not the attribute name: the
                # attribute is `metadata_` but the column is `metadata`, and a
                # kwarg would be rejected as an unconsumed column name.
                .values(
                    {
                        AppUser.metadata_: AppUser.metadata_.op("||")(
                            sa.cast({LEGAL_METADATA_KEY: payload}, JSONB)
                        )
                    }
                )
            )
            await session.commit()

    async def soft_delete_user(self, uid: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                AppUser.__table__.update()
                .where(AppUser.user_id == uid, AppUser.deleted_at.is_(None))
                .values(deleted_at=func.now())
            )
            await session.commit()
            return result.rowcount > 0

    async def purge_deleted_users(
        self, older_than: datetime, *, dry_run: bool = True
    ) -> list[PurgeSummary]:
        async with self._session_factory() as session:
            rows = list(
                (
                    await session.execute(
                        select(AppUser)
                        .where(
                            AppUser.deleted_at.isnot(None),
                            AppUser.deleted_at < older_than,
                        )
                        .order_by(AppUser.deleted_at)
                    )
                )
                .scalars()
                .all()
            )

            summaries: list[PurgeSummary] = []
            for row in rows:
                chat_ids = list(
                    (
                        await session.execute(
                            select(Chat.chat_id).where(Chat.user_id == row.user_id)
                        )
                    )
                    .scalars()
                    .all()
                )
                messages = 0
                if chat_ids:
                    messages = int(
                        (
                            await session.execute(
                                select(func.count())
                                .select_from(ChatMessage)
                                .where(ChatMessage.chat_id.in_(chat_ids))
                            )
                        ).scalar_one()
                    )
                summaries.append(
                    PurgeSummary(
                        uid=str(row.user_id),
                        email=row.email,
                        deleted_at=row.deleted_at,
                        chats=len(chat_ids),
                        messages=messages,
                    )
                )

                if not dry_run:
                    # chat_message has no FK to chat so it doesn't cascade;
                    # parts do cascade from chat_message, and chat cascades from
                    # app_user. token_usage is left alone on purpose — see the
                    # interface docstring.
                    if chat_ids:
                        await session.execute(
                            delete(ChatMessage).where(ChatMessage.chat_id.in_(chat_ids))
                        )
                    await session.execute(
                        delete(AppUser).where(AppUser.user_id == row.user_id)
                    )

            if not dry_run:
                await session.commit()
            return summaries

    # --- Token usage ---

    async def get_window_stats(self, quota_key: str, since: datetime) -> WindowStats:
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.coalesce(func.sum(TokenUsage.tokens_used), 0),
                    func.min(TokenUsage.recorded_at),
                ).where(
                    TokenUsage.quota_key == quota_key,
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
        quota_key: str,
        window_params: list[tuple[str, datetime]],
        estimated: int,
        user_limits: dict[str, int],
    ) -> list[WindowStats]:
        lock_key = _quota_lock_key(quota_key)
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
                        TokenUsage.quota_key == quota_key,
                        TokenUsage.recorded_at >= since,
                    )
                )
                total, oldest = result.one()
                ws = WindowStats(total_tokens=int(total), oldest_recorded_at=oldest)
                stats.append(ws)
                if int(total) + estimated > user_limits[name]:
                    violated = True

            if not violated:
                session.add(TokenUsage(quota_key=quota_key, tokens_used=estimated))

            await session.commit()
            return stats

    async def record_token_usage(self, quota_key: str, tokens: int) -> None:
        async with self._session_factory() as session:
            session.add(TokenUsage(quota_key=quota_key, tokens_used=tokens))
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
