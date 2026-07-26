import base64
import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional, cast
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from meeplemate.db.account_recovery import should_claim
from meeplemate.db.datalayer import (
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

if TYPE_CHECKING:
    from meeplemate.server.auth import AuthUser

log = structlog.get_logger(__name__)


def _user_lock_key(user_id: UUID) -> int:
    """Deterministic 63-bit positive integer for pg_advisory_xact_lock."""
    digest = hashlib.blake2b(str(user_id).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


def _email_lock_key(email: str) -> int:
    """Advisory-lock key for the account-resurrection path.

    Keyed on the email rather than the uid, because the whole point of that path
    is that the uid is new and unknown — the email is what two concurrent
    first-requests would collide on.
    """
    digest = hashlib.blake2b(email.strip().lower().encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


def _to_user_record(row: AppUser) -> UserRecord:
    return UserRecord(
        id=cast(UUID, row.id),
        uid=row.firebase_uid,
        email=row.email,
        name=row.name,
        sign_in_provider=row.sign_in_provider,
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
        self, game_id: str, user_id: UUID, pagination: Pagination
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

    async def ensure_chat(self, *, chat_id: UUID, game_id: str, user_id: UUID) -> ChatDict:
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
        self, user_id: UUID, pagination: Pagination
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

    async def get_message_owner(self, message_id: UUID) -> UUID | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Chat.user_id)
                .join(ChatMessage, ChatMessage.chat_id == Chat.chat_id)
                .where(ChatMessage.message_id == message_id)
            )
            row = result.scalar_one_or_none()
            return cast(UUID, row) if row is not None else None

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

    async def upsert_user(self, auth_user: "AuthUser") -> UserRecord:
        async with self._session_factory() as session:
            # Hot path: the uid is already known. Read first — this now runs on
            # every user-scoped request, so it must not write unconditionally.
            row = (
                await session.execute(
                    select(AppUser).where(AppUser.firebase_uid == auth_user.uid)
                )
            ).scalar_one_or_none()

            if row is not None:
                if row.deleted_at is not None:
                    # A stale token from the account that was just deleted.
                    # Never falls through to the resurrection path: doing so
                    # would let the deleting user's own in-flight request
                    # silently undo their deletion.
                    return _to_user_record(row)

                changed = {}
                if row.email != auth_user.email:
                    changed["email"] = auth_user.email
                if row.name != auth_user.name:
                    changed["name"] = auth_user.name
                if row.sign_in_provider != auth_user.sign_in_provider:
                    changed["sign_in_provider"] = auth_user.sign_in_provider
                if changed:
                    await session.execute(
                        AppUser.__table__.update()
                        .where(AppUser.id == row.id)
                        .values(**changed)
                    )
                    await session.commit()
                    await session.refresh(row)
                return _to_user_record(row)

            # Unknown uid: either a returning user whose account we can restore,
            # or somebody new. Lock on the email so two concurrent first
            # requests can't both decide to create/claim.
            if auth_user.email:
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:key)"),
                    {"key": _email_lock_key(auth_user.email)},
                )
                # Both a live account whose uid changed and a soft-deleted one
                # inside its grace period are claimable, so don't restrict to
                # deleted rows — that's what let a uid change silently fork a
                # user into a second account with a fresh rate-limit budget.
                #
                # Filter on provider in SQL, not just in the gate: one email can
                # own several rows (deleted once via Google, once via password),
                # and picking the most recent outright would let a non-matching
                # row shadow one the claimant is actually entitled to. Live rows
                # sort first, then most-recently-deleted.
                candidate = (
                    await session.execute(
                        select(AppUser)
                        .where(
                            func.lower(AppUser.email) == auth_user.email.strip().lower(),
                            AppUser.sign_in_provider == auth_user.sign_in_provider,
                        )
                        .order_by(
                            AppUser.deleted_at.is_(None).desc(),
                            AppUser.deleted_at.desc(),
                        )
                        .limit(1)
                    )
                ).scalar_one_or_none()

                if candidate is not None:
                    now = datetime.now(UTC)
                    if should_claim(candidate, auth_user, now):
                        previous_uid = candidate.firebase_uid
                        # The whole point of the internal id: re-pointing the
                        # external identity is one row. Chats and token usage
                        # already hang off `id` and don't move.
                        await session.execute(
                            AppUser.__table__.update()
                            .where(AppUser.id == candidate.id)
                            .values(
                                firebase_uid=auth_user.uid,
                                deleted_at=None,
                                name=auth_user.name,
                            )
                        )
                        was_deleted = candidate.deleted_at is not None
                        await session.commit()
                        await session.refresh(candidate)
                        log.info(
                            "account.resurrected" if was_deleted else "account.uid_adopted",
                            user_id=str(candidate.id),
                            firebase_uid=auth_user.uid,
                            previous_firebase_uid=previous_uid,
                            provider=auth_user.sign_in_provider,
                        )
                        return _to_user_record(candidate)

                    # Worth recording: a rejected claim is either a legitimate
                    # user coming back a different way, or someone probing.
                    log.warning(
                        "account.claim_rejected",
                        candidate_id=str(candidate.id),
                        firebase_uid=auth_user.uid,
                        claimant_provider=auth_user.sign_in_provider,
                        recorded_provider=candidate.sign_in_provider,
                        email_verified=auth_user.email_verified,
                        deleted_at=str(candidate.deleted_at),
                    )

            new_row = AppUser(
                firebase_uid=auth_user.uid,
                email=auth_user.email,
                name=auth_user.name,
                sign_in_provider=auth_user.sign_in_provider,
            )
            session.add(new_row)
            await session.commit()
            await session.refresh(new_row)
            return _to_user_record(new_row)

    async def soft_delete_user(self, user_id: UUID) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                AppUser.__table__.update()
                .where(AppUser.id == user_id, AppUser.deleted_at.is_(None))
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
                            select(Chat.chat_id).where(Chat.user_id == row.id)
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
                usage = int(
                    (
                        await session.execute(
                            select(func.count())
                            .select_from(TokenUsage)
                            .where(TokenUsage.user_id == row.id)
                        )
                    ).scalar_one()
                )
                summaries.append(
                    PurgeSummary(
                        id=cast(UUID, row.id),
                        email=row.email,
                        deleted_at=row.deleted_at,
                        chats=len(chat_ids),
                        messages=messages,
                        token_usage_rows=usage,
                    )
                )

                if not dry_run:
                    # chat_message has no FK to chat, so it doesn't cascade;
                    # parts do cascade from chat_message. Everything else
                    # (chat, token_usage) follows app_user.
                    if chat_ids:
                        await session.execute(
                            delete(ChatMessage).where(ChatMessage.chat_id.in_(chat_ids))
                        )
                    await session.execute(delete(AppUser).where(AppUser.id == row.id))

            if not dry_run:
                await session.commit()
            return summaries

    # --- Token usage ---

    async def get_window_stats(self, user_id: UUID, since: datetime) -> WindowStats:
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
        user_id: UUID,
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

    async def record_token_usage(self, user_id: UUID, tokens: int) -> None:
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
