from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)
from uuid import UUID

if TYPE_CHECKING:
    from meeplemate.server.auth import AuthUser


# ---------------------------------------------------------------------------
# Provider metadata
# Matches ai-sdk's ProviderMetadata: Record<string, Record<string, JsonValue>>
# ---------------------------------------------------------------------------

ProviderMetadata = dict[str, Any]

# ---------------------------------------------------------------------------
# Message part types
# Mirrors UIMessagePart from the Vercel AI SDK (frontend/node_modules/ai).
#
# Deviations from the SDK's TypeScript definitions:
#   - `state` is omitted from text/reasoning parts (always 'done' at persist
#     time; it is a streaming artifact).
#   - `id` is added to TextMessagePart and ReasoningMessagePart to preserve
#     the id carried by the corresponding streaming chunks (text-start /
#     reasoning-start).
# ---------------------------------------------------------------------------


class TextMessagePart(TypedDict):
    type: Literal["text"]
    id: NotRequired[str]  # from text-start chunk's id
    text: str
    providerMetadata: NotRequired[ProviderMetadata]


class ReasoningMessagePart(TypedDict):
    type: Literal["reasoning"]
    id: NotRequired[str]  # from reasoning-start chunk's id
    text: str
    providerMetadata: NotRequired[ProviderMetadata]


class StepStartMessagePart(TypedDict):
    type: Literal["step-start"]


class SourceUrlMessagePart(TypedDict):
    type: Literal["source-url"]
    sourceId: str
    url: str
    title: NotRequired[str]
    providerMetadata: NotRequired[ProviderMetadata]


class SourceDocumentMessagePart(TypedDict):
    type: Literal["source-document"]
    sourceId: str
    mediaType: str
    title: str
    filename: NotRequired[str]
    providerMetadata: NotRequired[ProviderMetadata]


class FileMessagePart(TypedDict):
    type: Literal["file"]
    url: str
    mediaType: str
    filename: NotRequired[str]
    providerMetadata: NotRequired[ProviderMetadata]


class ToolMessagePart(TypedDict):
    # type is 'tool-{name}' by convention; Literal cannot be parametrized in Python
    type: str
    toolCallId: str
    state: str  # 'input-available' | 'output-available' | 'output-error' | 'output-denied' | ...
    input: NotRequired[Any]
    output: NotRequired[Any]
    errorText: NotRequired[str]
    title: NotRequired[str]
    providerExecuted: NotRequired[bool]
    callProviderMetadata: NotRequired[ProviderMetadata]
    approval: NotRequired[dict[str, Any]]


class DynamicToolMessagePart(TypedDict):
    type: Literal["dynamic-tool"]
    toolName: str
    toolCallId: str
    state: str  # same state machine as ToolMessagePart
    input: NotRequired[Any]
    output: NotRequired[Any]
    errorText: NotRequired[str]
    title: NotRequired[str]
    providerExecuted: NotRequired[bool]
    callProviderMetadata: NotRequired[ProviderMetadata]
    approval: NotRequired[dict[str, Any]]


_D = TypeVar("_D")


class DataMessagePart(TypedDict, Generic[_D]):
    # type is 'data-{name}' by convention; Literal cannot be parametrized in Python
    type: str
    id: NotRequired[str]
    data: _D


MessagePart = Union[
    TextMessagePart,
    ReasoningMessagePart,
    StepStartMessagePart,
    SourceUrlMessagePart,
    SourceDocumentMessagePart,
    FileMessagePart,
    ToolMessagePart,
    DynamicToolMessagePart,
    DataMessagePart[Any],
]

# ---------------------------------------------------------------------------
# Pagination — mirrors Chainlit's Pagination / PageInfo / PaginatedResponse
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


@dataclass
class Pagination:
    first: int
    cursor: Optional[str] = None


@dataclass
class PageInfo:
    hasNextPage: bool
    startCursor: Optional[str]
    endCursor: Optional[str]


@dataclass
class PaginatedResponse(Generic[_T]):
    pageInfo: PageInfo
    data: list[_T]


# ---------------------------------------------------------------------------
# Conversation shapes
# ---------------------------------------------------------------------------


class ChatSummary(TypedDict):
    chat_id: str
    title: str  # derived from first user message text


class ChatDict(TypedDict):
    chat_id: str
    game_id: str
    user_id: str
    created_at: str  # ISO-8601


class MessageDict(TypedDict):
    id: str
    role: str  # "user" | "assistant" | "system"
    parts: list[MessagePart]  # same shape going in and coming out
    feedback: NotRequired[int | None]  # 0 = thumbs down, 1 = thumbs up, absent/None = no feedback


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


#: Key under which Terms/Privacy acceptance is stored in ``UserRecord.metadata``.
#: Lives here rather than beside the version constants in ``server/legal.py``
#: because the repository needs it to build the merge, and the db layer must not
#: import the server layer. The versions themselves are a server concern.
LEGAL_METADATA_KEY = "legal"


@dataclass
class UserRecord:
    """A persisted application user (backed by Firebase Auth)."""
    uid: str
    email: Optional[str]
    name: Optional[str]
    deleted_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quota_key(self) -> str:
        """Key under which this user's token consumption is recorded.

        The email, not the uid: a budget belongs to a person, and deleting an
        account mints a new uid, so a uid-keyed ledger would reset the budget.
        Falls back to the uid for tokens carrying no email (the deploy bot),
        which simply means those identities pool with nobody.

        Normalisation is deliberately limited to trimming and case. It is
        tempting to also canonicalise Gmail-style — strip dots, drop a ``+tag``
        — but that is only correct for ``@gmail.com``: on Workspace custom
        domains dots are significant, so ``a.b@corp.com`` and ``ab@corp.com``
        can be two different people, and pooling their budgets would let one
        throttle the other. Getting that wrong is worse than the thing it
        prevents.

        It also isn't needed while sign-in is Google-only: the user never types
        an address, they pick an account and Google returns its canonical form,
        so there is nowhere to inject a ``+tag``. Enabling email/password
        sign-up would change that, and is the point to revisit this — alongside
        requiring a verified email before pooling at all (see ``TokenUsage``).
        """
        # Normalise *before* testing for emptiness: a whitespace-only email
        # (a malformed claim, or a typo in MM_AUTH_BYPASS_USER) would otherwise
        # pass the truthiness check and key every such identity to "".
        normalised = (self.email or "").strip().lower()
        return normalised or self.uid


@dataclass
class PurgeSummary:
    """What a purge did (or would do, on a dry run) for one account."""
    uid: str
    email: Optional[str]
    deleted_at: Optional[datetime]
    chats: int
    messages: int


@dataclass
class WindowStats:
    """Aggregate token usage for a rolling time window."""
    total_tokens: int
    oldest_recorded_at: Optional[datetime]


class BaseDataLayer(ABC):
    """Abstract persistence interface for MeepleMate.

    Inspired by Chainlit's BaseDataLayer pattern. Implementations are free to
    back this with any storage technology; the application and API code depend
    only on this interface.
    """

    # --- Chats ---

    @abstractmethod
    async def get_chat(self, chat_id: UUID) -> Optional[ChatDict]:
        """Fetch a single chat or None if not found."""

    @abstractmethod
    async def list_chats(
        self, game_id: str, user_id: str, pagination: Pagination
    ) -> PaginatedResponse[ChatSummary]:
        """All chats for a game belonging to user_id, newest first, each with a derived title."""

    @abstractmethod
    async def ensure_chat(self, *, chat_id: UUID, game_id: str, user_id: str) -> ChatDict:
        """Idempotently create the chat for a client-supplied ``chat_id`` if absent, then
        return the current row.

        Never overwrites an existing row: if ``chat_id`` already exists (e.g. a retried
        first message, or a collision/tamper), the stored row is returned unchanged.
        Callers MUST verify ownership/game on the result before writing to the chat.
        """

    @abstractmethod
    async def delete_chat(self, chat_id: UUID) -> bool:
        """Delete a chat and cascade-delete its messages. Returns True if found."""

    @abstractmethod
    async def list_recent_game_ids(
        self, user_id: str, pagination: Pagination
    ) -> PaginatedResponse[str]:
        """Distinct game_ids with the most recent chat activity for user_id, newest first.

        Cursor encodes the MAX(created_at) of the last game's most recent chat.
        """

    # --- Messages ---

    @abstractmethod
    async def get_messages(self, chat_id: UUID) -> list[MessageDict]:
        """Active messages for a chat, ordered chronologically. Includes ``feedback`` (0/1/None)."""

    @abstractmethod
    async def get_message(self, message_id: UUID) -> Optional[dict]:
        """Return minimal message info dict with keys 'message_id', 'chat_id', 'role', or None if not found."""

    @abstractmethod
    async def deactivate_messages_from(self, chat_id: UUID, from_message_id: UUID) -> None:
        """Mark from_message_id and all subsequent messages in chat_id as is_active=false."""

    @abstractmethod
    async def save_message(
        self,
        *,
        message_id: UUID,
        chat_id: UUID,
        role: str,
        parts: list[MessagePart],
    ) -> None:
        """Persist a message and its parts.

        ``parts`` is the UIMessage.parts array — the assembled final state
        after streaming completes, not raw SSE chunks. Each element is a
        discriminated-union dict keyed on ``type``.
        """

    # --- Feedback ---

    @abstractmethod
    async def get_message_owner(self, message_id: UUID) -> str | None:
        """Return the user_id of the chat that owns this message, or None if not found."""

    @abstractmethod
    async def upsert_feedback(self, message_id: UUID, value: int) -> None:
        """Set feedback for a message (0 = thumbs down, 1 = thumbs up)."""

    @abstractmethod
    async def delete_feedback(self, message_id: UUID) -> None:
        """Remove feedback for a message."""

    # --- Users ---

    @abstractmethod
    async def upsert_user(self, uid: str, email: Optional[str], name: Optional[str]) -> UserRecord:
        """Upsert a user row (keyed by Firebase UID), syncing email/name.

        Returns the full record, *including* soft-deleted ones — callers decide
        what to do with those. Reads before writing: this runs on every
        user-scoped request, so an unchanged user must not cost a write.
        """

    @abstractmethod
    async def record_legal_acceptance(
        self, uid: str, terms_version: str, privacy_version: str
    ) -> None:
        """Record that ``uid`` accepted these Terms and Privacy Policy versions.

        Deliberately *not* folded into ``upsert_user``. That method answers
        "who is this?" on every account-scoped request, including ones with
        nothing to do with consent, and an acceptance must only ever be written
        as the result of a deliberate act of acceptance. Recording it there
        would let the row claim a user agreed to these versions when all they
        did was open their chat history — which is exactly the fact this record
        exists to establish.

        Records the versions **as given**. The caller passes what the client
        posted, which may not be the version currently in force — see
        ``record_legal_acceptance`` in ``server/api.py`` for why refusing those
        would couple the record to deploy order.

        A no-op when the stored versions already match, so re-accepting from a
        second device preserves the original ``accepted_at`` — the record should
        say when they first agreed, not when they last cleared a browser.

        Also a no-op when the stored acceptance is *newer* than the one given, so
        a stale client cannot downgrade a record another device already advanced.
        Versions are zero-padded ISO dates, so they order as strings.
        """

    @abstractmethod
    async def soft_delete_user(self, uid: str) -> bool:
        """Flag the account deleted. Returns False if absent or already flagged.

        Idempotent, so a client retrying after a partial failure is safe.
        """

    @abstractmethod
    async def purge_deleted_users(
        self, older_than: datetime, *, dry_run: bool = True
    ) -> list["PurgeSummary"]:
        """Hard-delete accounts soft-deleted before ``older_than``, and their data.

        Returns what was purged, or on a dry run what would be. Chats cascade
        from ``app_user``; messages are removed explicitly because
        ``chat_message`` has no FK to ``chat``.

        Deliberately leaves ``token_usage`` alone: those rows are keyed by email,
        not by account, so deleting them could clear a *live* account's budget
        for the same address — and after the retention window they are older than
        every rate-limit window anyway, so they no longer affect anyone.
        """

    # --- Token usage ---

    @abstractmethod
    async def get_window_stats(self, quota_key: str, since: datetime) -> WindowStats:
        """Sum of tokens and oldest record timestamp for quota_key in the rolling window [since, now]."""

    @abstractmethod
    async def get_app_window_stats(self, since: datetime) -> WindowStats:
        """Sum of tokens and oldest record timestamp across ALL users in the rolling window [since, now]."""

    @abstractmethod
    async def check_and_reserve_user(
        self,
        quota_key: str,
        window_params: list[tuple[str, datetime]],  # (window_name, since)
        estimated: int,
        user_limits: dict[str, int],
    ) -> list[WindowStats]:
        """Within a per-quota-key pg_advisory_xact_lock: fetch window stats and check limits.

        If any limit would be exceeded (used + estimated > limit): commit (release lock) and
        return the stats WITHOUT inserting a reservation.
        If all limits are satisfied: INSERT estimated tokens as a reservation, commit, and
        return the stats (before the reservation).

        The caller is responsible for re-checking whether the returned stats indicate a
        violation and raising 429 accordingly.
        """

    @abstractmethod
    async def record_token_usage(self, quota_key: str, tokens: int) -> None:
        """Persist a token usage record for quota_key."""

    # --- Lifecycle ---

    @abstractmethod
    async def close(self) -> None:
        """Release held resources (connections, sessions, etc.)."""
