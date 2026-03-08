from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, NotRequired, Optional, TypedDict, TypeVar, Union
from uuid import UUID

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


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


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
    async def create_chat(self, game_id: str, user_id: str) -> UUID:
        """Create a new chat session for user_id and return its UUID."""

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
        """All messages for a chat, ordered chronologically."""

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

    # --- Lifecycle ---

    @abstractmethod
    async def close(self) -> None:
        """Release held resources (connections, sessions, etc.)."""
