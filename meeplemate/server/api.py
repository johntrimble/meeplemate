import json
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from meeplemate.chatloop import ChatLoopServiceInput
from meeplemate.component_system import subsystem
from meeplemate.config import Config, System, create_app_system
from meeplemate.db.repository import ChatRepository
from meeplemate.server.deps import ApiDeps

_deps: ApiDeps


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _deps

    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(app_system, names=["api_deps"])

    async with system.astart() as started_system:
        _deps = started_system["api_deps"]
        yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class GameInfo(BaseModel):
    id: str
    name: str
    summary: str | None = None


class PageInfo(BaseModel):
    hasNextPage: bool
    startCursor: str | None = None
    endCursor: str | None = None


class GamesPage(BaseModel):
    pageInfo: PageInfo
    data: list[GameInfo]


@app.get("/api/games")
async def get_games(
    first: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> GamesPage:
    """List of all supported games with pagination."""
    games, has_next, start_cursor, end_cursor = await _deps.game_service.list_games(
        after=cursor, limit=first
    )
    return GamesPage(
        pageInfo=PageInfo(hasNextPage=has_next, startCursor=start_cursor, endCursor=end_cursor),
        data=[GameInfo(id=g["game_id"], name=g["name"], summary=g.get("summary")) for g in games],
    )


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

class ChatSummary(BaseModel):
    chat_id: str
    title: str


@app.get("/api/games/{game_id}/chats")
async def list_game_chats(game_id: str) -> list[ChatSummary]:
    """Return all chats for a game, newest first."""
    async with _deps.session_factory() as session:
        chats = await ChatRepository(session).get_chats_for_game(game_id=game_id)
    return [ChatSummary(**c) for c in chats]


class CreateChatResponse(BaseModel):
    chat_id: str


@app.post("/api/games/{game_id}/chats")
async def create_chat(game_id: str) -> CreateChatResponse:
    """
    Create a new chat for a game. Only called when the user actually sends
    their first message, so the chat record is created lazily.
    """
    manifest = await _deps.game_service.get_manifest(game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    async with _deps.session_factory() as session:
        chat_id = await ChatRepository(session).create_chat(game_id=game_id)

    return CreateChatResponse(chat_id=str(chat_id))


class MessagePart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    parts: list[MessagePart]


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str) -> list[ChatMessageOut]:
    """Return all messages for a chat in chronological order."""
    chat_uuid = UUID(chat_id)
    async with _deps.session_factory() as session:
        messages = await ChatRepository(session).get_messages(chat_id=chat_uuid)
    return [ChatMessageOut(**m) for m in messages]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class StreamChatRequest(BaseModel):
    message: str
    game_id: str


@app.post("/api/chats/{chat_id}/stream")
async def stream_chat(chat_id: str, request: StreamChatRequest):
    """
    Stream an assistant response using the Vercel AI UI message stream protocol.
    https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
    """
    manifest = await _deps.game_service.get_manifest(request.game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Game '{request.game_id}' not found")

    service_input: ChatLoopServiceInput = {
        "messages": [HumanMessage(content=request.message)],
        "manifest": manifest,
        "thread_id": chat_id,
    }

    chatloop_service = _deps.chatloop_service
    session_factory = _deps.session_factory
    chat_uuid = UUID(chat_id)

    async def sse_generator():
        msg_id = uuid4()
        text_id = str(uuid4())

        # Persist the user message before streaming starts.
        async with session_factory() as session:
            await ChatRepository(session).save_message(
                message_id=uuid4(),
                chat_id=chat_uuid,
                role="user",
                parts=[{"part_id": str(uuid4()), "part_type": "text", "payload": {"text": request.message}}],
            )

        yield f'data: {json.dumps({"type": "start", "messageId": str(msg_id)})}\n\n'
        yield f'data: {json.dumps({"type": "text-start", "id": text_id})}\n\n'

        accumulated: list[str] = []
        async for chunk in chatloop_service.astream_response(service_input):
            if chunk.content:
                delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                accumulated.append(delta)
                yield f'data: {json.dumps({"type": "text-delta", "id": text_id, "delta": delta})}\n\n'

        yield f'data: {json.dumps({"type": "text-end", "id": text_id})}\n\n'

        # Persist the complete assistant message.
        async with session_factory() as session:
            await ChatRepository(session).save_message(
                message_id=msg_id,
                chat_id=chat_uuid,
                role="assistant",
                parts=[{"part_id": text_id, "part_type": "text", "payload": {"text": "".join(accumulated)}}],
            )

        yield f'data: {json.dumps({"type": "finish"})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1"},
    )
