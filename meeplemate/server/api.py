import json
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from langchain import messages
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from meeplemate.chatloop import ChatLoopServiceInput, cast
from meeplemate.component_system import subsystem
from meeplemate.config import Config, System, create_app_system
from meeplemate.db.datalayer import TextMessagePart
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


class ChatsPage(BaseModel):
    pageInfo: PageInfo
    data: list[ChatSummary]


@app.get("/api/games/{game_id}/chats")
async def list_game_chats(
    game_id: str,
    first: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> ChatsPage:
    """Return chats for a game, newest first, with pagination."""
    from meeplemate.db.datalayer import Pagination
    result = await _deps.data_layer.list_chats(game_id, Pagination(first=first, cursor=cursor))
    return ChatsPage(
        pageInfo=PageInfo(
            hasNextPage=result.pageInfo.hasNextPage,
            startCursor=result.pageInfo.startCursor,
            endCursor=result.pageInfo.endCursor,
        ),
        data=[ChatSummary(**c) for c in result.data],
    )


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

    chat_id = await _deps.data_layer.create_chat(game_id)
    return CreateChatResponse(chat_id=str(chat_id))


class MessagePartOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    parts: list[MessagePartOut]


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str) -> list[ChatMessageOut]:
    """Return all messages for a chat in chronological order."""
    chat_uuid = UUID(chat_id)
    messages = await _deps.data_layer.get_messages(chat_uuid)
    return [ChatMessageOut.model_validate(m) for m in messages]


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

    chatloop_service = _deps.chatloop_service
    data_layer = _deps.data_layer
    chat_uuid = UUID(chat_id)


    # Persist the user message
    await data_layer.save_message(
        message_id=uuid4(),
        chat_id=chat_uuid,
        role="user",
        parts=[{"type": "text", "text": request.message}],
    )

    # Get all messages for this chat so far
    messages = await data_layer.get_messages(chat_uuid)

    # Convert messages to LangChain format for the service input
    langchain_messages: list[AnyMessage] = []
    for message in messages:
        # Get last text part, this should be the message content
        text_parts = [p for p in message["parts"] if p["type"] == "text"]
        content = cast(TextMessagePart, text_parts[-1])["text"] if text_parts else ""
        role = message["role"]
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        else:
            # For simplicity, we treat all non-user messages as assistant messages.
            # In a more complex implementation, we might have system messages or other roles.
            langchain_messages.append(AIMessage(content=content))


    service_input: ChatLoopServiceInput = {
        "messages": langchain_messages,
        "manifest": manifest,
        "thread_id": chat_id,
    }

    async def sse_generator():
        # https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
        msg_id = uuid4()
        text_id = str(uuid4())

        # Start the message
        yield f'data: {json.dumps({"type": "start", "messageId": str(msg_id)})}\n\n'

        step_is_open: bool = False

        def maybe_close_step():
            nonlocal step_is_open
            if step_is_open:
                step_is_open = False
                yield f'data: {json.dumps({"type": "finish-step"})}\n\n'

        def open_step(description: str):
            nonlocal step_is_open

            for e in maybe_close_step():
                yield e

            step_is_open = True
            reasoning_uuid = str(uuid4())
            description = f"{description}"
            yield f'data: {json.dumps({"type": "start-step"})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-start", "id": reasoning_uuid})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-delta", "id": reasoning_uuid, "delta": description})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_uuid})}\n\n'

        final_answer = ""

        async for _, _, event in chatloop_service.astream(service_input, subgraphs=True, stream_mode=["custom"]):
            match event:
                case {"type": "mm_step", "description": description}:
                    for e in open_step(description):
                        yield e
                case {"type": "mm_refined_user_query", "refined_query": refined_query}:
                    description = f"Refined user query: {refined_query}"
                    reasoning_uuid = str(uuid4())
                    yield f'data: {json.dumps({"type": "reasoning-start", "id": reasoning_uuid})}\n\n'
                    yield f'data: {json.dumps({"type": "reasoning-delta", "id": reasoning_uuid, "delta": description})}\n\n'
                    yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_uuid})}\n\n'
                case {"type": "mm_user_query_answered", "answer": answer}:
                    reasoning_uuid = str(uuid4())
                    final_answer = answer
                    yield f'data: {json.dumps({"type": "text-start", "id": reasoning_uuid})}\n\n'
                    yield f'data: {json.dumps({"type": "text-delta", "id": reasoning_uuid, "delta": answer})}\n\n'
                    yield f'data: {json.dumps({"type": "text-end", "id": reasoning_uuid})}\n\n'
                    for e in maybe_close_step():
                        yield e

        # Persist the complete assistant message.
        await data_layer.save_message(
            message_id=msg_id,
            chat_id=chat_uuid,
            role="assistant",
            parts=[{"type": "text", "id": text_id, "text": final_answer}],
        )

        yield f'data: {json.dumps({"type": "finish"})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1"},
    )
