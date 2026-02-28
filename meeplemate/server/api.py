import json
from uuid import uuid4
from contextlib import asynccontextmanager
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from meeplemate.chatloop import ChatLoopServiceInput
from meeplemate.component_system import subsystem
from meeplemate.config import Config, System, create_app_system
from meeplemate.server.deps import ApiDeps

_deps: ApiDeps

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _deps

    # Load configuration from YAML file (if MM_CONFIG_FILE is set), .env file, 
    # and environment variables
    # Priority: env vars > .env > YAML config > defaults
    settings: Config = Config()

    # Create the system
    app_system: System = create_app_system(settings)
    system = subsystem(app_system, names=["api_deps"])

    async with system.astart() as started_system:
        _deps = started_system["api_deps"]
        yield

app = FastAPI(lifespan=lifespan)

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
    """
    List of all supported games. Includes description of the game along with
    image URLs, game name, etc. Supports pagination.
    """
    games, has_next, start_cursor, end_cursor = await _deps.game_service.list_games(
        after=cursor, limit=first
    )
    return GamesPage(
        pageInfo=PageInfo(hasNextPage=has_next, startCursor=start_cursor, endCursor=end_cursor),
        data=[GameInfo(id=g["game_id"], name=g["name"], summary=g.get("summary")) for g in games],
    )

@app.get("/api/recently_used_games")
async def get_recently_used_games():
    """
    Returns a user's most recently used games. This can be used to show
    users which games they have interacted with recently.
    """
    pass

@app.get("/api/games/{game_id}/chats")
async def get_game_chats():
    """
    Returns a list of all a users chats for a given game, sorted by most
    recent message. This is used to populate the list of chats a user has
    for a given game. Supports pagination.
    """
    pass

# Maybe we don't need this endpoint if we just create a chat when the first message is posted to /api/chats/{chat_id}/stream. The frontend can keep track of which chats exist based on which chat_ids it has sent messages to.
# @app.post("/api/games/{game_id}/chats")
# async def create_game_chat():
#     """
#     Creates a new chat for a given game. The returned chat ID should then be
#     used with the /api/chats/{chat_id}/stream to post a message and receive
#     the response.
#     """
#     pass

@app.get("/api/chats/{chat_id}/messages")
async def chat_messages():
    """
    Returns all messages for a given chat, in chronological order. This is
    used to populate the chat history when a user opens a chat.
    """
    pass

class StreamChatRequest(BaseModel):
    message: str
    game_id: str


@app.post("/api/chats/{chat_id}/stream")
async def stream_chat(chat_id: str, request: StreamChatRequest):
    """
    Implementation of vercel streaming API (SSE):

    https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
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

    async def sse_generator():
        msg_id = str(uuid4())
        text_id = str(uuid4())
        yield f'data: {json.dumps({"type": "start", "messageId": msg_id})}\n\n'
        yield f'data: {json.dumps({"type": "text-start", "id": text_id})}\n\n'
        async for chunk in chatloop_service.astream_response(service_input):
            if chunk.content:
                delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                yield f'data: {json.dumps({"type": "text-delta", "id": text_id, "delta": delta})}\n\n'
        yield f'data: {json.dumps({"type": "text-end", "id": text_id})}\n\n'
        yield f'data: {json.dumps({"type": "finish"})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1"},
    )

