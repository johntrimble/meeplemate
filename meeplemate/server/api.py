import json
from uuid import UUID, uuid4
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from typing import Literal
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from meeplemate.chatloop import ChatLoopServiceInput, cast
from meeplemate.component_system import subsystem
from meeplemate.config import Config, System, create_app_system
from meeplemate.db.datalayer import Pagination, TextMessagePart
from meeplemate.server.auth import AuthUser, get_current_user
from meeplemate.server.deps import ApiDeps
from meeplemate.server.rate_limit import RateLimitState, TokenCountingCallback, check_rate_limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(app_system, names=["api_deps"])

    async with system.astart() as started_system:
        app.state.deps = started_system["api_deps"]
        yield


def get_deps(request: Request) -> ApiDeps:
    return request.app.state.deps


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class GameInfo(BaseModel):
    id: str
    name: str
    summary: str | None = None
    emoji: str | None = None
    background_color: str | None = None


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
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> GamesPage:
    """List of all supported games with pagination."""
    games, has_next, start_cursor, end_cursor = await deps.game_service.list_games(
        after=cursor, limit=first
    )
    return GamesPage(
        pageInfo=PageInfo(hasNextPage=has_next, startCursor=start_cursor, endCursor=end_cursor),
        data=[GameInfo(id=g["game_id"], name=g["name"], summary=g.get("summary"), emoji=g.get("emoji"), background_color=g.get("background_color")) for g in games],
    )


@app.get("/api/games/{game_id}")
async def get_game(
    game_id: str,
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> GameInfo:
    """Get a single game by ID."""
    manifest = await deps.game_service.get_manifest(game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return GameInfo(id=manifest["game_id"], name=manifest["name"], summary=manifest.get("summary"), emoji=manifest.get("emoji"), background_color=manifest.get("background_color"))


@app.get("/api/recent-games")
async def get_recent_games(
    first: int = Query(default=5, ge=1, le=20),
    cursor: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> GamesPage:
    """Games the current user has most recently chatted in, newest-first."""
    recent = await deps.data_layer.list_recent_game_ids(
        user_id=user.uid,
        pagination=Pagination(first=first, cursor=cursor),
    )
    games = await deps.game_service.get_games_by_ids(recent.data)
    return GamesPage(
        pageInfo=PageInfo(hasNextPage=recent.pageInfo.hasNextPage, startCursor=recent.pageInfo.startCursor, endCursor=recent.pageInfo.endCursor),
        data=[GameInfo(id=g["game_id"], name=g["name"], summary=g.get("summary"), emoji=g.get("emoji"), background_color=g.get("background_color")) for g in games],
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
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> ChatsPage:
    """Return chats for a game belonging to the authenticated user, newest first."""
    from meeplemate.db.datalayer import Pagination
    result = await deps.data_layer.list_chats(game_id, user.uid, Pagination(first=first, cursor=cursor))
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
async def create_chat(
    game_id: str,
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> CreateChatResponse:
    """
    Create a new chat for a game scoped to the authenticated user.
    Only called when the user actually sends their first message.
    """
    manifest = await deps.game_service.get_manifest(game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    chat_id = await deps.data_layer.create_chat(game_id, user.uid)
    return CreateChatResponse(chat_id=str(chat_id))


class MessagePartOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    parts: list[MessagePartOut]
    feedback: int | None = None


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> list[ChatMessageOut]:
    """Return all messages for a chat in chronological order."""
    chat_uuid = UUID(chat_id)
    chat = await deps.data_layer.get_chat(chat_uuid)
    if chat is None or chat["user_id"] != user.uid:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = await deps.data_layer.get_messages(chat_uuid)
    return [ChatMessageOut.model_validate(m) for m in messages]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    value: Literal[0, 1]


@app.put("/api/messages/{message_id}/feedback", status_code=204)
async def set_message_feedback(
    message_id: str,
    body: FeedbackRequest,
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> Response:
    """Upsert thumbs-up (1) or thumbs-down (0) feedback for a message."""
    msg_uuid = UUID(message_id)
    owner = await deps.data_layer.get_message_owner(msg_uuid)
    if owner != user.uid:
        raise HTTPException(status_code=404, detail="Message not found")
    await deps.data_layer.upsert_feedback(msg_uuid, body.value)
    return Response(status_code=204)


@app.delete("/api/messages/{message_id}/feedback", status_code=204)
async def delete_message_feedback(
    message_id: str,
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> Response:
    """Remove feedback for a message."""
    msg_uuid = UUID(message_id)
    owner = await deps.data_layer.get_message_owner(msg_uuid)
    if owner != user.uid:
        raise HTTPException(status_code=404, detail="Message not found")
    await deps.data_layer.delete_feedback(msg_uuid)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class StreamChatRequest(BaseModel):
    message: str
    game_id: str


@app.post("/api/chats/{chat_id}/stream")
async def stream_chat(
    chat_id: str,
    request: StreamChatRequest,
    user: AuthUser = Depends(get_current_user),
    rate_state: RateLimitState = Depends(check_rate_limit),
    deps: ApiDeps = Depends(get_deps),
):
    """
    Stream an assistant response using the Vercel AI UI message stream protocol.
    https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
    """
    manifest = await deps.game_service.get_manifest(request.game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Game '{request.game_id}' not found")

    chat_uuid = UUID(chat_id)
    chat = await deps.data_layer.get_chat(chat_uuid)
    if chat is None or chat["user_id"] != user.uid:
        raise HTTPException(status_code=404, detail="Chat not found")

    chatloop_service = deps.chatloop_service
    data_layer = deps.data_layer

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
        text_parts = [p for p in message["parts"] if p["type"] == "text"]
        content = cast(TextMessagePart, text_parts[-1])["text"] if text_parts else ""
        role = message["role"]
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        else:
            langchain_messages.append(AIMessage(content=content))

    service_input: ChatLoopServiceInput = {
        "messages": langchain_messages,
        "manifest": manifest,
        "thread_id": chat_id,
    }

    token_callback = TokenCountingCallback(
        estimated=deps.rate_limiter.config.estimated_tokens_per_request,
        output_token_multiplier=deps.rate_limiter.config.output_token_multiplier,
    )

    async def sse_generator():
        # https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
        msg_id = uuid4()
        text_id = str(uuid4())

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
            yield f'data: {json.dumps({"type": "start-step"})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-start", "id": reasoning_uuid})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-delta", "id": reasoning_uuid, "delta": description})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_uuid})}\n\n'

        final_answer = ""

        from langchain_core.runnables import RunnableConfig
        async for _, _, event in chatloop_service.astream(
            service_input,
            subgraphs=True,
            stream_mode=["custom"],
            config=RunnableConfig(callbacks=[token_callback]),
        ):
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

        await data_layer.save_message(
            message_id=msg_id,
            chat_id=chat_uuid,
            role="assistant",
            parts=[{"type": "text", "id": text_id, "text": final_answer}],
        )

        try:
            estimated = deps.rate_limiter.config.estimated_tokens_per_request
            await data_layer.record_token_usage(user.uid, token_callback.tokens - estimated)
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Failed to record token usage", exc_info=True)

        yield f'data: {json.dumps({"type": "finish"})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1", **rate_state.headers()},
    )


# ---------------------------------------------------------------------------
# Retry (regenerate) an assistant message
# ---------------------------------------------------------------------------


@app.post("/api/messages/{message_id}/retry")
async def retry_message(
    message_id: str,
    user: AuthUser = Depends(get_current_user),
    rate_state: RateLimitState = Depends(check_rate_limit),
    deps: ApiDeps = Depends(get_deps),
):
    """
    Regenerate an assistant response.

    Deactivates the specified assistant message and all subsequent messages in
    the same chat, then re-runs the QA pipeline on the remaining conversation
    history and streams back a new response using the Vercel AI UI message
    stream protocol.
    """
    msg_uuid = UUID(message_id)

    owner = await deps.data_layer.get_message_owner(msg_uuid)
    if owner is None or owner != user.uid:
        raise HTTPException(status_code=404, detail="Message not found")

    msg_info = await deps.data_layer.get_message(msg_uuid)
    if msg_info is None or msg_info["role"] != "assistant":
        raise HTTPException(status_code=400, detail="Can only retry assistant messages")

    chat_uuid = UUID(msg_info["chat_id"])
    chat = await deps.data_layer.get_chat(chat_uuid)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    manifest = await deps.game_service.get_manifest(chat["game_id"])
    if manifest is None:
        raise HTTPException(status_code=404, detail="Game not found")

    # Soft-delete the target message and all subsequent messages
    await deps.data_layer.deactivate_messages_from(chat_uuid, msg_uuid)

    # Fetch trimmed history (only active messages remain)
    messages = await deps.data_layer.get_messages(chat_uuid)

    langchain_messages: list[AnyMessage] = []
    for message in messages:
        text_parts = [p for p in message["parts"] if p["type"] == "text"]
        content = cast(TextMessagePart, text_parts[-1])["text"] if text_parts else ""
        if message["role"] == "user":
            langchain_messages.append(HumanMessage(content=content))
        else:
            langchain_messages.append(AIMessage(content=content))

    service_input: ChatLoopServiceInput = {
        "messages": langchain_messages,
        "manifest": manifest,
        "thread_id": str(chat_uuid),
    }

    token_callback = TokenCountingCallback(
        estimated=deps.rate_limiter.config.estimated_tokens_per_request,
        output_token_multiplier=deps.rate_limiter.config.output_token_multiplier,
    )

    chatloop_service = deps.chatloop_service
    data_layer = deps.data_layer

    async def sse_generator():
        msg_id = uuid4()
        text_id = str(uuid4())

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
            yield f'data: {json.dumps({"type": "start-step"})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-start", "id": reasoning_uuid})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-delta", "id": reasoning_uuid, "delta": description})}\n\n'
            yield f'data: {json.dumps({"type": "reasoning-end", "id": reasoning_uuid})}\n\n'

        final_answer = ""

        from langchain_core.runnables import RunnableConfig
        async for _, _, event in chatloop_service.astream(
            service_input,
            subgraphs=True,
            stream_mode=["custom"],
            config=RunnableConfig(callbacks=[token_callback]),
        ):
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

        await data_layer.save_message(
            message_id=msg_id,
            chat_id=chat_uuid,
            role="assistant",
            parts=[{"type": "text", "id": text_id, "text": final_answer}],
        )

        try:
            estimated = deps.rate_limiter.config.estimated_tokens_per_request
            await data_layer.record_token_usage(user.uid, token_callback.tokens - estimated)
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Failed to record token usage", exc_info=True)

        yield f'data: {json.dumps({"type": "finish"})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1", **rate_state.headers()},
    )
