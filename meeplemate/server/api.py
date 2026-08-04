import sys

# Block transformers from loading via langchain_core.language_models.base.
# That module does `from transformers import GPT2TokenizerFast` at import time
# (inside try/except ImportError) as a GPT-2 fallback tokenizer. We never use
# this fallback: ChatOpenAI uses tiktoken and we use LightweightTokenizer.
# sys.modules[name] = None causes any import of that name to raise
# ModuleNotFoundError, which langchain_core catches and sets _HAS_TRANSFORMERS=False.
# Safe for the API process only — ingest runs separately and is unaffected.
if "transformers" not in sys.modules:
    sys.modules["transformers"] = None  # type: ignore[assignment]

import json
import time
from datetime import date
from uuid import UUID, uuid4, uuid5
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict, Field, field_validator
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Literal
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

import structlog

# Configured here, between the import blocks, so that records emitted while
# `meeplemate.*` is still being imported are captured. Re-applied in create_app once
# Config has resolved values from .env / YAML that os.environ alone wouldn't see.
from meeplemate.logging_config import configure_logging

configure_logging()

from meeplemate.chatloop import ChatLoopServiceInput, cast
from meeplemate.component_system import subsystem
from meeplemate.config import Config, System, create_app_system
from meeplemate.db.datalayer import Pagination, TextMessagePart, UserRecord
from meeplemate.server.auth import AuthUser, delete_firebase_user, get_current_user
from meeplemate.server.deps import (
    ApiDeps,
    get_db_user,
    get_db_user_allow_deleted,
)
from meeplemate.server.legal import PRIVACY_VERSION, TERMS_VERSION
from meeplemate.server.logging_middleware import RequestContextMiddleware
from meeplemate.server.rate_limit import RateLimitState, TokenCountingCallback, check_rate_limit
from meeplemate.tracing import NoopTraceSink, PersistingTracer

logger = structlog.get_logger(__name__)


def get_deps(request: Request) -> ApiDeps:
    return request.app.state.deps


def _parse_chat_uuid(chat_id: str) -> UUID:
    """Parse a URL chat id into a UUID, returning 400 on malformed input.

    Chat ids are now client-supplied (the frontend mints one before the chat exists
    server-side), so junk ids are expected and must fail cleanly instead of raising a
    bare ValueError -> 500.
    """
    try:
        return UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat id")


# Namespace for deriving a stable message UUID from a client-supplied id that isn't
# already a UUID (the AI SDK generates short non-UUID ids like "tUph1jxlmpniKcBi").
_CLIENT_MESSAGE_NAMESPACE = UUID("6f9b9e6a-4c1e-4b6a-9c2d-000000000001")


def _coerce_message_uuid(value: str) -> UUID:
    """Map a client message id to a UUID for storage as the message PK.

    Pass real UUIDs through; deterministically derive one (uuid5) for the AI SDK's
    non-UUID ids. Determinism keeps `save_message` idempotent when a first stream is
    retried through a cold start with the same client id.
    """
    try:
        return UUID(value)
    except ValueError:
        return uuid5(_CLIENT_MESSAGE_NAMESPACE, value)


router = APIRouter()


def create_app(api_deps: ApiDeps | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    In production (api_deps=None), loads Config and starts the full component
    system. In tests, pass a pre-built ApiDeps to skip system startup entirely.
    """
    settings: Config | None = None
    if api_deps is None:
        settings = Config()
        settings.use_lightweight_tokenizer = True
        settings.use_approximate_tokenizer = True
        # Re-apply now that .env / YAML sources have been read.
        configure_logging(
            level=settings.log.level,
            fmt=settings.log.format,
            gcp_project_id=settings.log.gcp_project_id,
        )

    cors_origins = (
        api_deps.cors_config.allowed_origins
        if api_deps is not None
        else settings.cors.allowed_origins  # type: ignore[union-attr]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if api_deps is not None:
            app.state.deps = api_deps
            yield
        else:
            app_system: System = create_app_system(settings)  # type: ignore[arg-type]
            system = subsystem(app_system, names=["api_deps"])
            async with system.astart() as started_system:
                app.state.deps = started_system["api_deps"]
                logger.info("app.startup_complete")
                yield
                logger.info("app.shutdown")

    new_app = FastAPI(lifespan=lifespan)
    # Added after CORSMiddleware so it ends up outermost: Starlette applies
    # middleware in reverse registration order, and we want preflight and
    # CORS-rejected requests logged too.
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    new_app.add_middleware(RequestContextMiddleware)
    new_app.include_router(router)
    return new_app


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class GameInfo(BaseModel):
    id: str
    name: str
    emoji: str | None = None
    background_color: str | None = None
    example_questions: list[str] | None = None


class PageInfo(BaseModel):
    hasNextPage: bool
    startCursor: str | None = None
    endCursor: str | None = None


class GamesPage(BaseModel):
    pageInfo: PageInfo
    data: list[GameInfo]


@router.get("/api/games")
async def get_games(
    first: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> GamesPage:
    """List of all supported games with pagination.

    Auth is required so anonymous traffic can't reach the catalog DB query. The
    deploy pipeline that snapshots this endpoint into the static CDN `games.json`
    authenticates with a short-lived Firebase ID token minted from the service
    account (see the frontend deploy workflow) — do NOT drop this dependency to
    make that curl easier.

    Equally, do NOT "upgrade" this to `get_db_user`. That token is minted for a
    synthetic `deploy-bot` uid via signInWithCustomToken: it carries no email and
    has no `app_user` row, so resolving an account here would mint a junk row on
    every deploy. This endpoint is not user-scoped — the catalog is the same for
    everyone and `user` is deliberately unused — so a valid token is all it needs.
    """
    games, has_next, start_cursor, end_cursor = await deps.game_service.list_games(
        after=cursor, limit=first
    )
    questions = await deps.game_service.get_example_questions([g["game_id"] for g in games])
    return GamesPage(
        pageInfo=PageInfo(hasNextPage=has_next, startCursor=start_cursor, endCursor=end_cursor),
        data=[GameInfo(id=g["game_id"], name=g["name"], emoji=g.get("emoji"), background_color=g.get("background_color"), example_questions=q) for g, q in zip(games, questions)],
    )


@router.get("/api/games/{game_id}")
async def get_game(
    game_id: str,
    user: AuthUser = Depends(get_current_user),
    deps: ApiDeps = Depends(get_deps),
) -> GameInfo:
    """Get a single game by ID. Authed — same rationale as `get_games`."""
    manifest = await deps.game_service.get_manifest(game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Game not found")
    questions = (await deps.game_service.get_example_questions([game_id]))[0]
    return GameInfo(id=manifest["game_id"], name=manifest["name"], emoji=manifest.get("emoji"), background_color=manifest.get("background_color"), example_questions=questions)


@router.get("/api/recent-games")
async def get_recent_games(
    first: int = Query(default=5, ge=1, le=20),
    cursor: str | None = Query(default=None),
    db_user: UserRecord = Depends(get_db_user),
    deps: ApiDeps = Depends(get_deps),
) -> GamesPage:
    """Games the current user has most recently chatted in, newest-first."""
    recent = await deps.data_layer.list_recent_game_ids(
        user_id=db_user.uid,
        pagination=Pagination(first=first, cursor=cursor),
    )
    games = await deps.game_service.get_games_by_ids(recent.data)
    questions = await deps.game_service.get_example_questions([g["game_id"] for g in games])
    return GamesPage(
        pageInfo=PageInfo(hasNextPage=recent.pageInfo.hasNextPage, startCursor=recent.pageInfo.startCursor, endCursor=recent.pageInfo.endCursor),
        data=[GameInfo(id=g["game_id"], name=g["name"], emoji=g.get("emoji"), background_color=g.get("background_color"), example_questions=q) for g, q in zip(games, questions)],
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


@router.get("/api/games/{game_id}/chats")
async def list_game_chats(
    game_id: str,
    first: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    db_user: UserRecord = Depends(get_db_user),
    deps: ApiDeps = Depends(get_deps),
) -> ChatsPage:
    """Return chats for a game belonging to the authenticated user, newest first."""
    from meeplemate.db.datalayer import Pagination
    result = await deps.data_layer.list_chats(game_id, db_user.uid, Pagination(first=first, cursor=cursor))
    return ChatsPage(
        pageInfo=PageInfo(
            hasNextPage=result.pageInfo.hasNextPage,
            startCursor=result.pageInfo.startCursor,
            endCursor=result.pageInfo.endCursor,
        ),
        data=[ChatSummary(**c) for c in result.data],
    )


class MessagePartOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str


class ChatMessageOut(BaseModel):
    id: str
    role: str
    parts: list[MessagePartOut]
    feedback: int | None = None


@router.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    db_user: UserRecord = Depends(get_db_user),
    deps: ApiDeps = Depends(get_deps),
) -> list[ChatMessageOut]:
    """Return all messages for a chat in chronological order."""
    chat_uuid = _parse_chat_uuid(chat_id)
    chat = await deps.data_layer.get_chat(chat_uuid)
    if chat is None or chat["user_id"] != db_user.uid:
        raise HTTPException(status_code=404, detail="Chat not found")
    messages = await deps.data_layer.get_messages(chat_uuid)
    return [ChatMessageOut.model_validate(m) for m in messages]


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    value: Literal[0, 1]


@router.put("/api/messages/{message_id}/feedback", status_code=204)
async def set_message_feedback(
    message_id: str,
    body: FeedbackRequest,
    db_user: UserRecord = Depends(get_db_user),
    deps: ApiDeps = Depends(get_deps),
) -> Response:
    """Upsert thumbs-up (1) or thumbs-down (0) feedback for a message."""
    msg_uuid = UUID(message_id)
    owner = await deps.data_layer.get_message_owner(msg_uuid)
    if owner != db_user.uid:
        raise HTTPException(status_code=404, detail="Message not found")
    await deps.data_layer.upsert_feedback(msg_uuid, body.value)
    return Response(status_code=204)


@router.delete("/api/messages/{message_id}/feedback", status_code=204)
async def delete_message_feedback(
    message_id: str,
    db_user: UserRecord = Depends(get_db_user),
    deps: ApiDeps = Depends(get_deps),
) -> Response:
    """Remove feedback for a message."""
    msg_uuid = UUID(message_id)
    owner = await deps.data_layer.get_message_owner(msg_uuid)
    if owner != db_user.uid:
        raise HTTPException(status_code=404, detail="Message not found")
    await deps.data_layer.delete_feedback(msg_uuid)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

#: A document version is its effective date, verbatim, as ``YYYY-MM-DD``.
_VERSION_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

#: Validated shape, never membership. See `record_legal_acceptance`.
LegalVersion = Annotated[str, Field(pattern=_VERSION_PATTERN)]


class LegalAcceptanceRequest(BaseModel):
    terms_version: LegalVersion
    privacy_version: LegalVersion

    @field_validator("terms_version", "privacy_version")
    @classmethod
    def _must_be_a_real_date(cls, value: str) -> str:
        # The pattern alone admits "2026-13-45". Parsing is what makes the stored
        # version comparable: `has_accepted_current` and the repository's
        # downgrade guard both order these as strings, which is only sound while
        # every one of them is a real, zero-padded ISO date.
        try:
            date.fromisoformat(value)
        except ValueError:
            raise ValueError("must be a calendar date in YYYY-MM-DD form")
        return value


@router.post("/api/account/legal-acceptance", status_code=204)
async def record_legal_acceptance(
    body: LegalAcceptanceRequest,
    db_user: UserRecord = Depends(get_db_user),
    deps: ApiDeps = Depends(get_deps),
) -> Response:
    """Record which Terms and Privacy Policy versions the user accepted.

    Stores **what the client posted**, verbatim, rather than this process's
    constants. That is the honest record: the client is the only party that
    knows what it put in front of the user, and "user accepted 2026-01-01 at
    time T" is a true statement even when today's version is 2026-07-28.

    The only rejection is a **malformed** version string (see
    `LegalAcceptanceRequest`). Nothing is refused for being stale, or for being
    a version this process has never heard of, because either rule silently
    couples the record to deploy order: ship the frontend first and every client
    posts a version the backend does not yet know, so a perfectly valid
    acceptance would be thrown away. The property worth protecting is that **no
    deploy order can cost a user their acceptance record.**

    A version this process does not consider current is therefore logged, not
    refused — that is the signal a version allow-list would have provided, minus
    the data loss. A stale client self-heals without help from us: it records
    what it showed, and on its next load the newer bundle's constants no longer
    match its localStorage, so the frontend re-prompts.

    Idempotent — re-accepting the same versions is a 204 that leaves the stored
    `accepted_at` alone. The repository also refuses to overwrite a newer stored
    acceptance with an older one, so a stale tab cannot undo a fresh one.
    """
    if (
        body.terms_version != TERMS_VERSION
        or body.privacy_version != PRIVACY_VERSION
    ):
        logger.warning(
            "legal_acceptance_version_divergence",
            posted_terms_version=body.terms_version,
            posted_privacy_version=body.privacy_version,
            current_terms_version=TERMS_VERSION,
            current_privacy_version=PRIVACY_VERSION,
        )

    await deps.data_layer.record_legal_acceptance(
        db_user.uid, body.terms_version, body.privacy_version
    )
    return Response(status_code=204)


@router.delete("/api/account", status_code=204)
async def delete_account(
    db_user: UserRecord = Depends(get_db_user_allow_deleted),
    deps: ApiDeps = Depends(get_deps),
) -> Response:
    """Delete the authenticated user's account.

    A soft delete: the row is flagged so the user's existing ID token stops
    working at once, and `mm-admin purge-deleted-accounts` removes the account
    and its chats once the retention window has passed.

    Signing up again later creates a *new* account with no history — the uid is
    the account key and Firebase mints a fresh one. Their token budget does
    follow them, because `token_usage` is keyed on email precisely so that
    deleting an account can't be used to clear a rate limit.

    The database is flagged *before* Firebase is touched, so a failure partway
    leaves the account locked out rather than half-live. That also makes retries
    safe — hence `get_db_user_allow_deleted`, since the ordinary dependency
    rejects an already-flagged account and would block the client's second
    attempt. Deleting an already-deleted account is a no-op 204.
    """
    await deps.data_layer.soft_delete_user(db_user.uid)

    await delete_firebase_user(db_user.uid)

    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class StreamChatRequest(BaseModel):
    message: str
    game_id: str
    retry_message_id: str | None = None
    # Client-supplied id for the user message. Lets a retried first stream (cold
    # start) dedupe instead of double-saving the message. Optional for back-compat.
    message_id: str | None = None


@router.post("/api/chats/{chat_id}/stream")
async def stream_chat(
    chat_id: str,
    request: StreamChatRequest,
    db_user: UserRecord = Depends(get_db_user),
    rate_state: RateLimitState = Depends(check_rate_limit),
    deps: ApiDeps = Depends(get_deps),
):
    """
    Stream an assistant response using the Vercel AI UI message stream protocol.
    https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
    """
    # Bound for the rest of the request so every downstream chatloop / qa_graph /
    # search line is attributable to this chat.
    structlog.contextvars.bind_contextvars(
        chat_id=chat_id,
        game_id=request.game_id,
        is_retry=request.retry_message_id is not None,
    )

    manifest = await deps.game_service.get_manifest(request.game_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Game '{request.game_id}' not found")

    chat_uuid = _parse_chat_uuid(chat_id)
    # Create-on-first-message: the client mints the chat id and navigates optimistically,
    # so the chat may not exist yet. ensure_chat creates it atomically if absent and never
    # overwrites an existing row, so a collision/tamper can't hijack another user's chat —
    # we still verify ownership + game below before writing anything.
    chat = await deps.data_layer.ensure_chat(
        chat_id=chat_uuid, game_id=request.game_id, user_id=db_user.uid
    )
    if chat["user_id"] != db_user.uid or chat["game_id"] != request.game_id:
        raise HTTPException(status_code=404, detail="Chat not found")

    chatloop_service = deps.chatloop_service
    data_layer = deps.data_layer

    if request.retry_message_id is not None:
        # Regenerate path: validate the target message, then deactivate it and everything after.
        msg_uuid = UUID(request.retry_message_id)
        owner = await data_layer.get_message_owner(msg_uuid)
        if owner is None or owner != db_user.uid:
            raise HTTPException(status_code=404, detail="Message not found")
        msg_info = await data_layer.get_message(msg_uuid)
        if msg_info is None or msg_info["role"] != "assistant":
            raise HTTPException(status_code=400, detail="Can only retry assistant messages")
        await data_layer.deactivate_messages_from(chat_uuid, msg_uuid)
    else:
        # Normal path: persist the new user message. Use the client-supplied id when
        # present so a retried first stream dedupes (save_message is idempotent on it).
        user_message_id = _coerce_message_uuid(request.message_id) if request.message_id else uuid4()
        await data_layer.save_message(
            message_id=user_message_id,
            chat_id=chat_uuid,
            role="user",
            parts=[{"type": "text", "text": request.message}],
        )

    # Get all active messages for this chat (trimmed history for regenerate, full for normal).
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

        # This generator body runs *after* stream_chat has returned its response, so
        # an exception raised in here can no longer become an HTTP error — it just
        # truncates the SSE stream. Logging it is the only way it is ever visible.
        structlog.contextvars.bind_contextvars(message_id=str(msg_id))
        started = time.perf_counter()
        logger.info("chat.stream_started", message_count=len(langchain_messages))

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

        # Persist the agent's run trace, keyed by chat/message id. The tracer's
        # _persist_run fires when the root run completes — inside this generator,
        # before the response closes — so the upload lands within Cloud Run's
        # request CPU window. NoopTraceSink (the default) makes this a no-op.
        callbacks: list = [token_callback]
        if not isinstance(deps.trace_sink, NoopTraceSink):
            callbacks.append(
                PersistingTracer(
                    deps.trace_sink,
                    chat_id=chat_id,
                    message_id=str(msg_id),
                )
            )

        from langchain_core.runnables import RunnableConfig
        try:
            async for _, _, event in chatloop_service.astream(
                service_input,
                subgraphs=True,
                stream_mode=["custom"],
                config=RunnableConfig(callbacks=callbacks),
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
        except Exception:
            logger.exception(
                "chat.stream_failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                answer_chars=len(final_answer),
            )
            raise

        try:
            estimated = deps.rate_limiter.config.estimated_tokens_per_request
            await data_layer.record_token_usage(db_user.quota_key, token_callback.tokens - estimated)
        except Exception:
            logger.warning("chat.record_token_usage_failed", exc_info=True)

        logger.info(
            "chat.stream_finished",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            answer_chars=len(final_answer),
            tokens=token_callback.tokens,
        )

        yield f'data: {json.dumps({"type": "finish"})}\n\n'
        yield 'data: [DONE]\n\n'

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"x-vercel-ai-ui-message-stream": "v1", **rate_state.headers()},
    )
