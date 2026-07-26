from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from meeplemate.chatloop import ChatLoopService
from meeplemate.db.datalayer import BaseDataLayer, UserRecord
from meeplemate.game_service import GameService
from meeplemate.server.auth import AuthUser, get_current_user
from meeplemate.tracing import NoopTraceSink

if TYPE_CHECKING:
    from meeplemate.server.rate_limit import RateLimiter
    from meeplemate.tracing import TraceSink


class CorsConfig(BaseModel):
    """CORS configuration for the API server."""
    allowed_origins: list[str] = Field(
        default_factory=list,
        description="List of allowed CORS origins (MM_CORS__ALLOWED_ORIGINS). "
                    "E.g. ['https://app.example.com', 'https://staging.example.com']",
    )


@dataclass
class ApiDeps:
    """Dependencies for the API."""
    chatloop_service: ChatLoopService
    game_service: GameService
    data_layer: BaseDataLayer
    rate_limiter: RateLimiter
    cors_config: CorsConfig
    trace_sink: TraceSink = field(default_factory=NoopTraceSink)


# ---------------------------------------------------------------------------
# Current-user dependencies
# ---------------------------------------------------------------------------
#
# The uid in the token is enough to identify the account, so these dependencies
# exist for one reason: to check whether it has been deleted. Firebase ID tokens
# stay valid for up to an hour after the underlying user is destroyed and are not
# revocation-checked per request, so without this read a just-deleted user could
# keep using their own data until the token expired.
#
# Handlers that merely need a valid token (the game catalog) must keep depending
# on `get_current_user` alone — see the note on `get_games` in api.py.


async def get_db_user_allow_deleted(
    request: Request,
    auth_user: AuthUser = Depends(get_current_user),
) -> UserRecord:
    """Resolve the account for the current token, *including* deleted ones.

    Only for the deletion endpoint itself, which must stay callable after the
    account is flagged so a client can retry a partial failure.
    """
    data_layer: BaseDataLayer = request.app.state.deps.data_layer
    return await data_layer.upsert_user(auth_user.uid, auth_user.email, auth_user.name)


async def get_db_user(
    db_user: UserRecord = Depends(get_db_user_allow_deleted),
) -> UserRecord:
    """Resolve the account for the current token, rejecting deleted accounts."""
    if db_user.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account deleted",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return db_user
