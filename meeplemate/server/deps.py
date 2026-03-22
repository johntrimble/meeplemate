from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from meeplemate.chatloop import ChatLoopService
from meeplemate.db.datalayer import BaseDataLayer
from meeplemate.game_service import GameService

if TYPE_CHECKING:
    from meeplemate.server.rate_limit import RateLimiter


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
