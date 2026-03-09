from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from meeplemate.chatloop import ChatLoopService
from meeplemate.db.datalayer import BaseDataLayer
from meeplemate.game_service import GameService

if TYPE_CHECKING:
    from meeplemate.server.rate_limit import RateLimiter


@dataclass
class ApiDeps:
    """Dependencies for the API."""
    chatloop_service: ChatLoopService
    game_service: GameService
    data_layer: BaseDataLayer
    rate_limiter: RateLimiter
