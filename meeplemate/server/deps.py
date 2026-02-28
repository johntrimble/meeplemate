from dataclasses import dataclass
from meeplemate.chatloop import ChatLoopService
from meeplemate.game_service import GameService

@dataclass
class ApiDeps:
    """Dependencies for the API."""
    chatloop_service: ChatLoopService
    game_service: GameService