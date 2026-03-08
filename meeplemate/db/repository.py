from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from meeplemate.db.models import Chat, ChatMessage, ChatMessagePart


class ChatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_chats_for_game(self, *, game_id: str) -> list[dict]:
        """Return all chats for a game, newest first, with title from the first user message."""
        result = await self.session.execute(
            select(Chat)
            .where(Chat.game_id == game_id)
            .order_by(Chat.created_at.desc())
        )
        chats = result.scalars().all()

        out = []
        for chat in chats:
            first_msg_result = await self.session.execute(
                select(ChatMessage)
                .where(ChatMessage.chat_id == chat.chat_id, ChatMessage.role == "user")
                .order_by(ChatMessage.created_at)
                .limit(1)
                .options(selectinload(ChatMessage.parts))
            )
            first_msg = first_msg_result.scalar_one_or_none()

            title = "New Chat"
            if first_msg:
                title = "".join(
                    p.payload.get("text", "")
                    for p in sorted(first_msg.parts, key=lambda p: p.ordinal)
                    if p.part_type == "text"
                ) or "New Chat"

            out.append({"chat_id": str(chat.chat_id), "title": title})
        return out

    async def create_chat(self, *, game_id: str) -> UUID:
        chat = Chat(game_id=game_id)
        self.session.add(chat)
        await self.session.commit()
        await self.session.refresh(chat)
        return chat.chat_id

    async def get_messages(self, *, chat_id: UUID) -> list[dict]:
        """Return messages for a chat ordered by creation time, with parts ordered by ordinal."""
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at)
            .options(selectinload(ChatMessage.parts))
        )
        messages = result.scalars().all()

        out = []
        for msg in messages:
            parts = sorted(msg.parts, key=lambda p: p.ordinal)
            out.append({
                "id": str(msg.message_id),
                "role": msg.role,
                "parts": [
                    {"type": part.part_type, **part.payload}
                    for part in parts
                ],
            })
        return out

    async def save_message(
        self,
        *,
        message_id: UUID,
        chat_id: UUID,
        role: str,
        parts: list[dict],
    ) -> None:
        """Persist a chat message and its parts in one transaction.

        Each entry in ``parts`` must have:
          - part_id   (str)
          - part_type (str, e.g. "text")
          - payload   (dict, e.g. {"text": "..."})
        """
        msg = ChatMessage(message_id=message_id, chat_id=chat_id, role=role)
        self.session.add(msg)
        for ordinal, part in enumerate(parts):
            self.session.add(
                ChatMessagePart(
                    message_id=message_id,
                    part_id=part["part_id"],
                    part_type=part["part_type"],
                    ordinal=ordinal,
                    payload=part["payload"],
                )
            )
        await self.session.commit()
