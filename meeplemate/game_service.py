from dataclasses import dataclass

from langchain_core.stores import BaseStore

from meeplemate.ingest.gamepackage import GamePackage


@dataclass
class GameService:
    data_store: BaseStore
    version_store: BaseStore

    async def get_current_version_for_game(self, game_id: str) -> str | None:
        results = await self.version_store.amget([game_id])
        assert len(results) == 1 and results[0] is not None, "No version found for game_id"
        version = results[0]
        return str(version)

    async def get_manifest(self, game_id: str) -> GamePackage | None:
        game_key = await self.get_current_version_for_game(game_id)
        manifest = (await self.data_store.amget([game_key]))[0]
        return manifest

    async def list_games(
        self, *, after: str | None = None, limit: int = 20
    ) -> tuple[list[GamePackage], bool, str | None, str | None]:
        """Return a page of games sorted by game_id.

        Returns (games, has_next_page, start_cursor, end_cursor).
        """
        all_ids: list[str] = sorted([key async for key in await self.version_store.ayield_keys()])

        start = 0
        if after is not None:
            for i, gid in enumerate(all_ids):
                if gid == after:
                    start = i + 1
                    break

        page_ids = all_ids[start : start + limit]
        has_next_page = len(all_ids) > start + limit

        manifests = []
        for game_id in page_ids:
            manifest = await self.get_manifest(game_id)
            if manifest is not None:
                manifests.append(manifest)

        start_cursor = page_ids[0] if page_ids else None
        end_cursor = page_ids[-1] if page_ids else None
        return manifests, has_next_page, start_cursor, end_cursor
