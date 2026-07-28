import type { InfiniteData, QueryClient, QueryKey } from '@tanstack/react-query'
import { apiGameToGame, type Game, type GamesPage } from '@/data/games'

// ---------------------------------------------------------------------------
// Game detail from the already-cached game lists
// ---------------------------------------------------------------------------
//
// `/api/games/{id}` and `/api/games` return the exact same `GameInfo` per game
// (see `get_game` / `get_games` in meeplemate/server/api.py — both build the
// model from the same manifest fields), so a game already present in a cached
// list is a complete stand-in for the detail response.
//
// That matters on cold start: `seedGamesFromStatic` paints the list instantly
// from the static `/games.json` snapshot, but the chat page blocks on
// `useGame` until the scale-to-zero backend answers (~20-30s). Feeding the
// cached list entry in as `initialData` closes that gap.
//
// A miss (game absent from the snapshot — e.g. beyond the 100-item page the
// deploy pipeline snapshots, or ingested since the last deploy) is not an
// error: `useGame` just fetches as it always has.

export interface CachedGame {
  game: Game
  /**
   * `dataUpdatedAt` of the list entry the game came from. Passed through as
   * `initialDataUpdatedAt` so the seeded detail query inherits the list's real
   * freshness instead of claiming to be fresh as of now.
   */
  updatedAt: number
}

function successEntry(queryClient: QueryClient, queryKey: QueryKey) {
  const query = queryClient.getQueryCache().find({ queryKey, exact: true })
  if (!query || query.state.status !== 'success') return undefined
  return { data: query.state.data, updatedAt: query.state.dataUpdatedAt }
}

/**
 * Find a game in the cached game lists, newest cache entry first.
 *
 * Looks in `['games']` (the `useInfiniteQuery` catalog, also what the static
 * snapshot seeds) and `['recent-games']`. Returns `undefined` when the game is
 * in neither.
 */
export function findCachedGame(queryClient: QueryClient, gameId: string): CachedGame | undefined {
  const catalog = successEntry(queryClient, ['games'])
  if (catalog) {
    const { pages } = catalog.data as InfiniteData<GamesPage, string | undefined>
    for (const page of pages ?? []) {
      const hit = page.data?.find((g) => g.id === gameId)
      if (hit) return { game: apiGameToGame(hit), updatedAt: catalog.updatedAt }
    }
  }

  // `useRecentGames` stores already-mapped `Game`s, not raw `GameInfo`s.
  const recent = successEntry(queryClient, ['recent-games'])
  if (recent) {
    const hit = (recent.data as Game[]).find((g) => g.id === gameId)
    if (hit) return { game: hit, updatedAt: recent.updatedAt }
  }

  return undefined
}
