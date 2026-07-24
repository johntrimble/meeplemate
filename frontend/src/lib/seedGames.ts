import type { InfiniteData, QueryClient } from '@tanstack/react-query'
import type { GamesPage } from '@/data/games'

// ---------------------------------------------------------------------------
// Cold-start seed for the game catalog
// ---------------------------------------------------------------------------
//
// A new user (or one whose persisted cache was evicted) has nothing in
// IndexedDB, so the Select-Game page would otherwise block on `/api/games`
// while the scale-to-zero Cloud Run backend cold-starts (~20s). To avoid that,
// we publish a static `games.json` snapshot to the CDN (same origin as the SPA,
// generated from `/api/games` at deploy time) and seed the `['games']` query
// cache from it on startup. The list paints instantly; `useGameList` then
// revalidates against the authoritative `/api/games` in the background (which
// also warms the backend for the first question).
//
// `games.json` stores a single `GamesPage` (the exact shape `/api/games`
// returns). We wrap it in the `InfiniteData` envelope that `useInfiniteQuery`
// (queryKey `['games']`) reads.

const STATIC_GAMES_URL = '/games.json'

/**
 * Seed the `['games']` infinite-query cache from the static CDN snapshot, if
 * present. A no-op when:
 *   - the cache already holds a `['games']` entry (a persisted/warmer copy from
 *     a returning user is at least as fresh — never clobber it), or
 *   - the snapshot is missing/invalid (e.g. `vite dev`, where no file exists) —
 *     the app falls back to fetching `/api/games` exactly as before.
 *
 * Call this AFTER the persister finishes restoring (see `App.tsx` `onSuccess`)
 * so the async restore can't overwrite the seed and vice-versa.
 */
export async function seedGamesFromStatic(queryClient: QueryClient): Promise<void> {
  // Don't overwrite a persisted/warmer entry restored from IndexedDB.
  if (queryClient.getQueryData(['games'])) return

  try {
    const res = await fetch(STATIC_GAMES_URL)
    if (!res.ok) return
    const page = (await res.json()) as GamesPage
    // Basic shape guard: a snapshot with no games isn't worth seeding.
    if (!page?.data?.length) return

    // Re-check after the await — the persister may have restored in the meantime.
    if (queryClient.getQueryData(['games'])) return

    const seeded: InfiniteData<GamesPage, string | undefined> = {
      pages: [page],
      pageParams: [undefined],
    }
    queryClient.setQueryData(['games'], seeded)
  } catch {
    // Missing file, offline, malformed JSON — silently fall back to `/api/games`.
  }
}
