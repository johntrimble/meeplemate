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
 * Cap on the snapshot fetch. `CacheGate` awaits this seed before revealing
 * children, so an unresponsive CDN would otherwise hold the app on its splash
 * indefinitely (see the ordering note below). Bounded well
 * under the backend cold start we're avoiding: past this we just fall back to
 * `/api/games`, exactly as if no snapshot existed.
 */
const SEED_TIMEOUT_MS = 2_000

/**
 * Seed the `['games']` infinite-query cache from the static CDN snapshot, if
 * present. A no-op when:
 *   - the cache already holds a `['games']` entry (a persisted/warmer copy from
 *     a returning user is at least as fresh — never clobber it), or
 *   - the snapshot is missing/invalid (e.g. `vite dev`, where no file exists) —
 *     the app falls back to fetching `/api/games` exactly as before.
 *
 * Call this from `CacheGate`, after the persister has restored AND
 * `reconcileUserCache` has run — never earlier. Both of those write the cache:
 * restore would race the seed, and the reconcile *clears* it outright for a
 * new/changed owner, silently wiping a seed that landed first.
 *
 * `CacheGate` awaits this before revealing children. That ordering is
 * load-bearing: `useGame` reads the seeded catalog as `initialData`, which
 * react-query applies only when the `['game', id]` query is created. If a deep
 * link to `/chat/:gameId` mounted before the seed landed, the detail query would
 * be created empty and never pick the seed up. The early return above means a
 * returning (warm-cache) user pays nothing for this: they short-circuit before
 * the fetch.
 *
 * Never rejects — all failure modes fall back to fetching `/api/games`.
 */
export async function seedGamesFromStatic(
  queryClient: QueryClient,
  timeoutMs: number = SEED_TIMEOUT_MS
): Promise<void> {
  // Don't overwrite a persisted/warmer entry restored from IndexedDB.
  if (queryClient.getQueryData(['games'])) return

  try {
    const res = await fetch(STATIC_GAMES_URL, { signal: AbortSignal.timeout(timeoutMs) })
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
    // Missing file, offline, timed out, malformed JSON — silently fall back to
    // `/api/games`.
  }
}
