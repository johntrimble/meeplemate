import type { Query, QueryClient, QueryKey } from '@tanstack/react-query'
import type { PersistedClient, PersistRetryer } from '@tanstack/react-query-persist-client'

// ---------------------------------------------------------------------------
// Persisted-cache retention + tiered eviction
// ---------------------------------------------------------------------------
//
// We keep the cache warm for a long time so users don't hit a ~30s Cloud Run
// cold start on every visit. The persister stores the whole client as one blob
// in IndexedDB. To keep that blob bounded we (1) proactively cap what gets
// persisted (`makeShouldDehydrateQuery` - the primary bound) and (2) trim in a
// priority order if a write ever hits the storage quota (`tieredEvict` - a
// last-resort safety net).

/** Retention window for both `gcTime` (in-memory) and the persister `maxAge`. */
export const RETENTION_MS = 365 * 24 * 60 * 60 * 1000

/** How many chats' message histories to persist (the rest re-fetch on open). */
export const MAX_PERSISTED_MESSAGE_CHATS = 25

/**
 * Bump ONLY when the cached data shape changes incompatibly (not per deploy).
 * A mismatch makes the persister discard + delete the old blob on restore, so
 * clients with stale-schema caches rebuild fresh. See App.tsx `persistOptions`.
 */
export const CACHE_SCHEMA_VERSION = '1'

// Eviction priority, least-valuable first (lower = evicted first). Messages are
// large and cheap to re-fetch; the small lists are the cold-start payoff.
const EVICTION_TIER: Record<string, number> = {
  messages: 0,
  chats: 1,
  game: 2,
  games: 3,
  'recent-games': 4,
}
// Unknown/unclassified queries are kept longest (evicted last) to avoid
// dropping something we didn't anticipate before our known-valuable data.
const UNKNOWN_TIER = 99

export function evictionTier(queryKey: QueryKey): number {
  const head = queryKey[0]
  return typeof head === 'string' && head in EVICTION_TIER ? EVICTION_TIER[head] : UNKNOWN_TIER
}

/**
 * Decide whether a query is persisted. Non-success queries and message
 * histories beyond the {@link MAX_PERSISTED_MESSAGE_CHATS} most-recently-updated
 * chats are excluded; everything else (lists, sidebar chats, game detail) is
 * always kept. This is the steady-state bound on the persisted blob.
 */
export function makeShouldDehydrateQuery(queryClient: QueryClient): (query: Query) => boolean {
  return (query) => {
    if (query.state.status !== 'success') return false
    if (query.queryKey[0] !== 'messages') return true

    const keep = queryClient
      .getQueryCache()
      .findAll({ queryKey: ['messages'] })
      .filter((q) => q.state.status === 'success')
      .sort((a, b) => b.state.dataUpdatedAt - a.state.dataUpdatedAt)
      .slice(0, MAX_PERSISTED_MESSAGE_CHATS)
    return keep.some((q) => q.queryHash === query.queryHash)
  }
}

/**
 * Persister `retry`: on a failed (e.g. quota-exceeded) write, drop the single
 * lowest-priority query (oldest `dataUpdatedAt` within a tier) and let the
 * persister retry with a smaller blob. Returns `undefined` when nothing is left
 * to trim so the persister gives up gracefully.
 */
export const tieredEvict: PersistRetryer = ({ persistedClient }) => {
  const queries = persistedClient.clientState.queries
  if (queries.length === 0) return undefined

  let victim = 0
  for (let i = 1; i < queries.length; i++) {
    const tier = evictionTier(queries[i].queryKey)
    const victimTier = evictionTier(queries[victim].queryKey)
    if (
      tier < victimTier ||
      (tier === victimTier && queries[i].state.dataUpdatedAt < queries[victim].state.dataUpdatedAt)
    ) {
      victim = i
    }
  }

  const trimmed: PersistedClient = {
    ...persistedClient,
    clientState: {
      ...persistedClient.clientState,
      queries: queries.filter((_, i) => i !== victim),
    },
  }
  return trimmed
}
