import { describe, expect, it } from 'vitest'
import { QueryClient } from '@tanstack/react-query'
import type { PersistedClient } from '@tanstack/react-query-persist-client'
import type { DehydratedState, QueryKey } from '@tanstack/react-query'
import {
  MAX_PERSISTED_MESSAGE_CHATS,
  evictionTier,
  makeShouldDehydrateQuery,
  tieredEvict,
} from './cachePersist'

// --- helpers ---------------------------------------------------------------

let seq = 0
function dq(queryKey: unknown[], dataUpdatedAt: number) {
  return {
    queryKey,
    queryHash: JSON.stringify(queryKey) + '#' + seq++,
    state: { data: 'x', dataUpdatedAt, status: 'success' },
  }
}

function client(queries: ReturnType<typeof dq>[]): PersistedClient {
  return {
    timestamp: Date.now(),
    buster: '1',
    clientState: { mutations: [], queries } as unknown as DehydratedState,
  }
}

/** Drain tieredEvict repeatedly, returning the queryKeys in removal order. */
function evictionOrder(queries: ReturnType<typeof dq>[]): QueryKey[] {
  let pc: PersistedClient | undefined = client(queries)
  const removed: QueryKey[] = []
  for (;;) {
    const before = pc!.clientState.queries
    const next = tieredEvict({ persistedClient: pc!, error: new Error('quota'), errorCount: 1 })
    if (!next) break
    const after = next.clientState.queries
    const gone = before.find((b) => !after.some((a) => a.queryHash === b.queryHash))
    removed.push(gone!.queryKey)
    pc = next
    if (after.length === 0) break
  }
  return removed
}

// --- evictionTier ----------------------------------------------------------

describe('evictionTier', () => {
  it('orders messages < chats < game < games < recent-games', () => {
    expect(evictionTier(['messages', 'c1'])).toBeLessThan(evictionTier(['chats', 'g1']))
    expect(evictionTier(['chats', 'g1'])).toBeLessThan(evictionTier(['game', 'g1']))
    expect(evictionTier(['game', 'g1'])).toBeLessThan(evictionTier(['games']))
    expect(evictionTier(['games'])).toBeLessThan(evictionTier(['recent-games']))
  })
})

// --- tieredEvict -----------------------------------------------------------

describe('tieredEvict', () => {
  it('evicts by tier, then oldest dataUpdatedAt within a tier', () => {
    const queries = [
      dq(['recent-games'], 10),
      dq(['games'], 10),
      dq(['messages', 'newest'], 300),
      dq(['messages', 'oldest'], 100),
      dq(['messages', 'middle'], 200),
      dq(['chats', 'g1'], 50),
    ]
    const order = evictionOrder(queries)
    expect(order).toEqual([
      ['messages', 'oldest'],
      ['messages', 'middle'],
      ['messages', 'newest'],
      ['chats', 'g1'],
      ['games'],
      ['recent-games'],
    ])
  })

  it('returns undefined when there is nothing left to trim', () => {
    expect(tieredEvict({ persistedClient: client([]), error: new Error('x'), errorCount: 1 })).toBeUndefined()
  })
})

// --- makeShouldDehydrateQuery ----------------------------------------------

describe('makeShouldDehydrateQuery', () => {
  function addQuery(qc: QueryClient, key: QueryKey, updatedAt: number, status: 'success' | 'pending' = 'success') {
    const q = qc.getQueryCache().build(qc, { queryKey: key })
    if (status === 'success') q.setData('data', { updatedAt, manual: true })
    return q
  }

  it('always persists list / chat / game queries, drops non-success', () => {
    const qc = new QueryClient()
    const should = makeShouldDehydrateQuery(qc)
    expect(should(addQuery(qc, ['games'], 1))).toBe(true)
    expect(should(addQuery(qc, ['recent-games'], 1))).toBe(true)
    expect(should(addQuery(qc, ['chats', 'g1'], 1))).toBe(true)
    expect(should(addQuery(qc, ['game', 'g1'], 1))).toBe(true)
    expect(should(addQuery(qc, ['games-pending'], 0, 'pending'))).toBe(false)
  })

  it('keeps only the N most-recent message histories', () => {
    const qc = new QueryClient()
    const total = MAX_PERSISTED_MESSAGE_CHATS + 5
    const queries = Array.from({ length: total }, (_, i) => addQuery(qc, ['messages', `chat-${i}`], i + 1))
    const should = makeShouldDehydrateQuery(qc)

    const kept = queries.filter((q) => should(q))
    expect(kept).toHaveLength(MAX_PERSISTED_MESSAGE_CHATS)
    // The 5 oldest (lowest updatedAt) are the ones dropped.
    expect(should(queries[0])).toBe(false)
    expect(should(queries[4])).toBe(false)
    expect(should(queries[5])).toBe(true)
    expect(should(queries[total - 1])).toBe(true)
  })
})
