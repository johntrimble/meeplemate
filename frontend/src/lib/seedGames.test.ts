import { afterEach, describe, expect, it, vi } from 'vitest'
import { QueryClient } from '@tanstack/react-query'
import type { GamesPage } from '@/data/games'
import { seedGamesFromStatic } from './seedGames'

const PAGE: GamesPage = {
  pageInfo: { hasNextPage: false },
  data: [
    { id: 'catan', name: 'Catan', emoji: '🏝️', background_color: '#92400e' },
    { id: 'munchkin', name: 'Munchkin', emoji: '⚔️', background_color: '#7f1d1d' },
  ],
}

function mockFetch(impl: () => Promise<Response> | Response) {
  vi.stubGlobal('fetch', vi.fn(impl))
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('seedGamesFromStatic', () => {
  it('seeds the games cache in the InfiniteData shape the hook reads', async () => {
    mockFetch(() => new Response(JSON.stringify(PAGE), { status: 200 }))
    const qc = new QueryClient()

    await seedGamesFromStatic(qc)

    // Matches how useGameList reads `data.pages` (one page = one GamesPage).
    expect(qc.getQueryData(['games'])).toEqual({ pages: [PAGE], pageParams: [undefined] })
  })

  it('does not overwrite an existing (persisted/warmer) games entry', async () => {
    const fetchSpy = vi.fn(() => new Response(JSON.stringify(PAGE), { status: 200 }))
    vi.stubGlobal('fetch', fetchSpy)
    const qc = new QueryClient()
    const existing = { pages: [{ ...PAGE, data: [] }], pageParams: [undefined] }
    qc.setQueryData(['games'], existing)

    await seedGamesFromStatic(qc)

    expect(qc.getQueryData(['games'])).toBe(existing)
    expect(fetchSpy).not.toHaveBeenCalled() // short-circuits before fetching
  })

  it('no-ops when the snapshot is missing (404)', async () => {
    mockFetch(() => new Response('not found', { status: 404 }))
    const qc = new QueryClient()

    await seedGamesFromStatic(qc)

    expect(qc.getQueryData(['games'])).toBeUndefined()
  })

  it('no-ops on fetch failure (offline / malformed)', async () => {
    mockFetch(() => Promise.reject(new TypeError('network error')))
    const qc = new QueryClient()

    await seedGamesFromStatic(qc)

    expect(qc.getQueryData(['games'])).toBeUndefined()
  })

  it('gives up on a hanging snapshot instead of blocking startup forever', async () => {
    // Never resolves on its own; only the abort signal can end it. App.tsx
    // returns this promise to the restore gate, so it MUST settle regardless.
    mockFetch(
      (...args: unknown[]) =>
        new Promise<Response>((_resolve, reject) => {
          const { signal } = (args[1] ?? {}) as { signal?: AbortSignal }
          signal?.addEventListener('abort', () => reject(signal.reason))
        })
    )
    const qc = new QueryClient()

    await expect(seedGamesFromStatic(qc, 10)).resolves.toBeUndefined()

    expect(qc.getQueryData(['games'])).toBeUndefined()
  })

  it('no-ops when the snapshot has no games', async () => {
    const empty: GamesPage = { pageInfo: { hasNextPage: false }, data: [] }
    mockFetch(() => new Response(JSON.stringify(empty), { status: 200 }))
    const qc = new QueryClient()

    await seedGamesFromStatic(qc)

    expect(qc.getQueryData(['games'])).toBeUndefined()
  })
})
