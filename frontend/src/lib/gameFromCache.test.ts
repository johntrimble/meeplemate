import { describe, expect, it } from 'vitest'
import { QueryClient } from '@tanstack/react-query'
import type { Game, GamesPage } from '@/data/games'
import { findCachedGame } from './gameFromCache'

const CATAN = { id: 'catan', name: 'Catan', emoji: '🏝️', background_color: '#92400e' }
const MUNCHKIN = { id: 'munchkin', name: 'Munchkin', emoji: '⚔️', background_color: '#7f1d1d' }

function page(...data: GamesPage['data']): GamesPage {
  return { pageInfo: { hasNextPage: false }, data }
}

function withCatalog(...pages: GamesPage[]) {
  const qc = new QueryClient()
  qc.setQueryData(['games'], { pages, pageParams: pages.map(() => undefined) })
  return qc
}

describe('findCachedGame', () => {
  it('finds a game in the seeded catalog and maps it to the detail shape', () => {
    const qc = withCatalog(page(CATAN, MUNCHKIN))

    const hit = findCachedGame(qc, 'munchkin')

    // Same mapping useGame's queryFn applies to the detail response.
    expect(hit?.game).toEqual({
      id: 'munchkin',
      name: 'Munchkin',
      shortName: 'Munchkin',
      emoji: '⚔️',
      bgColor: '#7f1d1d',
      exampleQuestions: undefined,
    })
  })

  it('searches every page of the infinite query, not just the first', () => {
    const qc = withCatalog(page(CATAN), page(MUNCHKIN))

    expect(findCachedGame(qc, 'munchkin')?.game.id).toBe('munchkin')
  })

  it("reports the list entry's own freshness, not now", () => {
    const qc = withCatalog(page(CATAN))
    const seededAt = qc.getQueryCache().find({ queryKey: ['games'] })!.state.dataUpdatedAt

    expect(findCachedGame(qc, 'catan')?.updatedAt).toBe(seededAt)
  })

  it('falls back to recent-games, which stores already-mapped games', () => {
    const qc = new QueryClient()
    const recent: Game[] = [
      { id: 'gloomhaven', name: 'Gloomhaven', shortName: 'Gloomhaven', emoji: '🗡️', bgColor: '#1f2937' },
    ]
    qc.setQueryData(['recent-games'], recent)

    expect(findCachedGame(qc, 'gloomhaven')?.game).toEqual(recent[0])
  })

  it('prefers the catalog when a game is in both lists', () => {
    const qc = withCatalog(page(CATAN))
    qc.setQueryData(['recent-games'], [{ id: 'catan', name: 'stale', shortName: 'stale', emoji: '❓', bgColor: '#000' }])

    expect(findCachedGame(qc, 'catan')?.game.name).toBe('Catan')
  })

  it('misses when the game is in no cached list (beyond the snapshot, or newly ingested)', () => {
    const qc = withCatalog(page(CATAN))

    expect(findCachedGame(qc, 'wingspan')).toBeUndefined()
  })

  it('misses on an empty cache', () => {
    expect(findCachedGame(new QueryClient(), 'catan')).toBeUndefined()
  })

  it('ignores a non-success catalog entry', () => {
    const qc = new QueryClient()
    // An errored/pending query has no usable data even though the key exists.
    qc.getQueryCache().build(qc, { queryKey: ['games'] })

    expect(findCachedGame(qc, 'catan')).toBeUndefined()
  })
})
