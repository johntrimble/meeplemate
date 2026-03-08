import { useCallback, useEffect, useRef, useState } from 'react'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GamesPage } from '@/data/games'

interface UseGameListResult {
  games: Game[]
  isLoading: boolean
  error: string | null
  hasMore: boolean
  loadMore: () => void
}

export function useGameList(): UseGameListResult {
  const authFetch = useAuthFetch()
  const [games, setGames] = useState<Game[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(true)
  const cursorRef = useRef<string | undefined>(undefined)
  const loadingRef = useRef(false)

  const loadMore = useCallback(() => {
    if (loadingRef.current || !hasMore) return
    loadingRef.current = true
    setIsLoading(true)

    const params = new URLSearchParams({ first: '20' })
    if (cursorRef.current) {
      params.set('cursor', cursorRef.current)
    }

    authFetch(`/api/games?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch games: ${r.status}`)
        return r.json() as Promise<GamesPage>
      })
      .then((page) => {
        setGames((prev) => [...prev, ...page.data.map(apiGameToGame)])
        setHasMore(page.pageInfo.hasNextPage)
        cursorRef.current = page.pageInfo.endCursor
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load games')
      })
      .finally(() => {
        loadingRef.current = false
        setIsLoading(false)
      })
  }, [authFetch, hasMore])

  useEffect(() => {
    loadMore()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { games, isLoading, error, hasMore, loadMore }
}
