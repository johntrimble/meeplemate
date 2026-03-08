import { useEffect, useState } from 'react'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GamesPage } from '@/data/games'

interface UseRecentGamesResult {
  games: Game[]
  isLoading: boolean
}

export function useRecentGames(): UseRecentGamesResult {
  const authFetch = useAuthFetch()
  const [games, setGames] = useState<Game[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    authFetch('/api/recent-games')
      .then((r) => r.json() as Promise<GamesPage>)
      .then((page) => setGames(page.data.map(apiGameToGame)))
      .catch(() => {})
      .finally(() => setIsLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { games, isLoading }
}
