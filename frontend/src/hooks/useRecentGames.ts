import { useQuery } from '@tanstack/react-query'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GamesPage } from '@/data/games'

interface UseRecentGamesResult {
  games: Game[]
  isLoading: boolean
}

export function useRecentGames(): UseRecentGamesResult {
  const authFetch = useAuthFetch()

  const query = useQuery({
    queryKey: ['recent-games'],
    queryFn: ({ signal }) =>
      authFetch('/api/recent-games', { signal })
        .then((r) => r.json() as Promise<GamesPage>)
        .then((page) => page.data.map(apiGameToGame)),
  })

  return {
    games: query.data ?? [],
    isLoading: query.isPending,
  }
}
