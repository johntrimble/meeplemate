import { useQuery } from '@tanstack/react-query'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GameInfo } from '@/data/games'

interface UseGameResult {
  game: Game | null
  isLoading: boolean
  error: string | null
}

export function useGame(gameId: string): UseGameResult {
  const authFetch = useAuthFetch()

  const query = useQuery({
    queryKey: ['game', gameId],
    queryFn: ({ signal }) =>
      authFetch(`/api/games/${gameId}`, { signal })
        .then((r) => {
          if (!r.ok) throw new Error(`Game not found: ${r.status}`)
          return r.json() as Promise<GameInfo>
        })
        .then(apiGameToGame),
  })

  return {
    game: query.data ?? null,
    isLoading: query.isPending,
    error: query.error ? (query.error as Error).message : null,
  }
}
