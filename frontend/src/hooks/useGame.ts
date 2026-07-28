import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GameInfo } from '@/data/games'
import { findCachedGame } from '@/lib/gameFromCache'

interface UseGameResult {
  game: Game | null
  isLoading: boolean
  error: string | null
}

export function useGame(gameId: string): UseGameResult {
  const authFetch = useAuthFetch()
  const queryClient = useQueryClient()

  // A game already in a cached list is a complete detail response (see
  // `findCachedGame`), so seed from it rather than blocking the whole chat page
  // on a cold-starting backend. `initialData` (not `placeholderData`) so the
  // usual `['game', id]` cache entry is still created and persisted; with
  // `staleTime: 0` this always revalidates against `/api/games/{id}` in the
  // background. A miss returns undefined — the query then fetches as before.
  const cached = findCachedGame(queryClient, gameId)

  const query = useQuery({
    queryKey: ['game', gameId],
    queryFn: ({ signal }) =>
      authFetch(`/api/games/${gameId}`, { signal })
        .then((r) => {
          if (!r.ok) throw new Error(`Game not found: ${r.status}`)
          return r.json() as Promise<GameInfo>
        })
        .then(apiGameToGame),
    initialData: cached?.game,
    initialDataUpdatedAt: cached?.updatedAt,
  })

  return {
    game: query.data ?? null,
    isLoading: query.isPending,
    error: query.error ? (query.error as Error).message : null,
  }
}
