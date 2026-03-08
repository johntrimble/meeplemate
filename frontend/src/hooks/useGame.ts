import { useEffect, useState } from 'react'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GameInfo } from '@/data/games'

interface UseGameResult {
  game: Game | null
  isLoading: boolean
  error: string | null
}

export function useGame(gameId: string): UseGameResult {
  const authFetch = useAuthFetch()
  const [game, setGame] = useState<Game | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setIsLoading(true)
    setError(null)
    authFetch(`/api/games/${gameId}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Game not found: ${r.status}`)
        return r.json() as Promise<GameInfo>
      })
      .then((g) => setGame(apiGameToGame(g)))
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load game')
      })
      .finally(() => setIsLoading(false))
  // authFetch is stable within a session; gameId is the real dep.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameId])

  return { game, isLoading, error }
}
