import { useInfiniteQuery } from '@tanstack/react-query'
import { useAuthFetch } from '@/auth/authFetch'
import { apiGameToGame, type Game, type GamesPage } from '@/data/games'

interface UseGameListResult {
  games: Game[]
  isLoading: boolean
  isFetchingMore: boolean
  error: string | null
  hasMore: boolean
  loadMore: () => void
}

export function useGameList(): UseGameListResult {
  const authFetch = useAuthFetch()

  const query = useInfiniteQuery({
    queryKey: ['games'],
    queryFn: ({ pageParam, signal }) => {
      // Fetch the whole catalog in one page (100 = the endpoint's max). This
      // matches the deploy-time `games.json` snapshot exactly, so the seeded
      // first paint and this background revalidation are byte-identical (no
      // page-size mismatch, no list resize on revalidate).
      const params = new URLSearchParams({ first: '100' })
      if (pageParam) params.set('cursor', pageParam)
      return authFetch(`/api/games?${params}`, { signal })
        .then((r) => {
          if (!r.ok) throw new Error(`Failed to fetch games: ${r.status}`)
          return r.json() as Promise<GamesPage>
        })
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) =>
      lastPage.pageInfo.hasNextPage ? (lastPage.pageInfo.endCursor ?? undefined) : undefined,
  })

  return {
    games: query.data?.pages.flatMap((p) => p.data.map(apiGameToGame)) ?? [],
    isLoading: query.isPending,
    isFetchingMore: query.isFetchingNextPage,
    error: query.error ? (query.error as Error).message : null,
    hasMore: query.hasNextPage,
    loadMore: () => query.fetchNextPage(),
  }
}
