import { useInfiniteQuery } from '@tanstack/react-query'
import { useAuthFetch } from '@/auth/authFetch'

export interface ChatSummary {
  chat_id: string
  title: string
}

interface ChatsPage {
  pageInfo: { hasNextPage: boolean; endCursor: string | null }
  data: ChatSummary[]
}

interface UseChatsResult {
  chats: ChatSummary[]
  hasNextPage: boolean
  loadMore: () => void
  isFetching: boolean
  isPending: boolean
  isFetchingMore: boolean
}

export function useChats(gameId: string, enabled: boolean): UseChatsResult {
  const authFetch = useAuthFetch()

  const query = useInfiniteQuery({
    queryKey: ['chats', gameId],
    queryFn: ({ pageParam, signal }) => {
      const params = new URLSearchParams({ first: '20' })
      if (pageParam) params.set('cursor', pageParam)
      return authFetch(`/api/games/${gameId}/chats?${params}`, { signal })
        .then((r) => r.json() as Promise<ChatsPage>)
    },
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) =>
      lastPage.pageInfo.hasNextPage ? lastPage.pageInfo.endCursor : null,
    enabled,
  })

  return {
    chats: query.data?.pages.flatMap((p) => p.data) ?? [],
    hasNextPage: query.hasNextPage,
    loadMore: () => query.fetchNextPage(),
    isFetching: query.isFetching,
    isPending: query.isPending,
    isFetchingMore: query.isFetchingNextPage,
  }
}
