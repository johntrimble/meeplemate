import { QueryClient } from '@tanstack/react-query'
import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister'
import { get, set, del } from 'idb-keyval'
import { RETENTION_MS, tieredEvict } from './cachePersist'

const CACHE_KEY = 'boardbarian-cache-v1'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 0,
      gcTime: RETENTION_MS,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      retry: false,
    },
  },
})

// Persist the cache to IndexedDB (large quota, async writes - unlike the old
// synchronous localStorage blob). `tieredEvict` trims in priority order if a
// write ever hits the storage quota.
export const persister = createAsyncStoragePersister({
  storage: { getItem: get, setItem: set, removeItem: del },
  key: CACHE_KEY,
  retry: tieredEvict,
})

// One-time migration: the cache used to live in localStorage under this key.
// Drop the now-dead blob so it doesn't waste the localStorage quota.
localStorage.removeItem(CACHE_KEY)
