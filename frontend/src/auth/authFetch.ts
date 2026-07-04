import { useCallback } from 'react'
import { useAuth } from './useAuth'
import { fetchWithRetry } from '@/lib/fetchWithRetry'

const BASE_URL = import.meta.env.VITE_API_URL ?? ''

/**
 * Returns an async fetch wrapper that injects `Authorization: Bearer <id_token>`
 * on every request. Drop-in replacement for the global `fetch`.
 *
 * Requests are retried through Cloud Run cold starts (see fetchWithRetry), so a
 * transient boot-time failure never surfaces to the caller as an error.
 */
export function useAuthFetch() {
  const { getIdToken } = useAuth()

  return useCallback(async (url: string, options: RequestInit = {}): Promise<Response> => {
    const token = await getIdToken()
    return fetchWithRetry(fetch, `${BASE_URL}${url}`, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${token}`,
      },
    })
  }, [getIdToken])
}
