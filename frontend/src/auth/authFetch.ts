import { useCallback } from 'react'
import { useAuth } from './useAuth'

/**
 * Returns an async fetch wrapper that injects `Authorization: Bearer <id_token>`
 * on every request. Drop-in replacement for the global `fetch`.
 */
export function useAuthFetch() {
  const { getIdToken } = useAuth()

  return useCallback(async (url: string, options: RequestInit = {}): Promise<Response> => {
    const token = await getIdToken()
    return fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${token}`,
      },
    })
  }, [getIdToken])
}
