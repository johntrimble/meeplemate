import { queryClient, localStoragePersister } from './queryClient'

// ---------------------------------------------------------------------------
// Cache identity isolation
// ---------------------------------------------------------------------------
//
// Every data endpoint requires auth, and React Query keys are not user-scoped,
// so the persisted cache holds one user's data at a time. To avoid leaking one
// user's data to the next on a shared device, we track the last authenticated
// uid and clear the cache (memory + localStorage) whenever a *different* user
// is seen. A returning same user keeps their warm cache.

const LAST_UID_KEY = 'boardbarian-last-uid'

function dropCache() {
  queryClient.clear()
  localStoragePersister.removeClient()
}

/**
 * Reconcile the cache against the current identity. Clears everything if the
 * last authenticated uid differs from `uid`; a matching uid is a no-op so the
 * warm cache survives (including after a lapsed session + re-login). Always
 * records `uid` as the last identity.
 */
export function reconcileUserCache(uid: string) {
  const lastUid = localStorage.getItem(LAST_UID_KEY)
  if (lastUid && lastUid !== uid) {
    dropCache()
  }
  localStorage.setItem(LAST_UID_KEY, uid)
}

/**
 * Clear the cache on explicit logout - both for privacy at rest and so the next
 * user starts clean. A lapsed session (not an explicit logout) must NOT call
 * this, or a returning same user would lose their cache.
 */
export function clearUserCache() {
  dropCache()
  localStorage.removeItem(LAST_UID_KEY)
}
