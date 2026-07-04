import { queryClient, persister } from './queryClient'

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

async function dropCache() {
  queryClient.clear()
  // Await the IndexedDB deletion - the persister is async, so callers must not
  // record new ownership until the old blob is actually gone (see below).
  await persister.removeClient()
}

/**
 * Reconcile the cache against the current identity. Clears everything unless the
 * last recorded uid matches `uid` - so a matching uid is a no-op (warm cache
 * survives, including after a lapsed session + re-login), while a *different* or
 * *unknown* owner is dropped. Treating an unknown owner (no recorded uid but a
 * persisted cache present - e.g. a browser that had a cache before this feature
 * shipped) as different prevents serving that stale cache to whoever logs in.
 *
 * The new uid is recorded only *after* the old blob is deleted: if we're
 * interrupted mid-delete, `last-uid` stays old so the next load re-clears rather
 * than serving the previous user's data under the new owner.
 */
export async function reconcileUserCache(uid: string) {
  const lastUid = localStorage.getItem(LAST_UID_KEY)
  if (lastUid !== uid) {
    await dropCache()
  }
  localStorage.setItem(LAST_UID_KEY, uid)
}

/**
 * Clear the cache on explicit logout - both for privacy at rest and so the next
 * user starts clean. A lapsed session (not an explicit logout) must NOT call
 * this, or a returning same user would lose their cache.
 */
export async function clearUserCache() {
  await dropCache()
  localStorage.removeItem(LAST_UID_KEY)
}
