import { useEffect, useState, type ReactNode } from 'react'
import { useIsRestoring } from '@tanstack/react-query'
import { useAuth } from './useAuth'
import { LoginScreen } from './LoginScreen'
import { reconcileUserCache } from '@/lib/userCache'

/**
 * Gate for routes that render user-specific cached data. It:
 *   1. waits for the persisted cache to finish restoring AND auth to resolve,
 *   2. requires a signed-in user (shows the login screen otherwise),
 *   3. reconciles the cache against the current identity (clearing it if the
 *      user changed) BEFORE revealing children.
 *
 * Gating the reconcile on `!isRestoring` avoids the race where the persister
 * rehydrates the previous user's data right after we clear it; gating render on
 * "reconciled for the current uid" avoids any one-frame flash of stale data.
 */
export function CacheGate({ children }: { children: ReactNode }) {
  const isRestoring = useIsRestoring()
  const { user, isLoading } = useAuth()
  const [reconciledUid, setReconciledUid] = useState<string | null>(null)

  const uid = user?.uid ?? null

  useEffect(() => {
    if (isRestoring || isLoading || !uid) return
    let cancelled = false
    // Only reveal children once the reconcile (incl. the async IndexedDB delete)
    // has finished, and never set state after unmount or a uid change.
    reconcileUserCache(uid)
      .then(() => {
        if (!cancelled) setReconciledUid(uid)
      })
      .catch((err) => console.error('cache reconcile failed', err))
    return () => {
      cancelled = true
    }
  }, [isRestoring, isLoading, uid])

  // Still settling (restoring cache or resolving auth).
  if (isRestoring || isLoading) return <Splash />
  // Identity must be established before any user data is shown.
  if (!user) return <LoginScreen />
  // Hold until the cache has been reconciled for THIS user (no stale flash).
  if (reconciledUid !== user.uid) return <Splash />

  return <>{children}</>
}

function Splash() {
  return <div className="fixed inset-0 bg-background" />
}
