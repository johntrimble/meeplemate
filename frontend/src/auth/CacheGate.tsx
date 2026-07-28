import { useEffect, useState, type ReactNode } from 'react'
import { useIsRestoring } from '@tanstack/react-query'
import { useAuth } from './useAuth'
import { LoginScreen } from './LoginScreen'
import { ConsentScreen } from './ConsentScreen'
import { useLegalAcceptance } from './useLegalAcceptance'
import { reconcileUserCache } from '@/lib/userCache'

/**
 * Gate for routes that render user-specific cached data. It:
 *   1. waits for the persisted cache to finish restoring AND auth to resolve,
 *   2. requires a signed-in user (shows the login screen otherwise),
 *   3. requires acceptance of the current Terms and Privacy Policy,
 *   4. reconciles the cache against the current identity (clearing it if the
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
  const { state: legalState, markAccepted } = useLegalAcceptance(uid)

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
  // Consent before the reconcile check, not after: the decision is synchronous
  // (localStorage vs a bundled constant, no network), so there's no reason to
  // hold it behind the async IndexedDB reconcile's splash.
  if (legalState !== 'current') {
    return <ConsentScreen state={legalState} onAccepted={markAccepted} />
  }
  // Hold until the cache has been reconciled for THIS user (no stale flash).
  if (reconciledUid !== user.uid) return <Splash />

  return <>{children}</>
}

function Splash() {
  return <div className="fixed inset-0 bg-background" />
}
