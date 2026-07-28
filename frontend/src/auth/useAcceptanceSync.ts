import { useEffect, useRef } from 'react'
import { useAuthFetch } from './authFetch'
import { markSynced, readAcceptance } from '@/lib/legal'

// ---------------------------------------------------------------------------
// Getting the acceptance record to the server, eventually
// ---------------------------------------------------------------------------
//
// Accepting the terms writes localStorage and opens the app immediately; the
// POST that records it server-side is fired without being awaited. That is the
// whole point - awaiting it put a Cloud Run cold start (~26s, up to 90s of
// retries) in front of every new user's first screen.
//
// The cost of not awaiting is that the POST can be lost: a cold start that
// outlasts the retry window, a closed tab, a dead connection. Since the record
// is the one thing the backend is still kept around for, losing it silently is
// not acceptable - so an unsent acceptance is remembered in localStorage and
// retried on subsequent loads until it settles.

type AuthFetch = (url: string, options?: RequestInit) => Promise<Response>

/**
 * POST the *stored* acceptance for `uid` and clear its retry flag once there is
 * nothing more to try. Never throws, never blocks anything the user can see.
 *
 * Posts what is stored rather than today's constants: on a retry those can
 * differ (a version bump landed between accepting and syncing), and the record
 * has to say what the user actually agreed to, not what is current now.
 *
 * A no-op when nothing is stored.
 */
export async function postAcceptance(authFetch: AuthFetch, uid: string): Promise<void> {
  const stored = readAcceptance(uid)
  if (!stored) return

  try {
    const res = await authFetch('/api/account/legal-acceptance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        terms_version: stored.termsVersion,
        privacy_version: stored.privacyVersion,
      }),
    })

    if (res.ok) {
      markSynced(uid)
      return
    }

    // A 4xx will fail identically on every future load, so retrying it is an
    // infinite loop rather than eventual consistency: stop, and make the noise
    // visible instead. The server only rejects a malformed version, and both
    // sides' versions come from one repo, so reaching this is a bug.
    if (res.status < 500) {
      console.error('legal acceptance rejected', res.status)
      markSynced(uid)
      return
    }

    // 5xx that survived fetchWithRetry's cold-start handling - a real server
    // error. Leave it pending; the next load tries again.
    console.error('legal acceptance failed', res.status)
  } catch (err) {
    // No answer at all (offline, or a cold start outlasting the retry window).
    // Exactly the case another attempt can fix, so leave it pending.
    console.error('legal acceptance could not be sent', err)
  }
}

/**
 * Retry an acceptance that never reached the server.
 *
 * Call from inside `CacheGate` (it needs auth context). Runs after the gate has
 * opened and must never gate anything itself - a user whose record is still
 * pending is already using the app, which is the design.
 */
export function useAcceptanceSync(uid: string | null): void {
  const authFetch = useAuthFetch()
  // `markSynced` writes localStorage, which doesn't re-render, so a re-run can't
  // see the result of an attempt still in flight. Without this the effect firing
  // twice (a changed `authFetch` identity, StrictMode) would double-POST.
  const inFlightFor = useRef<string | null>(null)

  useEffect(() => {
    if (!uid) return
    const stored = readAcceptance(uid)
    if (!stored || stored.synced) return
    if (inFlightFor.current === uid) return

    inFlightFor.current = uid
    void postAcceptance(authFetch, uid).finally(() => {
      if (inFlightFor.current === uid) inFlightFor.current = null
    })
  }, [uid, authFetch])
}
