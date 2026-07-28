import { useEffect } from 'react'
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
 * In-flight POST per uid, so two callers can't race the same record.
 *
 * There are two entry points - `ConsentScreen` fires one the moment the user
 * accepts, and `useAcceptanceSync` retries one that never landed - and the flag
 * they coordinate through (`synced` in localStorage) is only written when a
 * request *settles*. For the whole time a request is outstanding the stored
 * state still reads "unsent", so nothing in localStorage can tell a second
 * caller that a first is already on its way. That window is at its widest
 * exactly when it matters: a cold backend holds the POST open for up to 90s, and
 * any re-render of `AuthProvider` in that time (a Firebase token refresh is
 * enough) re-runs the sync effect.
 *
 * Keyed by uid rather than a bare boolean so switching accounts mid-flight
 * doesn't suppress the new account's acceptance.
 */
const inFlight = new Map<string, Promise<void>>()

/**
 * POST the *stored* acceptance for `uid` and clear its retry flag once there is
 * nothing more to try. Never throws, never blocks anything the user can see.
 *
 * Posts what is stored rather than today's constants: on a retry those can
 * differ (a version bump landed between accepting and syncing), and the record
 * has to say what the user actually agreed to, not what is current now.
 *
 * Concurrent calls for the same uid share one request. A no-op when nothing is
 * stored.
 */
export function postAcceptance(authFetch: AuthFetch, uid: string): Promise<void> {
  const existing = inFlight.get(uid)
  if (existing) return existing

  const request = sendAcceptance(authFetch, uid).finally(() => {
    inFlight.delete(uid)
  })
  inFlight.set(uid, request)
  return request
}

async function sendAcceptance(authFetch: AuthFetch, uid: string): Promise<void> {
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

  useEffect(() => {
    if (!uid) return
    const stored = readAcceptance(uid)
    if (!stored || stored.synced) return
    // Deduplication lives in `postAcceptance`, not here: this effect is only one
    // of the two callers, and a guard local to it cannot see the request
    // `ConsentScreen` fired moments earlier. Re-running this - StrictMode's
    // double-invoke, a changed `authFetch` identity - is therefore free.
    void postAcceptance(authFetch, uid)
  }, [uid, authFetch])
}
