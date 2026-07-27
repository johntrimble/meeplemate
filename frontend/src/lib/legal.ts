// ---------------------------------------------------------------------------
// Terms of Use / Privacy Policy acceptance
// ---------------------------------------------------------------------------
//
// The gate decision is made entirely from local state - a bundled constant vs a
// localStorage entry - and never waits on the network. That is deliberate: the
// whole cold-start design (seedGames.ts, the IndexedDB persister, fetchWithRetry)
// exists so first paint never blocks on the scale-to-zero backend, and putting a
// "have you accepted?" request in front of the app would undo it. On a cold start
// the user would stare at a splash for up to 90s before seeing anything.
//
// Both halves of the comparison are already local, so no request is needed:
//
//   localStorage matches these constants -> render, zero network
//   localStorage holds an older version  -> prompt, zero network (a version bump)
//   localStorage empty                   -> prompt, zero network (new user, new
//                                           device, or a cleared browser)
//
// Blocking on empty re-prompts a returning user on a new device. That is the
// accepted trade for never letting an un-accepted user in and never issuing a
// status request. The POST is idempotent, so re-accepting costs them one click
// and leaves the recorded `accepted_at` alone.
//
// The server enforces the same rule in `get_db_user` and is the authority; this
// is the UX half. See docs/legal.md.

/** Effective date of src/content/terms.md. Must match `TERMS_VERSION` in meeplemate/server/legal.py. */
export const TERMS_VERSION = '2026-07-28'

/** Effective date of src/content/privacy.md. Must match `PRIVACY_VERSION` in meeplemate/server/legal.py. */
export const PRIVACY_VERSION = '2026-07-28'

const STORAGE_PREFIX = 'boardbarian-legal-v1'

export interface StoredAcceptance {
  termsVersion: string
  privacyVersion: string
}

/** Whether the gate should prompt, and if so whether this is a re-consent. */
export type AcceptanceState = 'current' | 'outdated' | 'none'

// Scoped by uid so switching accounts in one browser can't inherit the other
// account's acceptance - the same reason userCache.ts reconciles on uid.
function storageKey(uid: string): string {
  return `${STORAGE_PREFIX}:${uid}`
}

export function readAcceptance(uid: string): StoredAcceptance | null {
  try {
    const raw = localStorage.getItem(storageKey(uid))
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<StoredAcceptance>
    if (
      typeof parsed?.termsVersion !== 'string' ||
      typeof parsed?.privacyVersion !== 'string'
    ) {
      return null
    }
    return { termsVersion: parsed.termsVersion, privacyVersion: parsed.privacyVersion }
  } catch {
    // Malformed JSON, or storage blocked entirely (Safari private mode, a
    // hardened profile). Treat as "not accepted": prompting again is harmless,
    // silently letting someone through is not.
    return null
  }
}

/**
 * Record acceptance locally. Call this only *after* the server has confirmed -
 * writing it optimistically would let the client believe it had accepted while
 * the server disagreed, and the user would hit a 403 on their first question
 * with no way to get back to the prompt.
 */
export function writeAcceptance(uid: string): void {
  try {
    localStorage.setItem(
      storageKey(uid),
      JSON.stringify({
        termsVersion: TERMS_VERSION,
        privacyVersion: PRIVACY_VERSION,
      } satisfies StoredAcceptance),
    )
  } catch {
    // Storage full or blocked. The acceptance is already recorded server-side,
    // so the only cost is being prompted again next load.
  }
}

/** Classify what the stored acceptance means for the version now in force. */
export function acceptanceState(uid: string): AcceptanceState {
  const stored = readAcceptance(uid)
  if (!stored) return 'none'
  // Both documents must match. They're versioned separately so a privacy-only
  // revision needn't claim the Terms changed, but consent is collected for the
  // pair, so either being stale means they haven't agreed to today's terms.
  if (stored.termsVersion === TERMS_VERSION && stored.privacyVersion === PRIVACY_VERSION) {
    return 'current'
  }
  return 'outdated'
}
