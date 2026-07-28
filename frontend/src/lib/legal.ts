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
// The server does NOT enforce this. It used to 403 un-accepted accounts, which
// is why accepting had to await the POST before the app could open — putting a
// cold start in front of every new user's first screen. The server now only
// *records* acceptance; this gate is the whole enforcement. See docs/legal.md.

/** Effective date of src/content/terms.md. Must match `TERMS_VERSION` in meeplemate/server/legal.py. */
export const TERMS_VERSION = '2026-07-28'

/** Effective date of src/content/privacy.md. Must match `PRIVACY_VERSION` in meeplemate/server/legal.py. */
export const PRIVACY_VERSION = '2026-07-28'

const STORAGE_PREFIX = 'boardbarian-legal-v1'

export interface StoredAcceptance {
  termsVersion: string
  privacyVersion: string
  /**
   * Whether there is anything left to send to the server — NOT "the server has
   * it". Set false when acceptance is recorded locally, and true once the POST
   * has settled in a way retrying cannot improve: a success, or a 4xx that will
   * fail identically forever. Only a request that got no answer at all (offline,
   * cold start) leaves it false, because that is the one case another attempt
   * can fix. See `useAcceptanceSync`.
   */
  synced: boolean
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
    return {
      termsVersion: parsed.termsVersion,
      privacyVersion: parsed.privacyVersion,
      // Entries written before `synced` existed were only ever stored *after* a
      // successful POST, so they are synced by construction. Defaulting to false
      // would make the deploy that ships this re-POST for every existing user at
      // once, against the same scale-to-zero backend, for nothing.
      synced: parsed.synced !== false,
    }
  } catch {
    // Malformed JSON, or storage blocked entirely (Safari private mode, a
    // hardened profile). Treat as "not accepted": prompting again is harmless,
    // silently letting someone through is not.
    return null
  }
}

/**
 * Record acceptance locally, immediately, without waiting for the server.
 *
 * This used to be forbidden: while the server 403'd un-accepted accounts, a
 * client that believed it had accepted while the server disagreed would hit a
 * wall on its first question with no way back to the prompt. That 403 is gone,
 * so the hazard is gone with it — and the cost it was imposing was real, since
 * awaiting the POST meant every new user met a cold start before seeing the app.
 *
 * Written `synced: false`; `useAcceptanceSync` is what eventually clears it.
 */
export function writeAcceptance(uid: string): void {
  try {
    localStorage.setItem(
      storageKey(uid),
      JSON.stringify({
        termsVersion: TERMS_VERSION,
        privacyVersion: PRIVACY_VERSION,
        synced: false,
      } satisfies StoredAcceptance),
    )
  } catch {
    // Storage full or blocked. Acceptance still reaches the server, so the only
    // cost is being prompted again next load.
  }
}

/**
 * Mark the stored acceptance as needing no further POST.
 *
 * Preserves the stored *versions* rather than re-writing today's constants: what
 * settled with the server is what was sent, and if the two have since diverged
 * (a version bump between accepting and syncing) the gate must still see the old
 * versions and re-prompt.
 *
 * A no-op if nothing is stored, so a cleared browser mid-flight can't resurrect
 * an acceptance.
 */
export function markSynced(uid: string): void {
  const stored = readAcceptance(uid)
  if (!stored) return
  try {
    localStorage.setItem(
      storageKey(uid),
      JSON.stringify({ ...stored, synced: true } satisfies StoredAcceptance),
    )
  } catch {
    // Storage blocked. Costs one redundant POST next load; the endpoint is
    // idempotent, so nothing is harmed.
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
