import { useCallback, useEffect, useState } from 'react'
import { acceptanceState, type AcceptanceState } from '@/lib/legal'

/**
 * Whether the signed-in user has accepted the documents currently in force.
 *
 * Reads localStorage synchronously on mount rather than fetching, so the gate
 * can decide during the first render and never blocks first paint on the
 * backend - see the note at the top of lib/legal.ts.
 *
 * localStorage isn't reactive, so acceptance is mirrored into React state and
 * `markAccepted` is what moves the gate along after a successful POST.
 */
export function useLegalAcceptance(uid: string | null) {
  const [state, setState] = useState<AcceptanceState>(() =>
    uid ? acceptanceState(uid) : 'none',
  )

  // Re-read when the account changes: acceptance is stored per uid, so one
  // user's agreement must never carry over to another in the same browser.
  useEffect(() => {
    setState(uid ? acceptanceState(uid) : 'none')
  }, [uid])

  const markAccepted = useCallback(() => setState('current'), [])

  return { state, markAccepted }
}
