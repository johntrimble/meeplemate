import { useSyncExternalStore } from 'react'
import type { RetryOptions } from './fetchWithRetry'

// ---------------------------------------------------------------------------
// Cold-start signal
// ---------------------------------------------------------------------------
//
// Tracks how many in-flight requests are currently retrying through a cold
// start (see fetchWithRetry). The UI subscribes via `useColdStart()` to show a
// low-key "Waking up the server…" hint — not an error — while a slow Cloud Run
// instance boots.

let activeCount = 0
const listeners = new Set<() => void>()

function emit() {
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function getSnapshot(): boolean {
  return activeCount > 0
}

/** Returns `true` while at least one request is retrying through a cold start. */
export function useColdStart(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false)
}

/**
 * Retry callbacks that report cold-start activity to the store. Pass the result
 * straight into {@link fetchWithRetry}'s options. Marks the request active on
 * its first retry and clears it (idempotently) when it settles.
 */
export function createColdStartTracker(): Pick<RetryOptions, 'onRetry' | 'onSettle'> {
  let active = false
  return {
    onRetry: () => {
      if (active) return
      active = true
      activeCount += 1
      emit()
    },
    onSettle: () => {
      if (!active) return
      active = false
      activeCount -= 1
      emit()
    },
  }
}
