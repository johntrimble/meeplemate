import { useEffect, useState } from 'react'

/**
 * How long a loading indicator must be visible before we assume the backend is
 * likely cold-starting and switch to a reassuring message. Warm requests
 * resolve in well under a second; a Cloud Run cold start takes ~26s, so a few
 * seconds is long enough to avoid flashing the message on normal slowness while
 * still reassuring the user well before the cold start finishes.
 */
export const SLOW_LOADING_MS = 4000

/**
 * Returns `false`, then `true` once `delayMs` has elapsed since mount.
 *
 * Loading indicators are only rendered while a request is in flight, so mount
 * == "loading started". A cold-started request stays pending the whole time
 * (fetchWithRetry retries internally without resolving), so the indicator stays
 * mounted and this flips to `true` mid-wait.
 */
export function useSlowLoading(delayMs: number = SLOW_LOADING_MS): boolean {
  const [isSlow, setIsSlow] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setIsSlow(true), delayMs)
    return () => clearTimeout(timer)
  }, [delayMs])

  return isSlow
}
