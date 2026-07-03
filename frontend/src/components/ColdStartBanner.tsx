import { Loader2Icon } from 'lucide-react'
import { useColdStart } from '@/lib/coldStart'

/**
 * A low-key, fixed banner shown while a request is retrying through a Cloud Run
 * cold start. This is intentionally *not* an error — it just reassures the user
 * that a slow first request is still in progress while the backend wakes up.
 */
export function ColdStartBanner() {
  const isColdStart = useColdStart()
  if (!isColdStart) return null

  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 top-0 z-50 flex items-center justify-center gap-2 bg-muted/95 px-4 py-2 text-sm text-muted-foreground shadow-sm backdrop-blur"
    >
      <Loader2Icon className="size-4 animate-spin" />
      <span>Waking up the server…</span>
    </div>
  )
}
