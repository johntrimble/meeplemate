import { SLOW_LOADING_MS, useSlowLoading } from '@/hooks/useSlowLoading'

const DEFAULT_SLOW_TEXT = 'Waking up the server...'

interface LoadingLabelProps {
  /** The normal loading text, shown until the request looks slow. */
  children: React.ReactNode
  className?: string
  /** How long before switching to {@link slowText}. Defaults to {@link SLOW_LOADING_MS}. */
  slowAfterMs?: number
  /** Message shown once the load looks slow (likely a cold start). */
  slowText?: string
}

/**
 * A loading label that reassures the user during a Cloud Run cold start: it
 * shows its normal text first, then switches to "Waking up the server..." once
 * the request has been pending for a while. Reused across every loading
 * indicator so the slow-state message stays consistent.
 */
export function LoadingLabel({
  children,
  className,
  slowAfterMs = SLOW_LOADING_MS,
  slowText = DEFAULT_SLOW_TEXT,
}: LoadingLabelProps) {
  const isSlow = useSlowLoading(slowAfterMs)
  return <span className={className}>{isSlow ? slowText : children}</span>
}
