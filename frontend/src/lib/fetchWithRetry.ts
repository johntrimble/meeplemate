// ---------------------------------------------------------------------------
// Cold-start-aware fetch retry
// ---------------------------------------------------------------------------
//
// Cloud Run cold starts (~26s) cause transient request failures: the first
// request is queued and eventually succeeds, but requests that arrive while an
// instance is booting are rejected by the Cloud Run proxy. Those failures show
// up in the browser as either:
//
//   - a `TypeError` (an aborted CORS *preflight* - JS can't see the status/body), or
//   - an HTTP 429/500/503 with a *plain-text* body ("...no available instance.").
//
// Both are transient and resolve once the instance is warm, so we retry them
// with exponential backoff + jitter for ~90s. Genuine application errors (HTTP
// 5xx with a JSON body) and rate-limit responses (429 with a JSON body) are
// returned unchanged so existing handling still applies.
//
// See GitHub issue #48 for the full Cloud Run error taxonomy.

export const RETRY_WINDOW_MS = 90_000
export const REQUEST_TIMEOUT_MS = 90_000

const BACKOFF_BASE_MS = 500
const BACKOFF_CAP_MS = 8_000

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export type RetryReason = 'network' | 'infra'

export type ResponseClass = 'transient-infra' | 'app-error' | 'ok'

export interface RetryOptions {
  /** Called on each retry (after the first failure), before the backoff sleep. */
  onRetry?: (info: { attempt: number; reason: RetryReason; delayMs: number }) => void
  /** Called exactly once when the request finally settles (success or failure). */
  onSettle?: () => void
  /** Total time to keep retrying transient failures. Defaults to {@link RETRY_WINDOW_MS}. */
  retryWindowMs?: number
  /** Per-attempt time-to-headers timeout. Defaults to {@link REQUEST_TIMEOUT_MS}. */
  requestTimeoutMs?: number
}

/**
 * Classify an HTTP response. The `Content-Type` is the key signal:
 *   - non-JSON 429/503/5xx  -> Cloud Run infra abort (retry)
 *   - JSON 5xx              -> real FastAPI error (surface, don't retry)
 *   - everything else       -> return as-is (incl. JSON 429 rate limits, 401/404)
 */
export function classifyResponse(res: Response): ResponseClass {
  const contentType = res.headers.get('content-type') ?? ''
  const isJson = contentType.includes('application/json')

  if (!isJson && (res.status === 429 || res.status === 503 || res.status >= 500)) {
    return 'transient-infra'
  }
  if (isJson && res.status >= 500) {
    return 'app-error'
  }
  return 'ok'
}

function backoffDelay(attempt: number): number {
  // Exponential backoff with full jitter.
  const ceiling = Math.min(BACKOFF_CAP_MS, BACKOFF_BASE_MS * 2 ** attempt)
  return Math.random() * ceiling
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason)
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(signal.reason)
      },
      { once: true },
    )
  })
}

/**
 * Fetch with cold-start retries. Wraps any `fetch`-like function, respecting an
 * optional caller `AbortSignal` (in `init.signal`) - a caller abort propagates
 * immediately and is never retried.
 *
 * The per-attempt timeout governs only *getting the Response* (headers). Once
 * headers arrive the body/stream is left untouched, so a queued cold-start
 * request is never cut off and an in-progress stream is never interrupted.
 */
export async function fetchWithRetry(
  fetchImpl: FetchLike,
  input: RequestInfo | URL,
  init: RequestInit = {},
  opts: RetryOptions = {},
): Promise<Response> {
  const retryWindowMs = opts.retryWindowMs ?? RETRY_WINDOW_MS
  const requestTimeoutMs = opts.requestTimeoutMs ?? REQUEST_TIMEOUT_MS
  const callerSignal = init.signal ?? undefined
  const start = Date.now()
  let attempt = 0

  try {
    for (;;) {
      // Per-attempt controller: our own timeout, merged with the caller's signal.
      const controller = new AbortController()
      const timer = setTimeout(
        () => controller.abort(new DOMException('Request timed out', 'TimeoutError')),
        requestTimeoutMs,
      )
      const onCallerAbort = () => controller.abort(callerSignal?.reason)
      if (callerSignal) {
        if (callerSignal.aborted) controller.abort(callerSignal.reason)
        else callerSignal.addEventListener('abort', onCallerAbort, { once: true })
      }

      let response: Response | undefined
      let caught: unknown
      try {
        response = await fetchImpl(input, { ...init, signal: controller.signal })
      } catch (err) {
        caught = err
      } finally {
        clearTimeout(timer)
        callerSignal?.removeEventListener('abort', onCallerAbort)
      }

      // A caller abort is intentional cancellation - surface it, never retry.
      if (callerSignal?.aborted) {
        if (caught !== undefined) throw caught
        return response as Response
      }

      let reason: RetryReason | null = null
      if (caught !== undefined) {
        // A `TypeError` is a network-level failure (includes failed CORS
        // preflight). Our own per-attempt timeout aborts `controller` too;
        // treat that as transient as well.
        const timedOut = controller.signal.aborted
        if (caught instanceof TypeError || timedOut) reason = 'network'
        else throw caught
      } else if (classifyResponse(response!) === 'transient-infra') {
        reason = 'infra'
      }

      // Not transient -> return the response as-is ('ok' or 'app-error').
      if (reason === null) return response!

      // Transient: retry until the budget is exhausted, then surface the last
      // response/error so the request doesn't hang forever.
      if (Date.now() - start >= retryWindowMs) {
        if (response) return response
        throw caught
      }

      const delayMs = backoffDelay(attempt)
      opts.onRetry?.({ attempt, reason, delayMs })
      await sleep(delayMs, callerSignal)
      attempt += 1
    }
  } finally {
    opts.onSettle?.()
  }
}
