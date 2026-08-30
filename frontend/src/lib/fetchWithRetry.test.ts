import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fetchWithRetry } from './fetchWithRetry'

/** The plain-text body Cloud Run's proxy returns while an instance boots. */
const NO_INSTANCE_BODY =
  'The request was aborted because there was no available instance.'

// ---------------------------------------------------------------------------
// Retry-window termination
// ---------------------------------------------------------------------------
//
// The property under test is that the loop STOPS. Everything else about a cold
// start is designed to hide it from the user - the request is retried silently
// and the "Thinking..." indicator stays up throughout - which means a loop that
// never gave up would show no error, ever: the user would watch that indicator
// forever with nothing to click. Nothing in the e2e suite can catch that, since
// proving it terminates means letting the full 90s window elapse.
//
// Fake timers make the wall time free. `advanceTimersByTimeAsync` is required
// rather than the sync variant: the loop awaits between sleeps, so the promise
// chain has to be flushed as the clock moves.

/** A response the classifier treats as transient infra (non-JSON 5xx). */
function coldStartResponse() {
  return new Response(NO_INSTANCE_BODY, {
    status: 500,
    headers: { 'content-type': 'text/plain; charset=utf-8' },
  })
}

/** A real application failure: 5xx, but with a JSON body. */
function appErrorResponse() {
  return new Response(JSON.stringify({ detail: 'boom' }), {
    status: 500,
    headers: { 'content-type': 'application/json' },
  })
}

// Short window so the test is quick to reason about, and a per-attempt timeout
// that can never fire (the stub resolves at once) so it stays out of the way.
const OPTS = { retryWindowMs: 5_000, requestTimeoutMs: 1_000 }

describe('fetchWithRetry', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    // Pin the jitter so the backoff schedule is deterministic: half of each
    // ceiling, i.e. 250ms, 500ms, 1s, 2s, ... until the window is spent.
    vi.spyOn(Math, 'random').mockReturnValue(0.5)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('gives up once the retry window expires and returns the last transient response', async () => {
    const fetchImpl = vi.fn().mockImplementation(() => Promise.resolve(coldStartResponse()))

    const settled = vi.fn()
    const promise = fetchWithRetry(fetchImpl, '/api/x', {}, { ...OPTS, onSettle: settled })
    await vi.advanceTimersByTimeAsync(60_000)
    const response = await promise

    // The caller gets the infra failure back rather than hanging forever.
    expect(response.status).toBe(500)
    // It really did retry before giving up, and it really did stop.
    expect(fetchImpl.mock.calls.length).toBeGreaterThan(1)
    expect(settled).toHaveBeenCalledTimes(1)

    // No further attempts once the window is spent.
    const attempts = fetchImpl.mock.calls.length
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetchImpl.mock.calls.length).toBe(attempts)
  })

  it('gives up once the retry window expires and re-throws the last transient error', async () => {
    // A TypeError is what an aborted CORS preflight looks like from JS.
    const fetchImpl = vi.fn().mockImplementation(() => Promise.reject(new TypeError('Failed to fetch')))

    const promise = fetchWithRetry(fetchImpl, '/api/x', {}, OPTS)
    const assertion = expect(promise).rejects.toThrow(TypeError)
    await vi.advanceTimersByTimeAsync(60_000)
    await assertion

    expect(fetchImpl.mock.calls.length).toBeGreaterThan(1)
  })

  it('retries a transient failure and returns the eventual success', async () => {
    const ok = new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })
    const fetchImpl = vi
      .fn()
      .mockImplementationOnce(() => Promise.resolve(coldStartResponse()))
      .mockImplementationOnce(() => Promise.resolve(ok))

    const onRetry = vi.fn()
    const promise = fetchWithRetry(fetchImpl, '/api/x', {}, { ...OPTS, onRetry })
    await vi.advanceTimersByTimeAsync(60_000)
    const response = await promise

    expect(response.status).toBe(200)
    expect(fetchImpl).toHaveBeenCalledTimes(2)
    expect(onRetry).toHaveBeenCalledTimes(1)
    expect(onRetry.mock.calls[0][0]).toMatchObject({ attempt: 0, reason: 'infra' })
  })

  it('does not retry a JSON 5xx - that is a real application error', async () => {
    const fetchImpl = vi.fn().mockImplementation(() => Promise.resolve(appErrorResponse()))

    const promise = fetchWithRetry(fetchImpl, '/api/x', {}, OPTS)
    await vi.advanceTimersByTimeAsync(60_000)
    const response = await promise

    expect(response.status).toBe(500)
    // Surfaced on the first attempt, so the caller's error handling runs at once
    // instead of after a 90s wait.
    expect(fetchImpl).toHaveBeenCalledTimes(1)
  })
})
