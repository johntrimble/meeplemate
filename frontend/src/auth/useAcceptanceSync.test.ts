import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { postAcceptance } from './useAcceptanceSync'
import { readAcceptance, writeAcceptance } from '@/lib/legal'

const UID = 'user-1'

/**
 * Minimal in-memory localStorage — the suite runs in the `node` environment
 * (see vitest.config.ts), so there is no real one. Mirrors lib/legal.test.ts.
 */
function stubStorage() {
  const store = new Map<string, string>()
  const storage = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    clear: () => store.clear(),
  }
  vi.stubGlobal('localStorage', storage)
  return storage
}

/** A fetch that stays pending until released, so callers overlap on purpose. */
function deferredFetch() {
  const calls: RequestInit[] = []
  let release!: (res: Response) => void
  const pending = new Promise<Response>((resolve) => {
    release = resolve
  })
  const authFetch = (_url: string, options: RequestInit = {}) => {
    calls.push(options)
    return pending
  }
  return { authFetch, calls, release }
}

const ok = () => new Response(null, { status: 204 })

beforeEach(() => {
  stubStorage()
  vi.spyOn(console, 'error').mockImplementation(() => {})
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('postAcceptance', () => {
  it('collapses concurrent calls for the same uid into one request', async () => {
    // The case the `synced` flag cannot cover on its own: it is only written
    // when a request *settles*, so while one is outstanding localStorage still
    // reads "unsent" and a second caller has nothing telling it to stand down.
    // ConsentScreen fires one on click and useAcceptanceSync retries another, so
    // the two really can overlap - most easily against a cold backend, which is
    // precisely the situation this whole design exists for.
    writeAcceptance(UID)
    const { authFetch, calls, release } = deferredFetch()

    const first = postAcceptance(authFetch, UID)
    const second = postAcceptance(authFetch, UID)

    expect(calls).toHaveLength(1)

    release(ok())
    await Promise.all([first, second])

    expect(calls).toHaveLength(1)
    expect(readAcceptance(UID)?.synced).toBe(true)
  })

  it('allows a fresh request once the previous one has settled', async () => {
    // Deduplication must not become a permanent lock: a retry on a later load
    // has to be able to get through.
    writeAcceptance(UID)
    const first = deferredFetch()
    const started = postAcceptance(first.authFetch, UID)
    first.release(new Response(null, { status: 503 }))
    await started

    // 5xx leaves it pending on purpose - that is the retryable case.
    expect(readAcceptance(UID)?.synced).toBe(false)

    const second = deferredFetch()
    const retried = postAcceptance(second.authFetch, UID)
    second.release(ok())
    await retried

    expect(second.calls).toHaveLength(1)
    expect(readAcceptance(UID)?.synced).toBe(true)
  })

  it('does not let one account suppress another', async () => {
    // Keyed by uid, so switching accounts mid-flight can't swallow the second
    // account's acceptance.
    writeAcceptance(UID)
    writeAcceptance('user-2')
    const { authFetch, calls, release } = deferredFetch()

    const a = postAcceptance(authFetch, UID)
    const b = postAcceptance(authFetch, 'user-2')

    expect(calls).toHaveLength(2)
    release(ok())
    await Promise.all([a, b])
  })

  it('posts the stored versions, not the current constants', async () => {
    // A retry after a version bump has to say what the user actually agreed to.
    localStorage.setItem(
      `boardbarian-legal-v1:${UID}`,
      JSON.stringify({
        termsVersion: '1999-01-01',
        privacyVersion: '1999-01-01',
        synced: false,
      }),
    )
    const { authFetch, calls, release } = deferredFetch()

    const sent = postAcceptance(authFetch, UID)
    release(ok())
    await sent

    expect(JSON.parse(calls[0].body as string)).toEqual({
      terms_version: '1999-01-01',
      privacy_version: '1999-01-01',
    })
  })

  it('stops retrying after a 4xx, which would fail identically forever', async () => {
    writeAcceptance(UID)
    const { authFetch, release } = deferredFetch()

    const sent = postAcceptance(authFetch, UID)
    release(new Response(null, { status: 422 }))
    await sent

    expect(readAcceptance(UID)?.synced).toBe(true)
  })

  it('is a no-op when nothing is stored', async () => {
    const { authFetch, calls } = deferredFetch()
    await postAcceptance(authFetch, UID)
    expect(calls).toHaveLength(0)
  })
})
