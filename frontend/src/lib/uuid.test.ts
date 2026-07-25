import { afterEach, describe, expect, it, vi } from 'vitest'
import { randomUUID } from './uuid'

const V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('randomUUID', () => {
  it('returns a well-formed v4 UUID (native path)', () => {
    expect(randomUUID()).toMatch(V4)
  })

  it('falls back to getRandomValues when randomUUID is unavailable (insecure context)', () => {
    // Simulate an insecure-context browser: no crypto.randomUUID, but getRandomValues works.
    const realGetRandomValues = globalThis.crypto.getRandomValues.bind(globalThis.crypto)
    vi.stubGlobal('crypto', {
      getRandomValues: realGetRandomValues,
      // randomUUID intentionally absent
    })

    const id = randomUUID()
    expect(id).toMatch(V4)
  })

  it('produces unique ids', () => {
    const ids = new Set(Array.from({ length: 1000 }, () => randomUUID()))
    expect(ids.size).toBe(1000)
  })
})
