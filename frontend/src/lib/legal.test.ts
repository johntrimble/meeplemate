import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import {
  TERMS_VERSION,
  PRIVACY_VERSION,
  acceptanceState,
  readAcceptance,
  writeAcceptance,
} from './legal'

const UID = 'user-1'
const KEY = `boardbarian-legal-v1:${UID}`

/**
 * Minimal in-memory localStorage. The suite runs in the `node` environment (see
 * vitest.config.ts), so there is no real one, and stubbing it explicitly also
 * makes the "storage is blocked" cases easy to drive.
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

let storage: ReturnType<typeof stubStorage>

beforeEach(() => {
  storage = stubStorage()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('acceptanceState', () => {
  it('prompts when nothing is stored', () => {
    // A new user, but also a returning one on a new device - indistinguishable
    // without asking the server, and we deliberately don't ask.
    expect(acceptanceState(UID)).toBe('none')
  })

  it('lets a current acceptance through', () => {
    writeAcceptance(UID)
    expect(acceptanceState(UID)).toBe('current')
  })

  it('re-prompts when the terms version has moved on', () => {
    storage.setItem(
      KEY,
      JSON.stringify({ termsVersion: '1999-01-01', privacyVersion: PRIVACY_VERSION }),
    )
    expect(acceptanceState(UID)).toBe('outdated')
  })

  it('re-prompts when only the privacy version has moved on', () => {
    // Versioned separately, but consent is collected for the pair.
    storage.setItem(
      KEY,
      JSON.stringify({ termsVersion: TERMS_VERSION, privacyVersion: '1999-01-01' }),
    )
    expect(acceptanceState(UID)).toBe('outdated')
  })

  it("does not carry one account's acceptance over to another", () => {
    writeAcceptance(UID)
    expect(acceptanceState('user-2')).toBe('none')
  })

  it('makes the decision without touching the network', () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    writeAcceptance(UID)

    acceptanceState(UID)
    acceptanceState('someone-else')

    // The whole point of the design: the gate must never sit in front of a
    // cold-starting backend just to find out whether to render.
    expect(fetchSpy).not.toHaveBeenCalled()
  })
})

describe('readAcceptance', () => {
  it('treats malformed JSON as not accepted', () => {
    storage.setItem(KEY, 'not json')
    expect(readAcceptance(UID)).toBeNull()
    expect(acceptanceState(UID)).toBe('none')
  })

  it('treats a wrong-shaped entry as not accepted', () => {
    storage.setItem(KEY, JSON.stringify({ termsVersion: 42 }))
    expect(readAcceptance(UID)).toBeNull()
  })

  it('treats unreadable storage as not accepted rather than throwing', () => {
    // Safari private mode and hardened profiles both surface as a throw here.
    // Failing closed is the safe direction: prompting again is harmless,
    // silently letting someone in is not.
    vi.stubGlobal('localStorage', {
      ...storage,
      getItem: () => {
        throw new Error('storage blocked')
      },
    })
    expect(() => acceptanceState(UID)).not.toThrow()
    expect(acceptanceState(UID)).toBe('none')
  })
})

describe('writeAcceptance', () => {
  it('stores the versions currently in force', () => {
    writeAcceptance(UID)
    expect(readAcceptance(UID)).toEqual({
      termsVersion: TERMS_VERSION,
      privacyVersion: PRIVACY_VERSION,
    })
  })

  it('swallows a storage failure instead of breaking the accept flow', () => {
    // By this point the server has already recorded the acceptance, so the only
    // cost of not persisting locally is being prompted again next load.
    vi.stubGlobal('localStorage', {
      ...storage,
      setItem: () => {
        throw new Error('quota exceeded')
      },
    })
    expect(() => writeAcceptance(UID)).not.toThrow()
  })
})
