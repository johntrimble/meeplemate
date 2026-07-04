import { test, expect, type Page } from '@playwright/test'
import { CATAN_GAME, MUNCHKIN_GAME, mockGameListRoutes } from './helpers/routes'

// The bypass auth user's uid (MockAuthProvider defaults to 'local-dev').
const BYPASS_UID = 'local-dev'
const CACHE_KEY = 'boardbarian-cache-v1'
const LAST_UID_KEY = 'boardbarian-last-uid'
const SENTINEL = 'GHOST-SENTINEL'

// A structurally valid persisted React Query client holding one "orphan" query
// the app never uses. Because nothing refetches or overwrites it, its survival
// in localStorage is a clean signal for whether the cache was cleared.
function seedBlob(): string {
  const now = Date.now()
  return JSON.stringify({
    buster: '',
    timestamp: now,
    clientState: {
      mutations: [],
      queries: [
        {
          queryKey: ['orphan-sentinel'],
          queryHash: '["orphan-sentinel"]',
          state: {
            data: SENTINEL,
            dataUpdateCount: 1,
            dataUpdatedAt: now,
            error: null,
            errorUpdateCount: 0,
            errorUpdatedAt: 0,
            fetchFailureCount: 0,
            fetchFailureReason: null,
            fetchMeta: null,
            isInvalidated: false,
            status: 'success',
            fetchStatus: 'idle',
          },
        },
      ],
    },
  })
}

async function seedCache(page: Page, lastUid: string) {
  await page.addInitScript(
    ([key, blob, uidKey, uid]) => {
      localStorage.setItem(key, blob)
      localStorage.setItem(uidKey, uid)
    },
    [CACHE_KEY, seedBlob(), LAST_UID_KEY, lastUid] as const,
  )
}

const readCache = (page: Page) => page.evaluate((k) => localStorage.getItem(k) ?? '', CACHE_KEY)
const readLastUid = (page: Page) => page.evaluate((k) => localStorage.getItem(k), LAST_UID_KEY)

test('clears the persisted cache when a different user loads the app', async ({ page }) => {
  await seedCache(page, 'someone-else') // last user differs from the bypass user
  await mockGameListRoutes(page)

  await page.goto('/select-game')

  // Gate lets the (now current) user through with fresh data...
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  // ...and the previous user's cached data was dropped.
  await expect.poll(() => readCache(page)).not.toContain(SENTINEL)
  expect(await readLastUid(page)).toBe(BYPASS_UID)
})

test('keeps the persisted cache when the same user loads the app', async ({ page }) => {
  await seedCache(page, BYPASS_UID) // same user as the bypass user
  await mockGameListRoutes(page)

  await page.goto('/select-game')

  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
  // Same identity -> warm cache preserved (orphan survives the reconcile).
  await expect.poll(() => readCache(page)).toContain(SENTINEL)
  expect(await readLastUid(page)).toBe(BYPASS_UID)
})
