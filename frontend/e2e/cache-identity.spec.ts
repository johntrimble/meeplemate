import { test, expect, type Page } from '@playwright/test'
import { CATAN_GAME, MUNCHKIN_GAME, mockGameListRoutes } from './helpers/routes'

// The bypass auth user's uid (MockAuthProvider defaults to 'local-dev').
const BYPASS_UID = 'local-dev'
const LAST_UID_KEY = 'boardbarian-last-uid'

// Read the persisted cache blob out of idb-keyval's store (raw IndexedDB, since
// the app doesn't expose idb-keyval globally). The persister stores the value
// as a serialized string under 'boardbarian-cache-v1'.
function readIdbCache(page: Page): Promise<string> {
  return page.evaluate(
    () =>
      new Promise<string>((resolve) => {
        const req = indexedDB.open('keyval-store')
        // Match idb-keyval's store so probing never pre-creates an incompatible DB.
        req.onupgradeneeded = () => req.result.createObjectStore('keyval')
        req.onsuccess = () => {
          let store: IDBObjectStore
          try {
            store = req.result.transaction('keyval', 'readonly').objectStore('keyval')
          } catch {
            resolve('')
            return
          }
          const g = store.get('boardbarian-cache-v1')
          g.onsuccess = () => resolve(typeof g.result === 'string' ? g.result : '')
          g.onerror = () => resolve('')
        }
        req.onerror = () => resolve('')
      }),
  )
}

const getLastUid = (page: Page) => page.evaluate((k) => localStorage.getItem(k), LAST_UID_KEY)
const setLastUid = (page: Page, v: string) => page.evaluate(([k, val]) => localStorage.setItem(k, val), [LAST_UID_KEY, v] as const)
const clearLastUid = (page: Page) => page.evaluate((k) => localStorage.removeItem(k), LAST_UID_KEY)

/** Load the game list and wait until it has been persisted to IndexedDB. */
async function loadAndPersist(page: Page) {
  await mockGameListRoutes(page)
  await page.goto('/select-game')
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  await expect.poll(() => readIdbCache(page), { timeout: 10_000 }).toContain('Munchkin')
}

/** Make the game endpoints fail with a JSON 500 (an app error, not retried) so
 * only the cache could supply data on the next load. */
async function blockApi(page: Page) {
  const fail = (route: import('@playwright/test').Route) =>
    route.fulfill({ status: 500, contentType: 'application/json', body: '{}' })
  await page.route('**/api/games?*', fail)
  await page.route('**/api/recent-games', fail)
}

test('keeps the persisted cache on reload for the same user', async ({ page }) => {
  await loadAndPersist(page)
  await blockApi(page)
  await page.reload()

  // Same identity -> cache restored from IndexedDB despite the blocked API.
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
})

test('clears the persisted cache when a different user reloads', async ({ page }) => {
  await loadAndPersist(page)
  await setLastUid(page, 'someone-else') // a different prior identity
  await blockApi(page)
  await page.reload()

  // Cache dropped on reconcile -> with the API blocked, no cached games remain.
  await expect(page.getByText(/failed to fetch games/i)).toBeVisible()
  await expect(page.getByText(MUNCHKIN_GAME.name)).not.toBeVisible()
  expect(await getLastUid(page)).toBe(BYPASS_UID)
})

test('clears a pre-existing cache that has no recorded owner (migration)', async ({ page }) => {
  await loadAndPersist(page)
  await clearLastUid(page) // cache present but no owner (pre-feature browser)
  await blockApi(page)
  await page.reload()

  await expect(page.getByText(/failed to fetch games/i)).toBeVisible()
  await expect(page.getByText(MUNCHKIN_GAME.name)).not.toBeVisible()
  expect(await getLastUid(page)).toBe(BYPASS_UID)
})
