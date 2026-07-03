import { test, expect } from '@playwright/test'
import {
  CATAN_GAME,
  EMPTY_GAMES_PAGE,
  MUNCHKIN_GAME,
  RECENT_GAMES_PAGE,
  mockGameListRoutes,
} from './helpers/routes'

test.beforeEach(async ({ page }) => {
  await mockGameListRoutes(page)
})

test('shows games from mocked API', async ({ page }) => {
  await page.goto('/select-game')
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
})

test('shows "Recently Used" section when recent games are returned', async ({ page }) => {
  await page.goto('/select-game')
  await expect(page.getByText('Recently Used')).toBeVisible()
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
})

test('hides "Recently Used" section when no recent games', async ({ page }) => {
  await page.route('**/api/recent-games', (route) =>
    route.fulfill({ json: { ...RECENT_GAMES_PAGE, data: [] } })
  )
  await page.goto('/select-game')
  await expect(page.getByText('Recently Used')).not.toBeVisible()
})

test('clicking a game navigates to its chat page', async ({ page }) => {
  await page.goto('/select-game')
  // There may be multiple cards for Munchkin (recent + all games), click the first.
  await page.getByText(MUNCHKIN_GAME.name).first().click()
  await expect(page).toHaveURL(`/chat/${MUNCHKIN_GAME.id}`)
})

test('shows empty state when no games available', async ({ page }) => {
  await page.route('**/api/games?*', (route) =>
    route.fulfill({ json: EMPTY_GAMES_PAGE })
  )
  await page.goto('/select-game')
  await expect(page.getByText('No games available.')).toBeVisible()
})

test('shows error state when API fails', async ({ page }) => {
  // A real application error is a JSON 5xx (from FastAPI). It is surfaced
  // immediately, not retried as a transient cold-start failure. (A plain-text
  // 5xx would instead be treated as a Cloud Run cold start — see cold-start.spec.ts.)
  await page.route('**/api/games?*', (route) =>
    route.fulfill({ status: 500, contentType: 'application/json', json: { detail: 'boom' } })
  )
  await page.goto('/select-game')
  // The hook sets error state when the response is not ok; SelectGamePage renders it.
  await expect(page.getByText(/failed to fetch games/i)).toBeVisible()
})

// ---------------------------------------------------------------------------
// Cache: in-memory query cache
// ---------------------------------------------------------------------------

test('games are visible from in-memory cache after SPA navigation away and back', async ({ page }) => {
  await mockGameListRoutes(page)
  await page.goto('/select-game')
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()

  // Navigate away via SPA (click a game card).
  await page.getByText(MUNCHKIN_GAME.name).first().click()
  await expect(page).toHaveURL(`/chat/${MUNCHKIN_GAME.id}`)

  // Block the API so any fresh fetch would fail.
  await page.route('**/api/games?*', (route) => route.fulfill({ status: 500 }))
  await page.route('**/api/recent-games', (route) => route.fulfill({ status: 500 }))

  // Navigate back — React Query serves cached data without hitting the API.
  await page.goBack()
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
})

test('games are visible from localStorage cache after full page reload', async ({ page }) => {
  await mockGameListRoutes(page)
  await page.goto('/select-game')
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()

  // Wait for the persister to write to localStorage (throttleTime:0 is async via setTimeout).
  await page.waitForFunction(() => {
    const raw = window.localStorage.getItem('boardbarian-cache-v1')
    return raw !== null && raw.includes('Munchkin')
  })

  // Block API before reloading — the persisted localStorage cache should serve data.
  await page.route('**/api/games?*', (route) => route.fulfill({ status: 500 }))
  await page.route('**/api/recent-games', (route) => route.fulfill({ status: 500 }))

  await page.reload()
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
})
