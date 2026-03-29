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
  await page.route('**/api/games?*', (route) =>
    route.fulfill({ status: 500, body: 'Internal Server Error' })
  )
  await page.goto('/select-game')
  // The hook sets error state when the response is not ok; SelectGamePage renders it.
  await expect(page.getByText(/failed to fetch games/i)).toBeVisible()
})
