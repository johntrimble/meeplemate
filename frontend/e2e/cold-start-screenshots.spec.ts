import { test, expect, type Page } from '@playwright/test'
import {
  MUNCHKIN_GAME,
  CATAN_GAME,
  mockChatMessagesRoute,
  mockGameRoute,
  acceptLegal,
} from './helpers/routes'

// Captures the "Waking up the server..." cold-start hint on every loading
// indicator we changed. Each case keeps the relevant request failing (the
// plain-text 500 Cloud Run returns during a cold start), so the loading
// indicator stays visible long enough to cross the slow threshold, then we
// screenshot it into frontend/screenshots/.

const NO_INSTANCE_BODY = 'The request was aborted because there was no available instance.'
const WAKING = 'Waking up the server...'
const GAME_ID = MUNCHKIN_GAME.id
const CHAT_ID = 'screenshot-chat'

// Get past the consent gate; these specs are about other behaviour.
test.beforeEach(async ({ page }) => {
  await acceptLegal(page)
})

// Use a phone-sized viewport - this is a mobile-first app.
test.use({ viewport: { width: 390, height: 844 } })

/** Always respond with the Cloud Run cold-start plain-text 500. */
function coldStart(page: Page, urlGlob: string) {
  return page.route(urlGlob, (route) =>
    route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY }),
  )
}

async function shoot(page: Page, name: string) {
  await expect(page.getByText(WAKING).first()).toBeVisible({ timeout: 15_000 })
  await page.screenshot({ path: `screenshots/cold-start-${name}.png` })
}

test('select-game: initial game list load', async ({ page }) => {
  await coldStart(page, '**/api/recent-games')
  await coldStart(page, '**/api/games?*')
  await page.goto('/select-game')
  await shoot(page, 'select-game-list')
})

test('select-game: load-more pagination', async ({ page }) => {
  await coldStart(page, '**/api/recent-games')
  // First page loads; the next page (cursor set) cold-starts.
  await page.route('**/api/games?*', (route) => {
    const url = new URL(route.request().url())
    if (url.searchParams.get('cursor')) {
      route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
    } else {
      route.fulfill({
        json: {
          pageInfo: { hasNextPage: true, startCursor: null, endCursor: 'cursor-1' },
          data: [MUNCHKIN_GAME, CATAN_GAME],
        },
      })
    }
  })
  await page.goto('/select-game')
  // The short first page leaves the infinite-scroll sentinel in view, which
  // triggers the next-page fetch (and its "Loading more…" indicator).
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  await shoot(page, 'select-game-load-more')
})

test('chat: full-page game load', async ({ page }) => {
  await coldStart(page, `**/api/games/${GAME_ID}`)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
  await shoot(page, 'chat-game-load')
})

test('chat: message history load', async ({ page }) => {
  await mockGameRoute(page)
  await page.route(`**/api/games/${GAME_ID}/chats?*`, (route) =>
    route.fulfill({ json: { pageInfo: { hasNextPage: false, endCursor: null }, data: [] } }),
  )
  await coldStart(page, `**/api/chats/${CHAT_ID}/messages`)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
  await shoot(page, 'chat-messages-load')
})

test('chat: sidebar chat list load (reopen after idle)', async ({ page }) => {
  await mockGameRoute(page)
  await mockChatMessagesRoute(page, CHAT_ID)
  // The sidebar's chat-list request cold-starts when reopened.
  await coldStart(page, `**/api/games/${GAME_ID}/chats?*`)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
  await page.getByRole('button', { name: 'Open menu' }).click()
  await shoot(page, 'chat-sidebar-load')
})
