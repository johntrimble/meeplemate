import { test, expect } from '@playwright/test'
import {
  CATAN_GAME,
  GAMES_PAGE,
  MUNCHKIN_GAME,
  RECENT_GAMES_PAGE,
  mockChatMessagesRoute,
  mockGameChatsRoute,
  mockGameRoute,
} from './helpers/routes'

// The plain-text body the Cloud Run proxy returns while an instance cold-starts.
const NO_INSTANCE_BODY = 'The request was aborted because there was no available instance.'

const GAME_ID = MUNCHKIN_GAME.id
const CHAT_ID = 'cold-start-chat'

// ---------------------------------------------------------------------------
// REST requests (game list) - transparent retry through a cold start
// ---------------------------------------------------------------------------

test('REST: retries plain-text 500 cold-start failures then renders data', async ({ page }) => {
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: RECENT_GAMES_PAGE }))

  let calls = 0
  await page.route('**/api/games?*', (route) => {
    calls++
    if (calls <= 2) {
      // Cloud Run traffic-management abort: plain-text 500, not JSON.
      route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
    } else {
      route.fulfill({ json: GAMES_PAGE })
    }
  })

  await page.goto('/select-game')

  // Catan only comes from the games list (not "Recently Used"), so its
  // appearance proves the retried games-list request eventually succeeded.
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  await expect(page.getByText(/failed to fetch games/i)).not.toBeVisible()
  expect(calls).toBeGreaterThanOrEqual(3)
})

test('REST: retries a network abort (failed CORS preflight) then renders data', async ({ page }) => {
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: RECENT_GAMES_PAGE }))

  let calls = 0
  await page.route('**/api/games?*', (route) => {
    calls++
    if (calls === 1) {
      // Aborting the request surfaces as `TypeError: Failed to fetch` in the page.
      route.abort('failed')
    } else {
      route.fulfill({ json: GAMES_PAGE })
    }
  })

  await page.goto('/select-game')

  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  await expect(page.getByText(/failed to fetch games/i)).not.toBeVisible()
})

test('REST: the loading indicator switches to the cold-start hint, then recovers', async ({ page }) => {
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: RECENT_GAMES_PAGE }))

  // Keep the games request failing (retrying) until we let it succeed, so the
  // loading indicator stays visible long enough to cross the slow threshold.
  let allowSuccess = false
  let calls = 0
  await page.route('**/api/games?*', (route) => {
    calls++
    if (allowSuccess) {
      route.fulfill({ json: GAMES_PAGE })
    } else {
      route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
    }
  })

  await page.goto('/select-game')

  // The normal loading text shows first...
  await expect(page.getByText('Loading games…')).toBeVisible()
  // ...then, once it's been pending a while, it reassures that the server is waking up.
  await expect(page.getByText('Waking up the server...')).toBeVisible({ timeout: 15_000 })

  // Let the "cold start" finish and confirm the UI recovers with no error.
  allowSuccess = true
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText('Waking up the server...')).not.toBeVisible()
  await expect(page.getByText(/failed to fetch games/i)).not.toBeVisible()
  expect(calls).toBeGreaterThanOrEqual(2)
})

test('REST: a JSON 500 application error is NOT retried', async ({ page }) => {
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: RECENT_GAMES_PAGE }))

  let calls = 0
  await page.route('**/api/games?*', (route) => {
    calls++
    route.fulfill({ status: 500, contentType: 'application/json', json: { detail: 'boom' } })
  })

  await page.goto('/select-game')

  // The error surfaces immediately (no 90s retry loop).
  await expect(page.getByText(/failed to fetch games/i)).toBeVisible()
  // The request count stops growing - proving it was not retried as transient.
  const settled = calls
  await page.waitForTimeout(1500)
  expect(calls).toBe(settled)
})

// ---------------------------------------------------------------------------
// Chat stream - retry cold starts without disturbing rate-limit / app errors
// ---------------------------------------------------------------------------

const SSE_STREAM = [
  'data: {"type":"start","messageId":"msg-cold"}\n\n',
  'data: {"type":"finish"}\n\n',
  'data: [DONE]\n\n',
].join('')

const RATE_LIMIT_429_BODY = {
  detail: {
    error: 'rate_limit_exceeded',
    message: 'You have used your token quota for the 8H window. Your limit resets at 2030-01-01T12:00:00+00:00.',
    window: '8H',
    limit: 400_000,
    used: 400_000,
    resets_at: '2030-01-01T12:00:00+00:00',
  },
}

test.describe('chat stream', () => {
  test.beforeEach(async ({ page }) => {
    await mockGameRoute(page)
    await mockGameChatsRoute(page)
    await mockChatMessagesRoute(page, CHAT_ID)
  })

  test('retries a plain-text 500 cold-start failure, no error shown', async ({ page }) => {
    let calls = 0
    await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) => {
      calls++
      if (calls === 1) {
        route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
      } else {
        route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'text/event-stream', 'x-vercel-ai-ui-message-stream': 'v1' },
          body: SSE_STREAM,
        })
      }
    })

    await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
    await page.getByPlaceholder('Ask anything').fill('How does combat work?')
    await page.getByRole('button', { name: 'Send' }).click()

    await expect(page.getByText('Something went wrong. Please try again.')).not.toBeVisible()
    await expect.poll(() => calls).toBeGreaterThanOrEqual(2)
  })

  test('a rate-limit 429 (JSON) is surfaced, not retried', async ({ page }) => {
    let calls = 0
    await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) => {
      calls++
      route.fulfill({ status: 429, json: RATE_LIMIT_429_BODY })
    })

    await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
    await page.getByPlaceholder('Ask anything').fill('How does combat work?')
    await page.getByRole('button', { name: 'Send' }).click()

    await expect(page.getByText("You've reached your 8H usage limit.")).toBeVisible()
    const settled = calls
    await page.waitForTimeout(1500)
    expect(calls).toBe(settled)
  })
})
