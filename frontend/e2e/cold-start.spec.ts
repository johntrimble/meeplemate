import { test, expect } from '@playwright/test'
import {
  CATAN_GAME,
  EMPTY_GAMES_PAGE,
  GAMES_PAGE,
  MUNCHKIN_GAME,
  NO_INSTANCE_BODY,
  RECENT_GAMES_PAGE,
  mockChatMessagesRoute,
  mockGameChatsRoute,
  mockGameRoute,
  acceptLegal,
} from './helpers/routes'
import { readIdbCache } from './helpers/cache'

const GAME_ID = MUNCHKIN_GAME.id
const CHAT_ID = 'cold-start-chat'

// Get past the consent gate; these specs are about other behaviour.
test.beforeEach(async ({ page }) => {
  await acceptLegal(page)
})

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

test('REST: the loading skeleton stays visible through retries, then recovers', async ({ page }) => {
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: RECENT_GAMES_PAGE }))

  // Keep the games request failing (retrying) until we let it succeed, so the
  // loading skeleton remains visible throughout the cold start.
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

  const loading = page.getByRole('status', { name: 'Loading games' })
  await expect(loading).toBeVisible()
  await expect.poll(() => calls, { timeout: 15_000 }).toBeGreaterThanOrEqual(2)
  await expect(loading).toBeVisible()

  // Let the "cold start" finish and confirm the UI recovers with no error.
  allowSuccess = true
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible({ timeout: 15_000 })
  await expect(loading).not.toBeVisible()
  await expect(page.getByText(/failed to fetch games/i)).not.toBeVisible()
  expect(calls).toBeGreaterThanOrEqual(2)
})

// ---------------------------------------------------------------------------
// Static CDN snapshot (`/games.json`) - instant first paint, backend still cold
// ---------------------------------------------------------------------------

test('static seed: game list paints from /games.json while /api/games is cold', async ({ page }) => {
  // Fresh context => empty IndexedDB (no persisted cache), so the only way the
  // grid can appear is the static-snapshot seed.
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: EMPTY_GAMES_PAGE }))

  // The static snapshot the deploy publishes to the CDN, served same-origin.
  await page.route('**/games.json', (route) => route.fulfill({ json: GAMES_PAGE }))

  // The live endpoint stays "cold" the whole time: never resolves successfully,
  // so any rendered game must have come from the seed, not this request.
  let apiCalls = 0
  await page.route('**/api/games?*', (route) => {
    apiCalls++
    route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
  })

  await page.goto('/select-game')

  // Catan only comes from the all-games grid (not "Recently Used", which is
  // empty here), so its presence proves the seed rendered the catalog.
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  // Never shows the cold-start loading state and never errors - it's instant.
  await expect(page.getByRole('status', { name: 'Loading games' })).not.toBeVisible()
  await expect(page.getByText(/failed to fetch games/i)).not.toBeVisible()

  // The seed does NOT suppress revalidation: the live request still fires (and
  // in production that's what warms the backend for the first question).
  expect(apiCalls).toBeGreaterThanOrEqual(1)
})

test('static seed: chat page paints from the seeded catalog while /api/games/{id} is cold', async ({
  page,
}) => {
  // Fresh context => empty IndexedDB, so the seed is the only possible source.
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: EMPTY_GAMES_PAGE }))
  await page.route('**/games.json', (route) => route.fulfill({ json: GAMES_PAGE }))
  await page.route('**/api/games?*', (route) =>
    route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
  )
  await mockGameChatsRoute(page, GAME_ID)

  // The detail endpoint never succeeds, so anything rendered came from the
  // catalog seed - the whole point of the fix.
  let detailCalls = 0
  await page.route(`**/api/games/${GAME_ID}`, (route) => {
    detailCalls++
    route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
  })

  // Deep link straight to the chat page: no prior visit to /select-game, so
  // this is the mount-order case CacheGate guarantees by awaiting the seed
  // before it renders children.
  await page.goto(`/chat/${GAME_ID}`)

  // Before the fix this was a fullscreen "Loading…" for the whole cold start.
  await expect(page.getByPlaceholder('Ask anything')).toBeVisible()
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
  await expect(page.getByText('Loading…')).not.toBeVisible()

  // Seeding does not suppress revalidation against the authoritative endpoint.
  await expect.poll(() => detailCalls).toBeGreaterThanOrEqual(1)
})

test('warm cache: chat page paints from the persisted catalog, without refetching the snapshot', async ({
  page,
}) => {
  // 1. First visit: load the catalog normally and wait for the persister to
  //    write it to IndexedDB (and for the reconcile to record this uid).
  await page.route('**/api/recent-games', (route) => route.fulfill({ json: EMPTY_GAMES_PAGE }))
  await page.route('**/api/games?*', (route) => route.fulfill({ json: GAMES_PAGE }))
  await mockGameChatsRoute(page, GAME_ID)

  await page.goto('/select-game')
  await expect(page.getByText(CATAN_GAME.name).first()).toBeVisible()
  await expect.poll(() => readIdbCache(page), { timeout: 10_000 }).toContain(MUNCHKIN_GAME.name)

  // 2. Everything the page could fetch now stays cold, including the static
  //    snapshot - so only the RESTORED cache can supply the game.
  let snapshotCalls = 0
  await page.route('**/games.json', (route) => {
    snapshotCalls++
    route.fulfill({ status: 404, contentType: 'text/plain', body: 'not found' })
  })
  await page.route('**/api/games?*', (route) =>
    route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
  )
  let detailCalls = 0
  await page.route(`**/api/games/${GAME_ID}`, (route) => {
    detailCalls++
    route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
  })

  // 3. Deep link into the chat page on a fresh load (same uid => reconcile is a
  //    no-op, so the restored catalog survives).
  await page.goto(`/chat/${GAME_ID}`)

  await expect(page.getByPlaceholder('Ask anything')).toBeVisible()
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
  await expect(page.getByText('Loading…')).not.toBeVisible()

  // A warm user pays nothing for the seed: `seedGamesFromStatic` short-circuits
  // on the restored `['games']` entry before ever fetching the snapshot.
  expect(snapshotCalls).toBe(0)
  // Still revalidates the detail endpoint in the background.
  await expect.poll(() => detailCalls).toBeGreaterThanOrEqual(1)
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

// Carries real text, unlike SSE_STREAM, so a test can watch the indicator give
// way to an actual answer.
const ANSWER_STREAM = [
  'data: {"type":"start","messageId":"msg-cold"}\n\n',
  'data: {"type":"text-start","id":"t1"}\n\n',
  'data: {"type":"text-delta","id":"t1","delta":"Forty-two."}\n\n',
  'data: {"type":"text-end","id":"t1"}\n\n',
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

  // Issue #107. The sibling test above proves the retry loop recovers; this one
  // proves the user is told anything is happening WHILE it grinds. Deliberately
  // exercises the real `fetchWithRetry` loop rather than a held-open stream: the
  // held-stream test in chat.spec.ts reproduces the same observable state, but
  // only this one would catch an indicator that survives a pending request yet
  // gets torn down by a failed-and-retried attempt.
  test('shows the thinking indicator throughout cold-start retries', async ({ page }) => {
    let cold = true
    let calls = 0
    await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) => {
      calls++
      if (cold) {
        return route.fulfill({ status: 500, contentType: 'text/plain', body: NO_INSTANCE_BODY })
      }
      return route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream', 'x-vercel-ai-ui-message-stream': 'v1' },
        body: ANSWER_STREAM,
      })
    })

    await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
    await page.getByPlaceholder('Ask anything').fill('How does combat work?')
    await page.getByRole('button', { name: 'Send' }).click()

    // Still up after several failed attempts - not just for the first one.
    await expect.poll(() => calls).toBeGreaterThanOrEqual(3)
    await expect(page.getByRole('status')).toContainText('Thinking...')
    await expect(page.getByText('Something went wrong. Please try again.')).not.toBeVisible()

    // The instance comes up: the answer arrives and the indicator stands down.
    cold = false
    await expect(page.getByText('Forty-two.')).toBeVisible()
    await expect(page.getByRole('status')).toHaveCount(0)
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
