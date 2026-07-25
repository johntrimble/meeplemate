import { test, expect } from '@playwright/test'
import {
  ASSISTANT_MESSAGE,
  MUNCHKIN_GAME,
  USER_MESSAGE,
  VERIFIED_QUOTE_MESSAGE,
  mockChatMessagesRoute,
  mockGameChatsRoute,
  mockGameRoute,
} from './helpers/routes'

const GAME_ID = MUNCHKIN_GAME.id
const CHAT_ID = 'test-chat-123'
const NEW_CHAT_URL = new RegExp(`/chat/${GAME_ID}/[0-9a-f-]{36}$`)

// A minimal Vercel AI UI message stream that just starts and finishes.
const FINISH_SSE = [
  'data: {"type":"start","messageId":"m1"}\n\n',
  'data: {"type":"finish"}\n\n',
  'data: [DONE]\n\n',
].join('')
const SSE_HEADERS = { 'Content-Type': 'text/event-stream', 'x-vercel-ai-ui-message-stream': 'v1' }

test.beforeEach(async ({ page }) => {
  await mockGameRoute(page)
  await mockGameChatsRoute(page)
})

// ---------------------------------------------------------------------------
// New chat (no chatId)
// ---------------------------------------------------------------------------

test('shows empty state for new chat', async ({ page }) => {
  await page.goto(`/chat/${GAME_ID}`)
  // Game name appears in the header and in the empty state — just assert at least one is visible
  await expect(page.getByText(MUNCHKIN_GAME.name).first()).toBeVisible()
  await expect(page.getByText('Suggested')).toBeVisible()
})

test('shows suggested questions in empty state', async ({ page }) => {
  await page.goto(`/chat/${GAME_ID}`)
  // At least one suggestion button should be visible
  const suggestions = page.locator('button').filter({ hasText: /\?/ })
  await expect(suggestions.first()).toBeVisible()
})

test('clicking a suggestion navigates to a new client-minted chat and shows the message', async ({ page }) => {
  // No create round-trip: the client mints the id and the chat is created on the
  // first stream. Navigation is immediate and the question shows as a message.
  await page.route('**/api/chats/*/stream', (route) =>
    route.fulfill({ status: 200, headers: SSE_HEADERS, body: FINISH_SSE }),
  )

  await page.goto(`/chat/${GAME_ID}`)
  const suggestion = page.locator('button').filter({ hasText: /\?/ }).first()
  const questionText = (await suggestion.textContent())!.trim()
  await suggestion.click()

  await expect(page).toHaveURL(NEW_CHAT_URL)
  await expect(page.getByText(questionText).last()).toBeVisible()
})

test('new chat: a terminal stream error keeps the message and shows a dismissible error', async ({ page }) => {
  // A JSON 5xx is a terminal app error (not a cold-start retry) — it surfaces at once.
  await page.route('**/api/chats/*/stream', (route) =>
    route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'boom' }) }),
  )

  await page.goto(`/chat/${GAME_ID}`)
  const suggestion = page.locator('button').filter({ hasText: /\?/ }).first()
  const questionText = (await suggestion.textContent())!.trim()
  await suggestion.click()

  await expect(page).toHaveURL(NEW_CHAT_URL)
  // Optimistic message stays; the error renders below it with a dismiss.
  await expect(page.getByText(questionText).last()).toBeVisible()
  await expect(page.getByText('Something went wrong. Please try again.')).toBeVisible()
  await page.getByRole('button', { name: 'Dismiss' }).click()
  await expect(page.getByText('Something went wrong. Please try again.')).not.toBeVisible()
})

// ---------------------------------------------------------------------------
// Existing chat (with chatId)
// ---------------------------------------------------------------------------

test('renders user and assistant messages', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
  await expect(page.getByText(USER_MESSAGE.content)).toBeVisible()
  await expect(page.getByText(ASSISTANT_MESSAGE.content)).toBeVisible()
})

test('copy button shows "Copied!" feedback on click', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  // The copy button has a sr-only label "Copy"
  const copyBtn = page.getByRole('button', { name: 'Copy' })
  await expect(copyBtn).toBeVisible()
  await copyBtn.click()

  // After clicking, the label changes to "Copied!" for 2 seconds
  await expect(page.getByRole('button', { name: 'Copied!' })).toBeVisible()
})

// ---------------------------------------------------------------------------
// Navigation via sidebar menu
// ---------------------------------------------------------------------------

test('menu "Select game" navigates to /select-game', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  await page.getByRole('button', { name: 'Open menu' }).click()
  await page.getByRole('button', { name: 'Select game' }).click()
  await expect(page).toHaveURL('/select-game')
})

test('menu "New Chat" navigates to new chat URL', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  await page.getByRole('button', { name: 'Open menu' }).click()
  await page.getByRole('button', { name: 'New Chat' }).click()
  await expect(page).toHaveURL(`/chat/${GAME_ID}`)
})

// ---------------------------------------------------------------------------
// Verified blockquote rendering
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rate limit error handling
// ---------------------------------------------------------------------------

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

test('shows rate limit error inline after a 429 response', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) =>
    route.fulfill({ status: 429, json: RATE_LIMIT_429_BODY })
  )
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  await page.getByPlaceholder('Ask anything').fill('How does combat work?')
  await page.getByRole('button', { name: 'Send' }).click()

  await expect(page.getByText("You've reached your 8H usage limit.")).toBeVisible()
  await expect(page.getByText('You can try again')).toBeVisible()
})

test('rate limit error: user message stays visible above error', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) =>
    route.fulfill({ status: 429, json: RATE_LIMIT_429_BODY })
  )
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  await page.getByPlaceholder('Ask anything').fill('How does combat work?')
  await page.getByRole('button', { name: 'Send' }).click()

  await expect(page.getByText('How does combat work?')).toBeVisible()
  await expect(page.getByText("You've reached your 8H usage limit.")).toBeVisible()
})

test('rate limit error: input remains enabled', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) =>
    route.fulfill({ status: 429, json: RATE_LIMIT_429_BODY })
  )
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  await page.getByPlaceholder('Ask anything').fill('How does combat work?')
  await page.getByRole('button', { name: 'Send' }).click()

  await expect(page.getByText("You've reached your 8H usage limit.")).toBeVisible()
  await expect(page.getByPlaceholder('Ask anything')).toBeEnabled()
})

test('rate limit error: dismiss clears the error and orphaned message', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)
  await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) =>
    route.fulfill({ status: 429, json: RATE_LIMIT_429_BODY })
  )
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  await page.getByPlaceholder('Ask anything').fill('How does combat work?')
  await page.getByRole('button', { name: 'Send' }).click()
  await expect(page.getByText("You've reached your 8H usage limit.")).toBeVisible()

  await page.getByRole('button', { name: 'Dismiss' }).click()
  await expect(page.getByText("You've reached your 8H usage limit.")).not.toBeVisible()
  await expect(page.getByText('How does combat work?')).not.toBeVisible()
})

test('rate limit error: sending a new message clears the error', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID)

  let callCount = 0
  await page.route(`**/api/chats/${CHAT_ID}/stream`, (route) => {
    callCount++
    if (callCount === 1) {
      route.fulfill({ status: 429, json: RATE_LIMIT_429_BODY })
    } else {
      route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream', 'x-vercel-ai-ui-message-stream': 'v1' },
        body: [
          'data: {"type":"start","messageId":"msg-new"}\n\n',
          'data: {"type":"finish"}\n\n',
          'data: [DONE]\n\n',
        ].join(''),
      })
    }
  })

  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)
  await page.getByPlaceholder('Ask anything').fill('How does combat work?')
  await page.getByRole('button', { name: 'Send' }).click()
  await expect(page.getByText("You've reached your 8H usage limit.")).toBeVisible()

  await page.getByPlaceholder('Ask anything').fill('What are the basic rules?')
  await page.getByRole('button', { name: 'Send' }).click()
  await expect(page.getByText("You've reached your 8H usage limit.")).not.toBeVisible()
  await expect(page.getByText('How does combat work?')).not.toBeVisible()
})

// ---------------------------------------------------------------------------
// Cache: sidebar chat list
// ---------------------------------------------------------------------------

const SIDEBAR_CHAT = { chat_id: 'cached-chat-1', title: 'How does loot work?' }

test('sidebar shows cached chats immediately on reopen when API is slow', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID, [])

  let fetchCount = 0
  await page.route(`**/api/games/${GAME_ID}/chats?*`, async (route) => {
    fetchCount++
    if (fetchCount >= 2) {
      // Delay subsequent fetches to prove the cache renders first.
      await new Promise((r) => setTimeout(r, 3000))
    }
    await route.fulfill({
      json: {
        pageInfo: { hasNextPage: false, endCursor: null },
        data: [SIDEBAR_CHAT],
      },
    })
  })

  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  // First open: fetches from API and populates cache.
  await page.getByRole('button', { name: 'Open menu' }).click()
  await expect(page.getByText(SIDEBAR_CHAT.title)).toBeVisible()
  await page.getByRole('button', { name: 'Close menu' }).click()

  // Second open: API is slow but cached data renders instantly.
  await page.getByRole('button', { name: 'Open menu' }).click()
  await expect(page.getByText(SIDEBAR_CHAT.title)).toBeVisible({ timeout: 500 })

  // The background refetch over cached data must stay silent — no loading
  // indicator (and therefore no cold-start hint) when we already have chats.
  await expect(page.getByText('Loading…')).not.toBeVisible()
  await expect(page.getByText('Waking up the server...')).not.toBeVisible()
})

// ---------------------------------------------------------------------------
// Optimistic: chat creation invalidates sidebar
// ---------------------------------------------------------------------------

test('sidebar reflects the new chat after the first message', async ({ page }) => {
  const NEW_CHAT_TITLE = 'First question about Munchkin'

  await page.route('**/api/chats/*/stream', (route) =>
    route.fulfill({ status: 200, headers: SSE_HEADERS, body: FINISH_SSE }),
  )
  // Once the first message finishes, the chat exists server-side; the sidebar fetch
  // returns it (onFinish invalidated the list).
  await page.route(`**/api/games/${GAME_ID}/chats?*`, (route) =>
    route.fulfill({
      json: {
        pageInfo: { hasNextPage: false, endCursor: null },
        data: [{ chat_id: 'created-chat', title: NEW_CHAT_TITLE }],
      },
    }),
  )

  await page.goto(`/chat/${GAME_ID}`)

  // Submit a suggestion — navigates immediately; the chat is created on the stream.
  await page.locator('button').filter({ hasText: /\?/ }).first().click()
  await expect(page).toHaveURL(NEW_CHAT_URL)

  // Open sidebar — chat list should include the newly created chat.
  await page.getByRole('button', { name: 'Open menu' }).click()
  await expect(page.getByText(NEW_CHAT_TITLE)).toBeVisible()
})

// ---------------------------------------------------------------------------
// Verified blockquote rendering
// ---------------------------------------------------------------------------

test('verified blockquote div attribute is preserved in rendered HTML', async ({ page }) => {
  await mockChatMessagesRoute(page, CHAT_ID, [USER_MESSAGE, VERIFIED_QUOTE_MESSAGE])
  await page.goto(`/chat/${GAME_ID}/${CHAT_ID}`)

  // The marker div must be in the DOM (it is empty so visibility is not asserted)
  const verifiedDiv = page.locator('div[data-quote-status="verified"]')
  await expect(verifiedDiv).toBeAttached()

  // The blockquote it marks must immediately follow in the DOM
  const blockquote = page.locator('div[data-quote-status="verified"] + [data-streamdown="blockquote"]')
  await expect(blockquote).toBeVisible()
  await expect(blockquote).toContainText('You may move up to three spaces')
})
