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

test('clicking a suggestion creates a chat and navigates', async ({ page }) => {
  await page.route(`**/api/games/${GAME_ID}/chats`, (route) => {
    if (route.request().method() === 'POST') {
      route.fulfill({ json: { chat_id: CHAT_ID } })
    } else {
      route.continue()
    }
  })
  await mockChatMessagesRoute(page, CHAT_ID, [])

  await page.goto(`/chat/${GAME_ID}`)
  const suggestion = page.locator('button').filter({ hasText: /\?/ }).first()
  await suggestion.click()
  await expect(page).toHaveURL(`/chat/${GAME_ID}/${CHAT_ID}`)
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
