import { test, expect } from '@playwright/test'
import { MUNCHKIN_GAME, acceptLegal, mockGameChatsRoute, mockGameRoute } from './helpers/routes'

const GAME_ID = MUNCHKIN_GAME.id
const NEW_CHAT_URL = new RegExp(`/chat/${GAME_ID}/[0-9a-f-]{36}$`)

// Get past the consent gate; these specs are about other behaviour.
test.beforeEach(async ({ page }) => {
  await acceptLegal(page)
})

// A minimal Vercel AI UI message stream that renders one line of text then finishes.
const ANSWER_SSE = [
  'data: {"type":"start","messageId":"m1"}\n\n',
  'data: {"type":"text-start","id":"t1"}\n\n',
  'data: {"type":"text-delta","id":"t1","delta":"Forty-two."}\n\n',
  'data: {"type":"text-end","id":"t1"}\n\n',
  'data: {"type":"finish"}\n\n',
  'data: [DONE]\n\n',
].join('')
const SSE_HEADERS = { 'Content-Type': 'text/event-stream', 'x-vercel-ai-ui-message-stream': 'v1' }

// The fix for the prod report: on a NEW chat, clicking a suggested question navigates
// into the chat view and shows the user's message IMMEDIATELY — even while the backend
// is still cold — instead of freezing on the empty state until a create round-trip returns.
test('new-chat suggestion loads into the chat instantly while the backend is cold', async ({ page }) => {
  await mockGameRoute(page)
  await mockGameChatsRoute(page)

  // The history fetch must NOT happen for a brand-new chat (it would 404 and, worse,
  // block on the cold start). Track it to prove we skip it.
  let messagesFetched = false
  await page.route('**/api/chats/*/messages', (route) => {
    messagesFetched = true
    return route.fulfill({ json: [] })
  })

  // Hold the stream pending to simulate the Cloud Run cold start.
  let releaseStream!: () => void
  const streamHeld = new Promise<void>((r) => { releaseStream = r })
  await page.route('**/api/chats/*/stream', async (route) => {
    await streamHeld
    await route.fulfill({ status: 200, headers: SSE_HEADERS, body: ANSWER_SSE })
  })

  await page.goto(`/chat/${GAME_ID}`)
  const question = page.locator('button').filter({ hasText: /\?/ }).first()
  const questionText = (await question.textContent())!.trim()
  await question.click()

  // FIXED: immediate navigation to a client-minted chat id, and the user's question is
  // already on screen as a message while the (cold) response is still pending.
  await expect(page).toHaveURL(NEW_CHAT_URL)
  await expect(page.getByText(questionText).last()).toBeVisible()
  await expect(page.getByText('Suggested')).not.toBeVisible()

  // The cold start finishes → the answer streams in.
  releaseStream()
  await expect(page.getByText('Forty-two.')).toBeVisible()

  // We never blocked on a history fetch for the new chat.
  expect(messagesFetched).toBe(false)
})
