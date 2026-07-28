import type { Page } from '@playwright/test'
import { PRIVACY_VERSION, TERMS_VERSION } from '../../src/lib/legal'

// ---------------------------------------------------------------------------
// Consent gate
// ---------------------------------------------------------------------------

/** The bypass auth user's uid (MockAuthProvider defaults to 'local-dev'). */
export const BYPASS_UID = 'local-dev'

/**
 * Pre-accept the current Terms and Privacy Policy for `uid`.
 *
 * Every authed route sits behind the consent gate in CacheGate, which blocks on
 * an empty localStorage, so specs about anything else have to get past it
 * first. Uses `addInitScript` so the entry is in place before the app boots on
 * every navigation in the test - including reloads.
 *
 * Deliberately writes the same shape `lib/legal.ts` writes, importing the
 * versions from it rather than hard-coding them, so a version bump doesn't
 * quietly turn every spec here into a test of the consent screen.
 */
export async function acceptLegal(page: Page, uid: string = BYPASS_UID) {
  await page.addInitScript(
    ([key, value]) => localStorage.setItem(key, value),
    [
      `boardbarian-legal-v1:${uid}`,
      JSON.stringify({ termsVersion: TERMS_VERSION, privacyVersion: PRIVACY_VERSION }),
    ] as const,
  )
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

export const MUNCHKIN_GAME = {
  id: 'munchkin',
  name: 'Munchkin',
  emoji: '⚔️',
  background_color: '#7f1d1d',
}

export const CATAN_GAME = {
  id: 'catan',
  name: 'Catan',
  emoji: '🏝️',
  background_color: '#92400e',
}

/**
 * The plain-text body the Cloud Run proxy returns while an instance cold-starts.
 * Paired with a non-JSON 500 it is what `fetchWithRetry` classifies as transient
 * infra and keeps retrying, i.e. a request that never resolves for the test.
 */
export const NO_INSTANCE_BODY =
  'The request was aborted because there was no available instance.'

export const EMPTY_GAMES_PAGE = {
  pageInfo: { hasNextPage: false, startCursor: null, endCursor: null },
  data: [],
}

export const GAMES_PAGE = {
  pageInfo: { hasNextPage: false, startCursor: null, endCursor: null },
  data: [MUNCHKIN_GAME, CATAN_GAME],
}

export const RECENT_GAMES_PAGE = {
  pageInfo: { hasNextPage: false, startCursor: null, endCursor: null },
  data: [MUNCHKIN_GAME],
}

export const EMPTY_CHATS_PAGE = {
  pageInfo: { hasNextPage: false, endCursor: null },
  data: [],
}

// A minimal UIMessage pair for testing chat rendering.
// The `content` field is required by the ai SDK's UIMessage type.

const VERIFIED_BLOCKQUOTE_CONTENT = [
  'Here is a verified rule:',
  '',
  '<div data-quote-status="verified"></div>',
  '',
  '> You may move up to three spaces on your turn.',
  '> ',
  '> (Movement Rules, p. 12)',
  '',
  'That concludes the rule.',
].join('\n')

export const VERIFIED_QUOTE_MESSAGE = {
  id: 'msg-assistant-verified',
  role: 'assistant',
  content: VERIFIED_BLOCKQUOTE_CONTENT,
  parts: [{ type: 'text', text: VERIFIED_BLOCKQUOTE_CONTENT }],
  feedback: null,
}

export const USER_MESSAGE = {
  id: 'msg-user-1',
  role: 'user',
  content: 'Can I use a card during combat?',
  parts: [{ type: 'text', text: 'Can I use a card during combat?' }],
}

export const ASSISTANT_MESSAGE = {
  id: 'msg-assistant-1',
  role: 'assistant',
  content: 'Yes, you can play most cards during combat.',
  parts: [{ type: 'text', text: 'Yes, you can play most cards during combat.' }],
  feedback: null,
}

// ---------------------------------------------------------------------------
// Route setup helpers
// ---------------------------------------------------------------------------

/** Mock the game list and recent games endpoints. */
export async function mockGameListRoutes(page: Page) {
  await page.route('**/api/recent-games', (route) =>
    route.fulfill({ json: RECENT_GAMES_PAGE })
  )
  await page.route('**/api/games?*', (route) =>
    route.fulfill({ json: GAMES_PAGE })
  )
}

/** Mock a single game detail endpoint. */
export async function mockGameRoute(page: Page, game = MUNCHKIN_GAME) {
  await page.route(`**/api/games/${game.id}`, (route) =>
    route.fulfill({ json: game })
  )
}

/** Mock the chat list for a game's sidebar. */
export async function mockGameChatsRoute(page: Page, gameId = 'munchkin') {
  await page.route(`**/api/games/${gameId}/chats?*`, (route) =>
    route.fulfill({ json: EMPTY_CHATS_PAGE })
  )
}

/** Mock the messages endpoint for an existing chat. */
export async function mockChatMessagesRoute(
  page: Page,
  chatId: string,
  messages: object[] = [USER_MESSAGE, ASSISTANT_MESSAGE]
) {
  await page.route(`**/api/chats/${chatId}/messages`, (route) =>
    route.fulfill({ json: messages })
  )
}
