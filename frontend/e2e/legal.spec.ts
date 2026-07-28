import { test, expect, type Page } from '@playwright/test'
import { BYPASS_UID, acceptLegal, mockGameListRoutes } from './helpers/routes'

// The consent gate and the published documents.
//
// The gate's whole design goal is that it decides from local state alone, so
// most of these assertions are about what *doesn't* happen: no request before
// the prompt paints, no prompt once acceptance is stored.

const STORAGE_KEY = `boardbarian-legal-v1:${BYPASS_UID}`
const CHECKBOX = { name: /I am at least 13 years old/i }
const CONTINUE = { name: /Agree and continue/i }

const readStored = (page: Page) =>
  page.evaluate((k) => localStorage.getItem(k), STORAGE_KEY)

/** Record every acceptance POST so tests can assert on it. */
function captureAcceptance(page: Page, status = 204) {
  const calls: string[] = []
  page.route('**/api/account/legal-acceptance', (route) => {
    calls.push(route.request().postData() ?? '')
    return route.fulfill({ status, contentType: 'application/json', body: '{}' })
  })
  return calls
}

// ---------------------------------------------------------------------------
// The published documents
// ---------------------------------------------------------------------------

test('serves the terms without signing in', async ({ page }) => {
  await page.goto('/terms/')
  await expect(page.getByRole('heading', { name: 'Terms of Use', level: 1 })).toBeVisible()
  // The effective date is the version the acceptance record stores, so it has
  // to actually be on the page the user reads.
  await expect(page.getByText(/Effective date:/i).first()).toBeVisible()
})

test('serves the privacy policy without signing in', async ({ page }) => {
  await page.goto('/privacy/')
  await expect(page.getByRole('heading', { name: 'Privacy Policy', level: 1 })).toBeVisible()
})

test('needs no JavaScript at all', async ({ browser }) => {
  // The point of pre-rendering these. They are the pages Google's OAuth brand
  // review loads and crawlers index, and routing them through the SPA meant a
  // 2MB+ bundle had to execute before a privacy policy could paint.
  const context = await browser.newContext({ javaScriptEnabled: false })
  const page = await context.newPage()

  await page.goto('/privacy/')
  await expect(page.getByRole('heading', { name: 'Privacy Policy', level: 1 })).toBeVisible()
  await expect(page.getByRole('link', { name: /Read the Terms of Use/i })).toBeVisible()
  expect(await page.locator('script').count()).toBe(0)

  await context.close()
})

test('both documents scroll natively', async ({ page }) => {
  // As plain documents these scroll the window, with no app-shell scroll
  // container to get wrong - the bug that made these pages unreadable when
  // they lived inside the SPA cannot recur here.
  for (const path of ['/privacy/', '/terms/']) {
    await page.goto(path)
    const metrics = await page.evaluate(() => ({
      scrollHeight: document.documentElement.scrollHeight,
      clientHeight: document.documentElement.clientHeight,
    }))
    expect(metrics.scrollHeight).toBeGreaterThan(metrics.clientHeight)

    await page.evaluate(() => window.scrollTo(0, 1000))
    expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(0)
  }
})

test('renders the contact address as a working mailto link', async ({ page }) => {
  // support@ is the only route the Privacy Policy gives for exercising access,
  // correction, deletion and objection requests. The SPA's markdown renderer
  // used to emit inert <button>s here, making every one a dead end; `marked`
  // produces real anchors, so this now holds by construction.
  for (const path of ['/privacy/', '/terms/']) {
    await page.goto(path)
    const mailtos = page.locator('a[href^="mailto:support@boardbarian.com"]')
    expect(await mailtos.count()).toBeGreaterThan(0)
  }
})

test('the account menu opens the documents in a new tab', async ({ page, context }) => {
  // Not in-tab: ChatInput keeps the unsent question in local state, so
  // navigating away from a chat would silently discard whatever the user was
  // typing. A tab also preserves scroll position and costs nothing to close.
  await acceptLegal(page)
  await mockGameListRoutes(page)
  await page.goto('/select-game')

  await page.getByRole('button', { name: 'Account menu' }).click()
  const popup = context.waitForEvent('page')
  await page.getByRole('menuitem', { name: 'Terms of Use' }).click()

  const opened = await popup
  await opened.waitForLoadState()
  await expect(opened).toHaveURL(/\/terms\/$/)
  // The original tab never moved.
  await expect(page).toHaveURL(/\/select-game$/)
})

test('the consent screen opens the documents in a new tab too', async ({ page, context }) => {
  // Same reason as the account menu, plus the gate itself must survive: an
  // in-tab hop would tear down the screen and the ticked checkbox with it.
  await page.goto('/select-game')

  const popup = context.waitForEvent('page')
  await page.getByRole('link', { name: 'Privacy Policy' }).click()

  const opened = await popup
  await opened.waitForLoadState()
  await expect(opened).toHaveURL(/\/privacy\/$/)
  await expect(page).toHaveURL(/\/select-game$/)
})

test('reads as a standalone document, not an app screen', async ({ page }) => {
  await page.goto('/terms/')

  // No back button: usually read in its own tab, where "back" is meaningless,
  // and otherwise reached by deep link, where there is nothing to go back to.
  await expect(page.getByRole('button', { name: /back/i })).toHaveCount(0)
  // A masthead that links home instead - orientation and a way in, without
  // pretending the reader arrived from somewhere.
  await page.getByRole('link', { name: /Boardbarian/ }).first().click()
  await expect(page).toHaveURL(/localhost:\d+\/$/)
})

test('each document offers the other one, from the top', async ({ page }) => {
  // Whoever just read one of these wants the other far more than the
  // marketing page. The cross-link sits at the very bottom, so landing
  // mid-document would be the natural failure - a full page load starts at the
  // top for free, which the SPA version had to reimplement and got wrong.
  await page.goto('/terms/')
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
  expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(0)

  await page.getByRole('link', { name: /Read the Privacy Policy/i }).click()
  await expect(page).toHaveURL(/\/privacy\/$/)
  expect(await page.evaluate(() => window.scrollY)).toBe(0)
  await expect(page.getByRole('heading', { name: 'Privacy Policy', level: 1 })).toBeVisible()

  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
  await page.getByRole('link', { name: /Read the Terms of Use/i }).click()
  await expect(page).toHaveURL(/\/terms\/$/)
  expect(await page.evaluate(() => window.scrollY)).toBe(0)
})

test('new-tab links announce themselves to screen readers', async ({ page }) => {
  // WCAG G201. The arrow-out-of-a-box glyph is the visual convention, but it
  // is invisible to a screen reader - without the sr-only text those users
  // just find their back button dead. Assert the announcement, since that is
  // the part with no visual fallback.
  await page.goto('/select-game')

  const links = page.getByRole('link', { name: /opens in a new tab/i })
  await expect(links).toHaveCount(2)
  for (const link of await links.all()) {
    await expect(link).toHaveAttribute('target', '_blank')
    await expect(link).toHaveAttribute('rel', /noreferrer/)
  }
})

test('links to both documents from the home page footer', async ({ page }) => {
  // The anonymous path Google's OAuth brand review follows.
  await page.goto('/')
  await page.getByRole('link', { name: 'Privacy Policy' }).click()
  await expect(page).toHaveURL(/\/privacy\/$/)
})

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

test('blocks a user who has not accepted, without calling the API first', async ({ page }) => {
  await mockGameListRoutes(page)

  // Nothing account-scoped may be requested before the prompt paints - that
  // request would sit behind a cold start and the user would stare at a splash.
  const accountRequests: string[] = []
  await page.route('**/api/**', (route) => {
    accountRequests.push(route.request().url())
    return route.fallback()
  })

  await page.goto('/select-game')

  await expect(page.getByRole('heading', { name: /Before you start/i })).toBeVisible()
  expect(accountRequests.filter((u) => u.includes('legal-acceptance'))).toHaveLength(0)
})

test('requires the checkbox before continuing', async ({ page }) => {
  await page.goto('/select-game')

  await expect(page.getByRole('button', CONTINUE)).toBeDisabled()
  await page.getByRole('checkbox', CHECKBOX).click()
  await expect(page.getByRole('button', CONTINUE)).toBeEnabled()
})

test('the ticked checkbox is actually legible', async ({ page }) => {
  // `toBeChecked()` passes whether or not anything is visible, which is how a
  // black tick on a black box shipped: Tailwind sorts `dark:` after
  // `data-[state=checked]:`, so a bare `dark:bg-input/30` beats
  // `data-[state=checked]:bg-primary` while the tick still takes
  // `text-primary-foreground` (near-black in this theme). Assert the two
  // actually contrast, not merely that the state flipped.
  await page.goto('/select-game')
  await page.getByRole('checkbox', CHECKBOX).click()

  const swatch = await page.locator('[data-slot="checkbox"]').evaluate((box) => {
    const svg = box.querySelector('svg')!
    // Alpha matters as much as lightness here. The broken background computes
    // to `oklab(1 0 0 / 0.045)` - lightness 1, but 4.5% opaque over a near-black
    // page, so it renders almost black. Reading only the lightness channel
    // scores that as a bright background and misses the bug entirely.
    const parse = (s: string) => {
      const m = s.match(/(?:oklch|oklab)\(\s*([0-9.]+)[^/)]*(?:\/\s*([0-9.]+%?))?/)
      const raw = m?.[2]
      const alpha = raw === undefined ? 1 : raw.endsWith('%')
        ? parseFloat(raw) / 100
        : parseFloat(raw)
      return { lightness: parseFloat(m?.[1] ?? 'NaN'), alpha }
    }
    return {
      background: parse(getComputedStyle(box).backgroundColor),
      tick: parse(getComputedStyle(svg).color),
    }
  })

  // The checked background must actually be painted, not a near-transparent
  // wash that lets the dark page through behind a dark tick.
  expect(swatch.background.alpha).toBeGreaterThan(0.9)
  expect(
    Math.abs(swatch.background.lightness - swatch.tick.lightness),
  ).toBeGreaterThan(0.4)
})

test('records acceptance and lets the user through', async ({ page }) => {
  await mockGameListRoutes(page)
  const calls = captureAcceptance(page)

  await page.goto('/select-game')
  await page.getByRole('checkbox', CHECKBOX).click()
  await page.getByRole('button', CONTINUE).click()

  await expect(page.getByText('Munchkin').first()).toBeVisible()
  expect(calls).toHaveLength(1)
  expect(JSON.parse(calls[0])).toEqual({
    terms_version: expect.any(String),
    privacy_version: expect.any(String),
  })
})

test('does not prompt again on reload once accepted', async ({ page }) => {
  await mockGameListRoutes(page)
  captureAcceptance(page)

  await page.goto('/select-game')
  await page.getByRole('checkbox', CHECKBOX).click()
  await page.getByRole('button', CONTINUE).click()
  await expect(page.getByText('Munchkin').first()).toBeVisible()

  await page.reload()

  await expect(page.getByText('Munchkin').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: /Before you start/i })).not.toBeVisible()
})

test('does not store acceptance when the server rejects it', async ({ page }) => {
  await mockGameListRoutes(page)
  // 409: this bundle is stale, so what the user read is not what they'd be
  // agreeing to. Recording it locally would leave the client believing it had
  // accepted while the server disagreed.
  captureAcceptance(page, 409)

  await page.goto('/select-game')
  await page.getByRole('checkbox', CHECKBOX).click()
  await page.getByRole('button', CONTINUE).click()

  await expect(page.getByRole('alert')).toContainText(/updated/i)
  expect(await readStored(page)).toBeNull()
})

test('re-prompts when the accepted version is stale', async ({ page }) => {
  await mockGameListRoutes(page)
  await page.addInitScript(
    ([key, value]) => localStorage.setItem(key, value),
    [
      STORAGE_KEY,
      JSON.stringify({ termsVersion: '1999-01-01', privacyVersion: '1999-01-01' }),
    ] as const,
  )

  await page.goto('/select-game')

  // A version bump is knowable from the bundle alone, so this must resolve with
  // no network round trip - i.e. never mid-session, always on load.
  await expect(page.getByRole('heading', { name: /updated our terms/i })).toBeVisible()
})

test('offers a way out without accepting', async ({ page }) => {
  await page.goto('/select-game')

  await expect(page.getByRole('button', { name: /Sign out/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /Delete my account/i })).toBeVisible()
})

test('opens the documents from the gate without losing it', async ({ page, context }) => {
  await page.goto('/select-game')
  await page.getByRole('checkbox', CHECKBOX).click()

  const popup = context.waitForEvent('page')
  await page.getByRole('link', { name: 'Terms of Use' }).click()
  await (await popup).waitForLoadState()

  // New tab, so the gate - and the ticked checkbox - survive.
  await expect(page.getByRole('checkbox', CHECKBOX)).toBeChecked()
})

test('lets an accepted user straight through', async ({ page }) => {
  await acceptLegal(page)
  await mockGameListRoutes(page)

  await page.goto('/select-game')

  await expect(page.getByText('Munchkin').first()).toBeVisible()
  await expect(page.getByRole('heading', { name: /Before you start/i })).not.toBeVisible()
})
