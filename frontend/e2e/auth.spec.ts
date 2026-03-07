import { test, expect } from '@playwright/test'

const EMULATOR_HOST = process.env.FIREBASE_EMULATOR_HOST ?? 'firebase-emulator:9099'
const PROJECT_ID = 'boardbarian'
const TEST_EMAIL = 'e2e-test@example.com'
const TEST_PASSWORD = 'testpassword123'

test.beforeAll(async ({ request }) => {
  // Seed a test user in the Firebase Auth Emulator with a deterministic UID.
  // The admin endpoint requires "Authorization: Bearer owner".
  // Ignore failures — if the user already exists from a prior run in this session that's fine.
  await request.post(
    `http://${EMULATOR_HOST}/identitytoolkit.googleapis.com/v1/projects/${PROJECT_ID}/accounts`,
    {
      headers: { Authorization: 'Bearer owner' },
      data: {
        localId: 'e2e-test-user',
        email: TEST_EMAIL,
        password: TEST_PASSWORD,
      },
      failOnStatusCode: false,
    }
  )
})

test('login with emulator credentials redirects to game list', async ({ page }) => {
  await page.goto('/login')
  await page.getByPlaceholder('Email').fill(TEST_EMAIL)
  await page.getByPlaceholder('Password').fill(TEST_PASSWORD)
  await page.getByRole('button', { name: 'Sign in (emulator)' }).click()
  await expect(page).toHaveURL('/select-game')
  await expect(page.getByText('Select a Game')).toBeVisible()
})
