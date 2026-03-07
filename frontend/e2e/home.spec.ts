import { test, expect } from '@playwright/test'

test('homepage shows marketing copy and sign-in button', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Rules questions, answered instantly.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Sign In' })).toBeVisible()
})
