import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  webServer: {
    command: [
      'VITE_AUTH_BYPASS=true',
      'npx vite --port 5175',
    ].join(' '),
    port: 5175,
    reuseExistingServer: false,
    timeout: 30_000,
  },
  use: {
    baseURL: 'http://localhost:5175',
    browserName: 'chromium',
    headless: true,
  },
})
