import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e-full',
  webServer: [
    {
      command: [
        'VITE_FIREBASE_EMULATOR=true',
        'VITE_FIREBASE_PROJECT_ID=boardbarian',
        'VITE_FIREBASE_API_KEY=fake-api-key',
        'npx vite --port 5174',
      ].join(' '),
      port: 5174,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: [
        'MM_FIREBASE__EMULATOR_HOST=firebase-emulator:9099',
        'uvicorn meeplemate.server.api:app --host 0.0.0.0 --port 8000',
      ].join(' '),
      cwd: '/workspace',
      port: 8000,
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
  use: {
    baseURL: 'http://localhost:5174',
    browserName: 'chromium',
    headless: true,
  },
})
