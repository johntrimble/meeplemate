import path from 'path'
import { defineConfig } from 'vitest/config'

// Standalone config so unit tests don't pull in the app's Vite plugins.
export default defineConfig({
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
