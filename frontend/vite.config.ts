import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    allowedHosts: ['localhost', 'ubuntu-box.local'],
    host: '0.0.0.0',
    proxy: {
      '/api': 'http://localhost:8000',
    },
    headers: {
      // Firebase signInWithPopup requires cross-origin popup communication.
      // Without this, Chrome's COOP policy blocks window.close in the auth popup.
      'Cross-Origin-Opener-Policy': 'unsafe-none',
    },
  },
})
