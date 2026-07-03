import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

/**
 * Dev-only plugin that simulates a Cloud Run cold start so a developer can see
 * how the UI retries and recovers. When `VITE_SIMULATE_COLD_START=<seconds>` is
 * set, every `/api/*` request during the first N seconds after the dev server
 * boots gets a plain-text 500 ("...no available instance.") — exactly what the
 * Cloud Run proxy returns during a real cold start. After the window, requests
 * proxy to the real backend as usual.
 *
 *   VITE_SIMULATE_COLD_START=30 npm run dev
 */
function simulateColdStart(): Plugin {
  const seconds = Number(process.env.VITE_SIMULATE_COLD_START)
  return {
    name: 'simulate-cold-start',
    apply: 'serve',
    configureServer(server) {
      if (!Number.isFinite(seconds) || seconds <= 0) return
      const until = Date.now() + seconds * 1000
      // Registering in the body (not a returned hook) runs this before Vite's
      // internal middlewares, including the `/api` proxy.
      server.middlewares.use((req, res, next) => {
        if (req.url?.startsWith('/api') && Date.now() < until) {
          res.statusCode = 500
          res.setHeader('Content-Type', 'text/plain; charset=utf-8')
          res.end('The request was aborted because there was no available instance.')
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), simulateColdStart()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    allowedHosts: ['localhost', 'ubuntu-box.local', 'ubuntu-box.home'],
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
