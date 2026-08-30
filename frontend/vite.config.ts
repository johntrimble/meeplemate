import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'
import { legalPages } from './scripts/legal-pages'

/**
 * Dev-only plugin that simulates a Cloud Run cold start so a developer can see
 * how the UI retries and recovers. When `VITE_SIMULATE_COLD_START=<seconds>` is
 * set, every `/api/*` request during the first N seconds after the dev server
 * boots gets a plain-text 500 ("...no available instance.") - exactly what the
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

/**
 * Dev-only plugin that holds the chat stream open before any bytes are written,
 * so a developer can actually look at the pending "Thinking..." state (issue
 * #107) instead of blinking past it. When `VITE_SIMULATE_STREAM_DELAY=<seconds>`
 * is set, every `POST /api/chats/{id}/stream` waits N seconds and then proxies to
 * the real backend as usual.
 *
 *   VITE_SIMULATE_STREAM_DELAY=20 ./script/server
 *
 * Distinct from `simulateColdStart` above, which is about *recovery*: that one
 * fails `/api/*` outright for a window after boot, which also stops the chat page
 * from loading at all - fine for watching `fetchWithRetry` retry, useless for
 * watching the send path. This one touches only the stream, arms on every send
 * rather than once at boot, and delays rather than fails, so the response still
 * streams in normally at the end.
 */
function simulateStreamDelay(): Plugin {
  const seconds = Number(process.env.VITE_SIMULATE_STREAM_DELAY)
  return {
    name: 'simulate-stream-delay',
    apply: 'serve',
    configureServer(server) {
      if (!Number.isFinite(seconds) || seconds <= 0) return
      // Registered in the body so it runs ahead of Vite's `/api` proxy.
      server.middlewares.use((req, _res, next) => {
        if (/^\/api\/chats\/[^/]+\/stream/.test(req.url ?? '')) {
          setTimeout(next, seconds * 1000)
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), simulateColdStart(), simulateStreamDelay(), legalPages()],
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
