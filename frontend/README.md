# Boardbarian Frontend

Boardbarian is an AI assistant that answers board game rules questions, helping
players spend less time in rulebooks and more time playing.

This is the SPA. For the backend, the dev container, and the system as a whole,
see the [root README](../README.md) and [DEVELOPMENT.md](../DEVELOPMENT.md).

## Tech Stack

- [Vite](https://vite.dev/) — build tool and dev server
- [React 18](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS v4](https://tailwindcss.com/) — utility-first CSS
- [shadcn/ui](https://ui.shadcn.com/) — component library (new-york style)
- [AI Elements](https://elements.ai-sdk.dev/) — AI-specific UI components
- [React Router v7](https://reactrouter.com/) — client-side routing
- [Vercel AI SDK](https://ai-sdk.dev/) (`ai`, `@ai-sdk/react`) — the streaming chat
  protocol the backend speaks natively
- [TanStack Query](https://tanstack.com/query) + IndexedDB persistence — caching
  that survives reloads, so first paint never waits on a cold backend
- [Firebase JS SDK](https://firebase.google.com/docs/auth) — Google Sign-In

## Prerequisites

- Node.js 20+
- npm 10+

Nothing here needs the GPU stack. The SPA runs on plain Node against a remote
backend, or against a bypassed one — see **Configuration** below.

## Getting Started

```bash
# From the frontend/ directory:
npm install
npm run dev
```

The dev server starts at http://localhost:5173 and proxies `/api/*` to
`http://localhost:8000`.

## Configuration

Environment variables go in `frontend/.env.local` (never committed). Copy what
you need from [`.env.example`](.env.example), which documents every variable:
Firebase credentials, auth bypass, and the Firebase Auth emulator.

For local work without a Firebase project, set `VITE_AUTH_BYPASS=true`. The
reasoning behind bypass mode and the emulator — and why the emulator needs
persistence — is in [`docs/auth.md`](../docs/auth.md#bypass-mode-local-development).

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start dev server with HMR |
| `npm run build` | Type-check and build for production (output: `dist/`) |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run ESLint |
| `npm run test:unit` | Vitest unit tests (`src/**/*.test.ts`) |
| `npm run test:mocked` | Playwright against a mocked backend (`e2e/`) — this is what CI runs |
| `npm run test:e2e` | Playwright against the real backend + Firebase emulator (`e2e-full/`) |

`test:e2e` needs the `firebase-emulator` Docker service up; it starts its own
Vite server and uvicorn backend. See [DEVELOPMENT.md](../DEVELOPMENT.md).

## Project Structure

```
frontend/
├── e2e/                      # Playwright specs against a mocked backend
├── e2e-full/                 # Playwright specs against the real backend
├── public/                   # Static assets (mascot, icons)
├── scripts/
│   └── legal-pages.ts        # Vite plugin: pre-renders /terms/ and /privacy/
├── src/
│   ├── auth/                 # AuthProvider, CacheGate, consent + login screens
│   ├── components/
│   │   ├── ui/               # shadcn/ui primitives
│   │   └── ai-elements/      # AI Elements components
│   ├── content/              # terms.md, privacy.md (authoring source)
│   ├── data/                 # API clients (games, conversations, user)
│   ├── hooks/                # useChats, useGame, useGameList, ...
│   ├── lib/                  # cache persistence, fetchWithRetry, legal, seedGames
│   ├── pages/                # Home, SelectGame, Chat, Login, NotFound, ...
│   ├── App.tsx               # Route definitions
│   ├── index.css             # Tailwind v4 + CSS variables
│   ├── main.tsx              # React root + providers
│   └── vite-env.d.ts
├── components.json           # shadcn/ui config
├── index.html
├── package.json
├── playwright.config.ts      # real-backend E2E
├── playwright.mocked.config.ts
├── tsconfig{,.app,.node}.json
├── vite.config.ts
└── vitest.config.ts
```

## Deployment

A purely static SPA (no SSR). `npm run build` emits `dist/`, which deploys to any
CDN.

Two routing rules, in this order:

1. **`/terms/` and `/privacy/` are real static files**, pre-rendered at build time
   by `scripts/legal-pages.ts` into `dist/terms/index.html` and
   `dist/privacy/index.html`. They must be served ahead of the SPA fallback, or
   the app swallows them. Link to them **with the trailing slash**.
2. **Everything else falls back to `index.html`** for client-side routing.

The reasoning — why these two documents are deliberately outside the app bundle —
is in [`docs/legal.md`](../docs/legal.md).
