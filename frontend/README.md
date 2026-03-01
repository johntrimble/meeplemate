# Boardbarian Frontend

Boardbarian is an AI assistant that answers board game rules questions, helping players spend less time in rulebooks and more time playing.

## Tech Stack

- [Vite](https://vite.dev/) — build tool and dev server
- [React 18](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS v4](https://tailwindcss.com/) — utility-first CSS
- [shadcn/ui](https://ui.shadcn.com/) — component library (new-york style)
- [Elements AI](https://elements.ai-sdk.dev/) — AI-specific UI components
- [React Router v7](https://reactrouter.com/) — client-side routing
- [Vercel AI SDK](https://ai-sdk.dev/) — AI stream protocol integration (future)

## Prerequisites

- Node.js 20+
- npm 10+

## Getting Started

```bash
# From the frontend/ directory:
npm install
npm run dev
```

The dev server starts at http://localhost:5173.

## Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start dev server with HMR |
| `npm run build` | Type-check and build for production (output: `dist/`) |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | Run ESLint |

## Project Structure

```
frontend/
├── public/                   # Static assets
├── src/
│   ├── components/
│   │   ├── ui/               # shadcn/ui primitives
│   │   └── ai-elements/      # Elements AI components
│   ├── lib/
│   │   └── utils.ts          # cn() utility
│   ├── pages/
│   │   └── HomePage.tsx
│   ├── App.tsx               # Route definitions
│   ├── index.css             # Tailwind v4 + CSS variables
│   ├── main.tsx              # React root + providers
│   └── vite-env.d.ts
├── components.json           # shadcn/ui config
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
└── vite.config.ts
```

## Deployment

This is a purely static SPA (no SSR). Build with `npm run build` and deploy the `dist/` directory to any CDN. Configure the CDN to serve `index.html` for all routes to support client-side routing.
