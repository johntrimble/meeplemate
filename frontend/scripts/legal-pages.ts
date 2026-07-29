import { readFileSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { marked } from 'marked'
import type { Plugin } from 'vite'

/**
 * Pre-renders the Terms of Use and Privacy Policy to standalone HTML.
 *
 * These are static documents, and routing them through the SPA was pure cost.
 * The whole app bundle - React, Router, TanStack Query, Firebase, the markdown
 * renderer - had to download and execute before a privacy policy could paint,
 * on the very page Google's OAuth brand review loads and crawlers index. It
 * also imported the app shell's problems: the markdown renderer emitted inert
 * <button>s where the documents' `mailto:` contact links should be, the pages
 * inherited a scroll container from `#root { overflow: hidden }` and couldn't
 * be scrolled at all, and switching between the two documents kept the previous
 * one's scroll offset. None of that is inherent to a legal document; all of it
 * came from rendering one inside an application.
 *
 * Markdown stays the authoring format: `tests/test_legal_versions.py` parses
 * the `**Effective date:**` line as the version the acceptance record stores,
 * and prose is far easier to revise as markdown than as HTML.
 *
 * The plugin owns both halves so dev matches production. Vite's dev server
 * hands extensionless paths to the SPA fallback, while Firebase Hosting serves
 * a matching static file ahead of its `**` rewrite - without the middleware
 * below, `/terms` would render the app in dev and a document in production.
 */

const DOCS = [
  { slug: 'terms', source: 'terms.md', title: 'Terms of Use' },
  { slug: 'privacy', source: 'privacy.md', title: 'Privacy Policy' },
] as const

const CONTENT_DIR = path.resolve(import.meta.dirname, '../src/content')

/** The other document, for the cross-link in the footer. */
const sibling = (slug: string) => DOCS.find((d) => d.slug !== slug)!

// Values lifted from src/index.css so the documents match the app without
// pulling in Tailwind. The app is dark-only (index.html hard-codes `class="dark"`),
// so these are the dark theme's.
const CSS = `
:root {
  --background: oklch(0.145 0 0);
  --foreground: oklch(0.985 0 0);
  --muted-foreground: oklch(0.708 0 0);
  --border: oklch(1 0 0 / 10%);
  --link: oklch(0.922 0 0);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--background);
  color: var(--foreground);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  -webkit-text-size-adjust: 100%;
}
.page { max-width: 48rem; margin: 0 auto; padding: 2rem 1rem 3rem; }
@media (min-width: 640px) { .page { padding: 3rem 1.5rem 4rem; } }
header, footer { color: var(--muted-foreground); font-size: 0.8125rem; }
header {
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
header a {
  display: inline-flex; align-items: center; gap: 0.5rem;
  font-weight: 600; letter-spacing: 0.02em;
  color: var(--muted-foreground); text-decoration: none;
}
header a img { width: 1.25rem; height: 1.25rem; }
header a:hover, footer a:hover { color: var(--foreground); }
footer {
  margin-top: 3rem; padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  display: flex; flex-direction: column; gap: 0.5rem;
}
@media (min-width: 640px) { footer { flex-direction: row; justify-content: space-between; } }
footer a { color: var(--muted-foreground); text-decoration: none; }
h1 { font-size: 1.5rem; line-height: 1.3; margin: 0 0 0.5rem; }
h2 { font-size: 1.125rem; margin: 2rem 0 0.75rem; }
h3 { font-size: 1rem; margin: 1.5rem 0 0.5rem; }
p, li { margin: 0.75rem 0; }
ul { margin: 0.75rem 0; padding-left: 1.5rem; }
li { margin: 0.25rem 0; }
a { color: var(--link); text-decoration: underline; text-underline-offset: 2px; overflow-wrap: anywhere; }
strong { font-weight: 600; }
/* Legal documents get printed and filed. Dark-on-light, and drop the
   navigation chrome, which is meaningless on paper. */
@media print {
  :root { --background: #fff; --foreground: #000; --muted-foreground: #444; --border: #ccc; --link: #000; }
  body { font-size: 11pt; }
  header, footer { display: none; }
  h2, h3 { break-after: avoid; }
}
`.trim()

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
}

/** Render one document to a complete HTML page. */
export function renderDoc(slug: string): string {
  const doc = DOCS.find((d) => d.slug === slug)
  if (!doc) throw new Error(`unknown legal document: ${slug}`)

  const markdown = readFileSync(path.join(CONTENT_DIR, doc.source), 'utf8')
  // `marked` emits real anchors, so the `mailto:support@boardbarian.com` links
  // the documents rely on for access, correction and deletion requests work by
  // construction rather than by overriding a renderer.
  const body = marked.parse(markdown, { async: false, gfm: true })
  const other = sibling(doc.slug)

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(doc.title)} · Boardbarian</title>
<meta name="description" content="${escapeHtml(doc.title)} for Boardbarian, a board game rules assistant.">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png">
<style>${CSS}</style>
</head>
<body>
<div class="page">
<header><a href="/"><img src="/mascot.png" alt="">Boardbarian</a></header>
<main>
${body}
</main>
<footer>
<a href="/${other.slug}/">Read the ${escapeHtml(other.title)}</a>
<a href="mailto:support@boardbarian.com">support@boardbarian.com</a>
</footer>
</div>
</body>
</html>
`
}

/**
 * Serve the documents in dev and emit them into `dist/` at build.
 *
 * Emitted as `<slug>/index.html` rather than `<slug>.html` because Firebase
 * Hosting resolves a directory index for a clean `/terms/` URL without needing
 * `cleanUrls`, which lives in the (separate) infra repo.
 */
export function legalPages(): Plugin {
  return {
    name: 'legal-pages',

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // Accept `/terms`, `/terms/` and `/terms/index.html` so a dev hitting
        // any of them sees what production serves.
        const match = req.url?.match(/^\/(terms|privacy)(?:\/(?:index\.html)?)?(?:[?#]|$)/)
        if (!match) return next()
        res.statusCode = 200
        res.setHeader('Content-Type', 'text/html; charset=utf-8')
        res.end(renderDoc(match[1]))
      })
    },

    async writeBundle(options) {
      const outDir = options.dir ?? path.resolve(import.meta.dirname, '../dist')
      for (const doc of DOCS) {
        const dir = path.join(outDir, doc.slug)
        await mkdir(dir, { recursive: true })
        await writeFile(path.join(dir, 'index.html'), renderDoc(doc.slug), 'utf8')
      }
    },
  }
}

export const LEGAL_DOCS = DOCS
