# Terms of Use and Privacy Policy

The two documents users have to accept, where they live, how acceptance is
recorded, and what to do when they change.

## The documents

| | |
|---|---|
| Terms of Use | [`frontend/src/content/terms.md`](../frontend/src/content/terms.md) → `/terms/` |
| Privacy Policy | [`frontend/src/content/privacy.md`](../frontend/src/content/privacy.md) → `/privacy/` |

**They are pre-rendered to standalone HTML, not SPA routes.**
[`frontend/scripts/legal-pages.ts`](../frontend/scripts/legal-pages.ts) is a Vite
plugin that renders the markdown with `marked` and emits
`dist/{terms,privacy}/index.html` at build time. The result is ~20KB of
self-contained HTML with **zero script tags**.

Routing them through the app was pure cost. The whole bundle — React, Router,
TanStack Query, Firebase, the markdown renderer, 2.3MB — had to download and
execute before a privacy policy could paint, on the very page Google's OAuth
brand review loads and crawlers index. It also inherited the app shell's
problems: the markdown renderer emitted inert `<button>`s where the documents'
`mailto:` contact links should be, `#root { overflow: hidden }` left the pages
unscrollable, and switching documents kept the previous one's scroll offset.
None of that is inherent to a legal document; all of it came from rendering one
inside an application.

Markdown stays the authoring format — `tests/test_legal_versions.py` parses the
`**Effective date:**` line as the version, and prose is easier to revise as
markdown.

The plugin owns dev *and* build because the two resolve URLs differently: Vite's
dev server hands extensionless paths to the SPA fallback, while Firebase Hosting
serves a matching static file ahead of its `**` rewrite. Without the dev
middleware, `/terms` would render the app locally and a document in production.

**Use the trailing slash** (`/terms/`, not `/terms`). Files are emitted as
`<slug>/index.html` so Firebase resolves a clean URL without needing `cleanUrls`,
which lives in the separate infra repo; linking to `/terms` costs every reader a
301 hop. The home page footer uses plain `<a>` rather than react-router `<Link>`
for the same reason — a client-side navigation would find no route and fall
through to the 404.

**Every in-app surface links to them with `target="_blank"`** — the account menu,
the consent screen, and the sign-in screen. Not a style preference: `ChatInput`
holds the unsent question in local state, so an in-tab hop silently discards
whatever the user was typing; the consent screen would lose its ticked checkbox;
and the sign-in screen renders inline at whatever gated URL was requested, so
navigating away drops the very route the user was trying to reach.

Those links go through `components/ExternalLink.tsx`, which pairs the
arrow-out-of-a-box glyph with `sr-only` text reading "(opens in a new tab)".
WCAG technique G201 wants the new tab announced, and the icon alone is invisible
to a screen reader — bundling the two in one component means neither can be
added without the other. The account menu inlines the same pair rather than
using the component, because the dropdown item needs the icon right-aligned in
its own slot.

Each page is a **standalone document**: a wordmark masthead linking home, and a
footer cross-linking the other document and support@. Deliberately *no* back
button — these are normally read in their own tab, where "back" means nothing
and the reader just closes it, and otherwise reached by deep link or shared URL,
where there is no previous page to return to.

Styling is a ~60-line inline stylesheet in the plugin, using colour values
lifted from `src/index.css` so the documents match the app without pulling in
Tailwind. It includes a `@media print` block that flips to black-on-white and
drops the navigation chrome — legal documents get printed and filed, and the
app is dark-only.

## Versions

The version string is the document's **effective date**, verbatim, as
`YYYY-MM-DD`. Four places carry it and must agree:

| Place | Form |
|---|---|
| `frontend/src/content/terms.md` | `**Effective date: July 28, 2026**` |
| `frontend/src/content/privacy.md` | `**Effective date: July 28, 2026**` |
| `frontend/src/lib/legal.ts` | `TERMS_VERSION` / `PRIVACY_VERSION` |
| `meeplemate/server/legal.py` | `TERMS_VERSION` / `PRIVACY_VERSION` |

`tests/test_legal_versions.py` fails if they drift. A frontend/backend mismatch
is the nastiest failure mode available here: the client posts a version the
server refuses with 409, so the gate can never be satisfied and the app is
unusable for **everyone**.

## How acceptance works

The gate decides from local state alone and never waits on the network. That is
load-bearing — the cold-start design (`seedGames.ts`, the IndexedDB persister,
`fetchWithRetry`) exists so first paint never blocks on the scale-to-zero
backend, and a "have you accepted?" request in front of the app would undo it.
On a cold start the user would stare at a splash for up to 90 seconds.

Both halves of the comparison are already local:

| `localStorage` | Meaning | Action |
|---|---|---|
| Matches the bundled constants | Accepted current version | Render; zero network |
| Holds an older version | A version bump | Prompt; zero network |
| Empty | New user, new device, or cleared browser | Prompt; zero network |

Blocking on empty re-prompts a returning user on a new device. That is the
accepted trade: no status endpoint, no background request, and an un-accepted
user is never shown the app. The POST is idempotent, so it costs them one click
and leaves the recorded `accepted_at` alone.

**Accepting does not wait for the server.** Clicking "Agree and continue" writes
localStorage and opens the app in the same tick; the POST that records it is
fired unawaited. This matters more than it looks: for a new user that POST is the
*first* request the backend ever sees — every other call site lives inside
`CacheGate`'s children, and the home and login screens call nothing — so it eats
a full Cloud Run cold start by construction. Awaiting it meant 26–90s of spinner
on top of a game list that `seedGamesFromStatic` had already loaded from the CDN
and was ready to paint. It still doubles as the warm-up; it just no longer holds
the door shut while it warms.

An unsent acceptance is not lost. `writeAcceptance` stores `synced: false` and
`useAcceptanceSync` (called from `CacheGate`) retries on later loads until the
request settles. `synced` means "nothing left to send", not "the server has it":
a 4xx sets it too, because a payload the server rejects will be rejected
identically forever and retrying it is a loop, not eventual consistency.

### Server side

The server records acceptance. It does not enforce it.

That is a deliberate reversal. `get_db_user` used to 403 un-accepted accounts,
and that 403 is what forced the client to await the POST — a client that believed
it had accepted while the server disagreed would hit a wall on its first question
with no way back to the prompt. The gate bought very little: a bypass still needs
a valid Firebase token, rate limiting is a separate dependency, and the only
person served by skipping the checkbox is the person it binds. What has real
value is the record — this app's consent covers storing conversations and sending
questions to third-party providers, and localStorage is not evidence of anything.
So the record stays and the gate goes.

The consequence to accept with open eyes: someone who edits localStorage, or
drives the API directly, uses the app with no acceptance on file.

- Stored in `app_user.metadata` under the `legal` key (`LEGAL_METADATA_KEY` in
  `meeplemate/db/datalayer.py`), sharing the blob with the per-user rate-limit
  overrides. Written by `record_legal_acceptance` as a JSONB **merge**, never a
  whole-blob write, so it cannot clobber those overrides.
- **The endpoint stores what the client posted**, verbatim — not the server's own
  constants. "User accepted 2026-01-01 at time T" is a true statement even when
  today's version is 2026-07-28, and it is the only version anyone can honestly
  claim was put in front of them.
- **The only rejection is a malformed version string.** `LegalAcceptanceRequest`
  requires `YYYY-MM-DD` that parses as a real date. Nothing is refused for being
  stale, or for being a version this process has never heard of. Both of those
  rules — the old 409, and the version allow-list that looks like the obvious
  replacement for it — silently couple the record to deploy order: ship the
  frontend first and every client posts a version the backend does not yet know,
  so valid acceptances get thrown away. **No deploy order may cost a user their
  acceptance record.** A version that isn't current is logged, not refused; that
  is the monitoring signal an allow-list would have given, minus the data loss.
- A stale client self-heals with no help from the server. It records what it
  showed, and on its next load the newer bundle's constants no longer match its
  localStorage, so the gate re-prompts.
- `record_legal_acceptance` in the repository **will not overwrite a newer stored
  acceptance with an older one**. Now that clients can post non-current versions,
  a stale tab could otherwise downgrade a record another device just advanced.
  Versions are zero-padded ISO dates, so they order as strings.
- `has_accepted_current` no longer gates anything and has no caller in the
  request path. It survives as a predicate for reporting and tests.
- `DELETE /api/account` uses `get_db_user_allow_deleted` so it stays callable
  after the account is flagged. **This is the decline path**: signing out leaves
  the account sitting there un-accepted, so deletion is the only real withdrawal.

## Bumping a version

Only bump when you want every user re-prompted. A typo fix or a clarification
should leave the effective date — and therefore the version — alone; changing it
re-prompts everyone for nothing.

1. Edit the document and update its `**Effective date:**` line.
2. Update the matching constant in `frontend/src/lib/legal.ts`.
3. Update the matching constant in `meeplemate/server/legal.py`.
4. `pytest tests/test_legal_versions.py` — this is what catches step 2 or 3
   being forgotten.
5. Deploy in **either order**. This used to read "backend first, or the frontend's
   users post a version the old backend rejects with 409 and cannot get in";
   neither half is true any more, since there is no 409 and no gate. A frontend
   deployed ahead of the backend just logs `legal_acceptance_version_divergence`
   until the backend catches up, and every acceptance in between is recorded
   correctly.

## Not covered

Recorded here so they don't get rediscovered as surprises:

- **No GDPR/UK section.** Art. 3(2) turns on *offering services* to EU users,
  not mere reachability (Recital 23); Boardbarian is English-only, `.com`, with
  no EU targeting. A real section would need an Art. 6 lawful basis, an Art. 21
  right to object that contradicts Privacy §4's "retention for testing is a
  condition of use", and a paid Art. 27 EU representative. Privacy §9's
  jurisdiction-neutral commitment is the proportionate substitute. Revisit if
  the EU is ever deliberately targeted.
- **Deletion is not automated.** `mm-admin purge-deleted-accounts` is hand-run.
  Privacy §8 states the criteria rather than a fixed clock, which is honest
  today and required by Cal. Civ. Code §1798.100(a)(3); tighten it back to a
  hard commitment once a scheduler exists.
- **No pseudonymised testing pipeline.** Privacy §4 and §8 describe test
  identifiers as something Boardbarian *may* do. Nothing implements it — purge
  removes chats outright via CASCADE.
- **No age verification** beyond the checkbox affirmation on the consent screen.
- **No DMCA designated agent** registered with the Copyright Office. Terms §21
  gives a contact, but safe harbour requires registration.
- **No data export**, as Privacy §9 says.
- **Google OAuth branding is console config, not code.** Set the privacy policy
  and terms URLs under Google Auth Platform → Branding. Note that returning
  users usually skip the consent screen entirely, which is why the in-app gate
  is the actual mechanism rather than a supplement.