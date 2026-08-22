# Development

Everything here runs **from inside the dev container**. See the
[README](README.md#getting-started) for bootstrapping and opening it.

## Running the App

```bash
./script/server
```

Starts the API on `http://localhost:8000` and the Vite frontend on
`http://localhost:5173`, and stops both on Ctrl+C. The frontend proxies `/api/*`
to the backend automatically.

To run just one half:

```bash
uvicorn meeplemate.server.api:create_app --factory --host 0.0.0.0 --port 8000 --reload --reload-dir meeplemate
cd frontend && npm run dev
```

## Frontend Configuration

Frontend environment variables live in `frontend/.env.local`, which is never
committed. [`frontend/.env.example`](frontend/.env.example) documents all of
them — Firebase credentials, auth bypass, and the Firebase Auth emulator — and
[`docs/auth.md`](docs/auth.md#bypass-mode-local-development) explains what each
mode is for.

The short version: `VITE_AUTH_BYPASS=true` skips Firebase entirely, and
`VITE_FIREBASE_EMULATOR=true` swaps the Google popup for the emulator's
email/password form.

### Simulating a Cloud Run cold start

The frontend transparently retries requests that fail while a Cloud Run instance is cold-starting (see `frontend/src/lib/fetchWithRetry.ts`). To see this behavior locally, start the dev server with `VITE_SIMULATE_COLD_START` set to a number of seconds:

```bash
cd frontend && VITE_SIMULATE_COLD_START=30 npm run dev
```

For the first 30 seconds after the dev server boots, every `/api/*` request returns a plain-text `500` (`"...no available instance."`), exactly like the Cloud Run proxy during a real cold start. The UI keeps showing its normal loading indicators, and after a few seconds those indicators switch their text to "Waking up the server..." (not an error). Everything recovers automatically once the window elapses and requests reach the real backend.

`./script/server` picks the variable up too, and prints a banner when the
simulation is on.

---

## Running E2E Tests

Requires the `firebase-emulator` Docker service to be running:

```bash
# from the host (if not already up)
docker compose up firebase-emulator -d
```

Then from inside the dev container:

```bash
cd frontend && npm run test:e2e
```

Playwright starts its own Vite server on port 5174 (with emulator auth) and the API backend on port 8000 automatically. No manual server startup needed.

---

## Database Schema

### 1) Edit models in meeplemate/db/models.py

### 2) Generate migration:

```bash
alembic revision --autogenerate -m "describe change"
```

### 3) Review change in the generated file in alembic/versions

### 4) Apply migration:

```bash
alembic upgrade head
```

---

## Ingesting Rulebooks

Processing a rulebook PDF into an ingested package, and pushing one to the prod
database, are both covered in [docs/ingestion.md](docs/ingestion.md).
