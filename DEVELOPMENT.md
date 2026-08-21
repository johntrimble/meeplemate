## Running the Backend

From inside the dev container:

```bash
uvicorn meeplemate.server.api:create_app --factory --host 0.0.0.0 --port 8000 --reload --reload-dir meeplemate
```

The API will be available at `http://localhost:8000`. The frontend Vite proxy forwards `/api/*` requests there automatically.

---

## Running the Frontend

From inside the dev container:

```bash
cd frontend && npm run dev
```

The dev server starts on `http://localhost:5173`.

The frontend reads auth configuration from `frontend/.env.local`. To use the Firebase Auth Emulator (email/password login instead of Google OAuth):

```
VITE_FIREBASE_EMULATOR=true
```

The emulator host defaults to `window.location.hostname:9099`, so it automatically works whether you access via `localhost`, `ubuntu-box.local`, or any other hostname — as long as port 9099 is reachable at that address. Set `VITE_FIREBASE_EMULATOR_HOST` explicitly only when the browser can't reach the emulator at the same hostname (e.g. inside a Docker container).

### Simulating a Cloud Run cold start

The frontend transparently retries requests that fail while a Cloud Run instance is cold-starting (see `frontend/src/lib/fetchWithRetry.ts`). To see this behavior locally, start the dev server with `VITE_SIMULATE_COLD_START` set to a number of seconds:

```bash
cd frontend && VITE_SIMULATE_COLD_START=30 npm run dev
```

For the first 30 seconds after the dev server boots, every `/api/*` request returns a plain-text `500` (`"...no available instance."`), exactly like the Cloud Run proxy during a real cold start. The UI keeps showing its normal loading indicators, and after a few seconds those indicators switch their text to "Waking up the server..." (not an error). Everything recovers automatically once the window elapses and requests reach the real backend.

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

## Ingest Data into Prod

> To process a rulebook PDF into an ingested package in the first place (OCR →
> chunking → local import), see [docs/ingestion.md](docs/ingestion.md). The steps
> below cover only pushing an already-ingested package to the prod database.

Get database url from meeplemate-infra repo:

```bash
terraform output database_url
```

Add output to `.env.ingest.prod` as `MM_PG__URL`.

Run the ingest script with the environment variables:

```bash
env $(grep -v '^\s*#' .env.ingest.prod | grep -v '^\s*$' | xargs) mm-ingest import-documents data/ingested/some-game
```
