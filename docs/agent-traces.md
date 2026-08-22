# Agent Trace Persistence

Interim persistence of LangChain agent run traces
([issue #53](https://github.com/johntrimble/meeplemate/issues/53)), until a dedicated
tracing tool (Langfuse/LangSmith) is stood up.

Each assistant message's full run tree is serialized and stored under the key
`<chat_id>/<message_id>.json.gz`, making it easy to recover the trace for any message a
user reports on.

## How it works

- `meeplemate/tracing/tracer.py` — `PersistingTracer` subclasses `AsyncBaseTracer` and is
  attached to the graph's callbacks per-request in the streaming handler
  (`meeplemate/server/api.py`). Its `_persist_run` fires once, when the root run completes,
  which happens *inside* the streaming request before the response closes. This matters on
  Cloud Run, where CPU is throttled once the response completes — the upload is awaited
  within the request's CPU window.
- `meeplemate/tracing/sinks.py` — the `TraceSink` backends (`noop`, `local`, `gcs`).
  The GCS backend uses the synchronous `google-cloud-storage` client wrapped in
  `asyncio.to_thread`, authenticated via Application Default Credentials.
- `meeplemate/tracing/serialization.py` — run-tree serialization/transform helpers, shared
  with the eval harness. LangChain plumbing nodes (`Runnable*`) are pruned by default to
  keep traces small. The eval harness writes its run files the same way — gzipped JSON
  under a `.json.gz` name (see [eval.md](eval.md#where-the-output-lands)) — but persists
  to the eval output directory rather than through a `TraceSink`.
- Failures while serializing or uploading are swallowed and logged — persisting a trace
  never breaks the user-facing response.

## Configuration

Selected via `TraceConfig` (`MM_TRACE__*`), defaults to **off** (`noop`):

| Env var | Default | Description |
| --- | --- | --- |
| `MM_TRACE__BACKEND` | `noop` | `noop` (off), `local` (dev/testing), or `gcs` (prod) |
| `MM_TRACE__LOCAL_DIR` | `./data/traces` | Directory for the `local` backend |
| `MM_TRACE__BUCKET` | — | GCS bucket name (required when backend is `gcs`) |
| `MM_TRACE__PREFIX` | `` | Optional key prefix prepended to every GCS object |

- **Dev/testing:** `MM_TRACE__BACKEND=local` writes `./data/traces/<chat_id>/<message_id>.json.gz`.
- **Prod:** `MM_TRACE__BACKEND=gcs MM_TRACE__BUCKET=<bucket>`.

## Provisioning the GCS bucket (out-of-band)

The application never creates the bucket. Provision it once via console/IaC:

- A **dedicated, private** bucket (e.g. `boardbarian-agent-traces`) with uniform
  bucket-level access.
- A **90-day Object Lifecycle delete rule** so traces are purged automatically. Because the
  bucket holds only traces, the rule can be an unconditional "delete after 90 days".
- Grant the runtime service account `roles/storage.objectCreator` (write-only is sufficient).

Traces embed full prompts and rulebook content, so keep the bucket private. Auth uses ADC:
Workload Identity on Cloud Run; `gcloud auth application-default login` locally.

Verify the lifecycle rule:

```bash
gcloud storage buckets describe gs://<bucket> --format="default(lifecycle)"
```
