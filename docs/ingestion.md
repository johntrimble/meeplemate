# Ingesting a Rulebook

This guide covers the full pipeline for turning a rulebook PDF into searchable
data in MeepleMate: render → OCR → text → page numbers → metadata → chunking →
summaries → presentation → import.

All commands are exposed by the `mm-ingest` CLI (defined in
[meeplemate/ingest/__main__.py](../meeplemate/ingest/__main__.py) and registered
in [pyproject.toml](../pyproject.toml) as the `mm-ingest` entry point). You can
invoke them either way:

```bash
mm-ingest <command> <args>
# or, equivalently:
python -m meeplemate.ingest <command> <args>
```

Run everything **from inside the dev container**.

---

## Prerequisites

The OCR steps talk to two vLLM services. Their endpoints, model names and
sampling settings are the `IngestConfig` defaults in
[config.py](../meeplemate/config.py), which match the
[compose.yaml](../compose.yaml) service names and ports:

| Step | Compose service | Compose profile | Config key | Default URL |
|------|-----------------|-----------------|------------|-------------|
| `ocr` | `vllm-deepseek-ocr` | `deepseek-ocr` | `ingest.ocr` | `http://vllm-deepseek-ocr:8000/v1` |
| `page-number-ocr` | `vllm-glm-ocr` | `glm-ocr` | `ingest.page_number_ocr` | `http://vllm-glm-ocr:8080/v1` |

To change any of them, either set an environment variable —
`MM_INGEST__OCR__MODEL=... mm-ingest ocr $PKG` — or add an `ingest:` block to the
config file `MM_CONFIG_FILE` points at. The dev container uses
[config-dev.yaml](../config-dev.yaml); neither it nor
[config.yaml](../config.yaml) sets `ingest:` today, so the defaults above are
what actually runs. Environment variables win over the config file.

Both OCR services are gated behind Compose **profiles**, so a plain
`docker compose up -d` does **not** start them. Bring up the one you need
explicitly, e.g.:

```bash
docker compose --profile deepseek-ocr up -d vllm-deepseek-ocr
docker compose --profile glm-ocr up -d vllm-glm-ocr
```

Each also requires an NVIDIA GPU (see the `deploy.resources` reservation in
[compose.yaml](../compose.yaml)). The `import-documents` step and every
`generate-*` step need Postgres and the configured chat model running, which come
up with the default `docker compose up -d`.

---

## Dependency graph

The steps are **not** independent — most read files an earlier step wrote, and
many reads are unguarded (a missing input raises `FileNotFoundError` / `KeyError`
rather than being skipped). The arrows below are hard dependencies unless marked
otherwise:

```
init-game-package          → manifest.yaml, raw_documents/
        │
        ▼
     render                → images/<dk>/NNNN.png        (deterministic, CPU)
        │
        ├───────────────► page-number-ocr   → page_numbers_raw/<dk>/NNNN.txt
        │                        │
        ▼                        ▼
       ocr                 page-number-fixup → page_numbers/<dk>/NNNN.txt
   → ocr/<dk>/NNNN.raw.md         │
     ocr/<dk>/NNNN.structured.json│
        │                         │
        ▼                         │
    build-text                    │
   → text/<dk>/NNNN.md            │
     text/<dk>/NNNN.structured.fixed.json
     text/<dk>/document.md        │
        │                         │
        ▼                         ▼
      add-metadata (reads text/, page_numbers/)
        │                  → metadata/<dk>/NNNN.yaml
        │
        │  Everything below reads text/ + metadata/ (via page_to_document)
        ├──────────────┬─────────────────────┐
        ▼              ▼                     ▼
  build-chunks   generate-reference   generate-setting-summary
   → chunks/      → reference/game.md    → setting/game.md
        │           reference/<dk>.md         │
        │           reference/<dk>.short.md   ▼
        │              │              generate-presentation (reads setting/game.md)
        │              ▼                  → presentation.yaml
        │        generate-example-questions      │
        │         (reads reference/; soft)       │
        │              ▼                         │
        │        import-example-questions        │
        │        (→ game_questions_store,        │
        │          keyed by game_id;             │
        │          NOT part of import-documents) │
        ▼                                        ▼
                    import-documents  ◄───────────┘
   (reads: text/, metadata/, chunks/, presentation.yaml [required],
           version.txt, reference/ [used if present])
```

`update-version` writes `version.txt` and sits outside this graph — it is an
input to `import-documents`, not a step other steps depend on.

### What `import-documents` actually requires

`import-documents` will **fail** unless these have run (unguarded reads in
`run_import_documents` / `import_game_data`, [dataimport.py](../meeplemate/ingest/dataimport.py)):

- `build-text` → `text/<dk>/NNNN.md` (which requires `render` then `ocr`)
- `add-metadata` → `metadata/<dk>/NNNN.yaml` (read by `page_to_document`)
- `build-chunks` → `chunks/`
- `generate-presentation` → `presentation.yaml` — which in turn **requires**
  `generate-setting-summary` (it unguarded-reads `setting/game.md`).
- `update-version` → `version.txt`. Without it the page and chunk IDs written to
  Postgres carry an empty version field.

And `add-metadata` itself requires `page-number-fixup`, which writes a page
number for **every** page — including rulebooks with an explicit
`page_one_offset`, which are computed rather than OCR'd.

It will **succeed but produce a degraded game** without:

- `generate-reference` — writes `reference/game.md` and
  `reference/<dk>.short.md`. `load_game_package` folds these back in as the
  manifest's `summary` fields and `import_game_data` deep-copies them into the
  stored game data, so skipping this yields a summary-less game.

Example questions are separate and don't affect `import-documents`:
`generate-example-questions` needs `generate-reference`, and the resulting
`example_questions.yaml` is loaded by `import-example-questions` (keyed by
`game_id`), not by `import-documents`.

---

## Package layout

Ingestion operates on a **game package**. There are two directories in play, and
they deliberately name their manifests differently:

- **Source directory** (e.g. `data/rules/<game>/`): the raw PDFs plus a
  hand-written `rulebooks.yaml`. Checked in by hand; not written by any step.
- **Ingested / output directory** (e.g. `data/ingested/<game>/`): the working
  package, whose derived manifest is `manifest.yaml`. `init-game-package` creates
  this from the source directory.

### Source `rulebooks.yaml`

Minimal required fields are `name`, `game_id`, and a `rulebooks` list where each
entry has a `name` and a `path` (relative to the source directory). `url` and
`strategy` are optional. Example
([data/rules/munchkin_rules/rulebooks.yaml](../data/rules/munchkin_rules/rulebooks.yaml)):

```yaml
name: Munchkin
game_id: munchkin
rulebooks:
- name: Munchkin Rules
  path: munchkin_rules-1.pdf
  strategy: "ocr_only"
  url: "https://munchkin.game/site-munchkin/assets/files/1138/munchkin_rules-1.pdf"
- name: "Munchkin Princesses Rules"
  path: princesses_rules.pdf
  strategy: "ocr_only"
  url: "https://munchkin.game/site-munchkin/assets/files/1394/princesses_rules.pdf"
```

### One output per step

The package is organised so that **every step owns one or more top-level entries,
no entry has two owners, and no entry nests inside another**. Directories group by
producing step first and rulebook second — `images/<dk>/`, not `<dk>/images/` —
so a step's whole output is one nameable path.

That rule is enforced by [tests/ingest/test_layout.py](../tests/ingest/test_layout.py)
against `PackageLayout.STEP_OUTPUTS` in
[layout.py](../meeplemate/ingest/layout.py), which is the single place any
package path is constructed.

| Step | Owns |
|------|------|
| `init-game-package` | `manifest.yaml`, `raw_documents/` |
| `update-version` | `version.txt` |
| `render` | `images/` |
| `ocr` | `ocr/` |
| `build-text` | `text/` |
| `page-number-ocr` | `page_numbers_raw/` |
| `page-number-fixup` | `page_numbers/` |
| `add-metadata` | `metadata/` |
| `build-chunks` | `chunks/` |
| `generate-reference` | `reference/` |
| `generate-setting-summary` | `setting/` |
| `generate-presentation` | `presentation.yaml` |
| `generate-example-questions` | `example_questions.yaml` |

### What the pipeline produces

`<dk>` below is a rulebook's `document_key` (the base64 of its filename).

```
data/ingested/<game>/
├── manifest.yaml                         # name, game_id, rulebooks, page counts (init-game-package)
├── version.txt                           # game_version uuid                     (update-version)
├── raw_documents/<file>.pdf              # copied source PDF(s)                  (init-game-package)
├── images/<dk>/NNNN.png                  # rendered page image                   (render)
├── ocr/<dk>/NNNN.raw.md                  # raw model output (debug only)         (ocr)
├── ocr/<dk>/NNNN.structured.json         # structured OCR output                 (ocr)
├── text/<dk>/NNNN.md                     # per-page markdown                     (build-text)
├── text/<dk>/NNNN.structured.fixed.json  # sentence-/header-fixed structure      (build-text)
├── text/<dk>/document.md                 # concatenated full-document markdown   (build-text)
├── page_numbers_raw/<dk>/NNNN.txt        # OCR'd page number                     (page-number-ocr)
├── page_numbers/<dk>/NNNN.txt            # cleaned/interpolated page number      (page-number-fixup)
├── metadata/<dk>/NNNN.yaml               # per-page metadata + char offsets      (add-metadata)
├── chunks/<dk>/parent/NNNN_MMMM.json     # parent chunks                         (build-chunks)
├── chunks/<dk>/child/NNNN_MMMM_CCCC.json # child chunks                          (build-chunks)
├── reference/game.md                     # game-level rules reference            (generate-reference)
├── reference/<dk>.md                     # per-rulebook reference                (generate-reference)
├── reference/<dk>.short.md               # per-rulebook short summary            (generate-reference)
├── setting/game.md                       # theme/setting summary                 (generate-setting-summary)
├── setting/<dk>.md                       # per-rulebook setting summary          (generate-setting-summary)
├── presentation.yaml                     # emoji + background color              (generate-presentation)
└── example_questions.yaml                # suggested questions (optional)        (generate-example-questions)
```

### The manifest holds only what the source implies

`manifest.yaml` carries `name`, `game_id` and the `rulebooks` list with their
document keys and page counts — nothing else, and `init-game-package` is its only
writer. The version and the generated summaries live in their own files.

`load_game_package` folds them back in when reading, so in memory a package still
looks the way it always did: `gp["game_version"]` comes from `version.txt`,
`gp["summary"]` from `reference/game.md`, and each `rulebook["summary"]` from
`reference/<dk>.short.md`. Nothing outside `meeplemate/ingest/` sees the change.

### Migrating an existing package

Packages created before this layout can be converted in place. It is a rename
pass — no re-OCR, no GPU:

```bash
mm-ingest migrate-layout $PKG              # dry run, reports what would move
mm-ingest migrate-layout $PKG --apply --verify
```

It refuses to run if it meets a file it cannot classify, rather than leaving it
behind, and it is safe to re-run after an interruption. `--verify` checks that
every artifact the manifest implies now resolves. `data/ingested` is not in git,
so take a copy before `--apply`.

---

## The pipeline (run in order)

Below, `PKG` is the output package directory, e.g. `./data/ingested/munchkin_rules/`.
Steps 1–11 are all required for a complete import; see
[What `import-documents` actually requires](#what-import-documents-actually-requires)
for exactly which are hard vs. degrading.

### 1. Initialize the package

Copies the PDFs into `raw_documents/`, assigns a `document_key` per rulebook, and
computes `page_count` for each PDF. Writes `manifest.yaml` — and is its only
writer.

```bash
mm-ingest init-game-package ./data/rules/munchkin_rules/ ./data/ingested/munchkin_rules/
```

> Takes `<input>` (source dir) and `<output>` (package dir). Every later step
> takes only the package dir.

### 2. Stamp a version

```bash
mm-ingest update-version $PKG
```

**Required, including on a first ingest.** The version is written to
`version.txt`; it is no longer stamped by `init-game-package`. Page and chunk IDs
in Postgres embed it, and `add-metadata`, `build-chunks`, `import-documents` and
`rebuild-bm25` all refuse to run without it.

Run it again before *re-ingesting* an already-published game: `import-documents`
refuses to overwrite a version that is already in the DB, so a fresh version lands
the re-ingest alongside the live one until you're ready to swap. Afterwards remove
the superseded version with `mm-ingest clear-old-data`. To deliberately re-import
the same version in place, skip the bump and use `import-documents --overwrite`.

### 3. Render the pages

Rasterise every rulebook page to a PNG at the configured dpi. Deterministic and
CPU-only — no model required, so it does not need an OCR service running.

```bash
mm-ingest render $PKG
```

> These images are ~90% of a package's bytes and reproduce byte-for-byte from the
> same PDF, poppler and Pillow. Keeping them in their own step means re-running
> OCR never touches them.

### 4. OCR the pages

Run DeepSeek-OCR over the rendered images, writing `ocr/<dk>/NNNN.raw.md` (the raw
model output, kept for debugging) and `ocr/<dk>/NNNN.structured.json`.

```bash
mm-ingest ocr $PKG
```

> This is the expensive, non-reproducible step: vLLM's greedy decoding is not
> bit-identical across runs, so re-running produces different text. Everything it
> writes is ~30 MB.

### 5. Build the text

Merge the structured OCR output into readable markdown: repair sentences that
straddle a page boundary, annotate headers, then write `text/<dk>/NNNN.md`,
`text/<dk>/NNNN.structured.fixed.json` and the concatenated
`text/<dk>/document.md`. Deterministic.

```bash
mm-ingest build-text $PKG
```

### 6. Extract page numbers

OCR the printed page number from each page image (GLM-OCR) into
`page_numbers_raw/<dk>/NNNN.txt`. Rulebooks with an explicit `page_one_offset` in
the manifest skip the model call and get an empty file, so the step's output
covers every page either way.

```bash
mm-ingest page-number-ocr $PKG
```

### 7. Fix up page numbers

Reconcile the raw page numbers — interpolate gaps, repair misreads, handle
roman/arabic segments — and write `page_numbers/<dk>/NNNN.txt`. Rulebooks with an
explicit `page_one_offset` get their numbers computed from that offset here, so
every page has a page-number file regardless of route.

```bash
mm-ingest page-number-fixup $PKG
```

### 8. Add per-page metadata

Write `metadata/<dk>/NNNN.yaml` for each page: game/rulebook identifiers, page
ordinal, the cleaned `page_num`, and the page's character offsets within
`document.md`. **Required by everything downstream** — `page_to_document`
unguarded-reads this file.

```bash
mm-ingest add-metadata $PKG
```

> The metadata deliberately omits `game_version`. `import-documents` stamps the
> current version onto every document, id and `doc_id` as it loads them, so chunk
> files no longer have to be rebuilt after a version bump.

### 9. Build chunks

Split the page markdown into parent/child chunks used for retrieval. Sizes come
from `ingest.chunk_size` / `chunk_overlap` / `child_chunk_size` /
`child_chunk_overlap` (defaults 500/50 and 125/12; see
[Prerequisites](#prerequisites) for how to override). Reads `document.md` and
each page's metadata, writes `chunks/<dk>/parent/` and `chunks/<dk>/child/`.

```bash
mm-ingest build-chunks $PKG
```

### 10. Generate summaries and the presentation asset

Three chat-model steps that must run in this order — `generate-presentation`
reads the setting summary, and `import-documents` requires the presentation file:

```bash
mm-ingest generate-reference $PKG            # reference/game.md + reference/<dk>.md
mm-ingest generate-setting-summary $PKG      # setting/game.md
mm-ingest generate-presentation $PKG         # presentation.yaml (emoji + bg color)
```

- **`generate-reference`** writes `reference/game.md`, `reference/<dk>.md` and
  `reference/<dk>.short.md`. It no longer touches the manifest: `load_game_package`
  reads those files back as the `summary` fields, and `import_game_data` copies
  them into the stored game data, so skipping this yields a summary-less game.
- **`generate-setting-summary`** writes `setting/game.md` (theme/setting only, no
  mechanics).
- **`generate-presentation`** reads `setting/game.md` (unguarded) and
  writes `presentation.yaml`. `import-documents` unguarded-reads this file and
  lifts `unicode_character` → `emoji` and `background_color` into the game data,
  so a missing `presentation.yaml` fails the import with `FileNotFoundError`.

Optionally also generate example questions here — note these are imported with a
separate command, not by `import-documents` (see
[Example questions](#example-questions-per-game-not-per-version) below):

```bash
mm-ingest generate-example-questions $PKG    # example_questions.yaml (needs generate-reference)
```

### 11. Import into Postgres

Load chunks, embeddings, docstore parents, full-page markdown, and game data
(from the manifest + `presentation.yaml`) into the vectorstore/stores. Finishes
by publishing the new `game_version` as the current version for this `game_id`.
Example questions are **not** imported here — they're game-scoped and handled by a
separate command (see [Example questions](#example-questions-per-game-not-per-version)).

```bash
mm-ingest import-documents $PKG
```

**Import errors out if the package's `game_version` has already been imported**,
rather than silently overwriting it — bump the version first (see
[step 2](#2-stamp-a-version)). To deliberately re-import the
same version in place, pass `--overwrite`:

```bash
mm-ingest import-documents --overwrite $PKG
```

At this point the game is searchable by the local backend.

---

## Other generation steps

### Example questions (per game, not per version)

`generate-example-questions` reads the game-level and per-rulebook references
produced by `generate-reference` (skipping entirely if they don't exist) and
writes `example_questions.yaml`.

These questions are keyed by `game_id` — **not** `game_version` — so they carry
across game versions rather than being tied to a single ingest. For that reason
they are **not** imported by `import-documents`; persist them with a dedicated
command instead:

```bash
mm-ingest import-example-questions $PKG
```

This has no version check and doesn't touch the version-scoped document data, so
it's also the way to refresh questions for a game that's already been imported.

---

## Re-ingesting / updating

- **New content version:** bump the version before re-importing — see
  [step 2](#2-stamp-a-version). Without a bump, `import-documents` errors instead
  of overwriting the currently-published version.
- **A version bump alone needs no rebuild.** `metadata/` and `chunks/` no longer
  record the version; `import-documents` stamps it onto every document, id and
  `doc_id` as it loads. So `update-version` followed by `import-documents` is
  sufficient — re-running `build-chunks` is only needed when the *text* or the
  chunking settings changed.
- **Reprocess a version in place:** re-run `build-chunks` then
  `import-documents --overwrite` (this is exactly what
  [update_munchkin.sh](../update_munchkin.sh) does). Safe because chunking and
  import don't depend on the OCR services being up. `--overwrite` is required
  since the version already exists in the DB.
- **Change the chunking only:** `build-chunks` then `import-documents`. It reads
  `text/` and `metadata/`, so no render and no OCR.
- **Re-OCR a rulebook:** `ocr` then `build-text`, then everything downstream.
  `render` does not need re-running — the images are unchanged.
- **Remove superseded versions from the DB:** `mm-ingest clear-old-data`.

---

## Pushing to production

The steps above populate the **local** database. To load an already-ingested
package into the prod database, see the
[Ingest Data into Prod](../DEVELOPMENT.md) section of DEVELOPMENT.md — it uses
the same `import-documents` command with prod credentials from `.env.ingest.prod`.
