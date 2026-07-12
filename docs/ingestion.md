# Ingesting a Rulebook

This guide covers the full pipeline for turning a rulebook PDF into searchable
data in MeepleMate: OCR → page numbers → metadata → chunking → summaries →
presentation → import.

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

The OCR steps talk to two vLLM services. The CLI reaches them at fixed internal
URLs (see [__main__.py](../meeplemate/ingest/__main__.py)) that match the
corresponding [compose.yaml](../compose.yaml) service names and ports:

| Step | Compose service | Compose profile | URL used by the CLI |
|------|-----------------|-----------------|---------------------|
| `ocr` | `vllm-deepseek-ocr` | `deepseek-ocr` | `http://vllm-deepseek-ocr:8000/v1` |
| `page-number-ocr` | `vllm-glm-ocr` | `glm-ocr` | `http://vllm-glm-ocr:8080/v1` |

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
init-game-package          → raw_documents/, rulebooks.yaml (game_version, page_count)
        │
        ▼
      ocr                  → NNNN.png, NNNN.md, document.md,
        │                    NNNN.structured.json, NNNN.structured.fixed.json
        ├───────────────► page-number-ocr   → NNNN.page_number.raw.txt   (reads NNNN.png)
        │                        │ (soft)
        │                        ▼
        │                 page-number-fixup  → NNNN.page_number.txt
        │                        │
        ▼                        ▼
      add-metadata (reads document.md, NNNN.md, NNNN.page_number.txt)
        │                  → NNNN.metadata.yaml
        │
        │  Everything below reads NNNN.md + NNNN.metadata.yaml (via page_to_document)
        ├──────────────┬─────────────────────┐
        ▼              ▼                     ▼
  build-chunks   generate-reference   generate-setting-summary
   → chunks/      → summary.md,         → game_setting_summary.md
        │           manifest summaries         │
        │              │                       ▼
        │              ▼                 generate-presentation (reads game_setting_summary.md)
        │        generate-example-questions      → presentation.yaml
        │         (reads summary.md; soft)         │
        │              → example_questions.yaml    │
        ▼              ▼                           ▼
                    import-documents  ◄────────────┘
   (reads: NNNN.md, NNNN.metadata.yaml, chunks/, presentation.yaml [required],
           example_questions.yaml [optional], manifest summaries [used if present])
```

### What `import-documents` actually requires

`import-documents` will **fail** unless these have run (unguarded reads in
`run_import_documents` / `import_game_data`, [dataimport.py](../meeplemate/ingest/dataimport.py)):

- `ocr` → page `NNNN.md`
- `add-metadata` → `NNNN.metadata.yaml` (read by `page_to_document`)
- `build-chunks` → `chunks/`
- `generate-presentation` → `presentation.yaml` — which in turn **requires**
  `generate-setting-summary` (it unguarded-reads `game_setting_summary.md`).

And `add-metadata` itself requires `page-number-fixup` (it unguarded-reads
`NNNN.page_number.txt`), which normally follows `page-number-ocr`.

It will **succeed but produce a degraded game** without:

- `generate-reference` — writes the game-level and per-rulebook `summary` fields
  **into the manifest**, and `import_game_data` deep-copies the manifest into the
  stored game data. Skip it and the game imports with no summary. It's also a
  prerequisite for `generate-example-questions`.
- `generate-example-questions` — `example_questions.yaml` is imported only if it
  exists (existence-guarded).

---

## Package layout

Ingestion operates on a **game package** — a directory containing a
`rulebooks.yaml` manifest plus the source PDFs. There are two directories in
play:

- **Source directory** (e.g. `data/rules/<game>/`): the raw PDFs plus a
  hand-written `rulebooks.yaml`.
- **Ingested / output directory** (e.g. `data/ingested/<game>/`): the working
  package that every subsequent step reads and writes. `init-game-package`
  creates this from the source directory.

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

### What the pipeline produces

After `init-game-package`, the output directory holds a `raw_documents/` copy of
the PDFs and one sub-directory per rulebook (named by a base64 `document_key`).
As the steps run, each page accumulates artifacts inside that sub-directory, and
several package-level assets get written alongside the manifest:

```
data/ingested/<game>/
├── rulebooks.yaml                    # manifest (game_version, page_count, summaries)
├── raw_documents/<file>.pdf          # copied source PDF(s)                 (init-game-package)
├── summary.md                        # game-level rules reference           (generate-reference)
├── game_setting_summary.md           # theme/setting summary                (generate-setting-summary)
├── presentation.yaml                 # emoji + background color             (generate-presentation)
├── example_questions.yaml            # suggested questions (optional)       (generate-example-questions)
└── <document_key>/
    ├── 0000.png                      # rendered page image                  (ocr)
    ├── 0000.md                       # per-page OCR'd markdown              (ocr)
    ├── document.md                   # concatenated full-document markdown  (ocr)
    ├── 0000.structured.json          # structured OCR output                (ocr)
    ├── 0000.structured.fixed.json    # sentence-/header-fixed structure     (ocr)
    ├── 0000.page_number.raw.txt      # OCR'd page number                    (page-number-ocr)
    ├── 0000.page_number.txt          # cleaned/interpolated page number     (page-number-fixup)
    ├── 0000.metadata.yaml            # per-page metadata + char offsets     (add-metadata)
    ├── summary.md / summary.short.md # per-rulebook summaries               (generate-reference)
    └── chunks/                       # parent + child chunk JSON            (build-chunks)
```

---

## The pipeline (run in order)

Below, `PKG` is the output package directory, e.g. `./data/ingested/munchkin_rules/`.
Steps 1–8 are all required for a complete import; see
[What `import-documents` actually requires](#what-import-documents-actually-requires)
for exactly which are hard vs. degrading.

### 1. Initialize the package

Copies the PDFs into `raw_documents/`, assigns a `document_key` per rulebook,
computes `page_count` for each PDF, and stamps a `game_version` UUID into the
manifest.

```bash
mm-ingest init-game-package ./data/rules/munchkin_rules/ ./data/ingested/munchkin_rules/
```

> Takes `<input>` (source dir) and `<output>` (package dir). Every later step
> takes only the package dir.
>
> `init-game-package` stamps a fresh `game_version` UUID only when the manifest
> doesn't already have one. Source manifests normally omit it, so a first-time
> init (or a re-init from source) gets a new version automatically.

### 1.5. Bump the version (re-ingest only)

**Skip this on a first-time ingest.** If this game has **already been imported**
and you're re-processing the *existing* package (e.g. re-running `ocr` /
`build-chunks` in place rather than re-initializing from source), the manifest
still carries the previously-imported `game_version`. In that case
`import-documents` (step 8) will **refuse to run** rather than overwrite the
published version. To land the re-ingest as a *new* version — keeping the
currently-live version untouched until you're ready to swap — stamp a fresh
version first:

```bash
mm-ingest update-version $PKG
```

`import-documents` stores all data under the manifest's `game_version` and
publishes it as the current version for the `game_id` (step 8). So this only needs
to happen any time before the import. After the new version is live, remove the
superseded one with `mm-ingest clear-old-data`. (If you instead want to re-import
the *same* version in place, skip this and use `import-documents --overwrite`.)

### 2. OCR the pages

Renders each page to a PNG and runs DeepSeek-OCR. Produces, per page, `NNNN.md`
and `NNNN.structured.json`; then a fix-up pass writes `NNNN.structured.fixed.json`
(dangling-sentence / header cleanup) and the concatenated `document.md` per
rulebook. `document.md` and the page markdown are inputs to nearly every later
step.

```bash
mm-ingest ocr $PKG
```

### 3. Extract page numbers

OCR the printed page number from each page image (GLM-OCR) into
`NNNN.page_number.raw.txt`.

```bash
mm-ingest page-number-ocr $PKG
```

### 4. Fix up page numbers

Reconcile the raw page numbers — interpolate gaps, repair misreads, handle
roman/arabic segments — and write the cleaned `NNNN.page_number.txt`. Missing raw
files are tolerated (treated as blank and interpolated), so this step soft-depends
on step 3.

```bash
mm-ingest page-number-fixup $PKG
```

### 5. Add per-page metadata

Write `NNNN.metadata.yaml` for each page: game/rulebook identifiers, page
ordinal, the cleaned `page_num` (from step 4), and the page's character offsets
within `document.md`. **Required by everything downstream** — `page_to_document`
unguarded-reads this file.

```bash
mm-ingest add-metadata $PKG
```

### 6. Build chunks

Split the page markdown into parent/child chunks used for retrieval
(parent 500/50, child 125/12 tokens). Reads `document.md` and each page's
`metadata.yaml` (needs the `start_index` offset), writes `chunks/`.

```bash
mm-ingest build-chunks $PKG
```

### 7. Generate summaries and the presentation asset

Three chat-model steps that must run in this order — `generate-presentation`
reads the setting summary, and `import-documents` requires the presentation file:

```bash
mm-ingest generate-reference $PKG            # summary.md + manifest summaries
mm-ingest generate-setting-summary $PKG      # game_setting_summary.md
mm-ingest generate-presentation $PKG         # presentation.yaml (emoji + bg color)
```

- **`generate-reference`** writes the game-level `summary.md`, per-rulebook
  `summary.md` / `summary.short.md`, and folds the summaries back into the
  manifest. `import_game_data` copies those manifest summaries into the stored
  game data, so skipping this yields a summary-less game.
- **`generate-setting-summary`** writes `game_setting_summary.md` (theme/setting
  only, no mechanics).
- **`generate-presentation`** reads `game_setting_summary.md` (unguarded) and
  writes `presentation.yaml`. `import-documents` unguarded-reads this file and
  lifts `unicode_character` → `emoji` and `background_color` into the game data,
  so a missing `presentation.yaml` fails the import with `FileNotFoundError`.

Optionally also generate example questions here (see
[Example questions](#example-questions-per-game-not-per-version) below):

```bash
mm-ingest generate-example-questions $PKG    # example_questions.yaml (needs generate-reference)
```

### 8. Import into Postgres

Load chunks, embeddings, docstore parents, full-page markdown, game data (from
the manifest + `presentation.yaml`), and — if present — example questions into
the vectorstore/stores. Finishes by publishing the new `game_version` as the
current version for this `game_id`.

```bash
mm-ingest import-documents $PKG
```

**Import errors out if the manifest's `game_version` has already been imported**,
rather than silently overwriting it — bump the version first (see
[step 1.5](#15-bump-the-version-re-ingest-only)). To deliberately re-import the
same version in place, pass `--overwrite`:

```bash
mm-ingest import-documents --overwrite $PKG
```

At this point the game is searchable by the local backend.

---

## Other generation steps

### Example questions (per game, not per version)

`generate-example-questions` reads the game-level and per-rulebook `summary.md`
files produced by `generate-reference` (falling back to the manifest `summary`,
and skipping entirely if neither exists). `import-documents` then imports
`example_questions.yaml` **only if it exists** (existence-guarded, so it's
optional).

These questions are keyed by `game_id` — **not** `game_version` — so they carry
across game versions rather than being tied to a single ingest, which is why
they're handled differently from the version-scoped chunks and presentation data.

To persist just the example questions to the DB without re-importing everything:

```bash
mm-ingest import-example-questions $PKG
```

---

## Re-ingesting / updating

- **New content version:** bump the version before re-importing — see
  [step 1.5](#15-bump-the-version-re-ingest-only). Without a bump,
  `import-documents` errors instead of overwriting the currently-published
  version.
- **Reprocess a version in place:** re-run `build-chunks` then
  `import-documents --overwrite` (this is exactly what
  [update_munchkin.sh](../update_munchkin.sh) does). Safe because chunking and
  import don't depend on the OCR services being up. `--overwrite` is required
  since the version already exists in the DB.
- **Remove superseded versions from the DB:** `mm-ingest clear-old-data`.

---

## Pushing to production

The steps above populate the **local** database. To load an already-ingested
package into the prod database, see the
[Ingest Data into Prod](../DEVELOPMENT.md) section of DEVELOPMENT.md — it uses
the same `import-documents` command with prod credentials from `.env.ingest.prod`.
