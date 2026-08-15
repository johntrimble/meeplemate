# Ingesting a Rulebook

Turning a rulebook PDF into searchable data is done with the `mm-ingest` CLI
([meeplemate/ingest/__main__.py](../meeplemate/ingest/__main__.py)), equivalently
`python -m meeplemate.ingest`. Run everything **from inside the dev container**.
`PKG` below is the output package directory, e.g. `./data/ingested/munchkin_rules/`.

## Quick reference

**1. Start the dependencies.** The OCR services are gated behind Compose
profiles — a plain `up -d` does not start them — and each needs an NVIDIA GPU.

```bash
docker compose up -d                                            # postgres + chat model
docker compose --profile deepseek-ocr up -d vllm-deepseek-ocr   # needed by `ocr`
docker compose --profile glm-ocr up -d vllm-glm-ocr             # needed by `page-number-ocr`
```

**2. Run the pipeline, in this order.**

```bash
PKG=./data/ingested/munchkin_rules/

mm-ingest init-game-package ./data/rules/munchkin_rules/ $PKG  # manifest.yaml, raw_documents/
mm-ingest update-version $PKG              # version.txt — required, even on a first ingest
mm-ingest render $PKG                      # images/         (CPU, deterministic)
mm-ingest ocr $PKG                         # ocr/            (GPU, non-reproducible)
mm-ingest build-text $PKG                  # text/
mm-ingest page-number-ocr $PKG             # page_numbers_raw/  (GPU)
mm-ingest page-number-fixup $PKG           # page_numbers/
mm-ingest add-metadata $PKG                # metadata/
mm-ingest build-chunks $PKG                # chunks/
mm-ingest generate-reference $PKG          # reference/
mm-ingest generate-setting-summary $PKG    # setting/
mm-ingest generate-presentation $PKG       # presentation.yaml
mm-ingest generate-example-questions $PKG  # example_questions.yaml
mm-ingest import-documents $PKG            # → Postgres; publishes the new game_version
mm-ingest import-example-questions $PKG    # → Postgres, keyed by game_id (separate step)
```

After `import-documents` the game is searchable by the local backend.
[ingest-rulebooks.sh](../ingest-rulebooks.sh) runs this loop over several games.

## The source package

Ingestion reads a hand-written **source directory** (`data/rules/<game>/`: the
PDFs plus `rulebooks.yaml`) and writes an **ingested package**
(`data/ingested/<game>/`, whose derived manifest is `manifest.yaml`).
`rulebooks.yaml` needs `name`, `game_id`, and a `rulebooks` list whose entries
each have a `name` and a `path` relative to the source directory (`url` and
`strategy` optional) — [full example](../data/rules/munchkin_rules/rulebooks.yaml):

```yaml
name: Munchkin
game_id: munchkin
rulebooks:
- name: Munchkin Rules
  path: munchkin_rules-1.pdf
  strategy: "ocr_only"
  url: "https://munchkin.game/site-munchkin/assets/files/1138/munchkin_rules-1.pdf"
```

## What it produces

`<dk>` is a rulebook's `document_key` (the base64 of its filename). Every path is
constructed in [layout.py](../meeplemate/ingest/layout.py).

```
data/ingested/<game>/
├── manifest.yaml                        # name, game_id, rulebooks, page counts  (init-game-package)
├── version.txt                          # game_version uuid                      (update-version)
├── raw_documents/<file>.pdf             # copied source PDF(s)                   (init-game-package)
├── images/<dk>/NNNN.png                 # rendered page image                    (render)
├── ocr/<dk>/NNNN.{raw,structured}.*     # raw + structured model output          (ocr)
├── text/<dk>/                           # per-page markdown + document.md        (build-text)
├── page_numbers_raw/<dk>/NNNN.txt       # OCR'd page number                      (page-number-ocr)
├── page_numbers/<dk>/NNNN.txt           # cleaned/interpolated page number       (page-number-fixup)
├── metadata/<dk>/NNNN.yaml              # per-page metadata + char offsets       (add-metadata)
├── chunks/<dk>/{parent,child}/*.json    # retrieval chunks                       (build-chunks)
├── reference/{game,<dk>,<dk>.short}.md  # rules reference + summaries            (generate-reference)
├── setting/{game,<dk>}.md               # theme/setting summary                  (generate-setting-summary)
├── presentation.yaml                    # emoji + background color               (generate-presentation)
└── example_questions.yaml               # suggested questions (optional)         (generate-example-questions)
```

## Gotchas

- `update-version` is required, including on a first ingest. `add-metadata`,
  `build-chunks`, `import-documents` and `rebuild-bm25` all refuse to run
  without `version.txt`.
- `import-documents` errors if that version is already in the DB rather than
  overwriting it. Bump the version to land a re-ingest alongside the live one,
  or pass `--overwrite` to re-import in place. Drop superseded versions with
  `mm-ingest clear-old-data`.
- A version bump alone needs no rebuild: `metadata/` and `chunks/` don't record
  the version — `import-documents` stamps it onto every id at load time.
- A missing `presentation.yaml` fails the import, and `generate-presentation`
  itself requires `generate-setting-summary` to have run.
- Skipping `generate-reference` imports a summary-less game.
- Example questions are keyed by `game_id`, not `game_version`, so they are
  loaded by `import-example-questions` and never by `import-documents`. That
  command has no version check — it's also how you refresh questions for a game
  that is already imported.

## Re-ingesting

- **Chunking settings changed:** `build-chunks` then
  `import-documents --overwrite` — no render, no OCR. This is what
  [update_munchkin.sh](../update_munchkin.sh) does.
- **Text changed (re-OCR):** `ocr` then `build-text` and everything downstream.
  `render` does not need re-running; the images are unchanged.
- **New content version:** `update-version` then `import-documents`.

## Configuration

The two OCR steps talk to separate vLLM services. Defaults come from
`IngestConfig` in [config.py](../meeplemate/config.py) and match the
[compose.yaml](../compose.yaml) service names and ports:

| Step | Compose service | Profile | Config key | Default URL |
|------|-----------------|---------|------------|-------------|
| `ocr` | `vllm-deepseek-ocr` | `deepseek-ocr` | `ingest.ocr` | `http://vllm-deepseek-ocr:8000/v1` |
| `page-number-ocr` | `vllm-glm-ocr` | `glm-ocr` | `ingest.page_number_ocr` | `http://vllm-glm-ocr:8080/v1` |

Override these — and the chunk sizes `ingest.chunk_size` / `chunk_overlap` /
`child_chunk_size` / `child_chunk_overlap` (defaults 500/50 and 125/12) — with an
environment variable (`MM_INGEST__OCR__MODEL=... mm-ingest ocr $PKG`) or an
`ingest:` block in the config file `MM_CONFIG_FILE` points at; env vars win.
Neither [config-dev.yaml](../config-dev.yaml) nor [config.yaml](../config.yaml)
sets `ingest:` today, so the defaults above are what runs.

## Pushing to production

The steps above populate the **local** database. To load an already-ingested
package into prod, see the [Ingest Data into Prod](../DEVELOPMENT.md) section of
DEVELOPMENT.md — same `import-documents` command, prod credentials from
`.env.ingest.prod`.
