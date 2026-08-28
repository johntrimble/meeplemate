# Running Evals

Answer quality is measured against golden test cases in
[`meeplemate/eval/test_cases.yaml`](../meeplemate/eval/test_cases.yaml). Run
everything **from inside the dev container**, with Postgres up
(`docker compose up -d`). The chat model is served from
`https://llm.johntrimble.dev/v1`, not the local vLLM container —
`create_eval_system` points there regardless of what compose is running.

The CLI is `mm-eval`, equivalently `python -m meeplemate.eval`.

Evaluation happens in two phases, and you can run them together or separately:

1. **Generation** — run the QA pipeline over the test cases, persisting a full
   trace per run.
2. **Evaluation** — score those saved traces.

## The commands

```bash
# Both phases in one go — the usual entry point
mm-eval run-qa <filter> --number-of-runs=3

# Generation only (writes run files, scores nothing)
mm-eval run-qa-gen <filter> --number-of-runs=3

# Score runs that already exist
mm-eval run-qa-eval <filter> --group-run-id=2026-08-21
```

`<filter>` is a glob matched against test case names — `munchkin__*`, `*break*`,
or `*` (the default) for everything.

| Flag | Applies to | Meaning |
|---|---|---|
| `--number-of-runs=N` | `run-qa`, `run-qa-gen` | Run each case N times. Use ≥3 for anything you plan to compare — single runs are noisy. |
| `--group-run-id=ID` | all three | Names the output group. Defaults to today's date, auto-incremented (`2026-08-21`, `2026-08-21-2`, …). |
| `--skip-existing` | `run-qa`, `run-qa-gen` | Skip cases that already have run files — resumes an interrupted generation. |
| `--overwrite` | `run-qa`, `run-qa-gen` | Regenerate run files that already exist. |
| `--skip-retrieval` | `run-qa`, `run-qa-gen` | Skip the retrieval scoring pass. |

**`run-qa` and `run-qa-eval` do not score the same things.** `run-qa` runs all
five metrics — Correctness, First-Pass Quote Validity, Quote Retention, Valid
Quote, Runaway Generations. `run-qa-eval` runs **Correctness only**. If you care
about quote health, score through `run-qa`.

## Cheaper checks

```bash
# Retrieval only, NO LLM calls — scores each arm against the evidence: quotes
mm-eval retrieval <filter> [--token-budget=15000]

# One ad-hoc question against a game
mm-eval ask <game_id> "<query>"
```

`retrieval` finishes in seconds rather than minutes and attributes a result to
retrieval rather than generation. Reach for it first on anything
retrieval-shaped.

## Growing the eval corpus

```bash
# No LLM calls — seed corpus stats and the seed list
mm-eval mine-questions <game_id> --dry-run

# One seed, end to end, with the chunk text the model actually saw
mm-eval mine-questions <game_id> --limit 1 --offset 30 --include-context-text

# Full sweep
mm-eval mine-questions <game_id>

# Retune the dedupe threshold over an existing run — NO LLM calls
mm-eval mine-questions <game_id> --run-id <id> --dedupe-only --dupe-threshold 0.88

# Thinking model, large output budget
mm-eval mine-questions <game_id> --config-file config-mining.yaml --max-answer-tokens 16384
```

Steps 1–2 of eval-corpus generation. For each parent chunk it names two rule
concepts the chunk leaves unresolved, retrieves those concepts across the corpus,
assembles a ≤12-chunk context, and generates candidate questions from it — then
flags near-duplicates against the batch and against `test_cases.yaml`.

Answer generation, answer-based filtering and human review are downstream and
manual. **This writes a review queue, not test cases**: promote a case into
`test_cases.yaml` by hand, with an answer you have verified yourself.

Retrieval here deliberately uses *mining* settings rather than production ones —
large k, no adaptive-k, no token budget, no document-order re-sort. Assembling
context behind the production cutoffs would restrict the corpus to questions
production retrieval already finds, biasing e2e scores optimistically and
increasingly so as retrieval is tuned.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--limit` / `--offset` | all / 0 | Window into the deterministic seed order. |
| `--concepts-per-seed` | 2 | Concepts named per seed; each becomes a retrieval query. |
| `--questions-per-seed` | 5 | Candidates generated per seed. |
| `--max-question-words` | 25 | Length target written into the generation prompt. Advisory — the model is asked, not clipped. |
| `--max-answer-tokens` | 6144 | Cap for the generation call. The config default (3072) is tuned for a QA answer and truncates reasoning plus N questions. |
| `--dense-k` | 40 | Dense candidates per concept. Effectively capped at 40 by `hnsw.ef_search`; raising it alone does not widen the arm. |
| `--bm25-k` | 30 | Lexical candidates per concept. |
| `--max-context-chunks` | 12 | Context size including the seed. Kept small — generation is a synthesis task and the served model degrades well before its nominal window. |
| `--adjacency-radius` | 1 | Drop retrieved chunks within N positions of the seed. Neighbours restate it rather than interact with it. |
| `--min-seed-chars` | 200 | Skip title pages, credits and component tables. |
| `--dupe-threshold` | 0.92 | Cosine similarity at or above which a candidate is flagged. |
| `--resume` | off | Skip seeds already in the run file, failures included. |
| `--dedupe-only` | off | Recompute dedupe over an existing file. Zero LLM calls. |
| `--drop-duplicates` | off | Omit near-duplicates instead of annotating them. |
| `--config-file` | `$MM_CONFIG_FILE` | Config YAML for this run. See below. |
| `--endpoint` | config | Override the chat endpoint for this run. |
| `--max-concept-tokens` | 8192 | Cap for the concept-extraction call. Thinking needs far more than non-thinking. |

Two numbers in the output are worth reading before the questions themselves:

- **`concept_retrieval_overlap`** per candidate — Jaccard overlap between the two
  concepts' retrieved parents. Near 1.0 means the model returned near-synonyms
  ("Stage Card" / "Stage Deck"), both arms fetched the same chunks, and the
  context is one cluster rather than two. The questions may still be fine; they
  just came from the prompt rather than from the retrieval.
- **The question-length histogram** — Boardbarian is used mostly from a phone, so
  a good candidate reads like something typed with thumbs. The failure mode is a
  question that recites the rule before asking about it: self-contained, fluent,
  and nothing a real user would type. The hand-written cases in
  `test_cases.yaml` have a median of **13 words**; the histogram and the
  `word_count` on each candidate are how you tell whether a run drifted off that.
- **The `max_similarity` histogram** — the threshold is a guess until you have
  seen the distribution. bge-small has a high similarity floor on short text, so
  read the 0.88–0.95 band by hand and re-run `--dedupe-only` to adjust.

Runs are auditable, not reproducible: sampling is at `temperature: 0.7`, so the
same seed yields different questions each time. `run.params` records what
produced a file; diversity across seeds is the point.

### Running against a thinking model

[`config-mining.yaml`](../config-mining.yaml) is a mining-specific config with
`enable_thinking: true` and a 32k output budget:

```bash
mm-eval mine-questions <game_id> --config-file config-mining.yaml \
    --max-answer-tokens 16384
```

`--config-file` **replaces** `$MM_CONFIG_FILE` rather than merging with it —
`YamlConfigSettingsSource` reads one file and does no layering — which is why
`config-mining.yaml` restates the pg URL and embedding model. Those halves have
to be kept in sync with `config-dev.yaml` by hand; nothing checks them, and a
mining run reading a different corpus than the eval it feeds will not announce
itself.

Two things to know before turning thinking on:

- **Output tokens go up roughly 26x.** Measured against this endpoint on a
  concept-extraction call: 96 tokens without thinking, 2,536 with. Both token
  caps have to move with it, and a call that overruns loses the whole seed.
- **You cannot read the thinking from here.** The endpoint *does* return the
  trace — a raw call comes back with `choices[0].message.reasoning` populated —
  but langchain's structured-output path drops it, leaving only `parsed` and
  `refusal` in `additional_kwargs`. The thinking happens and is billed; you just
  cannot inspect it through the CLI. Judge the effect by question quality across
  two runs over the same `--offset`/`--limit` window.
- **`chat_template_kwargs` must be set before the component system starts.**
  `build_openai_extra_body` runs when the `chat_model` component is constructed
  in `astart()`, so mutating `cfg.chat.models[0]` *inside* the
  `async with system.astart()` block is silently a no-op — the run looks like it
  used thinking and did not. `--config-file` is safe because `MM_CONFIG_FILE` is
  applied before `Config()`. This one costs an afternoon if you hit it in a
  throwaway script.

## Inspecting results

```bash
mm-eval print-run-summary <filter> --group-run-id=<id>  # summarize one group
mm-eval analyze <group-run-id> [--top-n=5]              # variance across runs, best/worst cases
mm-eval compare-qa <group-a> <group-b>                  # diff two groups
```

## Token spend

Token accounting lives outside the CLI, in
[`script/count_tokens.py`](../script/count_tokens.py):

```bash
python script/count_tokens.py <group-run-id>              # per test case (default)
python script/count_tokens.py <group-run-id> --by-node    # per LangGraph node
python script/count_tokens.py <group-run-id> --per-run    # per run file
python script/count_tokens.py <group-run-id> --per-call   # per LLM call
```

It also accepts paths instead of a group name. Its prices track
`RateLimitConfig`, so the dollar figures match what the rate limiter charges —
which is why it is also how you re-derive the rate-limit request estimate, from
the `in/run` and `out/run` columns of the `OVERALL` row. See
[rate-limits.md](rate-limits.md#re-deriving-the-estimate).

## Where the output lands

```
data/evals/
├── generation_runs/<group-run-id>/       # one file per run: the full LangGraph trace
│   └── <suite>__<case>.run<NNN>.json.gz  #   inputs, outputs, child_runs
└── qa_evals/<group-run-id>/              # deepeval scoring output

eval_gen/<game_id>/<run-id>.yaml          # mine-questions review queue
```

`eval_gen/` is tracked, unlike `data/` — the whole point of a review queue is
that someone reads it.

A run file's `outputs` carries `response`, `evidence`, and (for questions
classified COMPLEX) `clarifying_questions` — enough to tell a retrieval failure
from a reasoning one without opening the trace tree.

Run files are written gzip-compressed. The traces are highly repetitive JSON and
shrink by better than 10x, which matters because a single run is routinely
several MB and a group run is hundreds of them. Readers take either form and
prefer the uncompressed one, so run groups generated before compression keep
working untouched and can be backfilled whenever:

```bash
find data/evals/generation_runs -name '*.json' -exec gzip -6 {} +
```

Nothing in the eval CLI writes uncompressed runs any more, so reach for `zcat`
(or `gzip -dc`) when inspecting one by hand. The same layout, and the same
compression, applies under `data/evals/sampling_param_search/<group>/` — the
grid search reuses the same runner.

## Related

- [qa-flow.md](qa-flow.md) — what the pipeline being measured actually does
- Prompts live in [`meeplemate/prompts/`](../meeplemate/prompts/); metric
  definitions and their thresholds are in
  [`meeplemate/eval/metrics.py`](../meeplemate/eval/metrics.py)
