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

Growing that corpus is a separate workflow with its own commands and its own
output tree — see [eval-corpus.md](eval-corpus.md).

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
```

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

- [eval-corpus.md](eval-corpus.md) — where new test cases come from
- [qa-flow.md](qa-flow.md) — what the pipeline being measured actually does
- Prompts live in [`meeplemate/prompts/`](../meeplemate/prompts/); metric
  definitions and their thresholds are in
  [`meeplemate/eval/metrics.py`](../meeplemate/eval/metrics.py)
