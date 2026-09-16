# Growing the eval corpus

Where new golden test cases come from. The output of this workflow is hand-
verified cases in
[`meeplemate/eval/test_cases.yaml`](../meeplemate/eval/test_cases.yaml); running
evals *against* those cases is [eval.md](eval.md).

Run everything **from inside the dev container**, with Postgres up
(`docker compose up -d`). The chat model is served from
`https://llm.johntrimble.dev/v1`, not the local vLLM container.

## Why this exists

The corpus was 42 hand-written cases across three games, 31 of them carrying
`evidence:`. That is too small to measure anything: at n≈15 per split the 95%
CI on correctness is roughly ±23 points, and a paired comparison needs at least
six questions to flip in one direction before it clears noise. Detecting a
change that helps 10% of questions and hurts 2% needs n≈100–150.

Corpus size — not split rigor, not metric choice — is the binding constraint on
every other eval improvement, including a dev/holdout split that could catch
overfitting.

## The workflow

| Step | Command | Output |
| --- | --- | --- |
| 1. Mine candidates | `mm-eval mine-questions` | `mining/<game>/seeds/*.yaml` |
| 2. Answer and filter | `mm-eval answer-candidates` | `answers/<game>/*.yaml` |
| 3. Review | `mm-eval review` | `decisions/decisions.json` |
| 4. Promote | `mm-eval promote` | `dataset/test_cases.yaml` |

Step 3 is the only manual one, and it is the binding constraint, not generation: a full Warhammer sweep would produce ~6,500
candidates, and the filter in step 2 removes only a few percent of them. Sample
rather than sweep, and treat every filter signal as a way to order the reading
rather than to avoid it.

Each step owns one directory and writes nothing outside it —
[`eval_gen_layout.py`](../meeplemate/eval/eval_gen_layout.py) is the single
source of truth for every path, the same rule `PackageLayout` follows for
ingest.

## 1. Mining candidate questions

```bash
# No LLM calls — seed corpus stats and the seed list
mm-eval mine-questions <game_id> --dry-run

# One seed, end to end, with the chunk text the model actually saw
mm-eval mine-questions <game_id> --limit 1 --offset 30 --include-context-text

# Partial sweep — 60 seeds spread across the whole game. Reach for this, not
# --limit, which takes a contiguous window and samples one chapter.
mm-eval mine-questions <game_id> --sample 60

# Every seed. Fine for a small game; Warhammer has 1,313 of them.
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

**This writes a review queue, not test cases.** Steps 2–4 turn it into one.

`--limit` takes a *contiguous* window, which on a large game means one chapter.
`--limit 12` on Warhammer returns twelve chunks from pages 0-5 of the Champions
of Chaos Army Book; `--sample 12` returns twelve spread across six rulebooks.
The failure is quiet — a contiguous run produces perfectly good questions that
all turn on the same handful of rules. Sampling is applied after
`--offset`/`--limit` and before `--resume`, so the seed set is stable across
invocations and a resumed run converges.

```bash
# 60 seeds spread across all 1,313 Warhammer parents
mm-eval mine-questions warhammer_5th_edition --sample 60
```

Retrieval here deliberately uses *mining* settings rather than production ones —
large k, no adaptive-k, no token budget, no document-order re-sort. Assembling
context behind the production cutoffs would restrict the corpus to questions
production retrieval already finds, biasing e2e scores optimistically and
increasingly so as retrieval is tuned.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--limit` / `--offset` | all / 0 | Window into the deterministic seed order. Contiguous — see `--sample`. |
| `--sample` | off | Take N seeds evenly spread across that window. **Use this, not `--limit`, for a partial sweep.** |
| `--concepts-per-seed` | 2 | Concepts named per seed; each becomes a retrieval query. |
| `--questions-per-seed` | 5 | Candidates generated per seed. |
| `--max-question-words` | 25 | Length target written into the generation prompt. Advisory — the model is asked, not clipped. |
| `--max-answer-tokens` | 6144 | Cap for the generation call. The config default (3072) is tuned for a QA answer and truncates reasoning plus N questions. |
| `--dense-k` | 40 | Dense candidates per concept. Effectively capped at 40 by `hnsw.ef_search`; raising it alone does not widen the arm. |
| `--bm25-k` | 30 | Lexical candidates per concept. |
| `--max-context-chunks` | 12 | Context size including the seed. Kept small — generation is a synthesis task and the served model degrades well before its nominal window. |
| `--adjacency-radius` | 1 | Drop retrieved chunks within N positions of the seed. Neighbours restate it rather than interact with it. |
| `--min-seed-chars` | 200 | Skip title pages, credits and component tables. |
| `--dupe-threshold` | 0.90 | Cosine similarity at or above which a candidate is flagged. |
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
- **The `max_similarity` histogram** — read the 0.85–0.92 band by hand and
  re-run `--dedupe-only` to adjust. The default was 0.92 and let obvious
  rewordings through: measured on the 2026-08-28 run, *"Can I give a Hireling an
  Item to carry during combat?"* and *"Can I equip an item to my Hireling
  mid-combat?"* sit at 0.914 and are the same question.

  **No threshold separates these cleanly.** In the 0.88–0.92 band roughly half
  are true duplicates and half are distinct questions about one rule — *"discard
  a Curse for a **Race** ability"* vs *"for a **Class** ability"* is 0.890 and
  both are worth having. 0.90 catches the unambiguous cases without killing
  those; going below ~0.88 trades real questions for a tidier batch. The residue
  is a job for the reviewer, not the threshold.

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

## 2. Answering and filtering

```bash
# Cost it first - no LLM calls
mm-eval answer-candidates <game_id> --run-id <id> --dry-run

# Answer and filter; one record per candidate under answers/<game>/
mm-eval answer-candidates <game_id> --run-id <id>

# Three answers per candidate; report the worst pairwise agreement
mm-eval answer-candidates <game_id> --run-id <id> --runs 3
```

Runs each candidate through the live QA pipeline, then judges the **question**
using the answer as evidence about it -- an answer that hedges, or
explains that a component does not exist, says the question was bad. Writes an
`answer:` block under each candidate and never deletes without `--drop`.

Three signals, kept separate in the output so a bad one can be ignored rather
than silently poisoning a verdict:

| Signal | Cost | What it catches |
| --- | --- | --- |
| `verified_quotes` | free | Answers the agent could not ground. Zero means the question is not answerable from this corpus. |
| `evidence_chunks` | free | Rule interactions. One passage means a lookup, not an interaction -- the mechanical version of what the generation prompt asks for and cannot enforce. |
| `verdict` | 1 LLM call | `false_premise`, `not_in_rules`, `single_rule`, `unclear`. What the answer reveals about the question. |
| `consistency` | `--runs` x | Ambiguity. The *lowest* pairwise similarity across repeated answers, so two-agree-one-doesn't is not averaged away. |

`recommendation` combines them: **drop** only for questions that are defective
(ungrounded, false premise, not in the rules); **review** for merely weak ones,
including every `single_rule` and every single-passage answer; **keep** for the
rest. A weak question is still a usable test case, and whether to promote it is
the reviewer's call rather than the filter's.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--run-id` | required | Group run id — the directory under `data/eval_gen/`. |
| `--runs` | 1 | Answers per candidate. `N>1` multiplies cost by N and populates `consistency`. |
| `--limit` | all | Answer at most N, contiguously. For costing a sweep before committing to it. |
| `--sample` | off | Answer N spread evenly across the run. **Use this, not `--limit`, for a partial pass** — candidates are in seed order. |
| `--concurrency` | 4 | Parallel candidates. Each is a full QA graph run. |
| `--include-duplicates` | off | Also answer near-duplicates. They are skipped by default -- a full QA run to re-derive an answer the queue already has. |
| `--drop` | off | Remove `drop` candidates instead of annotating them. |
| `--resume` | off | Skip candidates that already have a non-error answer. Errored ones are retried. |
| `--include-evidence-text` | off | Embed quoted passages. Debugging one run only. |

**`qa_service` does not return the inner graph's `valid` flag.** It returns the
*coordinating* graph's output, whose schema is `{response, evidence,
clarifying_questions}`; `valid` is dropped at that boundary despite the
`QAService` type annotation and the `cast` in `build_qa_service` saying
otherwise. Grounding is therefore read from the response text, counting the
`<div data-quote-status="verified">` markers that `materialize` wraps around
verified blockquotes. Reading `result["valid"]` instead silently marks *every*
answer ungrounded and drops the entire queue.

## 3. Reviewing

```bash
# Serve the review page; decisions persist under data/eval_gen/<run-id>/decisions/
mm-eval review munchkin one_deck_dungeon --run-id 2026-08-28 [--port 8765]
```

A two-pane console over every answered candidate in a run: question, the four
signals, the agent's draft answer with its verified quotes marked, and why the
filter recommended what it did. `j`/`k` to move, `1`/`2`/`3` to record a verdict
on the **question**, `f` to flag the **answer** as wrong or doubtful, plus a
free-text note.

A third field, **Reference answer**, is where a correction goes. *Start from
the draft* pre-fills it with the agent's answer minus the verified-quote
markers, so fixing a mostly-right answer is an edit rather than a retype —
which matters because retyping the rulebook quotes by hand invites
transcription errors in the text that becomes the case's `evidence:`. Leave it
empty and the draft is promoted as-is. `note` is for observations about the
case and is never promoted.

The **Needs an answer** filter lists exactly the outstanding work: flagged
draft, question promoted, no replacement written yet.

Each candidate also shows its **nearest neighbour** when the similarity is 0.85
or above, with that neighbour's question and — if you have already ruled on it —
your verdict, so *"0.897 similar to a candidate you promoted"* is visible at the
moment you decide. **Go to it** jumps there. This is deliberately a human
judgement rather than a lower `--dupe-threshold`: in the 0.88–0.92 band roughly
half are reworded duplicates and half are distinct questions about one rule, and
no cutoff separates them. The **Has a near twin** filter lists the whole band.

Those two axes are independent on purpose. A good question with a wrong answer
is one of the most valuable things the pass can find -- it is a real test case
*and* evidence the pipeline fails it -- so it is promoted and flagged, not
rejected. A high flag rate among otherwise-good questions is a finding, not a
nuisance.

Decisions are keyed by **run id**, not by game: one session spans every game in
the run and candidate ids already carry theirs. The file matches the page's
Export output, so a server-written file and a browser-exported one are
interchangeable.

The server binds `0.0.0.0` by default because it runs inside the dev container,
where a loopback socket is reachable only from inside the container while the
browser is on the host. VS Code forwards the port automatically; `compose.yaml`
publishes only 8000 and 5173, so add `8765:8765` there if you would rather not
rely on that. Nothing authenticates and the page writes into the repo, so keep
the forwarding local — `--host 127.0.0.1` restricts it to the container.

The page is re-rendered per request, so re-running `answer-candidates` against
the same run shows up on reload.

## 4. Promoting into test cases

```bash
mm-eval promote 2026-08-28 --dry-run    # report what would be written
mm-eval promote 2026-08-28              # → data/eval_gen/2026-08-28/dataset/test_cases.yaml

# Adopt the run into the tracked eval corpus
cp data/eval_gen/2026-08-28/dataset/test_cases.yaml \
   meeplemate/eval/test_cases/2026-08-28.yaml
```

Derived entirely from `decisions.json` plus the answer records, so re-running
after a correction gives a clean diff rather than a hand-merge. **Do not edit
the generated file** — fix the decision and re-run.

Each promoted candidate becomes a golden case:

| field | source |
| --- | --- |
| `query` | the question as reviewed |
| `reference_answer` | `corrected_answer` if the reviewer wrote one, else the agent's draft |
| `evidence` | the cited blockquotes in that answer |
| `source` | the candidate id, and whether the answer is the reviewer's or the agent's |

Only `reviewer: "yes"` is promoted. A rejected candidate is skipped **even when
a correction was written for it** — the answer may have been drafted while
working through the flagged list, and the verdict is the decision that counts.
`--include-rejected` overrides.

Evidence is extracted with
[`quote_util.find_quotes_in_text`](../meeplemate/quote_util.py), the same parser
the QA pipeline uses to locate and verify quotes, so what lands in a test case
is exactly what the pipeline considers a quote. A private regex here would drift
from it, and the failure shows up as evidence that no longer matches the corpus
— which reads as a retrieval regression rather than a promotion bug.

Two things the step reports rather than hides:

- **Blockquotes with an unparseable citation.** `extract_ref_name_and_page` only
  recognises numeric pages, so a quote cited `(Some Book, p. iv)` yields no
  rulebook and cannot become evidence. The case is still written, just with less
  evidence than its answer cites, and thin evidence reads as a retrieval problem
  much later.
- **Cases with no evidence at all**, which score for correctness but are
  invisible to `mm-eval retrieval`.

The verified-quote markers the pipeline wraps around its blockquotes are
stripped: they are a rendering hint for the frontend, not part of an answer, and
the hand-written cases do not carry them.

### Adopting a run into the corpus

The generated dataset lands under `data/` like every other step output, so it is
regenerable and backed up rather than tracked. **Adopting it is a separate,
explicit act**: copy it to `meeplemate/eval/test_cases/<run-id>.yaml`, which is
where the loader looks.

`meeplemate/eval/__init__.py` loads `test_cases.yaml` **plus** every
`test_cases/*.yaml`, sorted by filename. The hand-written file stays curated by
hand; each adopted run is a separate generated file that can be regenerated
wholesale without touching it.

The loader deliberately does **not** read from `data/`. If it did, a fresh clone
would evaluate against a different corpus than the machine that generated the
run, and an eval number would not mean the same thing on two checkouts.

Suites are named `Munchkin (mined 2026-08-28)` so provenance shows in eval
output and run file paths. This does not affect `--filter`, which fnmatches the
**case** name (`golden.name = test_case["name"]`) — so a filter cannot currently
select one mined batch. Filter by case-name pattern, or run everything.

## Where the output lands

```
data/eval_gen/<group-run-id>/
├── mining/<game-id>/
│   ├── seeds/<ordinal>-<hash8>.yaml   # one seed's candidates, written once
│   ├── dedupe.json                    # whole-batch reduce
│   └── run.json                       # params, seed stats
├── answers/<game-id>/
│   ├── <candidate-id>.yaml            # one answer, written once
│   └── run.json
├── decisions/decisions.json           # the reviewer's record
└── dataset/test_cases.yaml            # generated golden cases
```

**Every record is written once, in its final state.** An earlier design rebuilt
and rewrote one accumulating document after every seed, which is O(n²) in bytes
written — 22 GB to produce a 34 MB file on a full Warhammer sweep. Per-record
files also let two shells work the same run at different `--offset`s without
clobbering each other, and make `--resume` a directory listing instead of a
parse of the whole document.

Seed files carry the ordinal as a sortable prefix but are *identified* by a hash
of the parent id, and `--resume` matches on the parent id recorded inside each
record. Ordinals are positions in the filtered corpus order, so changing
`--min-seed-chars` shifts every one of them.

A failed seed still writes a record, holding the error. `--resume` then skips it
the way it skips a success, and "which seed failed and why" is answerable
without a run summary.

`dedupe.json` and `decisions.json` are the deliberate exceptions — one compares
every candidate against every other, the other is edited by a person. Both are
small and rewritten atomically in full.

None of this is tracked in git. It lives under `data/` and is backed up by
`data/backup_data.sh` like the eval runs. The tracked output of the whole
workflow is whatever you adopt into `meeplemate/eval/test_cases/`.

## Related

- [eval.md](eval.md) — running evals against the cases this produces
- [qa-flow.md](qa-flow.md) — the pipeline that answers candidates in step 2
- [ingestion.md](ingestion.md) — where the parent chunks being mined come from
