# Eval Debugging Guide

## Overview

The eval system has two phases: **generation** (run the QA pipeline) and **evaluation** (score the outputs). Understanding both is key to debugging failures.

## Commands

```bash
# Generate + evaluate in one step
python -m meeplemate.eval run-qa <filter> --number-of-runs=N

# Generate only (creates run files without scoring)
python -m meeplemate.eval run-qa-gen <filter> --number-of-runs=N

# Evaluate existing runs
python -m meeplemate.eval run-qa-eval <filter> --group-run-id=<id>
```

- `<filter>` is a glob pattern matched against test case names (e.g., `grail_knights_break_tests`, `*break*`, `*`)
- Group run IDs are date-based (e.g., `2026-02-18`). Auto-generated if not specified.
- `--skip-existing` skips test cases that already have output files (useful for resuming interrupted runs)

## File Locations

- **Test cases**: `meeplemate/eval/test_cases.yaml` — defines queries, reference answers, and evidence
- **Generation runs**: `data/evals/generation_runs/{group_run_id}/` — one JSON file per run
  - Filename pattern: `{test_suite}__{test_case}.run{NNN}.json`
  - Each file contains the full LangGraph trace (inputs, outputs, child_runs)
- **Eval results**: `data/evals/qa_evals/{group_run_id}/` — DeepEval output files (JSON)
- **Eval code**: `meeplemate/eval/__main__.py` — metrics, generation, evaluation logic
- **Analysis helpers**: `meeplemate/eval/analysis.py` — utilities for finding and grouping runs

## Eval Metrics

Defined in `get_correctness_metric()` and the `_run_qa_eval()` function in `meeplemate/eval/__main__.py`.

- **Correctness (GEval)**: LLM-as-judge using Qwen3-30B via vLLM. Scores 0.0-1.0 with threshold 0.5. Compares actual output conclusion against reference answer.
- **Answer Relevancy**: DeepEval's built-in metric. Scores 0.0-1.0 with threshold 0.5. Checks whether the response addresses the input query.
- A test case **passes** only if ALL metrics meet their thresholds.

## Debugging a Failed Run

### Step 1: Identify the failure type

Read the run file's `outputs.response` field (first ~120 lines of the JSON):

```python
import json
with open("data/evals/generation_runs/{group_run_id}/{file}.json") as f:
    d = json.load(f)
print(d["outputs"]["response"])
```

Determine: Is the **conclusion** correct or wrong?

### Step 2: Check evidence (retrieval)

```python
evidence = d["outputs"]["evidence"]
for e in evidence:
    print(f'{e["rulebook_name"]} p.{e["page"]} - {e["content"][:100]}')
```

Key question: **Did the pipeline retrieve the critical evidence needed to answer correctly?** If the key passage is missing from evidence, this is a retrieval failure — no amount of reasoning improvement will help.

### Step 3: Check classification and subquestions

```python
# Classification is in the trace
cqs = d["outputs"].get("clarifying_questions", [])
print(f"Subquestions: {len(cqs)}")
for cq in cqs:
    print(f"  Q: {cq['question']}")
    # Check subquestion evidence too
    for e in cq.get("evidence", []):
        print(f"    Evidence: {e['rulebook_name']} p.{e['page']}")
```

- If classified SIMPLE when it should be COMPLEX: classification issue (see `meeplemate/prompts/analyze_question.md`)
- If COMPLEX but subquestions didn't retrieve key evidence: retrieval issue at the subquestion level

### Step 4: Check the answer vs format_answer chain

The pipeline has two steps after retrieval:
1. `answer_question` — generates `<reasoning>...</reasoning>` + answer text
2. `format_answer` — reformats the answer into markdown with blockquotes

To find the pre-format answer, search the trace for the answer field in the combine_subanswers path:

```python
# Find the answer_question output (before formatting)
def find_answers(obj, path=''):
    if isinstance(obj, dict):
        if 'answer' in obj and 'reasoning' in obj and isinstance(obj.get('answer', ''), str) and len(obj.get('answer', '')) > 100:
            print(f'{path}:')
            print(f'  {obj["answer"][:200]}')
        for k, v in obj.items():
            find_answers(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_answers(v, f'{path}[{i}]')
find_answers(d)
```

If the pre-format answer is wrong, the issue is in reasoning. If it's correct but the formatted response is wrong, the issue is in `format_answer` (see `meeplemate/prompts/markdown_format_response.md`).

### Step 5: Check eval scores

```python
with open("data/evals/qa_evals/{group_run_id}/{timestamp_file}") as f:
    data = json.load(f)

for tc in sorted(data["testCases"], key=lambda x: x["name"]):
    name = tc["name"]
    success = tc["success"]
    scores = {m["name"]: m["score"] for m in tc.get("metricsData", [])}
    print(f'{name}: success={success}, scores={scores}')
```

Watch for:
- **False negatives**: Correctness=1.0 but low Answer Relevancy → correct answer failed by relevancy metric
- **False positives**: Wrong answer but Correctness > 0.5 → eval model gave partial credit for discussing right concepts
- Scores near the 0.5 threshold are unreliable either way

## Common Failure Patterns

### Retrieval failure (most common)
- **Symptom**: Wrong conclusion despite correct reasoning structure
- **Diagnosis**: Critical evidence passage missing from `outputs.evidence` AND from all subquestion evidence
- **Root cause**: Semantic search doesn't surface the needed passage. The query or subquestion phrasing isn't similar enough to the passage content.
- **Fix**: Retrieval improvements (chunking strategy, query rewriting, adding more search queries)

### Misclassification (SIMPLE vs COMPLEX)
- **Symptom**: Question classified SIMPLE when it requires multi-step reasoning
- **Diagnosis**: Check classification in trace. SIMPLE means no subquestions were generated.
- **Root cause**: The analysis prompt didn't recognize the question as complex
- **Fix**: Update classification examples/signals in `meeplemate/prompts/analyze_question.md`

### Answer framing mismatch
- **Symptom**: Correct conclusion but opening line contradicts it (e.g., "No, they must take the test")
- **Diagnosis**: The `answer_question` step frames its Yes/No around a rephrased question rather than the user's actual question
- **Root cause**: Subquestion answers prime the LLM to think in a different framing (e.g., "immunity" vs "need to take")
- **Fix**: Prompt instructions in `_build_answer_prompt()` in `qa_graph.py` and `markdown_format_response.md`

### Eval false positive
- **Symptom**: Wrong answer passes eval (Correctness > 0.5)
- **Diagnosis**: The response discusses correct concepts at length but reaches the wrong conclusion. The eval model (Qwen3-30B) gives partial credit.
- **Root cause**: GEval criteria or threshold too lenient for binary yes/no questions
- **Fix**: Tighten the criteria in `get_correctness_metric()` or raise the threshold

## QA Pipeline Architecture (for debugging context)

The coordinating graph (`build_coordinating_agent_graph` in `qa_graph.py`) orchestrates:

```
analyze_question → [SIMPLE] → ask_simple_question → produce_response
                 → [COMPLEX] → ask_subquestions → combine_subanswers → produce_response
```

Within `combine_subanswers`, a nested `build_question_answer_graph` runs:
```
retrieve_data → answer_question → format_answer → validate_and_fix_response → provide_response
```

Key prompts:
- `meeplemate/prompts/analyze_question.md` — classification + subquestion generation
- `meeplemate/prompts/query_documents_guide_system_prompt.md` — system prompt for answer_question (rules for reasoning, category separation, exception handling)
- `meeplemate/prompts/markdown_format_response.md` — reformatting answer into markdown with blockquotes
