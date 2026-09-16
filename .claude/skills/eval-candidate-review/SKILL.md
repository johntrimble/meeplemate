---
name: eval-candidate-review
description: Review mined eval candidates for a game - validate each generated answer against the rulebook, flag and rewrite the wrong ones, cluster duplicate questions, and set promote/reject verdicts in decisions.json. Use when asked to review, check, correct, dedupe, promote or reject candidates under data/eval_gen/<run-id>/, or to work through the mm-eval review page.
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Reviewing mined eval candidates

`mm-eval mine-questions` and `mm-eval answer-candidates` produce a review queue,
not test cases. This skill is the human-judgement pass that turns that queue
into a golden set: read every question and answer against the rulebook, correct
the answers that are wrong, and decide which questions are worth keeping.

Everything lands in `data/eval_gen/<run-id>/decisions/decisions.json`. See
[docs/eval-corpus.md](../../../docs/eval-corpus.md) for the surrounding pipeline.

## Two independent axes

A candidate carries a verdict on the **question** and a flag on the **answer**.
They do not imply each other, and a card can carry both:

| field | meaning |
|---|---|
| `reviewer` | `"yes"` promote, `"no"` reject, `"later"` undecided, `null` unreviewed |
| `answer_suspect` | the generated answer is wrong or unusable |
| `corrected_answer` | the reference answer you wrote. **This is what promotes** |
| `note` | observations. Never promoted — do not put answer content here |

`promote` uses `corrected_answer` in preference to the agent's draft
([promote.py:164-170](../../../meeplemate/eval/promote.py#L164-L170)) and skips
rejected candidates even when one was written. So a bad question with a
corrected answer is fine: the correction stays on record and simply never ships.

**Only set `reviewer` when the user asks for promote/reject decisions.** Flagging
answers and writing corrections is a separate job; leave `reviewer` null for it.

## 1. Find the run and read the corpus first

```bash
ls data/eval_gen/<run-id>/answers/<game>/          # one yaml per candidate
cat data/eval_gen/<run-id>/mining/<game>/dedupe.json
```

Rulebook text lives under `data/ingestion/artifacts/<package>/`. The package
directory is usually the game id but not always — munchkin is `munchkin_rules`.
Resolve it via `game_id` in `data/ingestion/rules/*/rulebooks.yaml`.

```bash
PKG=waterdeep
ls data/ingestion/artifacts/$PKG/text/*/           # NNNN.md per page + document.md
for d in data/ingestion/artifacts/$PKG/text/*/; do basename "$d" | base64 -d; done
```

Directory names are base64 of the PDF filename. `document.md` is the whole book;
the numbered files are pages.

**Printed page numbers come from `page_numbers/`, never from the file index.**
Packages skip pages — waterdeep's `0003.md` is printed page 7, not 4. Citations
that are off by three look like hallucinations later.

```bash
for f in data/ingestion/artifacts/$PKG/page_numbers/*/*.txt; do
  echo "$(basename $f .txt) => $(cat $f)"; done
```

**Check the corpus for gaps before judging anything.** An answer cannot be better
than what was ingested. Waterdeep's is 13 OCR pages of a 24-page rulebook with
one page file empty, so whole sections simply are not there. Look for zero-byte
page files, missing printed numbers, and sections the table of contents promises
but the text lacks. Report gaps to the user — they are an ingest bug, and they
change what counts as a fair question.

Expect OCR damage in what *is* there: jumbled section order (an action landing
under the wrong Building's heading), LaTeX-wrapped glyphs, and mangled sentences.
Never quote a garbled sentence as evidence, and never trust the heading a passage
sits under without sanity-checking it against the game.

## 2. Review the answers

Read the full `response` of every candidate, not the opening line. These are the
failure modes that actually show up:

- **Leaked chain-of-thought.** The response is the model's deliberation, often
  opening `<reasoning>` with no closing tag — usually a token-limit truncation.
- **Fabricated quotes.** A sentence the model wrote itself, formatted as a
  blockquote with a page citation. The tell is a missing
  `<div data-quote-status="verified">` on that one quote.
- **Self-contradictory opener.** "No, X counts toward the limit" — the headline
  negates the body. Read them as wrong; the reader stops at the first word.
- **Answers a different question.** Asked about a Building's *action*, answers
  about the *Owner benefit*.
- **False-premise dodge.** The question presupposes a condition; the answer opens
  "No, only if…" instead of answering under the stated condition.
- **Invented rules.** A fabricated rationale for a correct conclusion, e.g.
  "agents in the Palace are not subject to reassignment" as if the Building were
  special. A right answer resting on a made-up rule is still a bad gold answer.
- **Right headline, false body.** One wrong sentence buried in an otherwise
  correct answer still poisons a reference answer.

Judge the conclusion against the rules first, then the reasoning and citations.
Flag when the answer contains a materially false statement, a fabricated quote,
or a broken response — not merely because it is thin or reads awkwardly.

Where the rulebook genuinely does not settle a point, say so in your summary
rather than inventing certainty.

## 3. Write corrected answers

Match the house format the generated answers use:

```
<One or two sentences answering the question directly.>

<Sentence introducing the rule.>

> <verbatim rulebook text>
>
> (<Rulebook Name>, p. <N>)

<More rules the same way, then a closing paragraph tying them together.>
```

Rules that matter, because these blockquotes *become* the test case's
`evidence:` via `find_quotes_in_text`:

- **Quote verbatim.** Copy from the page file. Use `...` to elide.
- **`<Rulebook Name>` must match a `name:` in `rulebooks.yaml` exactly.** It is
  written straight into the evidence entry, and `mm-eval retrieval` scores
  against it.
- **Pages must be numeric.** `(Book, p. iv)` parses to an empty ref name and the
  quote is silently dropped from evidence
  ([promote.py:95-111](../../../meeplemate/eval/promote.py#L95-L111)).
- **Every blockquote needs a citation line**, or it is dropped.
- **Do not write `<div data-quote-status="verified">` markers.** They mean the
  pipeline's verifier checked that span; writing them by hand fakes the signal.
  `clean_answer` strips them at promotion anyway, so nothing is lost.
- Two to four quotes is the norm. Prefer the passage that settles the point over
  the passage that merely mentions the topic.

## 4. Verify the quotes

Always, before persisting:

```bash
python3 .claude/skills/eval-candidate-review/verify_quotes.py \
    data/eval_gen/<run-id>/decisions/decisions.json
```

It checks each blockquote parses, names a real rulebook, cites an ingested page,
and appears verbatim on that page **of that book**. Fix every failure — a quote
that does not match reads as a retrieval regression months later, not as a review
bug. When the text is real but the citation is wrong, the failure line names the
page it is actually on.

Watch the book name as closely as the page number. Most packages hold several
rulebooks whose page numbers overlap completely — oathsworn's two both run from
page 1, warhammer has six — so `(Story Rule Book, p. 14)` on a sentence lifted
from the Encounter Rule Book's page 13 looks entirely plausible. Worse, the two
books sometimes print the *same* entry with different wording: oathsworn's
Crippled token is "cannot move" in one and "cannot willingly move" in the other,
so the wrong book can flip what the evidence says.

## 5. Duplicates and verdicts

**Do not trust `dedupe.json`.** On a single small rulebook every question scores
0.80–0.90 cosine because they share vocabulary, so at the 0.9 threshold genuine
duplicates come through marked `unique`. In the waterdeep run it caught none of
nineteen. Cluster by hand instead: group candidates by the *rule* they turn on
and compare answers. Same rule plus same answer plus no extra distinguishing
element means duplicate.

```bash
python3 - <<'PY'
import yaml, pathlib
for p in sorted(pathlib.Path("data/eval_gen/<run-id>/answers/<game>").glob("*-*.yaml")):
    y = yaml.safe_load(p.read_text())
    print(f"{y['candidate_id']}: {y['question']}")
PY
```

When a cluster collapses, keep the variant that is clearest and demands the most
of retrieval — the one needing two rules over the one needing one, the concrete
scenario over the abstract restatement. Reject a question that is a near-verbatim
restatement of a single rule turned into a yes/no; it tests recall of one
sentence.

Reject for:

- **Duplicate** of a kept candidate. Name the survivor in the note.
- **False premise** — describes a state the rules make impossible.
- **Incoherent terminology** — conflates Agents with Adventurers, pool with
  Tavern, or asks about a Building doing something it cannot do.
- **Genuine ambiguity** — two readings with opposite answers.

Every rejection gets a reason in `note`. It is the only record of why, and the
user needs it to overrule you.

Promote everything else. Trivial-but-correct and distinct is not grounds for
rejection unless the user says otherwise.

## 6. Persist

Back up first, then write.

```bash
cp data/eval_gen/<run-id>/decisions/decisions.json /tmp/.../decisions.bak
curl -s http://localhost:8765/api/decisions >/dev/null && echo "server up"
```

**The browser page is a writer too, and it writes the whole document.** If the
reviewer has the page open, it holds every verdict in JS memory and PUTs all of
them — including on `beforeunload`, so closing or reloading a stale tab is
enough to push its state. Whoever writes last wins the whole file.

The server guards this with a precondition: a PUT must carry
`base_updated_at`, the `updated_at` it last read, or it is refused with 409 and
the current document to merge onto. A run that has never been saved has no
`updated_at`, so its first write needs no base.

Two safe ways to write, in order of preference:

**Through the API**, echoing the version back:

```python
cur = json.load(urllib.request.urlopen(API))
doc = {**edit(cur), "base_updated_at": cur.get("updated_at")}
# On 409: rebase your edits onto the returned `current` and retry once.
```

**Straight to the file**, which is equivalent in shape — use the same settings
as [review_server.py:55](../../../meeplemate/eval/review_server.py#L55):

```python
path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
```

A direct write bumps nothing the open page knows about, so its next save
conflicts and it rebases — which is the intended outcome, but the reviewer sees
their page jump. Prefer the API when the server is up. If you do write the file
while a page is open, tell the user to reload it.

Document shape, matching the page's own export
([review_page.py:262-282](../../../meeplemate/eval/review_page.py#L262-L282)):

```json
{
  "run_id": "...", "reviewed": 0, "total": 0,
  "answers_flagged": 0, "answers_corrected": 0, "updated_at": "...Z",
  "decisions": [{
    "id": "...", "game": "...", "question": "...",
    "filter_recommendation": "keep|review|drop",
    "filter_verdict": "good|single_rule|false_premise|...",
    "reviewer": null, "answer_suspect": false,
    "corrected_answer": null, "note": null
  }]
}
```

`filter_recommendation` and `filter_verdict` come from the candidate's
`answer:` block. A row is only carried in the export if at least one of verdict,
flag, note or corrected answer is set.

**Always merge, never regenerate.** Read the existing decisions and preserve
every field you are not changing — a second pass that rebuilds the document from
scratch will silently drop the corrections from the first. Then assert it:

```python
for old in previous["decisions"]:
    new = by_id[old["id"]]
    assert new["answer_suspect"] == old["answer_suspect"]
    assert new["corrected_answer"] == old["corrected_answer"]
    assert (new["note"] or "").startswith(old["note"] or "")
```

## 7. Report

Give the user, in the terminal:

- counts, and the file you wrote
- corrections grouped by failure mode, each with the rule that settles it
- duplicate clusters as a table: rule, kept id, rejected ids
- rejections that are not duplicates, one line of reasoning each
- **the calls you are least sure of**, named, so they can overrule you cheaply
- corpus problems you hit along the way

Do not claim a quote is verified unless `verify_quotes.py` passed on it.
