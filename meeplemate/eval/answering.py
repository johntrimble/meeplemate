"""Steps 3-4 of eval-corpus generation: answer the mined candidates, then filter.

`mine-questions` writes a review queue. At default settings the four games we
care about would produce ~7,400 candidates, and the binding constraint is no
longer generation but the human reading them. This module exists to cut that
pile down before anyone opens it.

Three independent signals, deliberately kept separate in the output so a bad
one can be ignored rather than silently poisoning a verdict:

1. **Grounding** (free) - the number of quotes in the answer that verified
   against the corpus, plus the evidence chunks the answer cites. An answer with
   no verified quote is one the agent could not ground, which in practice means
   the question is not answerable from this corpus.

2. **Interaction** (free) - the number of *distinct* evidence chunks the answer
   cites. This is the mechanical version of the property the generation prompt
   asks for and cannot enforce: a question settled by one passage is a lookup,
   not a rule interaction. It is a proxy, not proof - the agent can cite two
   chunks redundantly, or find both rules already colocated in one - so it
   ranks candidates for review rather than deciding them.

3. **Sense** (one LLM call) - a judge reading the question *and* the answer.
   This is the step that catches what the answer reveals about the question:
   that the game has no such component, that the premise is false, that the
   rulebook simply does not cover it.

Nothing here drops a candidate by default. `mine-questions` annotates rather
than deletes for the same reason, and the reason is the same: this file is
reviewed by a person, and "the filter wanted this gone" is information they may
disagree with.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterable, Mapping, Optional, Sequence, TypedDict

import numpy as np
import structlog
from langchain_core.prompts import ChatPromptTemplate

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Evidence (pure)
# ---------------------------------------------------------------------------


VERIFIED_QUOTE_MARKER = '<div data-quote-status="verified">'


def count_verified_quotes(response: str) -> int:
    """Quotes in the answer that verified against the corpus.

    `materialize` (qa_graph.py) wraps each verified blockquote in this marker as
    the last step before the response string is formed, and deliberately leaves
    unverified quotes unmarked -- absence of the marker is the signal.

    This is read off the response text rather than off the graph state because
    `qa_service` returns the COORDINATING graph's output, whose schema is
    `CoordinationOutputState` -- `{response, evidence, clarifying_questions}`.
    The inner graph's `valid` flag is dropped at that boundary, so
    `result["valid"]` is always absent no matter what the `QAService` type
    annotation and the `cast` at the end of `build_qa_service` claim. Reading it
    there silently marks every answer ungrounded.

    Counting verified quotes is also stricter than `valid` was: `valid` means
    "no INVALID quotes", which is trivially true of an answer containing no
    quotes at all.
    """
    return (response or "").count(VERIFIED_QUOTE_MARKER)


EvidenceKey = tuple[str, str, int]


def evidence_key(chunk: Mapping[str, Any]) -> EvidenceKey:
    """Identity of one evidence chunk.

    ``Chunk`` (qa_graph.py) carries no id, so identity is composed from the
    fields that do survive the graph: rulebook, page and start offset. Page is
    normalised to ``str`` because the corpus has genuine ``page`` /
    ``page_ordinal`` drift and the same chunk can arrive with either an int or a
    string page depending on whether a page number was OCR'd.
    """
    return (
        str(chunk.get("rulebook_name") or ""),
        str(chunk.get("page") if chunk.get("page") is not None else ""),
        int(chunk.get("start_index") or 0),
    )


@dataclass(frozen=True)
class EvidenceStats:
    chunk_count: int
    rulebook_count: int
    page_count: int
    rulebooks: tuple[str, ...]


def summarise_evidence(chunks: Sequence[Mapping[str, Any]]) -> EvidenceStats:
    """Distinct-chunk / distinct-rulebook counts over an answer's evidence.

    Deduplicates: the graph can carry the same chunk through several stages
    (analysis, subquestions, refinement) and emit it more than once. Counting
    raw list length instead would report every multi-stage answer as a rule
    interaction.
    """
    keys = {evidence_key(c) for c in chunks}
    books = {k[0] for k in keys if k[0]}
    pages = {(k[0], k[1]) for k in keys}
    return EvidenceStats(
        chunk_count=len(keys),
        rulebook_count=len(books),
        page_count=len(pages),
        rulebooks=tuple(sorted(books)),
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


JUDGE_TEMPLATE = """\
You are auditing a candidate question for a board game rules test set for the game
"{{game_name}}".

A rules assistant was asked the question and produced the answer below from the rulebooks.
Your job is to judge the QUESTION, using the answer as evidence about it. You are not
grading the answer.

<question>
{{question}}
</question>

<answer>
{{answer}}
</answer>

<evidence_passages>
{{#evidence}}
<passage rulebook="{{rulebook_name}}" page="{{page}}">
{{content}}
</passage>
{{/evidence}}
</evidence_passages>

Classify the question as exactly one of:

- "good" - a real question about this game, settled by the rules, and settling it needs more
  than one rule. This is what we are looking for.
- "single_rule" - a real, correctly-answered question, but one rule answers it outright. Fine
  as a question, weak as a test of a rules assistant.
- "false_premise" - the question assumes something that is not true of this game: a component,
  keyword, state or action that does not exist, or an interaction that cannot arise.
- "not_in_rules" - a coherent question the rulebooks do not answer. The answer hedges, says
  the rules are silent, or reaches for something outside them.
- "unclear" - the question is ambiguous, self-contradictory, or cannot be understood without
  seeing the passage it was generated from.

Judge on the game's own terms. An answer that confidently cites real rules is evidence the
premise holds; an answer that hedges, contradicts itself, or explains that something does not
exist is evidence it does not.

Give your response in the following JSON format:

{
"reasoning": "<one or two sentences of evidence for the verdict>",
"verdict": "<good|single_rule|false_premise|not_in_rules|unclear>",
"rules_involved": ["<short name of each distinct rule the answer leans on>"]
}
"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [("human", JUDGE_TEMPLATE)],
    template_format="mustache",
)


VERDICTS = ("good", "single_rule", "false_premise", "not_in_rules", "unclear")


class JudgeResult(TypedDict):
    reasoning: Annotated[str, ..., "Evidence for the verdict, one or two sentences."]
    verdict: Annotated[str, ..., "One of good, single_rule, false_premise, not_in_rules, unclear."]
    rules_involved: Annotated[
        list[str], ..., "Short name of each distinct rule the answer leans on."
    ]


def normalise_verdict(raw: Any) -> str:
    """Coerce the judge's verdict onto the known set.

    Structured output constrains the schema, not the vocabulary: the model
    returns "Good", "single-rule", or an invented sixth label often enough that
    an unmatched value has to mean something safe. It maps to "unclear", which
    routes to review rather than to either keep or drop.
    """
    v = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return v if v in VERDICTS else "unclear"


# ---------------------------------------------------------------------------
# Recommendation (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    status: str  # keep | review | drop
    reasons: tuple[str, ...]


def recommend(
    *,
    verified_quotes: int,
    stats: EvidenceStats,
    verdict: str,
    consistency: Optional[float] = None,
    min_consistency: float = 0.75,
) -> Recommendation:
    """Combine the three signals into keep / review / drop.

    "drop" is reserved for candidates that are defective as questions -
    ungrounded, false-premised, or unanswerable from the corpus. A question that
    is merely *weak* (single rule) lands in "review", not "drop": it is still a
    usable test case, just not the interaction case we were mining for, and that
    is a call for the person promoting it into test_cases.yaml.

    The signals are checked in order of how much they are trusted. Grounding is
    mechanical and comes from the pipeline under test; the judge is one LLM call
    and is trusted less; the evidence-count proxy is trusted least and can only
    downgrade a keep to a review, never produce a drop.
    """
    reasons: list[str] = []

    if verified_quotes <= 0:
        reasons.append("answer contains no verified quotes")
    if stats.chunk_count == 0:
        reasons.append("answer cites no evidence")
    if verdict in ("false_premise", "not_in_rules"):
        reasons.append(f"judge: {verdict}")
    if reasons:
        return Recommendation("drop", tuple(reasons))

    if verdict == "unclear":
        reasons.append("judge: unclear")
    if verdict == "single_rule":
        reasons.append("judge: answerable from one rule")
    if stats.chunk_count == 1:
        reasons.append("answer cites a single passage")
    if consistency is not None and consistency < min_consistency:
        reasons.append(f"answers disagree across runs ({consistency:.2f})")
    if reasons:
        return Recommendation("review", tuple(reasons))

    return Recommendation(
        "keep",
        (f"{stats.chunk_count} passages across {stats.rulebook_count} rulebook(s)",),
    )


def answer_consistency(vectors: np.ndarray) -> Optional[float]:
    """Lowest pairwise cosine among repeated answers to the same question.

    The minimum rather than the mean: three runs where two agree and one does
    not is exactly the case worth surfacing, and a mean would bury it. ``None``
    for a single run - there is nothing to compare, which is different from
    perfect agreement and must not be reported as 1.0.
    """
    if vectors is None or len(vectors) < 2:
        return None
    a = np.asarray(vectors, dtype=np.float64)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    sims = a @ a.T
    n = len(a)
    return float(min(sims[i, j] for i in range(n) for j in range(i + 1, n)))


# ---------------------------------------------------------------------------
# Result assembly (pure)
# ---------------------------------------------------------------------------


@dataclass
class AnswerResult:
    candidate_id: str
    question: str
    response: str = ""
    verified_quotes: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    stats: EvidenceStats = field(
        default_factory=lambda: EvidenceStats(0, 0, 0, ())
    )
    verdict: str = "unclear"
    judge_reasoning: str = ""
    rules_involved: list[str] = field(default_factory=list)
    consistency: Optional[float] = None
    alternates: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def answer_block(
    result: AnswerResult, *, include_evidence_text: bool = False
) -> dict[str, Any]:
    """The ``answer:`` mapping written under a candidate in the run YAML.

    ``recommendation`` leads because it is what a reviewer sorts on. Evidence
    text is omitted by default for the same reason the mining context is: a
    few hundred candidates each carrying several kilobytes of quoted rulebook
    turns a reviewable file into an unreviewable one.
    """
    if result.error is not None:
        return {"status": "error", "error": result.error}

    rec = recommend(
        verified_quotes=result.verified_quotes,
        stats=result.stats,
        verdict=result.verdict,
        consistency=result.consistency,
    )
    block: dict[str, Any] = {
        "recommendation": rec.status,
        "reasons": list(rec.reasons),
        "verdict": result.verdict,
        "judge_reasoning": result.judge_reasoning,
        "rules_involved": list(result.rules_involved),
        "verified_quotes": result.verified_quotes,
        "evidence_chunks": result.stats.chunk_count,
        "evidence_rulebooks": result.stats.rulebook_count,
        "rulebooks": list(result.stats.rulebooks),
        "response": result.response,
    }
    if result.consistency is not None:
        block["consistency"] = round(result.consistency, 4)
        block["alternate_responses"] = result.alternates
    if include_evidence_text:
        block["evidence"] = [
            {
                "rulebook": c.get("rulebook_name", ""),
                "page": c.get("page", ""),
                "content": c.get("content", ""),
            }
            for c in result.evidence
        ]
    return block


def candidates_to_answer(
    doc: Mapping[str, Any],
    *,
    resume: bool = False,
    include_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """Which candidates in a run document still need answering.

    Near-duplicates are skipped by default. Answering them costs a full QA graph
    run each to re-derive an answer for a question the queue already contains,
    and the reviewer will look at the keeper.
    """
    out: list[dict[str, Any]] = []
    for cand in doc.get("candidates") or ():
        if resume and isinstance(cand.get("answer"), Mapping):
            if cand["answer"].get("status") != "error":
                continue
        if not include_duplicates:
            status = (cand.get("dedupe") or {}).get("status")
            if status not in (None, "unique"):
                continue
        out.append(cand)
    return out


def apply_answers(
    doc: dict[str, Any],
    blocks: Mapping[str, dict[str, Any]],
    *,
    drop: bool = False,
) -> dict[str, Any]:
    """Attach answer blocks to their candidates, optionally dropping the rejects.

    Mutates and returns ``doc`` so the caller can flush the whole file after
    each completed candidate; a run of several hundred against a shared endpoint
    will be interrupted at least once.
    """
    kept: list[dict[str, Any]] = []
    for cand in doc.get("candidates") or ():
        block = blocks.get(cand.get("id"))
        if block is not None:
            cand["answer"] = block
        if drop and (cand.get("answer") or {}).get("recommendation") == "drop":
            continue
        kept.append(cand)
    doc["candidates"] = kept
    return doc


def recommendation_summary(doc: Mapping[str, Any]) -> dict[str, int]:
    counts = {"keep": 0, "review": 0, "drop": 0, "error": 0, "unanswered": 0}
    for cand in doc.get("candidates") or ():
        ans = cand.get("answer")
        if not isinstance(ans, Mapping):
            counts["unanswered"] += 1
        elif ans.get("status") == "error":
            counts["error"] += 1
        else:
            counts[ans.get("recommendation", "review")] += 1
    return counts


def verdict_summary(doc: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {v: 0 for v in VERDICTS}
    for cand in doc.get("candidates") or ():
        ans = cand.get("answer")
        if isinstance(ans, Mapping) and ans.get("verdict") in counts:
            counts[ans["verdict"]] += 1
    return counts


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


@dataclass
class AnsweringDeps:
    qa_service: Any
    manifest: Any
    judge_chain: Any
    embeddings: Any
    game_name: str


async def answer_once(deps: AnsweringDeps, question: str) -> dict[str, Any]:
    """One QA graph invocation. Each needs its own thread_id.

    A shared thread would make the graph treat the next candidate as a follow-up
    turn in the same conversation, so answer N would be conditioned on questions
    1..N-1. Cheap to get wrong and invisible in the output.
    """
    from uuid_utils.compat import uuid7

    return await deps.qa_service.ainvoke(
        {"manifest": deps.manifest, "query": question},
        config={"configurable": {"thread_id": str(uuid7())}},
    )


async def answer_and_judge(
    deps: AnsweringDeps,
    candidate: Mapping[str, Any],
    *,
    runs: int = 1,
) -> AnswerResult:
    """One candidate, end to end. Never raises - failures come back on the result."""
    cid = candidate.get("id", "?")
    question = (candidate.get("question") or "").strip()
    result = AnswerResult(candidate_id=cid, question=question)
    if not question:
        result.error = "candidate has no question"
        return result

    try:
        primary = await answer_once(deps, question)
        result.response = primary.get("response") or ""
        result.verified_quotes = count_verified_quotes(result.response)
        result.evidence = list(primary.get("evidence") or [])
        result.stats = summarise_evidence(result.evidence)

        if runs > 1:
            others = await asyncio.gather(
                *(answer_once(deps, question) for _ in range(runs - 1)),
                return_exceptions=True,
            )
            texts = [result.response]
            for other in others:
                if isinstance(other, BaseException):
                    logger.warning("Consistency run failed", cid=cid, error=repr(other))
                    continue
                texts.append(other.get("response") or "")
            if len(texts) > 1:
                # aembed_documents, not aembed_query: answer-to-answer similarity
                # is symmetric, and the query path is instruction-wrapped.
                vecs = np.asarray(await deps.embeddings.aembed_documents(texts))
                result.consistency = answer_consistency(vecs)
                result.alternates = texts[1:]

        judged = await deps.judge_chain.ainvoke(
            {
                "game_name": deps.game_name,
                "question": question,
                "answer": result.response,
                "evidence": [
                    {
                        "rulebook_name": c.get("rulebook_name", ""),
                        "page": c.get("page", ""),
                        "content": c.get("content", ""),
                    }
                    for c in result.evidence
                ],
            }
        )
        result.verdict = normalise_verdict(judged.get("verdict"))
        result.judge_reasoning = (judged.get("reasoning") or "").strip()
        result.rules_involved = [
            s.strip() for s in (judged.get("rules_involved") or []) if str(s).strip()
        ]
        return result
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        logger.exception("Candidate failed", cid=cid)
        result.error = repr(exc)
        return result
