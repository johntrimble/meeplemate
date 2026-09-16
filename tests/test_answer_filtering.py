"""Pure-function tests for eval-candidate answering and filtering.

No DB, no network, no LLM.
"""

import numpy as np
import pytest

from meeplemate.eval.answering import (
    AnswerResult,
    EvidenceStats,
    answer_block,
    answer_consistency,
    apply_answers,
    count_verified_quotes,
    candidates_to_answer,
    evidence_key,
    normalise_verdict,
    recommend,
    recommendation_summary,
    summarise_evidence,
    verdict_summary,
)


def chunk(book="Rulebook", page="12", start=0, content="text"):
    return {
        "rulebook_name": book,
        "page": page,
        "start_index": start,
        "end_index": start + len(content),
        "content": content,
    }


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


def test_evidence_key_normalises_page_type():
    # page arrives as int or str depending on whether a page number was OCR'd;
    # both must key to the same chunk or every count is inflated.
    assert evidence_key(chunk(page=12)) == evidence_key(chunk(page="12"))


def test_evidence_key_distinguishes_chunks_on_the_same_page():
    a = chunk(start=0)
    b = chunk(start=900)
    assert evidence_key(a) != evidence_key(b)


def test_summarise_evidence_deduplicates_repeated_chunks():
    """The graph can emit the same chunk from several stages.

    Counting raw list length would report every multi-stage answer as a rule
    interaction, which is exactly the signal being measured.
    """
    stats = summarise_evidence([chunk(), chunk(), chunk()])
    assert stats.chunk_count == 1
    assert stats.rulebook_count == 1


def test_summarise_evidence_counts_distinct_rulebooks():
    stats = summarise_evidence(
        [chunk(book="Battle Book"), chunk(book="Magic"), chunk(book="Magic", start=50)]
    )
    assert stats.chunk_count == 3
    assert stats.rulebook_count == 2
    assert stats.rulebooks == ("Battle Book", "Magic")


def test_summarise_evidence_handles_empty():
    stats = summarise_evidence([])
    assert stats.chunk_count == 0 and stats.rulebook_count == 0


# --------------------------------------------------------------------------
# Verdict normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("good", "good"),
        ("Good", "good"),
        ("single-rule", "single_rule"),
        ("single rule", "single_rule"),
        ("FALSE_PREMISE", "false_premise"),
    ],
)
def test_normalise_verdict_accepts_case_and_separator_variants(raw, expected):
    assert normalise_verdict(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "excellent", "maybe?"])
def test_unknown_verdicts_fall_back_to_unclear_not_good(raw):
    """An invented label must route to review, never to keep or drop.

    Structured output constrains the schema, not the vocabulary.
    """
    assert normalise_verdict(raw) == "unclear"


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------


TWO = EvidenceStats(chunk_count=2, rulebook_count=2, page_count=2, rulebooks=("A", "B"))
ONE = EvidenceStats(chunk_count=1, rulebook_count=1, page_count=1, rulebooks=("A",))
NONE_ = EvidenceStats(chunk_count=0, rulebook_count=0, page_count=0, rulebooks=())


def test_keep_requires_grounding_a_good_verdict_and_two_passages():
    rec = recommend(verified_quotes=2, stats=TWO, verdict="good")
    assert rec.status == "keep"


def test_ungrounded_answer_is_dropped():
    rec = recommend(verified_quotes=0, stats=TWO, verdict="good")
    assert rec.status == "drop"
    assert "no verified quotes" in rec.reasons[0]


def test_answer_with_no_evidence_is_dropped():
    assert recommend(verified_quotes=2, stats=NONE_, verdict="good").status == "drop"


@pytest.mark.parametrize("verdict", ["false_premise", "not_in_rules"])
def test_judge_rejections_are_dropped(verdict):
    assert recommend(verified_quotes=2, stats=TWO, verdict=verdict).status == "drop"


def test_single_rule_is_reviewed_not_dropped():
    """A weak question is still a usable test case.

    Whether to promote it is the reviewer's call, not the filter's.
    """
    rec = recommend(verified_quotes=2, stats=TWO, verdict="single_rule")
    assert rec.status == "review"


def test_single_passage_downgrades_keep_to_review_but_never_drops():
    rec = recommend(verified_quotes=2, stats=ONE, verdict="good")
    assert rec.status == "review"
    assert "single passage" in " ".join(rec.reasons)


def test_unclear_verdict_is_reviewed():
    assert recommend(verified_quotes=2, stats=TWO, verdict="unclear").status == "review"


def test_low_consistency_downgrades_to_review():
    rec = recommend(verified_quotes=2, stats=TWO, verdict="good", consistency=0.40)
    assert rec.status == "review"
    assert "disagree" in " ".join(rec.reasons)


def test_high_consistency_leaves_a_keep_alone():
    assert recommend(
        verified_quotes=2, stats=TWO, verdict="good", consistency=0.95
    ).status == "keep"


def test_drop_reasons_accumulate_rather_than_short_circuit():
    rec = recommend(verified_quotes=0, stats=NONE_, verdict="false_premise")
    assert rec.status == "drop"
    assert len(rec.reasons) == 3


# --------------------------------------------------------------------------
# Consistency
# --------------------------------------------------------------------------


def test_consistency_is_none_for_a_single_run():
    """None is not 1.0: nothing was compared, which is not perfect agreement."""
    assert answer_consistency(np.array([[1.0, 0.0]])) is None
    assert answer_consistency(np.zeros((0, 2))) is None


def test_consistency_reports_the_worst_pair_not_the_mean():
    # Two identical answers and one orthogonal: the mean would hide the outlier,
    # and the outlier is the reason to look.
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert answer_consistency(vecs) == pytest.approx(0.0, abs=1e-9)


def test_identical_answers_score_one():
    vecs = np.array([[0.6, 0.8], [0.6, 0.8]])
    assert answer_consistency(vecs) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Run-document plumbing
# --------------------------------------------------------------------------


def doc_with(*cands):
    return {"run": {"game_id": "g"}, "candidates": list(cands)}


def cand(cid, dedupe="unique", answer=None):
    c = {"id": cid, "question": f"q {cid}", "dedupe": {"status": dedupe}}
    if answer is not None:
        c["answer"] = answer
    return c


def test_near_duplicates_are_skipped_by_default():
    doc = doc_with(cand("a"), cand("b", dedupe="near_existing"))
    assert [c["id"] for c in candidates_to_answer(doc)] == ["a"]


def test_include_duplicates_answers_them():
    doc = doc_with(cand("a"), cand("b", dedupe="near_candidate"))
    assert len(candidates_to_answer(doc, include_duplicates=True)) == 2


def test_candidates_with_no_dedupe_block_are_answered():
    doc = doc_with({"id": "a", "question": "q"})
    assert len(candidates_to_answer(doc)) == 1


def test_resume_skips_answered_candidates():
    doc = doc_with(cand("a", answer={"recommendation": "keep"}), cand("b"))
    assert [c["id"] for c in candidates_to_answer(doc, resume=True)] == ["b"]


def test_resume_retries_candidates_that_errored():
    """A timeout must not permanently exclude a candidate from the queue."""
    doc = doc_with(cand("a", answer={"status": "error", "error": "TimeoutError()"}))
    assert [c["id"] for c in candidates_to_answer(doc, resume=True)] == ["a"]


def test_apply_answers_attaches_without_dropping_by_default():
    doc = doc_with(cand("a"), cand("b"))
    apply_answers(doc, {"a": {"recommendation": "drop"}, "b": {"recommendation": "keep"}})
    assert [c["id"] for c in doc["candidates"]] == ["a", "b"]
    assert doc["candidates"][0]["answer"]["recommendation"] == "drop"


def test_apply_answers_drop_removes_only_drops():
    doc = doc_with(cand("a"), cand("b"), cand("c"))
    apply_answers(
        doc,
        {
            "a": {"recommendation": "drop"},
            "b": {"recommendation": "review"},
            "c": {"recommendation": "keep"},
        },
        drop=True,
    )
    assert [c["id"] for c in doc["candidates"]] == ["b", "c"]


def test_apply_answers_leaves_unanswered_candidates_in_place_under_drop():
    """A partial run must not delete candidates it never reached."""
    doc = doc_with(cand("a"), cand("b"))
    apply_answers(doc, {"a": {"recommendation": "drop"}}, drop=True)
    assert [c["id"] for c in doc["candidates"]] == ["b"]


def test_summaries_count_unanswered_and_errors_separately():
    doc = doc_with(
        cand("a", answer={"recommendation": "keep", "verdict": "good"}),
        cand("b", answer={"status": "error", "error": "x"}),
        cand("c"),
    )
    assert recommendation_summary(doc) == {
        "keep": 1, "review": 0, "drop": 0, "error": 1, "unanswered": 1
    }
    assert verdict_summary(doc)["good"] == 1


# --------------------------------------------------------------------------
# Answer block
# --------------------------------------------------------------------------


def test_answer_block_for_a_failure_carries_the_error_and_no_verdict():
    block = answer_block(AnswerResult(candidate_id="a", question="q", error="Boom()"))
    assert block == {"status": "error", "error": "Boom()"}


def test_answer_block_omits_evidence_text_by_default():
    r = AnswerResult(
        candidate_id="a", question="q", response="ans", verified_quotes=2,
        evidence=[chunk()], stats=summarise_evidence([chunk()]), verdict="good",
    )
    assert "evidence" not in answer_block(r)
    assert "evidence" in answer_block(r, include_evidence_text=True)


def test_answer_block_omits_consistency_when_a_single_run():
    r = AnswerResult(
        candidate_id="a", question="q", response="ans", verified_quotes=2,
        evidence=[chunk(), chunk(book="B")],
        stats=summarise_evidence([chunk(), chunk(book="B")]), verdict="good",
    )
    block = answer_block(r)
    assert "consistency" not in block
    assert block["recommendation"] == "keep"


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


MARKER = '<div data-quote-status="verified"></div>'


def test_count_verified_quotes_counts_each_marker():
    text = f"Yes.\n\n{MARKER}\n\n> a rule\n\nAnd also\n\n{MARKER}\n\n> another"
    assert count_verified_quotes(text) == 2


def test_unmarked_quotes_do_not_count():
    """Unverified quotes are left unmarked on purpose; absence is the signal."""
    assert count_verified_quotes("Yes.\n\n> an unverified quote") == 0


def test_count_verified_quotes_handles_empty_and_none():
    assert count_verified_quotes("") == 0
    assert count_verified_quotes(None) == 0


def test_an_answer_with_evidence_but_no_verified_quote_is_dropped():
    """The regression that motivated this signal.

    Reading `valid` off the qa_service result marked every answer ungrounded,
    because the coordinating graph's output schema does not carry it. Grounding
    now comes from the response text, so an answer that cites chunks but
    verified nothing is still correctly dropped -- and one that verified a quote
    is not.
    """
    grounded = AnswerResult(
        candidate_id="a", question="q", response=f"Yes.{MARKER}",
        evidence=[chunk(), chunk(book="B")],
        stats=summarise_evidence([chunk(), chunk(book="B")]), verdict="good",
    )
    grounded.verified_quotes = count_verified_quotes(grounded.response)
    assert answer_block(grounded)["recommendation"] == "keep"

    ungrounded = AnswerResult(
        candidate_id="b", question="q", response="Yes, probably.",
        evidence=[chunk(), chunk(book="B")],
        stats=summarise_evidence([chunk(), chunk(book="B")]), verdict="good",
    )
    ungrounded.verified_quotes = count_verified_quotes(ungrounded.response)
    assert answer_block(ungrounded)["recommendation"] == "drop"
