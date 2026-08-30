"""Step 4: turn reviewed candidates into golden test cases.

`decisions.json` is the source of truth. This step is a pure derivation from it
plus the answer records, so it can be re-run after any correction and produce a
clean diff rather than a hand-merge.

Only candidates the reviewer marked `yes` are promoted. A rejected candidate is
skipped even when a `corrected_answer` was written for it -- the answer may have
been drafted while working through the flagged list, and the verdict is the
decision that counts.

Evidence is extracted with `quote_util.find_quotes_in_text`, the same parser the
QA pipeline uses to locate and verify quotes. A private regex here would drift
from it, and the failure mode is evidence that no longer matches the corpus --
which reads as a retrieval regression rather than as a promotion bug.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import structlog

from meeplemate.quote_util import find_quotes_in_text
from meeplemate.util import snake_case

#: The QA pipeline wraps each verified blockquote in this before returning, as a
#: hint for the frontend. It is pipeline output, not part of an answer, and the
#: hand-written cases do not carry it -- leaving it in would put presentation
#: markup in front of the correctness judge.
VERIFIED_MARKER = '<div data-quote-status="verified"></div>'

#: Names that read badly as the last word of a truncated case name.
_TRAILING_STOPWORDS = {
    "a", "an", "the", "as", "at", "by", "for", "from", "if", "in", "into", "of",
    "on", "or", "to", "with", "and", "but", "do", "does", "is", "are", "my",
}

logger = structlog.get_logger(__name__)

#: Case names become file names (`<suite>__<case>.json.gz`), so keep them short
#: enough that a run group stays listable.
MAX_NAME_WORDS = 10


def case_name(question: str, taken: set[str]) -> str:
    """A stable, readable, unique snake_case name for a test case.

    Derived from the question rather than from the candidate id: the id encodes
    a seed ordinal that means nothing outside the mining run, and these names
    are what appear in eval output and run filenames.
    """
    words = [w for w in snake_case(question.strip()).strip("_").split("_") if w]
    words = words[:MAX_NAME_WORDS]
    # Truncation lands mid-phrase often enough that trailing function words are
    # common ("..._during_combat_as_an"). Dropping them costs no meaning.
    while len(words) > 3 and words[-1] in _TRAILING_STOPWORDS:
        words.pop()
    base = "_".join(words)[:120].strip("_") or "question"
    name, n = base, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def clean_answer(answer_text: str) -> str:
    """An answer as a reference answer: pipeline markup removed.

    Only the verified-quote marker is stripped. The blockquotes themselves stay,
    because the hand-written cases quote the rulebook the same way and the
    correctness judge is comparing against that style.
    """
    text = (answer_text or "").replace(VERIFIED_MARKER, "")
    # Trailing whitespace forces PyYAML off literal block style onto an escaped
    # double-quoted scalar, which is unreadable in a diff. The agent emits "> "
    # as the blank line inside a blockquote, so this is not rare.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse the blank-line run the removed div leaves behind.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _page_value(raw: str) -> Any:
    """Ints where the page is numeric, matching the hand-written cases.

    Cosmetic: `retrieval.py` coerces to str and resolves gold parents by quote
    text, never by page.
    """
    page = str(raw).strip()
    return int(page) if page.isdigit() else page


def unparseable_quotes(answer_text: str) -> list[str]:
    """Blockquotes that will not become evidence, and why they are lost.

    `extract_ref_name_and_page` only recognises numeric pages, so a quote cited
    as `(Some Book, p. iv)` yields an empty `ref_name` and is dropped. Every
    quote in the first real run parsed, but the failure is otherwise invisible:
    the case still gets written, just with less evidence than the answer
    actually cites, and thin evidence reads as a retrieval problem later.
    """
    lost: list[str] = []
    for q in find_quotes_in_text(answer_text or ""):
        if q["quote_type"] != "blockquote":
            continue
        citation = q["citation"]
        if not citation or not citation["ref_name"]:
            lost.append((q["quote"] or q["text"] or "").strip()[:80])
    return lost


def extract_evidence(answer_text: str) -> list[dict[str, Any]]:
    """Evidence entries from the blockquotes in an answer.

    `quote` is the cleaned form the parser already produces -- no `>` markers,
    no trailing citation, no surrounding quotation marks -- which is the shape
    `test_cases.yaml` stores. Uncited blockquotes are dropped: an evidence entry
    without a rulebook and page cannot be scored by `mm-eval retrieval`.
    """
    out: list[dict[str, Any]] = []
    for q in find_quotes_in_text(answer_text or ""):
        if q["quote_type"] != "blockquote":
            continue
        citation = q["citation"]
        if not citation or not citation["ref_name"]:
            continue
        quote = (q["quote"] or "").strip()
        if not quote:
            continue
        out.append({
            "rulebook": citation["ref_name"].strip(),
            "page": _page_value(citation["page"]),
            "quote": quote + "\n",
        })
    return out


def suite_name(game_id: str, run_id: str) -> str:
    """`Munchkin (mined 2026-08-28)`.

    Provenance is visible in eval output and in run file paths. It does not
    affect `--filter`, which fnmatches the *case* name.
    """
    return f"{game_id.replace('_', ' ').title()} (mined {run_id})"


def promoted_decisions(
    decisions: Sequence[Mapping[str, Any]], *, include_rejected: bool = False
) -> list[Mapping[str, Any]]:
    if include_rejected:
        return [d for d in decisions if d.get("reviewer") in ("yes", "no", "later")]
    return [d for d in decisions if d.get("reviewer") == "yes"]


def build_test_case(
    decision: Mapping[str, Any],
    answer_record: Mapping[str, Any] | None,
    taken: set[str],
) -> dict[str, Any] | None:
    """One golden case, or None when there is no answer to promote.

    The reviewer's `corrected_answer` wins over the agent's draft; a case with
    neither is skipped and reported rather than written with an empty
    `reference_answer`, which would score as a failure for every run.
    """
    answer = clean_answer(decision.get("corrected_answer") or "")
    if not answer:
        answer = clean_answer(((answer_record or {}).get("answer") or {}).get("response", ""))
    if not answer:
        return None

    question = (decision.get("question") or "").strip()
    if not question:
        return None

    return {
        "name": case_name(question, taken),
        "query": question + "\n",
        "reference_answer": answer + "\n",
        "evidence": extract_evidence(answer),
        "source": {
            "candidate_id": decision["id"],
            "answer": "reviewer" if decision.get("corrected_answer") else "agent",
        },
    }


def build_suites(
    run_id: str,
    decisions: Sequence[Mapping[str, Any]],
    answers_by_id: Mapping[str, Mapping[str, Any]],
    *,
    include_rejected: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """(suites, skipped candidate ids), one suite per game, games in sorted order.

    Names are deduped across the whole file rather than per suite, because run
    file paths are `<suite>__<case>` and two games can phrase a question
    identically ("can i play a curse").
    """
    taken: set[str] = set()
    skipped: list[str] = []
    by_game: dict[str, list[dict[str, Any]]] = {}

    for decision in promoted_decisions(decisions, include_rejected=include_rejected):
        case = build_test_case(decision, answers_by_id.get(decision["id"]), taken)
        if case is None:
            skipped.append(decision["id"])
            continue
        by_game.setdefault(decision.get("game", "unknown"), []).append(case)

    suites = [
        {
            "name": suite_name(game, run_id),
            "params": {"game_id": game},
            "test_cases": cases,
        }
        for game, cases in sorted(by_game.items())
    ]
    return suites, skipped


def evidence_stats(suites: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    cases = [c for s in suites for c in s["test_cases"]]
    return {
        "cases": len(cases),
        "with_evidence": sum(1 for c in cases if c["evidence"]),
        "evidence_quotes": sum(len(c["evidence"]) for c in cases),
        "reviewer_answers": sum(1 for c in cases if c["source"]["answer"] == "reviewer"),
        "dropped_quotes": sum(
            len(unparseable_quotes(c["reference_answer"])) for c in cases
        ),
    }


class _CaseDumper(__import__("yaml").SafeDumper):
    """Local dumper: block scalars for multi-line strings.

    `safe_dump` renders a multi-line answer as one escaped double-quoted scalar,
    which is unreadable in a diff and unlike the hand-written cases. Subclassing
    keeps the style scoped here rather than changing every `spit_yaml` call in
    the process.
    """


def _block_scalar_for_multiline(dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_CaseDumper.add_representer(str, _block_scalar_for_multiline)


def dump_suites(suites, path, *, run_id: str, source) -> None:
    """Write the generated suites, atomically, in the hand-written file's style."""
    import yaml

    from meeplemate.util import atomic_write_text

    header = (f"# Generated by `mm-eval promote {run_id}` - do not edit by hand.\n"
              f"# Source: {source}\n"
              f"# Re-run promote after changing a decision or an answer.\n")

    def write(fp):
        fp.write(header)
        yaml.dump(suites, fp, Dumper=_CaseDumper, sort_keys=False,
                  default_flow_style=False, allow_unicode=True, width=100)

    atomic_write_text(path, write)
