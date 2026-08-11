"""Retrieval-only evaluation against the evidence in ``test_cases.yaml``.

Isolates retrieval from generation. The end-to-end eval takes ~200s per
question and mixes retrieval quality with sub-question generation, answer
reasoning and formatting — so a retrieval change shows up there as noise. This
harness calls the chunk search service directly with the raw query and asks one
question: did the passage the reference answer quotes come back?

Gold parents are resolved by *quote text*, never by page number. The page
numbers in ``test_cases.yaml`` drift from the corpus (the Warpstorm Scroll is
cited as ``Warhammer Magic p.44`` but ingested at p.41; the Grail Knights quote
is cited as p.62 in one case and p.63 in the next), because printed page labels
come from a per-rulebook offset with its own history. Page is carried through
as a diagnostic so the disagreement stays visible.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import structlog

from meeplemate import quote_util
from meeplemate.eval import test_suites
from meeplemate.search import ChunkSearchService

logger = structlog.get_logger(__name__)

#: Fuzzy-match floor for locating an evidence quote inside a parent chunk.
#: Matches ``validate_and_fix_response``'s threshold so "the retrieval found it"
#: means the same thing here as "the quote would verify" downstream.
QUOTE_MIN_SCORE = 92.0


@dataclass
class GoldEvidence:
    """One evidence entry, resolved to the parent chunks that contain it."""

    rulebook: str
    cited_page: str
    quote: str
    parent_ids: set[str] = field(default_factory=set)
    #: Pages the quote was actually found on, for comparison with cited_page.
    found_pages: set[str] = field(default_factory=set)

    @property
    def resolved(self) -> bool:
        return bool(self.parent_ids)


@dataclass
class RetrievalCase:
    suite: str
    name: str
    game_id: str
    query: str
    evidence: list[GoldEvidence]

    @property
    def gold_parent_ids(self) -> set[str]:
        ids: set[str] = set()
        for item in self.evidence:
            ids |= item.parent_ids
        return ids


def load_retrieval_cases(pattern: str = "*") -> list[RetrievalCase]:
    """Test cases that carry ``evidence:``, filtered by glob on the case name."""
    import fnmatch

    cases: list[RetrievalCase] = []
    for suite in test_suites:
        game_id = suite["params"]["game_id"]
        for tc in suite["test_cases"]:
            if not tc.get("evidence"):
                continue
            if not fnmatch.fnmatch(tc["name"], pattern):
                continue
            cases.append(
                RetrievalCase(
                    suite=suite["name"],
                    name=tc["name"],
                    game_id=game_id,
                    query=tc["query"],
                    evidence=[
                        GoldEvidence(
                            rulebook=e["rulebook"],
                            cited_page=str(e.get("page", "")),
                            quote=e["quote"],
                        )
                        for e in tc["evidence"]
                    ],
                )
            )
    return cases


def _quote_fragments(quote: str) -> list[str]:
    """The whole quote, then its sentences.

    Parent chunks are split on a token budget with overlap, so a multi-sentence
    evidence quote routinely straddles a boundary and matches no single parent
    in full. A parent carrying any one sentence is a legitimate retrieval hit.
    """
    whole = quote.strip()
    fragments = [whole]
    for part in whole.replace("\n", " ").split(". "):
        part = part.strip().rstrip(".")
        # Short fragments match everywhere and would inflate the gold set.
        if len(part) >= 40:
            fragments.append(part)
    return fragments


async def resolve_gold_parents(
    cases: Sequence[RetrievalCase],
    parents_by_game: Mapping[str, Sequence[tuple[str, str, str]]],
) -> None:
    """Populate ``GoldEvidence.parent_ids`` in place.

    ``parents_by_game`` maps game_id to (parent_id, rulebook_name, page_num,
    page_content) tuples — see ``load_parents``.
    """
    for case in cases:
        parents = parents_by_game.get(case.game_id, [])
        for item in case.evidence:
            candidates = [p for p in parents if p[1] == item.rulebook]
            if not candidates:
                logger.warning(
                    "No parents for cited rulebook",
                    case=case.name,
                    rulebook=item.rulebook,
                )
                continue
            for fragment in _quote_fragments(item.quote):
                for parent_id, _rulebook, page, content in candidates:
                    if parent_id in item.parent_ids:
                        continue
                    if quote_util.find_quote_with_gaps(
                        content, fragment, min_score=QUOTE_MIN_SCORE
                    ):
                        item.parent_ids.add(parent_id)
                        item.found_pages.add(page)
                # The full quote matching is the strongest signal; only fall
                # back to sentence fragments when it found nothing.
                if item.parent_ids:
                    break


@dataclass
class CaseResult:
    case: RetrievalCase
    returned_ids: list[str]
    tokens_used: int = 0
    #: Parent ids each arm contributed, for attribution.
    vector_ids: set[str] = field(default_factory=set)
    bm25_ids: set[str] = field(default_factory=set)

    @property
    def gold(self) -> set[str]:
        return self.case.gold_parent_ids

    @property
    def hits(self) -> set[str]:
        return self.gold & set(self.returned_ids)

    @property
    def recall(self) -> Optional[float]:
        """Fraction of gold parents that survive into the model's context.

        Deliberately the only rank-free-but-order-sensitive metric here: the
        token budget truncates in relevance order, so a badly-ranked gold parent
        shows up as a miss. Everything that survives the budget is read by the
        model, so nothing finer-grained than "did it make the cut" applies.

        There is no recall@k or MRR. `build_chunk_search_service_2` re-sorts its
        output by (rulebook, page, start_index) before returning, so position in
        the returned list is document order, not relevance order — measuring @k
        over it produces numbers that look like ranking quality and are not.
        Fusion appeared to halve recall@5 purely because it returns twice as
        many chunks, diluting the first five slots of a document-ordered list.
        """
        if not self.gold:
            return None
        return len(self.hits) / len(self.gold)

    def attribution(self) -> dict[str, int]:
        """Which arm found each gold parent — the number that decides whether
        the lexical arm is pulling its weight. If ``bm25_only`` is zero across a
        run, the arm is redundant however good the aggregate recall looks."""
        out = {"vector_only": 0, "bm25_only": 0, "both": 0, "neither": 0}
        for parent_id in self.gold:
            in_vec = parent_id in self.vector_ids
            in_bm25 = parent_id in self.bm25_ids
            if in_vec and in_bm25:
                out["both"] += 1
            elif in_vec:
                out["vector_only"] += 1
            elif in_bm25:
                out["bm25_only"] += 1
            else:
                out["neither"] += 1
        return out


async def run_case(
    search: ChunkSearchService,
    manifest: Any,
    case: RetrievalCase,
    token_budget: int = 15_000,
    chunk_id_to_parent: Optional[Mapping[str, str]] = None,
) -> tuple[list[str], int]:
    """Run one query through the search service, returning parent ids in order.

    Passes the raw ``test_case.query`` as a single query rather than going
    through ``search_chunks``, so the result reflects retrieval alone and not
    the LLM's sub-question generation.

    ``add_chunk_ids`` replaces each returned ``Document.id`` with a synthetic
    ``rulebook::page::start_index`` handle, so the ids coming out of the service
    are not docstore keys. ``chunk_id_to_parent`` translates them back; without
    it the gold set and the results are in different namespaces and every
    comparison silently scores zero.
    """
    output = await search.ainvoke(
        {"manifest": manifest, "query": [case.query], "token_budget": token_budget}
    )
    ids = [doc.id for doc in output["chunks"] if doc.id is not None]
    if chunk_id_to_parent is not None:
        ids = [chunk_id_to_parent.get(chunk_id, chunk_id) for chunk_id in ids]
    return ids, output["tokens_used"]


def summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if r.gold]
    if not scored:
        return {"cases": 0}
    totals = {"vector_only": 0, "bm25_only": 0, "both": 0, "neither": 0}
    for r in scored:
        for key, value in r.attribution().items():
            totals[key] += value
    return {
        "cases": len(scored),
        "unresolved_cases": len(results) - len(scored),
        "recall": sum(r.recall or 0 for r in scored) / len(scored),
        "full_recall_cases": sum(1 for r in scored if r.recall == 1.0),
        "chunks": sum(len(r.returned_ids) for r in scored) / len(scored),
        "tokens": sum(r.tokens_used for r in scored) / len(scored),
        "attribution": totals,
    }
