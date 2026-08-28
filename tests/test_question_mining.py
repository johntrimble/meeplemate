"""Pure-function tests for the question miner.

No database, network or LLM — the I/O half of ``meeplemate.eval.mining`` is thin
orchestration, and everything with a decision in it lives in a pure function.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from langchain_core.documents import Document

from meeplemate.eval.mining import (
    ContextEntry,
    ContextSelection,
    ExistingQuestion,
    ParentRef,
    RetrievedHit,
    SeedResult,
    adjacency_block,
    build_run_document,
    length_histogram,
    candidate_id,
    concept_retrieval_overlap,
    cosine_similarity_matrix,
    dedupe_candidates,
    dump_run_yaml,
    eval_gen_run_path,
    fuse_concept_hits,
    next_run_id,
    normalise_concepts,
    normalise_questions,
    order_parents,
    parent_ref_from_document,
    parse_parent_key,
    retrieve_for_concept,
    seed_ids_in_document,
    select_context,
)

GAME = "oathsworn"
VER = "019fe3c4-0a8a-7651-8ec1-8988bc7e669d"


def parent(doc_key="encounter", page=0, chunk=0, book="Encounter Rule Book", content="x" * 300):
    return ParentRef(
        parent_id=f"{GAME}#{VER}#{doc_key}#{page}#{chunk}",
        document_key=doc_key,
        rulebook_name=book,
        page_ordinal=page,
        page_num=str(page),
        page_chunk_index=chunk,
        content=content,
    )


# --------------------------------------------------------------------------
# Key parsing and ordering
# --------------------------------------------------------------------------


def test_parse_parent_key_roundtrip():
    key = f"{GAME}#{VER}#RW5jb3VudGVy#30#1"
    assert parse_parent_key(key) == (GAME, VER, "RW5jb3VudGVy", 30, 1)


def test_parse_parent_key_document_key_containing_separator():
    # The two trailing integers are taken with rsplit, so a '#' inside the
    # document_key must not shift page_ordinal or chunk_idx.
    key = f"{GAME}#{VER}#odd#key#30#1"
    assert parse_parent_key(key) == (GAME, VER, "odd#key", 30, 1)


@pytest.mark.parametrize("bad", ["", "nosep", f"{GAME}#{VER}", f"{GAME}#{VER}#doc#x#1"])
def test_parse_parent_key_malformed_returns_none(bad):
    assert parse_parent_key(bad) is None


def test_parent_ref_prefers_metadata_and_tolerates_missing_page_num():
    key = f"{GAME}#{VER}#doc#7#2"
    doc = Document(page_content="body", metadata={
        "document_key": "doc", "rulebook_name": "Encounter Rule Book",
        "page_ordinal": 7, "page_chunk_index": 2,
    })
    ref = parent_ref_from_document(key, doc)
    assert ref.rulebook_name == "Encounter Rule Book"
    assert ref.page_ordinal == 7
    # page_num is only written when a page number was OCR'd.
    assert ref.page_num == ""


def test_order_parents_is_numeric_not_lexicographic():
    # The regression this guards: sorting the docstore key as a string puts
    # page_ordinal 10 before 2, silently corrupting adjacency and --offset.
    unordered = [parent(page=10), parent(page=2), parent(page=1)]
    assert [p.page_ordinal for p in order_parents(unordered)] == [1, 2, 10]


def test_order_parents_does_not_interleave_rulebooks():
    mixed = [
        parent(doc_key="story", page=1),
        parent(doc_key="encounter", page=5),
        parent(doc_key="story", page=0),
        parent(doc_key="encounter", page=0),
    ]
    keys = [p.document_key for p in order_parents(mixed)]
    assert keys == ["encounter", "encounter", "story", "story"]


# --------------------------------------------------------------------------
# Adjacency
# --------------------------------------------------------------------------


def test_adjacency_radius_zero_is_seed_only():
    ordered = order_parents([parent(page=i) for i in range(5)])
    seed = ordered[2].parent_id
    assert adjacency_block(ordered, seed, 0) == {seed}


def test_adjacency_radius_one_takes_neighbours():
    ordered = order_parents([parent(page=i) for i in range(5)])
    block = adjacency_block(ordered, ordered[2].parent_id, 1)
    assert block == {ordered[1].parent_id, ordered[2].parent_id, ordered[3].parent_id}


def test_adjacency_clipped_at_list_ends():
    ordered = order_parents([parent(page=i) for i in range(3)])
    assert adjacency_block(ordered, ordered[0].parent_id, 1) == {
        ordered[0].parent_id, ordered[1].parent_id
    }


def test_adjacency_does_not_cross_rulebook_boundary():
    # A seed at the end of book A must not exclude the first chunk of book B:
    # that is a different rulebook, not a restatement, and cross-book pairs are
    # exactly what the miner is looking for.
    ordered = order_parents([
        parent(doc_key="encounter", page=0),
        parent(doc_key="encounter", page=1),
        parent(doc_key="story", page=0),
    ])
    seed = ordered[1]
    assert seed.document_key == "encounter"
    block = adjacency_block(ordered, seed.parent_id, 1)
    assert ordered[2].parent_id not in block


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_fuse_prefers_parent_found_by_both_arms():
    hits = fuse_concept_hits("knockback", ["a", "b"], ["b", "c"])
    assert hits[0].parent_id == "b"
    assert hits[0].arms == ("dense", "bm25")


def test_fuse_records_per_arm_ranks():
    hits = {h.parent_id: h for h in fuse_concept_hits("k", ["a", "b"], ["b", "c"])}
    assert (hits["a"].dense_rank, hits["a"].bm25_rank) == (0, None)
    assert (hits["b"].dense_rank, hits["b"].bm25_rank) == (1, 0)
    assert (hits["c"].dense_rank, hits["c"].bm25_rank) == (None, 1)
    assert hits["c"].arms == ("bm25",)


# --------------------------------------------------------------------------
# Context assembly
# --------------------------------------------------------------------------


def _hits(concept, ids):
    return [
        RetrievedHit(parent_id=i, concept=concept, fused_rank=r, arms=("dense",),
                     dense_rank=r, bm25_rank=None)
        for r, i in enumerate(ids)
    ]


def _corpus(n=12, doc_key="encounter"):
    parents = order_parents([parent(doc_key=doc_key, page=i) for i in range(n)])
    return parents, {p.parent_id: p for p in parents}


def test_select_context_seed_is_always_first_even_with_no_hits():
    parents, by_id = _corpus()
    sel = select_context(parents[0], {}, by_id, excluded_ids=set(),
                         per_concept_cap=6, max_context_chunks=12)
    assert sel.entries[0].role == "seed"
    assert sel.entries[0].parent.parent_id == parents[0].parent_id
    assert sel.retrieved == ()


def test_select_context_drops_excluded_and_records_them():
    parents, by_id = _corpus()
    seed = parents[5]
    excluded = {parents[4].parent_id, parents[5].parent_id, parents[6].parent_id}
    hits = _hits("c", [p.parent_id for p in parents[4:9]])
    sel = select_context(seed, {"c": hits}, by_id, excluded_ids=excluded,
                         per_concept_cap=6, max_context_chunks=12)
    got = {e.parent.parent_id for e in sel.retrieved}
    assert got.isdisjoint(excluded)
    assert set(sel.dropped_adjacent) == excluded & {h.parent_id for h in hits}


def test_select_context_merges_a_parent_found_by_both_concepts():
    parents, by_id = _corpus()
    seed, shared = parents[0], parents[5].parent_id
    a = [RetrievedHit(shared, "alpha", 0, ("dense",), 0, None)]
    b = [RetrievedHit(shared, "beta", 0, ("bm25",), None, 0)]
    sel = select_context(seed, {"alpha": a, "beta": b}, by_id, excluded_ids=set(),
                         per_concept_cap=6, max_context_chunks=12)
    assert len(sel.retrieved) == 1
    entry = sel.retrieved[0]
    assert set(entry.concepts) == {"alpha", "beta"}
    assert entry.arms == ("dense", "bm25")


def test_select_context_respects_max_context_chunks_including_seed():
    parents, by_id = _corpus()
    hits = _hits("c", [p.parent_id for p in parents[1:]])
    sel = select_context(parents[0], {"c": hits}, by_id, excluded_ids=set(),
                         per_concept_cap=99, max_context_chunks=5)
    assert len(sel.entries) == 5


def test_select_context_respects_per_concept_cap():
    parents, by_id = _corpus()
    hits = _hits("c", [p.parent_id for p in parents[1:]])
    sel = select_context(parents[0], {"c": hits}, by_id, excluded_ids=set(),
                         per_concept_cap=2, max_context_chunks=12)
    assert len(sel.retrieved) == 2


def test_select_context_fills_round_robin_across_concepts():
    # Concatenate-and-truncate would give concept A all four slots. The
    # two-cluster shape is the whole point of retrieving per concept.
    parents, by_id = _corpus(n=20)
    a = _hits("alpha", [p.parent_id for p in parents[1:9]])
    b = _hits("beta", [p.parent_id for p in parents[9:17]])
    sel = select_context(parents[0], {"alpha": a, "beta": b}, by_id,
                         excluded_ids=set(), per_concept_cap=5, max_context_chunks=5)
    concepts = [e.concepts[0] for e in sel.retrieved]
    assert concepts.count("alpha") == 2
    assert concepts.count("beta") == 2


def test_select_context_is_deterministic():
    parents, by_id = _corpus(n=20)
    a = _hits("alpha", [p.parent_id for p in parents[1:9]])
    b = _hits("beta", [p.parent_id for p in parents[9:17]])
    args = (parents[0], {"alpha": a, "beta": b}, by_id)
    kwargs = dict(excluded_ids=set(), per_concept_cap=3, max_context_chunks=7)
    first = select_context(*args, **kwargs)
    second = select_context(*args, **kwargs)
    assert [e.parent.parent_id for e in first.entries] == \
           [e.parent.parent_id for e in second.entries]


def test_select_context_skips_parents_missing_from_the_docstore():
    parents, by_id = _corpus()
    hits = _hits("c", ["ghost-id", parents[5].parent_id])
    sel = select_context(parents[0], {"c": hits}, by_id, excluded_ids=set(),
                         per_concept_cap=6, max_context_chunks=12)
    assert [e.parent.parent_id for e in sel.retrieved] == [parents[5].parent_id]


# --------------------------------------------------------------------------
# Retrieval sort direction — the highest-consequence bug available here
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_for_concept_sorts_distance_ascending():
    """asimilarity_search_with_score returns raw cosine DISTANCE, lower better.

    There is no HybridSearchConfig on this store, so these are not similarity
    scores. A descending sort returns the *worst* matches and the pipeline goes
    on producing confident-looking questions from irrelevant context.
    """
    def child(parent_id):
        return Document(page_content="", metadata={"doc_id": parent_id})

    # Deliberately shuffled so the function has to sort rather than pass through.
    scored = [(child("far"), 0.9), (child("near"), 0.1), (child("mid"), 0.5)]
    vector_store = SimpleNamespace(
        asimilarity_search_with_score=lambda *a, **k: _async(scored)
    )
    bm25 = SimpleNamespace(asearch=lambda *a, **k: _async([]))

    hits = await retrieve_for_concept(
        vector_store=vector_store, bm25=bm25, concept="knockback",
        game_id=GAME, game_version=VER, dense_k=10, bm25_k=10,
    )
    assert [h.parent_id for h in hits] == ["near", "mid", "far"]


async def _async(value):
    return value


@pytest.mark.asyncio
async def test_retrieve_for_concept_dedupes_children_to_parents():
    def child(parent_id):
        return Document(page_content="", metadata={"doc_id": parent_id})

    scored = [(child("p1"), 0.1), (child("p1"), 0.2), (child("p2"), 0.3)]
    vector_store = SimpleNamespace(
        asimilarity_search_with_score=lambda *a, **k: _async(scored)
    )
    bm25 = SimpleNamespace(asearch=lambda *a, **k: _async([("p3", 4.2)]))

    hits = await retrieve_for_concept(
        vector_store=vector_store, bm25=bm25, concept="c",
        game_id=GAME, game_version=VER, dense_k=10, bm25_k=10,
    )
    assert [h.parent_id for h in hits].count("p1") == 1
    assert "p3" in {h.parent_id for h in hits}


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_normalise_concepts_strips_dedupes_and_drops_sentences():
    raw = [
        "  psychology test  ",
        "Psychology Test",
        "a break test is not a psychology test and the two are quite separate",
        "",
        "break test",
    ]
    assert normalise_concepts(raw) == ["psychology test", "break test"]


def test_normalise_concepts_allows_a_single_survivor():
    # Guided decoding constrains shape, not content: asking for 2 can yield 1.
    assert normalise_concepts(["knockback", "knockback"]) == ["knockback"]


def test_normalise_concepts_drops_a_contained_concept():
    # Two overlapping concepts retrieve the same chunks, collapsing the
    # two-cluster context the retrieval step exists to build.
    assert normalise_concepts(["Stage Card", "stage"]) == ["Stage Card"]
    assert normalise_concepts(["stage", "Stage Card"]) == ["stage"]


def test_concept_retrieval_overlap_measures_cluster_collapse():
    a = _hits("alpha", ["p1", "p2", "p3"])
    identical = _hits("beta", ["p1", "p2", "p3"])
    disjoint = _hits("beta", ["p4", "p5", "p6"])
    half = _hits("beta", ["p2", "p3", "p9"])

    assert concept_retrieval_overlap({"alpha": a, "beta": identical}) == 1.0
    assert concept_retrieval_overlap({"alpha": a, "beta": disjoint}) == 0.0
    assert concept_retrieval_overlap({"alpha": a, "beta": half}) == pytest.approx(0.5)
    # One concept, or none, has nothing to overlap with.
    assert concept_retrieval_overlap({"alpha": a}) == 0.0
    assert concept_retrieval_overlap({}) == 0.0


def test_normalise_questions_truncates_and_dedupes():
    raw = ["  Does bleed apply?  ", "does bleed apply?", "What about knockback?", ""]
    assert normalise_questions(raw, count=5) == ["Does bleed apply?", "What about knockback?"]
    assert normalise_questions(raw, count=1) == ["Does bleed apply?"]


def test_normalise_questions_empty_input():
    assert normalise_questions([], count=5) == []


# --------------------------------------------------------------------------
# Dedupe
# --------------------------------------------------------------------------


def test_cosine_similarity_matrix_basics():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_matrix(a, a)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(0.0)


def test_cosine_similarity_matrix_normalises_input():
    a = np.array([[3.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    assert cosine_similarity_matrix(a, b)[0, 0] == pytest.approx(1.0)


def _vec(*xs):
    return np.array([list(xs)], dtype=float)


def test_dedupe_flags_a_candidate_matching_an_existing_case():
    existing = [ExistingQuestion("Oathsworn/knocked_back", GAME, "knocked back?")]
    verdicts = dedupe_candidates(
        ["c1"], ["knocked back?"], _vec(1.0, 0.0),
        existing, _vec(1.0, 0.0), game_id=GAME, threshold=0.92,
    )
    assert verdicts[0].status == "near_existing"
    assert verdicts[0].nearest_ref == "Oathsworn/knocked_back"
    assert verdicts[0].max_similarity == pytest.approx(1.0)


def test_dedupe_flags_the_second_of_two_identical_candidates():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0]])
    verdicts = dedupe_candidates(
        ["c1", "c2"], ["q", "q"], vecs, [], np.zeros((0, 2)),
        game_id=GAME, threshold=0.92,
    )
    assert verdicts[0].status == "unique"
    assert verdicts[1].status == "near_candidate"
    assert verdicts[1].nearest_ref == "c1"


def test_dedupe_is_greedy_keep_first_not_a_chain():
    # Three paraphrases collapse to one keeper plus two flagged against it.
    vecs = np.array([[1.0, 0.0]] * 3)
    verdicts = dedupe_candidates(
        ["c1", "c2", "c3"], ["q"] * 3, vecs, [], np.zeros((0, 2)),
        game_id=GAME, threshold=0.92,
    )
    assert [v.status for v in verdicts] == ["unique", "near_candidate", "near_candidate"]
    assert verdicts[2].nearest_ref == "c1"


def test_dedupe_annotates_below_threshold_candidates():
    existing = [ExistingQuestion("S/a", GAME, "unrelated")]
    verdicts = dedupe_candidates(
        ["c1"], ["q"], _vec(1.0, 1.0), existing, _vec(1.0, 0.0),
        game_id=GAME, threshold=0.92,
    )
    assert verdicts[0].status == "unique"
    # Annotate, don't just gate: the number is what tunes the threshold.
    assert verdicts[0].max_similarity == pytest.approx(0.7071, abs=1e-3)
    assert verdicts[0].nearest_ref == "S/a"


def test_dedupe_cross_game_match_reports_but_does_not_gate():
    # bge-small keys on surface form, so a Warhammer question sits high against
    # a structurally identical Oathsworn one. That must not suppress a case.
    existing = [ExistingQuestion("WFB/grail", "warhammer_5th_edition", "break test?")]
    verdicts = dedupe_candidates(
        ["c1"], ["break test?"], _vec(1.0, 0.0), existing, _vec(1.0, 0.0),
        game_id=GAME, threshold=0.92,
    )
    assert verdicts[0].status == "unique"
    assert verdicts[0].cross_game_max_similarity == pytest.approx(1.0)


def test_dedupe_empty_corpus_and_empty_candidates():
    assert dedupe_candidates([], [], np.zeros((0, 2)), [], np.zeros((0, 2)),
                             game_id=GAME, threshold=0.92) == []
    verdicts = dedupe_candidates(["c1"], ["q"], _vec(1.0, 0.0), [], np.zeros((0, 2)),
                                 game_id=GAME, threshold=0.92)
    assert verdicts[0].status == "unique"


def test_dedupe_boundary_is_inclusive():
    existing = [ExistingQuestion("S/a", GAME, "q")]
    verdicts = dedupe_candidates(
        ["c1"], ["q"], _vec(1.0, 0.0), existing, _vec(1.0, 0.0),
        game_id=GAME, threshold=1.0,
    )
    assert verdicts[0].status == "near_existing"


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def _seed_result(ordinal=3, questions=("q1", "q2")):
    p = parent(page=ordinal)
    sel = ContextSelection(
        entries=(
            ContextEntry(parent=p, role="seed", concepts=(), arms=(), fused_rank=None),
        ),
        dropped_adjacent=("x",),
        dropped_over_cap=(),
    )
    return SeedResult(seed=p, seed_ordinal=ordinal, concepts=["a", "b"],
                      selection=sel, questions=list(questions))


def _run_meta():
    return {"run_id": "2026-08-27", "game_id": GAME, "game_version": VER,
            "status": "complete", "params": {}, "seeds": {"total_parents": 10}}


def test_build_run_document_ids_are_unique_and_stable():
    results = [_seed_result(3), _seed_result(4)]
    a = build_run_document(_run_meta(), results, {})
    b = build_run_document(_run_meta(), results, {})
    ids = [c["id"] for c in a["candidates"]]
    assert ids == [c["id"] for c in b["candidates"]]
    assert len(ids) == len(set(ids)) == 4
    assert ids[0] == candidate_id(GAME, 3, 0)


def test_build_run_document_records_failures_and_emits_no_candidates_for_them():
    ok = _seed_result(3)
    bad = SeedResult(seed=parent(page=9), seed_ordinal=9, error="TimeoutError()")
    doc = build_run_document(_run_meta(), [ok, bad], {})
    failures = doc["run"]["seeds"]["failures"]
    assert len(failures) == 1 and failures[0]["seed_ordinal"] == 9
    assert doc["run"]["seeds"]["succeeded"] == 1
    assert all(c["seed"]["seed_ordinal"] != 9 for c in doc["candidates"])


def test_dump_run_yaml_preserves_key_order(tmp_path):
    doc = build_run_document(_run_meta(), [_seed_result()], {})
    path = tmp_path / "run.yaml"
    dump_run_yaml(doc, path)
    text = path.read_text()
    assert text.index("question:") < text.index("context:")
    assert yaml.safe_load(text)["run"]["game_id"] == GAME


def test_dump_run_yaml_does_not_touch_the_global_dumper(tmp_path):
    # Registering a str representer on yaml.SafeDumper itself would change every
    # spit_yaml call in the process, including ingest's game manifests.
    before = yaml.SafeDumper.yaml_representers.get(str)
    dump_run_yaml(build_run_document(_run_meta(), [_seed_result()], {}), tmp_path / "r.yaml")
    assert yaml.SafeDumper.yaml_representers.get(str) is before


def test_seed_ids_in_document_includes_failures(tmp_path):
    doc = {
        "run": {"seeds": {"failures": [{"parent_id": "p-failed"}]}},
        "candidates": [{"seed": {"parent_id": "p-ok"}}],
    }
    # Failures are included so a resumed sweep does not stall retrying a seed
    # that will time out again.
    assert seed_ids_in_document(doc) == {"p-ok", "p-failed"}


def test_next_run_id_increments_on_collision(tmp_path):
    from datetime import datetime

    base = datetime.now().strftime("%Y-%m-%d")
    game_dir = tmp_path / GAME
    game_dir.mkdir()
    assert next_run_id(game_dir) == base
    (game_dir / f"{base}.yaml").touch()
    assert next_run_id(game_dir) == f"{base}-2"


def test_eval_gen_run_path_layout(tmp_path):
    assert eval_gen_run_path(tmp_path, GAME, "2026-08-27") == \
        tmp_path / GAME / "2026-08-27.yaml"


# --------------------------------------------------------------------------
# Question length
# --------------------------------------------------------------------------


def test_length_histogram_buckets_by_word_count():
    got = dict(length_histogram(["a b c", "w " * 12, "w " * 17, "w " * 40]))
    assert got["1-10"] == 1
    assert got["11-15"] == 1
    assert got["16-20"] == 1
    assert got["31-40"] == 1
    assert got["41+"] == 0


def test_length_histogram_edges_are_inclusive_upper_bounds():
    # A 10-word question belongs in 1-10, not 11-15. Off-by-one here would put
    # the reported median in the wrong bucket exactly at the value being tuned.
    assert dict(length_histogram(["w " * 10]))["1-10"] == 1
    assert dict(length_histogram(["w " * 11]))["11-15"] == 1


def test_length_histogram_overflow_bucket_catches_the_long_tail():
    assert dict(length_histogram(["w " * 200]))["41+"] == 1


def test_build_run_document_records_word_count():
    doc = build_run_document(
        _run_meta(), [_seed_result(questions=("How many cards do I draw?",))], {}
    )
    assert doc["candidates"][0]["word_count"] == 6


def test_generation_prompt_renders_the_length_cap():
    from meeplemate.eval.mining import GENERATE_CANDIDATES_PROMPT

    rendered = GENERATE_CANDIDATES_PROMPT.format_messages(
        game_name="X",
        chunks=[{"rulebook": "R", "page_num": "1", "content": "c", "is_seed": True}],
        count=5,
        max_words=25,
    )[0].content
    assert "under 25 words" in rendered
    # The concepts are a retrieval device and must never reach this prompt.
    assert "{{" not in rendered


def test_generation_prompt_fences_off_the_worked_example():
    """The style example's nouns must be explicitly disowned.

    Without the guard the model copies them: an early Oathsworn run produced
    "Can I use +2 Fuel if my reserve is empty?" -- the example's vocabulary in a
    game that has no fuel. A leaked noun makes a candidate unanswerable rather
    than merely bad, and it reads as plausible until someone checks the corpus.
    """
    from meeplemate.eval.mining import GENERATE_CANDIDATES_PROMPT

    rendered = GENERATE_CANDIDATES_PROMPT.format_messages(
        game_name="Oathsworn",
        chunks=[{"rulebook": "R", "page_num": "1", "content": "c", "is_seed": True}],
        count=5,
        max_words=25,
    )[0].content
    assert "DIFFERENT game" in rendered
    assert "do not exist in this game" in rendered
    assert rendered.count("Oathsworn") >= 2
