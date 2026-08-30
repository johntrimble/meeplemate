"""Mine candidate eval questions from a game's rulebook chunks.

Steps 1-2 of eval-corpus generation: build a small multi-chunk context per seed
chunk, generate candidate questions from it, then flag near-duplicates. Answer
generation, answer-based filtering and human review are downstream and manual.

The corpus this feeds is 42 hand-written cases, which is too few to measure
anything: at n≈15 per split the 95% CI on correctness is roughly ±23 points, and
a paired comparison needs 6 questions to flip in one direction before it clears
noise. The questions worth adding are the ones that turn on rule *interactions* —
evidence in two or more places, far apart.

Four cheaper ways of finding those were measured against this corpus and all
failed:

* A marker regex for "distinction" prose ranked the known-good Grail Knights
  pivot 1st of 3,518 chunks, but fires at 16.5% on Warhammer and 0.0% on
  Catan/Wingspan/Waterdeep/Quacks. It was tuned to one verbose rulebook.
* A rare-term co-occurrence graph does not contain the Grail Knights case at
  all: the pivot and the break-test rule share no rare term. The linking
  vocabulary (``psycholog`` df=80, ``break`` df=106) is mid-frequency, because a
  rulebook's conceptual vocabulary *is* its core vocabulary. Widening the band
  far enough to include those terms pushes median 1-hop degree to 654 of 1,349
  chunks — there is no band that is both selective and connected.
* tf-idf term weighting on the pivot chunk returns ``engend, bloodlust, feud,
  berserk, hatr`` — the chapter's narrative preamble, not its mechanics — and
  ranks the break-test rule 54th of 60.
* Whole-book prompting fits 16 of 17 games in one prompt, but asks the model to
  do selection *and* synthesis in one call.

What survives is that naming the concepts requires a model to read the passage.
Every statistical method is drawn to rare words, and in a rulebook the rare words
are the prose. So: one LLM call to name concepts, cheap retrieval to expand them,
one LLM call to generate questions from a deliberately small context.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    AbstractSet,
    Annotated,
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
)

import numpy as np
import structlog
import yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from meeplemate.util import atomic_write_text

logger = structlog.get_logger(__name__)

T = TypeVar("T")


#: Arms a parent chunk can be retrieved by, in a stable order for the YAML.
ARM_ORDER = ("dense", "bm25")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentRef:
    """One parent chunk, flattened for retrieval bookkeeping and serialisation.

    ``page_ordinal`` is the ingest ordinal (0-based, dense) and ``page_num`` is
    the printed page label, which drifts from it — see the module docstring of
    ``meeplemate.eval.retrieval``. Ordering and adjacency use ``page_ordinal``;
    ``page_num`` is carried for the human reading the output.
    """

    parent_id: str
    document_key: str
    rulebook_name: str
    page_ordinal: int
    page_num: str
    page_chunk_index: int
    content: str

    @property
    def sort_key(self) -> tuple[str, int, int]:
        return (self.document_key, self.page_ordinal, self.page_chunk_index)


@dataclass(frozen=True)
class RetrievedHit:
    """A parent chunk retrieved for one concept, with per-arm provenance."""

    parent_id: str
    concept: str
    fused_rank: int
    arms: tuple[str, ...]
    dense_rank: Optional[int]
    bm25_rank: Optional[int]


@dataclass(frozen=True)
class ContextEntry:
    """One chunk in an assembled generation context."""

    parent: ParentRef
    role: Literal["seed", "retrieved"]
    concepts: tuple[str, ...]
    arms: tuple[str, ...]
    fused_rank: Optional[int]


@dataclass(frozen=True)
class ContextSelection:
    """An assembled context. ``entries[0]`` is always the seed."""

    entries: tuple[ContextEntry, ...]
    dropped_adjacent: tuple[str, ...]
    dropped_over_cap: tuple[str, ...]

    @property
    def seed(self) -> ParentRef:
        return self.entries[0].parent

    @property
    def retrieved(self) -> tuple[ContextEntry, ...]:
        return tuple(e for e in self.entries if e.role == "retrieved")


@dataclass
class SeedResult:
    """Outcome of mining one seed. ``error`` set means the seed failed."""

    seed: ParentRef
    seed_ordinal: int
    concepts: list[str] = field(default_factory=list)
    selection: Optional[ContextSelection] = None
    questions: list[str] = field(default_factory=list)
    concept_overlap: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.questions)


@dataclass(frozen=True)
class ExistingQuestion:
    """A question already in ``test_cases.yaml``."""

    ref: str
    game_id: str
    question: str


@dataclass(frozen=True)
class DedupeVerdict:
    status: Literal["unique", "near_candidate", "near_existing"]
    max_similarity: float
    nearest_kind: Optional[Literal["candidate", "existing"]]
    nearest_ref: Optional[str]
    nearest_question: Optional[str]
    near_duplicates: tuple[tuple[str, str, float], ...]
    cross_game_max_similarity: float


# ---------------------------------------------------------------------------
# Parent chunk handling (pure)
# ---------------------------------------------------------------------------


def parse_parent_key(key: str) -> Optional[tuple[str, str, str, int, int]]:
    """Split a parent docstore key into its five components.

    Key format is ``{game_id}#{game_version}#{document_key}#{page_ordinal}#{chunk_idx}``
    (``ingest/gamepackage.py`` builds the page id, ``chunkbuild.py`` appends the
    chunk index). ``document_key`` is base64 of the source filename, so it can
    contain ``=``, ``+`` and ``/`` but never ``#`` — still, the two trailing
    integers are taken with ``rsplit`` so a stray separator shifts nothing.

    Returns ``None`` rather than raising: this is used to sanity-check keys read
    back from a run file, where a malformed entry should be skipped, not fatal.
    """
    head, sep, rest = key.partition("#")
    if not sep:
        return None
    game_id = head
    version, sep, rest = rest.partition("#")
    if not sep or not rest:
        return None
    parts = rest.rsplit("#", 2)
    if len(parts) != 3:
        return None
    document_key, page_str, chunk_str = parts
    try:
        return (game_id, version, document_key, int(page_str), int(chunk_str))
    except ValueError:
        return None


def parent_ref_from_document(key: str, doc: Document) -> ParentRef:
    """Build a ``ParentRef``, preferring metadata over parsing the key.

    The key is treated as opaque identity; every field below is written by
    ingest (``gamepackage.get_page_metadata``, ``documentmetadata``,
    ``chunkbuild.split_document``). ``page_num`` is only written when a page
    number was actually OCR'd, so it defaults to empty rather than raising.
    """
    meta = doc.metadata or {}
    parsed = parse_parent_key(key)
    return ParentRef(
        parent_id=key,
        document_key=str(meta.get("document_key") or (parsed[2] if parsed else "")),
        rulebook_name=str(meta.get("rulebook_name") or ""),
        page_ordinal=int(meta.get("page_ordinal", parsed[3] if parsed else 0)),
        page_num=str(meta.get("page_num") or ""),
        page_chunk_index=int(meta.get("page_chunk_index", parsed[4] if parsed else 0)),
        content=doc.page_content or "",
    )


def order_parents(parents: Iterable[ParentRef]) -> list[ParentRef]:
    """Corpus reading order: rulebook, then page, then chunk within the page.

    Not the docstore key order — ``ayield_keys`` is unordered, and sorting the
    key as a string puts ``page_ordinal`` 10 before 2, which would make both the
    adjacency filter and ``--offset`` quietly wrong.
    """
    return sorted(parents, key=lambda p: p.sort_key)


def sample_seeds(seeds: Sequence[T], count: int) -> list[T]:
    """Evenly spaced subsample of ``seeds``, preserving corpus order.

    Evenly spaced rather than random, for two reasons. The seed set has to be
    stable so ``--resume`` picks up the same seeds on a second invocation, and
    proportional coverage across rulebooks and chapters falls out for free.

    Contiguous ``--offset``/``--limit`` windows give neither: an 8-seed Oathsworn
    run at offset 40 produced 40 questions that were all about one section of
    one rulebook (saving, swapping, Tally, Epilogue), because those eight
    parents sit next to each other in the book. A partial sweep of a 1,300-chunk
    game with ``--limit`` samples one chapter, not the game.

    ``count`` at or above the population is a no-op, so callers do not have to
    special-case a small game.
    """
    n = len(seeds)
    if count <= 0 or count >= n:
        return list(seeds)
    # Strictly increasing while count < n, so indices never repeat.
    return [seeds[(i * n) // count] for i in range(count)]


def adjacency_block(
    ordered: Sequence[ParentRef], seed_id: str, radius: int
) -> set[str]:
    """Parent ids within ``radius`` positions of the seed, in the same rulebook.

    Neighbours of the seed tend to restate it rather than interact with it, so
    they make for single-hop questions. The rulebook clamp matters: a flat
    game-wide window would exclude the first chunk of book B when the seed is
    the last chunk of book A, which is not a restatement, it is a different
    rulebook — and cross-book pairs are the ones we most want.
    """
    index = {p.parent_id: i for i, p in enumerate(ordered)}
    i = index.get(seed_id)
    if i is None:
        return {seed_id}
    seed_doc = ordered[i].document_key
    lo = max(0, i - radius)
    hi = min(len(ordered) - 1, i + radius)
    return {
        ordered[j].parent_id
        for j in range(lo, hi + 1)
        if ordered[j].document_key == seed_doc
    } | {seed_id}


# ---------------------------------------------------------------------------
# Fusion and context assembly (pure)
# ---------------------------------------------------------------------------


def fuse_concept_hits(
    concept: str,
    dense_parent_ids: Sequence[str],
    bm25_parent_ids: Sequence[str],
    *,
    rrf_k: float = 60.0,
) -> list[RetrievedHit]:
    """Fuse the two arms for one concept and reattach per-arm provenance.

    ``rrf_fuse`` consumes rank only and returns bare ids, which is right for two
    arms on incomparable scales (cosine distance vs BM25) but loses the
    attribution the run file needs — so the arm ranks are zipped back on here.
    """
    from meeplemate.search import rrf_fuse

    dense_rank = {pid: i for i, pid in enumerate(dense_parent_ids)}
    bm25_rank = {pid: i for i, pid in enumerate(bm25_parent_ids)}

    fused = rrf_fuse([list(dense_parent_ids), list(bm25_parent_ids)], rrf_k=rrf_k)

    hits: list[RetrievedHit] = []
    for rank, pid in enumerate(fused):
        arms = tuple(
            arm
            for arm, table in (("dense", dense_rank), ("bm25", bm25_rank))
            if pid in table
        )
        hits.append(
            RetrievedHit(
                parent_id=pid,
                concept=concept,
                fused_rank=rank,
                arms=arms,
                dense_rank=dense_rank.get(pid),
                bm25_rank=bm25_rank.get(pid),
            )
        )
    return hits


def _merge_arms(a: Sequence[str], b: Sequence[str]) -> tuple[str, ...]:
    merged = set(a) | set(b)
    return tuple(arm for arm in ARM_ORDER if arm in merged)


def select_context(
    seed: ParentRef,
    hits_by_concept: Mapping[str, Sequence[RetrievedHit]],
    parents_by_id: Mapping[str, ParentRef],
    *,
    excluded_ids: AbstractSet[str],
    per_concept_cap: int,
    max_context_chunks: int,
) -> ContextSelection:
    """Assemble the generation context: seed first, then retrieved chunks.

    Fills the budget round-robin across concepts rather than concatenating and
    truncating, so one concept with a long hit list cannot consume the whole
    context and leave the other unrepresented — the two-cluster shape is the
    entire point of the retrieval step.

    A parent found by both concepts collapses into one entry naming both, with
    the union of its arms and its best fused rank. Deterministic given the same
    inputs.
    """
    seed_entry = ContextEntry(
        parent=seed, role="seed", concepts=(), arms=(), fused_rank=None
    )

    dropped_adjacent: list[str] = []
    dropped_over_cap: list[str] = []

    # Per concept, in fused order, minus anything excluded or unresolvable.
    queues: dict[str, list[RetrievedHit]] = {}
    for concept, hits in hits_by_concept.items():
        queue: list[RetrievedHit] = []
        for hit in hits:
            if hit.parent_id in excluded_ids:
                if hit.parent_id not in dropped_adjacent:
                    dropped_adjacent.append(hit.parent_id)
                continue
            if hit.parent_id not in parents_by_id:
                # Retrieved a parent we have no Document for. Possible if the
                # vector store and docstore have drifted; skip rather than
                # fabricate an entry with an empty body.
                continue
            queue.append(hit)
        queues[concept] = queue

    budget = max(0, max_context_chunks - 1)  # the seed occupies one slot
    taken: dict[str, ContextEntry] = {}
    taken_per_concept: dict[str, int] = {c: 0 for c in queues}
    cursors: dict[str, int] = {c: 0 for c in queues}

    # Round-robin in a fixed concept order so the result is reproducible.
    concept_order = list(queues.keys())
    progress = True
    while progress and len(taken) < budget:
        progress = False
        for concept in concept_order:
            if len(taken) >= budget:
                break
            if taken_per_concept[concept] >= per_concept_cap:
                continue
            queue = queues[concept]
            i = cursors[concept]
            while i < len(queue):
                hit = queue[i]
                i += 1
                existing = taken.get(hit.parent_id)
                if existing is not None:
                    # Already in the context via the other concept: merge rather
                    # than spend a second slot on the same chunk.
                    taken[hit.parent_id] = ContextEntry(
                        parent=existing.parent,
                        role="retrieved",
                        concepts=tuple(
                            dict.fromkeys((*existing.concepts, hit.concept))
                        ),
                        arms=_merge_arms(existing.arms, hit.arms),
                        fused_rank=min(
                            existing.fused_rank
                            if existing.fused_rank is not None
                            else hit.fused_rank,
                            hit.fused_rank,
                        ),
                    )
                    continue
                taken[hit.parent_id] = ContextEntry(
                    parent=parents_by_id[hit.parent_id],
                    role="retrieved",
                    concepts=(hit.concept,),
                    arms=hit.arms,
                    fused_rank=hit.fused_rank,
                )
                taken_per_concept[concept] += 1
                progress = True
                break
            cursors[concept] = i

    # Anything left unconsumed was cut by a cap or the budget.
    for concept, queue in queues.items():
        for hit in queue[cursors[concept] :]:
            if hit.parent_id not in taken and hit.parent_id not in dropped_over_cap:
                dropped_over_cap.append(hit.parent_id)

    entries = (seed_entry, *taken.values())
    return ContextSelection(
        entries=entries,
        dropped_adjacent=tuple(dropped_adjacent),
        dropped_over_cap=tuple(dropped_over_cap),
    )


# ---------------------------------------------------------------------------
# LLM output normalisation (pure)
# ---------------------------------------------------------------------------


def normalise_concepts(raw: Sequence[str], *, max_words: int = 8) -> list[str]:
    """Clean the concept list from a structured LLM response.

    Guided decoding constrains the *shape* of the output, not its content: the
    model still returns sentences instead of noun phrases, repeats itself with
    different capitalisation, or emits one concept when asked for two. Dropping
    a long "concept" matters because it goes straight into a retrieval query,
    where a full sentence retrieves noise.

    Also drops a concept contained in one already kept ("Stage" alongside "Stage
    Card"). Two overlapping concepts retrieve the same chunks, which collapses
    the two-cluster context the retrieval step exists to build. It cannot catch
    near-synonyms that share no substring — "Stage Card" and "Stage Deck" get
    through — so ``concept_retrieval_overlap`` reports what survives.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or ():
        concept = re.sub(r"\s+", " ", (item or "").strip().strip('".,;:'))
        if not concept:
            continue
        if len(concept.split()) > max_words:
            continue
        key = concept.casefold()
        if key in seen:
            continue
        if any(key in kept or kept in key for kept in seen):
            continue
        seen.add(key)
        out.append(concept)
    return out


def concept_retrieval_overlap(
    hits_by_concept: Mapping[str, Sequence[RetrievedHit]]
) -> float:
    """Jaccard overlap between the concepts' retrieved parent sets.

    The point of retrieving per concept is to assemble a context with two
    *distant* clusters in it. When the model returns near-synonyms ("Stage Card"
    / "Stage Deck") both arms return the same parents, the context is one
    cluster, and the questions that come out are single-hop more often.

    Recorded per candidate rather than gated on: a high number does not mean the
    questions are bad — it means the concepts did no work and whatever came out
    came from the prompt. That distinction is what you need when tuning.
    """
    sets = [
        {hit.parent_id for hit in hits}
        for hits in hits_by_concept.values()
        if hits
    ]
    if len(sets) < 2:
        return 0.0
    union: set[str] = set().union(*sets)
    if not union:
        return 0.0
    intersection = set.intersection(*sets)
    return round(len(intersection) / len(union), 4)


def normalise_questions(raw: Sequence[str], *, count: int) -> list[str]:
    """Clean the question list from a structured LLM response."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or ():
        question = re.sub(r"\s+", " ", (item or "").strip())
        if not question:
            continue
        key = question.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(question)
    return out[:count]


# ---------------------------------------------------------------------------
# Prompts and schemas
# ---------------------------------------------------------------------------


class ExtractedConcepts(TypedDict):
    reasoning: Annotated[
        str, ..., "One or two sentences on what this passage leaves unresolved."
    ]
    concepts: Annotated[
        list[str],
        ...,
        "Short noun phrases, 1-4 words each, naming rule concepts the passage "
        "refers to but does not fully define.",
    ]


class GeneratedCandidates(TypedDict):
    reasoning: Annotated[
        str,
        ...,
        "Which rule interactions across the passages you found, and why each is "
        "not settled by a single passage.",
    ]
    questions: Annotated[
        list[str],
        ...,
        "Self-contained rules questions, each requiring more than one passage.",
    ]


EXTRACT_CONCEPTS_TEMPLATE = """\
You are reading one passage from the rulebooks of the board game "{{game_name}}".

<passage rulebook="{{rulebook_name}}" page="{{page_num}}">
{{seed_text}}
</passage>

Identify exactly {{count}} distinct rule concepts that this passage REFERS TO or DEPENDS ON
but does NOT itself fully define or resolve. These are the loose ends: a named test, a timing
window, a status, a keyword, a component, or a phase that the passage assumes the reader
already knows, or mentions without stating the whole rule.

Requirements:
- Each concept is a short noun phrase, 1-4 words, worded the way the rulebook words it
  (for example "psychology test", "break test", "line of sight", "knockback").
- The concepts must be distinct from each other, not two phrasings of one idea.
- Prefer a concept whose full rule plainly lives ELSEWHERE in the rulebooks over one this
  passage already answers on its own.
- Do not invent terminology, and do not write sentences.
- If the passage genuinely has only one such concept, give as the second one a concept a
  player would still need to look up in order to apply this passage correctly.

Give your response in the following JSON format:

{
"reasoning": "<...what this passage leaves unresolved...>",
"concepts": ["<concept 1>", "<concept 2>"]
}
"""

EXTRACT_CONCEPTS_PROMPT = ChatPromptTemplate.from_messages(
    [("human", EXTRACT_CONCEPTS_TEMPLATE)],
    template_format="mustache",
)


# The extracted concepts are deliberately ABSENT from this prompt. They are a
# retrieval device used to assemble a context with two distant clusters in it,
# not a specification for the question. Constraining the question to concept A x
# concept B can only lose the better A x C question sitting in the same window.
# Adding a {{#concepts}} section here would quietly re-narrow the eval set.
GENERATE_CANDIDATES_TEMPLATE = """\
You are a veteran player of the board game "{{game_name}}", mid-game, hitting situations the
rulebook does not obviously settle.

Below are passages from the rulebooks. The first one is the passage in front of you; the rest
come from elsewhere in the rules.

<passages>
{{#chunks}}
<passage rulebook="{{rulebook}}" page="{{page_num}}"{{#is_seed}} focus="true"{{/is_seed}}>
{{content}}
</passage>
{{/chunks}}
</passages>

Write {{count}} questions a real player would ask a rules expert about these rules.

They are thumb-typing on a phone, mid-game, with the box open in front of them. They are
terse. Write what they would actually type, not a well-formed paragraph.

Requirements:
- Each question must be answerable from the passages above, and answering it must require
  reading MORE THAN ONE of them. A question that a single passage settles on its own is not
  what we want.
- Go after the interactions: exceptions that collide, timing that is ambiguous when two rules
  both apply, a keyword defined in one place and used in another, behaviour at a boundary.
- Ask about whatever these passages actually put in tension. Do NOT restrict yourself to any
  particular topic, and you do not have to use every passage.
- Keep every question under {{max_words}} words. Shorter is better; the best ones are under
  fifteen. One sentence where you can.
- Do NOT state the rules inside the question. The player is asking BECAUSE they do not know
  the rule — someone who can recite it does not need to ask. Reciting the rule and then
  asking how it applies gives the answer away and is not how anyone types on a phone.
- Name the pieces, characters, keywords and states involved, in the game's own words, so a
  reader who cannot see these passages knows what is being asked. Naming a thing is enough:
  do NOT go on to explain what it does. Never say "this character" or "the ability above".
- No lore, no trivia, nothing about the rulebook as a document ("what does page 30 say"), no
  citations, no "according to the rules".
- The {{count}} questions must be about {{count}} different situations.

The difference this makes. The example below is from a DIFFERENT game, one that has nothing
to do with "{{game_name}}". Copy its shape only. Its words — fuel, reserve track, Fuel token
— do not exist in this game, so no question you write may contain them.

  TOO LONG - recites two rules before it asks anything:
    "I've spent all my fuel so my reserve track is completely empty, but I just picked up a
    +2 Fuel token. Can I spend it to move 2 fuel from reserve to available, or does the rule
    that says you can't refill more than you have in reserve prevent me from using it?"

  What a player actually types:
    "My fuel reserve is empty. Can I still use a +2 Fuel token?"

Both ask the same thing. The second one is the one we want. Write your {{count}} questions
about "{{game_name}}", using only vocabulary from the passages above.

Give your response in the following JSON format:

{
"reasoning": "<...which interactions you found...>",
"questions": ["<question 1>", "<question 2>", ...]
}
"""

GENERATE_CANDIDATES_PROMPT = ChatPromptTemplate.from_messages(
    [("human", GENERATE_CANDIDATES_TEMPLATE)],
    template_format="mustache",
)


def selection_to_prompt_chunks(selection: ContextSelection) -> list[dict[str, Any]]:
    """Render a context for the mustache ``{{#chunks}}`` section.

    ``is_seed`` must be a real boolean key for the ``{{#is_seed}}`` section to
    fire; mustache treats a missing key and ``False`` alike, but an empty string
    would also be falsey and is easy to introduce by accident.
    """
    return [
        {
            "rulebook": entry.parent.rulebook_name,
            "page_num": entry.parent.page_num or str(entry.parent.page_ordinal),
            "content": entry.parent.content,
            "is_seed": entry.role == "seed",
        }
        for entry in selection.entries
    ]


# ---------------------------------------------------------------------------
# Dedupe (pure)
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two matrices of embeddings.

    Normalises explicitly. bge-small already returns unit vectors, but relying
    on that couples this function to one embedding model for the sake of one
    line.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim == 2 else 0, b.shape[0] if b.ndim == 2 else 0))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def dedupe_candidates(
    candidate_ids: Sequence[str],
    candidate_questions: Sequence[str],
    candidate_vectors: np.ndarray,
    existing: Sequence[ExistingQuestion],
    existing_vectors: np.ndarray,
    *,
    game_id: str,
    threshold: float,
) -> list[DedupeVerdict]:
    """Flag near-duplicates against the batch and against ``test_cases.yaml``.

    Greedy keep-first: a candidate already flagged as a near-duplicate is not
    used as a comparison target for later candidates, so three paraphrases
    collapse to one keeper plus two flagged rather than forming a chain.

    ``status`` is gated on same-game existing questions only. ``test_suites``
    spans every game, and bge-small keys heavily on surface form, so a Warhammer
    question sits high against a structurally identical Oathsworn one. The
    cross-game number is reported as a diagnostic and never gates.
    """
    n = len(candidate_ids)
    if n == 0:
        return []

    same_idx = [i for i, e in enumerate(existing) if e.game_id == game_id]
    cross_idx = [i for i, e in enumerate(existing) if e.game_id != game_id]

    sim_existing = (
        cosine_similarity_matrix(candidate_vectors, existing_vectors)
        if len(existing)
        else np.zeros((n, 0))
    )
    sim_candidates = cosine_similarity_matrix(candidate_vectors, candidate_vectors)

    verdicts: list[DedupeVerdict] = []
    kept: list[int] = []

    for i in range(n):
        near: list[tuple[str, str, float]] = []

        best_existing_sim = -1.0
        best_existing_j: Optional[int] = None
        for j in same_idx:
            s = float(sim_existing[i, j])
            if s > best_existing_sim:
                best_existing_sim, best_existing_j = s, j
            if s >= threshold:
                near.append(("existing", existing[j].ref, s))

        best_candidate_sim = -1.0
        best_candidate_j: Optional[int] = None
        for j in kept:
            s = float(sim_candidates[i, j])
            if s > best_candidate_sim:
                best_candidate_sim, best_candidate_j = s, j
            if s >= threshold:
                near.append(("candidate", candidate_ids[j], s))

        cross_max = max(
            (float(sim_existing[i, j]) for j in cross_idx), default=0.0
        )

        # Existing wins ties: a candidate that duplicates a hand-written case is
        # more useful to report as such than as a duplicate of its own batch.
        if best_existing_sim >= best_candidate_sim:
            max_sim, kind, ref, question = (
                best_existing_sim,
                "existing",
                existing[best_existing_j].ref if best_existing_j is not None else None,
                existing[best_existing_j].question if best_existing_j is not None else None,
            )
        else:
            max_sim, kind, ref, question = (
                best_candidate_sim,
                "candidate",
                candidate_ids[best_candidate_j] if best_candidate_j is not None else None,
                candidate_questions[best_candidate_j] if best_candidate_j is not None else None,
            )

        if best_existing_sim >= threshold:
            status: Literal["unique", "near_candidate", "near_existing"] = "near_existing"
        elif best_candidate_sim >= threshold:
            status = "near_candidate"
        else:
            status = "unique"
            kept.append(i)

        verdicts.append(
            DedupeVerdict(
                status=status,
                max_similarity=round(max(max_sim, 0.0), 4),
                nearest_kind=kind if ref is not None else None,
                nearest_ref=ref,
                nearest_question=question,
                near_duplicates=tuple(
                    sorted(near, key=lambda t: -t[2])
                ),
                cross_game_max_similarity=round(max(cross_max, 0.0), 4),
            )
        )

    return verdicts


def similarity_histogram(
    verdicts: Sequence[DedupeVerdict], *, bins: int = 10
) -> list[tuple[float, float, int]]:
    """(lo, hi, count) buckets of ``max_similarity`` over [0, 1].

    Printed after a run because the threshold is a guess until you have seen the
    distribution: bge-small has a high similarity floor on short text, so the
    interesting band is narrow and corpus-specific.
    """
    counts = [0] * bins
    for v in verdicts:
        idx = min(bins - 1, max(0, int(v.max_similarity * bins)))
        counts[idx] += 1
    return [(i / bins, (i + 1) / bins, counts[i]) for i in range(bins)]


def length_histogram(
    questions: Sequence[str], *, edges: Sequence[int] = (10, 15, 20, 25, 30, 40)
) -> list[tuple[str, int]]:
    """(label, count) buckets of question word count.

    Printed after a run for the same reason as the similarity histogram: the
    prompt asks for brevity and cannot enforce it, so the distribution is the
    only evidence that it landed. The default edges bracket the median of the
    hand-written cases in ``test_cases.yaml`` (13 words).
    """
    counts = [0] * (len(edges) + 1)
    for q in questions:
        n = len(q.split())
        for i, edge in enumerate(edges):
            if n <= edge:
                counts[i] += 1
                break
        else:
            counts[-1] += 1

    labels: list[str] = []
    prev = 0
    for edge in edges:
        labels.append(f"{prev + 1}-{edge}")
        prev = edge
    labels.append(f"{prev + 1}+")
    return list(zip(labels, counts))


# ---------------------------------------------------------------------------
# Run document serialisation
# ---------------------------------------------------------------------------


class _MiningDumper(yaml.SafeDumper):
    """Local dumper subclass.

    Registering a str representer on ``yaml.SafeDumper`` itself would change
    every ``spit_yaml`` call in the process, including the game manifests ingest
    writes. Subclassing keeps the block-scalar style scoped to this module.
    """


def _literal_block_for_multiline(dumper: yaml.Dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_MiningDumper.add_representer(str, _literal_block_for_multiline)


def candidate_id(game_id: str, seed_ordinal: int, question_index: int) -> str:
    return f"{game_id}-{seed_ordinal:04d}-{question_index}"


def _context_rows(
    selection: ContextSelection, *, include_text: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in selection.entries:
        row: dict[str, Any] = {
            "parent_id": entry.parent.parent_id,
            "rulebook": entry.parent.rulebook_name,
            "page_ordinal": entry.parent.page_ordinal,
            "page_num": entry.parent.page_num,
            "role": entry.role,
            "concepts": list(entry.concepts),
            "arms": list(entry.arms),
            "rank": entry.fused_rank,
        }
        if include_text:
            row["content"] = entry.parent.content
        rows.append(row)
    return rows


def build_seed_record(
    game_id: str, result: SeedResult, *, include_context_text: bool = False
) -> dict[str, Any]:
    """One seed's outcome as a standalone record.

    Seed-level facts -- concepts, context, overlap -- live here once rather than
    being repeated on each of the five candidates, which is how the earlier flat
    document carried them.

    A failed seed still produces a record, holding the error and no candidates.
    That is what lets ``--resume`` skip it the way it skips a success, and makes
    "which seed failed and why" answerable without a run summary.
    """
    return {
        "seed_ordinal": result.seed_ordinal,
        "seed": {
            "parent_id": result.seed.parent_id,
            "rulebook": result.seed.rulebook_name,
            "page_ordinal": result.seed.page_ordinal,
            "page_num": result.seed.page_num,
        },
        "error": result.error,
        "concepts": list(result.concepts),
        # High overlap means the two concepts retrieved the same chunks, so the
        # context is one cluster rather than two and the questions came from the
        # prompt rather than from the retrieval.
        "concept_retrieval_overlap": result.concept_overlap,
        "context": (
            _context_rows(result.selection, include_text=include_context_text)
            if result.selection is not None else []
        ),
        "context_chunk_count": (
            len(result.selection.entries) if result.selection is not None else 0
        ),
        "dropped_adjacent": (
            len(result.selection.dropped_adjacent) if result.selection is not None else 0
        ),
        "candidates": [
            {
                "id": candidate_id(game_id, result.seed_ordinal, i),
                "question": q,
                "word_count": len(q.split()),
            }
            for i, q in enumerate(result.questions)
        ],
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


async def load_parents(docstore: Any, gkey: str) -> list[ParentRef]:
    """Every parent chunk for one game version, in corpus reading order."""
    prefix = f"{gkey}#"
    keys = [k async for k in docstore.ayield_keys(prefix=prefix)]
    docs = await docstore.amget(keys)
    refs = [
        parent_ref_from_document(k, d)
        for k, d in zip(keys, docs)
        if d is not None
    ]
    return order_parents(refs)


def load_existing_questions() -> list[ExistingQuestion]:
    """Questions already in ``test_cases.yaml``, across every game.

    Queries are ``|`` block scalars so each one ends in a newline; that would
    otherwise ride into the embedding and shift every similarity slightly.
    """
    from meeplemate.eval import test_suites

    out: list[ExistingQuestion] = []
    for suite in test_suites:
        game_id = (suite.get("params") or {}).get("game_id", "")
        for case in suite.get("test_cases") or ():
            query = (case.get("query") or "").strip()
            if not query:
                continue
            out.append(
                ExistingQuestion(
                    ref=f"{suite.get('name', '?')}/{case.get('name', '?')}",
                    game_id=game_id,
                    question=query,
                )
            )
    return out


async def retrieve_for_concept(
    *,
    vector_store: Any,
    bm25: Any,
    concept: str,
    game_id: str,
    game_version: str,
    dense_k: int,
    bm25_k: int,
    rrf_k: float = 60.0,
) -> list[RetrievedHit]:
    """Both arms for one concept, fused, at mining settings.

    Mining settings, not production ones: no adaptive-k, no token budget, no
    document-order re-sort. ``build_chunk_search_service_2`` applies all three,
    and assembling context behind those cutoffs would restrict the eval set to
    questions production retrieval already finds — biasing e2e scores
    optimistically, and increasingly so as retrieval is tuned.
    """
    filter = {"game_id": game_id, "game_version": game_version}

    docs_and_scores = await vector_store.asimilarity_search_with_score(
        concept, filter=filter, k=dense_k
    )
    # RAW COSINE DISTANCE, lower is better. There is no HybridSearchConfig on
    # this store (see the comment in config.py), so these are not similarities
    # and not RRF scores. Sorting descending here yields the *worst* dense_k
    # children and the pipeline goes on producing confident-looking questions.
    docs_and_scores.sort(key=lambda x: x[1])

    dense_parent_ids: list[str] = []
    seen: set[str] = set()
    for doc, _score in docs_and_scores:
        parent_id = (doc.metadata or {}).get("doc_id")
        if parent_id is not None and parent_id not in seen:
            seen.add(parent_id)
            dense_parent_ids.append(parent_id)

    bm25_hits = await bm25.asearch(concept, game_version, k=bm25_k)
    bm25_parent_ids = [parent_id for parent_id, _score in bm25_hits]

    return fuse_concept_hits(
        concept, dense_parent_ids, bm25_parent_ids, rrf_k=rrf_k
    )


@dataclass
class MiningDeps:
    """Everything ``mine_seed`` needs from the component system."""

    vector_store: Any
    bm25: Any
    concept_chain: Any
    question_chain: Any
    ordered_parents: list[ParentRef]
    parents_by_id: dict[str, ParentRef]
    game_id: str
    game_name: str
    game_version: str


@dataclass
class MiningParams:
    concepts_per_seed: int = 2
    questions_per_seed: int = 5
    dense_k: int = 40
    bm25_k: int = 30
    rrf_k: float = 60.0
    per_concept_cap: int = 6
    max_context_chunks: int = 12
    adjacency_radius: int = 1
    # Rendered into the generation prompt, not enforced afterwards. The
    # hand-written cases in test_cases.yaml have a median of 13 words; a cap
    # here is a nudge toward that, and `word_count` on every candidate says
    # whether the nudge landed. Truncating an over-long question instead would
    # produce a corpus of sentence fragments.
    max_question_words: int = 25


async def mine_seed(
    seed: ParentRef,
    seed_ordinal: int,
    *,
    deps: MiningDeps,
    params: MiningParams,
) -> SeedResult:
    """One seed, end to end. Never raises — failures come back on the result.

    A sweep is 100+ independent seeds against a shared endpoint; two timeouts
    must not cost the other 115.
    """
    result = SeedResult(seed=seed, seed_ordinal=seed_ordinal)
    try:
        raw = await deps.concept_chain.ainvoke(
            {
                "game_name": deps.game_name,
                "rulebook_name": seed.rulebook_name,
                "page_num": seed.page_num or str(seed.page_ordinal),
                "seed_text": seed.content,
                "count": params.concepts_per_seed,
            }
        )
        concepts = normalise_concepts(raw.get("concepts") or [])[
            : params.concepts_per_seed
        ]
        result.concepts = concepts
        if not concepts:
            result.error = "no usable concepts extracted"
            return result

        hits_by_concept: dict[str, list[RetrievedHit]] = {}
        for concept in concepts:
            hits_by_concept[concept] = await retrieve_for_concept(
                vector_store=deps.vector_store,
                bm25=deps.bm25,
                concept=concept,
                game_id=deps.game_id,
                game_version=deps.game_version,
                dense_k=params.dense_k,
                bm25_k=params.bm25_k,
                rrf_k=params.rrf_k,
            )

        result.concept_overlap = concept_retrieval_overlap(hits_by_concept)

        excluded = adjacency_block(
            deps.ordered_parents, seed.parent_id, params.adjacency_radius
        )
        selection = select_context(
            seed,
            hits_by_concept,
            deps.parents_by_id,
            excluded_ids=excluded,
            per_concept_cap=params.per_concept_cap,
            max_context_chunks=params.max_context_chunks,
        )
        result.selection = selection

        if not selection.retrieved:
            result.error = "no context retrieved beyond the seed"
            return result

        raw_q = await deps.question_chain.ainvoke(
            {
                "game_name": deps.game_name,
                "chunks": selection_to_prompt_chunks(selection),
                "count": params.questions_per_seed,
                "max_words": params.max_question_words,
            }
        )
        questions = normalise_questions(
            raw_q.get("questions") or [], count=params.questions_per_seed
        )
        result.questions = questions
        if not questions:
            result.error = "no usable questions generated"
        return result
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        logger.exception(
            "Seed failed", parent_id=seed.parent_id, seed_ordinal=seed_ordinal
        )
        result.error = repr(exc)
        return result
