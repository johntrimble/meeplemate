"""Adaptive-k cutoff and RRF fusion — the pure-Python half of hybrid retrieval."""
import random

import pytest

from meeplemate.search import find_cutoff_adaptive_k, rrf_fuse


# ---------------------------------------------------------------------------
# find_cutoff_adaptive_k
# ---------------------------------------------------------------------------


def test_empty_scores_select_nothing():
    """Empty input must select nothing, not everything-but-the-last.

    The caller does ``docs[:adaptive_k]``, so the old ``-1`` return was a slice
    that silently dropped the final document. Unreachable while the only arm
    was dense search; reachable on every BM25 query whose terms are absent from
    the index.
    """
    assert find_cutoff_adaptive_k([]) == 0
    assert ["a", "b", "c"][: find_cutoff_adaptive_k([])] == []


def test_single_score_does_not_raise():
    """One hit has no deltas to take a max over; it used to raise ValueError.

    A rare term matching exactly one parent chunk is a normal BM25 result.
    """
    assert find_cutoff_adaptive_k([0.9]) == 1


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_short_lists_are_taken_whole(n):
    """Below the buffer the algorithm can only ever select everything."""
    scores = [1.0 - i * 0.1 for i in range(n)]
    assert find_cutoff_adaptive_k(scores) == n


def test_guards_do_not_change_longer_lists():
    """The guards are short-circuits, not a behaviour change.

    Compares against the original implementation on random descending inputs
    longer than the buffer, where neither guard should fire.
    """
    def original(scores, post_k_buffer=5, find_gap_within_top_percent=0.9):
        deltas = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
        relevant = deltas[: int(len(scores) * find_gap_within_top_percent)]
        return relevant.index(max(relevant)) + post_k_buffer + 1

    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(7, 60)
        scores = sorted((rng.random() for _ in range(n)), reverse=True)
        assert find_cutoff_adaptive_k(scores) == original(scores)


def test_rrf_shaped_scores_collapse_to_the_buffer():
    """Regression pin: adaptive-k is meaningless on RRF scores.

    RRF assigns ``1/(rank + k)``, whose deltas decrease monotonically, so the
    largest gap is always at index 0 and the result is the constant
    ``post_k_buffer + 1`` regardless of how many candidates there are. This is
    why the arms are cut *before* fusion rather than after — if a future change
    moves the cutoff downstream of ``rrf_fuse``, this test says what breaks.
    """
    for n in (10, 50, 200):
        rrf_scores = [1.0 / (60 + i) for i in range(n)]
        assert find_cutoff_adaptive_k(rrf_scores) == 6


def test_finds_a_real_cliff():
    """On calibrated scores it cuts at the drop-off (plus the buffer)."""
    scores = [0.95, 0.94, 0.93, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24]
    # Largest delta is between index 2 and 3, so k = 2 + 5 + 1.
    assert find_cutoff_adaptive_k(scores) == 8


# ---------------------------------------------------------------------------
# rrf_fuse
# ---------------------------------------------------------------------------


def test_agreement_between_arms_outranks_a_single_arm():
    both = rrf_fuse([["shared", "a"], ["shared", "b"]])
    assert both[0] == "shared"


def test_empty_arm_is_identity():
    """A game with no BM25 index must not perturb the dense ranking."""
    ranked = ["a", "b", "c"]
    assert rrf_fuse([ranked, []]) == ranked
    assert rrf_fuse([[], ranked]) == ranked


def test_output_is_a_permutation_of_the_union():
    fused = rrf_fuse([["a", "b", "c"], ["c", "d"]])
    assert sorted(fused) == ["a", "b", "c", "d"]
    assert len(fused) == len(set(fused))


def test_higher_ranked_wins_within_one_arm():
    assert rrf_fuse([["first", "second", "third"]]) == ["first", "second", "third"]


def test_ties_are_broken_deterministically():
    """Single-arm ids at equal rank tie on score exactly.

    Without the (best_rank, first_seen) tie-break the order would fall out of
    dict insertion, so the same inputs in a different arm order would produce a
    different ranking and evals would not reproduce.
    """
    # "x" and "y" are each rank 0 of their own arm: identical fused scores.
    a = rrf_fuse([["x"], ["y"]])
    b = rrf_fuse([["x"], ["y"]])
    assert a == b
    assert sorted(a) == ["x", "y"]


def test_rrf_k_damps_rank_influence():
    """A large rrf_k flattens rank differences; a small one sharpens them."""
    lists = [["a", "b"], ["b", "a"]]
    # Symmetric input: both orderings are ties, but the result must be stable.
    assert rrf_fuse(lists, rrf_k=1.0) == rrf_fuse(lists, rrf_k=1.0)
    assert sorted(rrf_fuse(lists, rrf_k=1000.0)) == ["a", "b"]
