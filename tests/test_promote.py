"""Tests for promoting reviewed candidates into golden test cases."""

import yaml

from meeplemate.eval.promote import (
    VERIFIED_MARKER,
    build_suites,
    build_test_case,
    case_name,
    clean_answer,
    dump_suites,
    evidence_stats,
    extract_evidence,
    unparseable_quotes,
    promoted_decisions,
    suite_name,
)

ANSWER = (
    "No, it does not.\n\n"
    f"{VERIFIED_MARKER}\n\n"
    "> You may carry any number of Small items, but only one Big one.\n"
    "> \n"
    "> (Munchkin Rules, p. 3)\n\n"
    "So the limit stands.\n"
)


def decision(cid="munchkin-0001-0", reviewer="yes", corrected=None, game="munchkin"):
    return {"id": cid, "game": game, "question": "Can I play a Curse?",
            "reviewer": reviewer, "answer_suspect": bool(corrected),
            "corrected_answer": corrected, "note": None}


def answer_rec(response=ANSWER):
    return {"candidate_id": "munchkin-0001-0", "answer": {"response": response}}


# --------------------------------------------------------------------------
# Answer cleaning
# --------------------------------------------------------------------------


def test_verified_markers_are_stripped():
    """They are the pipeline's rendering hint, not part of an answer -- leaving
    them in puts presentation markup in front of the correctness judge."""
    out = clean_answer(ANSWER)
    assert VERIFIED_MARKER not in out
    assert "data-quote-status" not in out


def test_blockquotes_survive_cleaning():
    """The hand-written cases quote the rulebook the same way."""
    assert "> You may carry any number" in clean_answer(ANSWER)


def test_trailing_whitespace_is_removed_from_every_line():
    """Trailing space forces PyYAML off literal block style onto an escaped
    double-quoted scalar, which is unreadable in a diff."""
    assert not any(l != l.rstrip() for l in clean_answer(ANSWER).split("\n"))


def test_blank_line_runs_are_collapsed():
    assert "\n\n\n" not in clean_answer(ANSWER)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


def test_evidence_comes_from_cited_blockquotes():
    ev = extract_evidence(ANSWER)
    assert len(ev) == 1
    assert ev[0]["rulebook"] == "Munchkin Rules"
    assert ev[0]["page"] == 3
    assert ev[0]["quote"].startswith("You may carry any number")
    assert ">" not in ev[0]["quote"]
    assert "(Munchkin Rules" not in ev[0]["quote"]


def test_numeric_pages_become_ints_matching_the_handwritten_cases():
    assert extract_evidence(ANSWER)[0]["page"] == 3


def test_a_non_numeric_page_is_dropped_and_reported():
    """`extract_ref_name_and_page` only recognises numeric pages, so `p. iv`
    yields no ref_name. The quote cannot become evidence -- the point of the
    report is that the loss is visible rather than silent."""
    from meeplemate.eval.promote import unparseable_quotes
    text = "> A rule\n>\n> (Some Book, p. iv)\n"
    assert extract_evidence(text) == []
    assert unparseable_quotes(text)


def test_uncited_blockquotes_are_dropped():
    """An evidence entry with no rulebook cannot be scored by mm-eval retrieval."""
    assert extract_evidence("> A rule with no citation\n") == []


def test_no_blockquotes_yields_no_evidence():
    assert extract_evidence("Just prose, no quotes.") == []


# --------------------------------------------------------------------------
# Case names
# --------------------------------------------------------------------------


def test_case_name_is_snake_case_from_the_question():
    assert case_name("Can I play a Curse?", set()) == "can_i_play_a_curse"


def test_case_names_are_deduped():
    taken = set()
    a = case_name("Can I play a Curse?", taken)
    b = case_name("Can I play a Curse?", taken)
    assert a != b and b.endswith("_2")


def test_case_name_does_not_end_on_a_function_word():
    """Truncation lands mid-phrase often; '..._during_combat_as_an' reads badly."""
    n = case_name("Can I play a Steed card during combat as an Item?", set())
    assert not n.endswith(("_as", "_an", "_the", "_of"))


def test_case_name_survives_an_empty_question():
    assert case_name("???", set()) == "question"


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------


def test_only_promoted_candidates_are_included():
    """A rejected candidate is skipped even with a correction written: the
    answer may predate the verdict, and the verdict is the decision."""
    rows = [decision("a", "yes"), decision("b", "no", corrected="An answer."),
            decision("c", "later")]
    assert [d["id"] for d in promoted_decisions(rows)] == ["a"]


def test_include_rejected_opts_them_back_in():
    rows = [decision("a", "yes"), decision("b", "no")]
    assert len(promoted_decisions(rows, include_rejected=True)) == 2


def test_reviewer_answer_wins_over_the_draft():
    case = build_test_case(decision(corrected="The corrected answer."),
                           answer_rec(), set())
    assert case["reference_answer"].startswith("The corrected answer.")
    assert case["source"]["answer"] == "reviewer"


def test_draft_is_used_when_there_is_no_correction():
    case = build_test_case(decision(), answer_rec(), set())
    assert case["reference_answer"].startswith("No, it does not.")
    assert case["source"]["answer"] == "agent"


def test_case_with_no_answer_at_all_is_skipped():
    """Writing an empty reference_answer would score as a failure every run."""
    assert build_test_case(decision(), {"answer": {"response": ""}}, set()) is None


def test_suites_are_one_per_game_in_sorted_order():
    rows = [decision("o-1", game="one_deck_dungeon"), decision("m-1", game="munchkin")]
    answers = {"o-1": answer_rec(), "m-1": answer_rec()}
    suites, _ = build_suites("r1", rows, answers)
    assert [s["params"]["game_id"] for s in suites] == ["munchkin", "one_deck_dungeon"]


def test_suite_name_carries_provenance():
    assert suite_name("one_deck_dungeon", "2026-08-28") == \
        "One Deck Dungeon (mined 2026-08-28)"


def test_names_are_deduped_across_games_not_just_within_a_suite():
    """Run files are <suite>__<case>, and two games can phrase a question the
    same way."""
    rows = [decision("m-1", game="munchkin"), decision("o-1", game="one_deck_dungeon")]
    suites, _ = build_suites("r1", rows, {"m-1": answer_rec(), "o-1": answer_rec()})
    names = [c["name"] for s in suites for c in s["test_cases"]]
    assert len(names) == len(set(names))


def test_skipped_ids_are_reported():
    rows = [decision("a"), decision("b")]
    _, skipped = build_suites("r1", rows, {"a": answer_rec(), "b": answer_rec("")})
    assert skipped == ["b"]


def test_evidence_stats_counts_reviewer_answers():
    rows = [decision("a"), decision("b", corrected="Mine.")]
    suites, _ = build_suites("r1", rows, {"a": answer_rec(), "b": answer_rec()})
    stats = evidence_stats(suites)
    assert stats["cases"] == 2 and stats["reviewer_answers"] == 1


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_dump_uses_block_scalars_and_round_trips(tmp_path):
    suites, _ = build_suites("r1", [decision()], {"munchkin-0001-0": answer_rec()})
    out = tmp_path / "r1.yaml"
    dump_suites(suites, out, run_id="r1", source="decisions.json")
    text = out.read_text()
    assert text.startswith("# Generated by `mm-eval promote r1`")
    assert "reference_answer: |" in text          # not an escaped quoted scalar
    assert 'reference_answer: "' not in text
    assert yaml.safe_load(text)[0]["params"]["game_id"] == "munchkin"


def test_dumped_file_matches_the_loader_schema(tmp_path):
    suites, _ = build_suites("r1", [decision()], {"munchkin-0001-0": answer_rec()})
    out = tmp_path / "r1.yaml"
    dump_suites(suites, out, run_id="r1", source="d.json")
    for suite in yaml.safe_load(out.read_text()):
        assert {"name", "params", "test_cases"} <= set(suite)
        assert "game_id" in suite["params"]
        for case in suite["test_cases"]:
            assert {"name", "query", "reference_answer", "evidence"} <= set(case)


def test_dump_does_not_touch_the_global_dumper(tmp_path):
    before = dict(yaml.SafeDumper.yaml_representers)
    suites, _ = build_suites("r1", [decision()], {"munchkin-0001-0": answer_rec()})
    dump_suites(suites, tmp_path / "r1.yaml", run_id="r1", source="d.json")
    assert yaml.SafeDumper.yaml_representers == before
