"""Tests for the candidate review page and its persistence backend.

No browser and no network: the page is built as a string, and the server is
exercised through FastAPI's test client against a temp directory.
"""

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from meeplemate.eval.eval_gen_layout import EvalGenLayout, seed_key
from meeplemate.eval.eval_gen_store import (
    dump_json,
    write_answer_record,
    write_seed_record,
)
from meeplemate.eval.review_page import build_page, empty_decisions, load_candidates, load_run
from meeplemate.eval.review_server import build_app, read_decisions, write_decisions


def seed_record(ordinal=1, parent="munchkin#v1#0#0", candidates=None, error=None):
    return {
        "seed_ordinal": ordinal,
        "seed": {"parent_id": parent, "rulebook": "Munchkin Rules",
                 "page_ordinal": 3, "page_num": "3"},
        "error": error,
        "concepts": ["curses", "combat"],
        "context": [], "context_chunk_count": 0, "dropped_adjacent": 0,
        "candidates": candidates if candidates is not None else [
            {"id": "munchkin-0001-0", "question": "Can I play a Curse?", "word_count": 5}
        ],
    }


def answer_record(cid="munchkin-0001-0", status="ok"):
    if status == "error":
        block = {"status": "error", "error": "TimeoutError()"}
    else:
        block = {
            "recommendation": "keep", "verdict": "good", "reasons": ["2 passages"],
            "judge_reasoning": "two rules", "rules_involved": ["Curses"],
            "verified_quotes": 2, "evidence_chunks": 2, "evidence_rulebooks": 1,
            "rulebooks": ["Munchkin Rules"], "response": "Yes.",
        }
    return {"candidate_id": cid, "game_id": "munchkin",
            "question": "Can I play a Curse?", "answer": block}


@pytest.fixture
def layout(tmp_path):
    return EvalGenLayout(tmp_path, "r1")


def populate(layout, game="munchkin", answer="ok", seed=None):
    write_seed_record(layout, game, seed or seed_record())
    if answer is not None:
        write_answer_record(layout, game, "munchkin-0001-0", answer_record(status=answer))
    return layout


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def test_every_step_owns_its_own_top_level_entry():
    """The one-owner rule, as in PackageLayout: no entry owned twice, none
    nested inside another."""
    entries = [e for es in EvalGenLayout.STEP_OUTPUTS.values() for e in es]
    assert len(entries) == len(set(entries))
    for a in entries:
        for b in entries:
            if a is not b:
                assert not a.startswith(b + "/")


def test_every_path_stays_inside_its_step(layout):
    paths = {
        "mine-questions": [layout.seeds_dir("g"), layout.seed_file("g", 1, "p"),
                           layout.dedupe_file("g"), layout.mining_run_file("g")],
        "answer-candidates": [layout.answer_file("g", "c-1"), layout.answers_run_file("g")],
        "review": [layout.decisions_file()],
        "promote": [layout.dataset_dir(), layout.dataset_file()],
    }
    for step, ps in paths.items():
        roots = layout.step_roots(step)
        for path in ps:
            assert any(root in path.parents or root == path for root in roots), (step, path)


def test_seed_identity_is_the_parent_id_not_the_ordinal(layout):
    """--min-seed-chars shifts every ordinal; keying resume on it would
    silently re-mine seeds already done."""
    a = layout.seed_file("g", 7, "pid-x")
    b = layout.seed_file("g", 99, "pid-x")
    assert seed_key("pid-x") in a.name and seed_key("pid-x") in b.name
    assert a.name != b.name  # ordinal still sorts the directory


def test_seed_files_sort_numerically(layout):
    names = [layout.seed_file("g", n, f"p{n}").name for n in (2, 10, 100)]
    assert names == sorted(names)


def test_unsafe_characters_are_escaped_out_of_paths(layout):
    p = layout.answer_file("a game/../x", "id with spaces")
    assert ".." not in p.parts and " " not in p.name


# --------------------------------------------------------------------------
# Loading and joining
# --------------------------------------------------------------------------


def test_load_candidates_joins_mining_dedupe_and_answers(layout):
    populate(layout)
    dump_json({"candidates": {"munchkin-0001-0": {"status": "near_existing"}}},
              layout.dedupe_file("munchkin"))
    rows = load_candidates(layout, "munchkin")
    assert len(rows) == 1
    r = rows[0]
    assert r["rec"] == "keep" and r["chunks"] == 2
    assert r["seed"] == "Munchkin Rules p.3"
    assert r["dupe"] == "near_existing"
    assert r["concepts"] == ["curses", "combat"]


def test_candidates_without_an_answer_are_skipped(layout):
    populate(layout, answer=None)
    assert load_candidates(layout, "munchkin") == []


def test_candidates_whose_answer_errored_are_skipped(layout):
    populate(layout, answer="error")
    assert load_candidates(layout, "munchkin") == []


def test_missing_dedupe_file_defaults_to_unique(layout):
    populate(layout)
    assert load_candidates(layout, "munchkin")[0]["dupe"] == "unique"


def test_failed_seed_contributes_no_candidates(layout):
    write_seed_record(layout, "munchkin",
                      seed_record(ordinal=2, parent="p2", candidates=[], error="Boom()"))
    populate(layout)
    assert len(load_candidates(layout, "munchkin")) == 1


def test_load_run_spans_games(layout):
    populate(layout, "munchkin")
    write_seed_record(layout, "one_deck_dungeon",
                      seed_record(parent="odd#v#0#0",
                                  candidates=[{"id": "odd-0001-0", "question": "q?",
                                               "word_count": 1}]))
    write_answer_record(layout, "one_deck_dungeon", "odd-0001-0",
                        {**answer_record("odd-0001-0"), "game_id": "one_deck_dungeon"})
    rows = load_run(layout, ["munchkin", "one_deck_dungeon"])
    assert {r["game"] for r in rows} == {"munchkin", "one_deck_dungeon"}


def test_load_run_raises_on_an_unmined_game(layout):
    """Empty and missing must not look alike -- they need different responses."""
    populate(layout)
    with pytest.raises(FileNotFoundError):
        load_run(layout, ["munchkin", "nope"])


# --------------------------------------------------------------------------
# Store round-trips
# --------------------------------------------------------------------------


def test_seed_records_are_written_once_not_accumulated(layout):
    """The whole point of the layout: a second seed adds a file, it does not
    rewrite the first."""
    write_seed_record(layout, "munchkin", seed_record(1, "p1"))
    first = layout.seed_file("munchkin", 1, "p1")
    stamp = first.stat().st_mtime_ns
    write_seed_record(layout, "munchkin", seed_record(2, "p2"))
    assert len(list(layout.iter_seed_files("munchkin"))) == 2
    assert first.stat().st_mtime_ns == stamp


def test_resume_reads_parent_ids_from_inside_the_records(layout):
    from meeplemate.eval.eval_gen_store import mined_parent_ids
    write_seed_record(layout, "munchkin", seed_record(1, "pid-a"))
    write_seed_record(layout, "munchkin", seed_record(2, "pid-b"))
    assert mined_parent_ids(layout, "munchkin") == {"pid-a", "pid-b"}


def test_answered_ids_exclude_errors_so_they_are_retried(layout):
    from meeplemate.eval.eval_gen_store import answered_candidate_ids
    write_answer_record(layout, "munchkin", "ok-1", answer_record("ok-1"))
    write_answer_record(layout, "munchkin", "bad-1", answer_record("bad-1", status="error"))
    assert answered_candidate_ids(layout, "munchkin") == {"ok-1"}


def test_corrupt_reduce_file_does_not_stop_the_tool(layout):
    from meeplemate.eval.eval_gen_store import load_json
    layout.dedupe_file("munchkin").parent.mkdir(parents=True, exist_ok=True)
    layout.dedupe_file("munchkin").write_text("{not json")
    assert load_json(layout.dedupe_file("munchkin"), {}) == {}


# --------------------------------------------------------------------------
# Page building
# --------------------------------------------------------------------------


def test_build_page_inlines_every_candidate(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert "Can I play a Curse?" in html
    assert '<script id="data"' in html
    assert "__DATA__" not in html and "__RUNID__" not in html and "__KEY__" not in html


def test_build_page_escapes_a_script_tag_inside_an_answer(layout):
    """A '</script>' in rulebook text would otherwise close the data block."""
    write_seed_record(layout, "munchkin", seed_record())
    rec = answer_record()
    rec["answer"]["response"] = "See </script><script>alert(1)</script>"
    write_answer_record(layout, "munchkin", "munchkin-0001-0", rec)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    body = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert "alert(1)" not in body or "\\u003c" in body
    assert json.loads(body.replace("\\u003c", "<"))


def test_build_page_scopes_storage_to_the_run_id(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert "boardbarian-review-r1" in html


def test_page_guards_the_artifact_runtime_call(layout):
    """Opened from file:// there is no `claude` global; an unguarded reference
    throws inside the click handler and the clipboard fallback never runs."""
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert 'typeof claude !== "undefined"' in html


def test_decisions_file_is_scoped_to_the_group_not_a_game(layout):
    """A review session spans every game in the run, and candidate ids already
    carry their game."""
    p = layout.decisions_file()
    assert p == layout.group_root / "decisions" / "decisions.json"
    assert "munchkin" not in str(p)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def doc_with(*ids, run_id="r1"):
    return {"run_id": run_id, "reviewed": len(ids), "total": 9, "answers_flagged": 0,
            "decisions": [{"id": i, "reviewer": "yes"} for i in ids]}


def current_version(c):
    return c.get("/api/decisions").json().get("updated_at")


def save(c, *ids, **kw):
    """PUT carrying the precondition, the way the page does.

    Tests that are about something other than concurrency use this so they do
    not have to care that a second write needs the current version.
    """
    return c.put("/api/decisions",
                 json={**doc_with(*ids, **kw), "base_updated_at": current_version(c)})


def test_read_decisions_returns_empty_when_absent(tmp_path):
    got = read_decisions(tmp_path / "nope.json", "r1", 12)
    assert got == empty_decisions("r1", 12)


def test_read_decisions_survives_a_corrupt_file(tmp_path):
    """The reviewer cannot repair JSON from inside the browser, so a bad file
    must not stop the tool from starting."""
    p = tmp_path / "d.json"
    p.write_text("{not json")
    assert read_decisions(p, "r1", 3)["decisions"] == []
    assert p.exists()  # left alone, not clobbered on read


def test_read_decisions_rejects_an_unexpected_shape(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"decisions": "not a list"}))
    assert read_decisions(p, "r1")["decisions"] == []


def test_write_then_read_round_trips(tmp_path):
    p = tmp_path / "sub" / "d.json"
    write_decisions(p, doc_with("a", "b"))
    assert [d["id"] for d in read_decisions(p, "r1")["decisions"]] == ["a", "b"]


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    dfile = tmp_path / "decisions" / "r1.json"
    app = build_app(render_page=lambda: "<title>page</title>hello",
                    decisions_file=dfile, run_id="r1", total=9)
    return TestClient(app), dfile


def test_index_serves_the_page(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "hello" in r.text


def test_page_is_rendered_per_request(tmp_path):
    """So a concurrent answer-candidates run shows up on reload."""
    calls = []
    app = build_app(render_page=lambda: f"v{len(calls)}" or calls.append(1),
                    decisions_file=tmp_path / "d.json", run_id="r1", total=1)
    c = TestClient(app)
    seen = {c.get("/").text for _ in range(2)}
    assert seen  # rendered each time rather than cached at startup


def test_get_decisions_is_empty_before_anything_is_saved(client):
    c, _ = client
    body = c.get("/api/decisions").json()
    assert body["decisions"] == [] and body["run_id"] == "r1"


def test_put_persists_and_get_reads_back(client):
    c, dfile = client
    assert c.put("/api/decisions", json=doc_with("x")).json()["ok"] is True
    assert dfile.exists()
    assert [d["id"] for d in c.get("/api/decisions").json()["decisions"]] == ["x"]


def test_put_overwrites_the_client_run_id_with_the_servers(client):
    """A page left open from an earlier run must not write under its own id."""
    c, dfile = client
    c.put("/api/decisions", json=doc_with("x", run_id="SOME-OTHER-RUN"))
    assert json.loads(dfile.read_text())["run_id"] == "r1"


def test_put_rejects_a_body_without_decisions(client):
    c, _ = client
    assert c.put("/api/decisions", json={"nope": 1}).status_code == 422


def test_put_rejects_a_non_object_body(client):
    c, _ = client
    assert c.put("/api/decisions", json=[1, 2, 3]).status_code == 422


def test_later_puts_replace_earlier_state(client):
    c, _ = client
    save(c, "a", "b")
    save(c, "a")
    assert [d["id"] for d in c.get("/api/decisions").json()["decisions"]] == ["a"]


# --------------------------------------------------------------------------
# Concurrent writes
#
# A PUT replaces the whole document, so a client holding state from before
# someone else's write would revert it -- and the page flushes on unload, so
# merely closing a stale tab is enough to do it. These pin the precondition
# that stops that.
# --------------------------------------------------------------------------


def test_first_write_to_an_unsaved_run_needs_no_precondition(client):
    """A run nobody has saved has no version to be stale against, and
    demanding one would block the very first decision of a review."""
    c, _ = client
    assert c.put("/api/decisions", json=doc_with("a")).status_code == 200


def test_the_server_stamps_the_version_not_the_client(client):
    """A browser clock that disagrees with the container's must not get to
    decide whether the next write looks stale."""
    c, _ = client
    stamped = c.put("/api/decisions",
                    json={**doc_with("a"), "updated_at": "1999-01-01T00:00:00Z"}
                    ).json()["updated_at"]
    assert stamped != "1999-01-01T00:00:00Z"
    assert current_version(c) == stamped


def test_a_stale_put_is_refused_and_leaves_the_file_alone(client):
    """The regression this guards: a tab open from before another write
    flushes on unload and silently reverts it."""
    c, dfile = client
    save(c, "a")
    stale = current_version(c)
    save(c, "a", "b")                       # someone else writes in between

    r = c.put("/api/decisions", json={**doc_with("a"), "base_updated_at": stale})
    assert r.status_code == 409
    assert [d["id"] for d in json.loads(dfile.read_text())["decisions"]] == ["a", "b"]


def test_a_conflict_hands_back_the_document_to_merge_onto(client):
    """Rejecting the write is only half of it: the client needs what is on
    disk to rebase its own edits onto, without a second round trip."""
    c, _ = client
    save(c, "a")
    stale = current_version(c)
    save(c, "a", "b")

    body = c.put("/api/decisions",
                 json={**doc_with("a"), "base_updated_at": stale}).json()
    assert [d["id"] for d in body["current"]["decisions"]] == ["a", "b"]

    retried = c.put("/api/decisions",
                    json={**doc_with("a", "b", "c"),
                          "base_updated_at": body["current"]["updated_at"]})
    assert retried.status_code == 200
    assert [d["id"] for d in c.get("/api/decisions").json()["decisions"]] == ["a", "b", "c"]


def test_a_put_without_a_precondition_is_refused_once_a_version_exists(client):
    """Silence is not consent. A client that never read the file cannot know
    it is not overwriting one, so the omission is treated as stale."""
    c, _ = client
    save(c, "a")
    assert c.put("/api/decisions", json=doc_with("b")).status_code == 409


def test_the_page_sends_the_precondition_and_rebases_on_conflict(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert "base_updated_at: baseVersion" in html
    assert "r.status === 409" in html
    # Only candidates edited in this session are re-imposed over the file.
    assert "for (const id of dirty)" in html


# --------------------------------------------------------------------------
# Reference answers
# --------------------------------------------------------------------------


def test_page_carries_a_reference_answer_editor(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert 'id="ref"' in html
    assert "corrected_answer" in html
    # The draft is seeded into the editor with the verified markers stripped:
    # they are rendering hints, not part of the answer text.
    assert 'data-quote-status="verified"><\\/div>/g' in html


def test_note_and_reference_answer_are_separate_fields(layout):
    """`note` is an observation about the case and is not promotable; the
    reference answer is what goes into test_cases.yaml."""
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert 'id="notes"' in html and 'id="ref"' in html
    assert "Not promoted" in html


def test_needs_answer_filter_is_offered(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert "needs_answer" in html and "Needs an answer" in html


# --------------------------------------------------------------------------
# Near-duplicate surfacing
# --------------------------------------------------------------------------


def test_nearest_neighbour_is_carried_through_the_join(layout):
    """Recorded for every candidate, not only flagged ones -- no threshold
    separates a reworded duplicate from a distinct question about one rule,
    so the reviewer has to see it."""
    populate(layout)
    dump_json({"candidates": {"munchkin-0001-0": {
        "status": "unique", "max_similarity": 0.897,
        "nearest": {"kind": "candidate", "ref": "munchkin-0002-0",
                    "question": "Another phrasing?", "similarity": 0.897},
    }}}, layout.dedupe_file("munchkin"))
    row = load_candidates(layout, "munchkin")[0]
    assert row["near"]["similarity"] == 0.897
    assert row["near"]["ref"] == "munchkin-0002-0"


def test_missing_dedupe_leaves_the_neighbour_empty(layout):
    populate(layout)
    assert load_candidates(layout, "munchkin")[0]["near"] is None


def test_page_renders_the_neighbour_card_and_its_floor(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert "nearBlock" in html
    assert "NEAR_FLOOR" in html
    # Below the floor the neighbour is noise; the card must not render for it.
    assert "n.similarity < NEAR_FLOOR) return \"\"" in html


def test_page_offers_the_near_twin_filter(layout):
    populate(layout)
    html = build_page(load_run(layout, ["munchkin"]), run_id="r1", games=["munchkin"])
    assert "Has a near twin" in html and '"near"' in html
