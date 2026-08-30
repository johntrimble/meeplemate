"""Reading and writing eval-generation records.

Every write here is a whole file in its final state, placed atomically. Nothing
appends, and nothing rewrites a record that already exists, so two processes
working the same run at different `--offset`s cannot corrupt each other's work
and a kill mid-run costs at most the in-flight records.

The two reduces (`dedupe.json`, `decisions.json`) are the deliberate
exceptions -- see `eval_gen_layout`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import structlog
import yaml

from meeplemate.eval.eval_gen_layout import EvalGenLayout
from meeplemate.util import atomic_write_text

logger = structlog.get_logger(__name__)


class _RecordDumper(yaml.SafeDumper):
    """Local dumper subclass.

    Registering a representer on `yaml.SafeDumper` itself would change every
    `spit_yaml` call in the process, including the game manifests ingest
    writes.
    """


def _literal_block_for_multiline(dumper: yaml.Dumper, data: str):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_RecordDumper.add_representer(str, _literal_block_for_multiline)


def dump_yaml(doc: Mapping[str, Any], path: Path) -> None:
    """Atomically write one record, preserving key order.

    `sort_keys=False` because key order is meaningful in these records --
    `question` before `context`, `recommendation` before `response` -- so a
    reviewer opening one is not scrolling past bookkeeping to reach the point.
    """
    atomic_write_text(
        path,
        lambda fp: yaml.dump(
            dict(doc), fp, Dumper=_RecordDumper, sort_keys=False,
            default_flow_style=False, allow_unicode=True, width=100,
        ),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def dump_json(doc: Mapping[str, Any], path: Path) -> None:
    atomic_write_text(
        path, lambda fp: json.dump(dict(doc), fp, indent=2, ensure_ascii=False)
    )


def load_json(path: Path, default: Any = None) -> Any:
    """Read a reduce file, tolerating absence and corruption.

    A malformed file returns the default rather than raising. These are
    regenerable (`--dedupe-only`) or hand-editable, and a tool that refuses to
    start is worse than one that starts empty and says so.
    """
    if not Path(path).exists():
        return default
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("Unreadable file, using default", path=str(path), error=repr(exc))
        return default


# ---------------------------------------------------------------------------
# Mining records
# ---------------------------------------------------------------------------


def write_seed_record(
    layout: EvalGenLayout, game_id: str, record: Mapping[str, Any]
) -> Path:
    """One seed's outcome, success or failure.

    Failures get a file too, holding the error. `--resume` then skips them the
    way it skips successes, and which seed failed and why is visible without
    opening a run summary -- `--retry-failed` becomes "delete these and resume".
    """
    path = layout.seed_file(game_id, record["seed_ordinal"], record["seed"]["parent_id"])
    dump_yaml(record, path)
    return path


def read_seed_records(layout: EvalGenLayout, game_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in layout.iter_seed_files(game_id):
        try:
            rec = load_yaml(path)
        except yaml.YAMLError as exc:
            logger.warning("Skipping unreadable seed record", path=str(path), error=repr(exc))
            continue
        if rec:
            out.append(rec)
    return out


def mined_parent_ids(layout: EvalGenLayout, game_id: str) -> set[str]:
    """Parent ids already recorded, for `--resume`.

    Read from inside each record rather than parsed out of the filename: the
    filename's ordinal prefix moves when `--min-seed-chars` changes, the
    parent id does not.
    """
    ids: set[str] = set()
    for rec in read_seed_records(layout, game_id):
        pid = (rec.get("seed") or {}).get("parent_id")
        if pid:
            ids.add(pid)
    return ids


def iter_candidates(
    layout: EvalGenLayout, game_id: str
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """(candidate, its seed record) for every candidate mined for a game."""
    for rec in read_seed_records(layout, game_id):
        for cand in rec.get("candidates") or ():
            yield cand, rec


# ---------------------------------------------------------------------------
# Answer records
# ---------------------------------------------------------------------------


def write_answer_record(
    layout: EvalGenLayout, game_id: str, candidate_id: str, record: Mapping[str, Any]
) -> Path:
    path = layout.answer_file(game_id, candidate_id)
    dump_yaml(record, path)
    return path


def read_answers(layout: EvalGenLayout, game_id: str) -> dict[str, dict[str, Any]]:
    """Answer records keyed by candidate id.

    `run.json` shares the directory and is skipped by extension -- it is JSON,
    the records are YAML.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in layout.iter_answer_files(game_id):
        try:
            rec = load_yaml(path)
        except yaml.YAMLError as exc:
            logger.warning("Skipping unreadable answer", path=str(path), error=repr(exc))
            continue
        cid = (rec or {}).get("candidate_id")
        if cid:
            out[cid] = rec
    return out


def answered_candidate_ids(layout: EvalGenLayout, game_id: str) -> set[str]:
    """Candidate ids with a non-error answer, for `--resume`.

    An errored answer is retried: a timeout must not permanently exclude a
    candidate the way a considered verdict does.
    """
    return {
        cid for cid, rec in read_answers(layout, game_id).items()
        if (rec.get("answer") or {}).get("status") != "error"
    }
