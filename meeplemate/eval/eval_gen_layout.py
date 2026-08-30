"""The on-disk layout of an eval-generation run, in one place.

Every path under `data/eval_gen/<group-run-id>/` is constructed here and
nowhere else, mirroring `meeplemate.ingest.layout.PackageLayout`.

The organising rule, enforced by `tests/test_eval_gen_layout.py`:

    every step owns one or more top-level entries in the group, no entry is
    owned by two steps, and no entry nests inside another.

Records are written **once, in their final state**: one file per seed under
`mining/<game>/seeds/`, one per candidate under `answers/<game>/`. The earlier
design rebuilt and rewrote a single accumulating document after every seed,
which is O(n^2) in bytes written -- 22 GB to produce a 34 MB file on a full
Warhammer sweep. Per-record files also let two processes work the same run at
different `--offset`s without clobbering each other, and make `--resume` a
directory listing rather than a parse of the whole accumulated document.

Two steps are genuine whole-batch reduces and are exempt: `dedupe.json`
compares every candidate against every other, and `decisions.json` is edited by
a person. Both are small and rewritten atomically in full.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterator

#: Anything outside this is replaced in a filename component. Parent ids carry
#: '#' separators and rulebook names carry spaces and punctuation; neither is
#: safe to drop into a path unescaped.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def seed_key(parent_id: str) -> str:
    """Stable 8-hex identity for a seed's parent chunk.

    Filenames carry the seed ordinal as a sortable prefix, but identity comes
    from this. Ordinals are positions in the *filtered* corpus order, so
    changing `--min-seed-chars` shifts every one of them; keying resume on the
    ordinal would silently re-mine seeds already done.
    """
    return hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:8]


def safe_component(value: str) -> str:
    cleaned = _UNSAFE.sub("-", str(value)).strip("-")
    return cleaned or "unnamed"


@dataclass(frozen=True)
class EvalGenLayout:
    """Paths for one eval-generation group run."""

    root: Path
    group_run_id: str

    #: step name -> the top-level group entries that step is allowed to write.
    #: The machine-readable form of the one-owner rule.
    STEP_OUTPUTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "mine-questions": ("mining",),
        "answer-candidates": ("answers",),
        "review": ("decisions",),
        "promote": ("dataset",),
    }

    # --- group ---------------------------------------------------------------

    @property
    def group_root(self) -> Path:
        return Path(self.root) / self.group_run_id

    def step_roots(self, step: str) -> tuple[Path, ...]:
        return tuple(self.group_root / entry for entry in self.STEP_OUTPUTS[step])

    # --- mine-questions ------------------------------------------------------

    def mining_root(self, game_id: str) -> Path:
        return self.group_root / "mining" / safe_component(game_id)

    def seeds_dir(self, game_id: str) -> Path:
        return self.mining_root(game_id) / "seeds"

    def seed_file(self, game_id: str, seed_ordinal: int, parent_id: str) -> Path:
        """One seed's candidates. Ordinal sorts the directory, hash identifies it."""
        return self.seeds_dir(game_id) / f"{seed_ordinal:05d}-{seed_key(parent_id)}.yaml"

    def iter_seed_files(self, game_id: str) -> Iterator[Path]:
        d = self.seeds_dir(game_id)
        if not d.exists():
            return iter(())
        return iter(sorted(d.glob("*.yaml")))

    def dedupe_file(self, game_id: str) -> Path:
        """Whole-batch reduce: rewritten in full, by design."""
        return self.mining_root(game_id) / "dedupe.json"

    def mining_run_file(self, game_id: str) -> Path:
        return self.mining_root(game_id) / "run.json"

    # --- answer-candidates ---------------------------------------------------

    def answers_root(self, game_id: str) -> Path:
        return self.group_root / "answers" / safe_component(game_id)

    def answer_file(self, game_id: str, candidate_id: str) -> Path:
        return self.answers_root(game_id) / f"{safe_component(candidate_id)}.yaml"

    def iter_answer_files(self, game_id: str) -> Iterator[Path]:
        d = self.answers_root(game_id)
        if not d.exists():
            return iter(())
        return iter(sorted(d.glob("*.yaml")))

    def answers_run_file(self, game_id: str) -> Path:
        return self.answers_root(game_id) / "run.json"

    # --- review --------------------------------------------------------------

    def decisions_file(self) -> Path:
        """One file for the group: a review session spans every game in it, and
        candidate ids already carry their game."""
        return self.group_root / "decisions" / "decisions.json"

    # --- promote --------------------------------------------------------------

    def dataset_dir(self) -> Path:
        return self.group_root / "dataset"

    def dataset_file(self) -> Path:
        """The generated golden cases for this run.

        Under `data/` like every other step output, so it is regenerable and
        backed up rather than tracked. Adopting a run into the eval corpus is a
        separate, explicit act -- copying it to `meeplemate/eval/test_cases/`,
        which is where the loader looks. If the loader read from `data/`, a
        fresh clone would evaluate against a different corpus than the machine
        that generated it.
        """
        return self.dataset_dir() / "test_cases.yaml"

    # --- discovery -----------------------------------------------------------

    def mined_games(self) -> list[str]:
        root = self.group_root / "mining"
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())

    def answered_games(self) -> list[str]:
        root = self.group_root / "answers"
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())


def group_run_ids(root: Path) -> list[str]:
    """Existing group runs, newest-looking last (ids are dates)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def next_group_run_id(root: Path) -> str:
    """Today's date, auto-incremented on collision.

    Mirrors `next_group_run_id` in `meeplemate.eval` so ids read the same way
    across the eval commands.
    """
    from datetime import datetime

    base = datetime.now().strftime("%Y-%m-%d")
    existing = set(group_run_ids(root))
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"
