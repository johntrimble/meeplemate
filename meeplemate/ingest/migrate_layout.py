"""Move an existing game package from the flat layout to the per-step layout.

The old layout put every artifact for a rulebook in one directory keyed by
document_key. The new one groups by producing step first
(`images/<dk>/`, `ocr/<dk>/`, ...) so each step owns one top-level entry.

Everything maps deterministically, so this is a rename pass — no re-ingest, no
GPU, and the 2.4 GB of page images never move bytes because `Path.rename` on the
same filesystem is a metadata operation.

Run with no flags for a dry run; `--apply` performs the moves.
"""
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from structlog import get_logger

from meeplemate.ingest.layout import PackageLayout
from meeplemate.util import spit_yaml

logger = get_logger(__name__)

# Per-page suffixes, longest first. Order matters: ".structured.fixed.json"
# must be tried before ".structured.json", and ".raw.md" before ".md", or the
# shorter rule silently swallows the longer one's files.
PAGE_SUFFIX_RULES: list[tuple[str, str, str]] = [
    # (old suffix, new step directory, new suffix)
    (".structured.fixed.json", "text", ".structured.fixed.json"),
    (".structured.json", "ocr", ".structured.json"),
    (".page_number.raw.txt", "page_numbers_raw", ".txt"),
    (".page_number.txt", "page_numbers", ".txt"),
    (".metadata.yaml", "metadata", ".yaml"),
    (".raw.md", "ocr", ".raw.md"),
    (".png", "images", ".png"),
    (".md", "text", ".md"),
]

# Whole-file names inside a document directory.
DOCUMENT_FILE_RULES: dict[str, tuple[str, str]] = {
    "document.md": ("text", "{dk}/document.md"),
    "summary.md": ("reference", "{dk}.md"),
    "summary.short.md": ("reference", "{dk}.short.md"),
    "setting_summary.md": ("setting", "{dk}.md"),
}

# Package-level files that move.
ROOT_FILE_RULES: dict[str, str] = {
    "summary.md": "reference/game.md",
    "game_setting_summary.md": "setting/game.md",
}

# Package-level files that stay exactly where they are.
ROOT_FILES_UNCHANGED = {"presentation.yaml", "example_questions.yaml", "manifest.yaml", "version.txt"}

# Directories the new layout owns; present when resuming a partial migration.
NEW_STEP_DIRECTORIES = {
    entry
    for entries in PackageLayout.STEP_OUTPUTS.values()
    for entry in entries
    if "." not in entry
}


@dataclass
class Move:
    src: Path
    dst: Path


@dataclass
class MigrationPlan:
    root: Path
    moves: list[Move] = field(default_factory=list)
    unclassified: list[Path] = field(default_factory=list)
    manifest_version: str | None = None
    already_migrated: bool = False

    @property
    def ok(self) -> bool:
        return not self.unclassified


def _classify_page_file(name: str) -> tuple[str, str] | None:
    """Map `0007.page_number.txt` to ('page_numbers', '0007.txt')."""
    for old_suffix, step_dir, new_suffix in PAGE_SUFFIX_RULES:
        if not name.endswith(old_suffix):
            continue
        stem = name[: -len(old_suffix)]
        if not (stem.isdigit() and len(stem) == 4):
            continue
        return step_dir, f"{stem}{new_suffix}"
    return None


def _classify_chunk_file(name: str) -> tuple[str, str] | None:
    """`0007_chunk_0002_child_0003.json` -> ('child', '0007_0002_0003.json')."""
    if not name.endswith(".json"):
        return None
    stem = name[: -len(".json")]
    parts = stem.split("_")
    # child first: a parent rule would also match its prefix.
    if len(parts) == 5 and parts[1] == "chunk" and parts[3] == "child":
        return "child", f"{parts[0]}_{parts[2]}_{parts[4]}.json"
    if len(parts) == 3 and parts[1] == "chunk":
        return "parent", f"{parts[0]}_{parts[2]}.json"
    return None


def plan_migration(root: Path) -> MigrationPlan:
    plan = MigrationPlan(root=root)
    layout = PackageLayout(root)

    if layout.manifest().exists() and (root / "images").exists():
        plan.already_migrated = True
        return plan

    old_manifest = root / "rulebooks.yaml"
    manifest_path = old_manifest if old_manifest.exists() else layout.manifest()
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest in {root}")

    manifest = yaml.safe_load(manifest_path.read_text())
    plan.manifest_version = manifest.get("game_version")
    document_keys = [rb["document_key"] for rb in manifest["rulebooks"]]

    for document_key in document_keys:
        document_dir = root / document_key
        if not document_dir.is_dir():
            continue
        for path in sorted(document_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(document_dir)

            if relative.parts[0] == "chunks":
                classified = _classify_chunk_file(path.name)
                if classified is None:
                    plan.unclassified.append(path)
                    continue
                family, new_name = classified
                plan.moves.append(
                    Move(path, root / "chunks" / document_key / family / new_name)
                )
                continue

            if len(relative.parts) != 1:
                plan.unclassified.append(path)
                continue

            if path.name in DOCUMENT_FILE_RULES:
                step_dir, template = DOCUMENT_FILE_RULES[path.name]
                plan.moves.append(
                    Move(path, root / step_dir / template.format(dk=document_key))
                )
                continue

            classified = _classify_page_file(path.name)
            if classified is None:
                plan.unclassified.append(path)
                continue
            step_dir, new_name = classified
            plan.moves.append(Move(path, root / step_dir / document_key / new_name))

    # Package-level files.
    for path in sorted(root.iterdir()):
        if path.is_dir():
            if path.name == "raw_documents" or path.name in document_keys:
                continue
            if path.name in NEW_STEP_DIRECTORIES:
                continue
            plan.unclassified.append(path)
            continue
        if path.name == "rulebooks.yaml" or path.name in ROOT_FILES_UNCHANGED:
            continue
        if path.name in ROOT_FILE_RULES:
            plan.moves.append(Move(path, root / ROOT_FILE_RULES[path.name]))
            continue
        plan.unclassified.append(path)

    return plan


def rewrite_manifest(root: Path, *, apply: bool) -> str | None:
    """Split rulebooks.yaml into manifest.yaml plus version.txt.

    Drops the fields that now live in their own files: `game_version` becomes
    version.txt, and the summaries are already on disk under reference/.
    """
    layout = PackageLayout(root)
    old_manifest = root / "rulebooks.yaml"
    if not old_manifest.exists():
        return None

    manifest = yaml.safe_load(old_manifest.read_text())
    version = manifest.pop("game_version", None)
    manifest.pop("summary", None)
    manifest.pop("path", None)
    for rulebook in manifest.get("rulebooks", []):
        rulebook.pop("summary", None)

    if apply:
        spit_yaml(manifest, layout.manifest())
        if version:
            layout.version().write_text(str(version))
        old_manifest.unlink()
    return version


def backfill_raw_documents(root: Path, source_root: Path, *, apply: bool) -> list[Path]:
    """Copy source PDFs in for packages that never got a raw_documents/ copy.

    Three packages predate that step. `render` reads from raw_documents/, so
    without this they cannot be re-rendered.
    """
    layout = PackageLayout(root)
    manifest_path = layout.manifest() if layout.manifest().exists() else root / "rulebooks.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    copied: list[Path] = []
    for rulebook in manifest["rulebooks"]:
        target = layout.raw_document(rulebook["path"])
        if target.exists():
            continue
        source = source_root / root.name / rulebook["path"]
        if not source.exists():
            logger.warning("No source PDF to backfill", expected=str(source))
            continue
        copied.append(target)
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return copied


def apply_plan(plan: MigrationPlan) -> int:
    moved = 0
    for move in plan.moves:
        if not move.src.exists():
            # Resumable: a previous run already moved this one.
            continue
        move.dst.parent.mkdir(parents=True, exist_ok=True)
        move.src.rename(move.dst)
        moved += 1
    # Drop the now-empty per-document directories.
    for path in sorted(plan.root.iterdir(), reverse=True):
        if path.is_dir():
            for sub in sorted(path.rglob("*"), reverse=True):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            if not any(path.iterdir()):
                path.rmdir()
    return moved


def verify_package(root: Path) -> list[str]:
    """Assert every artifact the manifest implies resolves under the new layout."""
    from meeplemate.ingest.gamepackage import load_game_package

    problems: list[str] = []
    gp = load_game_package(root)
    layout = PackageLayout(root)

    for rulebook in gp["rulebooks"]:
        document_key = rulebook["document_key"]
        if not layout.document_md(document_key).exists():
            problems.append(f"missing {layout.document_md(document_key)}")
        for page_num in range(rulebook["page_count"]):
            for accessor in ("page_md", "page_metadata", "page_image", "page_number"):
                path = getattr(layout, accessor)(document_key, page_num)
                if not path.exists():
                    problems.append(f"missing {path}")
    return problems
