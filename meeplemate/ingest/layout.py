"""The on-disk layout of a game package, in one place.

Every path inside `data/ingested/<game>/` is constructed here and nowhere else.

The organising rule, enforced by `tests/ingest/test_layout.py`:

    every step owns one or more top-level entries in the package, no entry is
    owned by two steps, and no entry nests inside another.

Directories are grouped by *step* first and rulebook second — `images/<dk>/`
rather than `<dk>/images/` — so a step's entire output is one path that can be
named without knowing the document keys, which are not assigned until
`init-game-package` has run.

This module deliberately imports nothing from `gamepackage`, so the type surface
that non-ingest modules depend on stays independent of the layout.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class PackageLayout:
    root: Path

    #: step name -> the top-level package entries that step is allowed to write.
    #: This is the machine-readable form of the one-owner rule.
    STEP_OUTPUTS: ClassVar[dict[str, tuple[str, ...]]] = {
        "init-game-package": ("manifest.yaml", "raw_documents"),
        "update-version": ("version.txt",),
        "render": ("images",),
        "ocr": ("ocr",),
        "build-text": ("text",),
        "page-number-ocr": ("page_numbers_raw",),
        "page-number-fixup": ("page_numbers",),
        "add-metadata": ("metadata",),
        "build-chunks": ("chunks",),
        "generate-reference": ("reference",),
        "generate-setting-summary": ("setting",),
        "generate-presentation": ("presentation.yaml",),
        "generate-example-questions": ("example_questions.yaml",),
    }

    def step_roots(self, step: str) -> tuple[Path, ...]:
        return tuple(self.root / entry for entry in self.STEP_OUTPUTS[step])

    # --- init-game-package ----------------------------------------------------

    def manifest(self) -> Path:
        return self.root / "manifest.yaml"

    def raw_documents(self) -> Path:
        return self.root / "raw_documents"

    def raw_document(self, relative_path: str) -> Path:
        return self.raw_documents() / relative_path

    # --- update-version -------------------------------------------------------

    def version(self) -> Path:
        return self.root / "version.txt"

    # --- render ---------------------------------------------------------------

    def page_image(self, document_key: str, page_num: int) -> Path:
        return self.root / "images" / document_key / f"{page_num:04d}.png"

    # --- ocr ------------------------------------------------------------------

    def page_raw_md(self, document_key: str, page_num: int) -> Path:
        return self.root / "ocr" / document_key / f"{page_num:04d}.raw.md"

    def page_structured(self, document_key: str, page_num: int) -> Path:
        return self.root / "ocr" / document_key / f"{page_num:04d}.structured.json"

    # --- build-text -----------------------------------------------------------

    def page_md(self, document_key: str, page_num: int) -> Path:
        return self.root / "text" / document_key / f"{page_num:04d}.md"

    def page_structured_fixed(self, document_key: str, page_num: int) -> Path:
        return self.root / "text" / document_key / f"{page_num:04d}.structured.fixed.json"

    def document_md(self, document_key: str) -> Path:
        return self.root / "text" / document_key / "document.md"

    # --- page numbers ---------------------------------------------------------

    def page_number_raw(self, document_key: str, page_num: int) -> Path:
        return self.root / "page_numbers_raw" / document_key / f"{page_num:04d}.txt"

    def page_number(self, document_key: str, page_num: int) -> Path:
        return self.root / "page_numbers" / document_key / f"{page_num:04d}.txt"

    # --- add-metadata ---------------------------------------------------------

    def page_metadata(self, document_key: str, page_num: int) -> Path:
        return self.root / "metadata" / document_key / f"{page_num:04d}.yaml"

    # --- build-chunks ---------------------------------------------------------
    # Parent and child chunks get their own directories. They used to share one,
    # distinguished by a glob plus a regex that had to reject the other family.

    def chunks_dir(self, document_key: str) -> Path:
        return self.root / "chunks" / document_key

    def parent_chunks_dir(self, document_key: str) -> Path:
        return self.chunks_dir(document_key) / "parent"

    def child_chunks_dir(self, document_key: str) -> Path:
        return self.chunks_dir(document_key) / "child"

    def parent_chunk(self, document_key: str, page_num: int, chunk_idx: int) -> Path:
        return self.parent_chunks_dir(document_key) / f"{page_num:04d}_{chunk_idx:04d}.json"

    def child_chunk(
        self, document_key: str, page_num: int, chunk_idx: int, child_idx: int
    ) -> Path:
        return (
            self.child_chunks_dir(document_key)
            / f"{page_num:04d}_{chunk_idx:04d}_{child_idx:04d}.json"
        )

    # --- summaries ------------------------------------------------------------

    def game_reference(self) -> Path:
        return self.root / "reference" / "game.md"

    def rulebook_reference(self, document_key: str) -> Path:
        return self.root / "reference" / f"{document_key}.md"

    def rulebook_reference_short(self, document_key: str) -> Path:
        return self.root / "reference" / f"{document_key}.short.md"

    def game_setting(self) -> Path:
        return self.root / "setting" / "game.md"

    def rulebook_setting(self, document_key: str) -> Path:
        return self.root / "setting" / f"{document_key}.md"

    # --- presentation assets --------------------------------------------------

    def presentation(self) -> Path:
        return self.root / "presentation.yaml"

    def example_questions(self) -> Path:
        return self.root / "example_questions.yaml"
