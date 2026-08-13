from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal, NotRequired, Sequence, TypedDict

import yaml

from meeplemate.ingest.errors import MissingStepInput
from meeplemate.ingest.layout import PackageLayout
from meeplemate.util import amap, aslurp, aslurp_json, spit_yaml
from langchain_core.documents.base import Document

class RulebookDescriptor(TypedDict):
    name: str
    summary: NotRequired[str]
    path: str
    url: str
    document_key: str
    strategy: NotRequired[str]
    page_count: int
    page_one_offset: NotRequired[int | Literal["auto"]]


class Manifest(TypedDict):
    name: str
    game_id: str
    game_version: NotRequired[str]
    rulebooks: Sequence[RulebookDescriptor]
    summary: NotRequired[str]
    emoji: NotRequired[str]
    background_color: NotRequired[str]


class GamePackage(Manifest, TypedDict):
    path: Path


@dataclass
class Page:
    gp: GamePackage
    page_num: int
    document_key: str


def get_game_key_for_id_version(game_id: str, game_version: str) -> str:
    return f"{game_id}#{game_version}"


def get_game_key(gp: GamePackage) -> str:
    return get_game_key_for_id_version(gp["game_id"], gp.get("game_version", ""))


def document_keys(manifest: Manifest) -> Sequence[str]:
    document_keys: list[str] = []
    for rulebook in manifest["rulebooks"]:
        document_keys.append(rulebook["document_key"])
    return document_keys


def layout_for(gp: GamePackage) -> PackageLayout:
    return PackageLayout(gp["path"])


def load_manifest(target_dir: Path) -> Manifest:
    manifest_path = PackageLayout(target_dir).manifest()
    manifest = yaml.safe_load(manifest_path.read_text())
    return manifest

 
def _read_if_exists(path: Path) -> str | None:
    return path.read_text() if path.exists() else None


def load_game_package(target_dir: Path, *, require_version: bool = False) -> GamePackage:
    """Assemble the in-memory game package from its on-disk parts.

    The manifest holds only facts derived from the source rulebooks. The version
    and the generated summaries live in their own files, written by the steps
    that own them, and are folded back in here.

    Everything downstream — `import_game_data`, and through it `chatloop`,
    `search` and `qa_graph` — sees exactly the shape it saw when all of this was
    one YAML file.
    """
    manifest = load_manifest(target_dir)
    if "path" in manifest:
        del manifest["path"]
    gp: GamePackage = GamePackage(**manifest, path=target_dir)
    layout = PackageLayout(target_dir)

    version = _read_if_exists(layout.version())
    if version is None and require_version:
        raise MissingStepInput(
            what=f"No game version at {layout.version()}",
            run_step=f"update-version {target_dir}",
        )
    if version is not None:
        gp["game_version"] = version.strip()

    game_summary = _read_if_exists(layout.game_reference())
    if game_summary is not None:
        gp["summary"] = game_summary

    for rulebook in gp["rulebooks"]:
        short = _read_if_exists(layout.rulebook_reference_short(rulebook["document_key"]))
        if short is not None:
            rulebook["summary"] = short

    return gp


def write_manifest(manifest: Manifest, target_dir: Path) -> None:
    """Write the package manifest. `init-game-package` is its only caller.

    Keeping one writer is what lets the manifest be a stage output. The version
    and summaries that used to be merged in here are written by their own steps
    to their own files.
    """
    manifest_dict = dict(manifest)
    manifest_dict.pop("path", None)
    spit_yaml(manifest_dict, PackageLayout(target_dir).manifest())


def get_page(gp:GamePackage, document_key: str, page_num: int) -> Page:
    page = Page(
        gp=gp,
        document_key=document_key,
        page_num=page_num,
    )
    return page


def get_raw_documents_directory_path(gp: GamePackage) -> Path:
    return layout_for(gp).raw_documents()


def get_game_setting_summary_path(gp: GamePackage) -> Path:
    return layout_for(gp).game_setting()


def get_game_presentation_path(gp: GamePackage) -> Path:
    return layout_for(gp).presentation()


def get_game_example_questions_path(gp: GamePackage) -> Path:
    return layout_for(gp).example_questions()


def page_md_path(page: Page) -> Path:
    return layout_for(page.gp).page_md(page.document_key, page.page_num)


def page_raw_md_path(page: Page) -> Path:
    return layout_for(page.gp).page_raw_md(page.document_key, page.page_num)


def page_structured_path(page: Page) -> Path:
    return layout_for(page.gp).page_structured(page.document_key, page.page_num)


def page_structured_fixed_path(page: Page) -> Path:
    return layout_for(page.gp).page_structured_fixed(page.document_key, page.page_num)


def page_number_raw_path(page: Page) -> Path:
    return layout_for(page.gp).page_number_raw(page.document_key, page.page_num)


def page_number_path(page: Page) -> Path:
    return layout_for(page.gp).page_number(page.document_key, page.page_num)


async def page_md(page: Page) -> str:
    markdown_path = page_md_path(page)
    markdown = await aslurp(markdown_path)
    return markdown


async def page_structured(page: Page) -> Sequence[dict]:
    structured_path = page_structured_path(page)
    structured = await aslurp_json(structured_path)
    return structured


async def page_structured_fixed(page: Page) -> Sequence[dict]:
    structured_path = page_structured_fixed_path(page)
    structured = await aslurp_json(structured_path)
    return structured


async def get_pages_iter(gp: GamePackage, document_key: str|None = None) -> AsyncIterator[Page]:
    rulebook_descriptors = gp["rulebooks"]
    for descriptor in rulebook_descriptors:
        if document_key is not None and descriptor["document_key"] != document_key:
            continue

        for page_num in range(descriptor["page_count"]):
            page = get_page(gp, descriptor["document_key"], page_num)
            yield page


def get_page_metadata_path(page: Page) -> Path:
    return layout_for(page.gp).page_metadata(page.document_key, page.page_num)


def load_page_metadata(page: Page) -> dict:
    metadata_path = get_page_metadata_path(page)
    metadata = yaml.safe_load(metadata_path.read_text())
    return metadata


def get_rulebook(gp: GamePackage, document_key: str) -> RulebookDescriptor:
    return next(rb for rb in gp["rulebooks"] if rb["document_key"] == document_key)


def get_page_one_offset(rulebook: RulebookDescriptor) -> int | Literal["auto"]:
    return rulebook.get("page_one_offset", "auto")


def page_num_from_offset(page_num: int, page_one_offset: int) -> str:
    return str(page_num - page_one_offset + 1)


def get_page_metadata(page: Page) -> dict:
    """Per-page metadata baked into chunk JSON.

    Deliberately excludes `game_version`. It used to be written here, which made
    every chunk file depend on `update-version` and forced a full rebuild after
    each version bump. `add_game_metadata_to_document` in dataimport already
    stamps the current version onto metadata, `doc_id` and the document id at
    import time, so recording it here bought nothing.
    """
    rulebook = get_rulebook(page.gp, page.document_key)
    game_id = page.gp["game_id"]
    metadata = {
        "game_name": page.gp["name"],
        "game_id": game_id,
        "rulebook_name": rulebook["name"],
        "document_key": page.document_key,
        "page_ordinal": page.page_num,
    }
    return metadata


def get_page_id(game_id: str, game_version: str, document_key: str, page_ordinal: int) -> str:
    game_key = get_game_key_for_id_version(game_id, game_version)
    return f"{game_key}#{document_key}#{page_ordinal}"


def get_page_chunk_id(game_id: str, game_version: str, document_key: str, page_ordinal: int, chunk_idx: int) -> str:
    return f"{get_page_id(game_id, game_version, document_key, page_ordinal)}#{chunk_idx}"


async def page_to_document(page: Page) -> Document:
    game_id = page.gp["game_id"]
    metadata = get_page_metadata(page)
    addl_metadata = load_page_metadata(page)
    metadata.update(addl_metadata)
    # Built with an empty version slot; import fills it in. Keeps chunk JSON
    # identical across version bumps. See get_page_metadata.
    page_key = get_page_id(game_id, "", page.document_key, page.page_num)
    markdown = await page_md(page)
    return Document(id=page_key, page_content=markdown, metadata=metadata)


def get_document_page_aiter(gp: GamePackage, document_key: str|None = None) -> AsyncIterator[Document]:
    return amap(page_to_document, get_pages_iter(gp, document_key=document_key))