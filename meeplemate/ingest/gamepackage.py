from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, NotRequired, Sequence, TypedDict

import yaml

from meeplemate.util import amap, aslurp, aslurp_json
from langchain_core.documents.base import Document

class RulebookDescriptor(TypedDict):
    name: str
    path: str
    url: str
    document_key: str
    strategy: NotRequired[str]
    page_count: int


class Manifest(TypedDict):
    name: str
    game_id: str
    rulebooks: Sequence[RulebookDescriptor]


class GamePackage(Manifest, TypedDict):
    path: Path


@dataclass
class Page:
    gp: GamePackage
    page_num: int
    document_key: str


def document_keys(manifest: Manifest) -> Sequence[str]:
    document_keys: list[str] = []
    for rulebook in manifest["rulebooks"]:
        document_keys.append(rulebook["document_key"])
    return document_keys


def load_manifest(target_dir: Path) -> Manifest:
    manifest_path = target_dir / "rulebooks.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    return manifest

 
def load_game_package(target_dir: Path) -> GamePackage:
    manifest = load_manifest(target_dir)
    if "path" in manifest:
        del manifest["path"]
    gp: GamePackage = GamePackage(**manifest, path=target_dir)
    return gp


def get_page(gp:GamePackage, document_key: str, page_num: int) -> Page:
    page_base = (gp["path"] / document_key / f"{page_num:04d}")
    page_num = int(page_base.stem)
    page = Page(
        gp=gp,
        document_key=document_key,
        page_num=page_num,
    )
    return page



def page_md_path(page: Page) -> Path:
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    markdown_path = page_base.with_suffix(".md")
    return markdown_path


def page_structured_path(page: Page) -> Path:
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    structured_path = page_base.with_suffix(".structured.json")
    return structured_path


def page_structured_fixed_path(page: Page) -> Path:
    page_base = (page.gp["path"] / page.document_key / f"{page.page_num:04d}")
    structured_path = page_base.with_suffix(".structured.fixed.json")
    return structured_path


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


def get_page_metadata(page: Page) -> dict:
    rulebook = next(
        rb for rb in page.gp["rulebooks"] if rb["document_key"] == page.document_key
    )
    game_id = page.gp["game_id"]
    metadata = {
        "game_name": page.gp["name"],
        "game_id": game_id,
        "rulebook_name": rulebook["name"],
        "document_key": page.document_key,
        "page_num": page.page_num,
    }
    return metadata


def get_page_id(game_id: str, document_key: str, page_num: int) -> str:
    return f"{game_id}#{document_key}#{page_num}"


async def page_to_document(page: Page) -> Document:
    game_id = page.gp["game_id"]
    metadata = get_page_metadata(page)
    page_key = get_page_id(game_id, page.document_key, page.page_num)
    markdown = await page_md(page)
    return Document(id=page_key, page_content=markdown, metadata=metadata)


def get_document_page_aiter(gp: GamePackage) -> AsyncIterator[Document]:
    return amap(page_to_document, get_pages_iter(gp))
