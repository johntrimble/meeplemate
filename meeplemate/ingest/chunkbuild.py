import asyncio
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_text_splitters import TextSplitter

from meeplemate.ingest.gamepackage import GamePackage, Page, get_pages_iter, load_game_package, page_md, page_to_document, get_page_chunk_id
from meeplemate.text_splitters import FixedRecursiveCharacterTextSplitter
from meeplemate.util import amap, aspit


@dataclass
class ChunkDescriptor(Page):
    chunk_idx: int


@dataclass
class ChildChunkDescriptor(ChunkDescriptor):
    child_idx: int


def get_chunks_directory_path(gp: GamePackage, document_key: str) -> Path:
    return gp["path"] / document_key / "chunks"


def get_chunk_path_for_index(page: Page, chunk_idx: int) -> Path:
    return get_chunks_directory_path(page.gp, page.document_key) / f"{page.page_num:04d}_chunk_{chunk_idx:04d}.json"


def get_chunk_path(chunk: ChunkDescriptor) -> Path:
    return get_chunks_directory_path(chunk.gp, chunk.document_key) / f"{chunk.page_num:04d}_chunk_{chunk.chunk_idx:04d}.json"


def get_child_chunk_path_for_index(page: Page, chunk_idx: int, child_idx: int) -> Path:
    return get_chunks_directory_path(page.gp, page.document_key) / f"{page.page_num:04d}_chunk_{chunk_idx:04d}_child_{child_idx:04d}.json"


def get_child_chunk_path(chunk: ChildChunkDescriptor) -> Path:
    return get_chunks_directory_path(chunk.gp, chunk.document_key) / f"{chunk.page_num:04d}_chunk_{chunk.chunk_idx:04d}_child_{chunk.child_idx:04d}.json"


def get_child_chunk_id(game_id: str, game_version:str, document_key: str, page_num: int, chunk_idx: int, child_idx: int) -> str:
    chunk_key = get_page_chunk_id(game_id, game_version, document_key, page_num, chunk_idx)
    return f"{chunk_key}#{child_idx}"


def chunks_for_page_iter(page: Page) -> Iterator[ChunkDescriptor]:
    chunks_path = get_chunks_directory_path(page.gp, page.document_key)
    # Use glob to find all chunk files for this page
    chunk_pattern = re.compile(rf"^{page.page_num:04d}_chunk_(\d+)\.json$")
    for path in chunks_path.glob(f"{page.page_num:04d}_chunk_*.json"):
        match = chunk_pattern.match(path.name)
        if match:
            chunk_idx = int(match.group(1))
            yield ChunkDescriptor(
                gp=page.gp,
                document_key=page.document_key,
                page_num=page.page_num,
                chunk_idx=chunk_idx,
            )


def child_chunks_for_chunk_iter(chunk: ChunkDescriptor) -> Iterator[ChildChunkDescriptor]:
    chunks_path = get_chunks_directory_path(chunk.gp, chunk.document_key)
    # Use glob to find all child chunk files for this chunk
    child_chunk_pattern = re.compile(rf"^{chunk.page_num:04d}_chunk_{chunk.chunk_idx:04d}_child_(\d+)\.json$")
    for path in chunks_path.glob(f"{chunk.page_num:04d}_chunk_{chunk.chunk_idx:04d}_child_*.json"):
        match = child_chunk_pattern.match(path.name)
        if match:
            child_idx = int(match.group(1))
            yield ChildChunkDescriptor(
                gp=chunk.gp,
                document_key=chunk.document_key,
                page_num=chunk.page_num,
                chunk_idx=chunk.chunk_idx,
                child_idx=child_idx,
            )


def split_document(parent_splitter: TextSplitter, child_splitter: TextSplitter, document: Document) -> Sequence[Tuple[Document, Sequence[Document]]]:
    split_docs: list[Tuple[Document, list[Document]]] = []
    parent_chunks = parent_splitter.split_documents([document])
    for parent_idx, parent_chunk in enumerate(parent_chunks):
        parent_chunk.metadata = {**document.metadata, **parent_chunk.metadata}
        parent_chunk.metadata["chunk_index"] = parent_idx
        parent_chunk.metadata["chunk_count"] = len(parent_chunks)
        parent_chunk.id = f"{document.id}#{parent_idx}"
        child_chunks = child_splitter.split_documents([parent_chunk])
        for child_idx, child_chunk in enumerate(child_chunks):
            child_chunk.id = f"{parent_chunk.id}#{child_idx}"
            child_chunk.metadata = {**parent_chunk.metadata, **child_chunk.metadata}
            child_chunk.metadata["doc_id"] = parent_chunk.id
            child_chunk.metadata["child_chunk_index"] = child_idx
            child_chunk.metadata["child_chunk_count"] = len(child_chunks)
        split_docs.append((parent_chunk, child_chunks))
    return split_docs


async def write_parent_and_child_chunks(
    parent_splitter: TextSplitter,
    child_splitter: TextSplitter,
    page: Page,
):
    tasks = []
    document = await page_to_document(page)
    documents_split = split_document(
        parent_splitter,
        child_splitter,
        document,
    )
    for parent_idx, (parent_doc, child_docs) in enumerate(documents_split):
        parent_chunk_path = get_chunk_path_for_index(page, parent_idx)
        tasks.append(aspit(dumps(parent_doc), parent_chunk_path))
        for child_idx, child_doc in enumerate(child_docs):
            child_chunk_path = get_child_chunk_path_for_index(page, parent_idx, child_idx)
            tasks.append(aspit(dumps(child_doc), child_chunk_path))
    await asyncio.gather(*tasks)
 

@dataclass
class BuildChunksJob:
    path: Path
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    tokenizer: Any

    @property
    def gp(self) -> GamePackage:
        if not getattr(self, "_gp", None):
            self._gp = load_game_package(self.path)
        return self._gp
    
    async def run(self) -> None:
        parent_splitter = FixedRecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            add_start_index=True,
        )

        child_splitter = FixedRecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            add_start_index=True,
        )

        tasks = []
        for rulebook in self.gp["rulebooks"]:
            chunks_path = get_chunks_directory_path(self.gp, rulebook["document_key"])
            chunks_path.mkdir(exist_ok=True)
            async for page in get_pages_iter(self.gp, rulebook["document_key"]):
                tasks.append(
                    write_parent_and_child_chunks(
                        parent_splitter,
                        child_splitter,
                        page,
                    )
                )
        await asyncio.gather(*tasks)
