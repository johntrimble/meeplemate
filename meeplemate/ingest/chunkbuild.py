import asyncio
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_text_splitters import TextSplitter

from meeplemate.ingest.gamepackage import GamePackage, Page, layout_for, get_pages_iter, load_game_package, load_page_metadata, page_md, page_to_document, get_page_chunk_id
from meeplemate.text_splitters import FixedRecursiveCharacterTextSplitter
from meeplemate.util import amap, aslurp, aspit


@dataclass
class ChunkDescriptor(Page):
    chunk_idx: int


@dataclass
class ChildChunkDescriptor(ChunkDescriptor):
    child_idx: int


def get_chunk_path_for_index(page: Page, chunk_idx: int) -> Path:
    return layout_for(page.gp).parent_chunk(page.document_key, page.page_num, chunk_idx)


def get_chunk_path(chunk: ChunkDescriptor) -> Path:
    return layout_for(chunk.gp).parent_chunk(chunk.document_key, chunk.page_num, chunk.chunk_idx)


def get_child_chunk_path_for_index(page: Page, chunk_idx: int, child_idx: int) -> Path:
    return layout_for(page.gp).child_chunk(page.document_key, page.page_num, chunk_idx, child_idx)


def get_child_chunk_path(chunk: ChildChunkDescriptor) -> Path:
    return layout_for(chunk.gp).child_chunk(chunk.document_key, chunk.page_num, chunk.chunk_idx, chunk.child_idx)


def get_child_chunk_id(game_id: str, game_version:str, document_key: str, page_ordinal: int, chunk_idx: int, child_idx: int) -> str:
    chunk_key = get_page_chunk_id(game_id, game_version, document_key, page_ordinal, chunk_idx)
    return f"{chunk_key}#{child_idx}"


def chunks_for_page_iter(page: Page) -> Iterator[ChunkDescriptor]:
    # Parent and child chunks live in separate directories, so a page's parents
    # can be listed without a pattern that has to exclude its children.
    chunks_path = layout_for(page.gp).parent_chunks_dir(page.document_key)
    chunk_pattern = re.compile(rf"^{page.page_num:04d}_(\d+)\.json$")
    for path in chunks_path.glob(f"{page.page_num:04d}_*.json"):
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
    chunks_path = layout_for(chunk.gp).child_chunks_dir(chunk.document_key)
    child_chunk_pattern = re.compile(rf"^{chunk.page_num:04d}_{chunk.chunk_idx:04d}_(\d+)\.json$")
    for path in chunks_path.glob(f"{chunk.page_num:04d}_{chunk.chunk_idx:04d}_*.json"):
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
        parent_chunk.metadata["page_chunk_index"] = parent_idx
        parent_chunk.metadata["page_chunk_count"] = len(parent_chunks)
        parent_chunk.id = f"{document.id}#{parent_idx}"
        child_chunks = child_splitter.split_documents([parent_chunk])
        parent_start = parent_chunk.metadata.get("start_index", 0)
        for child_idx, child_chunk in enumerate(child_chunks):
            child_chunk.id = f"{parent_chunk.id}#{child_idx}"
            child_chunk.metadata = {**parent_chunk.metadata, **child_chunk.metadata}
            child_chunk.metadata["start_index"] += parent_start
            child_chunk.metadata["end_index"] += parent_start
            child_chunk.metadata["doc_id"] = parent_chunk.id
            child_chunk.metadata["child_chunk_index"] = child_idx
            child_chunk.metadata["child_chunk_count"] = len(child_chunks)
        split_docs.append((parent_chunk, child_chunks))
    return split_docs


async def write_parent_and_child_chunks(
    parent_splitter: TextSplitter,
    child_splitter: TextSplitter,
    page: Page,
    page_document_offset: int = 0,
):
    tasks = []
    document = await page_to_document(page)
    documents_split = split_document(
        parent_splitter,
        child_splitter,
        document,
    )
    for parent_idx, (parent_doc, child_docs) in enumerate(documents_split):
        parent_doc.metadata["start_index"] += page_document_offset
        parent_doc.metadata["end_index"] += page_document_offset
        for child_idx, child_doc in enumerate(child_docs):
            child_doc.metadata["start_index"] += page_document_offset
            child_doc.metadata["end_index"] += page_document_offset
            child_chunk_path = get_child_chunk_path_for_index(page, parent_idx, child_idx)
            tasks.append(aspit(dumps(child_doc), child_chunk_path))
        parent_chunk_path = get_chunk_path_for_index(page, parent_idx)
        tasks.append(aspit(dumps(parent_doc), parent_chunk_path))
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

        write_tasks = []
        for rulebook in self.gp["rulebooks"]:
            document_key = rulebook["document_key"]
            document_layout = layout_for(self.gp)
            document_layout.parent_chunks_dir(document_key).mkdir(parents=True, exist_ok=True)
            document_layout.child_chunks_dir(document_key).mkdir(parents=True, exist_ok=True)

            # Read full document.md
            full_document_markdown = await aslurp(layout_for(self.gp).document_md(document_key))

            # Phase 1: split all pages to determine document-level chunk count
            page_splits: list[Tuple[Page, int, Sequence[Tuple[Document, Sequence[Document]]]]] = []
            async for page in get_pages_iter(self.gp, document_key):
                page_metadata = load_page_metadata(page)
                page_document_offset = page_metadata["start_index"]
                document = await page_to_document(page)
                splits = split_document(parent_splitter, child_splitter, document)
                page_splits.append((page, page_document_offset, splits))

            document_chunk_count = sum(len(splits) for _, _, splits in page_splits)

            # Phase 2: assign document-level indices
            document_chunk_idx = 0
            for page, page_document_offset, splits in page_splits:
                for parent_idx, (parent_doc, child_docs) in enumerate(splits):
                    parent_doc.metadata["start_index"] += page_document_offset
                    parent_doc.metadata["end_index"] += page_document_offset
                    parent_doc.metadata["chunk_index"] = document_chunk_idx
                    parent_doc.metadata["chunk_count"] = document_chunk_count

                    for child_idx, child_doc in enumerate(child_docs):
                        child_doc.metadata["start_index"] += page_document_offset
                        child_doc.metadata["end_index"] += page_document_offset
                        child_doc.metadata["chunk_index"] = document_chunk_idx
                        child_doc.metadata["chunk_count"] = document_chunk_count

                    document_chunk_idx += 1

            # Phase 3: Validate document-level indices
            #
            # This is a sanity check to ensure that we can map our chunks have
            # the correct offsets back to the original markdown. This is
            # important because we use these offsets for merging and rearranging
            # chunks at runtime.
            for page, page_document_offset, splits in page_splits:
                for parent_idx, (parent_doc, child_docs) in enumerate(splits):
                    start_index = parent_doc.metadata["start_index"]
                    end_index = parent_doc.metadata["end_index"]
                    assert full_document_markdown[start_index:end_index] == parent_doc.page_content, f"Parent chunk index mismatch for {page.document_key} page {page.page_num} chunk {parent_idx}"
                    for child_idx, child_doc in enumerate(child_docs):
                        child_start = child_doc.metadata["start_index"]
                        child_end = child_doc.metadata["end_index"]
                        assert full_document_markdown[child_start:child_end] == child_doc.page_content, f"Child chunk index mismatch for {page.document_key} page {page.page_num} chunk {parent_idx} child {child_idx}"

            # Phase 4: Write chunks
            for page, page_document_offset, splits in page_splits:
                for parent_idx, (parent_doc, child_docs) in enumerate(splits):
                    for child_idx, child_doc in enumerate(child_docs):
                        child_chunk_path = get_child_chunk_path_for_index(page, parent_idx, child_idx)
                        write_tasks.append(aspit(dumps(child_doc), child_chunk_path))
                    parent_chunk_path = get_chunk_path_for_index(page, parent_idx)
                    write_tasks.append(aspit(dumps(parent_doc), parent_chunk_path))

        await asyncio.gather(*write_tasks)
