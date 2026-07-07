
import asyncio
import copy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import re
from typing import Any, AsyncIterator, Coroutine, Tuple, TypedDict

from langchain_core.documents.base import Document
from langchain_core.load import dumps, loads
from langchain_core.stores import BaseStore
from langchain_core.vectorstores.base import VectorStore

from meeplemate.ingest.chunkbuild import ChildChunkDescriptor, ChunkDescriptor, child_chunks_for_chunk_iter, chunks_for_page_iter, get_child_chunk_path, get_chunk_path
from meeplemate.ingest.gamepackage import GamePackage, get_game_example_questions_path, get_game_presentation_path, get_page, get_pages_iter, page_to_document, get_game_key
from structlog import get_logger

from meeplemate.util import amap, achain_from_aiterable, aslurp, aslurp_yaml, sem_guard

logger = get_logger(__name__)


@dataclass
class ImportDocumentsJob:
    gp: GamePackage
    vector_store: VectorStore
    full_page_store: BaseStore[str, Document]
    game_data_store: BaseStore[str, Any]
    game_version_store: BaseStore[str, Any]
    game_questions_store: BaseStore[str, Any]
    chunk_store: BaseStore[str, Document]
    path: Path
    concurrency: int


def get_page_path_bases(gp: GamePackage, doc_key: str|None=None) -> list[Tuple[str, int, Path]]:
    page_paths: list[Tuple[str, int, Path]] = []
    for rulebook in gp["rulebooks"]:
        if doc_key is not None and doc_key != rulebook["document_key"]:
            continue

        directory = (gp["path"] / rulebook["document_key"]).resolve()
        page_pattern = re.compile(r"^(\d+)(\..*)?$")
        paths = glob("*.md", root_dir=directory)
        
        for path in paths:
            match = page_pattern.match(Path(path).stem)
            if match:
                page_num = int(match.group(1))
                path = directory / path
                page_paths.append((rulebook["document_key"], page_num, path.with_suffix("")))

    page_paths.sort(key=lambda x: (x[0], x[1]))
    return page_paths


async def import_game_data(job: ImportDocumentsJob) -> None:
    game_data = copy.deepcopy(job.gp)
    game_data.pop("path")
    for rulebook in game_data["rulebooks"]:
        rulebook.pop("strategy", None)
        rulebook.pop("path", None)

    # Add presentation data to help with displaying the game on the frontend
    presentation_path = get_game_presentation_path(job.gp)
    presentation_data = await aslurp_yaml(presentation_path)

    if "unicode_character" in presentation_data:
        game_data["emoji"] = presentation_data["unicode_character"]
    
    if "background_color" in presentation_data:
        game_data["background_color"] = presentation_data["background_color"]
    
    logger.info("Saving game data", game_data=game_data)

    await job.game_data_store.amset([(get_game_key(job.gp), game_data)])


async def import_example_questions(gp: GamePackage, game_questions_store: BaseStore[str, Any]) -> None:
    # Example questions are an optional asset produced by ExampleQuestionsJob. They are keyed by
    # game_id (not version) since they carry across game versions. This is called both from the
    # full import job and standalone (see the `import-example-questions` CLI command).
    example_questions_path = get_game_example_questions_path(gp)
    if not example_questions_path.exists():
        logger.info("No example questions asset to import", game_id=gp["game_id"])
        return

    data = await aslurp_yaml(example_questions_path)
    questions = data.get("questions") if isinstance(data, dict) else None
    if not questions:
        logger.info("Example questions asset had no questions", game_id=gp["game_id"])
        return

    logger.info("Saving example questions", game_id=gp["game_id"], questions=questions)
    await game_questions_store.amset([(gp["game_id"], questions)])


def get_all_chunks_iter(gp: GamePackage) -> AsyncIterator[ChunkDescriptor]:
    pages_iter = get_pages_iter(gp)
    chunks_iter = achain_from_aiterable(
        amap(chunks_for_page_iter, pages_iter)
    )
    return chunks_iter


def get_all_child_chunks_iter(gp: GamePackage) -> AsyncIterator[ChildChunkDescriptor]:
    all_chunks = get_all_chunks_iter(gp)
    all_child_chunks = achain_from_aiterable(
        amap(child_chunks_for_chunk_iter, all_chunks)
    )
    return all_child_chunks
    

async def aslurp_document(path: Path) -> Document:
    content = await aslurp(path)
    document = loads(content)
    return document


def add_game_metadata_to_document(document: Document, gp: GamePackage) -> Document:
    game_metadata = {
        "game_id": gp["game_id"],
        "game_version": gp.get("game_version", ""),
    }
    document.metadata.update(game_metadata)

    # We need to fix the doc_id reference for child chunks to their parent
    if "doc_id" in document.metadata:
        parent_doc_id = document.metadata["doc_id"]
        # The parent doc ID will be the chunk ID which includes the page number, we need to replace that with the game version
        parts = parent_doc_id.split("#")
        if len(parts) > 1:
            parts[1] = gp.get("game_version", "")
            document.metadata["doc_id"] = "#".join(parts)

    # We also need to fix the ID for chunks as that includes the version number
    assert document.id and isinstance(document.id, str), "Document must have a string ID"
    parts = document.id.split("#")
    if len(parts) > 1:
        parts[1] = gp.get("game_version", "")
        document.id = "#".join(parts)

    return document


async def run_import_documents(job: ImportDocumentsJob) -> None:
    # Ensure the vector store partition exists before spawning concurrent tasks
    ensure_partition = getattr(job.vector_store, "ensure_partition", None)
    if callable(ensure_partition):
        game_version = job.gp.get("game_version")
        if game_version:
            await ensure_partition(game_version)  # type: ignore[misc]

    tasks = []
    sem = asyncio.Semaphore(job.concurrency)

    def add_sem_guarded_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.create_task(sem_guard(coro, sem))
        tasks.append(task)
        return task

    # Add full pages
    pages_iter = get_pages_iter(job.gp)
    page_documents_iter = amap(page_to_document, pages_iter)
    async for document in page_documents_iter:
        document = add_game_metadata_to_document(document, job.gp)
        assert document.id is not None, "Document must have an ID"
        add_sem_guarded_task(
            job.full_page_store.amset([(document.id, document)])
        )

    # Add chunks for each page
    chunks_iter = get_all_chunks_iter(job.gp)
    chunk_paths_iter = amap(get_chunk_path, chunks_iter)
    documents_iter = amap(aslurp_document, chunk_paths_iter)
    async for document in documents_iter:
        document = add_game_metadata_to_document(document, job.gp)
        assert document.id is not None, "Document must have an ID"
        add_sem_guarded_task(
            job.chunk_store.amset([(document.id, document)])
        )

    # Add vectors for child chunks
    child_chunks_iter = get_all_child_chunks_iter(job.gp)
    child_chunk_paths_iter = amap(get_child_chunk_path, child_chunks_iter)
    child_documents_iter = amap(aslurp_document, child_chunk_paths_iter)
    async for document in child_documents_iter:
        document = add_game_metadata_to_document(document, job.gp)
        assert document.id is not None, "Document must have an ID"
        add_sem_guarded_task(
            job.vector_store.aadd_documents([document])
        )

    # Import the game data
    add_sem_guarded_task(
        import_game_data(job)
    )

    # Import example questions (optional asset)
    add_sem_guarded_task(
        import_example_questions(job.gp, job.game_questions_store)
    )

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Data imported! Lets update the current game version
    game_key = get_game_key(job.gp)
    await job.game_version_store.amset([(job.gp['game_id'], game_key)])
