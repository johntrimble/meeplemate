
import asyncio
import copy
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import re
from typing import Any, Coroutine, Tuple, TypedDict

from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore

from meeplemate.ingest.gamepackage import GamePackage, get_page, page_to_document
from structlog import get_logger

logger = get_logger(__name__)


class ImportDocumentsServices(TypedDict):
    retriever: BaseRetriever


@dataclass
class ImportDocumentsJob:
    gp: GamePackage
    retriever: BaseRetriever
    full_page_store: BaseStore
    game_data_store: BaseStore
    input_dir: Path
    output_dir: Path
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
    
    logger.info("Saving game data", game_data=game_data)
    
    await job.game_data_store.amset([(game_data["game_id"], game_data)])


async def run_import_documents(job: ImportDocumentsJob) -> None:
    async def process_page(doc_key: str, page_num: int):
        # Get the page content
        page = get_page(job.gp, doc_key, page_num)
        # Get document for the page
        document = await page_to_document(page)
        # Add to retriever which will chunk and store in vector store
        assert hasattr(job.retriever, "aadd_documents"), "Retriever must support aadd_documents"
        await job.retriever.aadd_documents([document])
        # Also store full page in full page store
        await job.full_page_store.amset([(document.id, document)])

    concurrency = job.concurrency
    sem = asyncio.Semaphore(concurrency)

    async def sem_guard(coro: Coroutine[Any, Any, Any], sem: asyncio.Semaphore) -> Any:
        async with sem:
            return await coro

    doc_page_number_bases = get_page_path_bases(job.gp)

    tasks = []
    for doc_key, page_num, page_base in doc_page_number_bases:
        coro = sem_guard(process_page(doc_key, page_num), sem)
        tasks.append(asyncio.create_task(coro))

    await asyncio.gather(*tasks)
    await import_game_data(job)
