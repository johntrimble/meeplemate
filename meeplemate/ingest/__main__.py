import asyncio
from pathlib import Path
import click
from openai import AsyncOpenAI
import structlog

from meeplemate.component_system import System, afactory, factory, subsystem
from meeplemate.config import Config, create_app_system
from meeplemate.ingest.dataimport import ImportDocumentsJob, run_import_documents
from meeplemate.ingest.gamepackage import load_game_package
from meeplemate.ingest.ocr import IngestJob
from meeplemate.ingest.summary import ExtractTerminologyJob, GenerateGameReferenceJob

logger = structlog.get_logger(__name__)


@click.group()
def cli():
    pass


# uv run python -m meeplemate.ingest process --input ./data/rules/munchkin_rules/ --output ./data/ingested/munchkin_rules/

@cli.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
def process(input: Path, output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "ocr_client": (
                factory(AsyncOpenAI)(
                    api_key="EMPTY",
                    base_url="http://vllm-ocr:8000/v1",
                    timeout=3600
                ),
                []
            ),
            "ingest_job": (
                afactory(
                    IngestJob,
                    astart=IngestJob.run
                )(
                    input_dir=input,
                    output_dir=output,
                    max_size=2_000,
                    chunk_size=settings.ingest.chunk_size,
                    chunk_overlap=settings.ingest.chunk_overlap,
                    child_chunk_size=settings.ingest.child_chunk_size,
                    child_chunk_overlap=settings.ingest.child_chunk_overlap,
                ),
                {
                    "ocr_client": "ocr_client"
                }
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


@cli.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
def import_documents(input: Path, output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            **app_system._components,
            "import_job": (
                factory(ImportDocumentsJob)(
                    input_dir=input,
                    output_dir=output,
                    gp=load_game_package(output),
                    concurrency=100,
                ),
                {
                    "retriever": "retriever",
                    "full_page_store": "full_page_store",
                    "game_data_store": "game_data_store"
                },
            )
        }
    )
    system = System.subsystem(system, names=["import_job", "keyspace_creator"])

    async def _import_documents():
        async with system.astart() as services:
            await run_import_documents(services["import_job"])

    asyncio.run(_import_documents())


@cli.command()
@click.argument("output", type=Path)
def generate_reference(output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "generate_reference_job": (
                afactory(
                    GenerateGameReferenceJob,
                    astart=GenerateGameReferenceJob.run,
                )(
                    gp=load_game_package(output),
                    output_dir=output,
                ),
                {
                    "chat_model": "chat_model",
                    "tokenizer": "tokenizer",
                }
            )
        },
    )

    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


@cli.command()
@click.argument("output", type=Path)
def extract_terminology(output: Path):
    settings: Config = Config()
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "extract_terminology_job": (
                afactory(
                    ExtractTerminologyJob,
                    astart=ExtractTerminologyJob.run,
                )(
                    gp=load_game_package(output),
                    output_dir=output,
                ),
                {
                    "chat_model": "chat_model",
                    "tokenizer": "tokenizer",
                }
            )
        },
    )

    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())

if __name__ == "__main__":
    cli()
