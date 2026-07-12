import asyncio
from pathlib import Path
import click
from openai import AsyncOpenAI
import structlog

from meeplemate.component_system import System, afactory, factory, subsystem
from meeplemate.config import Config, create_app_system
from meeplemate.ingest.chunkbuild import BuildChunksJob
from meeplemate.ingest.cleardata import ClearOldDataJob
from meeplemate.ingest.dataimport import ImportDocumentsJob, import_example_questions, run_import_documents
from meeplemate.ingest.gamepackage import load_game_package
from meeplemate.ingest.initgp import InitGamePackageJob
from meeplemate.ingest.ocr import OcrJob, PageNumberFixUpJob, PageNumberOcrJob
from meeplemate.ingest.documentmetadata import DocumentMetadataJobJob
from meeplemate.ingest.summary import ExampleQuestionsJob, ExtractTerminologyJob, GenerateGameReferenceJob, PresentationJob, SettingSummaryJob, save_manifest

logger = structlog.get_logger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input", type=Path)
@click.argument("output", type=Path)
def init_game_package(input: Path, output: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "init_gp_job": (
                afactory(
                    InitGamePackageJob,
                    astart=InitGamePackageJob.run,
                )(
                    input_dir=input,
                    output_dir=output,
                ),
                []
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


# uv run python -m meeplemate.ingest process ./data/rules/munchkin_rules/ ./data/ingested/munchkin_rules/

@cli.command()
@click.argument("path", type=Path)
def ocr(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "ocr_client": (
                factory(AsyncOpenAI)(
                    api_key="EMPTY",
                    base_url="http://vllm-deepseek-ocr:8000/v1",
                    timeout=3600
                ),
                []
            ),
            "ocr_job": (
                afactory(
                    OcrJob,
                    astart=OcrJob.run
                )(
                    path=path,
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
@click.argument("path", type=Path)
def page_number_ocr(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "ocr_client": (
                factory(AsyncOpenAI)(
                    api_key="EMPTY",
                    base_url="http://vllm-glm-ocr:8080/v1",
                    timeout=3600
                ),
                []
            ),
            "ocr_job": (
                afactory(
                    PageNumberOcrJob,
                    astart=PageNumberOcrJob.run
                )(
                    path=path,
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
@click.argument("path", type=Path)
def page_number_fixup(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "page_number_fixup_job": (
                afactory(
                    PageNumberFixUpJob,
                    astart=PageNumberFixUpJob.run
                )(
                    path=path,
                ),
                []
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


@cli.command()
@click.argument("path", type=Path)
def add_metadata(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "page_metadata_job": (
                afactory(
                    DocumentMetadataJobJob,
                    astart=DocumentMetadataJobJob.run,
                )(
                    path=path,
                ),
                []
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


@cli.command()
@click.argument("path", type=Path)
def build_chunks(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "build_chunks_job": (
                afactory(
                    BuildChunksJob,
                    astart=BuildChunksJob.run,
                )(
                    path=path,
                    parent_chunk_size=500,
                    parent_chunk_overlap=50,
                    child_chunk_size=125,
                    child_chunk_overlap=12,
                ),
                {
                    "tokenizer": "tokenizer"
                }
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())

# uv run python -m meeplemate.ingest import-documents ./data/ingested/munchkin_rules/

@cli.command()
@click.argument("path", type=Path)
def import_documents(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            **app_system._components,
            "import_job": (
                factory(ImportDocumentsJob)(
                    path=path,
                    gp=load_game_package(path),
                    concurrency=5,
                ),
                {
                    "vector_store": "vector_store",
                    "game_version_store": "game_version_store",
                    "chunk_store": "docstore",
                    "full_page_store": "full_page_store",
                    "game_data_store": "game_data_store",
                    "game_questions_store": "game_questions_store"
                },
            )
        }
    )
    system = System.subsystem(system, names=["import_job"])

    async def _import_documents():
        async with system.astart() as services:
            await run_import_documents(services["import_job"])

    asyncio.run(_import_documents())


@cli.command()
@click.argument("path", type=Path)
def update_version(path: Path):
    from uuid_utils import uuid7
    manifest = load_game_package(path)
    new_version = str(uuid7())
    manifest["game_version"] = new_version
    save_manifest(manifest)


@cli.command()
def clear_old_data():
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "clear_data_job": (
                afactory(
                    ClearOldDataJob,
                    astart=ClearOldDataJob.run,
                )(),
                {
                    "game_version_store": "game_version_store",
                    "game_data_store": "game_data_store",
                    "docstore": "docstore",
                    "full_page_store": "full_page_store",
                    "vector_store": "vector_store",
                }
            )
        }
    )
    async def _run():
        async with system.astart() as services:
            pass
    
    asyncio.run(_run())


# uv run python -m meeplemate.ingest generate-reference ./data/ingested/munchkin_rules/

@cli.command()
@click.argument("path", type=Path)
def generate_reference(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "generate_reference_job": (
                afactory(
                    GenerateGameReferenceJob,
                    astart=GenerateGameReferenceJob.run,
                )(
                    gp=load_game_package(path),
                    path=path,
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
@click.argument("path", type=Path)
def generate_setting_summary(path: Path):
    settings: Config = Config() # type: ignore
    for model in settings.chat.models:
        model.max_new_tokens = 10_000
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "generate_setting_summary_job": (
                afactory(
                    SettingSummaryJob,
                    astart=SettingSummaryJob.run,
                )(
                    gp=load_game_package(path),
                    path=path,
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
@click.argument("path", type=Path)
def generate_presentation(path: Path):
    settings: Config = Config() # type: ignore
    for model in settings.chat.models:
        model.max_new_tokens = 10_000
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "generate_presentation_job": (
                afactory(
                    PresentationJob,
                    astart=PresentationJob.run,
                )(
                    gp=load_game_package(path),
                    path=path,
                ),
                {
                    "chat_model": "chat_model",
                }
            )
        },
    )

    async def _run():
        async with system.astart() as services:
            pass

    asyncio.run(_run())


@cli.command()
@click.argument("path", type=Path)
def generate_example_questions(path: Path):
    settings: Config = Config() # type: ignore
    for model in settings.chat.models:
        model.max_new_tokens = 10_000
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "generate_example_questions_job": (
                afactory(
                    ExampleQuestionsJob,
                    astart=ExampleQuestionsJob.run,
                )(
                    gp=load_game_package(path),
                    path=path,
                ),
                {
                    "chat_model": "chat_model",
                }
            )
        },
    )

    async def _run():
        async with system.astart() as services:
            pass

    asyncio.run(_run())


@cli.command("import-example-questions")
@click.argument("path", type=Path)
def import_example_questions_command(path: Path):
    """Persist the example_questions.yaml asset to the DB without a full re-import."""
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = System.subsystem(app_system, names=["game_questions_store"])

    async def _run():
        async with system.astart() as services:
            await import_example_questions(
                load_game_package(path),
                services["game_questions_store"],
            )

    asyncio.run(_run())


@cli.command()
@click.argument("output", type=Path)
def extract_terminology(output: Path):
    settings: Config = Config() # type: ignore
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
