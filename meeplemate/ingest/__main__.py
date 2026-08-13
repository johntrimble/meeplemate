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
from meeplemate.ingest.layout import PackageLayout
from meeplemate.ingest.initgp import InitGamePackageJob
from meeplemate.ingest.ocr import BuildTextJob, OcrJob, PageNumberFixUpJob, PageNumberOcrJob
from meeplemate.ingest.render import RenderJob
from meeplemate.ingest.documentmetadata import DocumentMetadataJobJob
from meeplemate.ingest.summary import ExampleQuestionsJob, GenerateGameReferenceJob, PresentationJob, SettingSummaryJob

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
def render(path: Path):
    """Render every rulebook page to a PNG. Deterministic; no model required."""
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "render_job": (
                afactory(
                    RenderJob,
                    astart=RenderJob.run,
                )(
                    path=path,
                    max_size=settings.ingest.render.max_image_size,
                    pdf_page_chunk=settings.ingest.render.pdf_page_chunk,
                    dpi=settings.ingest.render.dpi,
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
def ocr(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "ocr_client": (
                factory(AsyncOpenAI)(
                    api_key=settings.ingest.ocr.api_key,
                    base_url=settings.ingest.ocr.base_url,
                    timeout=settings.ingest.ocr.timeout,
                ),
                []
            ),
            "ocr_job": (
                afactory(
                    OcrJob,
                    astart=OcrJob.run
                )(
                    path=path,
                    model_config=settings.ingest.ocr,
                    max_ocr_workers=settings.ingest.max_ocr_workers,
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
def build_text(path: Path):
    """Merge OCR output into per-page and per-document markdown."""
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "build_text_job": (
                afactory(
                    BuildTextJob,
                    astart=BuildTextJob.run,
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
def page_number_ocr(path: Path):
    settings: Config = Config() # type: ignore
    app_system: System = create_app_system(settings)
    system = subsystem(
        app_system,
        extra_components={
            "ocr_client": (
                factory(AsyncOpenAI)(
                    api_key=settings.ingest.page_number_ocr.api_key,
                    base_url=settings.ingest.page_number_ocr.base_url,
                    timeout=settings.ingest.page_number_ocr.timeout,
                ),
                []
            ),
            "ocr_job": (
                afactory(
                    PageNumberOcrJob,
                    astart=PageNumberOcrJob.run
                )(
                    path=path,
                    model_config=settings.ingest.page_number_ocr,
                    max_ocr_workers=settings.ingest.max_ocr_workers,
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
                    parent_chunk_size=settings.ingest.chunk_size,
                    parent_chunk_overlap=settings.ingest.chunk_overlap,
                    child_chunk_size=settings.ingest.child_chunk_size,
                    child_chunk_overlap=settings.ingest.child_chunk_overlap,
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
@click.option("--overwrite", is_flag=True, help="Re-import in place even if this game_version already exists in the DB.")
def import_documents(path: Path, overwrite: bool):
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
                    "bm25_builder": "bm25_index_builder",
                },
            )
        }
    )
    system = System.subsystem(system, names=["import_job"])

    async def _import_documents():
        async with system.astart() as services:
            await run_import_documents(services["import_job"], overwrite=overwrite)

    asyncio.run(_import_documents())


@cli.command()
@click.argument("path", type=Path)
@click.option("--version", "version_override", default=None,
              help="Index this game_version instead of the currently-published one.")
def rebuild_bm25(path: Path, version_override: str | None):
    """Rebuild the BM25 postings index for a game's published version.

    The build reads parent chunks straight out of the docstore, so it needs
    nothing but (game_id, game_version) — no chunk files, no re-embedding.
    That makes it the cheap way to index a version imported before the index
    existed, and the fast loop for tuning k1/b without touching ingest.

    Defaults to the version the game_version_store currently points at, not the
    one in the local manifest: those diverge as soon as someone runs
    `update-version` without re-importing, and indexing an unimported version
    silently produces an empty index.
    """
    settings: Config = Config()  # type: ignore
    app_system: System = create_app_system(settings)
    system = System.subsystem(app_system, names=["bm25_index_builder", "game_version_store"])
    gp = load_game_package(path)
    game_id = gp["game_id"]

    async def _rebuild():
        async with system.astart() as services:
            game_version = version_override
            if game_version is None:
                game_key = (await services["game_version_store"].amget([game_id]))[0]
                if game_key:
                    game_version = game_key.split("#", 1)[1]
                else:
                    game_version = gp.get("game_version")
                    click.echo(
                        f"{game_id}: not published yet, falling back to the manifest version"
                    )
            if not game_version:
                raise click.ClickException(
                    f"No game_version for {game_id}: it has never been imported and the "
                    f"manifest at {path} has none."
                )
            counts = await services["bm25_index_builder"].abuild(
                game_id=game_id, game_version=game_version
            )
            click.echo(
                f"{game_id}#{game_version}: "
                f"{counts['doc_count']} parents, {counts['term_count']} terms, "
                f"{counts['posting_count']} postings"
            )

    asyncio.run(_rebuild())


@cli.command()
@click.argument("path", type=Path)
def update_version(path: Path):
    """Stamp a new game version.

    The version lives in its own file rather than the manifest: it changes on
    every re-import, while the manifest only changes when the source rulebooks
    do, and a step that owns one small file is easier to reason about than a
    third writer of a shared one.
    """
    from uuid_utils import uuid7
    new_version = str(uuid7())
    version_path = PackageLayout(path).version()
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(new_version)
    logger.info("Stamped game version", path=str(version_path), game_version=new_version)


@cli.command("migrate-layout")
@click.argument("path", type=Path)
@click.option("--apply", "apply_changes", is_flag=True, help="Perform the moves. Without this, only report them.")
@click.option("--verify", is_flag=True, help="After migrating, check every artifact the manifest implies exists.")
@click.option("--source-dir", type=Path, default=Path("data/rules"), show_default=True,
              help="Where to find source PDFs when backfilling raw_documents/.")
def migrate_layout_command(path: Path, apply_changes: bool, verify: bool, source_dir: Path):
    """Move a package from the flat layout to the per-step layout."""
    from meeplemate.ingest.migrate_layout import (
        apply_plan,
        backfill_raw_documents,
        plan_migration,
        rewrite_manifest,
        verify_package,
    )

    plan = plan_migration(path)
    if plan.already_migrated:
        click.echo(f"{path}: already on the new layout")
    else:
        if not plan.ok:
            click.echo(f"{path}: {len(plan.unclassified)} unrecognised path(s); refusing to migrate:")
            for unknown in plan.unclassified[:20]:
                click.echo(f"  {unknown}")
            raise click.ClickException(
                "Unrecognised files would be left behind. Classify them before migrating."
            )

        click.echo(f"{path}: {len(plan.moves)} file(s) to move")
        if not apply_changes:
            for move in plan.moves[:15]:
                click.echo(f"  {move.src.relative_to(path)} -> {move.dst.relative_to(path)}")
            if len(plan.moves) > 15:
                click.echo(f"  ... and {len(plan.moves) - 15} more")

        moved = apply_plan(plan) if apply_changes else 0
        version = rewrite_manifest(path, apply=apply_changes)
        backfilled = backfill_raw_documents(path, source_dir, apply=apply_changes)

        if apply_changes:
            click.echo(f"{path}: moved {moved} file(s); version {version or '(none)'}")
            if backfilled:
                click.echo(f"{path}: backfilled {len(backfilled)} source PDF(s)")
        else:
            click.echo(f"{path}: dry run, nothing changed. Pass --apply to migrate.")
            if backfilled:
                click.echo(f"{path}: would backfill {len(backfilled)} source PDF(s)")

    if verify:
        problems = verify_package(path)
        if problems:
            click.echo(f"{path}: {len(problems)} missing artifact(s)")
            for problem in problems[:20]:
                click.echo(f"  {problem}")
            raise click.ClickException("Verification failed")
        click.echo(f"{path}: verified")


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
                    "bm25_index": "bm25_index_builder",
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


if __name__ == "__main__":
    cli()
