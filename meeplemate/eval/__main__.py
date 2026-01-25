import json
from pathlib import Path
import click
from gradio import skip
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.stores import BaseStore
from uuid_utils import uuid7
from meeplemate.eval import collect_runs_by_name_iter, load_persisted_run, skip_run_types, test_suites, TestRunTracer, get_test_run_file_path
from dev_system import areload, get_service
from dataclasses import dataclass
import fnmatch

from meeplemate.ingest.gamepackage import GamePackage, get_game_key_for_id_version
from meeplemate.component_system import factory
from meeplemate.qa_graph import QAService, QuoteValidationException
from meeplemate.util import slurp_json

@dataclass
class GameService:
    data_store: BaseStore
    version_store: BaseStore
    
    async def get_current_version_for_game(self, game_id: str) -> str | None:
        results = await self.version_store.amget([game_id])
        assert len(results) == 1 and results[0] is not None, "No version found for game_id"
        version = results[0]
        return str(version)
    
    async def get_manifest(self, game_id: str) -> GamePackage | None:
        game_key = await self.get_current_version_for_game(game_id)
        manifest = self.data_store.mget([game_key])[0]
        return manifest


def get_eval_generation_runs_dir():
    import meeplemate
    from pathlib import Path

    meeplemate_file = Path(meeplemate.__file__)
    # If it is an __init__.py file, get the parent
    if meeplemate_file.name == "__init__.py":
        meeplemate_file = meeplemate_file.parent
    
    # Now get the parent directory
    project_dir = meeplemate_file.parent

    return project_dir / "data" / "evals" / "generation_runs"


def get_most_recent_group_run_id(eval_runs_dir: Path) -> str | None:
    import os
    from datetime import datetime

    if not eval_runs_dir.exists():
        return None

    subdirs = [d for d in os.listdir(eval_runs_dir) if (eval_runs_dir / d).is_dir()]
    if not subdirs:
        return None

    # Get created times and find the most recent
    subdir_times = [(d, (eval_runs_dir / d).stat().st_ctime) for d in subdirs]
    subdir_times.sort(key=lambda x: x[1], reverse=True)
    return subdir_times[0][0]


async def _print_summary_of_run(filter: str = "*", group_run_id: str | None = None):
    # Get most recent group_run_id if not provided, use created-at to find latest
    if group_run_id is None:
        eval_runs_dir = get_eval_generation_runs_dir()
        group_run_id = get_most_recent_group_run_id(eval_runs_dir)
    
    assert group_run_id is not None, "No group_run_id provided and no runs found"

    eval_runs_dir = get_eval_generation_runs_dir()

    for test_suite in test_suites:
        for test_case in test_suite["test_cases"]:
            if not fnmatch.fnmatch(test_case["name"], filter):
                continue

            run_file = get_test_run_file_path(eval_runs_dir, group_run_id, test_suite["name"], test_case["name"])
            if not run_file.exists():
                continue

            run = load_persisted_run(run_file)
            runs = list(collect_runs_by_name_iter(run, "check_answer_progress"))
            answer_run = runs[0] if runs else None
            if answer_run is None:
                continue

            assert answer_run.outputs is not None, "No outputs in answer_chain_run"
            response = answer_run.outputs["response"]

            answer_chain_run = list(collect_runs_by_name_iter(answer_run, "game_agent_answer_chain"))[0]
            input_documents = answer_chain_run.inputs["documents"]
            
            query = answer_chain_run.inputs["query"]
            assert answer_chain_run.outputs is not None, "No outputs in answer_chain_run"
            parsed = answer_chain_run.outputs["parsed"]

            document_summary = "\n".join(
                f"{doc['rulebook_name']}, {doc['page']}, {doc['content'][:100].replace('\n', ' ')}"
                for doc in input_documents
            )

            click.echo(f"Test Suite: {test_suite['name']}, Test Case: {test_case['name']}")
            click.echo(f"Query:")
            click.echo(f"{query.strip()}")
            click.echo(f"")
            click.echo(f"Response:")
            click.echo(f"{response}")
            click.echo(f"")
            click.echo(f"Used Documents:")
            click.echo(f"{document_summary}")
            click.echo(f"")
            click.echo(f"Answer:")
            click.echo(f"{json.dumps(parsed, indent=2)}")


async def _run_gen_qa_tests(filter: str, group_run_id: str):
    extra_components = {
        # Lets not persist the graph state during evals
        "checkpointer": (
            factory(InMemorySaver, ignore_context_manager=True)(),
            []
        )
    }

    await areload(
        ["game_service", "qa_service", "checkpointer"],
        extra_components=extra_components
    )

    game_service: GameService = get_service("game_service")
    qa_service: QAService = get_service("qa_service")

    tracer = TestRunTracer(
        get_eval_generation_runs_dir(),
        run_transformer=skip_run_types(
            {
                "RunnableLambda*",
                "RunnableSequence*",
                "RunnableAssign*",
                "RunnableParallel*",
                "RunnableWithFallbacks*", 
            }
        )
    )

    for test_suite in test_suites:
        game_id = test_suite["params"]["game_id"]
        manifest = await game_service.get_manifest(game_id)
        assert manifest is not None, f"Manifest not found for game_id: {game_id}"

        for test_case in test_suite["test_cases"]:
            # Compare test_case name to filter (contains wildcard *)
            import fnmatch
            if not fnmatch.fnmatch(test_case["name"], filter):
                continue

            query = test_case["query"]

            metadata = {
                "test_group_run_id": group_run_id,
                "test_suite": test_suite["name"],
                "test_case": test_case["name"],
            }
            config: RunnableConfig = {
                "callbacks": [
                    tracer,
                ],
                "configurable": {"thread_id": str(uuid7())},
                "metadata": metadata
            }

            _ = await qa_service.ainvoke(
                {"manifest": manifest, "query": query},
                config=config
            )
         

@click.group()
def cli():
    pass


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
def run_gen_qa_tests(filter: str, group_run_id: str | None = None):
    import asyncio
    # Default group_run_id is today's date in YYYY-MM-DD format
    if group_run_id is None:
        from datetime import datetime
        group_run_id = datetime.now().strftime("%Y-%m-%d")

    asyncio.run(_run_gen_qa_tests(filter, group_run_id))


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
def print_run_summary(filter: str = "*", group_run_id: str | None = None):
    import asyncio
    asyncio.run(_print_summary_of_run(filter, group_run_id))


if __name__ == "__main__":
    cli()