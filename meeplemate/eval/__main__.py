import asyncio
import json
from pathlib import Path
import struct
import click
from gradio import skip
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.stores import BaseStore
import structlog
from uuid_utils import uuid7
from meeplemate.eval import collect_runs_by_name_iter, load_persisted_run, skip_run_types, snake_case, test_suites, TestRunTracer, get_test_run_file_path, parse_group_run_id
from dev_system import areload, get_service
from dataclasses import dataclass
import fnmatch

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric
from deepeval.models.llms.local_model import LocalModel
from deepeval import evaluate
from deepeval.metrics import GEval, BaseMetric
from deepeval.test_case import LLMTestCaseParams

from typing import Optional, Tuple, Union
from pydantic import BaseModel
from openai.types.chat import ChatCompletion
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.constants import ProviderSlug as PS

from meeplemate.ingest.gamepackage import GamePackage, get_game_key_for_id_version
from meeplemate.component_system import factory
from meeplemate.qa_graph import Chunk, QAService, QAServiceInput, QuoteValidationException
from meeplemate.util import slurp_json
from meeplemate.config import GameService

logger = structlog.get_logger(__name__)

# Custom LocalModel that properly supports structured outputs with vllm
class StructuredLocalModel(LocalModel):
    """LocalModel subclass that properly uses vllm's structured output features.

    This adds the response_format parameter when a schema is requested, which tells
    vllm to enforce JSON schema compliance via guided decoding.
    """

    @create_retry_decorator(PS.LOCAL)
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        client = self.load_model(async_mode=False)

        # Build request kwargs
        kwargs = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            **self.generation_kwargs,
        }

        # If schema is provided, add response_format for structured outputs
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True
                }
            }

        response: ChatCompletion = client.chat.completions.create(**kwargs)
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0

    @create_retry_decorator(PS.LOCAL)
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = prompt

        client = self.load_model(async_mode=True)

        # Build request kwargs
        kwargs = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            **self.generation_kwargs,
        }

        # If schema is provided, add response_format for structured outputs
        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True
                }
            }

        response: ChatCompletion = await client.chat.completions.create(**kwargs)
        res_content = response.choices[0].message.content

        if schema:
            json_output = trim_and_load_json(res_content)
            return schema.model_validate(json_output), 0.0
        else:
            return res_content, 0.0


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

    # Parse group_run_id to extract base ID and run number
    base_group_run_id, run_number = parse_group_run_id(group_run_id)

    for test_suite in test_suites:
        for test_case in test_suite["test_cases"]:
            if not fnmatch.fnmatch(test_case["name"], filter):
                continue

            run_file = get_test_run_file_path(eval_runs_dir, base_group_run_id, test_suite["name"], test_case["name"], run_number=run_number)
            if not run_file.exists():
                continue

            run = load_persisted_run(run_file)
            query = run.inputs["query"]
            assert run.outputs is not None, "No outputs in run"
            evidence = run.outputs["evidence"] if "evidence" in run.outputs else []
            response = run.outputs["response"]
          
            document_summary = "\n".join(
                f"{doc['rulebook_name']}, {doc['page']}, {doc['content'][:100].replace('\n', ' ')}"
                for doc in evidence
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
            for child_run in collect_runs_by_name_iter(run, "game_agent_answer_chain"):
                assert child_run.outputs is not None, "No outputs in child_run"
                parsed = child_run.outputs["parsed"]
                click.echo(f"{json.dumps(parsed, indent=2)}")


async def _run_qa_gen_multiple_runs(filter: str, group_run_id: str, number_of_runs: int, skip_retrieval: bool = False):
    # TODO: Move this component setup elsewhere... maybe make these functions
    # part of the component system?
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

    # Generate multiple runs with run numbers
    tasks = []
    for i in range(number_of_runs):
        run_number = i + 1
        tasks.append(
            asyncio.create_task(
                _run_qa_gen_no_start_system(filter, group_run_id, skip_retrieval=skip_retrieval, run_number=run_number)
            )
        )
    await asyncio.gather(*tasks)


async def _run_qa_gen(filter: str, group_run_id: str, skip_retrieval: bool = False):
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
    await _run_qa_gen_no_start_system(filter, group_run_id, skip_retrieval=skip_retrieval, run_number=None)


async def _run_qa_gen_no_start_system(filter: str, group_run_id: str, skip_retrieval: bool = False, run_number: int | None = None):
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
    tasks = []
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
            evidence = test_case.get("evidence", None)

            if skip_retrieval and evidence is None:
                click.echo(f"Skipping retrieval for test case without evidence: {test_case['name']}")
                continue


            metadata = {
                "test_group_run_id": group_run_id,
                "test_suite": test_suite["name"],
                "test_case": test_case["name"],
                "skip_retrieval": skip_retrieval,
                "run_number": run_number,
            }
            config: RunnableConfig = {
                "callbacks": [
                    tracer,
                ],
                "configurable": {"thread_id": str(uuid7())},
                "metadata": metadata
            }

            input: QAServiceInput = {
                "manifest": manifest,
                "query": query,
                "messages": [],
                "recursion_depth": 0,
                "evidence": []
            }
            if skip_retrieval:
                logger.info("Skipping retrieval as per flag", test_case=test_case['name'])
                chunks: list[Chunk] = []
                for item in evidence:
                    chunks.append(
                        {
                            "rulebook_name": item["rulebook"],
                            "page": item["page"],
                            "start_index": item.get("start_index", -1),
                            "end_index": item.get("end_index", -1),
                            "content": item["quote"],
                        }
                    )
                input["evidence"] = chunks
                logger.info(f"Starting generation with evidence", number_of_chunks=len(chunks), test_case=test_case['name'])
            logger.info(f"Starting QA generation", test_case=test_case['name'])
            tasks.append(
                asyncio.create_task(
                    qa_service.ainvoke(
                        input,
                        config=config
                    )
                )
            )

    await asyncio.gather(*tasks)


def get_correctness_metric(model) -> BaseMetric:
    correctness_metric = GEval(
        model=model,
        name="Correctness",
        criteria="Determine whether the actual output reaches the same general conclusion as the expected output. Do not penalize if one explores exceptions or edge cases not mentioned in the other. Focus on whether both outputs agree on the main point.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    return correctness_metric


async def _run_qa_eval(filter: str, base_group_run_id: str):
    """Evaluate all runs for a given base group_run_id.

    Discovers all runs (e.g., __run001, __run002, etc.) and evaluates them.
    All results are stored in a single directory: qa_evals/{base_group_run_id}/
    """
    from meeplemate.eval.analysis import find_run_groups

    llm = StructuredLocalModel(
        model="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4",
        api_key="dummy",
        base_url="http://192.168.0.44:8000/v1"
    )

    # Find all runs for this group
    run_groups = find_run_groups(base_group_run_id)

    if not run_groups:
        click.echo(f"No runs found for {base_group_run_id}")
        return

    click.echo(f"Found {len(run_groups)} runs to evaluate: {run_groups}")
    click.echo()

    llm_tests = []

    # Iterate through all discovered runs
    for run_id in run_groups:
        # Parse to get base ID and run number
        parsed_base_id, run_number = parse_group_run_id(run_id)

        click.echo(f"Processing {run_id}...")

        for test_suite in test_suites:
            game_id = test_suite["params"]["game_id"]
            for test_case in test_suite["test_cases"]:
                if not fnmatch.fnmatch(test_case["name"], filter):
                    continue

                if "reference_answer" not in test_case:
                    continue

                run_file = get_test_run_file_path(
                    get_eval_generation_runs_dir(),
                    parsed_base_id,
                    test_suite["name"],
                    test_case["name"],
                    run_number=run_number
                )

                if not run_file.exists():
                    continue

                run = load_persisted_run(run_file)

                # Get the actual output from the run
                assert run.outputs is not None, "No outputs in run"
                actual_output = run.outputs["response"]

                # Create unique test name including run ID
                test_name = f"{snake_case(test_suite['name'])}__{snake_case(test_case['name'])}"
                if run_number is not None:
                    test_name += f"__run{str(run_number).zfill(3)}"

                llm_test = LLMTestCase(
                    input=test_case["query"],
                    actual_output=actual_output,
                    expected_output=test_case["reference_answer"],
                    name=test_name,
                )
                llm_tests.append(llm_test)

    if not llm_tests:
        click.echo("No test cases found to evaluate")
        return

    click.echo(f"\nEvaluating {len(llm_tests)} test cases across {len(run_groups)} runs...")
    metrics: list[BaseMetric] = [get_correctness_metric(model=llm), AnswerRelevancyMetric(model=llm)]
    results = evaluate(llm_tests, metrics)

    click.echo(f"\n✓ Evaluation complete! Results saved to: {get_eval_generation_runs_dir().parent / 'qa_evals' / base_group_run_id}")
         

@click.group()
def cli():
    pass


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
@click.option("--number-of-runs", default=None, type=int)
@click.option("--skip-retrieval", is_flag=True, default=False)
def run_qa_gen(filter: str, group_run_id: str | None = None, number_of_runs: int|None = None, skip_retrieval: bool = False):
    import asyncio
    # Default group_run_id is today's date in YYYY-MM-DD format
    if group_run_id is None:
        from datetime import datetime
        group_run_id = datetime.now().strftime("%Y-%m-%d")

    if number_of_runs is None:
        asyncio.run(_run_qa_gen(filter, group_run_id, skip_retrieval=skip_retrieval))
    else:
        asyncio.run(_run_qa_gen_multiple_runs(filter, group_run_id, number_of_runs, skip_retrieval=skip_retrieval))

@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
def run_qa_eval(filter: str, group_run_id: str | None = None):
    import os
    import asyncio

    # Default group_run_id is today's date in YYYY-MM-DD format
    if group_run_id is None:
        from datetime import datetime
        group_run_id = datetime.now().strftime("%Y-%m-%d")

    # Set environment variable DEEPEVAL_RESULTS_FOLDER to data/evals/qa_evals/{group_run_id}
    eval_runs_dir = get_eval_generation_runs_dir().parent / "qa_evals"
    eval_dir = eval_runs_dir / group_run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DEEPEVAL_RESULTS_FOLDER"] = str(eval_dir)

    asyncio.run(_run_qa_eval(filter, group_run_id))


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
@click.option("--skip-retrieval", is_flag=True, default=False)
def print_run_summary(filter: str = "*", group_run_id: str | None = None, skip_retrieval: bool = False):
    import asyncio
    asyncio.run(_print_summary_of_run(filter, group_run_id))

@cli.command()
@click.argument("game-id", required=True)
@click.argument("query", required=True)
def ask(game_id: str, query: str) -> None:
    import asyncio

    async def _run():
        await areload(["game_service", "qa_service"])
        game_service: GameService = get_service("game_service")
        qa_service: QAService = get_service("qa_service")

        manifest = await game_service.get_manifest(game_id)
        assert manifest is not None, f"Manifest not found for game_id: {game_id}"

        config: RunnableConfig = {
            "configurable": {"thread_id": str(uuid7())},
        }

        response = await qa_service.ainvoke(
            {"manifest": manifest, "query": query},
            config=config
        )
        print(response["response"])

    return asyncio.run(_run())


@cli.command()
@click.argument("group-run-id", required=True)
@click.option("--top-n", default=5, help="Number of top/bottom items to show")
def analyze(group_run_id: str, top_n: int):
    """Analyze multi-run evaluation results.

    Analyzes variance across multiple runs (e.g., __run001, __run002, etc.)
    and provides summary statistics showing consistency and variance.

    Example:
        python -m meeplemate.eval analyze 2026-01-25
    """
    from meeplemate.eval.analysis import (
        find_run_groups, get_summary, aggregate_by_test_case, load_all_runs
    )

    # Find all run groups
    run_groups = find_run_groups(group_run_id)

    if not run_groups:
        click.echo(f"Error: No runs found for {group_run_id}")
        click.echo(f"\nChecked directory: {get_eval_generation_runs_dir().parent / 'qa_evals'}")
        return

    # Get summary statistics
    summary = get_summary(group_run_id)

    # Print header
    click.echo("=" * 70)
    click.echo("Multi-Run Evaluation Analysis")
    click.echo("=" * 70)
    click.echo(f"Base group: {summary.base_group_run_id}")
    click.echo(f"Runs found: {summary.num_runs}")
    for rg in summary.run_groups:
        click.echo(f"  - {rg}")
    click.echo(f"Test cases: {summary.num_test_cases}")
    click.echo()

    # Print overall metrics
    click.echo("Overall Metrics:")
    click.echo("-" * 70)
    for metric_name, stats in summary.metrics_summary.items():
        click.echo(f"\n  {metric_name}:")
        click.echo(f"    Mean:      {stats['mean']:.3f} ± {stats['std']:.3f}")
        click.echo(f"    Median:    {stats['median']:.3f}")
        click.echo(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        click.echo(f"    Pass rate: {stats['pass_rate']:.1%} (threshold: {stats['threshold']})")
    click.echo()

    # Load full results for detailed analysis
    results = load_all_runs(group_run_id)
    agg = aggregate_by_test_case(results)

    if agg.empty:
        click.echo("Warning: No aggregated data available")
        return

    # Show most consistent test cases
    click.echo(f"\nTop {top_n} Most Consistent Test Cases (lowest std dev):")
    click.echo("-" * 70)
    consistent = agg.nsmallest(top_n, 'std')
    for idx, row in consistent.iterrows():
        click.echo(f"  {row['test_case'][:50]:50s} | {row['metric_name']:20s} | std: {row['std']:.3f} | mean: {row['mean']:.3f}")

    # Show most variant test cases
    click.echo(f"\nTop {top_n} Most Variant Test Cases (highest std dev):")
    click.echo("-" * 70)
    variant = agg.nlargest(top_n, 'std')
    for idx, row in variant.iterrows():
        click.echo(f"  {row['test_case'][:50]:50s} | {row['metric_name']:20s} | std: {row['std']:.3f} | mean: {row['mean']:.3f}")

    # Show test cases with lowest pass rates
    click.echo(f"\nLowest Pass Rates (test cases that failed most often):")
    click.echo("-" * 70)
    low_pass = agg.nsmallest(top_n, 'pass_rate')
    for idx, row in low_pass.iterrows():
        click.echo(f"  {row['test_case'][:50]:50s} | {row['metric_name']:20s} | pass: {row['pass_rate']:.1%} | mean: {row['mean']:.3f}")

    click.echo()
    click.echo("=" * 70)
    click.echo("\nFor detailed visualizations, use the Jupyter notebook:")
    click.echo("  jupyter notebook notebooks/eval_analysis.ipynb")
    click.echo()


if __name__ == "__main__":
    cli()