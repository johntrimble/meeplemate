import eval_logging

eval_logging.configure_logging()

import asyncio
import json
from pathlib import Path
from typing import Sequence

import click
import fnmatch
import structlog

from uuid_utils import uuid7

from deepeval import evaluate
from deepeval.dataset.golden import Golden
from deepeval.metrics import BaseMetric
from deepeval.models.llms.local_model import LocalModel
from deepeval.test_case import LLMTestCase

from langchain_core.runnables import RunnableConfig

from meeplemate.eval import (
    collect_runs_by_name_iter,
    load_persisted_run,
    snake_case,
    test_suites,
    get_test_run_file_path,
    parse_group_run_id,
    next_group_run_id,
    load_goldens
)
from meeplemate.eval.local_model import StructuredLocalModel
from meeplemate.eval.metrics import get_correctness_metric
from meeplemate.eval.runner import E2ERunner, create_eval_system
from meeplemate.game_service import GameService
from meeplemate.qa_graph import QAService


logger = structlog.get_logger(__name__)

LLM_COST_PER_1M_TOKENS_TABLE = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "input": 0.1,
        "output": 0.3
    }
}





def get_eval_and_generation_output_dir() -> Path:
    import meeplemate
    from pathlib import Path

    meeplemate_file = Path(meeplemate.__file__)
    # If it is an __init__.py file, get the parent
    if meeplemate_file.name == "__init__.py":
        meeplemate_file = meeplemate_file.parent
    
    # Now get the parent directory
    project_dir = meeplemate_file.parent

    return project_dir / "data" / "evals"


def get_eval_generation_runs_dir():
    return get_eval_and_generation_output_dir() / "generation_runs"


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


def build_local_model() -> LocalModel:
    return StructuredLocalModel(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        api_key="dummy",
        # base_url="http://vllm:8000/v1"
        base_url="http://192.168.0.44:8000/v1",
        temperature=0.7,
        generation_kwargs={
            "presence_penalty": 0.6,
            "top_p": 0.8,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.1,
            },
        },
    )


async def _run_qa_eval(filter: str, base_group_run_id: str, llm: LocalModel):
    """Evaluate all runs for a given base group_run_id.

    Discovers all runs (e.g., __run001, __run002, etc.) and evaluates them.
    All results are stored in a single directory: qa_evals/{base_group_run_id}/
    """
    from meeplemate.eval.analysis import find_run_groups

    llm = build_local_model()

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
                if run.outputs is None or "response" not in run.outputs:
                    click.echo(f"  WARNING: Run {run_number} has no response (failed run), treating as failed")
                    actual_output = "[Run failed to produce a response]"
                else:
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
    metrics: list[BaseMetric] = [
        get_correctness_metric(model=llm),
        # AnswerRelevancyMetric(model=llm)
    ]
    results = evaluate(llm_tests, metrics)

    click.echo(f"\n✓ Evaluation complete! Results saved to: {get_eval_generation_runs_dir().parent / 'qa_evals' / base_group_run_id}")


def build_runner(filter:str) -> E2ERunner:
    # Filter test cases based on the provided filter
    goldens = [
        golden 
        for golden in load_goldens() 
        if fnmatch.fnmatch(golden.name or "", filter)
    ]

    # Directory where results go, used for both generation and eval runs
    output_dir = get_eval_and_generation_output_dir()

    # Build the local llm for evaluation (though we won't actually us it here)
    llm = build_local_model()

    runner = E2ERunner(
        target_directory=output_dir,
        goldens=goldens,
        local_llm_model_eval=llm,
        generation_runs_directory_name="generation_runs",
        evaluation_results_directory_name="qa_evals",
    )

    return runner


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
@click.option("--number-of-runs", default=None, type=int)
@click.option("--skip-retrieval", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing run files")
@click.option("--skip-existing", is_flag=True, default=False, help="Skip test cases with existing output files")
def run_qa_gen(filter: str, group_run_id: str | None = None, number_of_runs: int|None = None, skip_retrieval: bool = False, overwrite: bool = False, skip_existing: bool = False):
    runner = build_runner(filter)

    if group_run_id is None:
        group_run_id = next_group_run_id(runner.generation_runs_directory)

    click.echo(f"Group run ID: {group_run_id}")

    if number_of_runs is None:
        number_of_runs = 1

    asyncio.run(
        runner.generate(
            group_run_id=group_run_id,
            number_of_runs=number_of_runs,
            skip_existing=skip_existing,
            overwrite=overwrite,
            skip_retrieval=skip_retrieval,
        )
    )


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
def run_qa_eval(filter: str, group_run_id: str | None = None):
    runner = build_runner(filter)
    if group_run_id is None:
        group_run_id = get_most_recent_group_run_id(runner.generation_runs_directory)
        if group_run_id is None:
            raise click.ClickException("No generation runs found. Run 'run-qa-gen' first.")

    click.echo(f"Group run ID: {group_run_id}")

    asyncio.run(runner.evaluate(group_run_id=group_run_id))


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
@click.option("--number-of-runs", default=None, type=int)
@click.option("--skip-retrieval", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing run files")
@click.option("--skip-existing", is_flag=True, default=False, help="Skip test cases with existing output files")
def run_qa(filter: str, group_run_id: str | None = None, number_of_runs: int | None = None, skip_retrieval: bool = False, overwrite: bool = False, skip_existing: bool = False):
    """Generate and evaluate in one step."""

    click.echo(f"Group run ID: {group_run_id}")

    runner = build_runner(filter)

    if group_run_id is None:
        group_run_id = next_group_run_id(runner.generation_runs_directory)

    if number_of_runs is None:
        number_of_runs = 1

    async def run():
        await runner.generate(
            group_run_id=group_run_id,
            number_of_runs=number_of_runs,
            skip_existing=skip_existing,
            overwrite=overwrite,
            skip_retrieval=skip_retrieval,
        )
        await runner.evaluate(group_run_id=group_run_id)

    asyncio.run(run())


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
@click.option("--skip-retrieval", is_flag=True, default=False)
def print_run_summary(filter: str = "*", group_run_id: str | None = None, skip_retrieval: bool = False):
    asyncio.run(_print_summary_of_run(filter, group_run_id))

@cli.command()
@click.argument("game-id", required=True)
@click.argument("query", required=True)
def ask(game_id: str, query: str) -> None:
    system = create_eval_system(["game_service", "qa_service"])

    async def _run():
        async with system.astart() as services:
            game_service: GameService = services["game_service"]
            qa_service: QAService = services["qa_service"]

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


@cli.command()
@click.argument("group-a", required=True)
@click.argument("group-b", required=True)
def compare_qa(group_a: str, group_b: str):
    """Compare evaluation results between two runs.

    Example:
        python -m meeplemate.eval compare-qa 2026-02-17 2026-02-17-2
    """
    from meeplemate.eval.analysis import compare_runs

    click.echo(f"Comparing: {group_a} -> {group_b}")
    click.echo()

    comparison = compare_runs(group_a, group_b)

    # Overall metrics
    click.echo("Overall Metrics:")
    click.echo("-" * 70)
    for m in comparison.overall:
        arrow = "+" if m.delta >= 0 else ""
        direction = "^" if m.delta > 0.01 else ("v" if m.delta < -0.01 else "=")
        click.echo(f"  {m.metric_name:25s} {m.mean_a:.3f} -> {m.mean_b:.3f}  ({arrow}{m.delta:.3f})  {direction}")
        click.echo(f"  {'':25s} pass: {m.pass_rate_a:.1%} -> {m.pass_rate_b:.1%}")
    click.echo()

    # Improvements
    if comparison.improvements:
        click.echo(f"Improvements ({len(comparison.improvements)}):")
        click.echo("-" * 70)
        for tc in comparison.improvements:
            click.echo(f"  {tc.test_case[:45]:45s} {tc.metric_name:20s} {tc.mean_a:.3f} -> {tc.mean_b:.3f}  (+{tc.delta:.3f})")
        click.echo()

    # Regressions
    if comparison.regressions:
        click.echo(f"Regressions ({len(comparison.regressions)}):")
        click.echo("-" * 70)
        for tc in comparison.regressions:
            click.echo(f"  {tc.test_case[:45]:45s} {tc.metric_name:20s} {tc.mean_a:.3f} -> {tc.mean_b:.3f}  ({tc.delta:.3f})")
        click.echo()

    if not comparison.improvements and not comparison.regressions:
        click.echo("No significant changes detected.")
        click.echo()


@cli.command()
@click.argument("group-run-id")
def token_usage(group_run_id: str):
    """Show average token usage per model per run for a generation group.

    Loads all run files under the group directory, sums tokens per LLM call
    (deduplicated by run ID to avoid double-counting), and prints averages.

    Example:
        python -m meeplemate.eval token-usage 2026-02-24-2
    """
    from collections import defaultdict
    from meeplemate.eval import extract_token_usage_from_run

    base_group_run_id, _ = parse_group_run_id(group_run_id)
    group_dir = get_eval_generation_runs_dir() / base_group_run_id

    if not group_dir.exists():
        raise click.ClickException(f"No runs found for: {group_run_id}")

    run_files = sorted(group_dir.glob("*.json"))
    if not run_files:
        raise click.ClickException(f"No .json files in {group_dir}")

    # model_name -> list of (input_tokens, output_tokens) per run file
    model_samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
    skipped = 0

    for run_file in run_files:
        try:
            run = load_persisted_run(run_file)
            usage = extract_token_usage_from_run(run)
            if not usage:
                skipped += 1
                continue
            for model_name, counts in usage.items():
                model_samples[model_name].append(
                    (counts["input_tokens"], counts["output_tokens"])
                )
        except Exception as e:
            click.echo(f"  Warning: skipping {run_file.name}: {e}", err=True)
            skipped += 1

    if not model_samples:
        click.echo("No token usage data found (run files may pre-date stream_usage fix).")
        return

    loaded = len(run_files) - skipped
    click.echo(f"Group: {base_group_run_id}  ({loaded} run files, {skipped} skipped)")
    click.echo()
    w = 55
    click.echo(f"{'Model':<{w}} {'Runs':>5} {'Avg In':>10} {'Avg Out':>10} {'Avg Total':>10} {'Avg Cost':>12}")
    click.echo("-" * (w + 52))
    for model_name, samples in sorted(model_samples.items()):
        n = len(samples)
        avg_in = sum(s[0] for s in samples) / n
        avg_out = sum(s[1] for s in samples) / n
        costs = LLM_COST_PER_1M_TOKENS_TABLE.get(model_name)
        if costs:
            avg_cost = (avg_in * costs["input"] + avg_out * costs["output"]) / 1_000_000
            cost_str = f"${avg_cost:.4f}"
        else:
            cost_str = "--"
        click.echo(f"{model_name:<{w}} {n:>5} {avg_in:>10.0f} {avg_out:>10.0f} {avg_in + avg_out:>10.0f} {cost_str:>12}")


if __name__ == "__main__":
    cli()