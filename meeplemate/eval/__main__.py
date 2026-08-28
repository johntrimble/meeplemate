from meeplemate import eval_logging

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
    resolve_run_file,
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

            run_file = resolve_run_file(get_test_run_file_path(eval_runs_dir, base_group_run_id, test_suite["name"], test_case["name"], run_number=run_number))
            if run_file is None:
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

                run_file = resolve_run_file(get_test_run_file_path(
                    get_eval_generation_runs_dir(),
                    parsed_base_id,
                    test_suite["name"],
                    test_case["name"],
                    run_number=run_number
                ))

                if run_file is None:
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
@click.argument("filter", required=False, default="*")
@click.option("--token-budget", default=15_000, type=int)
def retrieval(filter: str, token_budget: int):
    """Score retrieval alone against the evidence quotes in test_cases.yaml.

    No LLM calls: each test case's raw query goes straight to the chunk search
    service, and we ask whether the parent chunk containing the reference
    answer's quote came back. Runs three arm configurations so the lexical arm
    has to justify itself rather than just not hurting.
    """
    import logging

    from meeplemate.eval.retrieval import (
        CaseResult,
        load_retrieval_cases,
        resolve_gold_parents,
        run_case,
        summarise,
    )
    from meeplemate.search import build_chunk_search_service_2, get_chunk_id

    cases = load_retrieval_cases(filter)
    if not cases:
        raise click.ClickException(f"No test cases with evidence matched {filter!r}")

    async def _run():
        base = create_eval_system(
            names=["vector_store", "docstore", "tokenizer", "bm25_searcher",
                   "game_data_store", "game_version_store"]
        )
        async with base.astart() as svc:
            # Pull every parent chunk once per game, so gold resolution is a
            # local fuzzy match rather than N round trips.
            parents_by_game: dict[str, list[tuple[str, str, str, str]]] = {}
            manifests: dict[str, Any] = {}
            # The search service returns synthetic chunk ids, not docstore keys
            # (add_chunk_ids rewrites Document.id), so we need the translation
            # back or the gold set can never match.
            chunk_id_to_parent: dict[str, str] = {}
            for game_id in sorted({c.game_id for c in cases}):
                gkey = (await svc["game_version_store"].amget([game_id]))[0]
                manifests[game_id] = (await svc["game_data_store"].amget([gkey]))[0]
                prefix = f"{gkey}#"
                keys = [k async for k in svc["docstore"].ayield_keys(prefix=prefix)]
                docs = await svc["docstore"].amget(keys)
                parents_by_game[game_id] = [
                    (k, d.metadata.get("rulebook_name", ""), str(d.metadata.get("page_num", "")), d.page_content)
                    for k, d in zip(keys, docs) if d is not None
                ]
                for k, d in zip(keys, docs):
                    if d is not None:
                        chunk_id_to_parent[get_chunk_id(d)] = k
                click.echo(f"{game_id}: {len(parents_by_game[game_id])} parent chunks")

            await resolve_gold_parents(cases, parents_by_game)

            unresolved = [
                (c.name, e.rulebook, e.quote[:60])
                for c in cases for e in c.evidence if not e.resolved
            ]
            if unresolved:
                click.echo(f"\n{len(unresolved)} evidence entries did not resolve to any parent "
                           f"(excluded from scoring — check for OCR drift or a bad quote):")
                for name, rulebook, quote in unresolved[:10]:
                    click.echo(f"   {name}: [{rulebook}] {quote}...")

            # Page-number drift is worth seeing but must not gate the gold set.
            drift = [
                (c.name, e.cited_page, sorted(e.found_pages))
                for c in cases for e in c.evidence
                if e.resolved and e.cited_page and e.cited_page not in e.found_pages
            ]
            if drift:
                click.echo(f"\n{len(drift)} evidence entries were found on a different page "
                           f"than cited (diagnostic only):")
                for name, cited, found in drift[:10]:
                    click.echo(f"   {name}: cited p.{cited}, found p.{','.join(found)}")

            # Quiet the per-query retrieval logging: useful in production, pure
            # noise when the point is the summary table.
            structlog.configure(
                wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
            )

            configs = {
                "vector only": {"bm25": None},
                "bm25 only": {"bm25": svc["bm25_searcher"], "vector_top_k": 0},
                "fused": {"bm25": svc["bm25_searcher"]},
            }
            per_config: dict[str, list[CaseResult]] = {}
            for label, kwargs in configs.items():
                search = build_chunk_search_service_2(
                    vectorstore=svc["vector_store"],
                    docstore=svc["docstore"],
                    tokenizer=svc["tokenizer"],
                    default_token_budget=token_budget,
                    **kwargs,
                )
                results = []
                for case in cases:
                    ids, tokens = await run_case(
                        search, manifests[case.game_id], case, token_budget,
                        chunk_id_to_parent=chunk_id_to_parent,
                    )
                    results.append(CaseResult(case=case, returned_ids=ids, tokens_used=tokens))
                per_config[label] = results

            # Attribute using each arm's own single-arm run, so "found by
            # vector" means the vector arm actually surfaced it rather than
            # "was returned and BM25 didn't also have it".
            for fused, vec, bm25 in zip(
                per_config["fused"], per_config["vector only"], per_config["bm25 only"]
            ):
                fused.vector_ids = set(vec.returned_ids)
                fused.bm25_ids = set(bm25.returned_ids)

            # Recall is post-token-budget, so it already reflects ranking: the
            # budget truncates in relevance order. There is deliberately no
            # recall@k — the service re-sorts its output into document order
            # before returning, so position in that list is not rank.
            click.echo()
            click.echo(f"{'config':<12} {'recall':>7} {'all gold':>9} {'chunks':>7} {'tokens':>7}")
            for label, results in per_config.items():
                s = summarise(results)
                click.echo(
                    f"{label:<12} {s['recall']:>7.1%} "
                    f"{s['full_recall_cases']:>4}/{s['cases']:<4} "
                    f"{s['chunks']:>7.1f} {s['tokens']:>7.0f}"
                )

            summary = summarise(per_config["fused"])
            a = summary["attribution"]
            click.echo(
                f"\ngold parents found by: vector_only={a['vector_only']} "
                f"bm25_only={a['bm25_only']} both={a['both']} neither={a['neither']}"
            )
            click.echo(f"scored {summary['cases']} cases, {summary['unresolved_cases']} unscorable")

    asyncio.run(_run())


@cli.command("mine-questions")
@click.argument("game-id", required=True)
@click.option("--run-id", default=None,
              help="Output file stem. Defaults to today's date, auto-incremented.")
@click.option("--limit", default=None, type=int,
              help="Number of seed chunks to process. Default: every parent chunk.")
@click.option("--offset", default=0, type=int,
              help="Index into the deterministic seed order to start at.")
@click.option("--concepts-per-seed", default=2, type=int)
@click.option("--questions-per-seed", default=5, type=int)
@click.option("--max-question-words", default=25, type=int,
              help="Length target rendered into the generation prompt. Boardbarian is used "
                   "mostly from a phone, and the hand-written cases in test_cases.yaml have "
                   "a median of 13 words. Advisory: the model is asked, not clipped.")
@click.option("--max-answer-tokens", default=6144, type=int,
              help="Token cap for the generation call. The config default (3072) is "
                   "tuned for a QA answer and truncates reasoning plus N questions. "
                   "Raise well above this for a thinking model.")
@click.option("--max-concept-tokens", default=8192, type=int,
              help="Token cap for the concept-extraction call. Non-thinking needs ~100; "
                   "thinking measured at ~2500 on this endpoint, and a seed whose trace "
                   "runs past the cap is lost entirely.")
@click.option("--dense-k", default=40, type=int,
              help="Dense candidates per concept. Effectively capped at 40 by the index's "
                   "hnsw.ef_search default, so raising this alone does not widen the arm.")
@click.option("--bm25-k", default=30, type=int, help="Lexical candidates per concept.")
@click.option("--rrf-k", default=60.0, type=float)
@click.option("--per-concept-cap", default=6, type=int)
@click.option("--max-context-chunks", default=12, type=int,
              help="Chunks in the generation context, seed included. Keep it small.")
@click.option("--adjacency-radius", default=1, type=int,
              help="Drop retrieved chunks within N positions of the seed in corpus order.")
@click.option("--min-seed-chars", default=200, type=int,
              help="Skip parent chunks shorter than this (title pages, credits, tables).")
@click.option("--dupe-threshold", default=0.92, type=float)
@click.option("--concurrency", default=4, type=int)
@click.option("--config-file", default=None, type=click.Path(path_type=Path),
              help="Config YAML to use instead of $MM_CONFIG_FILE, e.g. config-mining.yaml "
                   "(thinking on, large output budget).")
@click.option("--endpoint", default=None,
              help="Override the chat endpoint for this run.")
@click.option("--output-dir", default=None, type=click.Path(path_type=Path),
              help="Where run files land. Default: <project>/eval_gen.")
@click.option("--resume", is_flag=True, default=False,
              help="Skip seeds already present in the run file.")
@click.option("--dedupe-only", is_flag=True, default=False,
              help="Recompute dedupe over an existing run file. Makes no LLM calls.")
@click.option("--drop-duplicates", is_flag=True, default=False,
              help="Omit near-duplicates from the output instead of annotating them.")
@click.option("--include-context-text", is_flag=True, default=False,
              help="Embed chunk text in the YAML. For single-seed debugging only.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the seed list and corpus stats. Makes no LLM calls.")
def mine_questions(game_id: str, run_id: str | None, limit: int | None, offset: int,
                   concepts_per_seed: int, questions_per_seed: int,
                   max_question_words: int,
                   max_answer_tokens: int, max_concept_tokens: int, dense_k: int,
                   bm25_k: int, rrf_k: float, per_concept_cap: int,
                   max_context_chunks: int, adjacency_radius: int, min_seed_chars: int,
                   dupe_threshold: float, concurrency: int,
                   config_file: Path | None, endpoint: str | None,
                   output_dir: Path | None, resume: bool, dedupe_only: bool,
                   drop_duplicates: bool, include_context_text: bool, dry_run: bool):
    """Mine candidate eval questions from a game's rulebook chunks.

    Steps 1-2 of eval-corpus generation: for each parent chunk, name two rule
    concepts it leaves unresolved, retrieve those concepts across the corpus,
    assemble a small multi-chunk context, and generate candidate questions from
    it. Then flag near-duplicates against the batch and test_cases.yaml.

    Answer generation, answer-based filtering and human review are downstream and
    manual — this writes a review queue, not test cases.

    Retrieval here deliberately uses MINING settings rather than production ones:
    large k, no adaptive-k, no token budget, no document-order re-sort. Assembling
    context behind the production cutoffs would restrict the eval set to questions
    production retrieval already finds.
    """
    import logging
    import os

    import numpy as np

    from meeplemate.config import Config
    from meeplemate.eval.mining import (
        EXTRACT_CONCEPTS_PROMPT,
        GENERATE_CANDIDATES_PROMPT,
        ExtractedConcepts,
        GeneratedCandidates,
        MiningDeps,
        MiningParams,
        SeedResult,
        build_run_document,
        candidate_id,
        dump_run_yaml,
        eval_gen_run_path,
        load_existing_questions,
        load_parents,
        load_run_document,
        dedupe_candidates,
        mine_seed,
        next_run_id,
        seed_ids_in_document,
    )

    if output_dir is None:
        output_dir = get_eval_and_generation_output_dir().parent.parent / "eval_gen"
    game_dir = Path(output_dir) / game_id

    async def _run():
        # MM_CONFIG_FILE is read by YamlConfigSettingsSource at Config()
        # construction, so the override has to land before that. There is no
        # layering: the named file replaces config-dev.yaml wholesale rather
        # than merging with it.
        if config_file:
            resolved = Path(config_file).resolve()
            if not resolved.exists():
                raise click.ClickException(f"Config file not found: {resolved}")
            os.environ["MM_CONFIG_FILE"] = str(resolved)
        cfg = Config()
        names = ["docstore", "vector_store", "bm25_searcher", "chat_model",
                 "embedding_model", "_embedding_model",
                 "game_data_store", "game_version_store"]
        system = create_eval_system(names=names, config=cfg)
        # create_eval_system rewrites config.chat.models[0].endpoint before
        # returning, so an override has to be applied to the same Config object
        # *after* the call. Component factories run at astart(), not here, which
        # is what makes the late mutation land. Setting it earlier is discarded.
        if endpoint:
            cfg.chat.models[0].endpoint = endpoint

        async with system.astart() as svc:
            gkey = (await svc["game_version_store"].amget([game_id]))[0]
            if gkey is None:
                raise click.ClickException(f"No current game version for game_id {game_id!r}")
            manifest = (await svc["game_data_store"].amget([gkey]))[0]
            if manifest is None:
                raise click.ClickException(f"No manifest for game key {gkey!r}")
            game_version = manifest["game_version"]
            game_name = manifest.get("name", game_id)

            parents = await load_parents(svc["docstore"], gkey)
            if not parents:
                raise click.ClickException(f"No parent chunks found for {game_id!r}")
            parents_by_id = {p.parent_id: p for p in parents}

            resolved_run_id = run_id or next_run_id(game_dir)
            run_path = eval_gen_run_path(output_dir, game_id, resolved_run_id)

            existing = load_existing_questions()
            embeddings = svc["_embedding_model"]

            async def _dedupe(results: Sequence[SeedResult]) -> dict:
                """Embed and compare. Symmetric, so aembed_documents both sides.

                embedding_model is instruction-wrapped for retrieval and prepends
                a query instruction on aembed_query only; mixing a query vector
                with a document vector gives similarities that look plausible and
                are wrong. _embedding_model is the raw model.
                """
                ids, questions = [], []
                for r in results:
                    for i, q in enumerate(r.questions):
                        ids.append(candidate_id(game_id, r.seed_ordinal, i))
                        questions.append(q)
                if not ids:
                    return {}
                cand_vecs = np.array(await embeddings.aembed_documents(questions))
                exist_vecs = (
                    np.array(await embeddings.aembed_documents([e.question for e in existing]))
                    if existing else np.zeros((0, cand_vecs.shape[1]))
                )
                verdicts = dedupe_candidates(
                    ids, questions, cand_vecs, existing, exist_vecs,
                    game_id=game_id, threshold=dupe_threshold,
                )
                return dict(zip(ids, verdicts))

            # --dedupe-only: reduce over an existing file, no LLM calls. Retuning
            # a threshold should not cost a full sweep of generations.
            if dedupe_only:
                if not run_path.exists():
                    raise click.ClickException(f"No run file at {run_path}")
                doc = load_run_document(run_path)
                ids = [c["id"] for c in doc.get("candidates") or []]
                questions = [c["question"] for c in doc.get("candidates") or []]
                if not ids:
                    raise click.ClickException(f"No candidates in {run_path}")
                click.echo(f"Re-deduping {len(ids)} candidates at threshold {dupe_threshold}")
                cand_vecs = np.array(await embeddings.aembed_documents(questions))
                exist_vecs = (
                    np.array(await embeddings.aembed_documents([e.question for e in existing]))
                    if existing else np.zeros((0, cand_vecs.shape[1]))
                )
                verdicts = dedupe_candidates(
                    ids, questions, cand_vecs, existing, exist_vecs,
                    game_id=game_id, threshold=dupe_threshold,
                )
                by_id = dict(zip(ids, verdicts))
                for candidate in doc.get("candidates") or []:
                    v = by_id[candidate["id"]]
                    candidate["dedupe"] = {
                        "status": v.status,
                        "max_similarity": v.max_similarity,
                        "nearest": (
                            {"kind": v.nearest_kind, "ref": v.nearest_ref,
                             "question": v.nearest_question, "similarity": v.max_similarity}
                            if v.nearest_ref is not None else None
                        ),
                        "cross_game_max_similarity": v.cross_game_max_similarity,
                        "near_duplicates": [
                            {"kind": k, "ref": r, "similarity": round(s, 4)}
                            for k, r, s in v.near_duplicates
                        ],
                    }
                doc["run"].setdefault("params", {})["dupe_threshold"] = dupe_threshold
                if drop_duplicates:
                    doc["candidates"] = [
                        c for c in doc["candidates"] if c["dedupe"]["status"] == "unique"
                    ]
                dump_run_yaml(doc, run_path)
                _report(verdicts, doc, run_path)
                return

            # Seed selection: deterministic corpus order, so --offset is stable
            # and a resumed run picks up where it left off.
            eligible = [p for p in parents if len(p.content) >= min_seed_chars]
            skipped_short = len(parents) - len(eligible)
            numbered = list(enumerate(eligible))[offset:]
            if limit is not None:
                numbered = numbered[:limit]

            already: set[str] = set()
            if resume and run_path.exists():
                already = seed_ids_in_document(load_run_document(run_path))
                numbered = [(i, p) for i, p in numbered if p.parent_id not in already]

            click.echo(f"{game_id}: {len(parents)} parent chunks, {skipped_short} below "
                       f"--min-seed-chars, {len(numbered)} seeds to process")
            if resume and already:
                click.echo(f"  resuming: {len(already)} seeds already in {run_path.name}")

            if dry_run:
                for i, p in numbered[:20]:
                    click.echo(f"  [{i:4d}] {p.rulebook_name} p.{p.page_num or p.page_ordinal} "
                               f"({len(p.content)} chars)")
                if len(numbered) > 20:
                    click.echo(f"  ... and {len(numbered) - 20} more")
                return

            if not numbered:
                raise click.ClickException("No seeds to process")

            # max_tokens has to go through with_structured_output; binding it on
            # the model first is silently discarded (see qa_graph.py). The config
            # default is 3072, tuned for a QA answer — reasoning plus N questions
            # runs past it and the seed dies with LengthFinishReasonError.
            concept_chain = EXTRACT_CONCEPTS_PROMPT | svc["chat_model"].with_structured_output(
                ExtractedConcepts, max_tokens=max_concept_tokens
            )
            question_chain = GENERATE_CANDIDATES_PROMPT | svc["chat_model"].with_structured_output(
                GeneratedCandidates, max_tokens=max_answer_tokens
            )
            deps = MiningDeps(
                vector_store=svc["vector_store"], bm25=svc["bm25_searcher"],
                concept_chain=concept_chain, question_chain=question_chain,
                ordered_parents=parents, parents_by_id=parents_by_id,
                game_id=game_id, game_name=game_name, game_version=game_version,
            )
            params = MiningParams(
                concepts_per_seed=concepts_per_seed, questions_per_seed=questions_per_seed,
                dense_k=dense_k, bm25_k=bm25_k, rrf_k=rrf_k,
                per_concept_cap=per_concept_cap, max_context_chunks=max_context_chunks,
                adjacency_radius=adjacency_radius,
                max_question_words=max_question_words,
            )
            run_meta = {
                "run_id": resolved_run_id, "game_id": game_id, "game_version": game_version,
                "status": "partial",
                "params": {
                    "chat_model": cfg.chat.models[0].model_name,
                    "endpoint": cfg.chat.models[0].endpoint,
                    "config_file": os.environ.get("MM_CONFIG_FILE", ""),
                    # Thinking is invisible in the output (the endpoint does not
                    # surface reasoning_content through structured output), so
                    # this flag is the only record that a run used it.
                    "enable_thinking": bool(
                        (cfg.chat.models[0].chat_template_kwargs or {}).get("enable_thinking")
                    ),
                    "model_max_new_tokens": cfg.chat.models[0].max_new_tokens,
                    "embedding_model": cfg.embedding.model,
                    "concepts_per_seed": concepts_per_seed,
                    "questions_per_seed": questions_per_seed,
                    "max_question_words": max_question_words,
                    "max_answer_tokens": max_answer_tokens,
                    "max_concept_tokens": max_concept_tokens,
                    "dense_k": dense_k, "bm25_k": bm25_k, "rrf_k": rrf_k,
                    "per_concept_cap": per_concept_cap,
                    "max_context_chunks": max_context_chunks,
                    "adjacency_radius": adjacency_radius,
                    "min_seed_chars": min_seed_chars,
                    "dupe_threshold": dupe_threshold,
                    "seed_offset": offset, "seed_limit": limit,
                    "concurrency": concurrency,
                },
                "seeds": {"total_parents": len(parents), "skipped_short": skipped_short},
            }

            # Quiet structlog so the per-seed progress lines stay readable; the
            # retrieval command does the same for its summary table.
            structlog.configure(
                wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
            )

            sem = asyncio.Semaphore(concurrency)

            async def _one(seed, ordinal):
                async with sem:
                    return await mine_seed(seed, ordinal, deps=deps, params=params)

            tasks = [asyncio.create_task(_one(p, i)) for i, p in numbered]
            results: list[SeedResult] = []
            # Preserve anything already on disk so --resume accumulates rather
            # than replacing. Prior candidates are re-emitted verbatim.
            prior_doc = load_run_document(run_path) if (resume and run_path.exists()) else None

            done = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                done += 1
                status = "ok" if result.ok else f"SKIP ({result.error})"
                # concept overlap: 1.0 means both concepts retrieved the same
                # chunks, so the context is one cluster instead of two.
                click.echo(f"  [{done}/{len(tasks)}] seed {result.seed_ordinal:4d} "
                           f"{result.seed.rulebook_name[:24]:24s} "
                           f"{len(result.questions)} questions  "
                           f"overlap={result.concept_overlap:.2f}  {status}")
                # Flush after every seed. The whole file is rewritten atomically,
                # which is cheap without chunk text and keeps one file format.
                doc = build_run_document(
                    run_meta, sorted(results, key=lambda r: r.seed_ordinal), {},
                    include_context_text=include_context_text,
                )
                if prior_doc:
                    doc["candidates"] = (prior_doc.get("candidates") or []) + doc["candidates"]
                dump_run_yaml(doc, run_path)

            if all(r.error is not None for r in results):
                raise click.ClickException(
                    f"Every seed failed. First error: {results[0].error}"
                )

            results.sort(key=lambda r: r.seed_ordinal)
            verdicts = await _dedupe(results)
            run_meta["status"] = "complete"
            doc = build_run_document(
                run_meta, results, verdicts,
                include_context_text=include_context_text,
                drop_duplicates=drop_duplicates,
            )
            if prior_doc:
                doc["candidates"] = (prior_doc.get("candidates") or []) + doc["candidates"]
            dump_run_yaml(doc, run_path)
            _report(list(verdicts.values()), doc, run_path)

    def _report(verdicts, doc, run_path: Path) -> None:
        from meeplemate.eval.mining import length_histogram, similarity_histogram

        counts = {"unique": 0, "near_candidate": 0, "near_existing": 0}
        for v in verdicts:
            counts[v.status] += 1
        click.echo()
        click.echo(f"candidates: {len(verdicts)}  unique={counts['unique']} "
                   f"near_candidate={counts['near_candidate']} "
                   f"near_existing={counts['near_existing']}")
        click.echo("max_similarity histogram:")
        for lo, hi, n in similarity_histogram(verdicts):
            if n:
                click.echo(f"  {lo:.1f}-{hi:.1f} {'#' * min(n, 60)} {n}")
        questions = [c["question"] for c in doc.get("candidates") or ()]
        if questions:
            words = sorted(len(q.split()) for q in questions)
            median = words[len(words) // 2]
            click.echo(f"\nquestion length: median {median} words "
                       f"(test_cases.yaml median is 13)")
            for label, n in length_histogram(questions):
                if n:
                    click.echo(f"  {label:>6} {'#' * min(n, 60)} {n}")
        click.echo(f"\nwrote {run_path}")

    asyncio.run(_run())


if __name__ == "__main__":
    cli()