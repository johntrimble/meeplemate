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
              help="Number of seed chunks to process, taken contiguously from --offset. "
                   "Default: every parent chunk. For a partial sweep use --sample "
                   "instead; a contiguous window covers one chapter, not the game.")
@click.option("--offset", default=0, type=int,
              help="Index into the deterministic seed order to start at.")
@click.option("--sample", default=None, type=int,
              help="Take N seeds evenly spread across the --offset/--limit window "
                   "instead of the whole of it. Use this rather than --limit for a "
                   "partial sweep: a contiguous window covers one chapter, not the game.")
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
@click.option("--dupe-threshold", default=0.90, type=float,
              help="Cosine similarity at or above which a candidate is flagged as a "
                   "near-duplicate. 0.92 let obvious rewordings through -- measured on "
                   "the 2026-08-28 run, pairs at 0.90-0.92 were the same question. "
                   "Below ~0.88 it starts killing distinct questions about one rule.")
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
                   sample: int | None, concepts_per_seed: int, questions_per_seed: int,
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
        build_seed_record,
        candidate_id,
        dedupe_candidates,
        load_existing_questions,
        load_parents,
        mine_seed,
        sample_seeds,
    )
    from meeplemate.eval.eval_gen_layout import EvalGenLayout, next_group_run_id
    from meeplemate.eval.eval_gen_store import (
        dump_json,
        load_json,
        mined_parent_ids,
        read_seed_records,
        write_seed_record,
    )

    if output_dir is None:
        output_dir = get_eval_and_generation_output_dir().parent / "eval_gen"
    output_dir = Path(output_dir)
    resolved_run_id = run_id or next_group_run_id(output_dir)
    layout = EvalGenLayout(output_dir, resolved_run_id)

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
                records = read_seed_records(layout, game_id)
                pairs = [(c["id"], c["question"])
                         for r in records for c in r.get("candidates") or ()]
                if not pairs:
                    raise click.ClickException(
                        f"No mined candidates under {layout.mining_root(game_id)}")
                ids = [i for i, _ in pairs]
                questions = [q for _, q in pairs]
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
                # Only the reduce file is rewritten. The seed records that fed it
                # are immutable, which is what makes re-running this cheap and
                # safe while a mining run is still in flight.
                _write_dedupe(layout, game_id, dict(zip(ids, verdicts)), dupe_threshold)
                _report(verdicts, layout, [game_id])
                return

            # Seed selection: deterministic corpus order, so --offset is stable
            # and a resumed run picks up where it left off.
            eligible = [p for p in parents if len(p.content) >= min_seed_chars]
            skipped_short = len(parents) - len(eligible)
            numbered = list(enumerate(eligible))[offset:]
            if limit is not None:
                numbered = numbered[:limit]
            # Before the --resume filter, not after: sampling a pool that has
            # already had the done seeds removed would pick a different set on
            # every invocation and a resumed run would never converge.
            if sample is not None:
                numbered = sample_seeds(numbered, sample)

            already: set[str] = set()
            if resume:
                # A directory listing, not a parse of the whole accumulated
                # document -- and it matches on the parent id recorded inside
                # each record, because the ordinal in the filename shifts when
                # --min-seed-chars changes.
                already = mined_parent_ids(layout, game_id)
                numbered = [(i, p) for i, p in numbered if p.parent_id not in already]

            click.echo(f"{game_id}: {len(parents)} parent chunks, {skipped_short} below "
                       f"--min-seed-chars, {len(numbered)} seeds to process")
            if resume and already:
                click.echo(f"  resuming: {len(already)} seeds already recorded")

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
                    "seed_offset": offset, "seed_limit": limit, "seed_sample": sample,
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
                # One file, written once, in its final state. Failures get a
                # record too, so --resume skips them and --retry-failed is
                # "delete the error records and resume".
                write_seed_record(layout, game_id, build_seed_record(
                    game_id, result, include_context_text=include_context_text))

            if results and all(r.error is not None for r in results):
                raise click.ClickException(
                    f"Every seed failed. First error: {results[0].error}"
                )

            run_meta["status"] = "complete"
            run_meta["seeds"].update({
                "attempted": len(results),
                "succeeded": sum(1 for r in results if r.ok),
                "failed": sum(1 for r in results if not r.ok),
            })
            dump_json(run_meta, layout.mining_run_file(game_id))

            # Dedupe reduces over every candidate on disk, not just this
            # invocation's -- a resumed run must compare against what came
            # before it, or the second half never sees the first half's dupes.
            records = read_seed_records(layout, game_id)
            pairs = [(c["id"], c["question"])
                     for r in records for c in r.get("candidates") or ()]
            if pairs:
                ids = [i for i, _ in pairs]
                questions = [q for _, q in pairs]
                cand_vecs = np.array(await embeddings.aembed_documents(questions))
                exist_vecs = (
                    np.array(await embeddings.aembed_documents([e.question for e in existing]))
                    if existing else np.zeros((0, cand_vecs.shape[1]))
                )
                verdicts = dedupe_candidates(
                    ids, questions, cand_vecs, existing, exist_vecs,
                    game_id=game_id, threshold=dupe_threshold,
                )
                _write_dedupe(layout, game_id, dict(zip(ids, verdicts)), dupe_threshold)
                _report(verdicts, layout, [game_id])

    def _write_dedupe(layout, game_id: str, by_id: dict, threshold: float) -> None:
        dump_json({
            "game_id": game_id, "threshold": threshold,
            "candidates": {
                cid: {
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
                for cid, v in by_id.items()
            },
        }, layout.dedupe_file(game_id))

    def _report(verdicts, layout, games) -> None:
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

        questions = [c["question"] for g in games
                     for r in read_seed_records(layout, g)
                     for c in r.get("candidates") or ()]
        if questions:
            words = sorted(len(q.split()) for q in questions)
            click.echo(f"\nquestion length: median {words[len(words) // 2]} words "
                       f"(test_cases.yaml median is 13)")
            for label, n in length_histogram(questions):
                if n:
                    click.echo(f"  {label:>6} {'#' * min(n, 60)} {n}")
        click.echo(f"\nwrote {layout.mining_root(games[0])}")

    asyncio.run(_run())


@cli.command("answer-candidates")
@click.argument("game-id", required=True)
@click.option("--run-id", required=True,
              help="Group run id — the directory under data/eval_gen/ holding the "
                   "mined seeds to answer.")
@click.option("--runs", default=1, type=int,
              help="Answer each candidate N times and report the lowest pairwise "
                   "similarity as `consistency`. N>1 multiplies cost by N.")
@click.option("--limit", default=None, type=int,
              help="Answer at most this many candidates, taken contiguously. That is "
                   "one chapter's worth on a large game -- prefer --sample.")
@click.option("--sample", default=None, type=int,
              help="Answer N candidates spread evenly across the run instead of the "
                   "first N. Candidates are in seed order, so a contiguous --limit "
                   "concentrates on one region of one rulebook.")
@click.option("--concurrency", default=4, type=int)
@click.option("--include-duplicates", is_flag=True, default=False,
              help="Also answer candidates flagged as near-duplicates.")
@click.option("--include-evidence-text", is_flag=True, default=False,
              help="Embed the quoted passages in the YAML. For debugging one run.")
@click.option("--drop", is_flag=True, default=False,
              help="Remove candidates recommended 'drop' instead of annotating them.")
@click.option("--resume", is_flag=True, default=False,
              help="Skip candidates that already have a non-error answer block.")
@click.option("--config-file", default=None, type=click.Path(path_type=Path),
              help="Config YAML to use instead of $MM_CONFIG_FILE.")
@click.option("--endpoint", default=None, help="Override the chat endpoint for this run.")
@click.option("--output-dir", default=None, type=click.Path(path_type=Path),
              help="Where run files live. Default: <project>/eval_gen.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report how many candidates would be answered. Makes no LLM calls.")
def answer_candidates(game_id: str, run_id: str, runs: int, limit: int | None,
                      sample: int | None, concurrency: int, include_duplicates: bool,
                      include_evidence_text: bool, drop: bool, resume: bool,
                      config_file: Path | None, endpoint: str | None,
                      output_dir: Path | None, dry_run: bool):
    """Answer mined candidates with the QA pipeline and filter on the result.

    Steps 3-4 of eval-corpus generation. Runs each candidate question through
    the live Boardbarian agent, then judges the QUESTION using the answer as
    evidence about it -- an answer that hedges, or explains that a component
    does not exist, says the question was bad.

    Annotates in place, under `answer:` on each candidate. Nothing is deleted
    without --drop: this file is read by a person, and a filter's opinion is
    something they are allowed to disagree with.
    """
    import os

    from meeplemate.config import Config
    from meeplemate.eval.answering import (
        AnsweringDeps,
        JUDGE_PROMPT,
        JudgeResult,
        answer_and_judge,
        answer_block,
        apply_answers,
        candidates_to_answer,
        recommendation_summary,
        verdict_summary,
    )
    from meeplemate.eval.eval_gen_layout import EvalGenLayout
    from meeplemate.eval.eval_gen_store import (
        answered_candidate_ids,
        dump_json,
        iter_candidates,
        load_json,
        read_answers,
        write_answer_record,
    )

    # MM_CONFIG_FILE is read by YamlConfigSettingsSource at Config()
    # construction, so the override has to land before that.
    if config_file:
        resolved = Path(config_file).resolve()
        if not resolved.exists():
            raise click.ClickException(f"Config file not found: {resolved}")
        os.environ["MM_CONFIG_FILE"] = str(resolved)
    cfg = Config()

    if output_dir is None:
        output_dir = get_eval_and_generation_output_dir().parent / "eval_gen"
    layout = EvalGenLayout(Path(output_dir), run_id)
    if not layout.seeds_dir(game_id).exists():
        raise click.ClickException(
            f"No mined seeds at {layout.seeds_dir(game_id)} -- run mine-questions first")

    dedupe = (load_json(layout.dedupe_file(game_id), {}) or {}).get("candidates", {})
    done = answered_candidate_ids(layout, game_id) if resume else set()

    all_candidates = [c for c, _ in iter_candidates(layout, game_id)]
    pending = []
    for cand in all_candidates:
        if cand["id"] in done:
            continue
        if not include_duplicates:
            status = (dedupe.get(cand["id"]) or {}).get("status")
            if status not in (None, "unique"):
                continue
        pending.append(cand)
    if limit is not None:
        pending = pending[:limit]
    # After --limit, so the two compose the way they do for mining, and the
    # sampled set is stable across invocations for --resume.
    if sample is not None:
        from meeplemate.eval.mining import sample_seeds
        pending = sample_seeds(pending, sample)

    total = len(all_candidates)
    click.echo(f"{layout.mining_root(game_id)}: {total} candidates, "
               f"{len(pending)} to answer ({runs} QA run(s) + 1 judge call each)")
    if dry_run:
        for cand in pending[:20]:
            click.echo(f"  {cand['id']}  {cand['question']}")
        if len(pending) > 20:
            click.echo(f"  ... and {len(pending) - 20} more")
        return
    if not pending:
        raise click.ClickException("Nothing to answer")

    system = create_eval_system(
        names=["game_service", "qa_service", "chat_model", "_embedding_model"],
        config=cfg,
    )
    # create_eval_system hardcodes the endpoint AFTER accepting config, so an
    # override has to be applied to the same object after the call and before
    # astart(), where the component factories actually read it.
    if endpoint:
        cfg.chat.models[0].endpoint = endpoint

    async def _run():
        async with system.astart() as svc:
            manifest = await svc["game_service"].get_manifest(game_id)
            if manifest is None:
                raise click.ClickException(f"No manifest for game_id: {game_id}")

            deps = AnsweringDeps(
                qa_service=svc["qa_service"],
                manifest=manifest,
                judge_chain=JUDGE_PROMPT | svc["chat_model"].with_structured_output(
                    JudgeResult
                ),
                embeddings=svc["_embedding_model"],
                game_name=getattr(manifest, "name", None) or game_id,
            )

            sem = asyncio.Semaphore(concurrency)

            async def one(cand):
                async with sem:
                    return await answer_and_judge(deps, cand, runs=runs)

            done_n = 0
            for coro in asyncio.as_completed([one(c) for c in pending]):
                result = await coro
                done_n += 1
                block = answer_block(
                    result, include_evidence_text=include_evidence_text
                )
                # One file per candidate, written once. No accumulating document
                # to rewrite, so two shells can answer the same run at different
                # --limit windows without clobbering each other.
                write_answer_record(layout, game_id, result.candidate_id, {
                    "candidate_id": result.candidate_id,
                    "game_id": game_id,
                    "question": result.question,
                    "answer": block,
                })
                label = block.get("error") or (
                    f"{block['recommendation']:<6} {block['verdict']}"
                )
                click.echo(f"  [{done_n}/{len(pending)}] {result.candidate_id}  {label}")

            dump_json({
                "group_run_id": run_id, "game_id": game_id,
                "chat_model": cfg.chat.models[0].model_name,
                "endpoint": cfg.chat.models[0].endpoint,
                "runs": runs, "concurrency": concurrency,
                "answered": len(pending),
            }, layout.answers_run_file(game_id))

            answers = read_answers(layout, game_id)
            rec_counts: dict[str, int] = {}
            ver_counts: dict[str, int] = {}
            for a in answers.values():
                blk = a.get("answer") or {}
                key = blk.get("recommendation") or blk.get("status") or "?"
                rec_counts[key] = rec_counts.get(key, 0) + 1
                if blk.get("verdict"):
                    ver_counts[blk["verdict"]] = ver_counts.get(blk["verdict"], 0) + 1
            click.echo()
            click.echo("  ".join(f"{k}={v}" for k, v in sorted(rec_counts.items())))
            click.echo("  ".join(f"{k}={v}" for k, v in sorted(ver_counts.items())))
            click.echo(f"\nwrote {layout.answers_root(game_id)}")

    asyncio.run(_run())


@cli.command("review")
@click.argument("games", nargs=-1, required=True)
@click.option("--run-id", required=True,
              help="Group run id — the directory under data/eval_gen/.")
@click.option("--port", default=8765, type=int, help="Port to serve on.")
@click.option("--host", default="0.0.0.0",
              help="Interface to bind. Defaults to all interfaces so the page is "
                   "reachable from the host browser -- loopback inside the dev "
                   "container is reachable only from inside it. Nothing here "
                   "authenticates; pass 127.0.0.1 to restrict it.")
@click.option("--decisions", default=None, type=click.Path(path_type=Path),
              help="Decisions file. Default: "
                   "data/eval_gen/<run-id>/decisions/decisions.json")
@click.option("--eval-gen", default=None, type=click.Path(path_type=Path),
              help="Directory holding <run-id>/. Default: <project>/data/eval_gen")
@click.option("--title", default="Candidate Rules Questions", help="Page title.")
def review(games: tuple[str, ...], run_id: str, port: int, host: str,
           decisions: Path | None, eval_gen: Path | None, title: str):
    """Serve the candidate review page for a mined + answered run.

    Reads the mined seeds and answers for each game named under
    data/eval_gen/<run-id>/, and persists every verdict, answer flag, note and
    reference answer to that run's decisions/decisions.json as the reviewer
    works. Decisions are scoped to the run id, not to a game: one
    session spans every game in the run and candidate ids already carry theirs.

        mm-eval review munchkin one_deck_dungeon --run-id 2026-08-28
    """
    from meeplemate.eval.eval_gen_layout import EvalGenLayout
    from meeplemate.eval.review_page import build_page, load_run
    from meeplemate.eval.review_server import build_app, read_decisions, serve

    if eval_gen is None:
        eval_gen = get_eval_and_generation_output_dir().parent / "eval_gen"
    layout = EvalGenLayout(Path(eval_gen), run_id)

    for game in games:
        if not layout.seeds_dir(game).exists():
            raise click.ClickException(
                f"No mined seeds for {game!r} at {layout.seeds_dir(game)}")

    try:
        rows = load_run(layout, games)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    if not rows:
        raise click.ClickException(
            f"No answered candidates in run {run_id}. "
            f"Run `mm-eval answer-candidates` first."
        )

    decisions_file = Path(decisions) if decisions else layout.decisions_file()
    existing = read_decisions(decisions_file, run_id, len(rows))

    kept = sum(1 for r in rows if r["rec"] == "keep")
    click.echo(f"{len(rows)} candidates ({kept} keep) across {len(games)} game(s)")
    click.echo(f"decisions: {decisions_file}"
               + (f"  ({len(existing['decisions'])} already recorded)"
                  if existing["decisions"] else "  (new)"))
    shown = "localhost" if host in ("0.0.0.0", "::") else host
    click.echo(f"\n  http://{shown}:{port}\n")
    click.echo("Ctrl+C to stop.")

    app = build_app(
        # Re-read per request so a concurrent answer-candidates run shows up on
        # reload rather than needing the server restarted.
        render_page=lambda: build_page(
            load_run(layout, games), run_id=run_id, games=games, title=title
        ),
        decisions_file=decisions_file,
        run_id=run_id,
        total=len(rows),
    )
    serve(app, host=host, port=port)


@cli.command("promote")
@click.argument("run-id", required=True)
@click.option("--games", default=None,
              help="Comma-separated game ids. Default: every game in the run.")
@click.option("--out", default=None, type=click.Path(path_type=Path),
              help="Output YAML. Default: the run's dataset/test_cases.yaml. Pass "
                   "meeplemate/eval/test_cases/<run-id>.yaml to adopt it into the "
                   "tracked eval corpus.")
@click.option("--eval-gen", default=None, type=click.Path(path_type=Path),
              help="Directory holding <run-id>/. Default: <project>/data/eval_gen")
@click.option("--decisions", default=None, type=click.Path(path_type=Path),
              help="Decisions file. Default: the run's decisions/decisions.json")
@click.option("--include-rejected", is_flag=True, default=False,
              help="Also promote candidates the reviewer rejected. Off by default: "
                   "a correction may have been drafted before the verdict, and the "
                   "verdict is the decision that counts.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Report what would be written. Writes nothing.")
def promote(run_id: str, games: str | None, out: Path | None, eval_gen: Path | None,
            decisions: Path | None, include_rejected: bool, dry_run: bool):
    """Turn reviewed candidates into golden test cases.

    Step 4 of eval-corpus generation. Reads the run's decisions and answer
    records and writes one suite per game to
    meeplemate/eval/test_cases/<run-id>.yaml, which the eval loader picks up
    alongside the hand-written test_cases.yaml.

    Derived entirely from decisions.json, so re-running after a correction
    produces a clean diff rather than a hand-merge.

        mm-eval promote 2026-08-28
    """
    from meeplemate.eval.eval_gen_layout import EvalGenLayout
    from meeplemate.eval.eval_gen_store import load_json, read_answers
    from meeplemate.eval.promote import build_suites, evidence_stats
    from meeplemate.util import atomic_write_text

    if eval_gen is None:
        eval_gen = get_eval_and_generation_output_dir().parent / "eval_gen"
    layout = EvalGenLayout(Path(eval_gen), run_id)

    decisions_file = Path(decisions) if decisions else layout.decisions_file()
    doc = load_json(decisions_file)
    if not doc or not isinstance(doc.get("decisions"), list):
        raise click.ClickException(f"No decisions at {decisions_file}")
    rows = doc["decisions"]

    game_ids = ([g.strip() for g in games.split(",") if g.strip()] if games
                else sorted({r.get("game", "") for r in rows if r.get("game")}))
    answers_by_id: dict[str, dict] = {}
    for game in game_ids:
        answers_by_id.update(read_answers(layout, game))

    suites, skipped = build_suites(
        run_id, [r for r in rows if r.get("game") in game_ids],
        answers_by_id, include_rejected=include_rejected,
    )
    if not suites:
        raise click.ClickException(
            f"Nothing to promote from {decisions_file} — no candidates marked 'yes'.")

    stats = evidence_stats(suites)
    click.echo(f"{decisions_file}: {len(rows)} decisions")
    for suite in suites:
        n = len(suite["test_cases"])
        ev = sum(len(c["evidence"]) for c in suite["test_cases"])
        click.echo(f"  {suite['name']}: {n} cases, {ev} evidence quotes")
    click.echo(f"\n{stats['cases']} cases  "
               f"{stats['with_evidence']} with evidence  "
               f"{stats['evidence_quotes']} quotes  "
               f"{stats['reviewer_answers']} reviewer-written answers")
    if stats.get("dropped_quotes"):
        # Only numeric page citations parse; a quote cited "p. iv" is lost.
        click.echo(f"  WARNING: {stats['dropped_quotes']} blockquote(s) had an "
                   f"unparseable citation and were not written as evidence")
    if stats["with_evidence"] < stats["cases"]:
        # A case with no evidence still scores for correctness but is invisible
        # to `mm-eval retrieval`, which is half of what a case is worth.
        click.echo(f"  WARNING: {stats['cases'] - stats['with_evidence']} cases "
                   f"have no evidence and will not be scored by `mm-eval retrieval`")
    if skipped:
        click.echo(f"  skipped (no answer to promote): {', '.join(skipped)}")

    if out is None:
        out = layout.dataset_file()
    if dry_run:
        click.echo(f"\nwould write {out}")
        return

    from meeplemate.eval.promote import dump_suites

    dump_suites(suites, out, run_id=run_id, source=decisions_file)
    click.echo(f"\nwrote {out}")
    if out == layout.dataset_file():
        # data/ is gitignored, so this file is not part of the eval corpus yet.
        click.echo("\nThis is the run's dataset, not the tracked corpus. To adopt it:")
        click.echo(f"  cp {out} meeplemate/eval/test_cases/{run_id}.yaml")


if __name__ == "__main__":
    cli()