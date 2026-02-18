"""Analysis utilities for multi-run DeepEval results.

This module provides functions for loading, aggregating, and analyzing
evaluation results across multiple runs (e.g., __run001, __run002, etc.).
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


def get_eval_results_dir():
    """Get the base directory for evaluation results."""
    import meeplemate
    meeplemate_file = Path(meeplemate.__file__)
    if meeplemate_file.name == "__init__.py":
        meeplemate_file = meeplemate_file.parent
    project_dir = meeplemate_file.parent
    return project_dir / "data" / "evals" / "qa_evals"


def get_generation_runs_dir():
    """Get the base directory for generation runs."""
    import meeplemate
    meeplemate_file = Path(meeplemate.__file__)
    if meeplemate_file.name == "__init__.py":
        meeplemate_file = meeplemate_file.parent
    project_dir = meeplemate_file.parent
    return project_dir / "data" / "evals" / "generation_runs"


def find_run_numbers(base_group_run_id: str) -> list[int | None]:
    """Find all run numbers for a given group_run_id by scanning generation files.

    Searches the generation_runs directory for files with .run00X.json suffix.

    Args:
        base_group_run_id: Base identifier (e.g., "2026-01-25")

    Returns:
        List of run numbers. None represents the base run.
        Example: [None, 1, 2, 3] means base run + run001, run002, run003
    """
    gen_dir = get_generation_runs_dir()

    if not gen_dir.exists():
        return []

    run_numbers = set()

    # Check for .run00X.json files in the base directory
    base_path = gen_dir / base_group_run_id
    if base_path.exists() and base_path.is_dir():
        for file_path in base_path.glob("*.json"):
            filename = file_path.name

            # Check if it's a numbered run file (*.run00X.json)
            if ".run" in filename:
                # Extract run number: "test_suite__test_case.run001.json" -> "001"
                parts = filename.rsplit(".run", 1)
                if len(parts) == 2:
                    run_part = parts[1].replace(".json", "")
                    try:
                        run_num = int(run_part)
                        run_numbers.add(run_num)
                    except ValueError:
                        pass
            else:
                # Base run (no .runXXX suffix)
                run_numbers.add(None)

    # Sort: None (base) first, then numbered runs
    sorted_nums = sorted([r for r in run_numbers if r is not None])
    if None in run_numbers:
        return [None] + sorted_nums
    else:
        return sorted_nums


def find_run_groups(base_group_run_id: str) -> list[str]:
    """Find all run groups matching the pattern.

    Given a base group_run_id like "2026-01-25", finds all matching runs by
    searching for files with .run00X.json suffix in the generation_runs directory.

    Args:
        base_group_run_id: Base identifier (e.g., "2026-01-25")

    Returns:
        List of group_run_ids sorted in order. Uses __run00X format for API compatibility.

    Example:
        >>> find_run_groups("2026-01-25")
        ['2026-01-25', '2026-01-25__run001', '2026-01-25__run002', '2026-01-25__run003']
    """
    run_numbers = find_run_numbers(base_group_run_id)

    result = []
    for run_num in run_numbers:
        if run_num is None:
            result.append(base_group_run_id)
        else:
            # Keep __run format for API compatibility
            result.append(f"{base_group_run_id}__run{str(run_num).zfill(3)}")

    return result


def load_eval_results(group_run_id: str) -> list[dict]:
    """Load evaluation results for a given group_run_id.

    With the new single-directory structure, all runs are stored in one directory
    (e.g., qa_evals/2026-01-31/) and test cases include run numbers in their names
    (e.g., test_case__run001, test_case__run002).

    This function loads from the base directory and filters test cases by run number.

    Args:
        group_run_id: The group run identifier (e.g., "2026-01-25" or "2026-01-25__run001")

    Returns:
        List of parsed evaluation results (DeepEval format)
        Each result contains a "testCases" list with individual test results

    Example:
        >>> results = load_eval_results("2026-01-25__run001")
        >>> len(results)  # Number of evaluation files
        1
        >>> results[0].keys()
        dict_keys(['testCases', '_source_file', '_group_run_id'])
    """
    # Parse group_run_id to extract base ID and run number
    if "__run" in group_run_id:
        parts = group_run_id.split("__run")
        base_group_run_id = parts[0]
        run_number = int(parts[1])
        run_suffix = f"__run{str(run_number).zfill(3)}"
    else:
        base_group_run_id = group_run_id
        run_suffix = None

    # Load from the base directory
    eval_dir = get_eval_results_dir() / base_group_run_id

    if not eval_dir.exists():
        return []

    results = []

    # Load all files in the directory (DeepEval uses timestamp-based filenames)
    for file_path in eval_dir.iterdir():
        if file_path.is_file():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # Filter test cases by run number
                    if run_suffix is not None:
                        # Only include test cases that match this run
                        filtered_test_cases = [
                            tc for tc in data.get('testCases', [])
                            if tc.get('name', '').endswith(run_suffix)
                        ]
                        data['testCases'] = filtered_test_cases

                    # Add metadata about which file this came from
                    data['_source_file'] = file_path.name
                    data['_group_run_id'] = group_run_id
                    results.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue

    return results


def load_all_runs(base_group_run_id: str) -> list[dict]:
    """Load evaluation results for all runs matching the base group.

    Convenience function that combines find_run_groups and load_eval_results.

    Args:
        base_group_run_id: Base identifier (e.g., "2026-01-25")

    Returns:
        List of all evaluation results from all matching runs

    Example:
        >>> all_results = load_all_runs("2026-01-25")
        >>> # Returns results from __run001, __run002, __run003, etc.
    """
    run_groups = find_run_groups(base_group_run_id)

    if not run_groups:
        print(f"Warning: No runs found for {base_group_run_id}")
        return []

    all_results = []
    for group_id in run_groups:
        results = load_eval_results(group_id)
        all_results.extend(results)

    return all_results


def extract_test_cases(eval_results: list[dict]) -> pd.DataFrame:
    """Extract all test cases from evaluation results into a flat DataFrame.

    Converts the nested DeepEval JSON structure into a flat table where each
    row represents one test case from one evaluation run.

    Args:
        eval_results: List of evaluation results from load_eval_results or load_all_runs

    Returns:
        DataFrame with columns:
        - group_run_id: Which run group this came from
        - source_file: Which evaluation file this came from
        - test_case: Test case name
        - input: Input query
        - actual_output: Generated answer
        - expected_output: Reference answer (may be empty)
        - success: Overall pass/fail
        - run_duration: Time to run test case
        - metric_name: Name of metric (one row per metric)
        - metric_score: Score for that metric
        - metric_success: Pass/fail for that metric
        - metric_threshold: Threshold for that metric
        - metric_reason: Explanation of score
    """
    rows = []

    for result in eval_results:
        group_run_id = result.get('_group_run_id', 'unknown')
        source_file = result.get('_source_file', 'unknown')

        for test_case in result.get('testCases', []):
            # Extract common fields
            test_case_name = test_case.get('name', 'unknown')

            # Strip run suffix from test case name for aggregation
            # e.g., "test_case__run001" -> "test_case"
            if '__run' in test_case_name:
                test_case_name = test_case_name.rsplit('__run', 1)[0]

            base_data = {
                'group_run_id': group_run_id,
                'source_file': source_file,
                'test_case': test_case_name,
                'input': test_case.get('input', ''),
                'actual_output': test_case.get('actualOutput', ''),
                'expected_output': test_case.get('expectedOutput', ''),
                'success': test_case.get('success', False),
                'run_duration': test_case.get('runDuration', 0.0),
            }

            # Extract metrics - create one row per metric
            for metric in test_case.get('metricsData', []):
                row = base_data.copy()
                row.update({
                    'metric_name': metric.get('name', 'unknown'),
                    'metric_score': metric.get('score', 0.0),
                    'metric_success': metric.get('success', False),
                    'metric_threshold': metric.get('threshold', 0.0),
                    'metric_reason': metric.get('reason', ''),
                })
                rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_test_case(eval_results: list[dict]) -> pd.DataFrame:
    """Aggregate metrics across runs, grouped by test case.

    For each test case and metric combination, calculates summary statistics
    across all runs (mean, median, std, min, max, count).

    Args:
        eval_results: List of evaluation results from load_all_runs

    Returns:
        DataFrame with columns:
        - test_case: Test case name
        - metric_name: Metric name
        - count: Number of runs
        - mean: Mean score
        - median: Median score
        - std: Standard deviation
        - min: Minimum score
        - max: Maximum score
        - pass_rate: Percentage of runs that passed

    Example:
        >>> results = load_all_runs("2026-01-25")
        >>> summary = aggregate_by_test_case(results)
        >>> summary[summary['test_case'] == 'munchkin__curse_in_combat']
    """
    df = extract_test_cases(eval_results)

    if df.empty:
        return pd.DataFrame()

    # Group by test case and metric
    grouped = df.groupby(['test_case', 'metric_name'])

    aggregated = grouped.agg({
        'metric_score': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'metric_success': lambda x: x.sum() / len(x)  # Pass rate
    }).reset_index()

    # Flatten column names
    aggregated.columns = [
        'test_case', 'metric_name',
        'count', 'mean', 'median', 'std', 'min', 'max',
        'pass_rate'
    ]

    return aggregated


def get_test_case_details(eval_results: list[dict], test_case_name: str) -> pd.DataFrame:
    """Get all individual results for a specific test case across runs.

    Useful for drilling down into a specific test case to see how answers
    and scores varied across different runs.

    Args:
        eval_results: List of evaluation results from load_all_runs
        test_case_name: Name of test case to filter for

    Returns:
        DataFrame with all runs of that test case, including:
        - group_run_id, source_file
        - input, actual_output, expected_output
        - metric_name, metric_score, metric_success
        - run_duration

    Example:
        >>> results = load_all_runs("2026-01-25")
        >>> details = get_test_case_details(results, "munchkin__curse_in_combat")
        >>> # See all answers and scores for this test case
    """
    df = extract_test_cases(eval_results)

    if df.empty:
        return pd.DataFrame()

    return df[df['test_case'] == test_case_name].sort_values(
        ['group_run_id', 'metric_name']
    )


@dataclass
class EvalRunSummary:
    """Summary statistics for a multi-run evaluation."""
    base_group_run_id: str
    run_groups: list[str]
    num_test_cases: int
    num_runs: int
    metrics_summary: dict[str, dict]  # metric_name -> {mean, std, pass_rate, etc.}


def get_summary(base_group_run_id: str) -> EvalRunSummary:
    """Get high-level summary of multi-run evaluation.

    Args:
        base_group_run_id: Base identifier (e.g., "2026-01-25")

    Returns:
        EvalRunSummary with aggregate statistics

    Example:
        >>> summary = get_summary("2026-01-25")
        >>> print(f"Analyzed {summary.num_runs} runs")
        >>> print(f"Mean Answer Relevancy: {summary.metrics_summary['Answer Relevancy']['mean']:.2f}")
    """
    run_groups = find_run_groups(base_group_run_id)
    all_results = load_all_runs(base_group_run_id)
    df = extract_test_cases(all_results)

    if df.empty:
        return EvalRunSummary(
            base_group_run_id=base_group_run_id,
            run_groups=run_groups,
            num_test_cases=0,
            num_runs=0,
            metrics_summary={}
        )

    # Calculate per-metric summary
    metrics_summary = {}
    for metric_name in df['metric_name'].unique():
        metric_df = df[df['metric_name'] == metric_name]
        metrics_summary[metric_name] = {
            'mean': metric_df['metric_score'].mean(),
            'std': metric_df['metric_score'].std(),
            'median': metric_df['metric_score'].median(),
            'min': metric_df['metric_score'].min(),
            'max': metric_df['metric_score'].max(),
            'pass_rate': metric_df['metric_success'].sum() / len(metric_df),
            'threshold': metric_df['metric_threshold'].iloc[0] if len(metric_df) > 0 else 0.0,
        }

    return EvalRunSummary(
        base_group_run_id=base_group_run_id,
        run_groups=run_groups,
        num_test_cases=len(df['test_case'].unique()),
        num_runs=len(run_groups),
        metrics_summary=metrics_summary
    )


@dataclass
class MetricComparison:
    """Comparison of a single metric between two runs."""
    metric_name: str
    mean_a: float
    mean_b: float
    delta: float
    pass_rate_a: float
    pass_rate_b: float


@dataclass
class TestCaseComparison:
    """Comparison of a single test case between two runs."""
    test_case: str
    metric_name: str
    mean_a: float
    mean_b: float
    delta: float


@dataclass
class RunComparison:
    """Full comparison between two evaluation runs."""
    group_a: str
    group_b: str
    overall: list[MetricComparison]
    improvements: list[TestCaseComparison]
    regressions: list[TestCaseComparison]


def compare_runs(group_a: str, group_b: str) -> RunComparison:
    """Compare evaluation results between two group_run_ids.

    Args:
        group_a: First (baseline) group_run_id
        group_b: Second (new) group_run_id

    Returns:
        RunComparison with overall metric deltas and per-test-case changes
    """
    summary_a = get_summary(group_a)
    summary_b = get_summary(group_b)

    # Overall metric comparison
    all_metrics = set(summary_a.metrics_summary.keys()) | set(summary_b.metrics_summary.keys())
    overall = []
    for metric_name in sorted(all_metrics):
        stats_a = summary_a.metrics_summary.get(metric_name, {})
        stats_b = summary_b.metrics_summary.get(metric_name, {})
        mean_a = stats_a.get('mean', 0.0)
        mean_b = stats_b.get('mean', 0.0)
        overall.append(MetricComparison(
            metric_name=metric_name,
            mean_a=mean_a,
            mean_b=mean_b,
            delta=mean_b - mean_a,
            pass_rate_a=stats_a.get('pass_rate', 0.0),
            pass_rate_b=stats_b.get('pass_rate', 0.0),
        ))

    # Per-test-case comparison
    results_a = load_all_runs(group_a)
    results_b = load_all_runs(group_b)
    agg_a = aggregate_by_test_case(results_a)
    agg_b = aggregate_by_test_case(results_b)

    improvements = []
    regressions = []

    if not agg_a.empty and not agg_b.empty:
        merged = pd.merge(
            agg_a[['test_case', 'metric_name', 'mean']],
            agg_b[['test_case', 'metric_name', 'mean']],
            on=['test_case', 'metric_name'],
            suffixes=('_a', '_b'),
            how='inner',
        )
        merged['delta'] = merged['mean_b'] - merged['mean_a']

        for _, row in merged.iterrows():
            tc = TestCaseComparison(
                test_case=row['test_case'],
                metric_name=row['metric_name'],
                mean_a=row['mean_a'],
                mean_b=row['mean_b'],
                delta=row['delta'],
            )
            if row['delta'] > 0.01:
                improvements.append(tc)
            elif row['delta'] < -0.01:
                regressions.append(tc)

        improvements.sort(key=lambda x: x.delta, reverse=True)
        regressions.sort(key=lambda x: x.delta)

    return RunComparison(
        group_a=group_a,
        group_b=group_b,
        overall=overall,
        improvements=improvements,
        regressions=regressions,
    )


@dataclass
class TestCaseAnalysis:
    """Detailed analysis of a single test case across multiple runs."""
    test_case_name: str
    num_runs: int
    answers: list[str]
    scores: dict[str, list[float]]  # metric_name -> list of scores
    mean_scores: dict[str, float]  # metric_name -> mean
    std_scores: dict[str, float]  # metric_name -> std


def analyze_test_case(eval_results: list[dict], test_case_name: str) -> TestCaseAnalysis:
    """Analyze a specific test case in detail.

    Args:
        eval_results: List of evaluation results from load_all_runs
        test_case_name: Name of test case to analyze

    Returns:
        TestCaseAnalysis with detailed statistics

    Example:
        >>> results = load_all_runs("2026-01-25")
        >>> analysis = analyze_test_case(results, "munchkin__curse_in_combat")
        >>> print(f"Generated {len(analysis.answers)} different answers")
        >>> print(f"Score variance: {analysis.std_scores['Answer Relevancy']:.2f}")
    """
    df = get_test_case_details(eval_results, test_case_name)

    if df.empty:
        return TestCaseAnalysis(
            test_case_name=test_case_name,
            num_runs=0,
            answers=[],
            scores={},
            mean_scores={},
            std_scores={}
        )

    # Get unique answers
    answers = df['actual_output'].unique().tolist()

    # Calculate per-metric statistics
    scores = {}
    mean_scores = {}
    std_scores = {}

    for metric_name in df['metric_name'].unique():
        metric_df = df[df['metric_name'] == metric_name]
        metric_scores = metric_df['metric_score'].tolist()
        scores[metric_name] = metric_scores
        mean_scores[metric_name] = np.mean(metric_scores)
        std_scores[metric_name] = np.std(metric_scores)

    return TestCaseAnalysis(
        test_case_name=test_case_name,
        num_runs=len(df['group_run_id'].unique()),
        answers=answers,
        scores=scores,
        mean_scores=mean_scores,
        std_scores=std_scores
    )
