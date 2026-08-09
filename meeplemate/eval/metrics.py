from typing import TypedDict

from deepeval.metrics import GEval, BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from langchain_core.tracers import Run

from meeplemate.eval import (
    QuoteCounts,
    QuoteStatsCounts,
    RunawayGenerationCounts,
    count_llm_runaway_generations,
    count_quote_stats,
    count_quotes_and_quote_errors,
)


QUOTE_COUNT_METRIC_KEY = "quote_counts"
QUOTE_STATS_METRIC_KEY = "quote_stats"
RUNAWAY_GENERATION_METRIC_KEY = "runaway_generation_counts"


def add_quote_stats(test_case: LLMTestCase, run: Run) -> None:
    if test_case.additional_metadata is None:
        test_case.additional_metadata = {}
    test_case.additional_metadata[QUOTE_STATS_METRIC_KEY] = count_quote_stats(run)


def get_quote_stats(test_case: LLMTestCase) -> QuoteStatsCounts:
    metadata = test_case.additional_metadata or {}
    stats = metadata.get(QUOTE_STATS_METRIC_KEY)
    if stats is None:
        raise ValueError("Quote stats not found in test case metadata. Make sure to run add_quote_stats first.")
    return stats


def add_quote_counts(test_case: LLMTestCase, run: Run) -> None:
    if test_case.additional_metadata is None:
        test_case.additional_metadata = {}
    test_case.additional_metadata[QUOTE_COUNT_METRIC_KEY] = count_quotes_and_quote_errors(run)


def get_quote_counts(test_case: LLMTestCase) -> QuoteCounts:
    metadata = test_case.additional_metadata or {}
    counts = metadata.get(QUOTE_COUNT_METRIC_KEY)
    if counts is None:
        raise ValueError("Quote counts not found in test case metadata. Make sure to run add_quote_counts first.")
    return counts


def add_runaway_generation_counts(test_case: LLMTestCase, run: Run) -> None:
    if test_case.additional_metadata is None:
        test_case.additional_metadata = {}
    runaway_counts = count_llm_runaway_generations(run)
    test_case.additional_metadata[RUNAWAY_GENERATION_METRIC_KEY] = runaway_counts


def get_runaway_generation_counts(test_case: LLMTestCase) -> RunawayGenerationCounts:
    metadata = test_case.additional_metadata or {}
    counts = metadata.get(RUNAWAY_GENERATION_METRIC_KEY)
    if counts is None:
        raise ValueError("Runaway generation counts not found in test case metadata. Make sure to run add_runaway_generation_counts first.")
    return counts


def get_correctness_metric(model) -> BaseMetric:
    correctness_metric = GEval(
        model=model,
        name="Correctness",
        criteria="Determine whether the actual output reaches the same general conclusion as the expected output. Do not penalize if one explores exceptions or edge cases not mentioned in the other. Focus on whether both outputs agree on the main point. IMPORTANT: Do not rely on the opening Yes/No word alone — read the full reasoning and conclusion to determine the actual position. A response may start with 'No' while its conclusion agrees with the expected output (e.g., 'No, they are not exempt' means the same as 'Yes, they must take the test'). Judge based on the substantive conclusion, not surface-level phrasing.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    return correctness_metric


class ValidQuoteMetric(BaseMetric):
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        quote_counts = get_quote_counts(test_case)
        total_quotes = quote_counts["total_quotes"]
        quote_errors = quote_counts["quote_errors"]
        self.score = 1.0 - (quote_errors / total_quotes) if total_quotes > 0 else 1.0
        return self.score
    
    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None or self.score is None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self): # type: ignore
        return "Valid Quote Metric"


class RunawayGenerationsMetric(BaseMetric):
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        generation_counts = get_runaway_generation_counts(test_case)
        total_generations = generation_counts["total_generations"]
        runaway_generations = generation_counts["runaway_generations"]
        if total_generations == 0:
            score = 1.0
        else:
            score = 1.0 - (runaway_generations / total_generations)
        self.score = score
        return self.score
    
    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None or self.score is None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self): # type: ignore
        return "Runaway Generations Metric"


def _is_stale(stats: QuoteStatsCounts) -> bool:
    """True when the run has validation units that never reported quote_stats.

    Such a run predates the instrumentation, so its counts are all zeros, which would
    otherwise divide out to a perfect 1.0 and quietly pass. Fail loudly instead so a
    stale generation group is obvious rather than flattering.
    """
    return stats["unit_total"] > stats["units"]


def _fail_stale(metric: BaseMetric) -> float:
    metric.score = 0.0
    metric.error = (
        "Generation run predates quote_stats instrumentation; regenerate to score this."
    )
    metric.reason = metric.error
    return metric.score


class FirstPassQuoteValidityMetric(BaseMetric):
    """Share of quotes the model got right unaided, before any repair.

    This is the one to drive towards 1.0. A repaired quote still cost an LLM call and a
    validation cycle, so it counts against the score even though the user never sees the
    difference. Read alongside Quote Retention: the gap between the two is exactly
    how much work the repair loop is absorbing.
    """

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        stats = get_quote_stats(test_case)
        if _is_stale(stats):
            return _fail_stale(self)
        generated = stats["generated"]
        self.score = stats["first_pass_valid"] / generated if generated > 0 else 1.0
        needed_repair = generated - stats["first_pass_valid"]
        self.reason = (
            f"{stats['first_pass_valid']} of {generated} quotes valid first pass "
            f"({self.score:.1%}); {needed_repair} needed repair"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None or self.score is None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self): # type: ignore
        return "First-Pass Quote Validity"


class QuoteRetentionMetric(BaseMetric):
    """Share of generated quotes that survived into the answer, repaired or not.

    Retention is always >= first-pass validity, since only invalid quotes are ever
    dropped, so it carries a tighter threshold. A fall here means the user is losing
    evidence: either the model is producing more bad quotes, or repair has stopped
    rescuing them.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        stats = get_quote_stats(test_case)
        if _is_stale(stats):
            return _fail_stale(self)
        generated = stats["generated"]
        lost = stats["lost"]
        self.score = (generated - lost) / generated if generated > 0 else 1.0
        self.reason = (
            f"{generated - lost} of {generated} quotes retained ({self.score:.1%}); "
            f"{generated - stats['first_pass_valid']} invalid, "
            f"{stats['repaired']} repaired, {lost} dropped"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None or self.score is None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self): # type: ignore
        return "Quote Retention"