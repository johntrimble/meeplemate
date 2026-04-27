from deepeval.metrics import GEval, BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def get_correctness_metric(model) -> BaseMetric:
    correctness_metric = GEval(
        model=model,
        name="Correctness",
        criteria="Determine whether the actual output reaches the same general conclusion as the expected output. Do not penalize if one explores exceptions or edge cases not mentioned in the other. Focus on whether both outputs agree on the main point. IMPORTANT: Do not rely on the opening Yes/No word alone — read the full reasoning and conclusion to determine the actual position. A response may start with 'No' while its conclusion agrees with the expected output (e.g., 'No, they are not exempt' means the same as 'Yes, they must take the test'). Judge based on the substantive conclusion, not surface-level phrasing.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    )
    return correctness_metric