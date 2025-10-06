import copy
from pathlib import Path
import re
import string
from typing import Any, Callable, Generic, Mapping, Optional, Protocol, Sequence, TextIO, Tuple, TypeVar, TypedDict, Literal, NotRequired, Unpack, cast
from langchain_core.runnables import (chain, Runnable, RunnableLambda, RunnablePassthrough)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.language_models import BaseChatModel, LanguageModelInput

from pyparsing import itemgetter
from sqlalchemy import case
import yaml


Grade = Literal["CORRECT", "INCORRECT", "UNKNOWN"]


class QASet(TypedDict):
    game_id: str
    grader_examples: Sequence["GradedQATestCase"]
    question_answers: Sequence["QATestCase"]


class QATestCase(TypedDict):
    question: str
    answers: Sequence[str]
    examples: NotRequired[Sequence["GradedQATestCase"]]


class QATestCaseWithPrediction(QATestCase):
    prediction: str


class ScoreResult(TypedDict):
    score: int
    grade: Grade
    reasoning: str
    ratio_correct: NotRequired[float]


class GradedQATestCase(QATestCaseWithPrediction, ScoreResult):
    pass


def load_question_answer_sets(path: str | Path) -> Sequence[QASet]:
    path = Path(path)
    data = yaml.safe_load(path.read_text())

    # Add any missing default fields
    for qa_set in data["question_answer_sets"]:
        for grader_example in qa_set.get("grader_examples", []):
            grade = grader_example.get("grade", "UNKNOWN")
            if not "score" in grader_example:
                grader_example["score"] = 1 if grade == "CORRECT" else 0

    return cast(Sequence[QASet], data["question_answer_sets"])


def get_test_cases(qa_set: QASet) -> Sequence[QATestCase]:
    grader_examples = qa_set.get("grader_examples", [])
    test_cases: list[QATestCase] = []
    for qa in qa_set["question_answers"]:
        examples = list(qa.get("examples", [])) + list(grader_examples)
        qa = copy.copy(qa)
        qa["examples"] = examples
        test_cases.append(qa)

    return test_cases


system_prompt_template = """\
You are a teacher grading a quiz. You are given a question, the student's \
answer, 1 or more true answers, and are asked to \
score the student answer as either CORRECT or INCORRECT by comparing it to the \
true answer(s) and wrong answer(s).
"""

correctness_evaluation_prompt_template = """\
Write out in a step by step manner your reasoning to be sure that your \
conclusion is correct. Avoid simply stating the correct answer at the outset. \
Compare the student anser to the true answers. Mark the \
student answer as CORRECT if it is the same as at least one of the true \
answers. A student answer may provide additional information so long as it does not contradict any of the true answers. \
If the student answer fails to provide an answer claiming that the \
rules are unclear or do not provide sufficient information, then the answer \
recevies a grade of INCORRECT. There is no partial credit. \
Answers are either CORRECT or INCORRECT.
At the end, always output "GRADE: CORRECT" or "GRADE: INCORRECT" (without the \
quotes) to indicate your final conclusion on a line all by itself.

Example Format:

QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER 1: true answer 1 here
TRUE ANSWER 2: true answer 2 here
EXPLANATION: step by step reasoning here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy with respect to \
the true answers. Ignore differences in punctuation and phrasing between the \
student answer and the true answers. Begin!
{examples}
QUESTION: {question}
STUDENT ANSWER: {prediction}
{answers}
EXPLANATION:\
"""

CORRECTNESS_EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", correctness_evaluation_prompt_template),
    ]
)


def format_answers(answers, prefix):
    return "\n".join(
        f"{prefix} {i+1}: {answer}"
        for i, answer in enumerate(answers)
    )


def format_example(example: GradedQATestCase) -> str:
    return f"QUESTION: {example['question']}\nSTUDENT ANSWER: {example['prediction']}\n{format_answers(example['answers'], 'TRUE ANSWER')}\nEXPLANATION: {example['reasoning']}\nGRADE: {example['grade']}\n"


def format_examples(examples: Sequence[GradedQATestCase]) -> str:
    return "\n".join(format_example(example) for example in examples)


grade_regex = re.compile(r"GRADE:\s*(CORRECT|INCORRECT)", re.IGNORECASE)


@chain
def get_score(output: str) -> ScoreResult:
    reasoning = output
    match = grade_regex.search(output.strip(), re.IGNORECASE)
    value: Grade = "UNKNOWN"
    score = 0

    if match:
        reasoning = output[:match.start()].strip()
        value = cast(Grade, match.group(1).upper())
        score = 1 if value == "CORRECT" else 0
    else:
        words = output.strip().upper().split()
        if len(words) > 0:
            first_word = words[0].strip(string.punctuation)
            last_word = words[-1].strip(string.punctuation)
            if "CORRECT" in [first_word, last_word]:
                score = 1
                value = "CORRECT"
            elif "INCORRECT" in [first_word, last_word]:
                score = 0
                value = "INCORRECT"
    
    return {"score": score, "grade": value, "reasoning": reasoning}


@chain
def pick_consensus_score(scores: Sequence[ScoreResult]) -> ScoreResult:
    correct_scores = [score for score in scores if score["grade"] == "CORRECT"]
    incorrect_scores = [score for score in scores if score["grade"] == "INCORRECT"]
    unknown_scores = [score for score in scores if score["grade"] == "UNKNOWN"]
    ratio_correct: float = len(correct_scores) / len(scores)

    if ratio_correct > 0.5:
        consensus_score = correct_scores[0]
    else:
        consensus_score = (incorrect_scores + unknown_scores)[0]

    consensus_score_with_ratio = cast(ScoreResult, {**consensus_score, "ratio_correct": ratio_correct})
    return consensus_score_with_ratio


Input = TypeVar("Input")
Output = TypeVar("Output")


def build_run_ntimes_chain(chain_to_sample:Runnable[Input, Output], n:int) -> Runnable[Input, list[Output]]:
    @chain
    def duplicate(x: Input) -> list[Input]:
        # TODO: Should we do a deepcopy here or is that overkill?
        return [copy.deepcopy(x) for _ in range(n)]

    return duplicate | chain_to_sample.map()


# def prepare_prompt_evaluation_inputs(test_case: QATestCaseWithPrediction) -> dict:
#     examples_str = format_examples(test_case.get("examples", []))
#     answers_str = format_answers(test_case["answers"], "TRUE ANSWER")
#     return {
#         "examples": examples_str,
#         "question": test_case["question"],
#         "prediction": test_case["prediction"],
#         "answers": answers_str,
#     }


def build_correctness_evaluation_chain(chat_model:Runnable[LanguageModelInput, BaseMessage], prompt:ChatPromptTemplate=CORRECTNESS_EVALUATION_PROMPT, consistency_samples:int=5) -> Runnable[QATestCaseWithPrediction, ScoreResult]:
    # If we are sampling for consistency, increase the temperature to get more
    # varied outputs
    if consistency_samples > 1:
        chat_model = chat_model.bind(temperature=0.7)

    eval_chain: Runnable[QATestCaseWithPrediction, ScoreResult] = (
        prepare_prompt_evaluation_inputs
        | prompt
        | chat_model
        | StrOutputParser()
        | get_score
    )

    if consistency_samples > 1:
        result_chain = (
            build_run_ntimes_chain(eval_chain, consistency_samples)
            | pick_consensus_score
        )
    else:
        result_chain = eval_chain

    return result_chain


class EvaluateCorrectnessResult(TypedDict):
    results: Sequence[GradedQATestCase]
    accuracy: float


def evaluate_correctness(eval_chain:Runnable[QATestCaseWithPrediction, ScoreResult], test_cases:Sequence[QATestCaseWithPrediction], prompt:ChatPromptTemplate=CORRECTNESS_EVALUATION_PROMPT, consistency_samples:int=5) -> EvaluateCorrectnessResult:
    graded: list[GradedQATestCase] = []
    batch_size = 10
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i+batch_size]
        results = eval_chain.batch(list(batch))
        for test_case, result in zip(batch, results):
            graded.append(
                cast(
                    GradedQATestCase,
                    {
                        **test_case,
                        **result,
                    }
                )
            )
    
    # Get accuracy
    accuracy = sum(test_case["score"] for test_case in graded) / len(graded)
    return {"results": graded, "accuracy": accuracy}


def print_correctness_results(results: EvaluateCorrectnessResult) -> None:
    for test_case in results["results"]:
        print(f"Question: {test_case['question']}")
        print(f"Prediction: {test_case['prediction']}")
        print(f"True Answer: {test_case['answers'][0]}")
        print(f"Reasoning: {test_case['reasoning']}")
        print(f"Score: {test_case['score']}")
        print(f"Value: {test_case['grade']}")
        if "ratio_correct" in test_case:
            print(f"Ratio Correct: {test_case['ratio_correct']}")
        print()
    print(f"Accuracy: {results['accuracy']}")


ObjectiveParams = TypeVar("ObjectiveParams")


def evaluate_all_params(params: Sequence[ObjectiveParams], objective_function: Callable[[ObjectiveParams], float], logfile: Optional[TextIO] = None):
    from tqdm import tqdm
    import json
    from operator import itemgetter

    results = []
    # Iterate over params and show progress bar
    for i, param in tqdm(enumerate(params), total=len(params)):
        tqdm.write(f"Evaluating param {i+1}/{len(params)}: {param}")
        score = objective_function(param)
        result = [score, param]
        results.append(result)
        if logfile is not None:
            logfile.write(json.dumps(result))
            logfile.write("\n")
            logfile.flush()
    results.sort(key=itemgetter(0), reverse=True)
    return results


def read_evaluation_logfile(logfile: TextIO|str|Path) -> list[tuple[float, ObjectiveParams]]:
    import json

    # Make sure we have TextIO
    if isinstance(logfile, (str, Path)):
        logfile = open(logfile, "r")

    # Read lines as json
    results = []
    try:
        for line in logfile:
            line = line.strip()
            if line == "":
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON line in logfile: {line}")
    finally:
        logfile.close()

    # Sort results by score descending
    results.sort(key=itemgetter(0), reverse=True)
    return results


class Example(TypedDict, Generic[Input, Output]):
    id: Any
    input: Input
    reference_outputs: Sequence[Output]
    metadata: NotRequired[dict[str, Any]]


class PredictionResult(TypedDict, Generic[Input, Output]):
    id: Any
    input: Input
    prediction: Output
    usage_metadata: NotRequired[dict[str, Any]]


class EvaluatorResult(TypedDict):
    id: Any
    key: str
    score: float | int
    metadata: NotRequired[dict[str, Any]]


UsageMetadata = Mapping[str, int | float]


class EvaluateResult(TypedDict):
    evaluator_results: Mapping[str, Sequence[EvaluatorResult]]
    summary_results: NotRequired[Mapping[str, float | int]]
    usage_metadata: NotRequired[Mapping[str, UsageMetadata]]


Evaluator = Callable[[Sequence[Example[Input, Output]], Sequence[PredictionResult[Input, Output]]], Sequence[EvaluatorResult]|Sequence[Sequence[EvaluatorResult]]]

SummaryFunctions = Callable[
    [
        Sequence[Example[Input, Output]], 
        Sequence[PredictionResult[Input, Output]], 
        Mapping[str, Sequence[EvaluatorResult]]
    ],
    Mapping[str, float | int]
]


def summarize_mean_all(examples: Sequence[Example[Input, Output]], predictions: Sequence[PredictionResult[Input, Output]], evaluator_results: Mapping[str, Sequence[EvaluatorResult]]) -> Mapping[str, float]:
    summary: dict[str, float] = {}
    for key, results in evaluator_results.items():
        scores = [result["score"] for result in results if "score" in result]
        if scores:
            mean_score = sum(scores) / len(scores)
            summary[f"mean_{key}"] = mean_score
    return summary


def predict_for_examples(predict_fn: Callable[[Sequence[Input]], Sequence[Output]], examples: Sequence[Example[Input, Output]]) -> Sequence[PredictionResult[Input, Output]]:
    inputs = [example["input"] for example in examples]
    predictions = predict_fn(inputs)
    if len(predictions) != len(examples):
        raise ValueError(f"Predict function returned {len(predictions)} predictions, but {len(examples)} examples were provided")
    
    results: list[PredictionResult[Input, Output]] = []
    for example, prediction in zip(examples, predictions):
        result: PredictionResult[Input, Output] = {
            "id": example["id"],
            "input": example["input"],
            "prediction": prediction,
        }
        results.append(result)
    
    return results


def add_usage_dictionaries(a: Mapping[str, UsageMetadata], b: Mapping[str, UsageMetadata]) -> Mapping[str, UsageMetadata]:
    keys = set(a.keys()).union(b.keys())
    result: dict[str, dict[str, int | float]] = {}
    for k in keys:
        result[k] = {}
        keys2 = set(a.get(k, {}).keys()).union(b.get(k, {}).keys())
        for k2 in keys2:
            result[k][k2] = a.get(k, {}).get(k2, 0) + b.get(k, {}).get(k2, 0)
    return result


def evaluate(
    examples: Sequence[Example[Input, Output]],
    predictions: Sequence[PredictionResult[Input, Output]],
    evaluators: Sequence[Evaluator],
    summary_functions: Optional[Sequence[SummaryFunctions]] = None,
) -> EvaluateResult:
    if len(examples) != len(predictions):
        raise ValueError(f"Number of examples ({len(examples)}) does not match number of predictions ({len(predictions)})")

    # Check that ids match
    for example, prediction in zip(examples, predictions):
        if example["id"] != prediction["id"]:
            raise ValueError(f"Example id {example['id']} does not match prediction id {prediction['id']}")
    
    # Call each evaluator
    evaluator_results: dict[str, list[EvaluatorResult]] = {}
    for evaluator in evaluators:
        _results = evaluator(examples, predictions)

        # Normalize to list of lists
        results: list[Sequence[EvaluatorResult]] = []
        for result in _results:
            if isinstance(result, Sequence):
                results.append(result)
            else:
                results.append([result])

        # Get all keys in results and make sure they are not already in use
        result_keys = {item["key"] for result in results for item in result}
        if not result_keys:
            raise ValueError("Evaluator returned results with no keys")
        if result_keys & evaluator_results.keys():
            raise ValueError(f"Evaluator returned duplicate keys: {result_keys & evaluator_results.keys()}")
        
        # Check that results have same length as examples and no duplicate ids
        if len(results) != len(examples):
            raise ValueError(f"Evaluator returned {len(results)} results, but {len(examples)} examples were provided")
    
        for result in results:
            for item in result:
                evaluator_results.setdefault(item["key"], []).append(item)
    
    # Execute summary functions
    summary_results: dict[str, float | int] = {}
    if summary_functions is not None:
        for summary_function in summary_functions:
            summary = summary_function(examples, predictions, evaluator_results)
            # Check for duplicate keys
            if summary.keys() & summary_results.keys():
                raise ValueError(f"Summary function returned duplicate keys: {summary.keys() & summary_results.keys()}")
            summary_results.update(summary)
    
    # Collect usage metadata if available
    usage_metadata = {}
    for prediction in predictions:
        if "usage_metadata" in prediction:
            usage_metadata = add_usage_dictionaries(usage_metadata, prediction["usage_metadata"])

    # Return results
    return_value: EvaluateResult = {
        "evaluator_results": evaluator_results,
        "summary_results": summary_results,
        "usage_metadata": usage_metadata
    }
    return return_value


def load_qa_examples(path: str | Path) -> Sequence[Example[str, str]]:
    import hashlib

    qa_sets = load_question_answer_sets(path)
    examples: list[Example[str, str]] = []
    for qa_set in qa_sets:
        game_id = qa_set["game_id"]
        for i, test_case in enumerate(get_test_cases(qa_set)):
            # Give each example a unique and reproducible ID
            id: str = f'{game_id}_{test_case["question"]}'
            id = hashlib.md5(id.encode()).hexdigest()

            example: Example[str, str] = {
                "id": id,
                "input": test_case["question"],
                "reference_outputs": test_case["answers"],
                "metadata": {
                    # Save the grader examples as metadata so our LLM-as-judge
                    # can use them
                    "grader_examples": test_case.get("examples", []),
                    # Allows us to use the correct document index for retrieval
                    "game_id": game_id,
                }
            }
            examples.append(example)

    return examples


def prepare_prompt_evaluation_inputs(example_prediction: Tuple[Example[Input, Output], PredictionResult[Input, Output]]) -> dict:
    example, prediction = example_prediction
    examples_str = format_examples(example.get("metadata", {}).get("grader_examples", []))
    answers_str = format_answers(example["reference_outputs"], "TRUE ANSWER")
    return {
        "examples": examples_str,
        "question": example["input"],
        "prediction": prediction["prediction"],
        "answers": answers_str,
    }


def build_correctness_evaluation_chain(chat_model:Runnable[LanguageModelInput, BaseMessage], prompt:ChatPromptTemplate=CORRECTNESS_EVALUATION_PROMPT, consistency_samples:int=5) -> Runnable[Tuple[Example[Input, Output], PredictionResult[Input, Output]], EvaluatorResult]:

    class ScoreResultAndExample(TypedDict):
        score_result: ScoreResult
        example: Example[Input, Output]

    def score_to_evaluator_result(input: ScoreResultAndExample) -> EvaluatorResult:
        score = input["score_result"]
        example = input["example"]

        return {
            "id": example["id"],
            "key": "correctness",
            "score": score["score"],
            "metadata": {
                "grade": score["grade"],
                "reasoning": score["reasoning"],
                **({"ratio_correct": score["ratio_correct"]} if "ratio_correct" in score else {}),
            }
        }

    # If we are sampling for consistency, increase the temperature to get more
    # varied outputs
    if consistency_samples > 1:
        chat_model = chat_model.bind(temperature=0.7)

    score_result_eval_chain: Runnable[Tuple[Example[Input, Output], PredictionResult[Input, Output]], ScoreResult] = (
        prepare_prompt_evaluation_inputs
        | prompt
        | chat_model
        | StrOutputParser()
        | get_score
    )

    if consistency_samples > 1:
        score_result_eval_chain = (
            build_run_ntimes_chain(score_result_eval_chain, consistency_samples)
            | pick_consensus_score
        )

    eval_chain: Runnable[Tuple[Example[Input, Output], PredictionResult[Input, Output]], EvaluatorResult] = (
        {
            "score_result": RunnablePassthrough() | score_result_eval_chain,
            "example": itemgetter(0),
        }
        | RunnableLambda(score_to_evaluator_result)
    )

    return eval_chain


def build_llm_grader_correctness_evaluator(chat_model) -> Evaluator[Input, Output]:
    eval_chain = build_correctness_evaluation_chain(chat_model)

    def evaluator(examples: Sequence[Example[Input, Output]], predictions: Sequence[PredictionResult[Input, Output]]) -> Sequence[EvaluatorResult]:
        return eval_chain.batch(list(zip(examples, predictions)))

    return evaluator


def print_evaluation_result(result: EvaluateResult) -> None:
    if "summary_results" in result:
        print("Summary Results:")
        for key, value in result["summary_results"].items():
            print(f"{key}: {value}")
        print()


def print_detailed_evaluation_results(result: EvaluateResult, examples: Sequence[Example[Input, Output]], predictions: Sequence[PredictionResult[Input, Output]]) -> None:
    evaluator_keys = list(result["evaluator_results"].keys())

    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        print(f"Input: {example['input']}")
        print(f"Prediction: {prediction['prediction']}")
        print(f"Reference:")
        for ref in example["reference_outputs"]:
            print(f"- {ref}")

        for key in evaluator_keys:
            if key in result["evaluator_results"]:
                evaluator_result = result["evaluator_results"][key][i]
                if "metadata" in evaluator_result:
                    if "reasoning" in evaluator_result["metadata"]:
                        print(f"Reasoning: {evaluator_result['metadata']['reasoning']}")
                    if "ratio_correct" in evaluator_result["metadata"]:
                        print(f"Ratio Correct: {evaluator_result['metadata']['ratio_correct']}")
                print(f"{key.capitalize()}: {evaluator_result.get('score', '')}")
                print()

        print_evaluation_result(result)