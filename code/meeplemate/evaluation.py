import copy
import re
import string
from operator import itemgetter
from langchain_core.runnables import (chain, Runnable, RunnablePassthrough, RunnableLambda)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from meeplemate.consistency import build_run_ntimes_chain

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
QUESTION: {query}
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


@chain
def format_true_answers_chain(true_answers):
    return format_answers(true_answers, "TRUE ANSWER")

@chain
def format_wrong_answers_chain(wrong_answers):
    return format_answers(wrong_answers, "WRONG ANSWER")

def format_example(example):
    return f"QUESTION: {example['query']}\nSTUDENT ANSWER: {example['prediction']}\n{format_answers(example['true_answers'], 'TRUE ANSWER')}\nEXPLANATION: {example['reasoning']}\nGRADE: {example['value']}\n"


@chain
def format_examples(examples):
    return "\n".join(format_example(example) for example in examples)


grade_regex = re.compile(r"GRADE:\s*(CORRECT|INCORRECT)", re.IGNORECASE)

@chain
def get_score(output):
    reasoning = output
    match = grade_regex.search(output.strip(), re.IGNORECASE)
    if match:
        reasoning = output[:match.start()].strip()
        value = match.group(1).upper()
        score = 1 if value == "CORRECT" else 0
    else:
        try:
            words = output.strip().upper().split()
            first_word = words[0].strip(string.punctuation)
            last_word = words[-1].strip(string.punctuation)
            if "CORRECT" in [first_word, last_word]:
                score = 1
                value = "CORRECT"
            elif "INCORRECT" in [first_word, last_word]:
                score = 0
                value = "INCORRECT"
            else:
                return None
        except IndexError:
            return None

    return {"score": score, "value": value, "reasoning": reasoning}


@chain
def pick_consensus_score(scores):
    scores = [score for score in scores if score is not None]
    if not scores:
        return {"score": 0, "value": "INCORRECT", "reasoning": "No consensus", "ratio_correct": 0.0}
    correct_scores = [score for score in scores if score["value"] == "CORRECT"]
    incorrect_scores = [score for score in scores if score["value"] == "INCORRECT"]
    ratio_correct = len(correct_scores) / len(scores)
    if ratio_correct > 0.5:
        consensus_score = correct_scores[0]
    else:
        consensus_score = incorrect_scores[0]
    consensus_score = {**consensus_score, "ratio_correct": ratio_correct}
    return consensus_score


def build_run_ntimes_chain(chain_to_sample:Runnable, n:int) -> Runnable:
    @chain
    def duplicate(x):
        # TODO: Should we do a deepcopy here or is that overkill?
        return [copy.deepcopy(x) for _ in range(n)]

    return duplicate | chain_to_sample.map()


def build_itemgetter_with_default(key, default):

    @chain
    def itemgetter_with_default(x):
        return x.get(key, default)

    return itemgetter_with_default

    
def build_test_case_correctness_evaluation_chain(chat_model, prompt=CORRECTNESS_EVALUATION_PROMPT, consistency_samples=5):
    eval_chain = (
        {
            "examples": build_itemgetter_with_default("examples", []) | format_examples,
            "query": itemgetter("query"),
            "answers": itemgetter("true_answers") | format_true_answers_chain,
            # "wrong_answers": itemgetter("wrong_answers") | format_wrong_answers_chain,
            "prediction": itemgetter("prediction"),
        }
        # | RunnablePassthrough.assign(answers=RunnableLambda(lambda x: x["true_answers"] + x["wrong_answers"]))
        | prompt
        | chat_model
        | StrOutputParser()
        | get_score
    )

    if consistency_samples > 1:
        result_chain = (
            build_run_ntimes_chain(eval_chain, consistency_samples)
            # | eval_chain.map()
            | pick_consensus_score
        )
    else:
        result_chain = eval_chain

    return result_chain


def preprocess_test_case(test_case):
    test_case = copy.deepcopy(test_case)
    examples = test_case.get("examples", [])
    for wrong_example in test_case.get("wrong_examples", []):
        wrong_example = {**test_case, **wrong_example, "value": "INCORRECT", "score": 0}
        examples.append(wrong_example)
    for correct_example in test_case.get("correct_examples", []):
        correct_example = {**test_case, **correct_example, "value": "CORRECT", "score": 1}
        examples.append(correct_example)
    test_case["examples"] = examples
    return test_case


def evaluate_correctness(chat_model, test_cases, prompt=CORRECTNESS_EVALUATION_PROMPT, consistency_samples=5):
    test_cases = [preprocess_test_case(test_case) for test_case in test_cases]
    eval_chain = build_test_case_correctness_evaluation_chain(chat_model, prompt=prompt, consistency_samples=consistency_samples)

    batch_size = 10
    for i in range(0, len(test_cases), batch_size):
        batch = test_cases[i:i+batch_size]
        results = eval_chain.batch(batch)
        for test_case, result in zip(batch, results):
            test_case.update(result)
    
    # Get accuracy
    accuracy = sum(test_case["score"] for test_case in test_cases) / len(test_cases)
    return {"results": test_cases, "accuracy": accuracy}


def print_correctness_results(results):
    for test_case in results["results"]:
        print(f"Question: {test_case['query']}")
        print(f"Prediction: {test_case['prediction']}")
        print(f"True Answer: {test_case['true_answers'][0]}")
        print(f"Reasoning: {test_case['reasoning']}")
        print(f"Score: {test_case['score']}")
        print(f"Value: {test_case['value']}")
        print(f"Ratio Correct: {test_case['ratio_correct']}")
        print()
    print(f"Accuracy: {results['accuracy']}")