import asyncio
import json
from pathlib import Path
import click
from gradio import skip
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.stores import BaseStore
from uuid_utils import uuid7
from meeplemate.eval import collect_runs_by_name_iter, load_persisted_run, skip_run_types, snake_case, test_suites, TestRunTracer, get_test_run_file_path
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
from meeplemate.qa_graph import QAService, QuoteValidationException
from meeplemate.util import slurp_json
from meeplemate.config import GameService


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


async def _run_qa_gen_multiple_runs(filter: str, group_run_id: str, number_of_runs: int):
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

    # Use format __run001, __run002, etc for each run suffix
    group_run_ids = [f"{group_run_id}__run{str(i+1).zfill(3)}" for i in range(number_of_runs)]
    tasks = []
    for run_id in group_run_ids:
        tasks.append(
            asyncio.create_task(_run_qa_gen_no_start_system(filter, run_id))
        )
    await asyncio.gather(*tasks)


async def _run_qa_gen(filter: str, group_run_id: str):
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
    await _run_qa_gen_no_start_system(filter, group_run_id)


async def _run_qa_gen_no_start_system(filter: str, group_run_id: str):
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

            tasks.append(
                asyncio.create_task(
                    qa_service.ainvoke(
                        {"manifest": manifest, "query": query},
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


async def _run_qa_eval(filter: str, group_run_id: str):
    # extra_components = {
    #     # Lets not persist the graph state during evals
    #     "checkpointer": (
    #         factory(InMemorySaver, ignore_context_manager=True)(),
    #         []
    #     )
    # }

    # await areload(
    #     ["game_service", "qa_service", "checkpointer"],
    #     extra_components=extra_components
    # )

    llm = StructuredLocalModel(
        model="Qwen/Qwen3-32B",
        api_key="dummy",
        base_url="http://192.168.0.44:8000/v1"
    )

    # game_service: GameService = get_service("game_service")
    llm_tests = []

    for test_suite in test_suites:
        game_id = test_suite["params"]["game_id"]
        for test_case in test_suite["test_cases"]:
            if not fnmatch.fnmatch(test_case["name"], filter):
                continue

            if "reference_answer" not in test_case:
                click.echo(f"Skipping test case without reference_answer: {test_case['name']}")
                continue

            run_file = get_test_run_file_path(get_eval_generation_runs_dir(), group_run_id, test_suite["name"], test_case["name"])
            if not run_file.exists():
                click.echo(f"Run file not found: {run_file}")
                continue
            run = load_persisted_run(run_file)

            # Get the actual output from the run
            assert run.outputs is not None, "No outputs in run"
            actual_output = run.outputs["response"]

            llm_test = LLMTestCase(
                input=test_case["query"],
                actual_output=actual_output,
                expected_output=test_case["reference_answer"],
                name=f"{snake_case(test_suite['name'])}__{snake_case(test_case['name'])}",
            )
            llm_tests.append(llm_test)
    
    metrics: list[BaseMetric] = [get_correctness_metric(model=llm), AnswerRelevancyMetric(model=llm)]
    results = evaluate(llm_tests, metrics)
         

@click.group()
def cli():
    pass


@cli.command()
@click.argument("filter", required=False, default="*")
@click.option("--group-run-id", default=None)
@click.option("--number-of-runs", default=None, type=int)
def run_qa_gen(filter: str, group_run_id: str | None = None, number_of_runs: int|None = None):
    import asyncio
    # Default group_run_id is today's date in YYYY-MM-DD format
    if group_run_id is None:
        from datetime import datetime
        group_run_id = datetime.now().strftime("%Y-%m-%d")

    if number_of_runs is None:
        asyncio.run(_run_qa_gen(filter, group_run_id))
    else:
        asyncio.run(_run_qa_gen_multiple_runs(filter, group_run_id, number_of_runs))

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
def print_run_summary(filter: str = "*", group_run_id: str | None = None):
    import asyncio
    asyncio.run(_print_summary_of_run(filter, group_run_id))


if __name__ == "__main__":
    cli()