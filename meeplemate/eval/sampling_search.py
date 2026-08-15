
from meeplemate import eval_logging
eval_logging.configure_logging()

from pathlib import Path
from typing import Mapping, Any, Iterator, Sequence, Callable

from meeplemate.eval import load_goldens, next_group_run_id
from meeplemate.eval.local_model import StructuredLocalModel
from meeplemate.eval.grid_search import GridSearchE2ERunner
from deepeval.models.llms.local_model import LocalModel

import click


def get_base_directory() -> Path:
    import meeplemate
    project_dir = Path(meeplemate.__file__).parent.parent
    return project_dir / "data" / "evals" / "sampling_param_search"


def hyperparameters_generator() -> Iterator[Mapping[str, Any]]:
    # `chat` is now a failover list of models; the grid search sweeps the
    # primary model's sampling params, so keys are pinned to models.0.
    grid = {
        "chat.models.0.model_name": ["Qwen/Qwen3-30B-A3B-Instruct-2507"],
        # "chat.models.0.model_name": ["Qwen/Qwen3.6-35B-A3B"],
        "chat.models.0.temperature": [0.7],
        # "chat.models.0.top_p": [1.0],
        "chat.models.0.top_p": [0.8],
        "chat.models.0.top_k": [-1],
        # "chat.models.0.min_p": [0.05, 0.1],
        "chat.models.0.min_p": [0.0],
        "chat.models.0.presence_penalty": [0.0, 0.4, 0.8, 1.2],
        "chat.models.0.frequency_penalty": [0.0, 0.2, 0.6, 1.0],
        "chat.models.0.repetition_penalty": [1.0]
    }

    # Yield all combinations of the above hyperparameters
    from itertools import product
    keys, values = zip(*grid.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))


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


def build_grid_search_runner(base_directory: Path, group_run_id: str|None = None) -> GridSearchE2ERunner:
    all_goldens = load_goldens()
    goldens = [
        g for g in all_goldens 
        if g.name in {"monster_pursue_too_low_loot_room", "grail_knights_break_tests", "warpstorm_flying_high_exception"}]

    base_directory = get_base_directory()
    base_directory.mkdir(parents=True, exist_ok=True)

    if group_run_id is None:
        group_run_id = next_group_run_id(base_directory)

    target_directory = base_directory / group_run_id
    target_directory.mkdir(parents=True, exist_ok=True)

    # Ensure generation_runs directory exists for E2ERunner
    generation_runs_directory = target_directory / "generation_runs"
    generation_runs_directory.mkdir(parents=True, exist_ok=True)

    # Ensure evaluation_runs directory exists for E2ERunner
    evaluation_runs_directory = target_directory / "qa_evals"
    evaluation_runs_directory.mkdir(parents=True, exist_ok=True)

    generator = GridSearchE2ERunner(
        hyperparameter_generator=hyperparameters_generator,
        results_directory=target_directory,
        number_of_runs=5,
        goldens=goldens,
        local_llm_model_eval=build_local_model(), # TODO: add local model eval
        generation_concurrency=5
    )

    return generator


@click.group()
def cli():
    pass

@cli.command()
@click.option("--group-run-id", default=None)
def search_gen(group_run_id: str|None = None):
    base_directory = get_base_directory()
    base_directory.mkdir(parents=True, exist_ok=True)

    runner = build_grid_search_runner(base_directory, group_run_id)

    import asyncio
    asyncio.run(runner.generate())


@cli.command()
@click.option("--group-run-id")
def search_eval(group_run_id: str):
    base_directory = get_base_directory()
    base_directory.mkdir(parents=True, exist_ok=True)

    runner = build_grid_search_runner(base_directory, group_run_id)

    import asyncio
    asyncio.run(runner.evaluate())


if __name__ == "__main__":
    cli()