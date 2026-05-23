from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping, Any, Iterator, Sequence, Callable

from deepeval.dataset.golden import Golden
from deepeval.models.llms.local_model import LocalModel

from structlog import get_logger

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from meeplemate.eval.runner import E2ERunner
from meeplemate.config import Config

logger = get_logger(__name__)

class E2ERunnerWithHyperparameters(E2ERunner):
    def apply_hyperparameters_to_config(self, config: Config, hyperparameters: Mapping[str, Any]) -> Config:
        # We will mutate the config based on the hyperparameters. The keys in
        # hyperparameters are dot-separated paths to the config fields, e.g.
        # "chat.temperature"
        for key, value in hyperparameters.items():
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                current = getattr(current, part)
            setattr(current, parts[-1], value)
        return config


@dataclass
class GridSearchE2ERunner:
    hyperparameter_generator: Callable[[], Iterator[Mapping[str, Any]]]
    results_directory: Path
    number_of_runs: int
    goldens: Sequence[Golden]
    local_llm_model_eval: LocalModel
    generation_concurrency: int = 5

    def _hyperparameters_to_group_run_name(self, hyperparameters: Mapping[str, Any]) -> str:
        # Sort keys by name
        keys = sorted(hyperparameters.keys())
        # Create a string representation of the hyperparameters
        s = "__".join(f"{key}={hyperparameters[key]}" for key in keys)
        # Replace all non-alphanumeric characters with underscores
        s = "".join(c if c.isalnum() else "_" for c in s)
        return s
    
    def _build_e2e_runner(self) -> E2ERunnerWithHyperparameters:
        return E2ERunnerWithHyperparameters(
            target_directory=self.results_directory,
            goldens=self.goldens,
            local_llm_model_eval=self.local_llm_model_eval,
            generation_concurrency=self.generation_concurrency,
        )
    
    async def generate(self) -> None:
        runner = self._build_e2e_runner()
        hyperparameters_combinations = list(self.hyperparameter_generator())
        with logging_redirect_tqdm():
            for hyperparameters in tqdm(hyperparameters_combinations, desc="Hyperparameter combinations"):
                group_run_name = self._hyperparameters_to_group_run_name(hyperparameters)
                await runner.generate(
                    group_run_id=group_run_name,
                    hyperparameters=hyperparameters,
                    number_of_runs=self.number_of_runs,
                    skip_existing=True,
                    overwrite=False,
                    skip_retrieval=False
                )
    
    async def evaluate(self) -> None:
        runner = self._build_e2e_runner()
        hyperparameters_combinations = list(self.hyperparameter_generator())
        for hyperparameters in hyperparameters_combinations:
            group_run_name = self._hyperparameters_to_group_run_name(hyperparameters)
            await runner.evaluate(
                group_run_id=group_run_name
            )
