import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Optional, Sequence, Mapping, Any

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.dataset.golden import Golden
from deepeval.evaluate import DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.models.llms.local_model import LocalModel

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import Run
from langchain_core.tracers.base import AsyncBaseTracer
from structlog import get_logger
from uuid_utils import uuid7

from meeplemate.chatloop import QAServiceInput
from meeplemate.component_system import subsystem, System
from meeplemate.config import Config, QAService, create_app_system, factory
from meeplemate.eval import (
    get_test_run_file_path,
    load_persisted_run,
    TestRunTracer
)
from meeplemate.eval.metrics import RunawayGenerationsMetric, ValidQuoteMetric, add_quote_counts, add_runaway_generation_counts, get_correctness_metric
from meeplemate.eval.local_model import StructuredLocalModel
from meeplemate.game_service import GameService
from meeplemate.qa_graph import Chunk
from meeplemate.tracing.serialization import skip_run_types
from meeplemate.util import snake_case

logger = get_logger(__name__)


class Runner:
    target_directory: Path

    def __init__(self, target_directory: Path):
        self.target_directory = target_directory

    async def generate(
        self,
        group_run_id:str,
        hyperparameters: Mapping[str, Any]|None=None,
        number_of_runs:int=1,
        skip_existing:bool=True,
        overwrite:bool=False,
        skip_retrieval:bool=False
    ) -> None:
        raise NotImplementedError
    
    async def evaluate(self, group_run_id:str) -> None:
        raise NotImplementedError


def create_eval_system(names: Optional[Sequence[str]]=None, config:Config|None=None) -> System:
    if config is None:
        config = Config()
    config.chat.models[0].endpoint = "http://192.168.0.44:8000/v1"

    extra_components = {
        "deepeval_llm": (
            factory(StructuredLocalModel)(
                model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                api_key="dummy",
                base_url=config.chat.models[0].endpoint,
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
            ),
            []
        )
    }

    app_system = create_app_system(config)
    system = subsystem(app_system, names=names, extra_components=extra_components)  
    return system

def get_test_suite_for_golden(golden: Golden) -> str:
    additional_metadata = golden.additional_metadata or {}
    return additional_metadata.get("test_suite", "default")

@dataclass
class E2ERunner(Runner):
    target_directory: Path
    goldens: Sequence[Golden]
    local_llm_model_eval: LocalModel
    generation_runs_directory_name: str = "generation_runs"
    evaluation_results_directory_name: str = "qa_evals"
    generation_semaphore: asyncio.Semaphore = field(init=False)
    generation_concurrency: int = 5

    def __post_init__(self):
        self.generation_semaphore = asyncio.Semaphore(self.generation_concurrency)

    @property
    def generation_runs_directory(self):
        return self.target_directory / self.generation_runs_directory_name
    
    @property
    def evaluation_results_directory(self):
        return self.target_directory / self.evaluation_results_directory_name
    
    def _get_generation_group_run_path(self, group_run_id:str, golden: Golden, run_number:int) -> Path:
        assert golden.name is not None, "Golden name is required for E2E tests"
        test_suite = get_test_suite_for_golden(golden)
        return get_test_run_file_path(
            self.generation_runs_directory,
            group_run_id,
            test_suite,
            golden.name,
            run_number
        )
    
    def _discover_existing_generation_runs(self, group_run_id:str) -> Sequence[Path]:
        # Discover existing runs for the given group_run_id by scanning the generation_runs directory
        generation_runs_dir = self.generation_runs_directory
        group_runs_dir = generation_runs_dir / group_run_id
        if not group_runs_dir.exists():
            return []
        
        existing_runs = []
        # Find all the *.run*.json files
        for run_file in group_runs_dir.glob("*.run*.json"):
            existing_runs.append(run_file)
        return existing_runs

    def _build_tracer(self) -> AsyncBaseTracer:
        tracer = TestRunTracer(
            self.generation_runs_directory,
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
        return tracer

    async def _run_agent(
        self,
        dependencies: Mapping[str, Any],
        golden:Golden,
        metadata: Mapping[str, Any],
        skip_retrieval:bool=False,
    ) -> None:
        assert golden.additional_metadata is not None, "Additional metadata is required for E2E tests"
        game_id = golden.additional_metadata.get("game_id")
        assert game_id is not None, "game_id is required in additional metadata for E2E tests"

        game_service: GameService = dependencies["game_service"]
        qa_service: QAService = dependencies["qa_service"]

        manifest = await game_service.get_manifest(game_id)
        assert manifest is not None, f"Manifest not found for game_id: {game_id}"

        tracer = self._build_tracer()

        config: RunnableConfig = {
            "callbacks": [tracer],
            "configurable": {
                "thread_id": str(uuid7()),
            },
            "metadata": dict(metadata),
        }

        input: QAServiceInput = {
            "manifest": manifest,
            "query": golden.input,
            "messages": [],
            "recursion_depth": 0,
            "evidence": [],
        }

        start_log_kwargs: dict[str, Any] = {
            "test_case": golden.name,
        }

        if skip_retrieval:
            logger.info("Skipping retrival as per flag", test_case=golden.name)
            golden_evidence = golden.additional_metadata.get("evidence")
            if golden_evidence is None:
                raise ValueError("Evidence must be provided in additional metadata when skip_retrieval is True")
            chunks: list[Chunk] = []
            for item in golden_evidence:
                chunks.append(
                    {
                        "rulebook_name": item["rulebook"],
                        "page": item["page"],
                        "start_index": item.get("start_index", -1),
                        "end_index": item.get("end_index", -1),
                        "content": item["quote"],
                    }
                )
                input["evidence"] = chunks
                start_log_kwargs["number_of_chunks"] = len(chunks)
        
        logger.info("Starting QA generation", **start_log_kwargs)

        await qa_service.ainvoke(
            input,
            config=config
        )
    
    def apply_hyperparameters_to_config(self, config: Config, hyperparameters: Mapping[str, Any]) -> Config:
        return config

    @asynccontextmanager
    async def generation_astart_dependencies(self, hyperparameters: Mapping[str, Any]|None=None) -> AsyncIterator[Mapping[str, Any]]:
        config = Config()
        if hyperparameters:
            config = self.apply_hyperparameters_to_config(config, hyperparameters)
        system = create_eval_system(["game_service", "qa_service"], config=config)
        async with system.astart() as services:
            yield services

    async def generate(
        self,
        group_run_id:str,
        hyperparameters: Mapping[str, Any]|None=None,
        number_of_runs:int=1,
        skip_existing:bool=True,
        overwrite:bool=False,
        skip_retrieval:bool=False
    ) -> None:
        async with self.generation_astart_dependencies(hyperparameters) as services:
            tasks = []
            for golden in self.goldens:
                additional_metadata = golden.additional_metadata or {}
                test_suite = additional_metadata.get("test_suite", "default")

                assert golden.name is not None, "Golden name is required for E2E tests"
                assert group_run_id is not None, "group_run_id is required for E2E tests"

                for run_number in range(number_of_runs):
                    run_file_path = get_test_run_file_path(
                        self.generation_runs_directory,
                        group_run_id,
                        test_suite,
                        golden.name,
                        run_number
                    )

                    if run_file_path.exists():
                        if skip_existing:
                            logger.info("Run file already exists, skipping generation", test_case=golden.name, run_file_path=str(run_file_path))
                            continue
                        elif overwrite:
                            run_file_path.unlink()
                            logger.info("Deleted existing run file", path=str(run_file_path))
                        else:
                            raise FileExistsError(f"Run file already exists: {run_file_path}")
                    
                    run_file_path.parent.mkdir(parents=True, exist_ok=True)

                    metadata = {
                        "test_group_run_id": group_run_id,
                        "test_case": golden.name,
                        "test_suite": test_suite,
                        "skip_retrieval": skip_retrieval,
                        "run_number": run_number,
                        "hyperparameters": hyperparameters,
                    }

                    async def ainvoke(*args, **kwargs):
                        async with self.generation_semaphore:
                            await self._run_agent(
                                *args, **kwargs
                            )

                    tasks.append(
                        ainvoke(
                            services,
                            golden,
                            metadata,
                            skip_retrieval=skip_retrieval
                        )
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for golden, result in zip(self.goldens, results):
                if isinstance(result, Exception):
                    logger.exception("Error during generation", test_case=golden.name, exc_info=result)

    def _create_test_case(self, golden: Golden, run: Run) -> LLMTestCase:
        assert golden.name is not None, "Golden name is required for creating test case"

        # With both the golden and the run, we can create the test case
        response = run.outputs.get("response") if run.outputs else None
        if response is None or not response.strip():
            actual_output = "[Run failed to produce a response]"
        else:
            actual_output = response
        
        run_metadata = run.metadata or {}
        run_number = run_metadata.get("run_number")
        test_suite = get_test_suite_for_golden(golden)

        # Create unique test name including run ID
        test_name = f"{snake_case(test_suite)}__{snake_case(golden.name)}"
        if run_number is not None:
            test_name += f"__run{str(run_number).zfill(3)}"

        test_case = LLMTestCase(
            name=test_name,
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output,
        )
        add_quote_counts(test_case, run)
        add_runaway_generation_counts(test_case, run)

        return test_case

    def _create_test_cases(self, group_run_id:str) -> Sequence[LLMTestCase]:
        test_cases = []
        run_files = self._discover_existing_generation_runs(group_run_id)
        for run_file in run_files:
            run = load_persisted_run(run_file)
            test_suite = run.metadata.get("test_suite")
            assert test_suite is not None, "test_suite is required in run metadata for evaluation"
            test_case_name = run.metadata.get("test_case")
            assert test_case_name is not None, "test_case is required in run metadata for evaluation"

            golden = next((g for g in self.goldens if g.name == test_case_name and get_test_suite_for_golden(g) == test_suite), None)
            if not golden:
                logger.warning("No matching golden found for run, skipping test case creation", test_suite=test_suite, test_case=test_case_name)
                continue
            
            test_case = self._create_test_case(golden, run)
            test_cases.append(test_case)
        
        return test_cases

    def _get_hyperparameters(self, group_run_id:str) -> Mapping[str, Any]:
        # Get hyperparameters from one of the runs for this group_run_id (they should all be the same)
        run_files = self._discover_existing_generation_runs(group_run_id)
        if not run_files:
            logger.warning("No run files found for group_run_id, cannot extract hyperparameters", group_run_id=group_run_id)
            return {}
        
        run = load_persisted_run(run_files[0])
        hyperparameters = run.metadata.get("hyperparameters")
        return hyperparameters or {}

    def get_evaluation_metrics(self) -> Sequence[BaseMetric]:
        return [
            get_correctness_metric(self.local_llm_model_eval),
            ValidQuoteMetric(threshold=0.9),
            RunawayGenerationsMetric(threshold=0.9),
        ]

    async def evaluate(self, group_run_id:str) -> None:
        test_cases = self._create_test_cases(group_run_id)
        hyperparameters:Mapping[str, Any] = self._get_hyperparameters(group_run_id)

        if not test_cases:
            logger.warning("No test cases found for evaluation", group_run_id=group_run_id)
            return

        logger.info("Evaluated test cases", count=len(test_cases), group_run_id=group_run_id)
        metrics = self.get_evaluation_metrics()

        eval_directory = self.evaluation_results_directory / group_run_id
        eval_directory.mkdir(parents=True, exist_ok=True)

        display_config = DisplayConfig(results_folder=str(eval_directory))

        evaluate(
            list(test_cases),
            list(metrics),
            hyperparameters=dict(hyperparameters),
            display_config=display_config
        )
