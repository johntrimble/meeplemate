from importlib import resources
from langchain.chat_models import BaseChatModel
from structlog import get_logger
from typing import Iterator, Sequence, TypedDict, override, Callable
import json
from uuid import UUID
from datetime import datetime
from fnmatch import fnmatch

from langchain_core.tracers import Run
from langchain_core.load import dumpd
from meeplemate.util import slurp_yaml, snake_case

# Create tracer for persisting langchain runs during tests
from langchain_core.tracers.base import AsyncBaseTracer
from pathlib import Path

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset.golden import Golden

logger = get_logger(__name__)    

# Load test_cases.yaml from this module
test_suites = slurp_yaml(resources.files(__package__).joinpath("test_cases.yaml"))

def load_goldens() -> Sequence[Golden]:
    goldens: list[Golden] = []
    for test_suite in test_suites:
        game_id = test_suite["params"]["game_id"]
        for test_case in test_suite["test_cases"]:
            if "reference_answer" not in test_case:
                continue
            metadata = {
                "test_suite": test_suite["name"],
                "game_id": game_id,
            }
            if "evidence" in test_case:
                metadata["evidence"] = test_case["evidence"]

            golden = Golden(
                input=test_case["query"],
                expected_output=test_case["reference_answer"],
                name=test_case["name"],
                additional_metadata=metadata,
                multimodal=False,
            )
            goldens.append(golden)
    return goldens


class RunEncoder(json.JSONEncoder):
    """Custom JSON encoder for Run objects that handles UUIDs, datetimes, and other LangChain types."""

    def default(self, o):
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        # For any other complex object, try to use dumpd
        try:
            return dumpd(o)
        except Exception:
            # Fall back to string representation
            return str(o)


def get_test_run_file_path(
    eval_runs_dir: Path,
    test_group_run_id: str,
    test_suite: str,
    test_case: str,
    run_number: int | None = None
) -> Path:
    """Get the file path for a test run.

    Args:
        eval_runs_dir: Base directory for evaluation runs
        test_group_run_id: Group run identifier (e.g., "2026-01-25")
        test_suite: Test suite name
        test_case: Test case name
        run_number: Optional run number (1, 2, 3, etc.). If None, creates base run file.

    Returns:
        Path to the run file. Examples:
        - run_number=None: eval_runs_dir/2026-01-25/test_suite__test_case.json
        - run_number=1: eval_runs_dir/2026-01-25/test_suite__test_case.run001.json
    """
    test_suite = snake_case(test_suite)
    test_case = snake_case(test_case)
    group_output_dir = eval_runs_dir / test_group_run_id
    group_output_dir.mkdir(exist_ok=True)

    # Construct filename with optional run suffix
    base_name = f"{test_suite}__{test_case}"
    if run_number is not None:
        filename = f"{base_name}.run{str(run_number).zfill(3)}.json"
    else:
        filename = f"{base_name}.json"

    run_file = group_output_dir / filename
    return run_file


def parse_group_run_id(group_run_id: str) -> tuple[str, int | None]:
    """Parse group_run_id into base ID and run number.

    Args:
        group_run_id: Group run identifier, optionally with __runXXX suffix

    Returns:
        Tuple of (base_group_run_id, run_number)

    Examples:
        "2026-01-25" -> ("2026-01-25", None)
        "2026-01-25__run001" -> ("2026-01-25", 1)
        "2026-01-25__run042" -> ("2026-01-25", 42)
    """
    if "__run" in group_run_id:
        parts = group_run_id.split("__run")
        base_id = parts[0]
        run_num = int(parts[1])
        return base_id, run_num
    else:
        return group_run_id, None


def load_persisted_run(run_file_path: Path|str) -> Run:
    """Load a persisted Run object from a JSON file.

    Args:
        run_file_path: Path to the JSON file containing the persisted run

    Returns:
        Run object reconstructed from the persisted data
    """
    if isinstance(run_file_path, str):
        run_file_path = Path(run_file_path)

    with run_file_path.open("r") as f:
        run_dict = json.load(f)

    # Use parse_obj to reconstruct the Run from the dict
    return Run.parse_obj(run_dict)


def print_run_tree(run: Run, indent: int = 0, max_depth: int = 10) -> None:
    """Print a hierarchical view of a run and its children.

    Args:
        run: The run to print
        indent: Current indentation level (for recursion)
        max_depth: Maximum depth to traverse
    """
    if indent > max_depth:
        return

    prefix = "  " * indent
    duration = ""
    if run.start_time and run.end_time:
        duration_secs = (run.end_time - run.start_time).total_seconds()
        duration = f" ({duration_secs:.2f}s)"

    print(f"{prefix}{run.name} [{run.run_type}]{duration}")

    # Print some additional info for certain run types
    if indent < 2:  # Only for top levels to avoid clutter
        if run.run_type == "llm" and run.outputs:
            # Show token usage if available
            generations = run.outputs.get("generations", [])
            if generations and len(generations) > 0:
                gen = generations[0][0]
                if hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                    metadata = gen.message.response_metadata
                    if "token_usage" in metadata:
                        tokens = metadata["token_usage"]
                        print(f"{prefix}  Tokens: {tokens}")

    # Recursively print children
    for child in run.child_runs:
        print_run_tree(child, indent + 1, max_depth)


def collect_runs_iter(run: Run) -> Iterator[Run]:
    """Recursively collect all runs from the run tree.

    Args:
        run: The root run to search
    Returns:
        List of all runs in the tree
    """
    yield run
    for child in run.child_runs:
        yield from collect_runs_iter(child)


def collect_runs_by_name_iter(run: Run, name: str) -> Iterator[Run]:
    """Recursively collect all runs with a specific name from the run tree.

    Args:
        run: The root run to search
        name: The name of runs to collect

    Returns:
        Iterator of all runs matching the specified name
    """
    for run in collect_runs_iter(run):
        if run.name == name:
            yield run


def collect_runs_by_type(run: Run, run_type: str) -> list[Run]:
    """Recursively collect all runs of a specific type from the run tree.

    Args:
        run: The root run to search
        run_type: The type of runs to collect (e.g., "llm", "chain", "tool")

    Returns:
        List of all runs matching the specified type
    """
    runs = []
    for run in collect_runs_iter(run):
        if run.run_type == run_type:
            runs.append(run)
    return runs


class QuoteCounts(TypedDict):
    total_quotes: int
    quote_errors: int


def count_quotes_and_quote_errors(run: Run) -> QuoteCounts:
    
    def count_quotes_in_response(response_text: str) -> int:
        import re
        # Count how many blocksquotes there are (groups of lines starting with
        # `>`). Keeping in mind that multiple consecutive lines starting with
        # `>` count as one quote. This will need to be multiline:
        quote_count = len(re.findall(r'(?m)^>.*(?:\n>.*)*', response_text))
        return quote_count

    fix_nodes = list(collect_runs_by_name_iter(run, "fix_quote_chain"))
    # fixed = sum(1 for n in fix_nodes if (n.outputs or {}).get('fixable') == True)
    # unfixable = sum(1 for n in fix_nodes if (n.outputs or {}).get('fixable') == False)
    provide_nodes = list(collect_runs_by_name_iter(run, 'provide_response'))
    responses = [
        (n.outputs or {}).get('response', '')
        for n in provide_nodes
    ]

    quote_errors = len(fix_nodes)
    quote_count = sum(
        count_quotes_in_response(response)
        for response in responses
    )

    return {"total_quotes": quote_count, "quote_errors": quote_errors}


class RunawayGenerationCounts(TypedDict):
    total_generations: int
    runaway_generations: int


def count_llm_runaway_generations(run: Run) -> RunawayGenerationCounts:
    llm_runs = collect_runs_by_type(run, "llm")
    total = 0
    runaway = 0
    for llm_run in llm_runs:
        generations = (llm_run.outputs or {}).get("generations", [])
        if not generations or not generations[0]:
            continue
        gen = generations[0][0]
        msg = gen.get("message", {}) if isinstance(gen, dict) else {}
        kwargs = msg.get("kwargs", {}) if isinstance(msg, dict) else {}
        finish_reason = kwargs.get("response_metadata", {}).get("finish_reason")
        total += 1
        if finish_reason == "length":
            runaway += 1
    return {"total_generations": total, "runaway_generations": runaway}


def get_llm_calls(run: Run) -> list[Run]:
    """Extract all LLM calls from a run tree.

    Args:
        run: The root run to search

    Returns:
        List of all LLM runs
    """
    return collect_runs_by_type(run, "llm")


def extract_token_usage_from_run(run: Run) -> dict[str, dict[str, int]]:
    """Sum token usage per model for a run, deduplicating by LLM run ID.

    Returns:
        Dict mapping model_name -> {"input_tokens": N, "output_tokens": N}
    """
    llm_runs = get_llm_calls(run)
    seen_ids: set[str] = set()
    usage_by_model: dict[str, dict[str, int]] = {}

    for llm_run in llm_runs:
        run_id = str(llm_run.id)
        if run_id in seen_ids:
            continue
        seen_ids.add(run_id)

        try:
            generations = (llm_run.outputs or {}).get("generations", [])
            if not generations or not generations[0]:
                continue
            gen = generations[0][0]
            msg = gen.get("message", {}) if isinstance(gen, dict) else {}
            kwargs = msg.get("kwargs", {}) if isinstance(msg, dict) else {}
            usage = kwargs.get("usage_metadata")
            if not usage:
                continue
            model_name = kwargs.get("response_metadata", {}).get("model_name", "unknown")
            if model_name not in usage_by_model:
                usage_by_model[model_name] = {"input_tokens": 0, "output_tokens": 0}
            usage_by_model[model_name]["input_tokens"] += usage.get("input_tokens", 0) or 0
            usage_by_model[model_name]["output_tokens"] += usage.get("output_tokens", 0) or 0
        except Exception:
            continue

    return usage_by_model


def get_run_summary(run: Run) -> dict:
    """Get a summary of a run including timing and statistics.

    Args:
        run: The run to summarize

    Returns:
        Dictionary containing summary statistics
    """
    from collections import Counter

    def count_by_type(r: Run) -> Counter:
        counts = Counter([r.run_type])
        for child in r.child_runs:
            counts.update(count_by_type(child))
        return counts

    duration = (run.end_time - run.start_time).total_seconds() if run.end_time and run.start_time else 0
    llm_calls = get_llm_calls(run)
    llm_duration = sum((llm.end_time - llm.start_time).total_seconds()
                      for llm in llm_calls
                      if llm.end_time and llm.start_time)

    return {
        "id": str(run.id),
        "name": run.name,
        "run_type": run.run_type,
        "duration_seconds": duration,
        "llm_calls": len(llm_calls),
        "llm_duration_seconds": llm_duration,
        "run_type_counts": dict(count_by_type(run)),
        "error": run.error if run.error else None,
    }


def transform_run_tree(
    run: Run,
    transform_fn: Callable[[Run, list[Run]], tuple[Run | None, list[Run]]]
) -> Run | None:
    """Generic tree transformer for Run objects.

    Recursively transforms a run tree by applying a transformation function to each node.
    The transformer can modify nodes, skip them (promoting their children), or filter children.

    Args:
        run: The run to transform
        transform_fn: Function that takes (run, transformed_children) and returns
                     (transformed_run_or_None, children_to_use).
                     - Return (None, children) to skip this node and promote children
                     - Return (run, children) to keep the node with new children
                     - Return (run, []) to keep node but remove all children

    Returns:
        Transformed run, or None if the root should be skipped
    """
    def _inner(run: Run) -> list[Run]:
        # Recursively transform children, collecting promoted ones
        transformed_children = []
        for child in run.child_runs:
            transformed_children.extend(_inner(child))

        # Apply transformation to this node
        transformed_run, final_children = transform_fn(run, transformed_children)

        if transformed_run is not None:
            transformed_run.child_runs = final_children
            return [transformed_run]
        else:
            # Node was skipped — promote its children to the parent
            return final_children

    results = _inner(run)
    return results[0] if results else None


def filter_children_by_predicate(
    predicate: Callable[[Run], bool]
) -> Callable[[Run, list[Run]], tuple[Run | None, list[Run]]]:
    """Create a transformer that filters children based on a predicate.

    Args:
        predicate: Function that returns True for children to keep

    Returns:
        A transformer function

    Example:
        transformer = filter_children_by_predicate(lambda r: r.run_type != "retriever")
        filtered_run = transform_run_tree(run, transformer)
    """
    def transformer(run: Run, children: list[Run]) -> tuple[Run | None, list[Run]]:
        filtered_children = [c for c in children if predicate(c)]
        return (run, filtered_children)
    return transformer


def limit_run_tree_depth(run: Run, max_depth: int) -> Run:
    """Limit the depth of a run tree.

    Unlike the transformer-based approach, this function directly limits tree depth
    by truncating children beyond the specified depth.

    Args:
        run: The run to limit
        max_depth: Maximum depth to preserve (0 means only root with no children)

    Returns:
        Run with depth limited

    Example:
        limited_run = limit_run_tree_depth(run, max_depth=3)
    """
    def limit_depth_recursive(node: Run, current_depth: int) -> Run:
        if current_depth >= max_depth:
            # Truncate children at this level
            node.child_runs = []
            return node

        # Recursively limit children
        node.child_runs = [
            limit_depth_recursive(child, current_depth + 1)
            for child in node.child_runs
        ]
        return node

    return limit_depth_recursive(run, 0)


def compose_transformers(
    *transformers: Callable[[Run, list[Run]], tuple[Run | None, list[Run]]]
) -> Callable[[Run, list[Run]], tuple[Run | None, list[Run]]]:
    """Compose multiple transformers into a single transformer.

    Transformers are applied in order. If any transformer removes a node,
    subsequent transformers are skipped for that node.

    Args:
        *transformers: Variable number of transformer functions

    Returns:
        A composed transformer function

    Example:
        transformer = compose_transformers(
            skip_run_types({"RunnableLambda"}),
            filter_children_by_predicate(lambda r: r.run_type != "retriever")
        )
        transformed_run = transform_run_tree(run, transformer)

        # For depth limiting, use limit_run_tree_depth separately:
        transformed_run = transform_run_tree(run, transformer)
        limited_run = limit_run_tree_depth(transformed_run, max_depth=5)
    """
    def composed(run: Run, children: list[Run]) -> tuple[Run | None, list[Run]]:
        current_run = run
        current_children = children

        for transformer in transformers:
            if current_run is None:
                # If a previous transformer removed the node, just promote children
                return (None, current_children)
            current_run, current_children = transformer(current_run, current_children)

        return (current_run, current_children)
    return composed


def _run_to_dict(run: Run) -> dict:
    """Serialize a Run to dict, including child_runs.

    langsmith >= 0.7 marks child_runs with exclude=True in the Pydantic model,
    so model_dump() silently drops them. We re-add them manually.
    """
    d = run.model_dump()
    if run.child_runs:
        d["child_runs"] = [_run_to_dict(c) for c in run.child_runs]
    return d


class TestRunTracer(AsyncBaseTracer):
    def __init__(
        self,
        eval_runs_dir: Path,
        run_transformer: Callable[[Run, list[Run]], tuple[Run | None, list[Run]]] | None = None
    ):
        """Create a test run tracer that persists LangChain runs to disk.

        Args:
            eval_runs_dir: Directory to store run files
            run_transformer: Optional transformer function to modify the run tree before persisting.
                           Use helpers like skip_run_types(), filter_children_by_predicate(),
                           or compose_transformers().

        Example:
            # Skip RunnableLambda and RunnableSequence nodes
            tracer = TestRunTracer(
                eval_runs_dir=Path("./eval_runs"),
                run_transformer=skip_run_types({"RunnableLambda", "RunnableSequence"})
            )

            # Combine multiple transformations
            tracer = TestRunTracer(
                eval_runs_dir=Path("./eval_runs"),
                run_transformer=compose_transformers(
                    skip_run_types({"RunnableLambda"}),
                    filter_children_by_predicate(lambda r: r.run_type != "retriever")
                )
            )
        """
        super().__init__()
        self.eval_runs_dir = eval_runs_dir
        self.eval_runs_dir.mkdir(exist_ok=True)
        self.run_transformer = run_transformer

    # Implement abstract methods
    @override
    async def _persist_run(self, run: Run) -> None:
        metadata = run.extra["metadata"]
        required_metadata_keys = ["test_group_run_id", "test_suite", "test_case"]
        if not all(key in metadata for key in required_metadata_keys):
            logger.info("Run metadata missing required keys, skipping persist", metadata=metadata)
            return

        # Transform the run tree if a transformer is provided
        if self.run_transformer:
            run = transform_run_tree(run, self.run_transformer)
            if run is None:
                logger.warning("Root run was filtered out by transformer, skipping persist")
                return

        # Update the run file
        run_number = metadata.get("run_number", None)  # Extract run number (optional)
        run_file = get_test_run_file_path(
            self.eval_runs_dir,
            metadata["test_group_run_id"],
            metadata["test_suite"],
            metadata["test_case"],
            run_number=run_number
        )
        run_file.parent.mkdir(parents=True, exist_ok=True)

        if run_file.exists():
            logger.error(
                "Attempted to overwrite existing run file!",
                run_file=str(run_file),
                existing_size=run_file.stat().st_size,
                metadata=metadata,
            )
            raise FileExistsError(
                f"Run file already exists: {run_file}. "
                f"This suggests a duplicate persist for the same test case/run number. "
                f"Metadata: test_suite={metadata['test_suite']}, "
                f"test_case={metadata['test_case']}, "
                f"run_number={metadata.get('run_number')}, "
                f"group_run_id={metadata['test_group_run_id']}"
            )

        with run_file.open("w") as f:
            # Convert Run to dict and serialize with custom encoder that handles
            # UUIDs, datetimes, and LangChain objects (AIMessage, etc.)
            run_dict = _run_to_dict(run)
            f.write(json.dumps(run_dict, cls=RunEncoder, indent=2))
            f.flush()


def next_group_run_id(generation_runs_dir: Path) -> str:
    """Generate the next available group_run_id based on today's date.

    Scans existing generation_runs directories and auto-increments:
    2026-02-17, 2026-02-17-2, 2026-02-17-3, etc.
    """
    from datetime import datetime
    base = datetime.now().strftime("%Y-%m-%d")
    gen_dir = generation_runs_dir

    if not gen_dir.exists() or not (gen_dir / base).exists():
        return base

    # Find the next available suffix
    n = 2
    while (gen_dir / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"
