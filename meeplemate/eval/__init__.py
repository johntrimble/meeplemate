from importlib import resources
from langchain.chat_models import BaseChatModel
from structlog import get_logger
from typing import IO, Iterator, Sequence, TypedDict, override, Callable
import gzip
import json

from langchain_core.tracers import Run
from meeplemate.util import slurp_yaml, snake_case

# Create tracer for persisting langchain runs during tests
from langchain_core.tracers.base import AsyncBaseTracer
from pathlib import Path

# Run-tree serialization/transform helpers now live in meeplemate.tracing so the
# server can reuse them without importing the eval package. Re-exported here for
# backward compatibility with existing eval call sites.
from meeplemate.tracing.serialization import (
    RunEncoder,
    compose_transformers,
    filter_children_by_predicate,
    limit_run_tree_depth,
    run_to_dict as _run_to_dict,
    skip_run_types,
    transform_run_tree,
)

from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset.golden import Golden

logger = get_logger(__name__)    

# Run files are written gzip-compressed: the traces are highly repetitive JSON and
# compress by better than 10x. Readers accept either form, so run groups generated
# before compression was introduced keep working untouched.
RUN_FILE_SUFFIX = ".json.gz"
UNCOMPRESSED_RUN_FILE_SUFFIX = ".json"
GZIP_MAGIC = b"\x1f\x8b"
COMPRESS_LEVEL = 6

def _load_test_suites() -> list:
    """Hand-written cases plus every promoted run.

    `test_cases.yaml` is curated by hand. `test_cases/<run-id>.yaml` is written
    by `mm-eval promote` from a review run and is regenerated wholesale, so the
    two are kept apart rather than merged into one file a person also edits.

    Sorted by filename so suite order is stable across machines; `iterdir()` is
    not ordered, and unstable order shows up as churn in eval output.
    """
    root = resources.files(__package__)
    suites = list(slurp_yaml(root.joinpath("test_cases.yaml")) or [])
    promoted_dir = root.joinpath("test_cases")
    if promoted_dir.is_dir():
        for entry in sorted(promoted_dir.iterdir(), key=lambda p: p.name):
            if entry.name.endswith((".yaml", ".yml")):
                suites.extend(slurp_yaml(entry) or [])
    return suites


test_suites = _load_test_suites()

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
        - run_number=None: eval_runs_dir/2026-01-25/test_suite__test_case.json.gz
        - run_number=1: eval_runs_dir/2026-01-25/test_suite__test_case.run001.json.gz
    """
    test_suite = snake_case(test_suite)
    test_case = snake_case(test_case)
    group_output_dir = eval_runs_dir / test_group_run_id
    group_output_dir.mkdir(exist_ok=True)

    # Construct filename with optional run suffix. This is the path new runs are
    # written to, so it is always the compressed spelling; resolve_run_file() maps
    # it back to whichever form is actually on disk.
    base_name = f"{test_suite}__{test_case}"
    if run_number is not None:
        filename = f"{base_name}.run{str(run_number).zfill(3)}{RUN_FILE_SUFFIX}"
    else:
        filename = f"{base_name}{RUN_FILE_SUFFIX}"

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


def uncompressed_run_file_path(run_file_path: Path) -> Path:
    """Strip the .gz from a run file path, if it has one."""
    if run_file_path.suffix == ".gz":
        return run_file_path.with_suffix("")
    return run_file_path


def existing_run_files(run_file_path: Path|str) -> list[Path]:
    """Every on-disk form of a run file, uncompressed first.

    A run is written as .json.gz but may exist as plain .json from before
    compression, and briefly as both if a compressed run was written alongside an
    old uncompressed one. Callers that delete a run need all of them; callers that
    read one want the first.
    """
    if isinstance(run_file_path, str):
        run_file_path = Path(run_file_path)

    uncompressed = uncompressed_run_file_path(run_file_path)
    compressed = uncompressed.with_name(uncompressed.name + ".gz")
    return [path for path in (uncompressed, compressed) if path.exists()]


def resolve_run_file(run_file_path: Path|str) -> Path|None:
    """Find the run file on disk, given either spelling of its path.

    Prefers the uncompressed file when a run exists in both forms. Returns None if
    neither exists, so callers can skip a missing run the way they skipped a failed
    .exists() before.
    """
    existing = existing_run_files(run_file_path)
    return existing[0] if existing else None


def open_run_file(run_file_path: Path) -> IO[str]:
    """Open a run file for reading as text, decompressing it if it is gzipped.

    Sniffs the magic bytes rather than trusting the extension, so a run file that
    was compressed or decompressed without being renamed still reads.
    """
    with run_file_path.open("rb") as f:
        is_gzipped = f.read(2) == GZIP_MAGIC

    if is_gzipped:
        return gzip.open(run_file_path, "rt", encoding="utf-8")
    return run_file_path.open("r", encoding="utf-8")


def load_persisted_run(run_file_path: Path|str) -> Run:
    """Load a persisted Run object from a run file.

    Args:
        run_file_path: Path to the run file, compressed (.json.gz) or not (.json).
            Either spelling resolves to whichever form is on disk.

    Returns:
        Run object reconstructed from the persisted data
    """
    resolved = resolve_run_file(run_file_path)
    if resolved is None:
        raise FileNotFoundError(f"No run file at {run_file_path}, with or without .gz")

    with open_run_file(resolved) as f:
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


class QuoteStatsCounts(TypedDict):
    generated: int
    first_pass_valid: int
    repaired: int
    lost: int
    units: int
    """Validation units that reported quote_stats."""
    unit_total: int
    """Validation units found at all. Exceeding `units` means the run predates the
    quote_stats output, so the counts are incomplete and must not be scored."""


def count_quote_stats(run: Run) -> QuoteStatsCounts:
    """Aggregate the per-pass quote accounting emitted by validate_and_fix_response.

    A request runs several independent validation units (one per subquestion, plus the
    combine stage), and each unit may validate several times as the loop retries. Within
    a unit, the *first* pass is the only one that describes what the model produced
    unaided, since later passes score a regenerated answer, while the *last* pass is the one
    whose repairs and losses actually reach the caller. So group by unit, then take the
    ends. Runs predating the `quote_stats` output contribute nothing rather than zeros,
    which would otherwise read as a perfect score.
    """
    by_unit: dict[str | None, list[Run]] = {}
    all_units: set[str] = set()
    for node in collect_runs_by_name_iter(run, "validate_and_fix_response"):
        all_units.add(str(node.parent_run_id))
        if not (node.outputs or {}).get("quote_stats"):
            continue
        by_unit.setdefault(str(node.parent_run_id), []).append(node)

    totals = QuoteStatsCounts(
        generated=0, first_pass_valid=0, repaired=0, lost=0,
        units=len(by_unit), unit_total=len(all_units),
    )
    for passes in by_unit.values():
        passes.sort(key=lambda r: r.start_time)
        first = (passes[0].outputs or {})["quote_stats"]
        last = (passes[-1].outputs or {})["quote_stats"]
        totals["generated"] += first["generated"]
        totals["first_pass_valid"] += first["first_pass_valid"]
        totals["repaired"] += last["repaired"]
        totals["lost"] += last["lost"]
    return totals


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

        existing_run_file = resolve_run_file(run_file)
        if existing_run_file is not None:
            logger.error(
                "Attempted to overwrite existing run file!",
                run_file=str(existing_run_file),
                existing_size=existing_run_file.stat().st_size,
                metadata=metadata,
            )
            raise FileExistsError(
                f"Run file already exists: {existing_run_file}. "
                f"This suggests a duplicate persist for the same test case/run number. "
                f"Metadata: test_suite={metadata['test_suite']}, "
                f"test_case={metadata['test_case']}, "
                f"run_number={metadata.get('run_number')}, "
                f"group_run_id={metadata['test_group_run_id']}"
            )

        with gzip.open(run_file, "wt", encoding="utf-8", compresslevel=COMPRESS_LEVEL) as f:
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
