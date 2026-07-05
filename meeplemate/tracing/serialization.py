"""Serialization and tree-transformation helpers for LangChain ``Run`` trees.

These utilities are shared between the eval harness (which persists run trees to
disk) and the server (which persists them to a trace sink). They live here, rather
than in ``meeplemate.eval``, so that the server can reuse them without importing the
eval package (and its heavy ``deepeval`` dependency).
"""

from __future__ import annotations

import fnmatch
import json
from datetime import datetime
from typing import Callable
from uuid import UUID

from langchain_core.load import dumpd
from langchain_core.tracers import Run

# Type alias for a run-tree transformer: given a run and its already-transformed
# children, return the (possibly replaced/None) run and the children to attach.
RunTransformer = Callable[[Run, "list[Run]"], "tuple[Run | None, list[Run]]"]


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


def run_to_dict(run: Run) -> dict:
    """Serialize a Run to dict, including child_runs.

    langsmith >= 0.7 marks child_runs with exclude=True in the Pydantic model,
    so model_dump() silently drops them. We re-add them manually.
    """
    d = run.model_dump()
    if run.child_runs:
        d["child_runs"] = [run_to_dict(c) for c in run.child_runs]
    return d


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


def skip_run_types(skip_types: set[str]) -> Callable[[Run, list[Run]], tuple[Run | None, list[Run]]]:
    """Create a transformer that skips certain run types/names.

    Nodes matching the skip criteria are removed from the tree, but their children
    are promoted to the parent level, preserving the execution trace.

    Supports wildcard patterns using shell-style glob syntax:
    - `*` matches everything
    - `?` matches any single character
    - `[seq]` matches any character in seq
    - `[!seq]` matches any character not in seq

    Args:
        skip_types: Set of run names or run_type values to skip. Supports wildcards.

    Returns:
        A transformer function

    Example:
        # Exact match
        transformer = skip_run_types({"RunnableLambda", "RunnableSequence"})

        # Wildcard patterns
        transformer = skip_run_types({"Runnable*"})  # Matches RunnableLambda, RunnableSequence, etc.
        transformer = skip_run_types({"*Lambda", "*Sequence"})

        pruned_run = transform_run_tree(run, transformer)
    """
    def matches_any_pattern(value: str, patterns: set[str]) -> bool:
        """Check if value matches any pattern in the set."""
        for pattern in patterns:
            # Try exact match first (faster)
            if value == pattern:
                return True
            # Try wildcard match
            if fnmatch.fnmatch(value, pattern):
                return True
        return False

    def transformer(run: Run, children: list[Run]) -> tuple[Run | None, list[Run]]:
        if matches_any_pattern(run.name, skip_types) or matches_any_pattern(run.run_type, skip_types):
            # Skip this run, promote children
            return (None, children)
        return (run, children)
    return transformer
