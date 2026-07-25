#!/usr/bin/env python
"""Count LLM token spend in eval generation-run traces.

Usage:
    python script/count_tokens.py <run-group-or-path> [<path>...] [mode]

    <run-group-or-path>  Run group name (e.g. 2026-07-07) or path(s) to JSON file(s)
    --per-run            One row per run file, grouped by test case
    --per-call           One row per LLM call, in trace order
    --by-node            Aggregate by LangGraph node

The default view answers "what does this question cost to answer?": one row per
test case, with per-run averages and the spread across that case's runs. Every
mode ends with an OVERALL row (averages per run) and a TOTAL row (absolute sums).

Only ``run_type == "llm"`` spans are counted, and traversal never descends below
one. The same AIMessage is echoed into the outputs of every enclosing chain and
graph node, so a naive walk over the JSON multi-counts it several-fold. This
also keeps FailoverChatModel honest: bound calls (_FailoverRunnable) emit one
span per child attempt, and unbound calls share the parent's run_manager so the
child emits no span -- either way, one span per attempt, never nested.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_GROUPS = PROJECT_ROOT / "data" / "evals" / "generation_runs"

# Prices used by calc_token_limits() in meeplemate/server/rate_limit.py.
# Qwen3 30b a3b token costs
# COST_PER_M_INPUT = 0.10
# COST_PER_M_OUTPUT = 0.30
# Qwen3.6 35b a3b token costs
COST_PER_M_INPUT = 0.14
COST_PER_M_OUTPUT = 1.00


def cost(input_tokens: float, output_tokens: float) -> float:
    return (input_tokens * COST_PER_M_INPUT + output_tokens * COST_PER_M_OUTPUT) / 1_000_000


@dataclass
class Call:
    """One provider call: an ``llm`` span with its reported usage."""
    run: str
    node: str
    input_tokens: int
    output_tokens: int


@dataclass
class Run:
    """One generation run -- the full cost of answering the question once."""
    run: str
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def case(self) -> str:
        return self.run.rsplit(".run", 1)[0]

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def usage(span: dict) -> tuple[int, int]:
    """(input, output) tokens for an llm span.

    Streaming responses report usage only via the message's usage_metadata;
    non-streaming ones also carry it as token_usage. Prefer the former.
    """
    outputs = span.get("outputs") or {}
    generations = outputs.get("generations") or []
    generation = generations[0][0] if generations and isinstance(generations[0], list) else {}
    message = generation.get("message") or {}
    message = message.get("kwargs", message)

    um = message.get("usage_metadata") or {}
    if um:
        return int(um.get("input_tokens") or 0), int(um.get("output_tokens") or 0)

    tu = (message.get("response_metadata") or {}).get("token_usage") or (
        outputs.get("llm_output") or {}
    ).get("token_usage") or {}
    return int(tu.get("prompt_tokens") or 0), int(tu.get("completion_tokens") or 0)


def collect(span: dict, run: str) -> list[Call]:
    if span.get("run_type") == "llm":
        node = ((span.get("extra") or {}).get("metadata") or {}).get("langgraph_node") or "-"
        return [Call(run, node, *usage(span))]
    calls = []
    for child in span.get("child_runs") or []:
        calls.extend(collect(child, run))
    return calls


def resolve(arg: str) -> list[Path]:
    """A path (file or directory) or a run group name under data/evals/generation_runs."""
    for candidate in (Path(arg), RUN_GROUPS / arg):
        if candidate.is_dir():
            return sorted(candidate.rglob("*.json"))
        if candidate.is_file():
            return [candidate]
    sys.exit(f"Error: could not find '{arg}' as a path or run group name")


def to_runs(calls: list[Call]) -> list[Run]:
    runs: dict[str, Run] = {}
    for call in calls:
        run = runs.setdefault(call.run, Run(call.run))
        run.input_tokens += call.input_tokens
        run.output_tokens += call.output_tokens
        run.calls += 1
    return list(runs.values())


class Table:
    """Left-aligned key column sized to the data, right-aligned numeric columns."""

    def __init__(self, key_header: str, columns: list[tuple[str, int]]) -> None:
        self.key_header = key_header
        self.columns = columns
        self.rows: list[tuple[str, list[str]]] = []

    def add(self, key: str, *values: str) -> None:
        self.rows.append((key, [str(v) for v in values]))

    def print(self, footers: list[tuple[str, list[str]]] = ()) -> None:
        width = max([len(self.key_header)] + [len(k) for k, _ in [*self.rows, *footers]])
        header = "".join(f"{name:>{w + 1}}" for name, w in self.columns)
        print(f"{self.key_header:<{width}}{header}")
        for key, values in [*self.rows, *footers]:
            cells = "".join(f"{v:>{w + 1}}" for v, (_, w) in zip(values, self.columns))
            print(f"{key:<{width}}{cells}")


def run_stats(runs: list[Run]) -> list[str]:
    """Per-run averages plus the spread of total tokens across the runs."""
    totals = sorted(r.total for r in runs)
    n = len(runs)
    mean_in = sum(r.input_tokens for r in runs) / n
    mean_out = sum(r.output_tokens for r in runs) / n
    return [
        f"{n}",
        f"{round(mean_in):,}",
        f"{round(mean_out):,}",
        f"{round(mean_in + mean_out):,}",
        f"{totals[0]:,}",
        f"{round(statistics.median(totals)):,}",
        f"{totals[-1]:,}",
        f"{sum(r.calls for r in runs) / n:.1f}",
        f"${cost(mean_in, mean_out):.4f}",
    ]


STAT_COLUMNS = [
    ("runs", 5), ("in/run", 9), ("out/run", 8), ("tok/run", 9),
    ("min", 9), ("med", 9), ("max", 9), ("calls", 6), ("$/run", 8),
]


def totals_footer(runs: list[Run]) -> list[tuple[str, list[str]]]:
    """OVERALL = averages per run across everything; TOTAL = absolute sums."""
    tin = sum(r.input_tokens for r in runs)
    tout = sum(r.output_tokens for r in runs)
    calls = sum(r.calls for r in runs)
    return [
        ("OVERALL", run_stats(runs)),
        ("TOTAL", [
            f"{len(runs)}", f"{tin:,}", f"{tout:,}", f"{tin + tout:,}",
            "", "", "", f"{calls}", f"${cost(tin, tout):.2f}",
        ]),
    ]


def by_case(runs: list[Run]) -> None:
    cases: dict[str, list[Run]] = {}
    for run in runs:
        cases.setdefault(run.case, []).append(run)

    table = Table("test_case", STAT_COLUMNS)
    for case, case_runs in sorted(
        cases.items(), key=lambda kv: sum(r.total for r in kv[1]) / len(kv[1]), reverse=True
    ):
        table.add(case, *run_stats(case_runs))
    table.print(totals_footer(runs))


def per_run(runs: list[Run]) -> None:
    table = Table("run", [
        ("in", 9), ("out", 8), ("total", 9), ("calls", 6), ("cost", 8),
    ])
    for run in sorted(runs, key=lambda r: (r.case, r.run)):
        table.add(
            run.run,
            f"{run.input_tokens:,}", f"{run.output_tokens:,}", f"{run.total:,}",
            f"{run.calls}", f"${cost(run.input_tokens, run.output_tokens):.4f}",
        )
    tin = sum(r.input_tokens for r in runs)
    tout = sum(r.output_tokens for r in runs)
    table.print([("TOTAL", [
        f"{tin:,}", f"{tout:,}", f"{tin + tout:,}",
        f"{sum(r.calls for r in runs)}", f"${cost(tin, tout):.2f}",
    ])])


def by_node(calls: list[Call], runs: list[Run]) -> None:
    nodes: dict[str, list[int]] = {}
    for call in calls:
        entry = nodes.setdefault(call.node, [0, 0, 0])
        entry[0] += call.input_tokens
        entry[1] += call.output_tokens
        entry[2] += 1

    n = len(runs)
    table = Table("node", [
        ("in", 10), ("out", 9), ("total", 10), ("calls", 6),
        ("calls/run", 9), ("tok/run", 9), ("$/run", 8),
    ])
    for node, (tin, tout, count) in sorted(nodes.items(), key=lambda kv: -sum(kv[1][:2])):
        table.add(
            node, f"{tin:,}", f"{tout:,}", f"{tin + tout:,}", f"{count}",
            f"{count / n:.1f}", f"{round((tin + tout) / n):,}", f"${cost(tin, tout) / n:.4f}",
        )
    tin = sum(r.input_tokens for r in runs)
    tout = sum(r.output_tokens for r in runs)
    table.print([("TOTAL", [
        f"{tin:,}", f"{tout:,}", f"{tin + tout:,}", f"{sum(r.calls for r in runs)}",
        f"{sum(r.calls for r in runs) / n:.1f}", f"{round((tin + tout) / n):,}",
        f"${cost(tin, tout) / n:.4f}",
    ])])


def per_call(calls: list[Call]) -> None:
    width = max(len(c.run) for c in calls)
    print(f"{'run':<{width}} {'node':<25} {'in':>9} {'out':>8}")
    for call in calls:
        print(f"{call.run:<{width}} {call.node:<25} "
              f"{call.input_tokens:>9,} {call.output_tokens:>8,}")


def main(argv: list[str]) -> None:
    mode = "by-case"
    targets: list[Path] = []
    for arg in argv:
        if arg in ("--per-run", "--per-call", "--by-node"):
            mode = arg[2:]
        elif arg in ("--help", "-h") or arg.startswith("-"):
            sys.exit(__doc__)
        else:
            targets.extend(resolve(arg))

    if not targets:
        sys.exit(__doc__)

    calls: list[Call] = []
    for path in targets:
        with path.open() as fh:
            calls.extend(collect(json.load(fh), path.stem))

    if not calls:
        sys.exit("No LLM calls found.")

    runs = to_runs(calls)
    if mode == "per-call":
        per_call(calls)
    elif mode == "per-run":
        per_run(runs)
    elif mode == "by-node":
        by_node(calls, runs)
    else:
        by_case(runs)


if __name__ == "__main__":
    main(sys.argv[1:])
