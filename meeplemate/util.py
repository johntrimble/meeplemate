import contextlib
import json
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, TextIO

from literalai import TypeVar
from pyparsing import Sequence

def dump_jsonl(obj_list, fp):
    for obj in obj_list:
        print(json.dumps(obj), file=fp)


def spit_jsonl(obj_list, f, flush_every_line=False):
    opened_file = False
    if hasattr(f, "write"):
        fp = f
    else:
        fp = open(f, "w")
        opened_file = True
    
    try:
        for obj in obj_list:
            print(json.dumps(obj), file=fp)
            if flush_every_line:
                fp.flush()
    finally:
        if opened_file:
            fp.close()


def slurp_jsonl(f, up_to_first_error=False):
    opened_file = False
    if hasattr(f, "read"):
        fp = f
    else:
        fp = open(f, "r")
        opened_file = True
    
    try:
        result = []
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result.append(obj)
            except json.JSONDecodeError:
                if up_to_first_error:
                    break
                else:
                    raise
        return result
    finally:
        if opened_file:
            fp.close()


def slurp_json(f):
    opened_file = False
    if hasattr(f, "read"):
        fp = f
    else:
        fp = open(f, "r")
        opened_file = True

    try:
        return json.load(fp)
    finally:
        if opened_file:
            fp.close()


def spit_json(obj, f):
    opened_file = False
    if hasattr(f, "write"):
        fp = f
    else:
        fp = open(f, "w")
        opened_file = True
    
    try:
        return json.dump(obj, fp)
    finally:
        if opened_file:
            fp.close()


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='.'):
    result_dict = {}
    for k, v in d.items():
        keys = k.split(sep)
        current_level = result_dict
        for part in keys[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[keys[-1]] = v
    return result_dict

Item = TypeVar('Item')

def partition_by(sequence: Sequence[Item], key_func: Callable[[Item], Any]) -> Sequence[Sequence[Item]]:
    sentinel = object()
    current_delimiter = sentinel
    results = []
    for item in sequence:
        delimiter = key_func(item)
        if delimiter != current_delimiter:
            current_delimiter = delimiter
            results.append([])
        results[-1].append(item)
    return results


def get_done_path(base_path: Path | str) -> Path:
    base_path = Path(base_path)
    return base_path.parent / (base_path.name + ".done")


def load_or_build_and_load_jsonl(
    jsonl_path: Path | str,
    build_func: Callable[[], Generator[Item, None, None] | Iterable[Item]],
    total=None,
    show_progress=None,
) -> Sequence[Item]:
    if total is not None:
        if show_progress is None:
            show_progress = True

    if show_progress is None:
        show_progress = False

    jsonl_path = Path(jsonl_path)
    done_path = get_done_path(jsonl_path)

    if not (jsonl_path.exists() and done_path.exists()):
        if show_progress:
            from tqdm import tqdm
            original_build_func = build_func
            build_func = lambda: tqdm(original_build_func(), total=total)
        spit_jsonl(build_func(), jsonl_path, flush_every_line=True)
        done_path.touch()

    return slurp_jsonl(jsonl_path)
    