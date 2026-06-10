from functools import reduce
import logging
import asyncio
import contextlib
import itertools
import json
from pathlib import Path
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Coroutine, Generator, Iterable, TextIO, Type, cast

from typing import Optional, Sequence, TypeVar, TypedDict
import yaml


def slurp_yaml(f):
    opened_file = False
    if hasattr(f, "read"):
        fp = f
    else:
        fp = open(f, "r")
        opened_file = True

    try:
        return yaml.safe_load(fp)
    finally:
        if opened_file:
            fp.close()


async def aslurp_yaml(path: Path | str) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, slurp_yaml, path)


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


async def aslurp_json(path: Path | str) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, slurp_json, path)


async def aslurp(path: Path | str) -> str:
    loop = asyncio.get_event_loop()
    def _read_file():
        with open(path, "r") as f:
            return f.read()
    return await loop.run_in_executor(None, _read_file)


async def aspit(text: str, path: Path | str) -> None:
    loop = asyncio.get_event_loop()
    def _write_file():
        with open(path, "w") as f:
            f.write(text)
    return await loop.run_in_executor(None, _write_file)


def spit_yaml(obj: Any, f: Path | str | TextIO) -> None:
    import yaml

    opened_file = False
    fp: TextIO
    if hasattr(f, "write"):
        # f is a file-like object (stream)
        fp = f  # type: ignore
    else:
        # f is a path (str or Path)
        assert isinstance(f, (str, Path))
        fp = open(f, "w")
        opened_file = True

    try:
        yaml.safe_dump(obj, fp)
    finally:
        if opened_file:
            fp.close()


async def aspit_yaml(obj: Any, path: Path | str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, spit_yaml, obj, path)


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


async def aspit_json(obj: Any, path: Path | str) -> None:
    loop = asyncio.get_event_loop()
    def _write_file():
        with open(path, "w") as f:
            json.dump(obj, f)
    await loop.run_in_executor(None, _write_file)


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


class _atee[Item]:
    def __init__(self, aiterable: AsyncIterable[Item]):
        if isinstance(aiterable, _atee):
            # Copy over internal state to avoid nesting
            self.aiterable = aiterable.aiterable
            self.buffer = aiterable.buffer
            self.lock = aiterable.lock
        else:
            self.aiterable: AsyncIterator[Item] = aiter(aiterable)
            self.buffer: list = [None, None]
            self.lock = asyncio.Lock()

    async def __anext__(self) -> Item:
        if self.buffer[1] is None:
            async with self.lock:
                if self.buffer[1] is None:
                    value = await anext(self.aiterable)
                    self.buffer[0] = value
                    self.buffer[1] = [None, None]
        value, self.buffer = self.buffer[0], self.buffer[1]
        return value
    
    def __aiter__(self) -> AsyncIterator[Item]:
        return self


def atee[Item](aiterable: AsyncIterable[Item], n:int=2) -> tuple[_atee[Item], ...]:
    t = _atee(aiterable)
    tees = tuple(_atee(t) for _ in range(n-1))
    return (t,) + tees


async def amap[R](func: Callable[..., R]|Callable[..., Coroutine[Any, Any, R]], *aiterables: AsyncIterable) -> AsyncIterator[R]:
    aiterators = [aiter(aiterable) for aiterable in aiterables]
    while True:
        args = []
        for aiterator in aiterators:
            try:
                arg = await anext(aiterator)
            except StopAsyncIteration:
                return
            args.append(arg)
        return_value = func(*args)
        if asyncio.iscoroutine(return_value):
            yield await return_value
        else:
            yield return_value


async def aenumerate[Item](aiterable: AsyncIterable[Item], start:int=0) -> AsyncIterator[tuple[int, Item]]:
    i = start
    async for item in aiter(aiterable):
        yield i, item
        i += 1


async def aislice[Item](iterable:AsyncIterable[Item], start:int, stop:int|None=None, step:int|None=None) -> AsyncIterator[Item]:
    if step is None:
        step = 1
    if stop is None:
        _stop = float('inf')
    else:
        _stop = stop
    
    iterator = aiter(iterable)

    async for i, item in aenumerate(iterator):
        if i >= _stop:
            break
        if i >= start and (i - start) % step == 0:
            yield item


def apairwise[Item](aiterable: AsyncIterable[Item]) -> AsyncIterator[tuple[Item, Item]]:
    i, j = atee(aiterable, 2)
    return amap(lambda x, y: (x, y), i, aislice(j, 1))


async def to_async_iter[Item](iterable: Iterable[Item]) -> AsyncIterator[Item]:
    for item in iterable:
        yield item


def arepeat[Item](value: Item, times: int | None = None) -> AsyncIterator[Item]:
    iter = itertools.repeat(value) if times is None else itertools.repeat(value, times)
    return to_async_iter(iter) # type: ignore


async def atakewhile[Item](predicate: Callable[[Item], bool], aiterable: AsyncIterable[Item]) -> AsyncIterator[Item]:
    async for item in aiterable:
        if predicate(item):
            yield item
        else:
            break


async def achain[Item](*aiterables: AsyncIterable[Item]) -> AsyncIterator[Item]:
    for aiterable in aiterables:
        async for item in aiterable:
            yield item


async def achain_from_aiterable[Item](aiterable_of_aiterables: AsyncIterable[AsyncIterable[Item]|Iterable[Item]]) -> AsyncIterator[Item]:
    async for aiterable in aiterable_of_aiterables:
        # If not an async iterable, but is a regular iterable, convert it
        if not isinstance(aiterable, AsyncIterable):
            aiterable = to_async_iter(aiterable)

        async for item in aiterable:
            yield item


async def sem_guard[Y,S,R](coro:Coroutine[Y,S,R], sem:asyncio.Semaphore) -> R:
    async with sem:
        return await coro


Reducer = Callable[[Any], Awaitable[None]]
Transducer = Callable[[Reducer], Reducer]

def xf_amap(fn: Callable[[Any], Awaitable[Any]]) -> Transducer:
    def xf(step: Reducer) -> Reducer:
        async def new_step(x: Any) -> None:
            y = await fn(x)
            await step(y)
        return new_step
    return xf

def xf_amapcat(fn: Callable[[Any], Awaitable[AsyncIterable[Any]]]) -> Transducer:
    def xf(step: Reducer) -> Reducer:
        async def new_step(x: Any) -> None:
            async for y in await fn(x):
                await step(y)
        return new_step
    return xf

async def pipeline(
    sink: Optional[asyncio.Queue],
    xf: Transducer,
    source: asyncio.Queue,
    *,
    concurrency: int = 10,
) -> None:
    # Our base reducer adds inputs to the sink queue
    async def base_reducer(item: Any) -> None:
        if sink is not None:
            await sink.put(item)

    # Now we apply the transducer to get the final reducer
    reducer = xf(base_reducer)
    
    async def worker() -> None:
        while True:
            try:
                item = await source.get()
                # If we get None, it's a signal from the producer that it is
                # done producing values. This means the worker should exit.
                # However, we still need to ensure all the other workers get the
                # signal too, so we re-enqueue the None to terminate the next
                # worker.
                if item is None:
                    await source.put(None)
                    break

                try:
                    await reducer(item)
                except Exception as e:
                    # We need to do something with this error. If we have a
                    # sink, just forward it on. Otherwise, log it. One downside
                    # of passing the exception on is that it means downstream
                    # consumers need to be able to handle exceptions as inputs.
                    if sink is not None:
                        await sink.put(e)
                    else:
                        logging.exception("Error in pipeline worker")
            finally:
                source.task_done()

    # Start the workers up based on the specified concurrency
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

    # Wait for all the workers to finish
    await asyncio.gather(*workers)

    # The last worker to exit will have re-enqueued the None sentinel, so we
    # need to remove it here to clean up.
    assert await source.get() is None, "Expected None sentinel at end of source queue"
    source.task_done()
    assert source.empty(), "Expected source queue to be empty after pipeline completion"

    # Now signal to the sink that we are done
    if sink is not None:
        await sink.put(None)


async def sink_into_queue(queue: asyncio.Queue, iterable: AsyncIterable[Any]|Iterable[Any], add_sentinel=True):
    # Is iterable not an async iterable? If so, wrap it
    if not isinstance(iterable, AsyncIterable):
        iterable = to_async_iter(iterable)

    async for item in aiter(iterable):
        await queue.put(item)
    if add_sentinel:
        await queue.put(None)  # Signal completion


async def queue_to_async_iter(queue: asyncio.Queue) -> AsyncIterator[Any]:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        yield item
        queue.task_done()


def compose(*functions):
  """
  Composes functions from right to left (mathematical composition order).
  f(g(h(x))) == compose(f, g, h)(x)
  """
  def inner(arg):
    return reduce(lambda acc, f: f(acc), reversed(functions), arg)
  return inner


def select_keys(d: Any, keys: Iterable[str]|Type) -> dict:
    if isinstance(keys, type) and hasattr(keys, "__annotations__"):
        _type: Type[dict] = keys
        keys = set()
        for base in _type.__mro__:
            if hasattr(base, "__annotations__"):
                keys.update(base.__annotations__.keys())

    keys = cast(Iterable[str], keys)
    return {k: d[k] for k in keys if k in d}


def slugify(text: str) -> str:
    # Lower case everything
    text = text.lower()
    # Replace spaces with hyphens
    text = text.replace(" ", "-")
    # Remove any characters that are not alphanumeric or hyphens
    text = ''.join(char for char in text if char.isalnum() or char == '-')
    return text


def snake_case(text: str) -> str:
    # Lower case everything
    text = text.lower()
    # Replace spaces with underscores
    text = text.replace(" ", "_")
    # Replace all punctuation with underscores
    punctuation = r'''!()-[]{};:'"\,<>./?@#$%^&*~'''
    for char in punctuation:
        text = text.replace(char, "_")
    # Dedupe underscores
    while "__" in text:
        text = text.replace("__", "_")
    # Remove trailing or leading underscores
    text = text.strip("_")
    # Remove any characters that are not alphanumeric or underscores
    text = ''.join(char for char in text if char.isalnum() or char == '_')
    return text


def load_template(template_path: str) -> str:
    import meeplemate
    path = Path(meeplemate.__path__[0]) / "prompts" / template_path
    return path.read_text()


def reorder_dict_by_typeddict(data, typeddict_class=None):
    """
    Recursively reorder dict/list structure to match TypedDict field definitions.

    Fields defined in the TypedDict are placed first in definition order.
    Extra fields not in the TypedDict are preserved and placed last (sorted alphabetically).

    The reordered dict will pass an equality test with the original dict (same keys/values).

    Args:
        data: The data to reorder (dict, list, or primitive)
        typeddict_class: The TypedDict class defining the expected structure

    Returns:
        Reordered data with dict keys matching TypedDict field order, extra keys at end

    Example:
        >>> from meeplemate.qa_graph import QaResponse
        >>> response = {...}  # Unordered dict, may have extra LLM-generated fields
        >>> ordered = reorder_dict_by_typeddict(response, QaResponse)
        >>> ordered == response  # True (same keys/values, different order)
    """
    from typing import get_type_hints, get_origin, get_args, Annotated

    # Handle non-dict cases
    if not isinstance(data, (dict, list)):
        return data

    if isinstance(data, list):
        # For lists, we don't know the item type unless passed in
        # So we just recurse without type info
        return [reorder_dict_by_typeddict(item, None) for item in data]

    # Must be a dict at this point
    if typeddict_class is None:
        # No type info, return as-is (or could try to infer)
        return data

    # Get type hints for the TypedDict
    hints = get_type_hints(typeddict_class, include_extras=True)

    # Get field order from __annotations__
    if not hasattr(typeddict_class, '__annotations__'):
        return data
    field_order = list(typeddict_class.__annotations__.keys())

    # First, add fields in the TypedDict-defined order
    reordered = {}
    for field in field_order:
        if field not in data:
            continue

        value = data[field]
        field_type = hints.get(field)

        if field_type is None:
            reordered[field] = value
            continue

        # Unwrap Annotated[Type, ...] -> Type
        if get_origin(field_type) is Annotated:
            field_type = get_args(field_type)[0]

        # Handle list[SomeTypedDict]
        if get_origin(field_type) is list:
            if not isinstance(value, list):
                reordered[field] = value
                continue

            item_type = get_args(field_type)[0] if get_args(field_type) else None

            # Check if item_type is a TypedDict
            if item_type and hasattr(item_type, '__annotations__'):
                reordered[field] = [
                    reorder_dict_by_typeddict(item, item_type)
                    for item in value
                ]
            else:
                reordered[field] = value

        # Handle nested TypedDict
        elif hasattr(field_type, '__annotations__'):
            reordered[field] = reorder_dict_by_typeddict(value, field_type)

        # Primitive types or unknown
        else:
            reordered[field] = value

    # Then, add any extra keys not in the TypedDict (preserves LLM-added fields)
    extra_keys = set(data.keys()) - set(field_order)
    for key in sorted(extra_keys):  # Sort extra keys alphabetically for consistency
        reordered[key] = data[key]

    return reordered


def serialize_typeddict(data: Any, typeddict_class, **json_kwargs) -> str:
    """
    Serialize a dict to JSON with field order matching TypedDict definition.

    Args:
        data: Dict to serialize (e.g., QaResponse instance)
        typeddict_class: The TypedDict class defining field order
        **json_kwargs: Additional arguments to pass to json.dumps()

    Returns:
        JSON string with fields in TypedDict definition order

    Example:
        >>> from meeplemate.qa_graph import QaResponse
        >>> response = {...}
        >>> json_str = serialize_typeddict(response, QaResponse, indent=2)
    """
    reordered = reorder_dict_by_typeddict(data, typeddict_class)
    return json.dumps(reordered, **json_kwargs)
