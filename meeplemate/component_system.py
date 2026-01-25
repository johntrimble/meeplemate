from contextlib import AbstractAsyncContextManager, ExitStack, AsyncExitStack, asynccontextmanager, contextmanager
from functools import partial
from typing import AsyncIterator, Awaitable, Callable, Collection, ContextManager, Iterator, Mapping, Self, Sequence, TypedDict, Tuple, Any, TypeVar, Type, Generic, cast, get_args


type ServiceDescriptor = Tuple[
    Callable[..., Any],
    Sequence[str]|Mapping[str, str]
] | Tuple[
    Callable[..., Any],
    Sequence[str],
    Mapping[str, str],
]


class NormalizedServiceDescriptor(TypedDict):
    factory: Callable[..., Any]
    dependencies: Sequence[str]
    args: Sequence[str]
    kwargs: Mapping[str, str]


SystemMapT = TypeVar("SystemMapT", bound=Mapping[str, Any])


class System(Generic[SystemMapT]):
    def __init__(self, components: Mapping[str, ServiceDescriptor]):
        self._components = components
        self._normalized_components = normalized_component_graph(components)
        validate_dependencies_exist(self._normalized_components)

    @classmethod
    def subsystem(cls, system: "System[Any]", names: Collection[str]) -> Self:
        subgraph = get_subgraph(system._normalized_components, names)
        components = {name: system._components[name] for name in subgraph.keys()}
        return cls(components)
 
    def _get_started_type(self) -> Type[SystemMapT]|None:
        orig = getattr(self, "__orig_class__", None)
        if orig is None:
            return None
        return get_args(orig)[0]
    
    @contextmanager
    def start(self) -> Iterator[SystemMapT]:
        with ExitStack() as stack:
            start_order = topological_sort(self._normalized_components)
            system_map: dict[str, Any] = {}
            for name in start_order:
                descriptor = self._normalized_components[name]
                factory = descriptor["factory"]
                args = [system_map[arg] for arg in descriptor["args"]]
                kwargs = {k: system_map[v] for k, v in descriptor["kwargs"].items()}
                component_cm = factory(*args, **kwargs)
                if is_sync_context_manager(component_cm):
                    component = stack.enter_context(component_cm)
                else:
                    # noop for non-context managers
                    component = component_cm
                # Remove system map entry on exit
                def _pop_entry(n: str) -> None:
                    system_map.pop(n, None)
                stack.callback(_pop_entry, name)
                system_map[name] = component

            yield validate_system_map(system_map, self._get_started_type())

    @asynccontextmanager
    async def astart(self) -> AsyncIterator[SystemMapT]:
        async with AsyncExitStack() as stack:
            start_order = topological_sort(self._normalized_components)
            system_map: dict[str, Any] = {}
            for name in start_order:
                descriptor = self._normalized_components[name]
                factory = descriptor["factory"]
                args = [system_map[arg] for arg in descriptor["args"]]
                kwargs = {k: system_map[v] for k, v in descriptor["kwargs"].items()}
                component_cm = factory(*args, **kwargs)
                if is_async_context_manager(component_cm):
                    component = await stack.enter_async_context(component_cm)  # type: ignore[arg-type]
                elif is_sync_context_manager(component_cm):
                    component = stack.enter_context(component_cm)  # type: ignore[arg-type]
                else:
                    component = component_cm
                # Remove system map entry on exit
                def _pop_entry(n: str) -> None:
                    system_map.pop(n, None)
                stack.callback(_pop_entry, name)
                system_map[name] = component

            yield validate_system_map(system_map, self._get_started_type())


class StartedSystem(Generic[SystemMapT]):
    system: System[SystemMapT]
    system_map: SystemMapT

    def __init__(self, system: System[SystemMapT], system_map: SystemMapT, stop: Callable[[], Any] | None = None, astop: Callable[[], Awaitable[Any]] | None = None):
        self.system = system
        self.system_map = system_map
        self._stop = stop
        self._astop = astop
    
    def stop(self) -> None:
        if self._stop is None:
            raise RuntimeError("Synchronous stop not implemented for this StartedSystem")
        self._stop()

    async def astop(self) -> None:
        if self._astop is None:
            raise RuntimeError("Asynchronous stop not implemented for this StartedSystem")
        await self._astop()


def extended_system(system: System[SystemMapT], extra_components: Mapping[str, ServiceDescriptor]) -> System[Mapping[str, Any]]:
    combined_components = {**system._components, **extra_components}
    return System[Mapping[str, Any]](combined_components)


def subsystem[SystemMapA: Mapping[str, Any]](system: System[SystemMapA], names: Collection[str]|None = None, extra_components: Mapping[str, ServiceDescriptor]|None = None) -> System[Mapping[str, Any]]:
    if extra_components is not None:
        new_system = extended_system(system, extra_components)
    else:
        new_system = system
    
    if names is not None:
        return System.subsystem(new_system, names)
    elif extra_components is not None:
        return System.subsystem(new_system, extra_components.keys())
    else:
        raise ValueError("Either names or extra_components must be provided")


def start_system(system: System[SystemMapT]) -> StartedSystem[SystemMapT]:
    cm = system.start()
    system_map = cm.__enter__()
    started = StartedSystem[SystemMapT](
        system=system,
        system_map=system_map,
        # __exit__ expects (exc_type, exc, tb); we pass Nones for a clean shutdown
        stop=lambda: cm.__exit__(None, None, None),
    )

    return started


def stop_system(started_system: StartedSystem[SystemMapT]) -> None:
    started_system.stop()


async def astart_system(system: System[SystemMapT]) -> StartedSystem[SystemMapT]:
    acm = system.astart()
    system_map = await acm.__aenter__()
    started = StartedSystem[SystemMapT](
        system=system,
        system_map=system_map,
        # __aexit__ expects (exc_type, exc, tb); we pass Nones for a clean shutdown
        astop=lambda: acm.__aexit__(None, None, None),
    )

    return started


async def astop_system(started_system: StartedSystem[SystemMapT]) -> None:
    await started_system.astop()


def is_sync_context_manager(obj: Any) -> bool:
    return hasattr(obj, "__enter__") and hasattr(obj, "__exit__")


def is_async_context_manager(obj: Any) -> bool:
    return hasattr(obj, "__aenter__") and hasattr(obj, "__aexit__")


def normalized_component_graph(components: Mapping[str, ServiceDescriptor]) -> Mapping[str, NormalizedServiceDescriptor]:
    normalized: dict[str, NormalizedServiceDescriptor] = {}
    for name, descriptor in components.items():
        factory = descriptor[0]
        if len(descriptor) == 2:
            if isinstance(descriptor[1], (list, tuple)):
                args: Sequence[str] = descriptor[1]
                kwargs: Mapping[str, str] = {}
            elif isinstance(descriptor[1], Mapping):
                args = []
                kwargs = descriptor[1]
            else:
                raise ValueError(f"Invalid service descriptor for {name}: {descriptor}")
        elif len(descriptor) == 3:
            args = descriptor[1]
            kwargs = descriptor[2]
        else:
            raise ValueError(f"Invalid service descriptor for {name}: {descriptor}")
        dependencies = list(args) + list(kwargs.values())
        normalized[name] = NormalizedServiceDescriptor(
            factory=factory,
            dependencies=dependencies,
            args=args,
            kwargs=kwargs,
        )
    return normalized


def get_subgraph(components: Mapping[str, NormalizedServiceDescriptor], names: Collection[str]) -> Mapping[str, NormalizedServiceDescriptor]:
    all_deps: set[str] = set()
    to_add = set(names)
    while len(to_add) > 0:
        name = to_add.pop()
        if name not in all_deps:
            all_deps.add(name)
            descriptor = components.get(name)
            if descriptor is None:
                raise ValueError(f"Component {name} not found in system")
            for dep in descriptor["dependencies"]:
                if dep not in all_deps:
                    to_add.add(dep)
    return {name: components[name] for name in all_deps}


def validate_dependencies_exist(components: Mapping[str, NormalizedServiceDescriptor]) -> None:
    for name, descriptor in components.items():
        for dep in descriptor["dependencies"]:
            if dep not in components:
                raise ValueError(f"Component {name} has unknown dependency {dep}")


def topological_sort(components: Mapping[str, NormalizedServiceDescriptor]) -> list[str]:
    from collections import defaultdict, deque

    in_degree = {name: 0 for name in components}
    graph = defaultdict(list)

    for name, descriptor in components.items():
        for dep in descriptor["dependencies"]:
            graph[dep].append(name)
            in_degree[name] += 1

    queue = deque([name for name, degree in in_degree.items() if degree == 0])
    sorted_order = []

    while queue:
        node = queue.popleft()
        sorted_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_order) != len(components):
        raise ValueError("Cycle detected in component dependencies")

    return sorted_order


def validate_system_map(system_map: Mapping[str, Any], started_type: Type[SystemMapT]|None) -> SystemMapT:
    if started_type is None:
        return cast(SystemMapT, system_map)

    # Only use pydantic if available
    try:
        from pydantic import ConfigDict, TypeAdapter
        from typing_extensions import is_typeddict
    except:
        TypeAdapter = None # type: ignore
        ConfigDict = None # type: ignore
        is_typeddict = None # type: ignore
    
    if TypeAdapter is None:
        return cast(SystemMapT, system_map)

    # If started_type is a TypedDict, patch in arbitrary_types_allowed via a lightweight subclass
    patched_type: Type[SystemMapT] = started_type
    if ConfigDict is not None and is_typeddict is not None and is_typeddict(started_type):
        patched_type = cast(
            Type[SystemMapT],
            type(
                f"_Patched{started_type.__name__}",
                (started_type,),
                {"__pydantic_config__": ConfigDict(arbitrary_types_allowed=True)},
            ),
        )

    adapter = TypeAdapter(patched_type)
    return adapter.validate_python(system_map)


ServiceT = TypeVar("ServiceT")

def factory(
    service_type: Callable[..., ServiceT|ContextManager[ServiceT]],
    start: Callable[[ServiceT], Any] | None = None,
    stop: Callable[[ServiceT], Any] | None = None,
    ignore_context_manager: bool = False,
) -> Callable[..., Callable[..., ContextManager[ServiceT]]]:
    @contextmanager
    def construct(*args, **kwargs) -> Iterator[ServiceT]:
        instance = service_type(*args, **kwargs)
        stack = ExitStack()
        with stack:
            # Add instance to the stack if it supports context management
            if isinstance(instance, ContextManager) and not ignore_context_manager:
                return_value = stack.enter_context(instance)
            else:
                return_value = instance
            
            return_value = cast(ServiceT, instance)

            # Add the stop callback to the stack if provided
            if stop:
                stack.callback(stop, return_value)

            if start:
                start(return_value)

            yield return_value


    def _receive_config(*args, **kwargs):
        return partial(construct, *args, **kwargs)
    return _receive_config


def afactory(
    service_type: Callable[..., ServiceT|AbstractAsyncContextManager[ServiceT]|ContextManager[ServiceT]],
    astart: Callable[[ServiceT], Awaitable[Any]] | None = None,
    astop: Callable[[ServiceT], Awaitable[Any]] | None = None,
    start: Callable[[ServiceT], Any] | None = None,
    stop: Callable[[ServiceT], Any] | None = None,
) -> Callable[..., Callable[..., AbstractAsyncContextManager[ServiceT]]]:
    @asynccontextmanager
    async def construct(*args, **kwargs) -> AsyncIterator[ServiceT]:
        instance = service_type(*args, **kwargs)
        stack = AsyncExitStack()
        async with stack:
            # Add instance to the stack if it supports async context management
            return_value: ServiceT
            if is_async_context_manager(instance):
                return_value = await stack.enter_async_context(instance)  # type: ignore[arg-type]
            elif is_sync_context_manager(instance):
                return_value = stack.enter_context(instance)  # type: ignore[arg-type]
            else:
                assert not isinstance(instance, (AbstractAsyncContextManager, ContextManager))
                return_value = instance

            # Add the astop callback to the stack if provided
            if astop:
                stack.push_async_callback(astop, return_value)
            if stop:
                stack.callback(stop, return_value)

            if astart:
                await astart(return_value)
            if start:
                start(return_value)

            yield return_value


    def _receive_config(*args, **kwargs):
        return partial(construct, *args, **kwargs)
    return _receive_config