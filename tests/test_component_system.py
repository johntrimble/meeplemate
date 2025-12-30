import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import AsyncIterator, ContextManager, Iterator, TypedDict

import pytest

from meeplemate.component_system import (
    normalized_component_graph,
    topological_sort,
    ServiceDescriptor,
    System,
)


def test_normalized_component_graph():
    @contextmanager
    def cluster() -> Iterator[None]:
        yield None

    @contextmanager
    def session(db) -> Iterator[None]:
        yield None
    
    @contextmanager
    def service(*, my_session) -> Iterator[None]:
        yield None
    
    @contextmanager
    def another_service(dependency, *, my_session) -> Iterator[None]:
        yield None
    
    components: dict[str, ServiceDescriptor] = {
        "cluster": (cluster, []),
        "session": (session, ["cluster"]),
        "my_service": (service, {"my_session": "session"}),
        "my_other_service": (another_service, ["my_service"], {"my_session": "session"}),
    }

    normalized = normalized_component_graph(components)
    expected = {
        "cluster": {
            "factory": cluster,
            "dependencies": [],
            "args": [],
            "kwargs": {}
        },
        "session": {
            "factory": session,
            "dependencies": [
                "cluster",
            ],
            "args": [
                "cluster",
            ],
            "kwargs": {}
        },
        "my_service": {
            "factory": service,
            "dependencies": [
                "session",
            ],
            "args": [],
            "kwargs": {
                "my_session": "session",
            }
        },
        "my_other_service": {
            "factory": another_service,
            "dependencies": [
                "my_service",
                "session",
            ],
            "args": [
                "my_service",
            ],
            "kwargs": {
                "my_session": "session",
            }
        },
    }

    assert normalized == expected


def test_topological_sort():
    @contextmanager
    def cluster() -> Iterator[None]:
        yield None

    @contextmanager
    def session(db) -> Iterator[None]:
        yield None
    
    @contextmanager
    def service(*, my_session) -> Iterator[None]:
        yield None
    
    @contextmanager
    def another_service(dependency, *, my_session) -> Iterator[None]:
        yield None
    
    components: dict[str, ServiceDescriptor] = {
        "cluster": (cluster, []),
        "my_other_service": (another_service, ["my_service"], {"my_session": "session"}),
        "my_service": (service, {"my_session": "session"}),
        "session": (session, ["cluster"]),
    }

    normalized = normalized_component_graph(components)
    sorted = topological_sort(normalized)

    assert sorted == ["cluster", "session", "my_service", "my_other_service"]


def test_async_system_supports_mixed_sync_and_async_components():
    events: list[str] = []

    @contextmanager
    def sync_dep() -> Iterator[str]:
        events.append("sync_enter")
        try:
            yield "sync_value"
        finally:
            events.append("sync_exit")

    @asynccontextmanager
    async def async_dep(sync_value) -> AsyncIterator[str]:
        events.append(f"async_enter:{sync_value}")
        try:
            yield "async_value"
        finally:
            events.append("async_exit")

    components: dict[str, ServiceDescriptor|AsyncServiceDescriptor] = {
        "sync_dep": (sync_dep, []),
        "async_dep": (async_dep, ["sync_dep"]),
    }

    async def run():
        async with System(components).astart() as system:
            assert system["sync_dep"] == "sync_value"
            assert system["async_dep"] == "async_value"

    asyncio.run(run())

    assert events == ["sync_enter", "async_enter:sync_value", "async_exit", "sync_exit"]


def test_validate_system_map() -> None:
    class Started(TypedDict):
        cluster: str
        session: str

    system_map = {
        "cluster": "cluster_value",
        "session": "session_value",
    }

    from meeplemate.component_system import validate_system_map
    validated = validate_system_map(system_map, Started)
    assert validated == system_map

    incomplete_system_map = {
        "cluster": "cluster_value",
    }

    with pytest.raises(ValueError):
        validate_system_map(incomplete_system_map, Started)


def test_validate_system_map_allows_arbitrary_types_without_modifying_typed_dict() -> None:
    class Service:
        pass

    class Started(TypedDict):
        service: Service

    service_instance = Service()
    system_map = {"service": service_instance}

    from meeplemate.component_system import validate_system_map

    validated = validate_system_map(system_map, Started)
    assert validated["service"] is service_instance
    assert not hasattr(Started, "__pydantic_config__")

    with pytest.raises(ValueError):
        validate_system_map({"service": "not-a-service"}, Started)


def test_system_validates_started_keys_with_typed_dict() -> None:
    class Started(TypedDict):
        cluster: str
        session: str

    @contextmanager
    def cluster() -> Iterator[str]:
        yield "cluster"

    @contextmanager
    def session(cluster) -> Iterator[str]:
        yield f"session-{cluster}"

    components: dict[str, ServiceDescriptor] = {
        "cluster": (cluster, []),
        "session": (session, ["cluster"]),
    }

    with System[Started](components).start() as system:
        assert system["cluster"] == "cluster"
        assert system["session"] == "session-cluster"

    class MissingKey(TypedDict):
        cluster: str
        session: str
        extra: str

    with pytest.raises(ValueError):
        with System[MissingKey](components).start():
            pass


def test_system_cleans_up_on_partial_start_failure() -> None:
    events: list[str] = []

    @contextmanager
    def good() -> Iterator[str]:
        events.append("good_enter")
        try:
            yield "good"
        finally:
            events.append("good_exit")

    @contextmanager
    def bad(good: str) -> Iterator[str]:
        events.append(f"bad_enter:{good}")
        raise RuntimeError("boom")
        yield "bad"  # pragma: no cover

    components: dict[str, ServiceDescriptor] = {
        "good": (good, []),
        "bad": (bad, ["good"]),
    }

    with pytest.raises(RuntimeError):
        with System(components).start():
            pass

    assert events == ["good_enter", "bad_enter:good", "good_exit"]
