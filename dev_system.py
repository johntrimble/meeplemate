from re import S
from typing import Any, cast
from meeplemate.config import Config, create_app_system
from meeplemate.component_system import (
    StartedSystem,
    System,
    subsystem,
    astart_system as _astart_system,
    astop_system as _astop_system
)

system = create_app_system(Config())

_started_system: tuple[StartedSystem, Any] | None = globals().get("_started_system")

async def astart(components=None):
    global system
    global _started_system

    import asyncio
    if components is not None:
        _system = subsystem(system, components)
    else:
        _system = system
    # TODO: Fix this typing issue properly, we should not use `cast` here
    started_system = await _astart_system(cast(System, _system))
    _started_system = (
        started_system,
        asyncio.get_event_loop(),
    )

async def astop():
    global system
    global _started_system
    if _started_system is not None:
        started_system, _ = _started_system
        if started_system is not None:
            await _astop_system(started_system)
            _started_system = None


async def areload(components=None):
    global system
    global _started_system

    # Get the components used if started_system exists
    if components is None and _started_system is not None:
        started_system, _ = _started_system
        components = list(started_system.system_map.keys())
    
    print("Reloading system...")
    await astop()
    await astart(components=components)
    print("System reloaded")


def get_service(name: str) -> Any:
    global system
    global _started_system

    if _started_system is None:
        raise RuntimeError("System is not started")
    
    started_system, _ = _started_system
    return started_system.system_map[name]

