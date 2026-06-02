"""Public M3 contracts for grid, state, halo, and precision."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import config


config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from .grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
    from .halo import HaloSpec
    from .physics_interfaces import PhysicsCarry, PhysicsDiagnostics, PhysicsStepResult, PhysicsStepSpec, PhysicsTendency
    from .physics_registry import NestFieldEntry, RegistryFieldSpec
    from .precision import DTypeRegistry
    from .state import State, Tendencies

__all__ = [
    "BCMetadata",
    "DEFAULT_DTYPES",
    "DTypeRegistry",
    "GridSpec",
    "HaloSpec",
    "NestFieldEntry",
    "PhysicsCarry",
    "PhysicsDiagnostics",
    "PhysicsStepResult",
    "PhysicsStepSpec",
    "PhysicsTendency",
    "Projection",
    "RegistryFieldSpec",
    "State",
    "Tendencies",
    "TerrainProvenance",
    "VerticalCoord",
    "apply_halo",
    "nest_field_list",
]


def __getattr__(name: str):
    """Lazily exposes contract symbols without importing executable submodules eagerly."""

    if name in {"BCMetadata", "GridSpec", "Projection", "TerrainProvenance", "VerticalCoord"}:
        from . import grid

        return getattr(grid, name)
    if name in {"HaloSpec", "apply_halo"}:
        from . import halo

        return getattr(halo, name)
    if name in {"PhysicsCarry", "PhysicsDiagnostics", "PhysicsStepResult", "PhysicsStepSpec", "PhysicsTendency"}:
        from . import physics_interfaces

        return getattr(physics_interfaces, name)
    if name in {"NestFieldEntry", "RegistryFieldSpec", "nest_field_list"}:
        from . import physics_registry

        return getattr(physics_registry, name)
    if name in {"DEFAULT_DTYPES", "DTypeRegistry"}:
        from . import precision

        return getattr(precision, name)
    if name in {"State", "Tendencies"}:
        from . import state

        return getattr(state, name)
    raise AttributeError(name)
