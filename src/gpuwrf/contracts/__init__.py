"""Public M3 contracts for grid, state, halo, and precision."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
    from .halo import HaloSpec
    from .precision import DTypeRegistry
    from .state import State, Tendencies

__all__ = [
    "BCMetadata",
    "DEFAULT_DTYPES",
    "DTypeRegistry",
    "GridSpec",
    "HaloSpec",
    "Projection",
    "State",
    "Tendencies",
    "TerrainProvenance",
    "VerticalCoord",
    "apply_halo",
]


def __getattr__(name: str):
    """Lazily exposes contract symbols without importing executable submodules eagerly."""

    if name in {"BCMetadata", "GridSpec", "Projection", "TerrainProvenance", "VerticalCoord"}:
        from . import grid

        return getattr(grid, name)
    if name in {"HaloSpec", "apply_halo"}:
        from . import halo

        return getattr(halo, name)
    if name in {"DEFAULT_DTYPES", "DTypeRegistry"}:
        from . import precision

        return getattr(precision, name)
    if name in {"State", "Tendencies"}:
        from . import state

        return getattr(state, name)
    raise AttributeError(name)
