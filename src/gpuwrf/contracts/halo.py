"""Single-GPU halo abstraction with future MPI-compatible call shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .state import State


@dataclass(frozen=True)
class HaloSpec:
    """Stores halo exchange metadata while M3 remains a single-GPU no-op."""

    width: int
    fields_to_exchange: tuple[str, ...]
    edge_type: Literal["periodic", "open", "nest_boundary"]
    sharding: Any = None

    def __post_init__(self) -> None:
        """Validates halo metadata once before timestep call sites use it."""

        if not 1 <= int(self.width) <= 8:
            raise ValueError("halo width must be in [1, 8]")
        if self.edge_type not in ("periodic", "open", "nest_boundary"):
            raise ValueError(f"unsupported halo edge_type {self.edge_type!r}")


def apply_halo(state: State, halo: HaloSpec) -> State:
    """Keeps future multi-GPU exchange call sites stable; M3 single-GPU returns state."""

    if halo.sharding is not None and bool(getattr(halo.sharding, "enabled", False)):
        from gpuwrf.runtime.sharding import exchange_state_halos

        return exchange_state_halos(state, halo)
    return state
