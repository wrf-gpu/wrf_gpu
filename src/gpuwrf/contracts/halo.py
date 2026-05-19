"""Single-GPU halo abstraction with future MPI-compatible call shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .state import State


@dataclass(frozen=True)
class HaloSpec:
    """Stores halo exchange metadata while M3 remains a single-GPU no-op."""

    width: int
    fields_to_exchange: tuple[str, ...]
    edge_type: Literal["periodic", "open", "nest_boundary"]

    def __post_init__(self) -> None:
        """Validates halo metadata once before timestep call sites use it."""

        if not 1 <= int(self.width) <= 4:
            raise ValueError("halo width must be in [1, 4]")
        if self.edge_type not in ("periodic", "open", "nest_boundary"):
            raise ValueError(f"unsupported halo edge_type {self.edge_type!r}")


def apply_halo(state: State, halo: HaloSpec) -> State:
    """Keeps future multi-GPU exchange call sites stable; M3 single-GPU returns state."""

    del halo
    return state
