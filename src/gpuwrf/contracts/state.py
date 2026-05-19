"""Device-resident prognostic state and tendency contracts for M3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp

from .grid import GridSpec
from .precision import DEFAULT_DTYPES


config.update("jax_enable_x64", True)


def _gpu_device() -> jax.Device:
    """Centralizes the mandatory GPU check used by constructors and self-test."""

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("State.zeros requires a GPU device; no JAX GPU backend is visible")
    return devices[0]


def _zeros(shape: tuple[int, ...], field: str, device: jax.Device) -> jax.Array:
    """Allocates one frozen state/tendency field on the selected GPU during init only."""

    return jax.device_put(jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(field)), device)


def _leaf_nbytes(leaves: Iterable[jax.Array]) -> int:
    """Computes persistent byte totals from pytree leaves for the spacetime budget."""

    return int(sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves))


@jax.tree_util.register_pytree_node_class
class Tendencies:
    """Pytree of preallocated tendency buffers matching every prognostic state field."""

    __slots__ = ("u", "v", "w", "theta", "qv", "p", "ph", "mu")

    def __init__(self, u, v, w, theta, qv, p, ph, mu) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.ph = ph
        self.mu = mu

    @classmethod
    def zeros(cls, grid: GridSpec) -> "Tendencies":
        """Allocates all tendency buffers once; reused by the timestep scan carry."""

        device = _gpu_device()
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        return cls(
            _zeros((nz, ny, nx + 1), "u", device),
            _zeros((nz, ny + 1, nx), "v", device),
            _zeros((nz + 1, ny, nx), "w", device),
            _zeros((nz, ny, nx), "theta", device),
            _zeros((nz, ny, nx), "qv", device),
            _zeros((nz, ny, nx), "p", device),
            _zeros((nz + 1, ny, nx), "ph", device),
            _zeros((ny, nx), "mu", device),
        )

    def replace(self, **updates) -> "Tendencies":
        """Returns an updated pytree with explicit field names; mirrors State.replace."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent tendency-buffer bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents tendency arrays as JAX scan carry leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds Tendencies after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class State:
    """Pytree of GPU-resident WRF-shaped prognostic fields."""

    __slots__ = ("u", "v", "w", "theta", "qv", "p", "ph", "mu")

    def __init__(self, u, v, w, theta, qv, p, ph, mu) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.ph = ph
        self.mu = mu

    @classmethod
    def zeros(cls, grid: GridSpec) -> "State":
        """Allocates the full M3 prognostic state once on the first visible GPU."""

        device = _gpu_device()
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        return cls(
            _zeros((nz, ny, nx + 1), "u", device),
            _zeros((nz, ny + 1, nx), "v", device),
            _zeros((nz + 1, ny, nx), "w", device),
            _zeros((nz, ny, nx), "theta", device),
            _zeros((nz, ny, nx), "qv", device),
            _zeros((nz, ny, nx), "p", device),
            _zeros((nz + 1, ny, nx), "ph", device),
            _zeros((ny, nx), "mu", device),
        )

    @classmethod
    def from_init(cls, grid: GridSpec, ic: Path) -> "State":
        """Keeps the future IC-loading call shape while M3 only supports zero init."""

        del ic
        return cls.zeros(grid)

    def replace(self, **updates) -> "State":
        """Returns an updated pytree with explicit field names for JAX functional steps."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent state bytes for the spacetime budget."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents state arrays as JAX scan carry leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds State after JAX transformations."""

        del aux
        return cls(*children)


def _self_test() -> None:
    """Runs a small allocation check for the sprint validation command."""

    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    leaves, _ = jax.tree_util.tree_flatten((state, tendencies))
    platforms = {leaf.devices().pop().platform for leaf in leaves}
    if platforms != {"gpu"}:
        raise RuntimeError(f"expected all leaves on gpu, got {sorted(platforms)}")
    print(f"ok state_bytes={state.bytes()} tendency_bytes={tendencies.bytes()} device=gpu")


def main(argv: list[str] | None = None) -> int:
    """CLI exists only to serve the contract's explicit self-test command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
