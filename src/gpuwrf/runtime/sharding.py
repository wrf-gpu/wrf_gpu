"""Optional multi-device sharding controls for DGX-scale runs.

The default path is deliberately host-level and inert: with
``ShardingConfig.enabled=False`` the selected runner is exactly
``run_forecast_operational``. That preserves the current single-GPU compiled graph
for users who do not opt in to sharding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal
import hashlib
import re

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State


ShardAxis = Literal["x", "y"]
ShardBackend = Literal["pmap"]

COLLECTIVE_TOKENS: tuple[str, ...] = (
    "collective-permute",
    "all-gather",
    "all-reduce",
    "all-to-all",
    "partition-id",
    "replica-id",
)


@dataclass(frozen=True)
class ShardingConfig:
    """Static opt-in configuration for domain decomposition."""

    enabled: bool = False
    axis: ShardAxis = "x"
    num_partitions: int | None = None
    halo_width: int = 2
    backend: ShardBackend = "pmap"
    axis_name: str = "shard"
    multi_node: bool = False
    coordinator_address: str | None = None
    process_id: int = 0
    process_count: int = 1

    def __post_init__(self) -> None:
        if self.axis not in ("x", "y"):
            raise ValueError(f"unsupported shard axis {self.axis!r}")
        if self.backend != "pmap":
            raise ValueError(f"unsupported sharding backend {self.backend!r}")
        if not self.axis_name:
            raise ValueError("axis_name must be nonempty")
        if not 1 <= int(self.halo_width) <= 4:
            raise ValueError("halo_width must be in [1, 4]")
        if self.num_partitions is not None and int(self.num_partitions) < 1:
            raise ValueError("num_partitions must be positive when supplied")
        if int(self.process_count) < 1:
            raise ValueError("process_count must be positive")
        if int(self.process_id) < 0 or int(self.process_id) >= int(self.process_count):
            raise ValueError("process_id must be in [0, process_count)")
        if self.multi_node and not self.coordinator_address:
            raise ValueError("multi_node sharding requires coordinator_address")

    @classmethod
    def disabled(cls) -> "ShardingConfig":
        """Return the canonical default-off config."""

        return cls(enabled=False)

    def resolved_partitions(self, *, platform: str | None = None) -> int:
        """Return the local partition count this process should use."""

        if self.num_partitions is not None:
            return int(self.num_partitions)
        devices = jax.devices(platform) if platform is not None else jax.local_devices()
        return max(1, len(devices))


def select_forecast_runner(config: ShardingConfig | None = None) -> Callable[..., Any]:
    """Select the public forecast runner for an optional sharding config.

    Disabled sharding returns the existing default function object, not a jitted
    wrapper. This is the zero-overhead contract.
    """

    from gpuwrf.runtime.operational_mode import run_forecast_operational

    cfg = ShardingConfig.disabled() if config is None else config
    if not bool(cfg.enabled):
        return run_forecast_operational
    return run_forecast_operational_sharded


def run_forecast_operational_optional_sharding(
    state,
    namelist,
    hours: float,
    *,
    sharding: ShardingConfig | None = None,
):
    """Host-level optional entry point.

    With the default config this calls the exact existing compiled entry point.
    """

    return select_forecast_runner(sharding)(state, namelist, hours)


def run_forecast_operational_sharded(state, namelist, hours: float):
    """Opt-in sharded runtime entry point.

    S2/S3/S4 fill this in behind explicit sharding gates. It is intentionally not
    reachable from the default path.
    """

    del state, namelist, hours
    raise NotImplementedError("sharded runtime execution is opt-in and implemented after halo/operator gates")


def hlo_operation_count(hlo_text: str) -> int:
    """Count HLO operation definition lines in a backend-stable way."""

    count = 0
    for line in hlo_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        if re.search(r"\b[-%.A-Za-z0-9_]+ = ", stripped):
            count += 1
    return count


def hlo_collective_counts(hlo_text: str) -> dict[str, int]:
    """Count collective/SPMD tokens that must be absent from flag-off HLO."""

    lower = hlo_text.lower()
    return {token: lower.count(token) for token in COLLECTIVE_TOKENS}


def hlo_sha256(hlo_text: str) -> str:
    """Stable hash for proof objects."""

    return hashlib.sha256(hlo_text.encode("utf-8")).hexdigest()


def hlo_graph_stats(hlo_text: str) -> dict[str, Any]:
    """Return the graph stats used by the disabled-sharding proof."""

    collectives = hlo_collective_counts(hlo_text)
    return {
        "op_count": int(hlo_operation_count(hlo_text)),
        "hlo_sha256": hlo_sha256(hlo_text),
        "hlo_size_bytes": len(hlo_text.encode("utf-8")),
        "collective_counts": collectives,
        "collectives_present": any(count > 0 for count in collectives.values()),
    }


def assert_flag_off_graph_unchanged(reference_hlo: str, candidate_hlo: str) -> None:
    """Raise if disabled sharding changes op count or introduces collectives."""

    reference = hlo_graph_stats(reference_hlo)
    candidate = hlo_graph_stats(candidate_hlo)
    if int(reference["op_count"]) != int(candidate["op_count"]):
        raise AssertionError(
            f"flag-off op count changed: {reference['op_count']} != {candidate['op_count']}"
        )
    if bool(candidate["collectives_present"]):
        raise AssertionError(f"flag-off HLO contains collectives: {candidate['collective_counts']}")


def x_partition_bounds(nx: int, num_partitions: int) -> tuple[tuple[int, int], ...]:
    """Return equal x-domain intervals as ``(start, end_exclusive)``."""

    nx = int(nx)
    n = int(num_partitions)
    if n < 1:
        raise ValueError("num_partitions must be positive")
    if nx % n != 0:
        raise ValueError(f"nx={nx} must be divisible by num_partitions={n}")
    width = nx // n
    return tuple((rank * width, (rank + 1) * width) for rank in range(n))


def _periodic_indices(start: int, stop: int, size: int) -> jax.Array:
    return jnp.mod(jnp.arange(int(start), int(stop), dtype=jnp.int32), int(size))


def _take_x_periodic(array: jax.Array, start: int, stop: int, *, period: int) -> jax.Array:
    indices = _periodic_indices(start, stop, period)
    source = array[..., :period]
    return jnp.take(source, indices, axis=-1)


def _take_x_face_periodic(array: jax.Array, start: int, stop: int, *, nx: int) -> jax.Array:
    raw = jnp.arange(int(start), int(stop), dtype=jnp.int32)
    wrapped = jnp.mod(raw, int(nx))
    indices = jnp.where(raw == int(nx), int(nx), wrapped)
    return jnp.take(array, indices, axis=-1)


def _x_kind(name: str, array: jax.Array, grid) -> str:
    if name.endswith("_bdy"):
        return "replicated"
    if name == "u" and array.ndim >= 1 and int(array.shape[-1]) == int(grid.nx) + 1:
        return "x_face"
    if array.ndim >= 1 and int(array.shape[-1]) == int(grid.nx):
        return "x_mass"
    return "replicated"


def _partition_leaf_x(
    name: str,
    array: jax.Array,
    *,
    grid,
    bounds: tuple[tuple[int, int], ...],
    halo_width: int,
    fill_halos: bool,
) -> jax.Array:
    kind = _x_kind(name, array, grid)
    if kind == "replicated":
        return jnp.stack([array for _ in bounds], axis=0)

    chunks = []
    h = int(halo_width)
    for start, end in bounds:
        if kind == "x_face":
            # Work over nx unique periodic faces; each shard owns L+1 faces by
            # appending the right boundary face, then optional halo slots. The
            # stored global nx face is preserved for exact host partition/merge.
            interior = _take_x_face_periodic(array, start, end + 1, nx=int(grid.nx))
            if h:
                if fill_halos:
                    left = _take_x_face_periodic(array, start - h, start, nx=int(grid.nx))
                    right = _take_x_face_periodic(array, end + 1, end + 1 + h, nx=int(grid.nx))
                else:
                    left = jnp.zeros(array.shape[:-1] + (h,), dtype=array.dtype)
                    right = jnp.zeros(array.shape[:-1] + (h,), dtype=array.dtype)
                interior = jnp.concatenate((left, interior, right), axis=-1)
            chunks.append(interior)
        else:
            interior = array[..., start:end]
            if h:
                if fill_halos:
                    left = _take_x_periodic(array, start - h, start, period=int(grid.nx))
                    right = _take_x_periodic(array, end, end + h, period=int(grid.nx))
                else:
                    left = jnp.zeros(array.shape[:-1] + (h,), dtype=array.dtype)
                    right = jnp.zeros(array.shape[:-1] + (h,), dtype=array.dtype)
                interior = jnp.concatenate((left, interior, right), axis=-1)
            chunks.append(interior)
    return jnp.stack(chunks, axis=0)


def partition_state_x(
    state: State,
    grid,
    *,
    num_partitions: int,
    halo_width: int = 0,
    fill_halos: bool = True,
) -> State:
    """Split a global State into a leading device-axis x-domain State."""

    bounds = x_partition_bounds(int(grid.nx), int(num_partitions))
    values = {
        name: _partition_leaf_x(
            name,
            getattr(state, name),
            grid=grid,
            bounds=bounds,
            halo_width=int(halo_width),
            fill_halos=bool(fill_halos),
        )
        for name in State.__slots__
    }
    return State(**values)


def _strip_x_halo(array: jax.Array, halo_width: int) -> jax.Array:
    h = int(halo_width)
    if h <= 0:
        return array
    return array[..., h:-h]


def _merge_leaf_x(name: str, array: jax.Array, *, grid, halo_width: int) -> jax.Array:
    del grid
    if name.endswith("_bdy"):
        kind = "replicated"
    elif name == "u":
        kind = "x_face"
    else:
        kind = "x_mass"
    if kind == "replicated":
        return array[0]
    pieces = [_strip_x_halo(array[i], halo_width) for i in range(int(array.shape[0]))]
    if kind == "x_face":
        body = [piece[..., :-1] for piece in pieces[:-1]]
        body.append(pieces[-1])
        return jnp.concatenate(body, axis=-1)
    return jnp.concatenate(pieces, axis=-1)


def merge_state_x(sharded_state: State, grid, *, halo_width: int = 0) -> State:
    """Merge a leading device-axis x-domain State back to global layout."""

    values = {
        name: _merge_leaf_x(name, getattr(sharded_state, name), grid=grid, halo_width=int(halo_width))
        for name in State.__slots__
    }
    return State(**values)


def exchange_periodic_halo_x(
    field: jax.Array,
    *,
    width: int,
    num_partitions: int,
    axis_name: str = "shard",
) -> jax.Array:
    """Refresh periodic x halos for one local shard using ``lax.ppermute``.

    ``field`` may already contain halo slots. The returned array always has
    ``width`` halo cells on each side and preserves shape when the input already
    had those slots.
    """

    h = int(width)
    if h <= 0:
        return field
    if int(field.shape[-1]) < 3 * h:
        raise ValueError("local x extent is too small for requested halo width")
    interior = field[..., h:-h]
    if int(interior.shape[-1]) < h:
        raise ValueError("local interior x extent is smaller than halo width")
    right_edge = interior[..., -h:]
    left_edge = interior[..., :h]
    n = int(num_partitions)
    send_right = tuple((rank, (rank + 1) % n) for rank in range(n))
    send_left = tuple((rank, (rank - 1) % n) for rank in range(n))
    left_halo = jax.lax.ppermute(right_edge, axis_name=axis_name, perm=send_right)
    right_halo = jax.lax.ppermute(left_edge, axis_name=axis_name, perm=send_left)
    return jnp.concatenate((left_halo, interior, right_halo), axis=-1)


def exchange_periodic_halo_x_face(
    field: jax.Array,
    *,
    width: int,
    num_partitions: int,
    axis_name: str = "shard",
) -> jax.Array:
    """Refresh periodic x halos for C-grid x-face fields.

    The local interior includes ``L+1`` faces for ``L`` mass columns. The two
    inter-shard boundary faces are duplicated between neighbors, so halo sends
    exclude the duplicated boundary face.
    """

    h = int(width)
    if h <= 0:
        return field
    if int(field.shape[-1]) < 3 * h + 1:
        raise ValueError("local x-face extent is too small for requested halo width")
    interior = field[..., h:-h]
    if int(interior.shape[-1]) <= h:
        raise ValueError("local x-face interior must exceed halo width")
    send_to_right = interior[..., -(h + 1) : -1]
    send_to_left = interior[..., 1 : h + 1]
    n = int(num_partitions)
    right_perm = tuple((rank, (rank + 1) % n) for rank in range(n))
    left_perm = tuple((rank, (rank - 1) % n) for rank in range(n))
    left_halo = jax.lax.ppermute(send_to_right, axis_name=axis_name, perm=right_perm)
    right_halo = jax.lax.ppermute(send_to_left, axis_name=axis_name, perm=left_perm)
    return jnp.concatenate((left_halo, interior, right_halo), axis=-1)


def exchange_state_halos(state: State, halo) -> State:
    """Apply opt-in periodic x halo exchange to the requested State fields."""

    cfg = getattr(halo, "sharding", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return state
    if getattr(cfg, "axis", "x") != "x":
        raise NotImplementedError("only x-axis sharding is implemented")
    if getattr(halo, "edge_type", None) != "periodic":
        raise NotImplementedError("only periodic sharded halos are implemented")
    width = int(getattr(halo, "width"))
    num_partitions = int(cfg.resolved_partitions())
    axis_name = str(getattr(cfg, "axis_name", "shard"))
    fields = set(getattr(halo, "fields_to_exchange", ()))
    updates = {
        name: (
            exchange_periodic_halo_x_face(
                getattr(state, name),
                width=width,
                num_partitions=num_partitions,
                axis_name=axis_name,
            )
            if name == "u"
            else exchange_periodic_halo_x(
                getattr(state, name),
                width=width,
                num_partitions=num_partitions,
                axis_name=axis_name,
            )
        )
        for name in fields
    }
    return state.replace(**updates)


__all__ = [
    "COLLECTIVE_TOKENS",
    "ShardingConfig",
    "assert_flag_off_graph_unchanged",
    "hlo_collective_counts",
    "hlo_graph_stats",
    "hlo_operation_count",
    "hlo_sha256",
    "exchange_periodic_halo_x",
    "exchange_periodic_halo_x_face",
    "exchange_state_halos",
    "merge_state_x",
    "partition_state_x",
    "run_forecast_operational_optional_sharding",
    "run_forecast_operational_sharded",
    "select_forecast_runner",
    "x_partition_bounds",
]
