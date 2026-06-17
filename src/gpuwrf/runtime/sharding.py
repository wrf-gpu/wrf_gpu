"""Optional multi-device sharding controls for DGX-scale runs.

The default path is deliberately host-level and inert: with
``ShardingConfig.enabled=False`` the selected runner is exactly
``run_forecast_operational``. That preserves the current single-GPU compiled graph
for users who do not opt in to sharding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import os
from typing import Any, Callable, Literal
import hashlib
import re

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State
from gpuwrf.contracts.state import Tendencies


ShardAxis = Literal["x", "y"]
ShardBackend = Literal["pmap"]
_TRUE_ENV = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE_ENV = {"", "0", "false", "no", "off", "disable", "disabled"}

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
    forecast_halo_width: int | None = None
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
        if self.forecast_halo_width is not None and int(self.forecast_halo_width) < 1:
            raise ValueError("forecast_halo_width must be positive when supplied")
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

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        prefix: str = "GPUWRF_K2",
    ) -> "ShardingConfig":
        """Build the K2 experimental sharding config from environment variables.

        K2 is default-off.  Operators must explicitly set
        ``GPUWRF_K2_EXPERIMENTAL=1`` (or ``GPUWRF_K2_ENABLED=1``) before this
        returns an enabled config.  This helper is intentionally host-side; it is
        not referenced by the default forecast path and does not appear in the
        single-GPU compiled graph.
        """

        source = os.environ if env is None else env
        enabled = _env_bool(source, f"{prefix}_EXPERIMENTAL", default=False)
        if not enabled:
            enabled = _env_bool(source, f"{prefix}_ENABLED", default=False)
        if not enabled:
            return cls.disabled()

        num_partitions = _env_int(source, f"{prefix}_PARTITIONS", default=None)
        if num_partitions is None:
            num_partitions = _env_int(source, f"{prefix}_NUM_PARTITIONS", default=None)
        forecast_halo = _env_int(source, f"{prefix}_FORECAST_HALO_WIDTH", default=None)
        multi_node = _env_bool(source, f"{prefix}_MULTI_NODE", default=False)
        coordinator = _env_str(source, f"{prefix}_COORDINATOR_ADDRESS", default=None)
        return cls(
            enabled=True,
            axis=_env_str(source, f"{prefix}_AXIS", default="x"),  # type: ignore[arg-type]
            num_partitions=num_partitions,
            halo_width=int(_env_int(source, f"{prefix}_HALO_WIDTH", default=2)),
            forecast_halo_width=forecast_halo,
            backend=_env_str(source, f"{prefix}_BACKEND", default="pmap"),  # type: ignore[arg-type]
            axis_name=_env_str(source, f"{prefix}_AXIS_NAME", default="shard"),
            multi_node=multi_node,
            coordinator_address=coordinator,
            process_id=int(_env_int(source, f"{prefix}_PROCESS_ID", default=0)),
            process_count=int(_env_int(source, f"{prefix}_PROCESS_COUNT", default=1)),
        )

    def resolved_partitions(self, *, platform: str | None = None) -> int:
        """Return the local partition count this process should use."""

        if self.num_partitions is not None:
            return int(self.num_partitions)
        devices = jax.devices(platform) if platform is not None else jax.local_devices()
        return max(1, len(devices))

    def operational_halo_width(self) -> int:
        """Return the halo depth used by the host-level sharded forecast runner."""

        return int(self.halo_width if self.forecast_halo_width is None else self.forecast_halo_width)


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


def _env_str(env: Mapping[str, str], name: str, *, default: str | None) -> str | None:
    value = env.get(name)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _env_int(env: Mapping[str, str], name: str, *, default: int | None) -> int | None:
    value = _env_str(env, name, default=None)
    if value is None:
        return default
    return int(value)


def _env_bool(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE_ENV:
        return True
    if text in _FALSE_ENV:
        return False
    raise ValueError(f"{name} must be a boolean token, got {value!r}")


def _env_device_ids(env: Mapping[str, str], name: str) -> tuple[int, ...] | None:
    value = _env_str(env, name, default=None)
    if value is None:
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def initialize_k2_distributed_from_env(
    config: ShardingConfig | None = None,
    env: Mapping[str, str] | None = None,
    *,
    prefix: str = "GPUWRF_K2",
) -> bool:
    """Initialize JAX distributed for K2 multi-node runs when explicitly enabled.

    Returns ``True`` only when ``config.multi_node`` is true and initialization is
    attempted.  Call this before any JAX device enumeration.  The lab path on this
    one-GPU workstation leaves it unused; NCAR/UCAR can set the env variables and
    run the same verification script under their cluster launcher.
    """

    cfg = ShardingConfig.from_env(env, prefix=prefix) if config is None else config
    if not bool(cfg.enabled) or not bool(cfg.multi_node):
        return False
    source = os.environ if env is None else env
    jax.distributed.initialize(
        coordinator_address=cfg.coordinator_address,
        num_processes=int(cfg.process_count),
        process_id=int(cfg.process_id),
        local_device_ids=_env_device_ids(source, f"{prefix}_LOCAL_DEVICE_IDS"),
        initialization_timeout=int(_env_int(source, f"{prefix}_INIT_TIMEOUT_S", default=300)),
        heartbeat_timeout_seconds=int(_env_int(source, f"{prefix}_HEARTBEAT_TIMEOUT_S", default=100)),
        coordinator_bind_address=_env_str(source, f"{prefix}_COORDINATOR_BIND_ADDRESS", default=None),
    )
    return True


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

    cfg = ShardingConfig.disabled() if sharding is None else sharding
    if not bool(cfg.enabled):
        return select_forecast_runner(cfg)(state, namelist, hours)
    return run_forecast_operational_sharded(state, namelist, hours, sharding=cfg)


def run_forecast_operational_k2_experimental(
    state,
    namelist,
    hours: float,
    *,
    env: Mapping[str, str] | None = None,
    initialize_distributed: bool = False,
):
    """Environment-gated K2 experimental forecast entry.

    With no K2 environment set this is a direct call to the existing default
    forecast runner.  With ``GPUWRF_K2_EXPERIMENTAL=1`` it dispatches to the
    opt-in sharded runner.  This is the public cluster/lab switch; production
    code should continue to call ``run_forecast_operational`` unless K2 is being
    explicitly tested.
    """

    cfg = ShardingConfig.from_env(env)
    if initialize_distributed:
        initialize_k2_distributed_from_env(cfg, env)
    return run_forecast_operational_optional_sharding(state, namelist, hours, sharding=cfg)


def run_forecast_operational_sharded(
    state: State,
    namelist,
    hours: float,
    *,
    sharding: ShardingConfig | None = None,
) -> State:
    """Opt-in sharded runtime entry point.

    This is an opt-in ``pmap`` runner for the DGX-D2 proof. It partitions a real
    operational ``State`` into x slabs with periodic halos, traces the existing
    ``run_forecast_operational`` entry point under a fake/real device mesh, routes
    operational ``apply_halo`` calls through the D1 ``ppermute`` halo substrate,
    strips halos, and merges the owned output. The default public forecast path
    never calls this function.

    Current fidelity boundary: lateral boundary replay is rejected. The D1
    substrate implements periodic x halos, not the real WRF specified-boundary
    decomposition. Running each shard with ``run_boundary=True`` would incorrectly
    apply global lateral forcing at internal shard seams.
    """

    from gpuwrf.runtime.operational_mode import run_forecast_operational

    cfg = ShardingConfig(enabled=True) if sharding is None else sharding
    if not bool(cfg.enabled):
        return run_forecast_operational(state, namelist, hours)
    if cfg.axis != "x":
        raise NotImplementedError("operational sharded forecast currently supports x-axis decomposition only")
    if bool(getattr(namelist, "run_boundary", False)):
        raise NotImplementedError(
            "operational sharded forecast currently rejects run_boundary=True; "
            "specified/nested boundary decomposition is not implemented"
        )
    if getattr(namelist, "radiation_static", None) is not None:
        raise NotImplementedError("partitioning radiation_static for sharded operational forecast is not implemented")

    num_partitions = int(cfg.resolved_partitions())
    devices = tuple(jax.local_devices())
    if num_partitions > len(devices):
        raise ValueError(f"requested {num_partitions} partitions but only {len(devices)} local devices are visible")
    halo_width = int(cfg.operational_halo_width())
    if halo_width < int(getattr(namelist.grid, "halo_width", 1)):
        raise ValueError("forecast_halo_width must cover the grid halo width")
    if halo_width > 8:
        raise ValueError("forecast_halo_width must be <= 8 for operational ppermute halos")

    sharded_state = partition_state_x(
        state,
        namelist.grid,
        num_partitions=num_partitions,
        halo_width=halo_width,
        fill_halos=True,
    )
    sharded_tendencies = partition_tendencies_x(
        namelist.tendencies,
        namelist.grid,
        num_partitions=num_partitions,
        halo_width=halo_width,
        fill_halos=True,
    )
    sharded_metrics = partition_metrics_x(
        namelist.metrics,
        namelist.grid,
        num_partitions=num_partitions,
        halo_width=halo_width,
        fill_halos=True,
    )
    sharded_terrain = partition_terrain_height_x(
        namelist.grid,
        num_partitions=num_partitions,
        halo_width=halo_width,
        fill_halos=True,
    )
    def host_leaf(value: jax.Array) -> jax.Array:
        return jax.device_get(value)

    sharded_state = jax.tree_util.tree_map(host_leaf, sharded_state)
    sharded_tendencies = jax.tree_util.tree_map(host_leaf, sharded_tendencies)
    sharded_metrics = jax.tree_util.tree_map(host_leaf, sharded_metrics)
    sharded_terrain = host_leaf(sharded_terrain)

    def local_forecast(
        local_state: State,
        local_tendencies: Tendencies,
        local_metrics: DycoreMetrics,
        local_terrain: jax.Array,
    ) -> State:
        local_grid = local_grid_for_x_shard(
            namelist.grid,
            local_nx=int(local_state.theta.shape[-1]),
            terrain_height=local_terrain,
            metrics=local_metrics,
            halo_width=min(4, halo_width),
        )
        local_namelist = replace(
            namelist,
            grid=local_grid,
            tendencies=local_tendencies,
            metrics=local_metrics,
            run_boundary=False,
        )
        return run_forecast_operational(local_state, local_namelist, hours)

    from gpuwrf.contracts.halo import HaloSpec
    from gpuwrf.dynamics.advection import DYCORE_HALO_FIELDS
    import gpuwrf.dynamics.acoustic_wrf as acoustic_wrf
    import gpuwrf.dynamics.flux_advection as flux_advection
    import gpuwrf.dynamics.core.acoustic as acoustic_core
    import gpuwrf.dynamics.core.rk_addtend_dry as rk_addtend_dry
    import gpuwrf.dynamics.core.small_step_prep as small_step_prep
    import gpuwrf.runtime.operational_mode as operational_mode

    def sharded_halo_spec(grid: GridSpec) -> HaloSpec:
        del grid
        return HaloSpec(
            width=halo_width,
            fields_to_exchange=DYCORE_HALO_FIELDS,
            edge_type="periodic",
            sharding=cfg,
        )

    original_halo_spec = operational_mode.halo_spec
    original_acoustic_wrf_context = acoustic_wrf._SHARDED_HALO_CONTEXT
    original_flux_advection_context = flux_advection._SHARDED_HALO_CONTEXT
    original_acoustic_context = acoustic_core._SHARDED_HALO_CONTEXT
    original_rk_dry_context = rk_addtend_dry._SHARDED_HALO_CONTEXT
    original_small_step_prep_context = small_step_prep._SHARDED_HALO_CONTEXT
    original_carry_context = operational_mode._SHARDED_CARRY_HALO_CONTEXT
    operational_mode.halo_spec = sharded_halo_spec
    acoustic_wrf._SHARDED_HALO_CONTEXT = (cfg, halo_width)
    flux_advection._SHARDED_HALO_CONTEXT = (cfg, halo_width)
    acoustic_core._SHARDED_HALO_CONTEXT = (cfg, halo_width)
    rk_addtend_dry._SHARDED_HALO_CONTEXT = (cfg, halo_width)
    small_step_prep._SHARDED_HALO_CONTEXT = (cfg, halo_width)
    operational_mode._SHARDED_CARRY_HALO_CONTEXT = (cfg, halo_width)
    try:
        outputs = jax.pmap(local_forecast, axis_name=cfg.axis_name)(
            sharded_state,
            sharded_tendencies,
            sharded_metrics,
            sharded_terrain,
        )
        jax.block_until_ready(outputs.theta)
    finally:
        operational_mode._SHARDED_CARRY_HALO_CONTEXT = original_carry_context
        small_step_prep._SHARDED_HALO_CONTEXT = original_small_step_prep_context
        rk_addtend_dry._SHARDED_HALO_CONTEXT = original_rk_dry_context
        acoustic_core._SHARDED_HALO_CONTEXT = original_acoustic_context
        flux_advection._SHARDED_HALO_CONTEXT = original_flux_advection_context
        acoustic_wrf._SHARDED_HALO_CONTEXT = original_acoustic_wrf_context
        operational_mode.halo_spec = original_halo_spec
    merged = merge_state_x(outputs, namelist.grid, halo_width=halo_width)
    return jax.tree_util.tree_map(lambda value: jnp.asarray(jax.device_get(value)), merged)


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
    # The default-off graph is byte-identical to the reference (the disabled
    # selector returns the exact same function object), so assert the strongest
    # available invariant: identical HLO sha256.
    if str(reference["hlo_sha256"]) != str(candidate["hlo_sha256"]):
        raise AssertionError(
            f"flag-off HLO sha256 changed: {reference['hlo_sha256']} != {candidate['hlo_sha256']}"
        )


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
                    if int(end) == int(grid.nx):
                        right = array[..., :h]
                    else:
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
        name: (
            None
            if getattr(state, name) is None
            else _partition_leaf_x(
                name,
                getattr(state, name),
                grid=grid,
                bounds=bounds,
                halo_width=int(halo_width),
                fill_halos=bool(fill_halos),
            )
        )
        for name in State.__slots__
    }
    return State(**values)


def partition_tendencies_x(
    tendencies: Tendencies,
    grid,
    *,
    num_partitions: int,
    halo_width: int = 0,
    fill_halos: bool = True,
) -> Tendencies:
    """Split operational tendency buffers into leading-device x slabs."""

    bounds = x_partition_bounds(int(grid.nx), int(num_partitions))
    values = {
        name: _partition_leaf_x(
            name,
            getattr(tendencies, name),
            grid=grid,
            bounds=bounds,
            halo_width=int(halo_width),
            fill_halos=bool(fill_halos),
        )
        for name in Tendencies.__slots__
    }
    return Tendencies(**values)


def _partition_metric_leaf_x(
    name: str,
    array: jax.Array,
    *,
    grid,
    bounds: tuple[tuple[int, int], ...],
    halo_width: int,
    fill_halos: bool,
) -> jax.Array:
    """Partition a DycoreMetrics leaf along x, preserving WRF staggering."""

    del name
    if array.ndim == 0 or int(array.shape[-1]) not in (int(grid.nx), int(grid.nx) + 1):
        return jnp.stack([array for _ in bounds], axis=0)

    h = int(halo_width)
    chunks = []
    is_x_face = int(array.shape[-1]) == int(grid.nx) + 1
    for start, end in bounds:
        if is_x_face:
            interior = _take_x_face_periodic(array, start, end + 1, nx=int(grid.nx))
            if h:
                if fill_halos:
                    left = _take_x_face_periodic(array, start - h, start, nx=int(grid.nx))
                    if int(end) == int(grid.nx):
                        right = array[..., :h]
                    else:
                        right = _take_x_face_periodic(array, end + 1, end + 1 + h, nx=int(grid.nx))
                else:
                    left = jnp.zeros(array.shape[:-1] + (h,), dtype=array.dtype)
                    right = jnp.zeros(array.shape[:-1] + (h,), dtype=array.dtype)
                interior = jnp.concatenate((left, interior, right), axis=-1)
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


def partition_metrics_x(
    metrics: DycoreMetrics,
    grid,
    *,
    num_partitions: int,
    halo_width: int = 0,
    fill_halos: bool = True,
) -> DycoreMetrics:
    """Split DycoreMetrics into leading-device x slabs."""

    bounds = x_partition_bounds(int(grid.nx), int(num_partitions))
    values = {
        name: _partition_metric_leaf_x(
            name,
            getattr(metrics, name),
            grid=grid,
            bounds=bounds,
            halo_width=int(halo_width),
            fill_halos=bool(fill_halos),
        )
        for name in DycoreMetrics._array_names()
    }
    # The stacked p_top leaf has shape (num_partitions,), which is not a valid
    # single-domain DycoreMetrics scalar until a rank slice is taken. Rebuild via
    # the pytree inverse so the host-level runner can slice first and then validate
    # the per-rank GridSpec normally.
    return DycoreMetrics.tree_unflatten(
        f"{metrics.provenance}:x-sharded",
        tuple(values[name] for name in DycoreMetrics._array_names()),
    )


def partition_terrain_height_x(
    grid,
    *,
    num_partitions: int,
    halo_width: int = 0,
    fill_halos: bool = True,
) -> jax.Array:
    """Split GridSpec terrain height into leading-device x slabs."""

    bounds = x_partition_bounds(int(grid.nx), int(num_partitions))
    return _partition_metric_leaf_x(
        "terrain_height",
        grid.terrain_height,
        grid=grid,
        bounds=bounds,
        halo_width=int(halo_width),
        fill_halos=bool(fill_halos),
    )


def local_grid_for_x_shard(
    global_grid: GridSpec,
    *,
    local_nx: int,
    terrain_height: jax.Array,
    metrics: DycoreMetrics,
    halo_width: int | None = None,
) -> GridSpec:
    """Build a local-shard GridSpec whose static shape matches haloed shard leaves."""

    projection = Projection(
        global_grid.projection.kind,
        global_grid.projection.lat_0,
        global_grid.projection.lon_0,
        global_grid.projection.dx_m,
        global_grid.projection.dy_m,
        int(local_nx),
        global_grid.projection.ny,
    )
    terrain = TerrainProvenance(
        source_path=f"{global_grid.terrain.source_path}:x-shard",
        sha256=global_grid.terrain.sha256,
        shape=(projection.ny, projection.nx),
        units=global_grid.terrain.units,
        projection_transform=global_grid.terrain.projection_transform,
        max_elevation_m=global_grid.terrain.max_elevation_m,
        coastline_sanity_check_passed=global_grid.terrain.coastline_sanity_check_passed,
    )
    vertical = VerticalCoord(
        global_grid.vertical.kind,
        global_grid.vertical.nz,
        global_grid.vertical.top_pressure_pa,
        global_grid.eta_levels,
    )
    return GridSpec(
        projection,
        terrain,
        vertical,
        global_grid.bc,
        global_grid.eta_levels,
        terrain_height,
        metrics=metrics,
        halo_width=int(global_grid.halo_width if halo_width is None else halo_width),
        staggering=global_grid.staggering,
    )


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
        name: (
            None
            if getattr(sharded_state, name) is None
            else _merge_leaf_x(name, getattr(sharded_state, name), grid=grid, halo_width=int(halo_width))
        )
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
    rank = jax.lax.axis_index(axis_name)
    send_to_left_default = interior[..., 1 : h + 1]
    send_to_left_wrap = interior[..., :h]
    send_to_left = jnp.where(rank == 0, send_to_left_wrap, send_to_left_default)
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
    "local_grid_for_x_shard",
    "merge_state_x",
    "partition_metrics_x",
    "partition_state_x",
    "partition_tendencies_x",
    "partition_terrain_height_x",
    "run_forecast_operational_optional_sharding",
    "run_forecast_operational_sharded",
    "select_forecast_runner",
    "x_partition_bounds",
]
