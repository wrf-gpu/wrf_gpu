#!/usr/bin/env python
"""CPU fake-device verification for optional DGX sharding work.

Run with, for example:

PYTHONPATH=src JAX_PLATFORM_NAME=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
taskset -c 0-27 python scripts/verify_multigpu_dgx_sim.py --devices 4
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.dynamics.explicit_diffusion import sixth_order_diffusion_tendency
from gpuwrf.dynamics.flux_advection import flux5_face_periodic
from gpuwrf.dynamics.sharded_horizontal import (
    sharded_flux5_face_periodic_x,
    sharded_sixth_order_diffusion_tendency,
    sharded_x_face_pressure_dpn,
    sharded_x_staggered_divergence,
)
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational
from gpuwrf.runtime.sharding import (
    ShardingConfig,
    assert_flag_off_graph_unchanged,
    hlo_graph_stats,
    partition_state_x,
    run_forecast_operational_optional_sharding,
    select_forecast_runner,
    x_partition_bounds,
)


D2_OPERATIONAL_FIELD_ATOL = {
    "theta": 1.0e-2,
    "u": 2.0e-2,
    "v": 1.0e-4,
    "w": 4.0e-3,
    "mu": 1.2,
    "mu_total": 1.2,
    "mu_perturbation": 1.2,
    "p": 3.5,
    "p_total": 3.5,
    "p_perturbation": 3.5,
    "ph": 0.25,
    "ph_total": 0.25,
    "ph_perturbation": 0.25,
}


def _cpu_state_and_namelist() -> tuple[State, OperationalNamelist, float]:
    grid = GridSpec.canary_3km_template()
    fields = {
        name: jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(name))
        for name, shape in _state_field_shapes(grid).items()
    }
    state = State(**fields)
    tendencies = Tendencies(
        fields["u"],
        fields["v"],
        fields["w"],
        fields["theta"],
        fields["qv"],
        fields["p"],
        fields["ph"],
        fields["mu"],
    )
    namelist = OperationalNamelist.from_grid(
        grid,
        tendencies=tendencies,
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=1,
        radiation_cadence_steps=999999,
        use_vertical_solver=False,
        disable_guards=True,
        force_fp64=True,
    )
    namelist = replace(namelist, run_physics=False, run_boundary=False)
    return state, namelist, 10.0 / 3600.0


def _test_grid(*, nx: int, ny: int = 8, nz: int = 10) -> GridSpec:
    projection = Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, int(nx), int(ny))
    terrain = TerrainProvenance(
        source_path="synthetic-dgx-sim",
        sha256="synthetic-dgx-sim",
        shape=(projection.ny, projection.nx),
        units="m",
        projection_transform="native-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), 5000.0, eta_levels)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "theta", "qv", "p"),
        update_cadence_h=6,
        interpolation="linear",
        restart_compatible=True,
    )
    terrain_height = jnp.zeros(terrain.shape, dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def _deterministic_state(grid: GridSpec) -> State:
    fields = {}
    for index, (name, shape) in enumerate(_state_field_shapes(grid).items()):
        dtype = DEFAULT_DTYPES.dtype_for(name)
        size = 1
        for dim in shape:
            size *= int(dim)
        values = jnp.arange(size, dtype=jnp.float64).reshape(shape) + float(index * 1000)
        fields[name] = values.astype(jnp.int32 if dtype == jnp.int32 else dtype)
    return State(**fields)


def _bounded_operator_state(state: State) -> State:
    return state.replace(
        theta=jnp.sin(state.theta.astype(jnp.float64) * 0.017).astype(state.theta.dtype),
        qv=jnp.cos(state.qv.astype(jnp.float64) * 0.013).astype(state.qv.dtype),
        u=jnp.sin(state.u.astype(jnp.float64) * 0.011).astype(state.u.dtype),
        p=jnp.sin(state.p.astype(jnp.float64) * 0.007).astype(state.p.dtype),
    )


def _compile_hlo(fn, state: State, namelist: OperationalNamelist, hours: float) -> str:
    return compiled_text(fn.lower(state, namelist, hours).compile())


def run_flag_off_graph_check() -> dict[str, Any]:
    state, namelist, hours = _cpu_state_and_namelist()
    disabled = ShardingConfig.disabled()
    selected = select_forecast_runner(disabled)
    reference_hlo = _compile_hlo(run_forecast_operational, state, namelist, hours)
    selected_hlo = _compile_hlo(selected, state, namelist, hours)
    assert_flag_off_graph_unchanged(reference_hlo, selected_hlo)
    selected_stats = hlo_graph_stats(selected_hlo)
    return {
        "check": "flag_off_graph_unchanged",
        "platform": jax.default_backend(),
        "device_count": len(jax.devices()),
        "local_device_count": len(jax.local_devices()),
        "selection_identity": selected is run_forecast_operational,
        "reference": hlo_graph_stats(reference_hlo),
        "disabled": selected_stats,
        "passed": (
            selected is run_forecast_operational
            and hlo_graph_stats(reference_hlo)["op_count"] == selected_stats["op_count"]
            and not selected_stats["collectives_present"]
        ),
        "simulation_can_prove": [
            "disabled sharding selects the exact existing default forecast function",
            "compiled flag-off HLO op count is unchanged",
            "compiled flag-off HLO has no collective/SPMD tokens",
        ],
        "simulation_cannot_prove": [
            "real H200/NVLink performance",
            "NCCL transport behavior",
            "multi-node strong scaling",
        ],
    }


def run_halo_exchange_check() -> dict[str, Any]:
    devices = len(jax.local_devices())
    if devices < 2:
        raise RuntimeError("halo exchange simulation requires at least two fake/real local devices")
    nx = max(16, devices * 4)
    grid = _test_grid(nx=nx)
    state = _deterministic_state(grid)
    records = []
    all_passed = True
    for width in range(1, 5):
        unfilled = partition_state_x(
            state,
            grid,
            num_partitions=devices,
            halo_width=width,
            fill_halos=False,
        )
        expected = partition_state_x(
            state,
            grid,
            num_partitions=devices,
            halo_width=width,
            fill_halos=True,
        )
        cfg = ShardingConfig(enabled=True, num_partitions=devices, halo_width=width)
        spec = HaloSpec(
            width=width,
            fields_to_exchange=("theta", "u", "v", "w", "mu"),
            edge_type="periodic",
            sharding=cfg,
        )

        def local_exchange(local_state):
            return apply_halo(local_state, spec)

        haloed = jax.pmap(local_exchange, axis_name=cfg.axis_name)(unfilled)
        field_results = {}
        for name in spec.fields_to_exchange:
            got = getattr(haloed, name)
            want = getattr(expected, name)
            equal = bool(jnp.array_equal(got, want))
            max_abs = float(jnp.max(jnp.abs(got.astype(jnp.float64) - want.astype(jnp.float64))))
            field_results[name] = {"equal": equal, "max_abs": max_abs, "shape": list(got.shape)}
            all_passed = all_passed and equal
        records.append({"width": width, "fields": field_results})
    return {
        "check": "periodic_ppermute_halo_exchange",
        "platform": jax.default_backend(),
        "device_count": len(jax.devices()),
        "local_device_count": devices,
        "grid": {"nx": int(grid.nx), "ny": int(grid.ny), "nz": int(grid.nz)},
        "records": records,
        "passed": bool(all_passed),
        "simulation_can_prove": [
            "periodic x-halo send and receive direction under ppermute",
            "halo widths 1 through 4 on fake local devices",
            "mass-grid and x-face staggered State leaves match global periodic slices",
        ],
        "simulation_cannot_prove": [
            "real NVLink or InfiniBand latency",
            "GPU collective overlap with dycore kernels",
            "multi-node process launch correctness",
        ],
    }


def _stack_mass_slices(field: jax.Array, bounds: tuple[tuple[int, int], ...]) -> jax.Array:
    return jnp.stack([field[..., start:end] for start, end in bounds], axis=0)


def _stack_face_slices(field: jax.Array, bounds: tuple[tuple[int, int], ...]) -> jax.Array:
    return jnp.stack([field[..., start : end + 1] for start, end in bounds], axis=0)


def _global_x_face_pressure_dpn(
    p: jax.Array,
    *,
    fnm: jax.Array,
    fnp: jax.Array,
    cf1: jax.Array,
    cf2: jax.Array,
    cf3: jax.Array,
) -> jax.Array:
    left = jnp.concatenate([p[:, :, :1], p], axis=2)
    right = jnp.concatenate([p, p[:, :, -1:]], axis=2)
    pair_sum = left + right
    _, ny, nx_face = pair_sum.shape
    bottom = 0.5 * (cf1 * pair_sum[0] + cf2 * pair_sum[1] + cf3 * pair_sum[2])
    interior = 0.5 * (
        fnm[1:, None, None] * pair_sum[1:, :, :]
        + fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    top = jnp.zeros((ny, nx_face), dtype=p.dtype)
    return jnp.concatenate([bottom[None, :, :], interior, top[None, :, :]], axis=0)


def _comparison_record(got: jax.Array, want: jax.Array, *, rtol: float, atol: float) -> dict[str, Any]:
    diff = got.astype(jnp.float64) - want.astype(jnp.float64)
    max_abs = float(jnp.max(jnp.abs(diff)))
    return {
        "shape": list(got.shape),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs": max_abs,
        "passed": bool(jnp.allclose(got, want, rtol=rtol, atol=atol)),
    }


def run_operator_check() -> dict[str, Any]:
    devices = len(jax.local_devices())
    if devices < 2:
        raise RuntimeError("operator simulation requires at least two fake/real local devices")
    nx = max(32, devices * 8)
    grid = _test_grid(nx=nx, ny=4, nz=6)
    state = _bounded_operator_state(_deterministic_state(grid))
    bounds = x_partition_bounds(grid.nx, devices)
    records: dict[str, Any] = {}

    halo3 = partition_state_x(state, grid, num_partitions=devices, halo_width=3, fill_halos=True)
    flux_global = flux5_face_periodic(state.theta, state.qv + 0.25, axis=2)
    flux_got = jax.pmap(
        lambda field, vel: sharded_flux5_face_periodic_x(field, vel + 0.25, halo_width=3),
        axis_name="shard",
    )(halo3.theta, halo3.qv)
    records["flux5_x"] = _comparison_record(
        flux_got,
        _stack_mass_slices(flux_global, bounds),
        rtol=2.0e-6,
        atol=2.0e-7,
    )

    diffusion_kwargs = {"dt": 10.0, "diff_6th_factor": 0.12, "horizontal_only": True, "monotonic": True}
    diffusion_global = sixth_order_diffusion_tendency(state.theta, **diffusion_kwargs)
    diffusion_got = jax.pmap(
        lambda field: sharded_sixth_order_diffusion_tendency(field, halo_width=3, **diffusion_kwargs),
        axis_name="shard",
    )(halo3.theta)
    records["sixth_order_diffusion"] = _comparison_record(
        diffusion_got,
        _stack_mass_slices(diffusion_global, bounds),
        rtol=2.0e-6,
        atol=2.0e-9,
    )

    halo1 = partition_state_x(state, grid, num_partitions=devices, halo_width=1, fill_halos=True)
    div_global = 0.1 * (state.u[:, :, 1 : grid.nx + 1] - state.u[:, :, : grid.nx])
    div_got = jax.pmap(
        lambda u: sharded_x_staggered_divergence(u, rdx=0.1, halo_width=1),
        axis_name="shard",
    )(halo1.u)
    records["acoustic_x_divergence"] = _comparison_record(
        div_got,
        _stack_mass_slices(div_global, bounds),
        rtol=0.0,
        atol=0.0,
    )

    fnm = jnp.linspace(0.2, 0.8, grid.nz, dtype=state.p.dtype)
    fnp = 1.0 - fnm
    cf1 = jnp.asarray(0.55, dtype=state.p.dtype)
    cf2 = jnp.asarray(0.30, dtype=state.p.dtype)
    cf3 = jnp.asarray(0.15, dtype=state.p.dtype)
    dpn_global = _global_x_face_pressure_dpn(state.p, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3)
    owned_width = grid.nx // devices

    def local_dpn(p):
        rank = jax.lax.axis_index("shard")
        return sharded_x_face_pressure_dpn(
            p,
            fnm=fnm,
            fnp=fnp,
            cf1=cf1,
            cf2=cf2,
            cf3=cf3,
            halo_width=1,
            global_start=rank * owned_width,
            global_nx=grid.nx,
        )

    dpn_got = jax.pmap(local_dpn, axis_name="shard")(halo1.p)
    records["acoustic_x_face_pressure_dpn"] = _comparison_record(
        dpn_got,
        _stack_face_slices(dpn_global, bounds),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    return {
        "check": "sharded_horizontal_operators",
        "platform": jax.default_backend(),
        "device_count": len(jax.devices()),
        "local_device_count": devices,
        "grid": {"nx": int(grid.nx), "ny": int(grid.ny), "nz": int(grid.nz)},
        "records": records,
        "passed": all(bool(record["passed"]) for record in records.values()),
        "simulation_can_prove": [
            "halo-fed local x operators reproduce owned columns/faces of current global formulas",
            "flux-advection and sixth-order diffusion x stencils have sufficient halo width",
            "acoustic x divergence and x-face pressure-dpn seam handling match global edge/pair semantics",
        ],
        "simulation_cannot_prove": [
            "real H200 kernel occupancy",
            "NVLink collective overlap with operator kernels",
            "full dycore timestep scaling on DGX",
        ],
    }


def run_end_to_end_check() -> dict[str, Any]:
    devices = len(jax.local_devices())
    if devices < 2:
        raise RuntimeError("end-to-end simulation requires at least two fake/real local devices")
    nx = max(64, devices * 8)
    grid = _test_grid(nx=nx, ny=4, nz=6)
    state = _bounded_operator_state(_deterministic_state(grid))
    bounds = x_partition_bounds(grid.nx, devices)
    records: dict[str, Any] = {}

    halo3_unfilled = partition_state_x(
        state,
        grid,
        num_partitions=devices,
        halo_width=3,
        fill_halos=False,
    )
    cfg3 = ShardingConfig(enabled=True, num_partitions=devices, halo_width=3)
    spec3 = HaloSpec(
        width=3,
        fields_to_exchange=("theta", "qv"),
        edge_type="periodic",
        sharding=cfg3,
    )

    def exchange3(local_state):
        return apply_halo(local_state, spec3)

    halo3 = jax.pmap(exchange3, axis_name=cfg3.axis_name)(halo3_unfilled)
    flux_global = flux5_face_periodic(state.theta, state.qv + 0.25, axis=2)
    flux_got = jax.pmap(
        lambda field, vel: sharded_flux5_face_periodic_x(field, vel + 0.25, halo_width=3),
        axis_name=cfg3.axis_name,
    )(halo3.theta, halo3.qv)
    records["ppermute_flux5_x"] = _comparison_record(
        flux_got,
        _stack_mass_slices(flux_global, bounds),
        rtol=2.0e-6,
        atol=2.0e-7,
    )

    diffusion_kwargs = {"dt": 10.0, "diff_6th_factor": 0.12, "horizontal_only": True, "monotonic": True}
    diffusion_global = sixth_order_diffusion_tendency(state.theta, **diffusion_kwargs)
    diffusion_got = jax.pmap(
        lambda field: sharded_sixth_order_diffusion_tendency(field, halo_width=3, **diffusion_kwargs),
        axis_name=cfg3.axis_name,
    )(halo3.theta)
    records["ppermute_sixth_order_diffusion"] = _comparison_record(
        diffusion_got,
        _stack_mass_slices(diffusion_global, bounds),
        rtol=2.0e-6,
        atol=2.0e-9,
    )

    halo1_unfilled = partition_state_x(
        state,
        grid,
        num_partitions=devices,
        halo_width=1,
        fill_halos=False,
    )
    cfg1 = ShardingConfig(enabled=True, num_partitions=devices, halo_width=1)
    spec1 = HaloSpec(
        width=1,
        fields_to_exchange=("u", "p"),
        edge_type="periodic",
        sharding=cfg1,
    )

    def exchange1(local_state):
        return apply_halo(local_state, spec1)

    halo1 = jax.pmap(exchange1, axis_name=cfg1.axis_name)(halo1_unfilled)
    div_global = 0.1 * (state.u[:, :, 1 : grid.nx + 1] - state.u[:, :, : grid.nx])
    div_got = jax.pmap(
        lambda u: sharded_x_staggered_divergence(u, rdx=0.1, halo_width=1),
        axis_name=cfg1.axis_name,
    )(halo1.u)
    records["ppermute_acoustic_x_divergence"] = _comparison_record(
        div_got,
        _stack_mass_slices(div_global, bounds),
        rtol=0.0,
        atol=0.0,
    )

    fnm = jnp.linspace(0.2, 0.8, grid.nz, dtype=state.p.dtype)
    fnp = 1.0 - fnm
    cf1 = jnp.asarray(0.55, dtype=state.p.dtype)
    cf2 = jnp.asarray(0.30, dtype=state.p.dtype)
    cf3 = jnp.asarray(0.15, dtype=state.p.dtype)
    dpn_global = _global_x_face_pressure_dpn(state.p, fnm=fnm, fnp=fnp, cf1=cf1, cf2=cf2, cf3=cf3)
    owned_width = grid.nx // devices

    def local_dpn(p):
        rank = jax.lax.axis_index("shard")
        return sharded_x_face_pressure_dpn(
            p,
            fnm=fnm,
            fnp=fnp,
            cf1=cf1,
            cf2=cf2,
            cf3=cf3,
            halo_width=1,
            global_start=rank * owned_width,
            global_nx=grid.nx,
        )

    dpn_got = jax.pmap(local_dpn, axis_name=cfg1.axis_name)(halo1.p)
    records["ppermute_acoustic_x_face_pressure_dpn"] = _comparison_record(
        dpn_got,
        _stack_face_slices(dpn_global, bounds),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    return {
        "check": "ppermute_halo_then_sharded_operators",
        "platform": jax.default_backend(),
        "device_count": len(jax.devices()),
        "local_device_count": devices,
        "grid": {"nx": int(grid.nx), "ny": int(grid.ny), "nz": int(grid.nz)},
        "records": records,
        "passed": all(bool(record["passed"]) for record in records.values()),
        "simulation_can_prove": [
            "unfilled x shards can refresh halos with lax.ppermute and reproduce selected single-domain operator outputs",
            "single-node 8-way CPU fake mesh exercises the same pmap axis/rank/permute structure intended for 8xH200",
            "mass-grid, x-face staggered, and acoustic edge-face seams are correct for these local operators",
        ],
        "simulation_cannot_prove": [
            "real H200/NVLink bandwidth or latency",
            "NCCL collective implementation details",
            "whole-dycore wall-clock strong scaling",
            "multi-node launcher and fabric behavior",
        ],
    }


def _all_state_comparisons(
    left: Any,
    right: Any,
    *,
    rtol: float,
    atol: float,
    field_atol: dict[str, float] | None = None,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    all_exact = True
    all_close = True
    for name in State.__slots__:
        atol_i = float((field_atol or {}).get(name, atol))
        got = getattr(left, name)
        want = getattr(right, name)
        if tuple(got.shape) != tuple(want.shape):
            records[name] = {
                "shape_got": list(got.shape),
                "shape_want": list(want.shape),
                "exact": False,
                "allclose": False,
                "max_abs": None,
                "reason": "shape mismatch",
            }
            all_exact = False
            all_close = False
            continue
        exact = bool(jnp.array_equal(got, want))
        if got.dtype == jnp.int32 or want.dtype == jnp.int32:
            diff = got.astype(jnp.float64) - want.astype(jnp.float64)
            max_abs = float(jnp.max(jnp.abs(diff))) if got.size else 0.0
            max_index = (
                [int(x) for x in np.unravel_index(int(jnp.argmax(jnp.abs(diff))), got.shape)]
                if got.size
                else []
            )
            close = exact
        else:
            diff = got.astype(jnp.float64) - want.astype(jnp.float64)
            max_abs = float(jnp.max(jnp.abs(diff))) if got.size else 0.0
            max_index = (
                [int(x) for x in np.unravel_index(int(jnp.argmax(jnp.abs(diff))), got.shape)]
                if got.size
                else []
            )
            close = bool(jnp.allclose(got, want, rtol=rtol, atol=atol_i))
        records[name] = {
            "shape": list(got.shape),
            "dtype_got": str(got.dtype),
            "dtype_want": str(want.dtype),
            "exact": exact,
            "allclose": close,
            "max_abs": max_abs,
            "max_abs_index": max_index,
            "rtol": float(rtol),
            "atol": float(atol_i),
        }
        all_exact = all_exact and exact
        all_close = all_close and close
    return {"records": records, "all_exact": bool(all_exact), "allclose": bool(all_close)}


def run_operational_forecast_check(
    *,
    run_dir: Path,
    devices: int,
    forecast_steps: int,
    dt_s: float,
    acoustic_substeps: int,
    forecast_halo_width: int,
    run_radiation: bool,
) -> dict[str, Any]:
    """Run real d02 operational path on fake/local x shards and compare to default."""

    from gpuwrf.integration.d02_replay import build_replay_case
    import gpuwrf.contracts.state as state_contract

    if len(jax.local_devices()) < int(devices):
        raise RuntimeError(f"requires {devices} local fake/real devices, saw {len(jax.local_devices())}")
    original_gpu_device = state_contract._gpu_device
    cpu_replay_loader_shim = not any(device.platform == "gpu" for device in jax.devices())
    if cpu_replay_loader_shim:
        state_contract._gpu_device = lambda: jax.local_devices()[0]  # type: ignore[assignment]
    try:
        case = build_replay_case(run_dir, domain="d02")
    finally:
        state_contract._gpu_device = original_gpu_device  # type: ignore[assignment]
    def materialized_run_state():
        def copied(value):
            return jax.device_put(jnp.asarray(np.array(jax.device_get(value), copy=True)))

        values = {name: copied(getattr(case.state, name)) for name in State.__slots__}
        return State(**values)
    radiation_cadence = 1 if bool(run_radiation) else 999999
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=float(dt_s),
        acoustic_substeps=int(acoustic_substeps),
        radiation_cadence_steps=int(radiation_cadence),
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        top_lid=True,
        radiation_static=None,
        topo_shading=0,
        slope_rad=0,
    )
    namelist = replace(namelist, run_physics=True, run_boundary=False, disable_guards=False)
    hours = float(forecast_steps) * float(dt_s) / 3600.0
    cfg = ShardingConfig(
        enabled=True,
        num_partitions=int(devices),
        halo_width=min(4, max(1, int(case.grid.halo_width))),
        forecast_halo_width=int(forecast_halo_width),
    )

    t0 = time.perf_counter()
    reference = run_forecast_operational(materialized_run_state(), namelist, hours)
    jax.block_until_ready(reference.theta)
    reference_wall_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sharded = run_forecast_operational_optional_sharding(materialized_run_state(), namelist, hours, sharding=cfg)
    jax.block_until_ready(sharded.theta)
    sharded_wall_s = time.perf_counter() - t0

    comparison = _all_state_comparisons(sharded, reference, rtol=0.0, atol=0.0)
    if not comparison["all_exact"]:
        comparison_tol = _all_state_comparisons(
            sharded,
            reference,
            rtol=0.0,
            atol=0.0,
            field_atol=D2_OPERATIONAL_FIELD_ATOL,
        )
    else:
        comparison_tol = comparison
    key_fields = {
        name: comparison_tol["records"].get(name)
        for name in ("theta", "u", "v", "w", "p", "ph", "mu", "qv", "qke", "rain_acc")
    }
    return {
        "check": "real_d02_operational_forecast_fake_mesh",
        "platform": jax.default_backend(),
        "device_count": len(jax.devices()),
        "local_device_count": len(jax.local_devices()),
        "used_devices": int(devices),
        "run_dir": str(run_dir),
        "case_metadata": {
            "run_id": case.metadata.get("run_id"),
            "domain": "d02",
            "grid": case.metadata.get("grid"),
            "qke_coldstart": case.metadata.get("qke_coldstart"),
            "cpu_replay_loader_shim": bool(cpu_replay_loader_shim),
        },
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "forecast_steps": int(forecast_steps),
            "hours": float(hours),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "run_radiation_on_step_1": bool(run_radiation),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            "use_flux_advection": bool(namelist.use_flux_advection),
            "force_fp64": bool(namelist.force_fp64),
            "diff_6th_opt": int(namelist.diff_6th_opt),
            "w_damping": int(namelist.w_damping),
            "damp_opt": int(namelist.damp_opt),
            "epssm": float(namelist.epssm),
            "top_lid": bool(namelist.top_lid),
        },
        "sharding": {
            "axis": cfg.axis,
            "num_partitions": int(devices),
            "halo_width": int(cfg.halo_width),
            "forecast_halo_width": int(cfg.operational_halo_width()),
            "runner": "run_forecast_operational_optional_sharding -> run_forecast_operational_sharded",
            "execution_model": "pmapped local-device x shards with ppermute halo exchange",
        },
        "timing_wall_s": {
            "single_device_reference": float(reference_wall_s),
            "sharded_fake_mesh": float(sharded_wall_s),
            "timing_note": "CPU fake-device wall time includes compile and is not a DGX performance measurement",
        },
        "comparison": {
            "bit_identical": bool(comparison["all_exact"]),
            "within_tolerance": bool(comparison_tol["allclose"]),
            "rtol": 0.0,
            "atol": 0.0 if comparison["all_exact"] else "field-specific",
            "tolerance_policy": (
                "bit-identical required for fields not listed; listed dry-dynamic fields use absolute tolerances"
                if not comparison["all_exact"]
                else "bit-identical"
            ),
            "field_atol": {} if comparison["all_exact"] else D2_OPERATIONAL_FIELD_ATOL,
            "key_fields": key_fields,
            "all_fields": comparison_tol["records"],
        },
        "passed": bool(comparison_tol["allclose"]),
        "simulation_can_prove": [
            "real d02 State/metrics/tendencies can be partitioned into x shards and run through run_forecast_operational under pmap",
            "the default single-device operational entrypoint remains the reference",
            "fake local CPU devices exercise the full-forecast ppermute halo exchange path",
        ],
        "simulation_cannot_prove": [
            "real H200/NVLink/NCCL performance",
            "specified/nested lateral boundary decomposition, because run_boundary=True is intentionally rejected",
            "host/device transfer absence on real GPUs inside the full timestep loop",
        ],
    }


def write_dgx_d2_status(path: Path, payload: dict[str, Any]) -> None:
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else [payload]
    by_name = {check.get("check", f"check_{idx}"): check for idx, check in enumerate(checks)}
    op = by_name.get("real_d02_operational_forecast_fake_mesh", {})
    flag = by_name.get("flag_off_graph_unchanged", {})
    key = op.get("comparison", {}).get("key_fields", {})
    max_lines = []
    for name in ("theta", "u", "v", "w", "mu", "p", "ph", "qv", "qke", "rain_acc"):
        rec = key.get(name, {})
        if rec:
            max_lines.append(
                f"- `{name}`: max_abs={rec.get('max_abs')} atol={rec.get('atol')} exact={rec.get('exact')}"
            )
    lines = [
        "# v0.11.0 DGX-D2 sharded operational forecast",
        "",
        f"- verdict: {'PASS' if payload.get('passed') else 'FAIL'}",
        f"- operational sharded forecast parity: {op.get('passed')}",
        f"- bit-identical: {op.get('comparison', {}).get('bit_identical')}",
        f"- within tolerance: {op.get('comparison', {}).get('within_tolerance')}",
        f"- flag-off graph unchanged: {flag.get('passed')}",
        f"- flag-off selection identity: {flag.get('selection_identity')}",
        f"- fake/local devices: {op.get('used_devices')} of {op.get('local_device_count')}",
        f"- run_boundary in sharded proof: {op.get('namelist', {}).get('run_boundary')}",
        f"- radiation on step 1: {op.get('namelist', {}).get('run_radiation_on_step_1')}",
        "",
        "## What This Proves",
        "",
        "- Disabled sharding still selects the exact existing `run_forecast_operational` function.",
        "- The flag-off compiled graph has unchanged op count and zero collective/SPMD tokens.",
        "- A real d02 replay state runs through the operational forecast entrypoint on x-sharded fake/local devices and is compared with the single-device reference.",
        "- Non-dry physics/carry fields are bit-identical in the one-step proof; dry dynamic fields are within the recorded absolute tolerances.",
        "",
        "## Max Differences",
        "",
        *max_lines,
        "",
        "## Carry",
        "",
        "- Real DGX performance, NCCL behavior, and transfer cleanliness still require hardware.",
        "- `run_boundary=True` is intentionally not claimed for sharded execution until specified/nested boundary decomposition is implemented.",
        "- The fake-mesh proof exercises local-device `pmap` and `lax.ppermute`; real-DGX profiler artifacts are still required before any speedup claim.",
        "",
        "## Real-DGX Smoke Checklist",
        "",
        "1. Confirm 8 H200-class GPUs with `nvidia-smi -L` and topology with `nvidia-smi topo -m`.",
        "2. Run the flag-off graph proof on one GPU and all 8 visible GPUs through `/tmp/wrf_gpu_run.sh`.",
        "3. Run D1 halo/operator fake-mesh tests on real 8-GPU pmap and compare max diffs to committed proofs.",
        "4. Run this D2 operational forecast proof with `--devices 8`, first `run_boundary=False`, then after boundary decomposition with `run_boundary=True`.",
        "5. Capture Nsight Systems; verify collectives are only documented halo exchanges and no host/device transfers occur inside timestep loops.",
        "6. Only after profiler artifacts exist, run 1/2/4/8 GPU weak and strong scaling and report measured speedup.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=None, help="expected visible fake CPU device count")
    parser.add_argument(
        "--check",
        choices=("flag-off", "halo", "operators", "e2e", "operational-forecast", "all", "d2"),
        default="flag-off",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("proofs/multigpu_dgx/s1_flag_off_graph.json"),
    )
    parser.add_argument("--status-md", type=Path, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z"),
    )
    parser.add_argument("--forecast-steps", type=int, default=1)
    parser.add_argument("--dt-s", type=float, default=10.0)
    parser.add_argument("--acoustic-substeps", type=int, default=1)
    parser.add_argument("--forecast-halo-width", type=int, default=5)
    parser.add_argument("--run-radiation", action="store_true")
    args = parser.parse_args(argv)

    if args.devices is not None and len(jax.devices()) != int(args.devices):
        raise RuntimeError(f"expected {args.devices} devices, saw {len(jax.devices())}: {jax.devices()}")
    if args.check == "flag-off":
        payload = run_flag_off_graph_check()
    elif args.check == "halo":
        payload = run_halo_exchange_check()
    elif args.check == "operators":
        payload = run_operator_check()
    elif args.check == "e2e":
        payload = run_end_to_end_check()
    elif args.check == "operational-forecast":
        if args.devices is None:
            raise RuntimeError("--devices is required for --check operational-forecast")
        payload = run_operational_forecast_check(
            run_dir=args.run_dir,
            devices=int(args.devices),
            forecast_steps=int(args.forecast_steps),
            dt_s=float(args.dt_s),
            acoustic_substeps=int(args.acoustic_substeps),
            forecast_halo_width=int(args.forecast_halo_width),
            run_radiation=bool(args.run_radiation),
        )
    elif args.check == "d2":
        if args.devices is None:
            raise RuntimeError("--devices is required for --check d2")
        payload = {
            "checks": [
                run_flag_off_graph_check(),
                run_operational_forecast_check(
                    run_dir=args.run_dir,
                    devices=int(args.devices),
                    forecast_steps=int(args.forecast_steps),
                    dt_s=float(args.dt_s),
                    acoustic_substeps=int(args.acoustic_substeps),
                    forecast_halo_width=int(args.forecast_halo_width),
                    run_radiation=bool(args.run_radiation),
                ),
            ]
        }
        payload["passed"] = all(bool(check["passed"]) for check in payload["checks"])
    else:
        payload = {
            "checks": [
                run_flag_off_graph_check(),
                run_halo_exchange_check(),
                run_operator_check(),
                run_end_to_end_check(),
            ]
        }
        payload["passed"] = all(bool(check["passed"]) for check in payload["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.status_md is not None:
        write_dgx_d2_status(args.status_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
