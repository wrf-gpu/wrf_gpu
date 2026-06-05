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
from typing import Any

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational
from gpuwrf.runtime.sharding import (
    ShardingConfig,
    assert_flag_off_graph_unchanged,
    hlo_graph_stats,
    partition_state_x,
    select_forecast_runner,
)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=None, help="expected visible fake CPU device count")
    parser.add_argument("--check", choices=("flag-off", "halo", "all"), default="flag-off")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("proofs/multigpu_dgx/s1_flag_off_graph.json"),
    )
    args = parser.parse_args(argv)

    if args.devices is not None and len(jax.devices()) != int(args.devices):
        raise RuntimeError(f"expected {args.devices} devices, saw {len(jax.devices())}: {jax.devices()}")
    if args.check == "flag-off":
        payload = run_flag_off_graph_check()
    elif args.check == "halo":
        payload = run_halo_exchange_check()
    else:
        payload = {"checks": [run_flag_off_graph_check(), run_halo_exchange_check()]}
        payload["passed"] = all(bool(check["passed"]) for check in payload["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
