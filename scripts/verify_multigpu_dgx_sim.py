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

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational
from gpuwrf.runtime.sharding import (
    ShardingConfig,
    assert_flag_off_graph_unchanged,
    hlo_graph_stats,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=None, help="expected visible fake CPU device count")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("proofs/multigpu_dgx/s1_flag_off_graph.json"),
    )
    args = parser.parse_args(argv)

    if args.devices is not None and len(jax.devices()) != int(args.devices):
        raise RuntimeError(f"expected {args.devices} devices, saw {len(jax.devices())}: {jax.devices()}")
    payload = run_flag_off_graph_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
