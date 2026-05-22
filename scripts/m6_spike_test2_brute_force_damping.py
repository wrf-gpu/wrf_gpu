#!/usr/bin/env python3
"""Run the M6.x c1 brute-force smdiv plus Rayleigh-sponge damping diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

import m6_spike_test1_flat_vs_mountain as spike1
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

DEFAULT_OUTPUT = ROOT / "artifacts" / "m6" / "spike" / "test2_brute_damping_result.json"


def _finite_float(value: Any) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def _value_at(times: list[float], values: list[float | None], target: float) -> float | None:
    index = times.index(float(target))
    value = values[index]
    return None if value is None else float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dx-m", type=float, default=400.0)
    parser.add_argument("--dy-m", type=float, default=400.0)
    parser.add_argument("--dz-m", type=float, default=100.0)
    parser.add_argument("--dt-s", type=float, default=2.0)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--n-acoustic", type=int, default=8)
    parser.add_argument("--bubble-center-z-m", type=float, default=2000.0)
    parser.add_argument("--bubble-radius-m", type=float, default=2000.0)
    parser.add_argument("--bubble-amplitude-k", type=float, default=2.0)
    args = parser.parse_args(argv)

    steps = int(round(float(args.duration_s) / float(args.dt_s)))
    if abs(steps * float(args.dt_s) - float(args.duration_s)) > 1.0e-9:
        raise ValueError("duration must be an integer number of dt steps")

    flat_terrain = np.zeros((int(args.ny), int(args.nx)), dtype=np.float64)
    grid = spike1._grid(args.nx, args.ny, args.nz, args.dx_m, args.dy_m, args.dz_m, flat_terrain, label="flat-brute-damping")
    state, tendencies, theta_reference, z_mass = spike1._initial_state(
        grid,
        dz_m=args.dz_m,
        bubble_center_x_m=0.5 * args.nx * args.dx_m,
        bubble_center_z_m=args.bubble_center_z_m,
        bubble_radius_m=args.bubble_radius_m,
        bubble_amplitude_k=args.bubble_amplitude_k,
    )
    initial = spike1._diagnose(state, theta_reference, z_mass)
    start = time.perf_counter()
    final, scanned = spike1._run(
        state,
        tendencies,
        grid,
        theta_reference,
        z_mass,
        dt_s=args.dt_s,
        steps=steps,
        n_acoustic=args.n_acoustic,
    )
    block_until_ready(final)
    elapsed_s = time.perf_counter() - start

    diag = spike1._series(initial, scanned)
    nonfinite_step = spike1._first_nonfinite_step(diag["finite_state"])
    surviving_seconds = float(args.duration_s) if nonfinite_step is None else max(0.0, (nonfinite_step - 1) * float(args.dt_s))
    times_s = [float(i) * float(args.dt_s) for i in range(steps + 1)]
    survived_600s = nonfinite_step is None and surviving_seconds >= 600.0
    interpretation = "STABILIZED_PAST_600S_DAMPING_INFRASTRUCTURE_IMPLICATED" if survived_600s else "STILL_UNSTABLE_BEFORE_600S_FORMULATION_ERROR_IMPLICATED"

    payload = {
        "artifact_type": "m6x_numerical_stability_spike_test2_brute_force_damping",
        "description": "Flat warm-bubble c1 dycore run after temporary branch-only smdiv=0.1 plus top-10-level Rayleigh w sponge.",
        "setup": {
            "grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz, "dx_m": args.dx_m, "dy_m": args.dy_m, "dz_m": args.dz_m},
            "dt_s": args.dt_s,
            "duration_s": args.duration_s,
            "steps": steps,
            "n_acoustic": args.n_acoustic,
            "physics": "off",
            "dycore_path": "gpuwrf.dynamics.step.step, c1 acoustic + buoyancy via acoustic.py",
            "damping": {
                "smdiv": "SMDIV_DIVERGENCE_DAMPING = 0.1 in acoustic.py",
                "rayleigh_sponge": "temporary branch-only top 10 w-face levels, factor 1 - 0.5 * (k - nz + 10) / 10 for k > nz - 10",
            },
            "bubble": {
                "center_x_m": 0.5 * args.nx * args.dx_m,
                "center_z_m": args.bubble_center_z_m,
                "radius_m": args.bubble_radius_m,
                "amplitude_k": args.bubble_amplitude_k,
            },
        },
        "first_nonfinite_step": nonfinite_step,
        "surviving_seconds": surviving_seconds,
        "survived_600s": bool(survived_600s),
        "interpretation": interpretation,
        "runtime_s_including_compile": elapsed_s,
        "time_s": times_s,
        "w_max_t_m_s": diag["w_max_m_s"],
        "centroid_z_t_m": diag["centroid_z_m"],
        "measured": {
            "w_max_t_300s_m_s": _value_at(times_s, diag["w_max_m_s"], 300.0),
            "w_max_t_600s_m_s": _value_at(times_s, diag["w_max_m_s"], 600.0),
            "centroid_z_t_300s_m": _value_at(times_s, diag["centroid_z_m"], 300.0),
            "centroid_z_t_600s_m": _value_at(times_s, diag["centroid_z_m"], 600.0),
            "p_min_t_600s_pa": _value_at(times_s, diag["p_min_pa"], 600.0),
            "theta_max_t_600s_k": _value_at(times_s, diag["theta_max_k"], 600.0),
            "mu_min_t_600s_pa": _value_at(times_s, diag["mu_min_pa"], 600.0),
            "mu_max_t_600s_pa": _value_at(times_s, diag["mu_max_pa"], 600.0),
        },
        "diagnostics_time_series": diag,
        "runtime": {"jax_devices": [str(device) for device in jax.devices()]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "interpretation": interpretation,
                "first_nonfinite_step": nonfinite_step,
                "surviving_seconds": surviving_seconds,
                "survived_600s": survived_600s,
                "measured": payload["measured"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
