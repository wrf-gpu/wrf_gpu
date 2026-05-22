#!/usr/bin/env python3
"""Run the c2 idealized warm-bubble acoustic-scan validation probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import BaseState, State
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig, run_acoustic_scan_carry
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig


config.update("jax_enable_x64", True)

GRAVITY_M_S2 = 9.80665
R_DRY_AIR = 287.05
CP_DRY_AIR = 1004.0
KAPPA = R_DRY_AIR / CP_DRY_AIR
P0_PA = 100000.0
T0_K = 300.0
DEFAULT_OUTPUT = ROOT / ".agent" / "sprints" / "2026-05-22-m6x-c2-A2-pgf-acoustic-implementation" / "proofs" / "warm_bubble_600s.json"


class Diagnostics(NamedTuple):
    w_max_m_s: object
    w_min_m_s: object
    centroid_z_m: object
    p_min_pa: object
    p_max_pa: object
    theta_min_k: object
    theta_max_k: object
    mu_min_pa: object
    mu_max_pa: object
    finite_state: object


def _finite_float(value: Any) -> float | None:
    number = float(np.asarray(value))
    return number if math.isfinite(number) else None


def _pressure_at_height(z_m: np.ndarray | float) -> np.ndarray | float:
    return P0_PA * np.exp(-GRAVITY_M_S2 * np.asarray(z_m) / (R_DRY_AIR * T0_K))


def _theta_from_pressure(pressure_pa: np.ndarray) -> np.ndarray:
    return T0_K * (P0_PA / pressure_pa) ** KAPPA


def _grid(nx: int, ny: int, nz: int, dx_m: float, dy_m: float, dz_m: float) -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, float(dx_m), float(dy_m), int(nx), int(ny))
    z_top = float(nz) * float(dz_m)
    pressure_top = float(_pressure_at_height(z_top))
    terrain = TerrainProvenance(
        source_path="synthetic://m6-warm-bubble/flat",
        sha256="analytic-m6-warm-bubble-flat",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="cartesian",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), pressure_top, eta)
    bc = BCMetadata("ideal", ("u", "v", "w", "theta", "p", "pb", "ph", "mu"), 0, "linear", False)
    terrain_height = jnp.zeros((int(ny), int(nx)), dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta, terrain_height)


def _initial_state(
    grid: GridSpec,
    *,
    dz_m: float,
    bubble_center_x_m: float,
    bubble_center_z_m: float,
    bubble_radius_m: float,
    bubble_amplitude_k: float,
) -> tuple[State, BaseState, jax.Array, jax.Array]:
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    dx_m = float(grid.projection.dx_m)
    z_face_1d = np.arange(nz + 1, dtype=np.float64) * float(dz_m)
    z_mass_1d = 0.5 * (z_face_1d[:-1] + z_face_1d[1:])
    z_face = np.broadcast_to(z_face_1d[:, None, None], (nz + 1, ny, nx))
    z_mass = np.broadcast_to(z_mass_1d[:, None, None], (nz, ny, nx))

    pb_1d = _pressure_at_height(z_mass_1d)
    pb = np.broadcast_to(pb_1d[:, None, None], (nz, ny, nx)).copy()
    theta_base_1d = _theta_from_pressure(pb_1d)
    theta_base = np.broadcast_to(theta_base_1d[:, None, None], (nz, ny, nx)).copy()
    phb = GRAVITY_M_S2 * z_face
    mub = np.full((ny, nx), P0_PA - float(grid.vertical.top_pressure_pa), dtype=np.float64)

    x_mass = (np.arange(nx, dtype=np.float64) + 0.5) * dx_m
    domain_x = nx * dx_m
    periodic_dx = np.minimum(np.abs(x_mass - bubble_center_x_m), domain_x - np.abs(x_mass - bubble_center_x_m))
    r2 = periodic_dx[None, None, :] ** 2 + (z_mass - bubble_center_z_m) ** 2
    theta_perturbation = bubble_amplitude_k * np.exp(-r2 / (bubble_radius_m * bubble_radius_m))
    theta = theta_base + theta_perturbation

    state = State.zeros(grid).replace(
        theta=jnp.asarray(theta),
        p_total=jnp.asarray(pb),
        p_perturbation=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        ph_total=jnp.asarray(phb),
        ph_perturbation=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        mu_total=jnp.asarray(mub),
        mu_perturbation=jnp.zeros((ny, nx), dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), T0_K, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float32),
        mavail=jnp.ones((ny, nx), dtype=jnp.float32),
        roughness_m=jnp.full((ny, nx), 0.1, dtype=jnp.float64),
        rhosfc=jnp.full((ny, nx), P0_PA / (R_DRY_AIR * T0_K), dtype=jnp.float64),
    )
    base = BaseState(
        pb=jnp.asarray(pb),
        phb=jnp.asarray(phb),
        mub=jnp.asarray(mub),
        t0=jnp.asarray(theta_base),
        theta_base=jnp.asarray(theta_base),
    )
    return state, base, jnp.asarray(theta_base), jnp.asarray(z_mass)


def _diagnose(state: State, theta_base: jax.Array, z_mass: jax.Array) -> Diagnostics:
    theta_perturbation = state.theta - theta_base
    positive = jnp.maximum(theta_perturbation, 0.0)
    weight = jnp.sum(positive)
    centroid_z = jnp.where(weight > 0.0, jnp.sum(positive * z_mass) / weight, jnp.nan)
    finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]))
    return Diagnostics(
        w_max_m_s=jnp.max(state.w),
        w_min_m_s=jnp.min(state.w),
        centroid_z_m=centroid_z,
        p_min_pa=jnp.min(state.p_total),
        p_max_pa=jnp.max(state.p_total),
        theta_min_k=jnp.min(state.theta),
        theta_max_k=jnp.max(state.theta),
        mu_min_pa=jnp.min(state.mu_total),
        mu_max_pa=jnp.max(state.mu_total),
        finite_state=finite,
    )


@partial(jax.jit, static_argnames=("config", "dt_s", "steps"))
def _run(
    state: State,
    previous_pressure: jax.Array,
    metrics,
    base: BaseState,
    theta_base: jax.Array,
    z_mass: jax.Array,
    config: AcousticConfig,
    dt_s: float,
    steps: int,
):
    def body(carry, _):
        carry_state, carry_previous_pressure = carry
        next_carry = run_acoustic_scan_carry(
            carry_state,
            previous_pressure=carry_previous_pressure,
            metrics=metrics,
            config=config,
            dt=float(dt_s),
            base_state=base,
        )
        return (next_carry.state, next_carry.previous_pressure), _diagnose(next_carry.state, theta_base, z_mass)

    return jax.lax.scan(body, (state, previous_pressure), xs=None, length=int(steps))


def _series(initial: Diagnostics, scanned: Diagnostics) -> dict[str, list[float | bool | None]]:
    out: dict[str, list[float | bool | None]] = {}
    for name in initial._fields:
        first = np.asarray(jax.device_get(getattr(initial, name))).reshape(1)
        rest = np.asarray(jax.device_get(getattr(scanned, name))).reshape(-1)
        values = np.concatenate((first, rest))
        if name == "finite_state":
            out[name] = [bool(item) for item in values.tolist()]
        else:
            out[name] = [_finite_float(item) for item in values]
    return out


def _first_nonfinite_step(finite_state: list[float | bool | None]) -> int | None:
    for index, finite in enumerate(finite_state):
        if not bool(finite):
            return int(index)
    return None


def _sample_at(series: list[float | bool | None], dt_s: float, target_s: float) -> float | bool | None:
    index = int(round(target_s / dt_s))
    return series[index] if 0 <= index < len(series) else None


def _verdict(payload: dict[str, Any]) -> str:
    samples = payload["samples"]
    if payload["first_nonfinite_step"] is not None:
        return "FAIL_NONFINITE"
    w300 = samples["300s"]["w_max_m_s"]
    w600 = samples["600s"]["w_max_m_s"]
    z300 = samples["300s"]["centroid_z_m"]
    z600 = samples["600s"]["centroid_z_m"]
    if not all(isinstance(value, (int, float)) for value in (w300, w600, z300, z600)):
        return "FAIL_MISSING_SAMPLE"
    if 5.0 <= float(w300) <= 10.0 and 5.0 <= float(w600) <= 10.0 and float(z300) > 2500.0 and float(z600) > 3000.0:
        return "PASS_WARM_BUBBLE_600S"
    return "FAIL_TARGETS_NOT_MET"


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
    parser.add_argument("--smdiv", type=float, default=0.0)
    parser.add_argument("--rayleigh", type=float, default=0.0)
    parser.add_argument("--bubble-center-z-m", type=float, default=2000.0)
    parser.add_argument("--bubble-radius-m", type=float, default=2000.0)
    parser.add_argument("--bubble-amplitude-k", type=float, default=2.0)
    args = parser.parse_args(argv)

    steps = int(round(float(args.duration_s) / float(args.dt_s)))
    if abs(steps * float(args.dt_s) - float(args.duration_s)) > 1.0e-9:
        raise ValueError("duration must be an integer number of dt steps")

    grid = _grid(args.nx, args.ny, args.nz, args.dx_m, args.dy_m, args.dz_m)
    state, base, theta_base, z_mass = _initial_state(
        grid,
        dz_m=args.dz_m,
        bubble_center_x_m=0.5 * args.nx * args.dx_m,
        bubble_center_z_m=args.bubble_center_z_m,
        bubble_radius_m=args.bubble_radius_m,
        bubble_amplitude_k=args.bubble_amplitude_k,
    )
    acoustic_config = AcousticConfig(
        n_substeps=int(args.n_acoustic),
        dx_m=float(args.dx_m),
        dy_m=float(args.dy_m),
        smdiv=SmdivConfig(enabled=bool(args.smdiv), coefficient=float(args.smdiv)),
        rayleigh=RayleighConfig(enabled=bool(args.rayleigh), coefficient=float(args.rayleigh)),
    )
    initial = _diagnose(state, theta_base, z_mass)
    start = time.perf_counter()
    (final_state, _final_previous_pressure), scanned = _run(
        state,
        state.p_perturbation,
        grid.metrics,
        base,
        theta_base,
        z_mass,
        acoustic_config,
        float(args.dt_s),
        steps,
    )
    jax.block_until_ready(final_state.p_total)
    elapsed_s = time.perf_counter() - start
    series = _series(initial, scanned)
    nonfinite_step = _first_nonfinite_step(series["finite_state"])
    samples = {
        "300s": {
            "w_max_m_s": _sample_at(series["w_max_m_s"], args.dt_s, 300.0),
            "centroid_z_m": _sample_at(series["centroid_z_m"], args.dt_s, 300.0),
        },
        "600s": {
            "w_max_m_s": _sample_at(series["w_max_m_s"], args.dt_s, 600.0),
            "centroid_z_m": _sample_at(series["centroid_z_m"], args.dt_s, 600.0),
        },
    }
    payload: dict[str, Any] = {
        "artifact_type": "m6x_c2_a2_warm_bubble_600s",
        "description": "Idealized flat Skamarock-Klemp-style warm-bubble probe using c2 acoustic_wrf scan.",
        "setup": {
            "grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz, "dx_m": args.dx_m, "dy_m": args.dy_m, "dz_m": args.dz_m},
            "dt_s": args.dt_s,
            "duration_s": args.duration_s,
            "steps": steps,
            "n_acoustic": args.n_acoustic,
            "bubble": {
                "center_x_m": 0.5 * args.nx * args.dx_m,
                "center_z_m": args.bubble_center_z_m,
                "radius_m": args.bubble_radius_m,
                "amplitude_k": args.bubble_amplitude_k,
            },
            "smdiv": args.smdiv,
            "rayleigh": args.rayleigh,
        },
        "first_nonfinite_step": nonfinite_step,
        "surviving_seconds": float(args.duration_s) if nonfinite_step is None else max(0.0, (nonfinite_step - 1) * float(args.dt_s)),
        "samples": samples,
        "diagnostics_time_series": series,
        "runtime_s_including_compile": elapsed_s,
        "jax_devices": [str(device) for device in jax.devices()],
        "wrf_source_anchors": {
            "horizontal_pgf": "module_small_step_em.F:828-862,902-936",
            "diagnostic_pressure_al_alt": "module_big_step_utilities_em.F:1025-1030,1082-1087,910-943",
            "mu_continuity": "module_small_step_em.F:1094-1108",
        },
    }
    payload["verdict"] = _verdict(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": payload["verdict"],
                "first_nonfinite_step": payload["first_nonfinite_step"],
                "surviving_seconds": payload["surviving_seconds"],
                "samples": payload["samples"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["verdict"] == "PASS_WARM_BUBBLE_600S" else 2


if __name__ == "__main__":
    raise SystemExit(main())
