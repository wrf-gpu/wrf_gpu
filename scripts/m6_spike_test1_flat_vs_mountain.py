#!/usr/bin/env python3
"""Run the M6.x c1 flat-vs-Schar-mountain numerical-stability diagnostic."""

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
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.dynamics.step import step as dycore_step
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

GRAVITY_M_S2 = 9.80665
R_DRY_AIR = 287.05
CP_DRY_AIR = 1004.0
KAPPA = R_DRY_AIR / CP_DRY_AIR
P0_PA = 100000.0
T0_K = 300.0
DEFAULT_OUTPUT = ROOT / "artifacts" / "m6" / "spike" / "test1_flat_vs_mountain_result.json"


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


def _schar_terrain(nx: int, ny: int, dx_m: float, *, peak_m: float, half_width_m: float) -> np.ndarray:
    """Compact Schar-style 2-D ridge, peak-centered and constant in y."""

    x = (np.arange(nx, dtype=np.float64) + 0.5) * float(dx_m)
    x0 = 0.5 * float(nx) * float(dx_m)
    radius = np.abs(x - x0)
    ridge = np.where(radius <= half_width_m, peak_m * np.cos(0.5 * np.pi * radius / half_width_m) ** 2, 0.0)
    return np.broadcast_to(ridge[None, :], (int(ny), int(nx))).copy()


def _grid(
    nx: int,
    ny: int,
    nz: int,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    terrain_height: np.ndarray,
    *,
    label: str,
) -> GridSpec:
    projection = Projection("lambert", 0.0, 0.0, float(dx_m), float(dy_m), int(nx), int(ny))
    z_top = float(nz) * float(dz_m)
    pressure_top = float(_pressure_at_height(z_top))
    terrain = TerrainProvenance(
        source_path=f"synthetic://m6x-spike/{label}",
        sha256=f"analytic-m6x-spike-{label}",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="cartesian",
        max_elevation_m=float(np.max(terrain_height)),
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), pressure_top, eta)
    bc = BCMetadata("ideal", ("u", "v", "w", "theta", "p", "pb", "ph", "mu"), 0, "linear", False)
    return GridSpec(projection, terrain, vertical, bc, eta, jnp.asarray(terrain_height, dtype=jnp.float64))


def _terrain_following_interfaces(terrain_height: np.ndarray, nz: int, dz_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return interface and mass heights for a flat-lid terrain-following column."""

    z_top = float(nz) * float(dz_m)
    sigma_face = np.linspace(0.0, 1.0, int(nz) + 1, dtype=np.float64)[:, None, None]
    terrain = terrain_height[None, :, :]
    z_face = terrain + sigma_face * (z_top - terrain)
    z_mass = 0.5 * (z_face[:-1, :, :] + z_face[1:, :, :])
    return z_face, z_mass


def _initial_state(
    grid: GridSpec,
    *,
    dz_m: float,
    bubble_center_x_m: float,
    bubble_center_z_m: float,
    bubble_radius_m: float,
    bubble_amplitude_k: float,
) -> tuple[State, Tendencies, jax.Array, jax.Array]:
    """Create a terrain-following warm-bubble state with a flat-column PB reference."""

    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    dx_m = float(grid.projection.dx_m)
    terrain_height = np.asarray(grid.terrain_height, dtype=np.float64)
    z_face, z_mass = _terrain_following_interfaces(terrain_height, nz, dz_m)

    pressure_top = float(grid.vertical.top_pressure_pa)
    pressure_surface = np.asarray(_pressure_at_height(terrain_height), dtype=np.float64)
    mu = pressure_surface - pressure_top
    eta_face = np.linspace(1.0, 0.0, nz + 1, dtype=np.float64)
    eta_mass = 0.5 * (eta_face[:-1] + eta_face[1:])
    pressure = pressure_top + eta_mass[:, None, None] * mu[None, :, :]

    flat_mu = float(P0_PA - pressure_top)
    pb_profile = pressure_top + eta_mass * flat_mu
    pb = np.broadcast_to(pb_profile[:, None, None], (nz, ny, nx))

    x_mass = (np.arange(nx, dtype=np.float64) + 0.5) * dx_m
    domain_x = nx * dx_m
    periodic_dx = np.minimum(np.abs(x_mass - bubble_center_x_m), domain_x - np.abs(x_mass - bubble_center_x_m))
    r2 = periodic_dx[None, None, :] ** 2 + (z_mass - bubble_center_z_m) ** 2
    theta_reference = _theta_from_pressure(pressure)
    theta_pert = bubble_amplitude_k * np.exp(-r2 / (bubble_radius_m * bubble_radius_m))
    theta = theta_reference + theta_pert
    ph = GRAVITY_M_S2 * z_face

    state = State.zeros(grid).replace(
        theta=jnp.asarray(theta),
        p=jnp.asarray(pressure),
        pb=jnp.asarray(pb),
        ph=jnp.asarray(ph),
        mu=jnp.asarray(mu),
        t_skin=jnp.full((ny, nx), T0_K, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float32),
        mavail=jnp.ones((ny, nx), dtype=jnp.float32),
        roughness_m=jnp.full((ny, nx), 0.1, dtype=jnp.float64),
        rhosfc=jnp.asarray(pressure_surface / (R_DRY_AIR * T0_K)),
    )
    return state, Tendencies.zeros(grid), jnp.asarray(theta_reference), jnp.asarray(z_mass)


def _diagnose(state: State, theta_reference: jax.Array, z_mass: jax.Array) -> Diagnostics:
    theta_pert = state.theta - theta_reference
    positive = jnp.maximum(theta_pert, 0.0)
    weight = jnp.sum(positive)
    centroid_z = jnp.where(weight > 0.0, jnp.sum(positive * z_mass) / weight, jnp.nan)
    finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]))
    return Diagnostics(
        w_max_m_s=jnp.max(state.w),
        w_min_m_s=jnp.min(state.w),
        centroid_z_m=centroid_z,
        p_min_pa=jnp.min(state.p),
        p_max_pa=jnp.max(state.p),
        theta_min_k=jnp.min(state.theta),
        theta_max_k=jnp.max(state.theta),
        mu_min_pa=jnp.min(state.mu),
        mu_max_pa=jnp.max(state.mu),
        finite_state=finite,
    )


@partial(jax.jit, static_argnames=("grid", "dt_s", "steps", "n_acoustic"))
def _run(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    theta_reference: jax.Array,
    z_mass: jax.Array,
    *,
    dt_s: float,
    steps: int,
    n_acoustic: int,
):
    def body(carry, _):
        nxt = dycore_step(carry, tendencies, grid, float(dt_s), n_acoustic=int(n_acoustic), debug=False)
        return nxt, _diagnose(nxt, theta_reference, z_mass)

    return jax.lax.scan(body, state, xs=None, length=int(steps))


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


def _case_result(
    label: str,
    grid: GridSpec,
    args: argparse.Namespace,
    *,
    steps: int,
) -> dict[str, Any]:
    state, tendencies, theta_reference, z_mass = _initial_state(
        grid,
        dz_m=args.dz_m,
        bubble_center_x_m=0.5 * args.nx * args.dx_m,
        bubble_center_z_m=args.bubble_center_z_m,
        bubble_radius_m=args.bubble_radius_m,
        bubble_amplitude_k=args.bubble_amplitude_k,
    )
    initial = _diagnose(state, theta_reference, z_mass)
    start = time.perf_counter()
    final, scanned = _run(
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
    diag = _series(initial, scanned)
    nonfinite_step = _first_nonfinite_step(diag["finite_state"])
    surviving_seconds = float(args.duration_s) if nonfinite_step is None else max(0.0, (nonfinite_step - 1) * float(args.dt_s))
    times_s = [float(i) * float(args.dt_s) for i in range(steps + 1)]
    return {
        "label": label,
        "terrain_max_m": float(jnp.max(grid.terrain_height)),
        "terrain_min_m": float(jnp.min(grid.terrain_height)),
        "first_nonfinite_step": nonfinite_step,
        "surviving_seconds": surviving_seconds,
        "runtime_s_including_compile": elapsed_s,
        "time_s": times_s,
        "w_max_t_m_s": diag["w_max_m_s"],
        "centroid_z_t_m": diag["centroid_z_m"],
        "diagnostics_time_series": diag,
    }


def _interpret(flat: dict[str, Any], mountain: dict[str, Any]) -> str:
    flat_survives = flat["first_nonfinite_step"] is None and float(flat["surviving_seconds"]) >= 3000.0
    mountain_blows_early = mountain["first_nonfinite_step"] is not None and float(mountain["surviving_seconds"]) < 300.0
    flat_blows_early = flat["first_nonfinite_step"] is not None and float(flat["surviving_seconds"]) < 300.0
    mountain_survives = mountain["first_nonfinite_step"] is None and float(mountain["surviving_seconds"]) >= 3000.0
    if flat_survives and mountain_blows_early:
        return "METRIC_TERMS_OR_BASE_STATE_MISSING"
    if flat_blows_early:
        return "FLAT_ALSO_UNSTABLE_FORMULATION_DOMINANT"
    if flat_survives and mountain_survives:
        return "C1_STABLE_IN_THIS_ISOLATED_TERRAIN_PROBE"
    return "MIXED_OR_SLOW_INSTABILITY"


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
    parser.add_argument("--duration-s", type=float, default=3000.0)
    parser.add_argument("--n-acoustic", type=int, default=8)
    parser.add_argument("--mountain-peak-m", type=float, default=250.0)
    parser.add_argument("--mountain-half-width-m", type=float, default=5000.0)
    parser.add_argument("--bubble-center-z-m", type=float, default=2000.0)
    parser.add_argument("--bubble-radius-m", type=float, default=2000.0)
    parser.add_argument("--bubble-amplitude-k", type=float, default=2.0)
    args = parser.parse_args(argv)

    steps = int(round(float(args.duration_s) / float(args.dt_s)))
    if abs(steps * float(args.dt_s) - float(args.duration_s)) > 1.0e-9:
        raise ValueError("duration must be an integer number of dt steps")

    flat_terrain = np.zeros((int(args.ny), int(args.nx)), dtype=np.float64)
    mountain_terrain = _schar_terrain(
        int(args.nx),
        int(args.ny),
        float(args.dx_m),
        peak_m=float(args.mountain_peak_m),
        half_width_m=float(args.mountain_half_width_m),
    )
    flat_grid = _grid(args.nx, args.ny, args.nz, args.dx_m, args.dy_m, args.dz_m, flat_terrain, label="flat")
    mountain_grid = _grid(
        args.nx,
        args.ny,
        args.nz,
        args.dx_m,
        args.dy_m,
        args.dz_m,
        mountain_terrain,
        label="schar-mountain",
    )

    flat = _case_result("flat", flat_grid, args, steps=steps)
    mountain = _case_result("schar_mountain", mountain_grid, args, steps=steps)
    interpretation = _interpret(flat, mountain)
    payload = {
        "artifact_type": "m6x_numerical_stability_spike_test1_flat_vs_mountain",
        "description": "Flat terrain vs compact Schar-style 250 m ridge warm-bubble c1 dycore diagnostic.",
        "citation_note": "Schar-style compact ridge used as the cheap mountain-wave proxy requested by the sprint prompt.",
        "setup": {
            "grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz, "dx_m": args.dx_m, "dy_m": args.dy_m, "dz_m": args.dz_m},
            "dt_s": args.dt_s,
            "duration_s": args.duration_s,
            "steps": steps,
            "n_acoustic": args.n_acoustic,
            "physics": "off",
            "dycore_path": "gpuwrf.dynamics.step.step, c1 acoustic + buoyancy via acoustic.py",
            "terrain_pressure_reference": "state.p follows terrain-column sigma pressure; state.pb is flat-column pressure profile to expose missing terrain metric/base-state cancellation.",
            "mountain": {"peak_m": args.mountain_peak_m, "half_width_m": args.mountain_half_width_m, "profile": "compact cos^2 ridge"},
            "bubble": {
                "center_x_m": 0.5 * args.nx * args.dx_m,
                "center_z_m": args.bubble_center_z_m,
                "radius_m": args.bubble_radius_m,
                "amplitude_k": args.bubble_amplitude_k,
            },
        },
        "cases": {"flat": flat, "schar_mountain": mountain},
        "interpretation": interpretation,
        "runtime": {"jax_devices": [str(device) for device in jax.devices()]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "interpretation": interpretation,
                "flat": {
                    "first_nonfinite_step": flat["first_nonfinite_step"],
                    "surviving_seconds": flat["surviving_seconds"],
                },
                "schar_mountain": {
                    "first_nonfinite_step": mountain["first_nonfinite_step"],
                    "surviving_seconds": mountain["surviving_seconds"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
