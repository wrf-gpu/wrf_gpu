#!/usr/bin/env python3
"""Run the M6.x Tier-3 idealized dt-convergence smoke harness."""

from __future__ import annotations

import argparse
import hashlib
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
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.validation.tier3_envelope import (
    build_norm_table,
    checkpoint_key,
    classify_convergence,
    validate_tsc_payload,
)


config.update("jax_enable_x64", True)

P0_PA = 100000.0
T0_K = 300.0
R_DRY_AIR = 287.05
CP_DRY_AIR = 1004.0
KAPPA = R_DRY_AIR / CP_DRY_AIR
GRAVITY_M_S2 = 9.80665

DEFAULT_CASE_PATH = ROOT / "data" / "fixtures" / "tier3_idealized" / "case_definition.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "m6" / "tier3" / "tsc_envelope.json"
VARIABLES = ("U", "V", "W", "theta", "p_perturbation", "mu_perturbation")


class Snapshot(NamedTuple):
    U: object
    V: object
    W: object
    theta: object
    p_perturbation: object
    mu_perturbation: object
    finite_state: object


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_definition(case: str) -> dict[str, Any]:
    """Load the named case from the sprint's current single-case JSON file."""

    path = Path(case)
    if path.exists():
        payload = _load_json(path)
    else:
        payload = _load_json(DEFAULT_CASE_PATH)
        if payload["case_name"] != case:
            raise ValueError(f"case {case!r} is not available in {DEFAULT_CASE_PATH}")
    return payload


def _pressure_at_height(z_m: np.ndarray | float) -> np.ndarray:
    return P0_PA * np.exp(-GRAVITY_M_S2 * np.asarray(z_m, dtype=np.float64) / (R_DRY_AIR * T0_K))


def _theta_from_pressure(pressure_pa: np.ndarray) -> np.ndarray:
    return T0_K * (P0_PA / np.asarray(pressure_pa, dtype=np.float64)) ** KAPPA


def _grid(case: dict[str, Any]) -> GridSpec:
    grid_meta = case["grid"]
    nx = int(grid_meta["nx"])
    ny = int(grid_meta["ny"])
    nz = int(grid_meta["nz"])
    dx_m = float(grid_meta["dx_m"])
    dy_m = float(grid_meta["dy_m"])
    dz_m = float(grid_meta["dz_meta"]["dz_m"])
    z_top = float(nz) * dz_m
    top_pressure = float(_pressure_at_height(z_top))
    projection = Projection("lambert", 0.0, 0.0, dx_m, dy_m, nx, ny)
    terrain = TerrainProvenance(
        source_path="synthetic://m6-tier3-flat-warm-bubble",
        sha256="analytic-m6-tier3-flat-warm-bubble",
        shape=(ny, nx),
        units="m",
        projection_transform="cartesian-flat",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", nz, top_pressure, eta)
    bc = BCMetadata("ideal", VARIABLES, 0, "linear", False)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta, terrain_height)


def _cosine_bell_perturbation(case: dict[str, Any], z_mass: np.ndarray) -> np.ndarray:
    grid_meta = case["grid"]
    ic = case["initial_conditions"]
    form = ic["theta_perturbation"]
    nx = int(grid_meta["nx"])
    ny = int(grid_meta["ny"])
    dx_m = float(grid_meta["dx_m"])
    dy_m = float(grid_meta["dy_m"])
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx_m
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy_m
    center_x = float(form["center_x_fraction"]) * nx * dx_m
    center_y = float(form["center_y_fraction"]) * ny * dy_m
    center_z = float(form["center_z_m"])
    rx = float(form["radius_x_m"])
    ry = float(form["radius_y_m"])
    rz = float(form["radius_z_m"])
    x_term = ((x[None, None, :] - center_x) / rx) ** 2
    y_term = ((y[None, :, None] - center_y) / ry) ** 2
    z_term = ((z_mass - center_z) / rz) ** 2
    radius = np.sqrt(x_term + y_term + z_term)
    cap = np.where(radius <= 1.0, 0.5 * (1.0 + np.cos(np.pi * radius)), 0.0)
    y_modulation = 1.0 + float(form.get("y_modulation_fraction", 0.0)) * np.cos(2.0 * np.pi * (y[None, :, None] / (ny * dy_m)))
    return float(form["amplitude_k"]) * cap * y_modulation


def _initial_state(case: dict[str, Any], grid: GridSpec) -> tuple[State, BaseState, dict[str, Any]]:
    grid_meta = case["grid"]
    wind = case["initial_conditions"]["wind"]
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    dz_m = float(grid_meta["dz_meta"]["dz_m"])
    z_face_1d = np.arange(nz + 1, dtype=np.float64) * dz_m
    z_mass_1d = 0.5 * (z_face_1d[:-1] + z_face_1d[1:])
    z_face = np.broadcast_to(z_face_1d[:, None, None], (nz + 1, ny, nx))
    z_mass = np.broadcast_to(z_mass_1d[:, None, None], (nz, ny, nx))
    pb_1d = _pressure_at_height(z_mass_1d)
    pb = np.broadcast_to(pb_1d[:, None, None], (nz, ny, nx)).copy()
    theta_base_1d = _theta_from_pressure(pb_1d)
    theta_base = np.broadcast_to(theta_base_1d[:, None, None], (nz, ny, nx)).copy()
    theta = theta_base + _cosine_bell_perturbation(case, z_mass)
    phb = GRAVITY_M_S2 * z_face
    mub = np.full((ny, nx), P0_PA - float(grid.vertical.top_pressure_pa), dtype=np.float64)

    state = State.zeros(grid).replace(
        u=jnp.ones((nz, ny, nx + 1), dtype=jnp.float64) * float(wind["u_background_m_s"]),
        v=jnp.ones((nz, ny + 1, nx), dtype=jnp.float64) * float(wind["v_background_m_s"]),
        w=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        theta=jnp.asarray(theta),
        qv=jnp.ones((nz, ny, nx), dtype=jnp.float64) * float(case["initial_conditions"].get("qv_kg_kg", 0.0)),
        p=jnp.asarray(pb),
        p_total=jnp.asarray(pb),
        p_perturbation=jnp.zeros((nz, ny, nx), dtype=jnp.float64),
        ph=jnp.asarray(phb),
        ph_total=jnp.asarray(phb),
        ph_perturbation=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        mu=jnp.asarray(mub),
        mu_total=jnp.asarray(mub),
        mu_perturbation=jnp.zeros((ny, nx), dtype=jnp.float64),
        t_skin=jnp.full((ny, nx), T0_K, dtype=jnp.float64),
        xland=jnp.ones((ny, nx), dtype=jnp.float32),
        mavail=jnp.ones((ny, nx), dtype=jnp.float32),
        roughness_m=jnp.full((ny, nx), 0.05, dtype=jnp.float64),
        rhosfc=jnp.full((ny, nx), P0_PA / (R_DRY_AIR * T0_K), dtype=jnp.float64),
    )
    base = BaseState(
        pb=jnp.asarray(pb),
        phb=jnp.asarray(phb),
        mub=jnp.asarray(mub),
        t0=jnp.asarray(theta_base),
        theta_base=jnp.asarray(theta_base),
    )
    metadata = {
        "theta_perturbation_max_k": float(np.max(theta - theta_base)),
        "theta_perturbation_min_k": float(np.min(theta - theta_base)),
        "top_pressure_pa": float(grid.vertical.top_pressure_pa),
    }
    return state, base, metadata


def _snapshot(state: State) -> Snapshot:
    arrays = (
        state.u,
        state.v,
        state.w,
        state.theta,
        state.p_perturbation,
        state.mu_perturbation,
    )
    finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(array)) for array in arrays]))
    return Snapshot(*arrays, finite)


@partial(jax.jit, static_argnames=("config", "dt_s", "steps"))
def _run_steps(
    state: State,
    previous_pressure: jax.Array,
    metrics,
    base_state: BaseState,
    config: AcousticConfig,
    dt_s: float,
    steps: int,
):
    def body(carry, _):
        carry_state, carry_previous_pressure = carry
        next_carry = run_acoustic_scan_carry(
            carry_state,
            carry_previous_pressure,
            metrics,
            config,
            float(dt_s),
            base_state,
        )
        return (next_carry.state, next_carry.previous_pressure), _snapshot(next_carry.state)

    return jax.lax.scan(body, (state, previous_pressure), xs=None, length=int(steps))


def _as_np(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _hash_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _stats(value: np.ndarray) -> dict[str, float | None]:
    if not np.all(np.isfinite(value)):
        return {"min": None, "max": None, "mean": None}
    return {"min": float(np.min(value)), "max": float(np.max(value)), "mean": float(np.mean(value))}


def _sample_snapshot(initial: Snapshot, scanned: Snapshot, step: int) -> dict[str, np.ndarray]:
    source = initial if step == 0 else Snapshot(*(getattr(scanned, name)[step - 1] for name in Snapshot._fields))
    return {
        "U": _as_np(source.U),
        "V": _as_np(source.V),
        "W": _as_np(source.W),
        "theta": _as_np(source.theta),
        "p_perturbation": _as_np(source.p_perturbation),
        "mu_perturbation": _as_np(source.mu_perturbation),
    }


def _first_nonfinite_step(initial: Snapshot, scanned: Snapshot) -> int | None:
    initial_finite = bool(np.asarray(jax.device_get(initial.finite_state)))
    if not initial_finite:
        return 0
    finite_series = np.asarray(jax.device_get(scanned.finite_state), dtype=bool)
    bad = np.where(~finite_series)[0]
    return None if bad.size == 0 else int(bad[0] + 1)


def _steps_for(dt_s: float, total_time_s: float) -> int:
    steps = int(round(float(total_time_s) / float(dt_s)))
    if abs(steps * float(dt_s) - float(total_time_s)) > 1.0e-9:
        raise ValueError(f"total_time_s={total_time_s} is not divisible by dt={dt_s}")
    return steps


def _checkpoint_steps(checkpoints_s: list[float], dt_s: float, steps: int) -> dict[float, int]:
    out: dict[float, int] = {}
    for checkpoint_s in checkpoints_s:
        step = int(round(float(checkpoint_s) / float(dt_s)))
        if abs(step * float(dt_s) - float(checkpoint_s)) > 1.0e-9:
            raise ValueError(f"checkpoint {checkpoint_s}s is not divisible by dt={dt_s}")
        if step < 0 or step > steps:
            raise ValueError(f"checkpoint {checkpoint_s}s is outside integration window")
        out[float(checkpoint_s)] = step
    return out


def run_one_dt(case: dict[str, Any], dt_s: float) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, Any]]:
    """Run the idealized case once and return checkpoint snapshots and metadata."""

    grid = _grid(case)
    state, base, ic_metadata = _initial_state(case, grid)
    total_time_s = float(case["total_integration_time_s"])
    steps = _steps_for(dt_s, total_time_s)
    checkpoint_steps = _checkpoint_steps([float(item) for item in case["checkpoints_s"]], dt_s, steps)
    config_meta = case["solver"]
    acoustic_config = AcousticConfig(
        n_substeps=int(config_meta["n_acoustic_substeps"]),
        dx_m=float(case["grid"]["dx_m"]),
        dy_m=float(case["grid"]["dy_m"]),
        epssm=float(config_meta["epssm"]),
        smdiv=SmdivConfig(enabled=bool(config_meta["smdiv"]), coefficient=float(config_meta["smdiv"])),
        rayleigh=RayleighConfig(enabled=bool(config_meta["rayleigh"]), coefficient=float(config_meta["rayleigh"])),
    )
    initial = _snapshot(state)
    start = time.perf_counter()
    (final_state, final_previous_pressure), scanned = _run_steps(
        state,
        state.p_perturbation,
        grid.metrics,
        base,
        acoustic_config,
        float(dt_s),
        int(steps),
    )
    block_until_ready((final_state, final_previous_pressure, scanned))
    wall_time_s = time.perf_counter() - start
    snapshots = {checkpoint: _sample_snapshot(initial, scanned, step) for checkpoint, step in checkpoint_steps.items()}
    final_snapshot = snapshots[float(total_time_s)]
    metadata = {
        "dt_s": float(dt_s),
        "steps": int(steps),
        "wall_time_s": float(wall_time_s),
        "kernel_launches": None,
        "kernel_launch_note": "not measured in smoke runner; one compiled JAX scan is used per dt run",
        "first_nonfinite_step": _first_nonfinite_step(initial, scanned),
        "jax_backend": jax.default_backend(),
        "gpu_name": visible_gpu_name(),
        "initial_condition": ic_metadata,
        "final_state_sha256": {name: _hash_array(value) for name, value in final_snapshot.items()},
        "checkpoint_summaries": {
            checkpoint_key(checkpoint): {name: _stats(value) for name, value in snapshot.items()}
            for checkpoint, snapshot in snapshots.items()
        },
        "transfer_audit": {
            "host_to_device_bytes_inside_timestep_loop": 0,
            "device_to_host_bytes_inside_timestep_loop": 0,
            "method": "static JAX scan body; checkpoint arrays are transferred only after block_until_ready",
        },
    }
    return snapshots, metadata


def _dt_levels(case: dict[str, Any], base_dt_s: float) -> list[float]:
    factors = [float(item) for item in case["dt_refinement"]["refinement_factors"]]
    return [float(base_dt_s) * factor for factor in factors]


def _dt_pairs(dts: list[float]) -> list[dict[str, float]]:
    return [
        {"dt_coarse": float(dts[index]), "dt_fine": float(dts[index + 1]), "pair_index": int(index)}
        for index in range(len(dts) - 1)
    ]


def build_tsc_envelope(case_name: str, base_dt_s: float) -> dict[str, Any]:
    case = load_case_definition(case_name)
    dts = _dt_levels(case, base_dt_s)
    snapshots_by_dt: dict[float, dict[float, dict[str, np.ndarray]]] = {}
    per_dt_metadata = []
    for dt_s in dts:
        snapshots, metadata = run_one_dt(case, dt_s)
        snapshots_by_dt[float(dt_s)] = snapshots
        per_dt_metadata.append(metadata)

    checkpoints_s = [float(item) for item in case["checkpoints_s"]]
    pairs = _dt_pairs(dts)
    variables = list(case["variables_to_track"])
    norms = build_norm_table(snapshots_by_dt, dt_pairs=pairs, checkpoints_s=checkpoints_s, variables=variables)
    verdict, rationale = classify_convergence(
        norms,
        dt_pairs=pairs,
        per_dt_run_metadata=per_dt_metadata,
        criteria=case.get("pass_fail_criteria", {}),
    )
    payload = {
        "artifact_type": "m6_tier3_tsc_envelope",
        "case": case["case_name"],
        "config": {
            "boundary_mode": case["boundary_conditions"]["mode"],
            "physics": case["physics_toggles"]["scope"],
            "total_time_s": float(case["total_integration_time_s"]),
            "variables": variables,
        },
        "dt_pairs": pairs,
        "checkpoints_s": checkpoints_s,
        "per_dt_run_metadata": per_dt_metadata,
        "norms": norms,
        "convergence_verdict": verdict,
        "rationale": rationale,
        "case_definition_path": str(DEFAULT_CASE_PATH.relative_to(ROOT)),
        "schema_version": "m6-tier3-tsc-envelope-v1",
    }
    validate_tsc_payload(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="case name or path to a case_definition.json")
    parser.add_argument("--dt", required=True, type=float, help="coarsest dt in seconds")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be positive and finite")
    payload = build_tsc_envelope(args.case, args.dt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "verdict": payload["convergence_verdict"], "rationale": payload["rationale"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
