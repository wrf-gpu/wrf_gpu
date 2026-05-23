#!/usr/bin/env python3
"""Run the M6.x warm-bubble operator-sanity probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

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
CURRENT_SPRINT = "2026-05-23-m6x-warm-bubble-gate-redesign"
DEFAULT_OUTPUT = ROOT / ".agent" / "sprints" / CURRENT_SPRINT / "proof_current_state_verdict.json"

PASS_OPERATOR_SANITY = "PASS_OPERATOR_SANITY"
FAIL_FINITENESS = "FAIL_FINITENESS"
FAIL_PHYSICAL_BOUNDS = "FAIL_PHYSICAL_BOUNDS"
FAIL_ANTI_CLAMP_DETECTION = "FAIL_ANTI_CLAMP_DETECTION"

PRODUCTION_SCAN_PATHS = (
    ROOT / "src" / "gpuwrf" / "dynamics" / "acoustic_wrf.py",
    ROOT / "src" / "gpuwrf" / "dynamics" / "vertical_implicit_solver.py",
)

SAMPLE_FIELDS = (
    "w_max_m_s",
    "w_min_m_s",
    "w_abs_max_m_s",
    "theta_perturbation_max_K",
    "theta_perturbation_min_K",
    "p_perturbation_max_Pa",
    "p_perturbation_min_Pa",
    "mu_perturbation_max_Pa",
    "mu_perturbation_min_Pa",
    "centroid_z_m",
    "mu_residual_Pa",
)

PHYSICAL_BOUND_CHECKS = (
    ("theta_perturbation_max_K", "max", 50.0, "K"),
    ("theta_perturbation_min_K", "min", -50.0, "K"),
    ("p_perturbation_max_Pa", "max", 50000.0, "Pa"),
    ("p_perturbation_min_Pa", "min", -50000.0, "Pa"),
    ("mu_perturbation_max_Pa", "max", 50000.0, "Pa"),
)

AMPLITUDE_LITERAL_RE = re.compile(r"(?<![\w.])(?:[5-9](?:\.0+)?|10(?:\.0+)?)(?![\w.])")
TARGET_TANH_RE = re.compile(r"(?:jnp\.)?tanh\s*\([^#\n]*/\s*(?:[5-9](?:\.0+)?|10(?:\.0+)?)")
TARGET_SCALE_TANH_RE = re.compile(r"(?<![\w.])(?:[5-9](?:\.0+)?|10(?:\.0+)?)\s*\*\s*(?:jnp\.)?tanh")
POSITIVE_ONLY_W_RE = re.compile(
    r"(?:jnp\.)?maximum\s*\(\s*(?:state\.)?(?:w|w_next|next_w|vertical_velocity)\b[^,]*,\s*0(?:\.0+)?"
)
THETA_CLIP_RE = re.compile(r"(?:jnp\.)?(?:minimum|clip)\s*\([^#\n]*(?:theta_target|theta_[a-z_]*target)")


class Diagnostics(NamedTuple):
    w_max_m_s: object
    w_min_m_s: object
    w_abs_max_m_s: object
    theta_perturbation_max_K: object
    theta_perturbation_min_K: object
    p_perturbation_max_Pa: object
    p_perturbation_min_Pa: object
    mu_perturbation_max_Pa: object
    mu_perturbation_min_Pa: object
    centroid_z_m: object
    mu_residual_Pa: object
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
) -> tuple[State, BaseState, jax.Array, jax.Array, jax.Array]:
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
    return state, base, jnp.asarray(theta_base), jnp.asarray(z_mass), state.mu_perturbation


def _diagnose(state: State, theta_base: jax.Array, z_mass: jax.Array, initial_mu_perturbation: jax.Array) -> Diagnostics:
    theta_perturbation = state.theta - theta_base
    positive = jnp.maximum(theta_perturbation, 0.0)
    weight = jnp.sum(positive)
    centroid_z = jnp.where(weight > 0.0, jnp.sum(positive * z_mass) / weight, jnp.nan)
    finite = jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(state)]))
    return Diagnostics(
        w_max_m_s=jnp.max(state.w),
        w_min_m_s=jnp.min(state.w),
        w_abs_max_m_s=jnp.max(jnp.abs(state.w)),
        theta_perturbation_max_K=jnp.max(theta_perturbation),
        theta_perturbation_min_K=jnp.min(theta_perturbation),
        p_perturbation_max_Pa=jnp.max(state.p_perturbation),
        p_perturbation_min_Pa=jnp.min(state.p_perturbation),
        mu_perturbation_max_Pa=jnp.max(state.mu_perturbation),
        mu_perturbation_min_Pa=jnp.min(state.mu_perturbation),
        centroid_z_m=centroid_z,
        mu_residual_Pa=jnp.max(jnp.abs(state.mu_perturbation - initial_mu_perturbation)),
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
    initial_mu_perturbation: jax.Array,
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
        diagnostics = _diagnose(next_carry.state, theta_base, z_mass, initial_mu_perturbation)
        return (next_carry.state, next_carry.previous_pressure), diagnostics

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


def _sample_payload(series: dict[str, list[float | bool | None]], dt_s: float, target_s: float) -> dict[str, Any]:
    index = int(round(target_s / dt_s))
    sample = {"time_s": float(target_s), "step": int(index)}
    for field in SAMPLE_FIELDS:
        sample[field] = _sample_at(series[field], dt_s, target_s)
    return sample


def _collect_bound_violations(series: dict[str, list[float | bool | None]], tolerance: float = 0.0) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for field, direction, bound, units in PHYSICAL_BOUND_CHECKS:
        values = np.asarray([np.nan if value is None else float(value) for value in series[field]], dtype=np.float64)
        if values.size == 0 or np.all(np.isnan(values)):
            continue
        if direction == "max":
            violating = np.where(values > bound + tolerance)[0]
            if violating.size == 0:
                continue
            step = int(violating[np.nanargmax(values[violating])])
            comparator = "<="
        else:
            violating = np.where(values < bound - tolerance)[0]
            if violating.size == 0:
                continue
            step = int(violating[np.nanargmin(values[violating])])
            comparator = ">="
        violations.append(
            {
                "field": field,
                "step": step,
                "value": float(values[step]),
                "bound": float(bound),
                "comparator": comparator,
                "tolerance": float(tolerance),
                "units": units,
            }
        )
    return violations


def _warning(
    *,
    path: str,
    line_number: int,
    rule: str,
    message: str,
    snippet: str,
    hard_fail: bool,
) -> dict[str, Any]:
    return {
        "path": path,
        "line": int(line_number),
        "rule": rule,
        "message": message,
        "snippet": snippet.strip(),
        "hard_fail": bool(hard_fail),
    }


def _scan_text_for_anti_clamp_patterns(path: str, text: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if TARGET_TANH_RE.search(line) or TARGET_SCALE_TANH_RE.search(line):
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="target_band_tanh_clamp",
                    message="target-shaped tanh clamp tied to a 5-10 m/s amplitude literal",
                    snippet=line,
                    hard_fail=True,
                )
            )
        if POSITIVE_ONLY_W_RE.search(line):
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="positive_only_w_velocity",
                    message="vertical velocity is clipped through a positive-only maximum",
                    snippet=line,
                    hard_fail=True,
                )
            )
        if THETA_CLIP_RE.search(line):
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="theta_target_clipping",
                    message="theta perturbation appears clipped to a target value",
                    snippet=line,
                    hard_fail=True,
                )
            )
        if "lift_bias" in lower or "updraft_drag" in lower:
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="lift_bias_or_updraft_drag",
                    message="explicit lift-bias or updraft-drag stabilizer name found",
                    snippet=line,
                    hard_fail=True,
                )
            )
        if (
            AMPLITUDE_LITERAL_RE.search(line)
            and "w" in lower
            and any(token in lower for token in ("clip", "clamp", "bound", "target", "limit", "maximum", "minimum", "tanh"))
        ):
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="target_band_w_bound_literal",
                    message="w-related bound/target logic contains a 5-10 m/s amplitude literal",
                    snippet=line,
                    hard_fail=True,
                )
            )
        if re.search(r"(?<![\w.])0\.38(?:0+)?(?![\w.])", line):
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="documented_mpas_slice_magic_0_38",
                    message="documented ADR-023 slice-oracle constant; warning only",
                    snippet=line,
                    hard_fail=False,
                )
            )
        if re.search(r"(?<![\w.])1\.35(?:0+)?(?![\w.])", line):
            warnings.append(
                _warning(
                    path=path,
                    line_number=line_number,
                    rule="documented_mpas_slice_magic_1_35",
                    message="documented ADR-023 slice-oracle conversion constant; warning only",
                    snippet=line,
                    hard_fail=False,
                )
            )
    return warnings


def scan_anti_clamp_patterns(
    paths: Sequence[Path] | None = None,
    *,
    source_text_by_path: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Scans production-path source for target-shaped warm-bubble clamps."""

    if source_text_by_path is not None:
        warnings: list[dict[str, Any]] = []
        for path, text in source_text_by_path.items():
            warnings.extend(_scan_text_for_anti_clamp_patterns(path, text))
        return warnings

    warnings = []
    for path in paths or PRODUCTION_SCAN_PATHS:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            warnings.append(
                _warning(
                    path=str(path),
                    line_number=0,
                    rule="missing_scan_path",
                    message="anti-clamp scan path is missing",
                    snippet="",
                    hard_fail=True,
                )
            )
            continue
        warnings.extend(_scan_text_for_anti_clamp_patterns(str(Path(path).relative_to(ROOT)), text))
    return warnings


def _run_pytest_precondition(name: str, nodeids: Sequence[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", *nodeids, "-q"]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_precondition_checks() -> dict[str, Any]:
    """Runs the oracle prerequisites required before an operator-sanity pass."""

    return {
        "r7_oracle": _run_pytest_precondition(
            "r7_oracle",
            (
                "tests/test_m6x_vertical_acoustic_oracle.py::test_linear_acoustic_period_matches_dispersion_relation",
                "tests/test_m6x_vertical_acoustic_oracle.py::test_amplitude_decay_within_2pct_of_analytic",
            ),
        ),
        "hydrostatic_rest": _run_pytest_precondition(
            "hydrostatic_rest",
            ("tests/test_m6x_vertical_acoustic_oracle.py::test_no_drift_in_hydrostatic_rest_state",),
        ),
    }


def _preconditions_ok(preconditions: Mapping[str, Any] | None) -> bool:
    if not preconditions:
        return True
    return all(bool(result.get("ok", False)) for result in preconditions.values())


def _verdict(payload: Mapping[str, Any]) -> str:
    if not _preconditions_ok(payload.get("preconditions")):
        return FAIL_FINITENESS
    if payload["first_nonfinite_step"] is not None:
        return FAIL_FINITENESS
    if payload["bound_violations"]:
        return FAIL_PHYSICAL_BOUNDS
    if any(bool(warning.get("hard_fail", False)) for warning in payload["anti_clamp_warnings"]):
        return FAIL_ANTI_CLAMP_DETECTION
    return PASS_OPERATOR_SANITY


def run_warm_bubble_operator_sanity(
    *,
    nx: int = 64,
    ny: int = 64,
    nz: int = 40,
    dx_m: float = 400.0,
    dy_m: float = 400.0,
    dz_m: float = 100.0,
    dt_s: float = 2.0,
    duration_s: float = 600.0,
    n_acoustic: int = 8,
    smdiv: float = 0.0,
    rayleigh: float = 0.0,
    bubble_center_z_m: float = 2000.0,
    bubble_radius_m: float = 2000.0,
    bubble_amplitude_k: float = 2.0,
    run_preconditions: bool = True,
) -> dict[str, Any]:
    steps = int(round(float(duration_s) / float(dt_s)))
    if abs(steps * float(dt_s) - float(duration_s)) > 1.0e-9:
        raise ValueError("duration must be an integer number of dt steps")

    grid = _grid(nx, ny, nz, dx_m, dy_m, dz_m)
    state, base, theta_base, z_mass, initial_mu_perturbation = _initial_state(
        grid,
        dz_m=dz_m,
        bubble_center_x_m=0.5 * nx * dx_m,
        bubble_center_z_m=bubble_center_z_m,
        bubble_radius_m=bubble_radius_m,
        bubble_amplitude_k=bubble_amplitude_k,
    )
    acoustic_config = AcousticConfig(
        n_substeps=int(n_acoustic),
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        smdiv=SmdivConfig(enabled=bool(smdiv), coefficient=float(smdiv)),
        rayleigh=RayleighConfig(enabled=bool(rayleigh), coefficient=float(rayleigh)),
    )
    initial = _diagnose(state, theta_base, z_mass, initial_mu_perturbation)
    start = time.perf_counter()
    (final_state, _final_previous_pressure), scanned = _run(
        state,
        state.p_perturbation,
        grid.metrics,
        base,
        theta_base,
        z_mass,
        initial_mu_perturbation,
        acoustic_config,
        float(dt_s),
        steps,
    )
    jax.block_until_ready(final_state.p_total)
    elapsed_s = time.perf_counter() - start
    series = _series(initial, scanned)
    nonfinite_step = _first_nonfinite_step(series["finite_state"])
    samples = {
        "300s": _sample_payload(series, dt_s, 300.0),
        "600s": _sample_payload(series, dt_s, 600.0),
    }
    payload: dict[str, Any] = {
        "artifact_type": "m6x_warm_bubble_operator_sanity",
        "description": "Idealized flat warm-bubble acoustic-scan operator-sanity probe; w amplitude is diagnostic only.",
        "setup": {
            "grid": {"nx": nx, "ny": ny, "nz": nz, "dx_m": dx_m, "dy_m": dy_m, "dz_m": dz_m},
            "dt_s": dt_s,
            "duration_s": duration_s,
            "steps": steps,
            "n_acoustic": n_acoustic,
            "bubble": {
                "center_x_m": 0.5 * nx * dx_m,
                "center_z_m": bubble_center_z_m,
                "radius_m": bubble_radius_m,
                "amplitude_k": bubble_amplitude_k,
            },
            "smdiv": smdiv,
            "rayleigh": rayleigh,
        },
        "legacy_amplitude_band_m_s": {
            "range": [5.0, 10.0],
            "gate": False,
            "reason": "unsourced for this pure-small-step Gaussian harness",
        },
        "first_nonfinite_step": nonfinite_step,
        "surviving_seconds": float(duration_s) if nonfinite_step is None else max(0.0, (nonfinite_step - 1) * float(dt_s)),
        "samples": samples,
        "bound_violations": _collect_bound_violations(series),
        "anti_clamp_warnings": scan_anti_clamp_patterns(),
        "diagnostics_time_series": series,
        "runtime_s_including_compile": elapsed_s,
        "jax_devices": [str(device) for device in jax.devices()],
        "wrf_source_anchors": {
            "horizontal_pgf": "module_small_step_em.F:828-862,902-936",
            "diagnostic_pressure_al_alt": "module_big_step_utilities_em.F:1025-1030,1082-1087,910-943",
            "mu_continuity": "module_small_step_em.F:1094-1108",
        },
    }
    payload["preconditions"] = run_precondition_checks() if run_preconditions else {}
    payload["verdict"] = _verdict(payload)
    return payload


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
    parser.add_argument(
        "--skip-preconditions",
        action="store_true",
        help="skip R7 oracle and hydrostatic-rest preconditions; intended only for focused unit tests",
    )
    args = parser.parse_args(argv)

    payload = run_warm_bubble_operator_sanity(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        dx_m=args.dx_m,
        dy_m=args.dy_m,
        dz_m=args.dz_m,
        dt_s=args.dt_s,
        duration_s=args.duration_s,
        n_acoustic=args.n_acoustic,
        smdiv=args.smdiv,
        rayleigh=args.rayleigh,
        bubble_center_z_m=args.bubble_center_z_m,
        bubble_radius_m=args.bubble_radius_m,
        bubble_amplitude_k=args.bubble_amplitude_k,
        run_preconditions=not args.skip_preconditions,
    )
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
                "bound_violations": payload["bound_violations"],
                "anti_clamp_hard_failures": [
                    warning for warning in payload["anti_clamp_warnings"] if warning.get("hard_fail")
                ],
                "preconditions_ok": _preconditions_ok(payload.get("preconditions")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["verdict"] == PASS_OPERATOR_SANITY else 2


if __name__ == "__main__":
    raise SystemExit(main())
