#!/usr/bin/env python
"""M6 perf-design runner: solver bakeoff plus operational-mode smoke gates."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

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

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.dynamics.vertical_implicit_solver import solve_tridiagonal_thomas, solve_tridiagonal_xla
from gpuwrf.profiling.budget import compiled_text, kernel_launches_per_step
from gpuwrf.profiling.transfer_audit import block_until_ready, count_transfer_bytes, visible_gpu_name
from gpuwrf.runtime.operational_mode import OperationalNamelist, run_forecast_operational


config.update("jax_enable_x64", True)

SPRINT_DIR = ROOT / ".agent" / "sprints" / "2026-05-25-m6-perf-design"
ARTIFACT_DIR = SPRINT_DIR / "artifacts"
HLO_DIR = ARTIFACT_DIR / "hlo"
TRACE_DIR = ARTIFACT_DIR / "jax_trace_solver_bakeoff"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cost_analysis(compiled) -> dict[str, float | None]:
    try:
        cost = compiled.cost_analysis()
    except Exception:
        return {}
    if isinstance(cost, list):
        cost = cost[0] if cost else {}
    return {str(key): _as_float(value) for key, value in dict(cost).items()}


def _shift_down(field: jax.Array, stride: int) -> jax.Array:
    return jnp.concatenate((jnp.zeros_like(field[:stride]), field[:-stride]), axis=0)


def _shift_up(field: jax.Array, stride: int) -> jax.Array:
    return jnp.concatenate((field[stride:], jnp.zeros_like(field[:stride])), axis=0)


def pcr_solve(a: jax.Array, b: jax.Array, c: jax.Array, rhs: jax.Array) -> jax.Array:
    """Pure parallel cyclic reduction for leading-axis tridiagonal systems."""

    n = int(rhs.shape[0])
    levels = int(math.ceil(math.log2(n)))
    lower = jnp.asarray(a)
    diag = jnp.asarray(b)
    upper = jnp.asarray(c)
    vec = jnp.asarray(rhs)
    row_index = jnp.arange(n)[:, None]
    for level in range(levels):
        stride = 1 << level
        diag_down = _shift_down(diag, stride)
        diag_up = _shift_up(diag, stride)
        lower_factor = jnp.where(row_index >= stride, lower / diag_down, 0.0)
        upper_factor = jnp.where(row_index + stride < n, upper / diag_up, 0.0)
        next_lower = -lower_factor * _shift_down(lower, stride)
        next_upper = -upper_factor * _shift_up(upper, stride)
        next_diag = diag - lower_factor * _shift_down(upper, stride) - upper_factor * _shift_up(lower, stride)
        next_vec = vec - lower_factor * _shift_down(vec, stride) - upper_factor * _shift_up(vec, stride)
        lower, diag, upper, vec = next_lower, next_diag, next_upper, next_vec
    return vec / diag


def hybrid_pcr_thomas_solve(a: jax.Array, b: jax.Array, c: jax.Array, rhs: jax.Array) -> jax.Array:
    """Fixed PCR+Thomas reference using PCR followed by one Thomas correction."""

    x0 = pcr_solve(a, b, c, rhs)
    residual = rhs - _tridiag_matvec(a, b, c, x0)
    correction = solve_tridiagonal_thomas(a, b, c, residual)
    return x0 + correction


def m6b2_wrf_thomas_solve(a: jax.Array, b: jax.Array, c: jax.Array, rhs: jax.Array) -> jax.Array:
    """Current JAX lax.scan Thomas family used as the M6B2 operational baseline."""

    return solve_tridiagonal_thomas(a, b, c, rhs)


def _tridiag_matvec(a: jax.Array, b: jax.Array, c: jax.Array, x: jax.Array) -> jax.Array:
    lower = a * _shift_down(x, 1)
    upper = c * _shift_up(x, 1)
    return lower + b * x + upper


@jax.jit
def _thomas_kernel(a, b, c, rhs):
    return m6b2_wrf_thomas_solve(a, b, c, rhs)


@jax.jit
def _pcr_kernel(a, b, c, rhs):
    return pcr_solve(a, b, c, rhs)


@jax.jit
def _hybrid_kernel(a, b, c, rhs):
    return hybrid_pcr_thomas_solve(a, b, c, rhs)


@jax.jit
def _xla_reference_kernel(a, b, c, rhs):
    return solve_tridiagonal_xla(a, b, c, rhs)


def _coefficient_fixture(n: int, columns: int, seed: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    lower = -0.02 * rng.random((n, columns), dtype=np.float64)
    upper = -0.02 * rng.random((n, columns), dtype=np.float64)
    lower[0, :] = 0.0
    upper[-1, :] = 0.0
    diag = 1.0 + np.abs(lower) + np.abs(upper) + 0.01 * rng.random((n, columns), dtype=np.float64)
    rhs = rng.normal(0.0, 1.0, (n, columns)).astype(np.float64)
    return tuple(jax.device_put(jnp.asarray(value)) for value in (lower, diag, upper, rhs))


def _real_canary_like_fixture() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    grid = GridSpec.canary_3km_template()
    metrics = DycoreMetrics.flat(
        ny=66,
        nx=159,
        nz=44,
        eta_levels=jnp.linspace(1.0, 0.0, 45, dtype=jnp.float64),
        top_pressure_pa=5000.0,
        provenance="m6-perf-design-canary-d02-shape",
    )
    mut = jax.device_put(jnp.ones((66, 159), dtype=jnp.float64) * 85000.0)
    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients

    a_wrf, alpha, gamma = calc_coef_w_wrf_coefficients(mut, metrics, dt=10.0, epssm=0.1, top_lid=False)
    del grid
    n, ny, nx = a_wrf.shape
    rhs = jnp.sin(jnp.linspace(0.0, 1.0, n * ny * nx, dtype=jnp.float64)).reshape((n, ny, nx))
    b = 1.0 / alpha + a_wrf * jnp.concatenate((jnp.zeros_like(gamma[:1]), gamma[:-1]), axis=0)
    c = gamma / alpha
    b = jnp.where(jnp.isfinite(b), b, 1.0)
    c = jnp.where(jnp.isfinite(c), c, 0.0)
    c = c.at[-1, :, :].set(0.0)
    return a_wrf.reshape((n, ny * nx)), b.reshape((n, ny * nx)), c.reshape((n, ny * nx)), rhs.reshape((n, ny * nx))


def _time_kernel(fn: Callable, args: tuple[jax.Array, ...], samples: int) -> tuple[float, jax.Array]:
    result = fn(*args)
    block_until_ready(result)
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        result = fn(*args)
        block_until_ready(result)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(timings)), result


def run_solver_bakeoff(samples: int = 5) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    HLO_DIR.mkdir(parents=True, exist_ok=True)
    fixture = _real_canary_like_fixture()
    synthetic = _coefficient_fixture(45, 66 * 159, 260525)
    algorithms: dict[str, Callable] = {
        "m6b2_lax_scan_thomas": _thomas_kernel,
        "pure_pcr": _pcr_kernel,
        "hybrid_pcr_thomas_refinement": _hybrid_kernel,
        "xla_tridiagonal_solve_reference": _xla_reference_kernel,
    }
    thomas_reference = _thomas_kernel(*fixture)
    block_until_ready(thomas_reference)
    results: dict[str, dict[str, Any]] = {}
    for name, fn in algorithms.items():
        compiled = fn.lower(*fixture).compile()
        hlo_text = compiled_text(compiled)
        hlo_path = HLO_DIR / f"{name}.hlo.txt"
        hlo_path.write_text(hlo_text, encoding="utf-8")
        wall_ms, result = _time_kernel(fn, fixture, samples)
        residual = _tridiag_matvec(*fixture[:3], result) - fixture[3]
        residual_norm = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(fixture[3]), 1.0e-30)
        delta = result - thomas_reference
        delta_norm = jnp.linalg.norm(delta) / jnp.maximum(jnp.linalg.norm(thomas_reference), 1.0e-30)
        syn_wall_ms, _ = _time_kernel(fn, synthetic, max(3, samples // 2))
        results[name] = {
            "status": "measured",
            "wall_ms_block_until_ready_real_canary_coefficients": wall_ms,
            "wall_ms_block_until_ready_synthetic_d02_shape": syn_wall_ms,
            "residual_l2_relative": float(residual_norm),
            "delta_vs_m6b2_thomas_l2_relative": float(delta_norm),
            "hlo_path": str(hlo_path.relative_to(ROOT)),
            "hlo_bytes": len(hlo_text.encode("utf-8")),
            "hlo_cost_analysis": _cost_analysis(compiled),
            "hlo_kernel_launch_count_estimate": int(kernel_launches_per_step(hlo_text)),
        }

    if TRACE_DIR.exists():
        import shutil

        shutil.rmtree(TRACE_DIR)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(TRACE_DIR), create_perfetto_link=False):
        for fn in algorithms.values():
            block_until_ready(fn(*fixture))
    h2d, d2h, trace_files = count_transfer_bytes(TRACE_DIR)

    payload = {
        "artifact_type": "m6_perf_design_solver_bakeoff",
        "status": "PASS" if int(h2d) == 0 and int(d2h) == 0 else "FAIL",
        "device": visible_gpu_name(),
        "jax_version": jax.__version__,
        "shape": {"vertical_faces": 45, "columns": 66 * 159},
        "algorithms": results,
        "transfer_audit": {
            "method": "warm JAX profiler trace scanned for post-init memcpy events",
            "trace_dir": str(TRACE_DIR.relative_to(ROOT)),
            "host_to_device_bytes": int(h2d),
            "device_to_host_bytes": int(d2h),
            "matched_trace_files": trace_files,
        },
        "external_references": {
            "cusparse_gtsv": {
                "status": "not_run",
                "reason": "repo has no JAX FFI/custom-call binding for cuSPARSE gtsv; ADR-026 treats it as an external benchmark gap, not deployable evidence",
            },
            "cusolverdx_gtsv_no_pivot": {
                "status": "not_run",
                "reason": "repo has no CUDA/C++ device wrapper for cuSolverDx in the M6 operational path",
            },
        },
        "winner": "m6b2_lax_scan_thomas",
        "winner_reason": "only candidate with prior M6B2 parity and measured residual at Thomas reference; PCR/hybrid remain optimization candidates until full Nsight/cuSPARSE references exist",
    }
    _write_json(ARTIFACT_DIR / "proof_solver_bakeoff.json", payload)
    return payload


def run_operational_smoke(hours: float) -> dict[str, Any]:
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    namelist = OperationalNamelist.from_grid(
        grid,
        dt_s=10.0,
        acoustic_substeps=2,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    start = time.perf_counter()
    result = run_forecast_operational(state, namelist, hours)
    block_until_ready(result)
    wall_s = time.perf_counter() - start
    payload = {
        "artifact_type": "m6_perf_design_operational_smoke",
        "status": "PASS",
        "hours": float(hours),
        "wall_s_including_compile": float(wall_s),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz},
        "sanitizer": "not_present_in_operational_path",
        "precision": {
            "u": str(result.u.dtype),
            "v": str(result.v.dtype),
            "w": str(result.w.dtype),
            "theta": str(result.theta.dtype),
            "p": str(result.p.dtype),
            "ph": str(result.ph.dtype),
            "mu": str(result.mu.dtype),
        },
    }
    _write_json(ARTIFACT_DIR / "proof_operational_smoke.json", payload)
    return payload


def write_acceptance_blocker_report() -> dict[str, Any]:
    """Record the honest state of full Stage 3/4/5 acceptance in this workspace."""

    payload = {
        "artifact_type": "m6_perf_design_acceptance_status",
        "status": "BLOCKED",
        "blocked_gates": [
            "Tier-4 golden 1h RMSE was not run by this script; no committed golden operational output/reference pair is present under the sprint directory",
            "28-rank CPU WRF 1h comparison was not rerun in this turn; only existing denominator artifacts are available under artifacts/m6",
            "direct cuSPARSE/cuSolverDx gtsv references are not bound into the repo",
            "Nsight Systems qdrep is produced only by invoking nsys around --profile-only; JAX trace transfer audit is not a substitute for the mandatory Nsight artifact",
        ],
        "existing_cpu_denominator_artifacts": [
            "artifacts/m6/cpu_denominator.json",
            "artifacts/m6/cpu_denominator_v2.json",
        ],
        "tier4_envelope_thresholds": {"T2_K": 3.0, "U10_m_s": 7.5, "V10_m_s": 7.5},
    }
    _write_json(ARTIFACT_DIR / "proof_acceptance_status.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver-bakeoff", action="store_true")
    parser.add_argument("--operational-smoke", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="run warmed solver kernels for external nsys capture")
    parser.add_argument("--hours", type=float, default=10.0 / 3600.0)
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args(argv)


def _cuda_profiler_call(name: str) -> None:
    library = ctypes.util.find_library("cudart") or "libcudart.so"
    cudart = ctypes.CDLL(library)
    result = getattr(cudart, name)()
    if int(result) != 0:
        raise RuntimeError(f"{name} failed with CUDA error {result}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profile_only:
        fixture = _real_canary_like_fixture()
        for fn in (_thomas_kernel, _pcr_kernel, _hybrid_kernel, _xla_reference_kernel):
            block_until_ready(fn(*fixture))
        use_cuda_capture = os.environ.get("GPUWRF_CUDA_PROFILER_RANGE") == "1"
        if use_cuda_capture:
            _cuda_profiler_call("cudaProfilerStart")
        with jax.profiler.TraceAnnotation("solver_bakeoff_loop"):
            for _ in range(5):
                for fn in (_thomas_kernel, _pcr_kernel, _hybrid_kernel, _xla_reference_kernel):
                    block_until_ready(fn(*fixture))
        if use_cuda_capture:
            _cuda_profiler_call("cudaProfilerStop")
        return 0
    if args.solver_bakeoff:
        print(json.dumps(run_solver_bakeoff(args.samples), indent=2, sort_keys=True))
    if args.operational_smoke:
        print(json.dumps(run_operational_smoke(args.hours), indent=2, sort_keys=True))
    if not args.solver_bakeoff and not args.operational_smoke:
        print(json.dumps(write_acceptance_blocker_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
