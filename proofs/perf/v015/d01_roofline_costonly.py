"""v0.15 roofline on the REAL Switzerland d01 128x128x44 case (the 173 ms/step problem grid).

Settles compute-vs-bandwidth bound for the v0.14 dominant per-step kernel on the
ACTUAL problem grid + commit (not the 2026-05-30 d02 461k-cell analysis). Uses XLA
cost_analysis (LOWER+COMPILE only, no device execution) for FLOPs/bytes -> low VRAM,
GPU-lock-safe. Warmed per-step WALL is measured separately (d01_warm_step.py).

Run (GPU lock required because .compile() touches the device):
  scripts/with_gpu_lock.sh --label perf-fix -- \
    taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python proofs/perf/v015/d01_roofline_costonly.py
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax

import gpuwrf.contracts.state as _stmod
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational_single_scan

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT = Path("proofs/perf/v015")

_orig_asarray = _stmod.jnp.asarray


def _safe_asarray(x, dtype=None, **kw):
    try:
        return _orig_asarray(x, dtype=dtype, **kw) if dtype is not None else _orig_asarray(x, **kw)
    except (TypeError, ValueError):
        return x


_stmod.jnp.asarray = _safe_asarray

# RTX 5090 GB202 peak specs (consumer Blackwell, fp64 = fp32/64).
PEAK = {
    "fp32_tflops": 2 * 21760 * 2.41e9 / 1e12,          # 104.9
    "fp64_tflops": (2 * 21760 * 2.41e9 / 1e12) / 64.0,  # 1.64
    "hbm_tbytes_s": 512 / 8 * 28e9 / 1e12,              # 1.792
}


def _cost(state, nl, hours):
    comp = run_forecast_operational_single_scan.lower(state, nl, float(hours)).compile()
    ca = comp.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0]
    return (
        float(ca.get("flops", float("nan"))),
        float(ca.get("bytes accessed", ca.get("bytes_accessed", float("nan")))),
        float(ca.get("transcendentals", 0.0)),
    )


def roof(flops, byts, dt_s, label):
    # cost_analysis returns the scan BODY cost ONCE (constant in trip count) -> per step.
    hbm_floor_ms = (byts / (PEAK["hbm_tbytes_s"] * 1e12)) * 1000.0 if byts > 0 else float("nan")
    fp64_floor_ms = (flops / (PEAK["fp64_tflops"] * 1e12)) * 1000.0 if flops > 0 else float("nan")
    fp32_floor_ms = (flops / (PEAK["fp32_tflops"] * 1e12)) * 1000.0 if flops > 0 else float("nan")
    return {
        "label": label,
        "dt_s": dt_s,
        "gflops_per_step": flops / 1e9,
        "gbytes_per_step": byts / 1e9,
        "arithmetic_intensity_flop_per_byte": flops / byts if byts else float("nan"),
        "hbm_bound_floor_ms": hbm_floor_ms,
        "fp64_compute_floor_ms": fp64_floor_ms,
        "fp32_compute_floor_ms": fp32_floor_ms,
        "ridge_AI_fp64": PEAK["fp64_tflops"] / PEAK["hbm_tbytes_s"],
        "ridge_AI_fp32": PEAK["fp32_tflops"] / PEAK["hbm_tbytes_s"],
        "bound_verdict": (
            "fp64-compute-bound" if flops / byts > PEAK["fp64_tflops"] / PEAK["hbm_tbytes_s"]
            else "bandwidth-or-launch-bound (below fp64 ridge)"
        ),
    }


def main() -> int:
    cfg = DailyPipelineConfig(run_id="run_h36", run_root=PROBE, domain="d01", hours=1)
    case, run_dir = _build_real_case(cfg)
    nl = case.namelist
    ny, nx, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)

    base_nl = dataclasses.replace(
        nl, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=180,
    )
    dyn_nl = dataclasses.replace(base_nl, run_physics=False, run_boundary=False)

    dt_s = float(nl.dt_s)
    # one step worth of hours so the single-scan body is the per-step body
    hours_1step = dt_s / 3600.0

    out = {
        "scope": "v0.15 roofline on REAL Switzerland d01 (the 173 ms/step problem grid)",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": ny, "nx": nx, "nz": nz, "mass_cells": ny * nx * nz},
        "namelist": {
            "dt_s": dt_s,
            "acoustic_substeps": int(getattr(nl, "acoustic_substeps", -1)) if hasattr(nl, "acoustic_substeps") else None,
            "force_fp64": bool(getattr(nl, "force_fp64", False)),
            "epssm": float(getattr(nl, "epssm", float("nan"))),
            "top_lid": bool(getattr(nl, "top_lid", False)),
        },
        "peak_specs": PEAK,
        "method": "single-scan cost_analysis body-once (constant in trip count) = per-step FLOPs/bytes",
    }

    # 1-step (the body) at hours = dt; verify body-once by also doing 2 steps and diffing.
    cf1, cb1, ct1 = _cost(_build_real_case(cfg)[0].state, base_nl, hours_1step)
    cf2, cb2, ct2 = _cost(_build_real_case(cfg)[0].state, base_nl, 2 * hours_1step)
    out["coupled_step"] = {
        "flops_1step": cf1, "bytes_1step": cb1, "transcendentals_1step": ct1,
        "flops_2step": cf2, "bytes_2step": cb2,
        "body_once_check": {"flops_equal": abs(cf1 - cf2) < 1.0, "bytes_equal": abs(cb1 - cb2) < 1.0},
        "roofline": roof(cf1, cb1, dt_s, "coupled (phys+bdy) non-radiation step, d01 128x128x44"),
    }

    yf1, yb1, yt1 = _cost(_build_real_case(cfg)[0].state, dyn_nl, hours_1step)
    yf2, yb2, yt2 = _cost(_build_real_case(cfg)[0].state, dyn_nl, 2 * hours_1step)
    out["dycore_only_step"] = {
        "flops_1step": yf1, "bytes_1step": yb1, "transcendentals_1step": yt1,
        "flops_2step": yf2, "bytes_2step": yb2,
        "body_once_check": {"flops_equal": abs(yf1 - yf2) < 1.0, "bytes_equal": abs(yb1 - yb2) < 1.0},
        "roofline": roof(yf1, yb1, dt_s, "dycore-only step, d01 128x128x44"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "d01_roofline_costonly.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "grid": out["grid"], "namelist": out["namelist"],
        "coupled": out["coupled_step"]["roofline"],
        "dycore": out["dycore_only_step"]["roofline"],
        "coupled_body_once": out["coupled_step"]["body_once_check"],
        "dycore_body_once": out["dycore_only_step"]["body_once_check"],
    }, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
