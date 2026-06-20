"""Wave-B measurement-only timing probe for v0.10.0 scope.

Runs one L2 d02 operational variant in a fresh process and reports compile time,
warm cache-hit samples, and a short finiteness summary.  The probe intentionally
does not edit model code; variants are namelist/config toggles only.

Examples:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
    python proofs/v0100/wave_b_timing_probe.py --variant full --force-fp64 true \
      --steps 240 --out proofs/v0100/wave_b_timing_full_fp64.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry


L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"


def _parse_bool(text: str) -> bool:
    lowered = str(text).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {text!r}")


def _peak_mb() -> float | None:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return None


def _variant_namelist(nl: Any, variant: str):
    if variant == "full":
        return nl
    if variant == "no_thompson":
        return dataclasses.replace(nl, mp_physics=0)
    if variant == "no_pbl":
        return dataclasses.replace(nl, bl_pbl_physics=0)
    if variant == "no_boundary":
        return dataclasses.replace(nl, run_boundary=False)
    if variant == "dycore_only":
        return dataclasses.replace(nl, run_physics=False, run_boundary=False)
    raise ValueError(f"unknown variant {variant!r}")


def _field_stat(state, name: str) -> dict[str, Any]:
    arr = np.asarray(jax.device_get(getattr(state, name)), dtype=np.float64)
    return {
        "finite": bool(np.isfinite(arr).all()),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full", "no_thompson", "no_pbl", "no_boundary", "dycore_only"), default="full")
    parser.add_argument("--force-fp64", type=_parse_bool, default=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--cadence", type=int, default=1000000, help="large default keeps the timed segment non-radiation")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--discard-first-sample", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not [d for d in jax.devices() if d.platform == "gpu"]:
        raise RuntimeError("No JAX GPU backend visible; refusing to produce timing proof")
    if int(args.steps) < 200:
        raise ValueError("--steps must be >=200 for Wave-B timing")

    acoustic_unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))
    cfg = DailyPipelineConfig(
        hours=1,
        dt_s=10.0,
        acoustic_substeps=10,
        run_id=L2_RUN_ID,
        run_root=paths.wrf_l2_root(),
        domain="d02",
        radiation_cadence_steps=int(args.cadence),
    )
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=False,
        force_fp64=bool(args.force_fp64),
        radiation_cadence_steps=int(args.cadence),
        time_utc=case.run_start,
    )
    nl = _variant_namelist(nl, args.variant)
    state0 = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    carry = initial_operational_carry(state0)

    print(
        f"variant={args.variant} force_fp64={bool(nl.force_fp64)} "
        f"steps={args.steps} cadence={args.cadence} unroll={acoustic_unroll} "
        f"device={jax.devices()[0]}",
        flush=True,
    )

    t0 = time.perf_counter()
    carry = _advance_chunk(
        carry,
        nl,
        jnp.asarray(1, dtype=jnp.int32),
        n_steps=int(args.steps),
        cadence=int(args.cadence),
    )
    jax.block_until_ready(carry.state.theta)
    compile_s = time.perf_counter() - t0
    peak_after_compile = _peak_mb()

    samples = []
    for repeat in range(int(args.repeats)):
        start = jnp.asarray(1 + (repeat + 1) * int(args.steps), dtype=jnp.int32)
        t0 = time.perf_counter()
        carry_next = _advance_chunk(
            carry,
            nl,
            start,
            n_steps=int(args.steps),
            cadence=int(args.cadence),
        )
        jax.block_until_ready(carry_next.state.theta)
        ms = (time.perf_counter() - t0) * 1000.0 / float(args.steps)
        samples.append(float(ms))
        print(f"  sample {repeat}: {ms:.3f} ms/step", flush=True)
        carry = carry_next

    used_samples = samples[1:] if bool(args.discard_first_sample) and len(samples) > 1 else samples
    final_ranges = {name: _field_stat(carry.state, name) for name in ("u", "v", "w", "theta", "qv", "p_total", "ph_total", "mu_total")}
    all_finite = all(item["finite"] for item in final_ranges.values())
    physical = (
        all_finite
        and abs(final_ranges["u"]["min"]) < 150.0 and abs(final_ranges["u"]["max"]) < 150.0
        and abs(final_ranges["v"]["min"]) < 150.0 and abs(final_ranges["v"]["max"]) < 150.0
        and abs(final_ranges["w"]["min"]) < 60.0 and abs(final_ranges["w"]["max"]) < 60.0
        and final_ranges["theta"]["min"] > 150.0 and final_ranges["theta"]["max"] < 600.0
        and final_ranges["mu_total"]["min"] > 0.0 and final_ranges["p_total"]["min"] > 0.0
        and final_ranges["qv"]["min"] >= -1.0e-6 and final_ranges["qv"]["max"] < 0.1
    )
    out = {
        "schema": "V0100WaveBTimingProbe",
        "schema_version": 1,
        "status": "PASS" if physical else "FAIL",
        "run_id": L2_RUN_ID,
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
        "variant": args.variant,
        "steps": int(args.steps),
        "cadence": int(args.cadence),
        "repeats": int(args.repeats),
        "discard_first_sample": bool(args.discard_first_sample),
        "config": {
            "force_fp64": bool(nl.force_fp64),
            "run_physics": bool(nl.run_physics),
            "run_boundary": bool(nl.run_boundary),
            "disable_guards": bool(nl.disable_guards),
            "mp_physics": int(nl.mp_physics),
            "bl_pbl_physics": int(nl.bl_pbl_physics),
            "sf_sfclay_physics": int(nl.sf_sfclay_physics),
            "use_noahmp": bool(nl.use_noahmp),
            "dt_s": float(nl.dt_s),
            "acoustic_substeps": int(nl.acoustic_substeps),
            "acoustic_unroll": acoustic_unroll,
            "GPUWRF_THOMPSON_NSED": os.environ.get("GPUWRF_THOMPSON_NSED"),
            "GPUWRF_THOMPSON_SED_UNROLL": os.environ.get("GPUWRF_THOMPSON_SED_UNROLL"),
            "GPUWRF_THOMPSON_FP32": os.environ.get("GPUWRF_THOMPSON_FP32"),
        },
        "compile_s": float(compile_s),
        "warmed_per_step_ms": float(min(used_samples)),
        "warmed_per_step_ms_samples": samples,
        "warmed_per_step_ms_samples_used": used_samples,
        "peak_gpu_mem_mb_after_compile": peak_after_compile,
        "peak_gpu_mem_mb_final": _peak_mb(),
        "final_state_ranges": final_ranges,
        "all_finite": bool(all_finite),
        "physically_plausible_short_run": bool(physical),
        "methodology_note": (
            "Fresh process; compile/warm call outside samples; warmed_per_step_ms is "
            "the min of cache-hit repeat calls. Default cadence is deliberately larger "
            "than the segment so the measured step is the non-radiation coupled step "
            "used by the Wave-A ~75 ms gate, with Thompson/MYNN/surface/boundary active."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("status", "compile_s", "warmed_per_step_ms", "all_finite", "physically_plausible_short_run")}, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
