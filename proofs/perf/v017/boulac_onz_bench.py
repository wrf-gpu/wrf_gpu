#!/usr/bin/env python3
"""v0.17 MYNN BouLac ONZ benchmark.

Run one mode per clean process.  The mode is resolved from environment at
``gpuwrf.physics.mynn_pbl`` import time:

  default dense:
    GPUWRF_MYNN_BOULAC_ONZ=0 GPUWRF_MYNN_BOULAC_CHUNKED=0

  production ONZ:
    GPUWRF_MYNN_BOULAC_ONZ=1 GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN=0

  legacy scan pathology probe:
    GPUWRF_MYNN_BOULAC_ONZ=1 GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN=1

Every GPU invocation must be wrapped in ``scripts/with_gpu_lock.sh``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import run_forecast_operational
from gpuwrf.physics import mynn_pbl as MYNN

import proofs.perf.v015.viability.fp32_fp64_ab_bench as AB


SWISS_RUN_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
SWISS_RUN_ID = "run_h36"
SWISS_DOMAIN = "d01"
SWISS_DT_S = 18.0

CANARY_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
CANARY_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
CANARY_DOMAIN = "d03"
CANARY_DT_S = 10.0


def _mode() -> str:
    if MYNN._MYNN_BOULAC_ONZ:
        if MYNN._MYNN_BOULAC_ONZ_LEGACY_SCAN:
            return "onz_legacy_scan"
        return "onz_chunk1_production"
    if MYNN._MYNN_BOULAC_CHUNKED:
        return f"chunked{MYNN._MYNN_BOULAC_CHUNK}"
    return "dense"


def _case_config(case_name: str) -> DailyPipelineConfig:
    if case_name == "swiss-1km":
        return DailyPipelineConfig(
            run_id=SWISS_RUN_ID,
            run_root=SWISS_RUN_ROOT,
            domain=SWISS_DOMAIN,
            hours=1,
            dt_s=SWISS_DT_S,
            acoustic_substeps=10,
        )
    if case_name == "canary-d03":
        return DailyPipelineConfig(
            run_id=CANARY_RUN_ID,
            run_root=CANARY_RUN_ROOT,
            domain=CANARY_DOMAIN,
            hours=1,
            dt_s=CANARY_DT_S,
            acoustic_substeps=10,
        )
    raise ValueError(f"unknown case {case_name!r}")


def _parse_tiles(raw: str) -> list[tuple[int, int]]:
    out = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        fy, fx = item.split("x", 1)
        out.append((int(fy), int(fx)))
    if not out:
        raise ValueError("need at least one tile factor")
    return out


def _block(tree: Any) -> None:
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _peak_gib() -> float:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**3)
    except Exception:
        return float("nan")


def _reset_peak() -> bool:
    dev = jax.devices()[0]
    if not hasattr(dev, "reset_memory_stats"):
        return False
    try:
        dev.reset_memory_stats()
        return True
    except Exception:
        return False


def _fresh_state(base_state, ny0: int, nx0: int, fy: int, fx: int):
    st = AB._tile_state(base_state, ny0, nx0, fy, fx)
    st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
    st = AB._cast_state_all_fp64(st)
    _block(st)
    return st


def _namelist(base_nl, ny0: int, nx0: int, fy: int, fx: int):
    nl = AB._tile_namelist(base_nl, ny0, nx0, fy, fx)
    return dataclasses.replace(nl, force_fp64=True)


def _finite_core_fields(state) -> bool:
    fields = ("theta", "u", "v", "p_perturbation", "tke")
    flags = []
    for name in fields:
        value = getattr(state, name, None)
        if value is not None:
            flags.append(jnp.all(jnp.isfinite(value)))
    return bool(jnp.stack(flags).all()) if flags else True


def _time_run(state, nl, hours: float):
    t0 = time.perf_counter()
    out = run_forecast_operational(state, nl, float(hours))
    _block(out)
    return time.perf_counter() - t0, out


def _measured_vs_projected(case_name: str, mode: str, ncol: int, rec: dict[str, Any]) -> dict[str, Any]:
    if case_name == "swiss-1km" and ncol == 147456:
        if mode == "dense":
            return {
                "projection": "dense fp64 expected to OOM at 147456 columns on the BouLac (B,nz,nz) allocation",
                "measured": "OOM" if rec.get("oom") else "FIT",
                "meets_projection": bool(rec.get("oom")),
                "prior_reference": "v016 Opus dense OOM attempted 18.80 GiB single allocation",
            }
        if mode == "onz_chunk1_production":
            return {
                "projection": "ONZ production path should fit 147456 columns with finite output near prior 18.25-21.31 GiB band",
                "measured": "FIT" if rec.get("ran_ok") and rec.get("out_finite") else "FAIL",
                "meets_projection": bool(rec.get("ran_ok") and rec.get("out_finite")),
                "prior_reference": "v016 Opus chunk=1 clean-process peak 18.25 GiB finite",
            }
        if mode == "onz_legacy_scan":
            return {
                "projection": "legacy pure scan expected to remain pathological in full jit at 147456 columns",
                "measured": "OOM" if rec.get("oom") else "FIT",
                "meets_projection": bool(rec.get("oom")),
                "prior_reference": "v016 Opus command-buffer OOM with 20 alive graphs",
            }
    return {
        "projection": "no hard fit/oom projection for this smaller Canary/control point",
        "measured": "FIT" if rec.get("ran_ok") else "FAIL",
        "meets_projection": bool(rec.get("ran_ok")),
    }


def measure(base_state, base_nl, ny0: int, nx0: int, fy: int, fx: int, steps: int, warm_reps: int,
            case_name: str) -> dict[str, Any]:
    ny, nx = fy * ny0, fx * nx0
    hours = float(steps) * float(base_nl.dt_s) / 3600.0
    nl = _namelist(base_nl, ny0, nx0, fy, fx)
    mode = _mode()
    rec: dict[str, Any] = {
        "case": case_name,
        "boulac_mode": mode,
        "ny": ny,
        "nx": nx,
        "nz": int(base_nl.grid.nz),
        "ncol": int(ny * nx),
        "tile_factor": [int(fy), int(fx)],
        "n_steps": int(steps),
        "hours": hours,
        "force_fp64": True,
        "env": {
            "GPUWRF_MYNN_BOULAC_ONZ": os.environ.get("GPUWRF_MYNN_BOULAC_ONZ", "<unset>"),
            "GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN": os.environ.get("GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN", "<unset>"),
            "GPUWRF_MYNN_BOULAC_CHUNKED": os.environ.get("GPUWRF_MYNN_BOULAC_CHUNKED", "<unset>"),
            "GPUWRF_MYNN_BOULAC_CHUNK": os.environ.get("GPUWRF_MYNN_BOULAC_CHUNK", "<unset>"),
        },
    }
    try:
        _reset_peak()
        st = _fresh_state(base_state, ny0, nx0, fy, fx)
        cold_t0 = time.perf_counter()
        cold_s, out = _time_run(st, nl, hours)
        cold_wall_s = time.perf_counter() - cold_t0

        warm_s = []
        warm_out = out
        for _ in range(int(warm_reps)):
            st = _fresh_state(base_state, ny0, nx0, fy, fx)
            wall_s, warm_out = _time_run(st, nl, hours)
            warm_s.append(wall_s)

        warm_min_s = min(warm_s) if warm_s else cold_s
        rec.update({
            "ran_ok": True,
            "oom": False,
            "compile_plus_first_run_s": cold_s,
            "cold_ms_per_step": cold_s / float(steps) * 1000.0,
            "warm_s": warm_s,
            "warm_min_s": warm_min_s,
            "warm_ms_per_step": warm_min_s / float(steps) * 1000.0,
            "estimated_compile_s": max(0.0, cold_s - warm_min_s),
            "peak_vram_gib": _peak_gib(),
            "out_finite": _finite_core_fields(warm_out),
            "wall_clock_cold_s": cold_wall_s,
        })
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        is_oom = (
            "RESOURCE_EXHAUSTED" in str(exc)
            or "out of memory" in str(exc).lower()
            or "OOM" in str(exc)
        )
        rec.update({
            "ran_ok": False,
            "oom": bool(is_oom),
            "error": msg[:1200],
            "peak_vram_gib": _peak_gib(),
            "_tb": "".join(traceback.format_exc().splitlines(keepends=True)[-12:]),
        })
    rec["measured_vs_projected"] = _measured_vs_projected(case_name, mode, int(ny * nx), rec)
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("swiss-1km", "canary-d03"), default="swiss-1km")
    parser.add_argument("--tiles", default=None, help="comma-separated tile factors, e.g. 3x3")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warm-reps", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--continue-after-oom", action="store_true")
    args = parser.parse_args(argv)

    cfg = _case_config(args.case)
    tiles = _parse_tiles(args.tiles or ("3x3" if args.case == "swiss-1km" else "1x1"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    dev = jax.devices()[0]
    case, run_dir = _build_real_case(cfg)
    base_nl = case.namelist
    base_state = case.state
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    mode = _mode()

    records = []
    print(
        f"[base] case={args.case} mode={mode} grid={ny0}x{nx0}x{nz} "
        f"dt={base_nl.dt_s}s run_dir={run_dir} device={dev}",
        flush=True,
    )
    for fy, fx in tiles:
        print(f"[measure] {args.case} {mode} tile={fy}x{fx} ncol={fy*ny0*fx*nx0}", flush=True)
        rec = measure(base_state, base_nl, ny0, nx0, fy, fx, args.steps, args.warm_reps, args.case)
        records.append(rec)
        partial = {
            "schema": "gpuwrf.v017.boulac_onz_bench.v1",
            "case": args.case,
            "mode": mode,
            "device": str(dev),
            "records": records,
        }
        args.out.write_text(json.dumps(partial, indent=2) + "\n")
        if rec.get("ran_ok"):
            print(
                f"  OK warm={rec['warm_ms_per_step']:.2f} ms/step "
                f"cold={rec['cold_ms_per_step']:.2f} ms/step "
                f"compile~{rec['estimated_compile_s']:.1f}s "
                f"vram={rec['peak_vram_gib']:.2f} GiB finite={rec['out_finite']}",
                flush=True,
            )
        else:
            print(f"  FAIL oom={rec.get('oom')} error={rec.get('error', '')[:240]}", flush=True)
            if rec.get("oom") and not args.continue_after_oom:
                break

    ok = [r for r in records if r.get("ran_ok")]
    payload = {
        "schema": "gpuwrf.v017.boulac_onz_bench.v1",
        "scope": "MYNN BouLac dense vs production ONZ memory-shape benchmark",
        "case": args.case,
        "mode": mode,
        "device": str(dev),
        "run_dir": str(run_dir),
        "base_grid": {"ny": ny0, "nx": nx0, "nz": nz, "ncol": ny0 * nx0},
        "timing_method": "one cold compile+first execution, then best-of-N warm executions; per-step wall = wall / n_steps",
        "records": records,
        "largest_ok": (
            {
                "ncol": ok[-1]["ncol"],
                "ny": ok[-1]["ny"],
                "nx": ok[-1]["nx"],
                "peak_vram_gib": ok[-1]["peak_vram_gib"],
                "warm_ms_per_step": ok[-1]["warm_ms_per_step"],
                "cold_ms_per_step": ok[-1]["cold_ms_per_step"],
                "estimated_compile_s": ok[-1]["estimated_compile_s"],
            }
            if ok else None
        ),
        "flags": {
            "GPUWRF_MYNN_BOULAC_ONZ": bool(MYNN._MYNN_BOULAC_ONZ),
            "GPUWRF_MYNN_BOULAC_ONZ_LEGACY_SCAN": bool(MYNN._MYNN_BOULAC_ONZ_LEGACY_SCAN),
            "GPUWRF_MYNN_BOULAC_CHUNKED": bool(MYNN._MYNN_BOULAC_CHUNKED),
            "GPUWRF_MYNN_BOULAC_CHUNK": int(MYNN._MYNN_BOULAC_CHUNK),
            "GPUWRF_MYNN_BOULAC_FP32": bool(MYNN._MYNN_BOULAC_FP32),
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
