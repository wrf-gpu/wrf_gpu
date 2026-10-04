"""ADR-031 S2 mixed-perturb fp32 viability ladder.

This is the S2 replacement for the v0.15 viability ladder whose "fp32" lane kept
the acoustic perturbation solve fp64-pinned.  Here the precision axis is:

* fp64      : force_fp64=True, default acoustic mode.
* mixed_s2  : force_fp64=False, acoustic_precision_mode=MIXED_PERTURB_FP32.

The harness reuses the trusted v0.15 real-case tiling/timing method, but records
the compile-plus-first-run wall and an estimated compile wall for each point so
the mixed fp64 island cannot hide a compile-pathology regression.

Run under the GPU lock, e.g.:

  scripts/with_gpu_lock.sh --label gpt-fp32-s2 -- \
    taskset -c 0-3 env PYTHONPATH=src:. JAX_ENABLE_X64=true \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_GPU_ALLOCATOR=cuda_malloc_async \
      python proofs/perf/v016/fp32_s2_mixed_ladder.py --steps 36
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.contracts.precision import (
    AcousticPrecisionMode,
    DEFAULT_DTYPES,
    STATE_FIELD_ORDER,
)
from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
from gpuwrf.runtime.operational_mode import run_forecast_operational

import proofs.perf.v015.viability.fp32_fp64_ab_bench as AB


OUT_JSON = Path("proofs/perf/v016/fp32_s2_mixed_ladder.json")

ANCHOR_RUN_ROOT = "<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable"
ANCHOR_RUN_ID = "run_h36"
ANCHOR_DOMAIN = "d01"
ANCHOR_DT_S = 18.0

DEFAULT_TILE_FACTORS = (
    (1, 1),  # 16,384 columns
    (2, 2),  # 65,536
    (3, 3),  # 147,456
    (3, 4),  # 196,608
    (4, 4),  # 262,144
    (5, 5),  # 409,600
    (6, 6),  # 589,824
)

MIXED_PERT_FIELDS = frozenset(("p_perturbation", "ph_perturbation", "mu_perturbation"))


def _block(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _peak_gib() -> float:
    dev = jax.devices()[0]
    try:
        return float(dev.memory_stats()["peak_bytes_in_use"]) / (1024.0**3)
    except Exception:
        return float("nan")


def _reset_peak() -> bool:
    dev = jax.devices()[0]
    if hasattr(dev, "reset_memory_stats"):
        try:
            dev.reset_memory_stats()
            return True
        except Exception:
            return False
    return False


def _cast_state_mixed_s2(state):
    updates = {}
    for field in STATE_FIELD_ORDER:
        value = getattr(state, field)
        if not hasattr(value, "dtype"):
            continue
        target = jnp.float32 if field in MIXED_PERT_FIELDS else DEFAULT_DTYPES.dtype_for(field)
        if value.dtype != target:
            updates[field] = value.astype(target)
    return state.replace(_cast=False, **updates) if updates else state.replace(_cast=False)


def _fresh_state(base_state, ny0: int, nx0: int, fy: int, fx: int, precision: str):
    st = AB._tile_state(base_state, ny0, nx0, fy, fx)
    st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
    if precision == "fp64":
        st = AB._cast_state_all_fp64(st)
    elif precision == "mixed_s2":
        st = _cast_state_mixed_s2(st)
    else:
        raise ValueError(f"unknown precision {precision!r}")
    _block(st)
    return st


def _namelist(base_nl, ny0: int, nx0: int, fy: int, fx: int, precision: str):
    nl = AB._tile_namelist(base_nl, ny0, nx0, fy, fx)
    if precision == "fp64":
        return dataclasses.replace(
            nl,
            force_fp64=True,
            acoustic_precision_mode=AcousticPrecisionMode.FP64_DEFAULT,
        )
    if precision == "mixed_s2":
        return dataclasses.replace(
            nl,
            force_fp64=False,
            acoustic_precision_mode=AcousticPrecisionMode.MIXED_PERTURB_FP32,
        )
    raise ValueError(f"unknown precision {precision!r}")


def _time_run(state, nl, hours: float) -> float:
    t0 = time.perf_counter()
    out = run_forecast_operational(state, nl, float(hours))
    _block(out)
    return time.perf_counter() - t0, out


def measure(base_state, base_nl, ny0: int, nx0: int, fy: int, fx: int, precision: str, steps: int) -> dict:
    ny, nx = fy * ny0, fx * nx0
    hours = float(steps) * float(base_nl.dt_s) / 3600.0
    nl = _namelist(base_nl, ny0, nx0, fy, fx, precision)
    rec = {
        "precision": precision,
        "ny": ny,
        "nx": nx,
        "nz": int(base_nl.grid.nz),
        "ncol": ny * nx,
        "tile_factor": [fy, fx],
        "n_steps": int(steps),
        "hours": hours,
        "force_fp64": bool(nl.force_fp64),
        "acoustic_precision_mode": str(nl.acoustic_precision_mode),
    }
    try:
        _reset_peak()
        st = _fresh_state(base_state, ny0, nx0, fy, fx, precision)
        rec["input_dtypes"] = {
            "u": str(st.u.dtype),
            "theta": str(st.theta.dtype),
            "p_perturbation": str(st.p_perturbation.dtype),
            "ph_perturbation": str(st.ph_perturbation.dtype),
            "mu_perturbation": str(st.mu_perturbation.dtype),
            "p_total": str(st.p_total.dtype),
            "ph_total": str(st.ph_total.dtype),
            "mu_total": str(st.mu_total.dtype),
        }
        cold_s, out = _time_run(st, nl, hours)
        warm_s = []
        warm_out = out
        for _ in range(2):
            st = _fresh_state(base_state, ny0, nx0, fy, fx, precision)
            wall, warm_out = _time_run(st, nl, hours)
            warm_s.append(wall)
        warm_min = min(warm_s)
        finite = bool(
            jnp.all(jnp.isfinite(warm_out.theta))
            and jnp.all(jnp.isfinite(warm_out.u))
            and jnp.all(jnp.isfinite(warm_out.p_perturbation))
        )
        rec.update(
            {
                "ran_ok": True,
                "oom": False,
                "compile_plus_first_run_s": cold_s,
                "warm_s": warm_s,
                "warm_min_s": warm_min,
                "estimated_compile_s": max(0.0, cold_s - warm_min),
                "ms_per_step": warm_min / float(steps) * 1000.0,
                "ms_per_forecast_hour": warm_min / float(steps) * (3600.0 / float(base_nl.dt_s)) * 1000.0,
                "peak_vram_gib": _peak_gib(),
                "out_finite": finite,
                "out_dtypes": {
                    "u": str(warm_out.u.dtype),
                    "theta": str(warm_out.theta.dtype),
                    "p_perturbation": str(warm_out.p_perturbation.dtype),
                    "ph_perturbation": str(warm_out.ph_perturbation.dtype),
                    "mu_perturbation": str(warm_out.mu_perturbation.dtype),
                    "p_total": str(warm_out.p_total.dtype),
                    "ph_total": str(warm_out.ph_total.dtype),
                    "mu_total": str(warm_out.mu_total.dtype),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        is_oom = (
            "RESOURCE_EXHAUSTED" in str(exc)
            or "out of memory" in str(exc).lower()
            or "OOM" in str(exc)
        )
        rec.update(
            {
                "ran_ok": False,
                "oom": bool(is_oom),
                "error": msg[:700],
                "peak_vram_gib": _peak_gib(),
                "_tb": "".join(traceback.format_exc().splitlines(keepends=True)[-8:]),
            }
        )
    return rec


def _parse_tiles(raw: str) -> list[tuple[int, int]]:
    out = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" in item:
            a, b = item.split("x", 1)
        elif ":" in item:
            a, b = item.split(":", 1)
        else:
            raise ValueError(f"bad tile factor {item!r}; expected FYxFX")
        out.append((int(a), int(b)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--tiles", default=",".join(f"{a}x{b}" for a, b in DEFAULT_TILE_FACTORS))
    ap.add_argument("--precisions", default="fp64,mixed_s2")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--continue-after-oom", action="store_true")
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tiles = _parse_tiles(args.tiles)
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    dev = jax.devices()[0]
    cfg = DailyPipelineConfig(
        run_id=ANCHOR_RUN_ID,
        run_root=ANCHOR_RUN_ROOT,
        domain=ANCHOR_DOMAIN,
        hours=1,
        dt_s=ANCHOR_DT_S,
        acoustic_substeps=10,
    )
    case, _ = _build_real_case(cfg)
    base_nl = case.namelist
    base_state = case.state
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    print(
        f"[base] {ny0}x{nx0}x{nz} dt={base_nl.dt_s}s steps={args.steps} device={dev}",
        flush=True,
    )

    records: list[dict] = []
    dead = {p: False for p in precisions}
    for fy, fx in tiles:
        ny, nx = fy * ny0, fx * nx0
        for precision in precisions:
            if dead.get(precision) and not args.continue_after_oom:
                continue
            print(f"[measure] {precision} {fy}x{fx} -> {ny}x{nx} ncol={ny*nx}", flush=True)
            rec = measure(base_state, base_nl, ny0, nx0, fy, fx, precision, int(args.steps))
            records.append(rec)
            args.out.write_text(json.dumps({"device": str(dev), "records": records}, indent=2) + "\n")
            if rec.get("ran_ok"):
                print(
                    f"  OK {precision} compile+first={rec['compile_plus_first_run_s']:.1f}s "
                    f"est_compile={rec['estimated_compile_s']:.1f}s "
                    f"ms/step={rec['ms_per_step']:.2f} VRAM={rec['peak_vram_gib']:.2f}G "
                    f"finite={rec['out_finite']}",
                    flush=True,
                )
            else:
                print(
                    f"  FAIL {precision} oom={rec.get('oom')} :: {rec.get('error','')[:160]}",
                    flush=True,
                )
                if rec.get("oom"):
                    dead[precision] = True
        if all(dead.get(p, False) for p in precisions) and not args.continue_after_oom:
            break

    by_ncol: dict[int, dict[str, dict]] = {}
    for rec in records:
        if rec.get("ran_ok"):
            by_ncol.setdefault(int(rec["ncol"]), {})[str(rec["precision"])] = rec
    ratios = []
    for ncol in sorted(by_ncol):
        pair = by_ncol[ncol]
        if "fp64" in pair and "mixed_s2" in pair:
            fp64 = pair["fp64"]
            mixed = pair["mixed_s2"]
            ratios.append(
                {
                    "ncol": ncol,
                    "fp64_ms_per_step": fp64["ms_per_step"],
                    "mixed_s2_ms_per_step": mixed["ms_per_step"],
                    "mixed_s2_speedup_over_fp64": fp64["ms_per_step"] / mixed["ms_per_step"],
                    "fp64_peak_gib": fp64["peak_vram_gib"],
                    "mixed_s2_peak_gib": mixed["peak_vram_gib"],
                    "fp64_over_mixed_s2_vram": (
                        fp64["peak_vram_gib"] / mixed["peak_vram_gib"]
                        if mixed["peak_vram_gib"]
                        else None
                    ),
                    "fp64_compile_s": fp64["estimated_compile_s"],
                    "mixed_s2_compile_s": mixed["estimated_compile_s"],
                }
            )

    ok_by_precision = {
        p: [r for r in records if r.get("ran_ok") and r["precision"] == p]
        for p in precisions
    }
    payload = {
        "schema": "ADR031S2MixedPerturbFp32Ladder",
        "scope": "S2 MIXED_PERTURB_FP32 vs fp64 on real tiled Switzerland d01 operational step",
        "device": str(dev),
        "anchor": {
            "run_root": ANCHOR_RUN_ROOT,
            "run_id": ANCHOR_RUN_ID,
            "domain": ANCHOR_DOMAIN,
            "base_grid": {"ny": ny0, "nx": nx0, "nz": nz, "ncol": ny0 * nx0},
            "dt_s": float(base_nl.dt_s),
            "steps": int(args.steps),
        },
        "precision_semantics": {
            "fp64": "force_fp64=True, acoustic_precision_mode=fp64_default, all operational state leaves fp64",
            "mixed_s2": (
                "force_fp64=False, acoustic_precision_mode=mixed_perturb_fp32; "
                "p'/ph'/mu' and acoustic work arrays fp32, compact fp64 stage island for base/c2a/alb/coef_w"
            ),
        },
        "timing_method": (
            "one cold compile+first execution, then best-of-2 warm executions at fixed N steps; "
            "estimated_compile_s = compile_plus_first_run_s - warm_min_s"
        ),
        "config_notes": {
            "run_boundary": False,
            "gwd_opt": 0,
            "use_noahmp": False,
            "tiling": "same spatial tiling helpers as proofs/perf/v015/viability/fp32_fp64_ab_bench.py",
        },
        "records": records,
        "mixed_s2_vs_fp64_ratios": ratios,
        "largest_ok": {
            p: (
                {
                    "ncol": vals[-1]["ncol"],
                    "ny": vals[-1]["ny"],
                    "nx": vals[-1]["nx"],
                    "peak_vram_gib": vals[-1]["peak_vram_gib"],
                    "ms_per_step": vals[-1]["ms_per_step"],
                    "estimated_compile_s": vals[-1]["estimated_compile_s"],
                }
                if vals
                else None
            )
            for p, vals in ok_by_precision.items()
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n============ S2 mixed ladder ============", flush=True)
    print(
        f"{'ncol':>9} {'fp64 ms':>9} {'mixed ms':>9} {'speedup':>8} "
        f"{'fp64 GiB':>9} {'mixed GiB':>9} {'fp64 comp':>10} {'mixed comp':>11}",
        flush=True,
    )
    for row in ratios:
        print(
            f"{row['ncol']:>9d} {row['fp64_ms_per_step']:>9.2f} "
            f"{row['mixed_s2_ms_per_step']:>9.2f} {row['mixed_s2_speedup_over_fp64']:>8.3f} "
            f"{row['fp64_peak_gib']:>9.2f} {row['mixed_s2_peak_gib']:>9.2f} "
            f"{row['fp64_compile_s']:>10.1f} {row['mixed_s2_compile_s']:>11.1f}",
            flush=True,
        )
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
