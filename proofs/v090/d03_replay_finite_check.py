#!/usr/bin/env python
"""v0.9.0 d03 1km replay FINITE check — qke-fix follow-up.

Confirms the d02-replay hour-1 stability fix carries to the 1km Tenerife d03 case:
the SHIPPED ``scripts/d03_replay.build_l3_d03_daily_case`` already routes the d03
forecast through the validated operational stability namelist (top_lid, epssm=0.5,
w_damping, damp_opt=3, zdamp, dampcoef, diff_6th, use_flux_advection, force_fp64)
AND inherits the WRF-faithful MYNN qke cold-start seed from
``d02_replay.build_replay_case`` (``_wrf_mynn_coldstart_qke``).

This harness drives that exact case in fixed ``step_h``-hour increments via the
production ``run_forecast_operational`` (one compiled program reused per increment;
peak memory bounded to one increment) for a few hours and records qke + the
dynamics maxima staying bounded + physical.  If any OTHER instability beyond the
qke/namelist issue appears (e.g. the historical d03 nested-ph geopotential pump),
the first-non-finite field + step is recorded precisely.

GPU (cuda); ONE GPU job at a time (lock held by the calling lane).  CPUs pinned to
0-3 (cores 4-31 are a live CPU-WRF backfill).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

from d03_replay import (  # noqa: E402
    D03_ACOUSTIC_SUBSTEPS,
    D03_DT_S,
    D03_RADIATION_CADENCE_STEPS,
    L3_RUN_ROOT,
    build_l3_d03_daily_case,
)
from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    DailyPipelineConfig,
    finite_summary,
    resolve_run_dir,
)
from gpuwrf.runtime.operational_mode import run_forecast_operational_single_scan  # noqa: E402


def _signature(state):
    summ = finite_summary(state)
    fields = summ["fields"]
    dyn = {}
    for k in ("u", "v", "w", "theta", "mu", "ph", "p", "qke", "qv"):
        if k in fields:
            f = fields[k]
            dyn[k] = {"fin": f["finite"], "max": f["max"], "min": f["min"]}
    nonfinite = {k: v["nonfinite_count"] for k, v in fields.items() if not v["finite"]}
    return {"all_finite": summ["all_finite"], "dyn": dyn, "nonfinite_fields": nonfinite}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="20260521_18z_l3_24h_20260522T133443Z")
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--hours", type=float, default=3.0,
                    help="forecast hours for the finite check (a few hours)")
    ap.add_argument("--dt-s", type=float, default=D03_DT_S)
    ap.add_argument("--step-h", type=float, default=0.25,
                    help="fixed forecast increment in hours (one compiled program reused; "
                         "0.25h @ dt=3s = 300 steps, memory-bounded)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "proofs" / "v090" / "d03_replay_finite_check.json")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ["JAX_PLATFORMS"] = args.platform
    import jax
    devices = [str(d) for d in jax.devices()]

    run_dir = resolve_run_dir(args.run_id, L3_RUN_ROOT)
    cfg = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(round(args.hours)),
        output_dir=Path("/tmp/v090_d03finite"),
        proof_dir=Path("/tmp/v090_d03finite"),
        run_root=L3_RUN_ROOT,
        score=False,
        domain="d03",
        dt_s=float(args.dt_s),
        acoustic_substeps=D03_ACOUSTIC_SUBSTEPS,
        radiation_cadence_steps=D03_RADIATION_CADENCE_STEPS,
    )
    case, _ = build_l3_d03_daily_case(cfg)
    nl = case.namelist
    state = case.state
    grid = case.grid
    grid_shape = [int(grid.nz), int(grid.ny), int(grid.nx)]
    qke_coldstart = case.metadata.get("qke_coldstart", {})
    qke_t0 = float(jax.numpy.max(jax.numpy.asarray(state.qke)))
    flags = {k: (bool(getattr(nl, k)) if isinstance(getattr(nl, k), bool) else getattr(nl, k))
             for k in ("top_lid", "epssm", "w_damping", "damp_opt", "dampcoef", "zdamp",
                       "diff_6th_opt", "diff_6th_factor", "use_flux_advection", "force_fp64")}
    print(f"d03 grid(zyx)={grid_shape} dt={args.dt_s} qke_t0_max={qke_t0:.4g} "
          f"qke_seeded={qke_coldstart.get('seeded')} flags={flags}")

    # Advance in FIXED step_h-hour increments via the production
    # run_forecast_operational (same entry daily_pipeline drives).  Every advance
    # has the SAME static hours=step_h so it compiles ONCE and is reused for all
    # subsequent increments (no per-checkpoint recompile, bounded peak memory).
    checkpoints = []
    h = args.step_h
    while h <= args.hours + 1e-9:
        checkpoints.append(round(h, 6))
        h += args.step_h
    if not checkpoints or abs(checkpoints[-1] - args.hours) > 1e-9:
        checkpoints.append(round(args.hours, 6))

    trace = []
    first_blow = None
    done_h = 0.0
    t_start = time.perf_counter()
    err = None
    aborted_oom = False
    while done_h < args.hours - 1e-9:
        adv = min(args.step_h, args.hours - done_h)
        try:
            # single_scan is @jax.jit(static hours, donate state): first adv call
            # compiles once, subsequent same-adv calls hit the jit cache.
            state = run_forecast_operational_single_scan(state, nl, float(adv))
            jax.block_until_ready(state)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            err = f"{type(exc).__name__}: {exc}"
            aborted_oom = "RESOURCE_EXHAUSTED" in str(exc)
            break
        done_h = round(done_h + adv, 6)
        sig = _signature(state)
        step = int(round(done_h * 3600.0 / args.dt_s))
        row = {"hours": round(done_h, 3), "step": step, "all_finite": sig["all_finite"],
               "dyn": sig["dyn"], "nonfinite_fields": sig["nonfinite_fields"]}
        trace.append(row)
        q = sig["dyn"].get("qke", {})
        w = sig["dyn"].get("w", {})
        th = sig["dyn"].get("theta", {})
        mu = sig["dyn"].get("mu", {})
        print(f"  {done_h:>5.2f}h (step {step:>5d}) finite={sig['all_finite']} "
              f"qke.max={q.get('max')} w.max={w.get('max')} theta.max={th.get('max')} "
              f"mu.max={mu.get('max')} nonfin={list(sig['nonfinite_fields'].keys())[:6]}")
        if not sig["all_finite"] and first_blow is None:
            first_blow = done_h
            break

    finite_trace = [t for t in trace if t["all_finite"]]
    last = finite_trace[-1] if finite_trace else {}
    survived_h = float(last.get("hours", 0.0))

    # d03-specific instability classification (honest): if it blew, which field
    # first, and is it the qke/namelist class (cured for d02) or a DISTINCT d03
    # mechanism (e.g. the nested-ph geopotential pump, force_geopotential=False by
    # default here)?
    d03_specific = None
    if first_blow is not None:
        blow_row = next((t for t in trace if not t["all_finite"]), {})
        nf = blow_row.get("nonfinite_fields", {})
        first_fields = list(nf.keys())
        d03_specific = {
            "first_nonfinite_hours": first_blow,
            "first_nonfinite_fields": first_fields,
            "note": (
                "qke first non-finite => same qke-closure class as d02 (should be cured here)"
                if "qke" in first_fields else
                "dynamics/other field first non-finite => potential d03-specific mechanism "
                "(inspect nested-ph forcing / 1km CFL), NOT the d02 qke/namelist issue"
            ),
        }

    verdict = (
        "D03_FINITE" if (first_blow is None and not err and survived_h >= min(args.hours, 1.0) - 1e-9)
        else "D03_OOM_INCONCLUSIVE" if aborted_oom
        else "D03_NONFINITE" if first_blow is not None
        else "D03_ERROR"
    )
    payload = {
        "schema": "V090D03ReplayFiniteCheck",
        "generated": datetime.now(timezone.utc).isoformat(),
        "branch": "worker/opus/v090-qkefix-followup",
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "platform": args.platform,
        "devices": devices,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "grid_mass_shape_zyx": grid_shape,
        "dt_s": float(args.dt_s),
        "acoustic_substeps": D03_ACOUSTIC_SUBSTEPS,
        "radiation_cadence_steps": D03_RADIATION_CADENCE_STEPS,
        "step_h": args.step_h,
        "runner": "run_forecast_operational in fixed step_h increments (one compiled program reused; memory-bounded)",
        "fix_under_test": "shipped build_l3_d03_daily_case = validated stability namelist (already present) + WRF-faithful MYNN qke cold-start seed (inherited from build_replay_case)",
        "wrf_seed_ref": "phys/module_bl_mynnedmf.F:618-691 (mym_initialize INITIALIZE_QKE)",
        "qke_coldstart": qke_coldstart,
        "qke_t0_max": qke_t0,
        "namelist_flags": flags,
        "checkpoints_hours": checkpoints,
        "trace": trace,
        "first_nonfinite_hours": first_blow,
        "survived_hours": survived_h,
        "survived_steps": int(last.get("step", 0)),
        "error": err,
        "aborted_oom": aborted_oom,
        "d03_specific_instability": d03_specific,
        "elapsed_s": time.perf_counter() - t_start,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nverdict: {verdict}  survived_hours={survived_h}  d03_specific={d03_specific}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
