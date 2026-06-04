#!/usr/bin/env python
"""v0.9.0 d02-replay qke-fix FOLLOW-UP — memory-leaner 3h+ re-verify.

The decisive qke-fix verify (proofs/v090/d02replay_qke_fix_verify.json) proved the
20260521 d02 replay is FINITE + physical through hour 1 (300 steps) under the
shipped fix (validated stability namelist + WRF-faithful MYNN qke cold-start seed),
but its 3h confirmation OOM'd: that harness drove ``run_forecast_operational`` with
a single large incremental fp64 advance (600-step jump = 14.6 GiB single segment),
which is a VERIFY-HARNESS memory artifact, NOT a model blow-up.

This follow-up re-runs the SAME shipped d02 case (``build_l2_daily_case`` -> the
validated _build_real_case stability namelist + the build_replay_case qke seed) but
in a MEMORY-BOUNDED way:

  * Fixed ``step_h``-hour increments via the production ``run_forecast_operational``
    (the SAME entry daily_pipeline drives per hour) -- every advance has the same
    static ``hours=step_h`` so it compiles ONCE and is reused, with device scratch
    freed (``block_until_ready``) between increments.  step_h is sized so each
    advance's step count stays well under the 600-step (14.6 GiB) single advance
    that OOM'd: at dt=12 s, step_h=0.5 -> 150 steps.  Peak GPU memory is bounded to
    one increment regardless of forecast length.
  * The OPERATIONAL gated-fp32 precision matrix (ADR-007; ``force_fp64=False`` ->
    ``_enforce_operational_precision`` uses DEFAULT_DTYPES: theta/u/v fp32,
    w/mu/ph fp64) -- LEANER than the fp64 probe.  We ALSO run the shipped fp64
    stable config segmented as the faithful cross-check (it is the exact shipped
    namelist; fp64 is the v0.1.0-validated operating point).

Checkpoints record qke + mu/ph/theta/u/v/w maxima staying bounded out to >=3h and,
budget permitting, toward 24h.  GPU (cuda) job; ONE GPU job at a time (lock claimed
by the calling lane).  CPUs pinned to 0-3 (cores 4-31 are a live CPU-WRF backfill).
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
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # release freed scratch promptly
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    except OSError:
        pass

# Reuse the SHIPPED d02 case builder so we verify EXACTLY what ships (no bespoke
# namelist): build_l2_daily_case routes to the validated stability set and the
# d02_replay.build_replay_case WRF-faithful qke cold-start seed.
sys.path.insert(0, str(ROOT / "scripts"))
from m7_l2_d02_replay import build_l2_daily_case  # noqa: E402
from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    DailyPipelineConfig,
    finite_summary,
    resolve_run_dir,
)
from gpuwrf.runtime.operational_mode import run_forecast_operational  # noqa: E402

L2_RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l2")
DT_S = 12.0
SUBSTEPS = 10
RAD_CADENCE = 180  # 36 min @ dt=12, matches the qke-fix verify cadence


def _signature(state):
    summ = finite_summary(state)
    fields = summ["fields"]
    dyn = {}
    for k in ("u", "v", "w", "theta", "mu", "ph", "p", "qke"):
        if k in fields:
            f = fields[k]
            dyn[k] = {"fin": f["finite"], "max": f["max"], "min": f["min"]}
    nonfinite = {k: v["nonfinite_count"] for k, v in fields.items() if not v["finite"]}
    return {"all_finite": summ["all_finite"], "dyn": dyn, "nonfinite_fields": nonfinite}


def _run_config(name, *, force_fp64, run_dir, hours, step_h, checkpoint_hours):
    import jax

    cfg = DailyPipelineConfig(
        run_id=run_dir.name,
        hours=int(round(hours)),
        output_dir=Path("/tmp/v090_d02reverify") / name,
        proof_dir=Path("/tmp/v090_d02reverify") / name,
        run_root=L2_RUN_ROOT,
        score=False,
        domain="d02",
        dt_s=DT_S,
        acoustic_substeps=SUBSTEPS,
        radiation_cadence_steps=RAD_CADENCE,
    )
    case, _ = build_l2_daily_case(cfg)
    # The shipped d02 case is fp64; for the gated-fp32 leaner config we flip
    # force_fp64 off on a COPY of the shipped namelist (everything else identical:
    # same stability flags, same seeded qke state).  ADR-007 operational matrix.
    import dataclasses
    nl = case.namelist
    if not force_fp64:
        nl = dataclasses.replace(nl, force_fp64=False)
    state = case.state

    qke_t0 = float(jax.numpy.max(jax.numpy.asarray(state.qke)))
    print(f"\n=== {name}: force_fp64={force_fp64} step_h={step_h} qke_t0_max={qke_t0:.4g} ===")
    flags = {k: (bool(getattr(nl, k)) if isinstance(getattr(nl, k), bool)
                 else getattr(nl, k))
             for k in ("top_lid", "epssm", "w_damping", "damp_opt", "dampcoef", "zdamp",
                       "diff_6th_opt", "diff_6th_factor", "use_flux_advection", "force_fp64")}

    # Advance in FIXED ``step_h``-hour increments via the production
    # ``run_forecast_operational`` (the SAME entry daily_pipeline drives per hour).
    # Every advance has the SAME static ``hours=step_h`` so it compiles ONCE and is
    # REUSED for all subsequent increments (no per-checkpoint recompile).  step_h is
    # sized so each increment's step count stays well under the 600-step (14.6 GiB)
    # single-advance that OOM'd: at dt=12 s, step_h=0.5 -> 150 steps/advance.  This
    # is the memory-bounded, production-faithful path.
    trace = []
    first_blow = None
    done_h = 0.0
    t_start = time.perf_counter()
    checkpoint_set = sorted(set(checkpoint_hours))
    target_iter = iter(checkpoint_set)
    next_checkpoint = next(target_iter, None)
    while done_h < hours - 1e-9:
        adv = min(step_h, hours - done_h)
        try:
            state = run_forecast_operational(state, nl, float(adv))
            jax.block_until_ready(state)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return {
                "flags": flags, "force_fp64": force_fp64, "step_h": step_h,
                "qke_t0_max": qke_t0, "trace": trace,
                "error": f"{type(exc).__name__}: {exc}",
                "aborted_oom": "RESOURCE_EXHAUSTED" in str(exc),
                "first_nonfinite_hours": first_blow,
                "survived_hours": done_h,
                "elapsed_s": time.perf_counter() - t_start,
            }
        done_h = round(done_h + adv, 6)
        # only record a trace row at/after a requested checkpoint (keep output lean)
        record = False
        while next_checkpoint is not None and done_h >= next_checkpoint - 1e-9:
            record = True
            next_checkpoint = next(target_iter, None)
        if not record and done_h < hours - 1e-9:
            # still check finiteness cheaply each increment (catch early blowup)
            sig_quick = _signature(state)
            if not sig_quick["all_finite"]:
                record = True
        if not record:
            continue
        sig = _signature(state)
        step = int(round(done_h * 3600.0 / DT_S))
        row = {"hours": round(done_h, 3), "step": step, "all_finite": sig["all_finite"],
               "dyn": sig["dyn"], "nonfinite_fields": sig["nonfinite_fields"]}
        trace.append(row)
        q = sig["dyn"].get("qke", {})
        mu = sig["dyn"].get("mu", {})
        w = sig["dyn"].get("w", {})
        th = sig["dyn"].get("theta", {})
        print(f"  {done_h:>5.2f}h (step {step:>5d}) finite={sig['all_finite']} "
              f"qke.max={q.get('max')} mu.max={mu.get('max')} w.max={w.get('max')} "
              f"theta.max={th.get('max')} nonfin={list(sig['nonfinite_fields'].keys())[:5]}")
        if not sig["all_finite"] and first_blow is None:
            first_blow = done_h
            break

    finite_trace = [t for t in trace if t["all_finite"]]
    last = finite_trace[-1] if finite_trace else {}
    survived_h = round(done_h, 3) if first_blow is None else float(last.get("hours", 0.0))
    return {
        "flags": flags, "force_fp64": force_fp64, "step_h": step_h,
        "qke_t0_max": qke_t0, "trace": trace,
        "first_nonfinite_hours": first_blow,
        "survived_hours": survived_h,
        "survived_steps": int(round(survived_h * 3600.0 / DT_S)),
        "aborted_oom": False,
        "stable_through_3h": bool(first_blow is None and survived_h >= 3.0 - 1e-9),
        "elapsed_s": time.perf_counter() - t_start,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="20260521_18z_l2_72h_20260522T133443Z")
    ap.add_argument("--platform", default="cuda")
    ap.add_argument("--hours", type=float, default=6.0,
                    help="max forecast hours (>=3 required; pushes toward 24 if budget allows)")
    ap.add_argument("--step-h", type=float, default=0.5,
                    help="fixed forecast increment in hours (one compiled program, reused; "
                         "0.5h=150 steps @ dt=12, well under the 600-step OOM)")
    ap.add_argument("--configs", nargs="+", default=["gated_fp32", "fp64"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "proofs" / "v090" / "d02replay_2to3h_reverify.json")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ["JAX_PLATFORMS"] = args.platform
    import jax
    devices = [str(d) for d in jax.devices()]
    run_dir = resolve_run_dir(args.run_id, L2_RUN_ROOT)

    # Record-trace checkpoints (every step_h increment is run regardless; we only
    # emit a trace row at these hour marks to keep output lean): every step_h up to
    # 3h, then hourly toward --hours.
    checkpoints = []
    h = args.step_h
    while h <= 3.0 + 1e-9 and h <= args.hours + 1e-9:
        checkpoints.append(round(h, 6))
        h += args.step_h
    h = 4.0
    while h <= args.hours + 1e-9:
        checkpoints.append(float(h))
        h += 1.0
    checkpoints.append(round(args.hours, 6))

    config_defs = {
        "gated_fp32": dict(force_fp64=False),
        "fp64": dict(force_fp64=True),
    }
    results = {}
    for name in args.configs:
        opt = config_defs[name]
        results[name] = _run_config(
            name, run_dir=run_dir, hours=args.hours,
            step_h=args.step_h, checkpoint_hours=checkpoints, **opt
        )

    any_3h = any(r.get("stable_through_3h") for r in results.values())
    max_finite_h = max((r.get("survived_hours", 0.0) for r in results.values()), default=0.0)
    payload = {
        "schema": "V090D02Replay2to3hReverify",
        "generated": datetime.now(timezone.utc).isoformat(),
        "branch": "worker/opus/v090-qkefix-followup",
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "platform": args.platform,
        "devices": devices,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "dt_s": DT_S,
        "acoustic_substeps": SUBSTEPS,
        "radiation_cadence_steps": RAD_CADENCE,
        "max_hours_requested": args.hours,
        "step_h": args.step_h,
        "runner": "run_forecast_operational in fixed step_h increments (one compiled program reused per increment; memory-bounded; no 600-step single advance / no incremental fp64 probe)",
        "fix_under_test": "shipped build_l2_daily_case = validated stability namelist + WRF-faithful MYNN qke cold-start seed (no clamps)",
        "wrf_seed_ref": "phys/module_bl_mynnedmf.F:618-691 (mym_initialize INITIALIZE_QKE)",
        "checkpoints_hours": checkpoints,
        "results": results,
        "any_config_finite_through_3h": bool(any_3h),
        "max_finite_hours": float(max_finite_h),
        "verdict": ("FINITE_THROUGH_3H_PLUS" if any_3h else "DID_NOT_REACH_3H"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"\nverdict: {payload['verdict']}  max_finite_hours={max_finite_h}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
