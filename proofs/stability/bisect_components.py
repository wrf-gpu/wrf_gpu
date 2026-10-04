"""STABILITY component-disable bisection (task #36, READ-ONLY on src).

Once the onset hour is known, run SHORT variants that each disable ONE component
and observe whether the onset is delayed/removed.  We disable components by
monkeypatching the adapter names the operational step looks up (this does NOT edit
src; the closure resolves bare names against the module global table) and by
flipping the damping / boundary / top_lid namelist flags.

Variants (each run to --hours, a small margin past the baseline onset):
  baseline      : nothing disabled (must reproduce the onset)
  no_thompson   : thompson_adapter -> identity
  no_mynn       : mynn_adapter -> identity
  no_rrtmg      : rrtmg_adapter -> identity   (also kills the lumped-heating pulse)
  no_surface    : surface_adapter -> identity
  no_boundary   : run_boundary=False
  no_wdamp      : w_damping=0
  no_raydamp    : damp_opt=0 (no upper Rayleigh)
  no_physics    : run_physics=False (pure dycore + boundary)
  opentop       : top_lid=False (sanity: should be WORSE/faster if top mode)

For each variant we record the first non-finite hour (or "survived") and the
theta/w/u abs-max growth, written incrementally to JSONL.

Run:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
    PYTHONPATH=src taskset -c 2-3 python proofs/stability/bisect_components.py \
      --hours 14 --variants baseline,no_rrtmg,no_raydamp,no_boundary,no_physics
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
import gpuwrf.runtime.operational_mode as om
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

FIELDS = ("u", "v", "w", "theta", "ph", "qv", "qc", "qr", "qi", "qs", "qg")


def _identity_dt(state, dt, *a, **k):
    return state


def _summary(state):
    out = {}
    bad = False
    for f in FIELDS:
        a = np.asarray(jax.device_get(getattr(state, f)), dtype=np.float64)
        fm = np.isfinite(a)
        finite = bool(fm.all())
        fv = a[fm]
        lev = (np.where(fm, np.abs(a), 0.0).reshape(a.shape[0], -1).max(axis=1)
               if a.ndim == 3 else None)
        out[f] = {"finite": finite,
                  "absmax": float(np.abs(fv).max()) if fv.size else None,
                  "worst_level": int(np.argmax(lev)) if lev is not None else None}
        if not finite:
            bad = True
    out["_all_finite"] = not bad
    return out


def _make_nl(case, *, run_physics=True, run_boundary=True, w_damping=1,
             damp_opt=3, top_lid=True):
    return dataclasses.replace(
        case.namelist, run_physics=run_physics, run_boundary=run_boundary,
        disable_guards=True, radiation_cadence_steps=180, time_utc=case.run_start,
        w_damping=w_damping, damp_opt=damp_opt, top_lid=top_lid,
    )


# variant -> (namelist-kwargs, set-of-adapters-to-noop)
VARIANTS = {
    "baseline":    (dict(), set()),
    "no_thompson": (dict(), {"thompson_adapter"}),
    "no_mynn":     (dict(), {"mynn_adapter"}),
    "no_rrtmg":    (dict(), {"rrtmg_adapter"}),
    "no_surface":  (dict(), {"surface_adapter"}),
    "no_boundary": (dict(run_boundary=False), set()),
    "no_wdamp":    (dict(w_damping=0), set()),
    "no_raydamp":  (dict(damp_opt=0), set()),
    "no_damp_all": (dict(w_damping=0, damp_opt=0), set()),
    "no_physics":  (dict(run_physics=False), set()),
    "opentop":     (dict(top_lid=False), set()),
}

# the real adapter objects so we can restore between variants
_REAL = {n: getattr(om, n) for n in
         ("thompson_adapter", "mynn_adapter", "rrtmg_adapter", "surface_adapter")}


def _run_variant(case, name, hours, seg, fh):
    nl_kw, noops = VARIANTS[name]
    nl = _make_nl(case, **nl_kw)
    # apply monkeypatches BEFORE any trace
    for n in _REAL:
        setattr(om, n, _identity_dt if n in noops else _REAL[n])
    # CRITICAL: _advance_chunk is @jax.jit; its cache key is (static args, shapes),
    # NOT the Python globals it closes over.  Without clearing the cache, a second
    # variant in the same process would reuse the FIRST variant's compiled graph and
    # silently ignore the new monkeypatch.  Clear so this variant re-traces with the
    # adapters patched right now.  (One variant per process would also work; this is
    # the in-process-safe equivalent and keeps the case build amortized.)
    jax.clear_caches()
    cadence = 180
    dt_s = 10.0
    steps = int(round(hours * 3600.0 / dt_s))
    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )
    start = 1
    t0 = time.perf_counter()
    first_bad = None
    last_finite_h = 0.0
    last_summary = None
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                               n_steps=int(n), cadence=cadence)
        jax.block_until_ready(carry.state.theta)
        end = start + n - 1
        lead = end * dt_s / 3600.0
        s = _summary(carry.state)
        last_summary = s
        rec = {"kind": "variant_hour", "variant": name, "lead_hours": round(lead, 2),
               "all_finite": s["_all_finite"],
               "theta_absmax": s["theta"]["absmax"], "theta_k": s["theta"]["worst_level"],
               "w_absmax": s["w"]["absmax"], "w_k": s["w"]["worst_level"],
               "u_absmax": s["u"]["absmax"], "u_k": s["u"]["worst_level"]}
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(f"[{name}] lead={lead:.2f}h finite={s['_all_finite']} "
              f"theta={s['theta']['absmax']}@k{s['theta']['worst_level']} "
              f"w={s['w']['absmax']}@k{s['w']['worst_level']} "
              f"u={s['u']['absmax']}@k{s['u']['worst_level']}", flush=True)
        if s["_all_finite"]:
            last_finite_h = lead
        elif first_bad is None:
            first_bad = lead
            bad_fields = [f for f in FIELDS if not s[f]["finite"]]
            print(f"  >>> {name} NON-FINITE at {lead:.2f}h: {bad_fields}", flush=True)
            break
        start += n
    # restore
    for n in _REAL:
        setattr(om, n, _REAL[n])
    res = {"kind": "variant_result", "variant": name,
           "first_nonfinite_hours": first_bad,
           "last_finite_hours": last_finite_h,
           "survived": first_bad is None,
           "wall_s": round(time.perf_counter() - t0, 1),
           "final_summary": last_summary}
    fh.write(json.dumps(res) + "\n"); fh.flush()
    print(f"=== {name}: first_nonfinite={first_bad}h survived={first_bad is None} "
          f"wall={res['wall_s']}s ===", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--seg-steps", type=int, default=360)
    ap.add_argument("--variants", type=str, default="baseline")
    ap.add_argument("--out", type=str, default="proofs/stability/bisect")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choices={list(VARIANTS)}")

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    jsonl = Path(f"{args.out}.jsonl")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = jsonl.open("a")
    fh.write(json.dumps({"kind": "meta", "run_dir": str(run_dir),
                         "hours": args.hours, "variants": variants}) + "\n")
    fh.flush()
    results = {}
    for v in variants:
        results[v] = _run_variant(case, v, args.hours, args.seg_steps, fh)
    fh.close()
    print("\nSUMMARY:", flush=True)
    for v in variants:
        r = results[v]
        print(f"  {v:14s} first_nonfinite={r['first_nonfinite_hours']}h "
              f"survived={r['survived']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
