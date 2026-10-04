"""STABILITY onset localizer (task #36, READ-ONLY diagnostic).

Walk the coupled real-case forecast (full physics + boundaries, guards OFF, fp64)
hour-by-hour using the SAME compiled segment that the operational segmented entry
uses (``_advance_chunk``), and at every hour boundary record, for EVERY prognostic
field:

  * global finiteness,
  * global min/max,
  * per-LEVEL (axis-0 == vertical k) abs-max  -> tells us WHICH level blows first,
  * the (k, j, i) location of the global |max| -> tells us WHERE in the column/domain,
  * |max| growth so we can see gradual-runaway vs sudden-NaN.

Everything is written to a JSONL log line-by-line (flush) so a watchdog kill loses
at most the in-flight hour.  This script does NOT modify src; it only reads State.

Run (GPU-shared; cap to half the card):
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
    PYTHONPATH=src taskset -c 2-3 python proofs/stability/onset_localize.py \
      --hours 24 --seg-steps 360 --out proofs/stability/onset_trace
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
from gpuwrf.runtime.operational_mode import (
    _advance_chunk,
    _enforce_operational_precision,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

# Every prognostic 3-D (or 2-D) field we care about.  3-D fields have axis 0 == k.
FIELDS_3D = ("u", "v", "w", "theta", "ph", "qv", "qc", "qr", "qi", "qs", "qg",
             "qke", "Ni", "Nr", "Ns", "Ng")
FIELDS_2D = ("mu", "mu_total")


def _field_report(state, name):
    a = np.asarray(jax.device_get(getattr(state, name)), dtype=np.float64)
    finite_mask = np.isfinite(a)
    all_finite = bool(finite_mask.all())
    rep = {"finite": all_finite, "shape": list(a.shape)}
    if not all_finite:
        rep["n_nonfinite"] = int(a.size - int(finite_mask.sum()))
        rep["n_nan"] = int(np.isnan(a).sum())
        rep["n_inf"] = int(np.isinf(a).sum())
        idx = np.argwhere(~finite_mask)
        if idx.size:
            rep["first_nonfinite_idx"] = [int(x) for x in idx[0]]
            if a.ndim == 3:
                per_k_bad = (~finite_mask).reshape(a.shape[0], -1).sum(axis=1)
                bad_levels = np.argwhere(per_k_bad > 0).ravel()
                rep["nonfinite_levels"] = [int(k) for k in bad_levels[:8]]
                rep["first_nonfinite_level"] = int(bad_levels[0]) if bad_levels.size else None
    fv = a[finite_mask]
    if fv.size:
        rep["fmin"] = float(fv.min())
        rep["fmax"] = float(fv.max())
        rep["absmax"] = float(np.abs(fv).max())
        masked = np.where(finite_mask, np.abs(a), -np.inf)
        flat = int(np.argmax(masked))
        rep["absmax_idx"] = [int(x) for x in np.unravel_index(flat, a.shape)]
        if a.ndim == 3:
            lev_absmax = np.where(finite_mask, np.abs(a), 0.0).reshape(a.shape[0], -1).max(axis=1)
            rep["per_level_absmax"] = [round(float(x), 4) for x in lev_absmax]
            rep["worst_level"] = int(np.argmax(lev_absmax))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seg-steps", type=int, default=360)  # 1 hour at dt=10s
    ap.add_argument("--out", type=str, default="proofs/stability/onset_trace")
    ap.add_argument("--guards", choices=["off", "on"], default="off")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    cadence = 180
    dt_s = 10.0
    steps = int(round(args.hours * 3600.0 / dt_s))
    seg = int(args.seg_steps)

    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True,
        disable_guards=(args.guards == "off"),
        radiation_cadence_steps=cadence, time_utc=case.run_start,
    )

    jsonl = Path(f"{args.out}_{args.guards}.jsonl")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = jsonl.open("w")
    meta = {"kind": "meta", "run_dir": str(run_dir), "guards": args.guards,
            "hours": args.hours, "seg_steps": seg, "cadence": cadence,
            "grid": {"ny": int(case.grid.ny), "nx": int(case.grid.nx), "nz": int(case.grid.nz)},
            "config": {"epssm": float(nl.epssm), "top_lid": bool(nl.top_lid),
                       "w_damping": int(nl.w_damping), "damp_opt": int(nl.damp_opt),
                       "zdamp": float(nl.zdamp), "dampcoef": float(nl.dampcoef),
                       "force_fp64": bool(nl.force_fp64),
                       "use_flux_advection": bool(nl.use_flux_advection)}}
    fh.write(json.dumps(meta) + "\n"); fh.flush()
    print("META", json.dumps(meta), flush=True)

    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )
    start = 1
    t0 = time.perf_counter()
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(
            carry, nl, jnp.asarray(start, dtype=jnp.int32), n_steps=int(n), cadence=cadence
        )
        jax.block_until_ready(carry.state.theta)
        end = start + n - 1
        rec = {"kind": "hour", "global_step": end, "lead_hours": end * dt_s / 3600.0,
               "wall_s": round(time.perf_counter() - t0, 1), "fields": {}}
        any_bad = False
        for f in FIELDS_3D + FIELDS_2D:
            try:
                r = _field_report(carry.state, f)
            except Exception as e:
                r = {"err": str(e)}
            rec["fields"][f] = r
            if not r.get("finite", True):
                any_bad = True
        rec["all_finite"] = not any_bad
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        th = rec["fields"]["theta"]; w = rec["fields"]["w"]; u = rec["fields"]["u"]
        print(f"[{args.guards}] step={end} lead={end*dt_s/3600.0:.2f}h finite={not any_bad} "
              f"theta_absmax={th.get('absmax', float('nan')):.1f}@k{th.get('worst_level','?')} "
              f"w_absmax={w.get('absmax', float('nan')):.2f}@k{w.get('worst_level','?')} "
              f"u_absmax={u.get('absmax', float('nan')):.1f}@k{u.get('worst_level','?')} "
              f"wall={rec['wall_s']}s", flush=True)
        if any_bad:
            bad_fields = [f for f, r in rec["fields"].items() if not r.get("finite", True)]
            print(f"  >>> NON-FINITE at step {end}: {bad_fields}", flush=True)
            break
        start += n
    fh.close()
    print(f"wrote {jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
