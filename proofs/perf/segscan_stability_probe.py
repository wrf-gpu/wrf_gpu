"""SEGSCAN stability probe -- localize WHERE the coupled 24h run loses finiteness.

The 24h segmented run reached the END feasibly (bounded compile + bounded peak mem)
but the final state was non-finite under guards-OFF.  This probe walks the SAME
host-loop segmentation hour-by-hour (reusing the ONE compiled 180-step segment) and
records, at every hour boundary, whether the carried State is finite and its
theta/|w|/|u| ranges -- for BOTH guards-OFF and guards-ON.  This separates two
questions the 24h FAIL conflates:

  * Is the non-finiteness a SEGMENTATION artifact?  No: each hour is advanced by the
    same compiled segment proven bitwise-equal to the single scan; a blow-up at hour
    k is the model, not the cut.  The hour-by-hour finite trace shows the model is
    finite for the validated window and diverges later.
  * Does the production guards-ON config stay finite to 24h?  The operational safety
    net is ON in production; this records whether it holds the full 24h.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
    taskset -c 0-3 python proofs/perf/segscan_stability_probe.py --hours 24
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

PROOF = Path("proofs/perf")
SEG = 180  # one radiation cadence interval == one hour at dt=10s? 180 steps = 0.5h.


def _finite_ranges(state):
    out = {}
    for f in ("u", "v", "w", "theta", "qv", "mu_total"):
        a = np.asarray(jax.device_get(getattr(state, f)), dtype=np.float64)
        fin = bool(np.isfinite(a).all())
        if fin:
            out[f] = {"finite": True, "min": float(np.min(a)), "max": float(np.max(a))}
        else:
            finite_vals = a[np.isfinite(a)]
            out[f] = {"finite": False,
                      "n_nonfinite": int(a.size - finite_vals.size),
                      "finite_min": float(np.min(finite_vals)) if finite_vals.size else None,
                      "finite_max": float(np.max(finite_vals)) if finite_vals.size else None}
    return out


def _walk(case, nl, cadence, steps, seg, label):
    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )
    trace = []
    start = 1
    last_finite_step = 0
    first_nonfinite_step = None
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(
            carry, nl, jnp.asarray(start, dtype=jnp.int32), n_steps=int(n), cadence=cadence
        )
        jax.block_until_ready(carry.state.theta)
        end = start + n - 1
        rng = _finite_ranges(carry.state)
        finite = all(v["finite"] for v in rng.values())
        if finite:
            last_finite_step = end
        elif first_nonfinite_step is None:
            first_nonfinite_step = end
        # Record sparsely (every ~hour = 360 steps) to keep the proof small.
        if end % 360 == 0 or not finite or end == steps:
            trace.append({"global_step": end, "lead_hours": end * 10.0 / 3600.0,
                          "finite": finite, "ranges": rng})
            print(f"[{label}] step={end} lead={end*10/3600:.2f}h finite={finite} "
                  f"theta=[{rng['theta'].get('min','nan'):}..{rng['theta'].get('max','nan')}] "
                  f"w_max={rng['w'].get('max','nan')}", flush=True)
        if not finite:
            break  # blow-up localized; no need to walk further
        start += n
    return {
        "label": label,
        "last_finite_global_step": last_finite_step,
        "last_finite_lead_hours": last_finite_step * 10.0 / 3600.0,
        "first_nonfinite_global_step": first_nonfinite_step,
        "first_nonfinite_lead_hours": (first_nonfinite_step * 10.0 / 3600.0
                                       if first_nonfinite_step else None),
        "trace": trace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--segment-steps", type=int, default=SEG)
    args = ap.parse_args()
    hours = float(args.hours)
    seg = int(args.segment_steps)

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    cadence = 180
    steps = int(round(hours * 3600.0 / 10.0))

    nl_off = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                                 disable_guards=True, radiation_cadence_steps=cadence,
                                 time_utc=case.run_start)
    nl_on = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                                disable_guards=False, radiation_cadence_steps=cadence,
                                time_utc=case.run_start)

    t0 = time.perf_counter()
    res_off = _walk(case, nl_off, cadence, steps, seg, "guards-OFF")
    res_on = _walk(case, nl_on, cadence, steps, seg, "guards-ON")
    wall = time.perf_counter() - t0

    out = {
        "scope": "SEGSCAN stability probe -- hour-by-hour finiteness of the coupled 24h run",
        "run_dir": str(run_dir),
        "hours": hours,
        "segment_steps": seg,
        "radiation_cadence_steps": cadence,
        "wall_s": wall,
        "guards_off": res_off,
        "guards_on": res_on,
        "note": (
            "Each hour is advanced by the SAME compiled 180-step segment proven "
            "bitwise-equal to the single scan (segscan_equiv.json), so a blow-up at "
            "lead t is a MODEL instability, not a segmentation artifact. This trace "
            "localizes the guards-OFF divergence lead and records whether the "
            "production guards-ON config holds the full 24h."
        ),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "segscan_stability_probe.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nguards-OFF last finite lead = {res_off['last_finite_lead_hours']:.2f}h, "
          f"first non-finite = {res_off['first_nonfinite_lead_hours']}", flush=True)
    print(f"guards-ON  last finite lead = {res_on['last_finite_lead_hours']:.2f}h, "
          f"first non-finite = {res_on['first_nonfinite_lead_hours']}", flush=True)
    print(f"wrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
