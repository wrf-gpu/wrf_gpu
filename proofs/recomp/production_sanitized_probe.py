"""Probe the PRODUCTION sanitized d02 replay path: is the guard load-bearing?

Runs the real coupled replay through run_replay_scan (sanitiser ON, the actual
production code path) for a short horizon and reports the per-step clip/nonfinite
counts and field extrema. If the w clip_count is large every step, the sanitiser
is masking a dry-dynamics w instability (i.e. the guard is load-bearing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.d02_replay import (
    DEFAULT_REPLAY_RUN_DIR,
    ReplayConfig,
    build_replay_case,
    run_replay_scan,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--n-acoustic", type=int, default=4)
    ap.add_argument("--no-physics", action="store_true",
                    help="dry-only: skip building physics into the candidate is not "
                         "supported by run_replay_scan; this flag is informational")
    ap.add_argument("--out", type=str, default="proofs/recomp/production_sanitized_probe.json")
    args = ap.parse_args()

    case = build_replay_case(DEFAULT_REPLAY_RUN_DIR, domain="d02")
    cfg = ReplayConfig(dt_s=args.dt, duration_s=args.dt * args.steps,
                       n_acoustic=args.n_acoustic, radiation_cadence_steps=max(args.steps, 1))
    final_state, _pp, diags = run_replay_scan(
        case.state, case.previous_pressure, case.tendencies,
        case.grid, case.metrics, case.base_state, cfg,
    )
    # diags is StepDiagnostics with per-step arrays
    clip = np.asarray(diags.candidate_clip_count)
    nonf = np.asarray(diags.candidate_nonfinite_count)
    changed = np.asarray(diags.candidate_changed_count)
    wmax = np.asarray(diags.w_abs_max_m_s)
    tmin = np.asarray(diags.theta_min_k)
    tmax = np.asarray(diags.theta_max_k)
    finite = np.asarray(diags.finite_after_sanitize)

    per_step = []
    for i in range(len(clip)):
        per_step.append({
            "step": i + 1,
            "clip_count": int(clip[i]),
            "nonfinite_count": int(nonf[i]),
            "changed_count": int(changed[i]),
            "w_absmax_post_sanitize": float(wmax[i]),
            "theta_min_post_sanitize": float(tmin[i]),
            "theta_max_post_sanitize": float(tmax[i]),
            "finite": bool(finite[i]),
        })
        print(f"[prod] step {i+1}: clip={int(clip[i])} nonfinite={int(nonf[i])} "
              f"changed={int(changed[i])} |w|post={float(wmax[i]):.2f} "
              f"theta[{float(tmin[i]):.1f},{float(tmax[i]):.1f}] finite={bool(finite[i])}",
              flush=True)

    result = {
        "dt_s": args.dt, "n_acoustic": args.n_acoustic, "steps": args.steps,
        "total_clip_count": int(clip.sum()),
        "total_nonfinite_count": int(nonf.sum()),
        "per_step": per_step,
        "verdict": ("GUARD_LOAD_BEARING" if clip.sum() > 0 or nonf.sum() > 0
                    else "GUARD_NOT_ENGAGING"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[prod] total clip={int(clip.sum())} nonfinite={int(nonf.sum())} "
          f"=> {result['verdict']}", flush=True)
    print(f"[prod] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
