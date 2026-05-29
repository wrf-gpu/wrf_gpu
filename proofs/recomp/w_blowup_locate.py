"""Locate WHERE the dry-dycore w instability lives (vertical level + map).

Runs dycore-only guards-off for a few steps and reports, per step, the (z,y,x)
of max|w| and the per-level max|w| profile. Distinguishes a model-top problem
(top damping / rhs_ph stub) from a terrain-following lower-level problem
(pressure gradient over steep slopes).
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
    _dycore_step_adr023,
    build_replay_case,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--n-acoustic", type=int, default=4)
    ap.add_argument("--out", type=str, default="proofs/recomp/w_blowup_locate.json")
    args = ap.parse_args()

    case = build_replay_case(DEFAULT_REPLAY_RUN_DIR, domain="d02")
    cfg = ReplayConfig(dt_s=args.dt, duration_s=args.dt * args.steps, n_acoustic=args.n_acoustic)
    grid = case.grid
    nz1 = grid.nz + 1

    state = case.state
    pp = case.previous_pressure
    steps_out = []
    for i in range(args.steps):
        state, pp = _dycore_step_adr023(
            state, pp, case.tendencies, grid, case.metrics, case.base_state, cfg
        )
        w = np.asarray(state.w)  # (nz+1, ny, nx)
        absw = np.abs(w)
        if not np.all(np.isfinite(absw)):
            absw = np.where(np.isfinite(absw), absw, 0.0)
        flat = int(np.argmax(absw))
        zk, yj, xi = np.unravel_index(flat, absw.shape)
        per_level_max = [float(np.nanmax(absw[k])) for k in range(absw.shape[0])]
        rec = {
            "step": i + 1,
            "w_absmax": float(np.nanmax(absw)),
            "argmax_zyx": [int(zk), int(yj), int(xi)],
            "argmax_z_frac": float(zk) / float(nz1 - 1),
            "per_level_max_w": per_level_max,
        }
        steps_out.append(rec)
        topk = np.argsort(per_level_max)[-5:][::-1]
        print(f"[wloc] step {i+1}: |w|max={rec['w_absmax']:.2f} at z={zk}/{nz1-1} "
              f"(y={yj},x={xi}); top-5 noisy levels(z:|w|)="
              + ", ".join(f"{int(k)}:{per_level_max[k]:.1f}" for k in topk), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dt_s": args.dt, "n_acoustic": args.n_acoustic,
                               "nz_stag": nz1, "steps": steps_out}, indent=2))
    print(f"[wloc] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
