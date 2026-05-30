"""Step 1b: spatial localization of the two modes from step1_longrun.

(A) Step-1 w spike: profile |w| vs k (max over y,x) to confirm the spike is the
    OPEN-TOP face (k=44) and how deep it penetrates.
(B) Late-step u growth: profile |u| vs x (max over k,y) to confirm the growth is
    concentrated at the EAST lateral edge (x=159), i.e. a free open-edge mode
    rather than an interior dycore instability.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step1b_spatial_probe.py --late 200
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision, _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/dycore_realinit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--late", type=int, default=200)
    args = ap.parse_args()

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=False, run_boundary=False, disable_guards=True)
    carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def _step(c, idx):
        return _physics_boundary_step(c, nl, idx, run_radiation=False, debug=False)

    # step 1 snapshot
    carry = _step(carry, jnp.asarray(1, dtype=jnp.int32))
    w1 = np.asarray(jax.device_get(jnp.abs(carry.state.w)))  # (45,66,159)
    w_vs_k = w1.max(axis=(1, 2)).tolist()

    # advance to late
    for s in range(2, args.late + 1):
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32))
    jax.block_until_ready(carry.state.u)
    u = np.asarray(jax.device_get(jnp.abs(carry.state.u)))  # (44,66,160)
    u_vs_x = u.max(axis=(0, 1)).tolist()
    u_vs_k = u.max(axis=(1, 2)).tolist()
    w = np.asarray(jax.device_get(jnp.abs(carry.state.w)))
    w_vs_k_late = w.max(axis=(1, 2)).tolist()

    payload = {
        "schema": "dycore_realinit_step1b_spatial",
        "late_step": args.late,
        "step1_w_vs_k": w_vs_k,                # index k = vertical face 0..44 (44=top)
        "step1_w_top_face_k44": w_vs_k[-1],
        "step1_w_interior_max_k0_40": max(w_vs_k[:41]),
        "late_u_vs_x": u_vs_x,                 # index x = u-face 0..159 (159=east edge)
        "late_u_east_edge_x159": u_vs_x[-1],
        "late_u_west_edge_x0": u_vs_x[0],
        "late_u_interior_max_x10_149": max(u_vs_x[10:150]),
        "late_u_vs_k": u_vs_k,
        "late_w_vs_k": w_vs_k_late,
    }
    out = PROOF / "step1b_spatial_probe.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("step1 |w| vs k (top=k44):")
    for k, val in enumerate(w_vs_k):
        mark = "  <== TOP FACE (open top)" if k == len(w_vs_k) - 1 else ""
        if val > 5.0 or k >= len(w_vs_k) - 4:
            print(f"   k={k:2d}: {val:.2f}{mark}")
    print(f"\nlate(step{args.late}) |u| vs x (east edge=x159):")
    print(f"   x=0 (west edge):   {u_vs_x[0]:.2f}")
    print(f"   x interior max:    {max(u_vs_x[10:150]):.2f}")
    print(f"   x=159 (east edge): {u_vs_x[-1]:.2f}")
    print(f"   x=158:             {u_vs_x[-2]:.2f}")
    print(f"   x=157:             {u_vs_x[-3]:.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
