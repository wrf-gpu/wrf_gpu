"""Step 2: isolate the two boundary modes from step1.

Mode A (top-face w spike, k=44): is it the OPEN top? Test top_lid=True (rigid
lid) vs the validated top_lid=False, dycore-only, run_boundary=False.

Mode B (lateral-edge u growth, x=0/x=159 symmetric -> periodic wrap on a LAM):
is it the missing lateral-boundary specification? Test run_boundary=True (the
operational lateral boundary apply) vs False.

We run a short window (default 60 steps) for each variant and report the w/u
extrema and the edge-vs-interior u localization, so each mode is attributed to a
specific BC switch with one number.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step2_bc_isolation.py --steps 60
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


def run_variant(case, nl, steps):
    carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def _step(c, idx):
        return _physics_boundary_step(c, nl, idx, run_radiation=False, debug=False)

    hist = []
    for s in range(1, steps + 1):
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32))
        w = jnp.abs(carry.state.w)
        u = jnp.abs(carry.state.u)
        rec = {
            "step": s,
            "w_absmax": float(jnp.max(w)),
            "w_top_face": float(jnp.max(w[-1])),
            "u_absmax": float(jnp.max(u)),
            "theta_max": float(jnp.max(carry.state.theta)),
        }
        hist.append(rec)
    jax.block_until_ready(carry.state.u)
    u = np.asarray(jax.device_get(jnp.abs(carry.state.u)))  # (nz,ny,nx+1)
    u_vs_x = u.max(axis=(0, 1))
    return {
        "step1_w_absmax": hist[0]["w_absmax"],
        "step1_w_top_face": hist[0]["w_top_face"],
        "final_w_absmax": hist[-1]["w_absmax"],
        "final_u_absmax": hist[-1]["u_absmax"],
        "final_theta_max": hist[-1]["theta_max"],
        "final_u_west_edge": float(u_vs_x[0]),
        "final_u_east_edge": float(u_vs_x[-1]),
        "final_u_interior_max": float(u_vs_x[10:150].max()),
        "history": hist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    base = dataclasses.replace(case.namelist, run_physics=False, run_boundary=False, disable_guards=True)

    variants = {
        "open_top_no_bndy (validated baseline)": base,
        "rigid_lid_no_bndy": dataclasses.replace(base, top_lid=True),
        "open_top_with_bndy": dataclasses.replace(base, run_boundary=True),
        "rigid_lid_with_bndy": dataclasses.replace(base, top_lid=True, run_boundary=True),
    }
    out = {}
    for name, nl in variants.items():
        print(f"\n=== variant: {name} (top_lid={nl.top_lid} run_boundary={nl.run_boundary}) ===", flush=True)
        try:
            r = run_variant(case, nl, args.steps)
            print(f"  step1 |w|={r['step1_w_absmax']:.2f} (top face {r['step1_w_top_face']:.2f}); "
                  f"final |w|={r['final_w_absmax']:.2f} |u|={r['final_u_absmax']:.2f} "
                  f"theta_max={r['final_theta_max']:.1f}", flush=True)
            print(f"  final u: west_edge={r['final_u_west_edge']:.2f} "
                  f"interior={r['final_u_interior_max']:.2f} east_edge={r['final_u_east_edge']:.2f}", flush=True)
            out[name] = r
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            out[name] = {"error": f"{type(e).__name__}: {e}"}

    (PROOF / "step2_bc_isolation.json").write_text(json.dumps({"steps": args.steps, "variants": out}, indent=2) + "\n")
    print(f"\nwrote {PROOF / 'step2_bc_isolation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
