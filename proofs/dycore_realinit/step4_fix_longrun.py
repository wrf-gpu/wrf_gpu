"""Step 4: confirm the FIX over the full hour (>=360 steps).

Findings from steps 1-3:
  * Mode A (top-face w=307 spike, step 1): 100% the OPEN-TOP w BC. top_lid=True
    -> step1 top face exactly 0, |w| 13.5 (step2_bc_isolation).
  * Mode B (slow u growth at BOTH lateral edges): periodic advection on a
    non-periodic LAM run with run_boundary=False; cured by run_boundary=True
    (real d02 specified+relaxation boundaries) -- but ONLY with a rigid lid.

Candidate fix configs, dycore-only (run_physics=False), disable_guards=True,
validated operational namelist otherwise, epssm=0.5 (real d02 namelist.input):

  A. rigid_lid + no_bndy          -- isolate the top fix alone
  B. rigid_lid + with_bndy        -- the full LAM config (expected stable)

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step4_fix_longrun.py --steps 360
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


def run(case, nl, steps, label):
    carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def _step(c, idx):
        return _physics_boundary_step(c, nl, idx, run_radiation=False, debug=False)

    hist = []
    first_nonfinite = None
    u = np.zeros(1)
    for s in range(1, steps + 1):
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32))
        wabs = jnp.abs(carry.state.w)
        uabs = jnp.abs(carry.state.u)
        rec = {
            "step": s,
            "w_absmax": float(jnp.max(wabs)),
            "w_top_face": float(jnp.max(wabs[-1])),
            "u_absmax": float(jnp.max(uabs)),
            "v_absmax": float(jnp.max(jnp.abs(carry.state.v))),
            "theta_min": float(jnp.min(carry.state.theta)),
            "theta_max": float(jnp.max(carry.state.theta)),
        }
        hist.append(rec)
        if first_nonfinite is None and not np.isfinite(rec["w_absmax"]):
            first_nonfinite = s
            break
        if s <= 5 or s % 30 == 0:
            print(f"  [{label}] step {s:4d}: |w|={rec['w_absmax']:.2f} "
                  f"(top {rec['w_top_face']:.2f}) |u|={rec['u_absmax']:.2f} "
                  f"|v|={rec['v_absmax']:.2f} theta[{rec['theta_min']:.1f},{rec['theta_max']:.1f}]", flush=True)
    jax.block_until_ready(carry.state.u)
    u = np.asarray(jax.device_get(jnp.abs(carry.state.u)))
    u_vs_x = u.max(axis=(0, 1))
    f = hist[-1]
    stable = (
        first_nonfinite is None
        and f["w_absmax"] < 30.0
        and f["u_absmax"] < 150.0
        and 150.0 <= f["theta_min"] and f["theta_max"] <= 550.0
    )
    return {"label": label, "top_lid": bool(nl.top_lid), "run_boundary": bool(nl.run_boundary),
            "epssm": nl.epssm, "steps": len(hist), "first_nonfinite": first_nonfinite,
            "step1_w": hist[0]["w_absmax"], "step1_top_face": hist[0]["w_top_face"],
            "final": f, "final_u_west_edge": float(u_vs_x[0]),
            "final_u_east_edge": float(u_vs_x[-1]), "final_u_interior_max": float(u_vs_x[10:150].max()),
            "verdict": "STABLE" if stable else "UNSTABLE", "history": hist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=360)
    args = ap.parse_args()
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    # epssm=0.5 to match the real Gen2 d02 namelist.input (&dynamics epssm=0.5).
    base = dataclasses.replace(case.namelist, run_physics=False, disable_guards=True, epssm=0.5)

    variants = {
        "rigid_lid_no_bndy":   dataclasses.replace(base, top_lid=True, run_boundary=False),
        "rigid_lid_with_bndy": dataclasses.replace(base, top_lid=True, run_boundary=True),
    }
    out = {}
    for name, nl in variants.items():
        print(f"\n=== {name} (top_lid={nl.top_lid} run_boundary={nl.run_boundary} "
              f"epssm={nl.epssm}, {args.steps} steps) ===", flush=True)
        out[name] = run(case, nl, args.steps, name)
        r = out[name]
        print(f"  VERDICT[{name}]: {r['verdict']} step1_top={r['step1_top_face']:.2f} "
              f"final |w|={r['final']['w_absmax']:.2f} |u|={r['final']['u_absmax']:.2f} "
              f"(W_edge={r['final_u_west_edge']:.1f} int={r['final_u_interior_max']:.1f} "
              f"E_edge={r['final_u_east_edge']:.1f})", flush=True)

    (PROOF / "step4_fix_longrun.json").write_text(json.dumps({"steps": args.steps, "variants": out}, indent=2) + "\n")
    print(f"\nwrote {PROOF / 'step4_fix_longrun.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
