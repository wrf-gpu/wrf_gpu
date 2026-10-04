"""Step 5: can the WRF-faithful OPEN top be stable WITH the real LAM boundaries?

The real Gen2 d02 namelist.input has no explicit top_lid -> WRF default open top.
step4 proved rigid_lid+with_bndy is stable, but rigid lid deviates from the real
namelist. This tests the open top WITH the real lateral boundaries at epssm=0.5
(real namelist) over the full hour, to decide whether the operational path should
use the open top (faithful) or the rigid lid (validated idealized top).

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step5_opentop_bndy.py --steps 360
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
    for s in range(1, steps + 1):
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32))
        wabs = jnp.abs(carry.state.w)
        rec = {"step": s, "w_absmax": float(jnp.max(wabs)),
               "w_top_face": float(jnp.max(wabs[-1])),
               "u_absmax": float(jnp.max(jnp.abs(carry.state.u))),
               "v_absmax": float(jnp.max(jnp.abs(carry.state.v))),
               "theta_min": float(jnp.min(carry.state.theta)),
               "theta_max": float(jnp.max(carry.state.theta))}
        hist.append(rec)
        if first_nonfinite is None and not np.isfinite(rec["w_absmax"]):
            first_nonfinite = s
            break
        if s <= 5 or s % 30 == 0:
            print(f"  [{label}] step {s:4d}: |w|={rec['w_absmax']:.2f} "
                  f"(top {rec['w_top_face']:.2f}) |u|={rec['u_absmax']:.2f} "
                  f"theta[{rec['theta_min']:.1f},{rec['theta_max']:.1f}]", flush=True)
    jax.block_until_ready(carry.state.u)
    f = hist[-1]
    stable = (first_nonfinite is None and f["w_absmax"] < 30.0 and f["u_absmax"] < 150.0
              and 150.0 <= f["theta_min"] and f["theta_max"] <= 550.0)
    return {"label": label, "top_lid": bool(nl.top_lid), "run_boundary": bool(nl.run_boundary),
            "epssm": nl.epssm, "steps": len(hist), "first_nonfinite": first_nonfinite,
            "step1_top_face": hist[0]["w_top_face"], "final": f,
            "verdict": "STABLE" if stable else "UNSTABLE", "history": hist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=360)
    args = ap.parse_args()
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    base = dataclasses.replace(case.namelist, run_physics=False, disable_guards=True)

    variants = {
        "open_top_with_bndy_eps0.5": dataclasses.replace(base, top_lid=False, run_boundary=True, epssm=0.5),
        "open_top_with_bndy_eps0.1": dataclasses.replace(base, top_lid=False, run_boundary=True, epssm=0.1),
    }
    out = {}
    for name, nl in variants.items():
        print(f"\n=== {name} (top_lid={nl.top_lid} run_boundary={nl.run_boundary} "
              f"epssm={nl.epssm}) ===", flush=True)
        out[name] = run(case, nl, args.steps, name)
        r = out[name]
        print(f"  VERDICT[{name}]: {r['verdict']} final |w|={r['final']['w_absmax']:.2f} "
              f"|u|={r['final']['u_absmax']:.2f} first_nonfinite={r['first_nonfinite']}", flush=True)

    (PROOF / "step5_opentop_bndy.json").write_text(json.dumps({"steps": args.steps, "variants": out}, indent=2) + "\n")
    print(f"\nwrote {PROOF / 'step5_opentop_bndy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
