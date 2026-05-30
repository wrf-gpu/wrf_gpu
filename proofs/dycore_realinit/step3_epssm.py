"""Step 3: the real Gen2 d02 namelist.input sets epssm=0.5, but the JAX
operational config (_build_real_case) leaves epssm at the 0.1 default.

epssm is the vertically-implicit acoustic off-centering coefficient
(module_small_step_em.F): eps_m = 1-epssm weights the EXPLICIT old-time
pressure/w in the top-face w update and the phi RHS. With epssm=0.1 (eps_m=0.9)
the open-top w solve is nearly centered and weakly damped -> the 307 m/s
top-face spike; the real run uses epssm=0.5 (eps_m=0.5), strongly off-centered
and damped.

Test: dycore-only (run_boundary=False), disable_guards=True, validated config,
epssm in {0.1 (baseline), 0.5 (WRF namelist)}, long run, open top (top_lid=False
faithful to the real LAM open top). Report per-step extrema + top-face origin.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step3_epssm.py --steps 360
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
        w = jnp.abs(carry.state.w)
        u = jnp.abs(carry.state.u)
        rec = {
            "step": s,
            "w_absmax": float(jnp.max(w)),
            "w_top_face": float(jnp.max(w[-1])),
            "u_absmax": float(jnp.max(u)),
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
                  f"theta[{rec['theta_min']:.1f},{rec['theta_max']:.1f}]", flush=True)
    jax.block_until_ready(carry.state.u)
    f = hist[-1]
    stable = (
        first_nonfinite is None
        and f["w_absmax"] < 30.0
        and f["u_absmax"] < 150.0
        and 150.0 <= f["theta_min"] and f["theta_max"] <= 550.0
    )
    return {"label": label, "epssm": nl.epssm, "steps": len(hist),
            "first_nonfinite": first_nonfinite, "step1_w": hist[0]["w_absmax"],
            "step1_top_face": hist[0]["w_top_face"], "final": f,
            "verdict": "STABLE" if stable else "UNSTABLE", "history": hist}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=360)
    args = ap.parse_args()
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    base = dataclasses.replace(case.namelist, run_physics=False, run_boundary=False, disable_guards=True)

    out = {}
    for eps in (0.1, 0.5):
        label = f"epssm={eps}"
        print(f"\n=== {label} (open top, dycore-only, {args.steps} steps) ===", flush=True)
        nl = dataclasses.replace(base, epssm=float(eps))
        out[label] = run(case, nl, args.steps, label)
        print(f"  VERDICT[{label}]: {out[label]['verdict']} step1_top={out[label]['step1_top_face']:.1f} "
              f"final |w|={out[label]['final']['w_absmax']:.2f} |u|={out[label]['final']['u_absmax']:.2f}", flush=True)

    (PROOF / "step3_epssm.json").write_text(json.dumps({"steps": args.steps, "variants": out}, indent=2) + "\n")
    print(f"\nwrote {PROOF / 'step3_epssm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
