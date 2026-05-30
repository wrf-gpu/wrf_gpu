"""Step 6 (ACCEPTANCE): the FIX wired into _build_real_case, dycore-only,
guards-off, >=360 steps on the real d02 init.

This does NOT override the namelist -- it uses _build_real_case exactly as the
operational pipeline does, so it proves the committed fix (epssm=0.5, top_lid=True)
is wired correctly. Dycore-only (run_physics=False) keeps the real lateral
boundaries (run_boundary stays True, the operational default) so this is the
faithful LAM dry-dynamics configuration. disable_guards=True so the guards are
proven NOT load-bearing.

Goal: STABLE for >=360 steps (~1 h) -- finite, |w|<30, theta in [150,550],
|u|<150 -- which is the DYCORE_REALINIT_STABLE gate.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step6_accept.py --steps 360
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _enforce_operational_precision, _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/dycore_realinit")
THETA_LO, THETA_HI, W_ABS, UV_ABS = 150.0, 550.0, 30.0, 150.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=360)
    args = ap.parse_args()
    PROOF.mkdir(parents=True, exist_ok=True)

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    # dycore-only: physics off, guards off. run_boundary stays at the operational
    # default (True) -- the real LAM lateral boundaries are part of the dycore-only
    # dry dynamics for a limited-area domain.
    nl = dataclasses.replace(case.namelist, run_physics=False, disable_guards=True)
    assert bool(nl.top_lid) is True, "fix not wired: top_lid must be True"
    assert abs(float(nl.epssm) - 0.5) < 1e-9, "fix not wired: epssm must be 0.5"
    assert bool(nl.run_boundary) is True, "real LAM must keep lateral boundaries"

    carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

    @jax.jit
    def _step(c, idx):
        return _physics_boundary_step(c, nl, idx, run_radiation=False, debug=False)

    def dtypes(state):
        return {n: str(np.asarray(jax.device_get(getattr(state, n))).dtype) for n in ("u", "v", "w", "theta", "ph", "mu")}

    init_dtypes = dtypes(carry.state)
    hist = []
    first_nonfinite = first_unphysical = None
    for s in range(1, args.steps + 1):
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32))
        wabs = jnp.abs(carry.state.w)
        rec = {"step": s, "w_absmax": float(jnp.max(wabs)),
               "w_top_face": float(jnp.max(wabs[-1])),
               "u_absmax": float(jnp.max(jnp.abs(carry.state.u))),
               "v_absmax": float(jnp.max(jnp.abs(carry.state.v))),
               "theta_min": float(jnp.min(carry.state.theta)),
               "theta_max": float(jnp.max(carry.state.theta))}
        if s % 20 == 0 or s <= 3:
            w = np.asarray(jax.device_get(wabs))
            k, y, x = np.unravel_index(np.argmax(w), w.shape)
            rec["w_max_k"] = int(k)
        hist.append(rec)
        finite = all(np.isfinite(v) for v in (rec["w_absmax"], rec["u_absmax"], rec["theta_max"]))
        unphys = (rec["theta_min"] < THETA_LO or rec["theta_max"] > THETA_HI
                  or rec["w_absmax"] > W_ABS or rec["u_absmax"] > UV_ABS or not finite)
        if first_nonfinite is None and not finite:
            first_nonfinite = s
        if first_unphysical is None and unphys:
            first_unphysical = s
        if s <= 5 or s % 30 == 0:
            print(f"  step {s:4d}: |w|={rec['w_absmax']:.2f} (top {rec['w_top_face']:.3f}) "
                  f"|u|={rec['u_absmax']:.2f} |v|={rec['v_absmax']:.2f} "
                  f"theta[{rec['theta_min']:.1f},{rec['theta_max']:.1f}]", flush=True)
        if not finite:
            break

    jax.block_until_ready(carry.state.u)
    final_dtypes = dtypes(carry.state)
    all_fp64 = all(v == "float64" for v in final_dtypes.values())
    f = hist[-1]
    stable = (first_nonfinite is None and f["w_absmax"] < W_ABS and f["u_absmax"] < UV_ABS
              and THETA_LO <= f["theta_min"] and f["theta_max"] <= THETA_HI and len(hist) >= args.steps)
    verdict = "STABLE" if (stable and all_fp64) else "UNSTABLE"

    payload = {
        "schema": "dycore_realinit_step6_accept", "schema_version": 1,
        "objective": "FIX wired in _build_real_case; dycore-only guards-off >=360-step stability on real d02.",
        "run_dir": str(run_dir),
        "config": {"dt_s": float(nl.dt_s), "acoustic_substeps": int(nl.acoustic_substeps),
                   "epssm": float(nl.epssm), "top_lid": bool(nl.top_lid),
                   "run_boundary": bool(nl.run_boundary), "run_physics": bool(nl.run_physics),
                   "disable_guards": bool(nl.disable_guards), "force_fp64": bool(nl.force_fp64),
                   "w_damping": int(nl.w_damping), "damp_opt": int(nl.damp_opt),
                   "zdamp": float(nl.zdamp), "dampcoef": float(nl.dampcoef),
                   "use_flux_advection": bool(nl.use_flux_advection), "diff_6th_opt": int(nl.diff_6th_opt)},
        "grid": {"nz": int(case.grid.nz), "ny": int(case.grid.ny), "nx": int(case.grid.nx)},
        "steps_run": len(hist), "first_unphysical_step": first_unphysical,
        "first_nonfinite_step": first_nonfinite,
        "init_dtypes": init_dtypes, "final_dtypes": final_dtypes, "all_prognostics_fp64": bool(all_fp64),
        "final": f, "verdict": verdict, "history": hist,
        "cpu_affinity": sorted(int(c) for c in os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "na",
        "device": str(jax.devices()[0]),
    }
    (PROOF / "step6_accept.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nVERDICT: {verdict} (steps_run={len(hist)}, first_unphysical={first_unphysical}, "
          f"first_nonfinite={first_nonfinite}, all_fp64={all_fp64})")
    print(f"final: |w|={f['w_absmax']:.2f} |u|={f['u_absmax']:.2f} |v|={f['v_absmax']:.2f} "
          f"theta[{f['theta_min']:.1f},{f['theta_max']:.1f}]")
    print(f"wrote {PROOF / 'step6_accept.json'}")
    return 0 if verdict == "STABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
