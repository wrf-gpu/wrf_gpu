"""Step 1 (dycore-realinit): long-run guards-off dycore-only stability on real d02.

Resolve the Sprint-U-vs-recomp discrepancy. This runs the SAME init dycore-only
(physics off, boundaries off), disable_guards=True, with the EXACT validated
``_build_real_case`` operational namelist (top_lid=False open top, w_damping=1,
damp_opt=3, zdamp=5000, dampcoef=0.2, use_flux_advection, force_fp64,
diff_6th_opt=2, operational dt_s=10/acoustic_substeps=10), for a LONG run
(default 360 steps ~= 1 h at the operational dt).

Records per-step w_absmax / theta_min/max / u_absmax / v_absmax / ph_absmax /
mu_absmax AND, on the final state, WHERE (which k-level) the |w| and |u| maxima
live, so we can confirm/deny the recomp claim that the blow-up originates at the
top of the column (z=40/44).

This is the decisive Step-1 verdict:
  * STABLE for the full hour  -> NO dycore defect; recomp instability was its
    non-validated damping/dt; re-investigate the COUPLED path.
  * UNSTABLE with validated config -> real dycore defect on long real-d02 runs.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/dycore_realinit/step1_longrun.py --steps 360
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
from gpuwrf.runtime.operational_mode import (
    _enforce_operational_precision,
    _physics_boundary_step,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/dycore_realinit")

THETA_LO, THETA_HI = 150.0, 550.0
W_ABS = 50.0
UV_ABS = 150.0


def _extrema(state) -> dict:
    return {
        "theta_min": float(jnp.min(state.theta)),
        "theta_max": float(jnp.max(state.theta)),
        "w_absmax": float(jnp.max(jnp.abs(state.w))),
        "u_absmax": float(jnp.max(jnp.abs(state.u))),
        "v_absmax": float(jnp.max(jnp.abs(state.v))),
        "ph_absmax": float(jnp.max(jnp.abs(state.ph))),
        "mu_absmax": float(jnp.max(jnp.abs(state.mu))),
    }


def _origin_levels(state) -> dict:
    """k-level (and y,x) of the |w| and |u| maxima -- to confirm top-of-column."""
    w = np.asarray(jax.device_get(jnp.abs(state.w)))
    u = np.asarray(jax.device_get(jnp.abs(state.u)))
    wk, wy, wx = np.unravel_index(np.argmax(w), w.shape)
    uk, uy, ux = np.unravel_index(np.argmax(u), u.shape)
    return {
        "w_max_k": int(wk), "w_max_y": int(wy), "w_max_x": int(wx),
        "w_nz": int(w.shape[0]),
        "u_max_k": int(uk), "u_max_y": int(uy), "u_max_x": int(ux),
        "u_nz": int(u.shape[0]),
    }


def _finite(ext) -> bool:
    return all(np.isfinite(v) for v in ext.values())


def _unphysical(ext) -> bool:
    return (
        ext["theta_min"] < THETA_LO
        or ext["theta_max"] > THETA_HI
        or ext["w_absmax"] > W_ABS
        or ext["u_absmax"] > UV_ABS
        or not np.isfinite(ext["theta_max"])
        or not np.isfinite(ext["w_absmax"])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=360)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--acoustic", type=int, default=10)
    ap.add_argument("--out", type=str, default=str(PROOF / "step1_longrun.json"))
    # record per-step origin every N steps (cheap-ish device_get) plus on blow-up
    ap.add_argument("--origin-every", type=int, default=20)
    args = ap.parse_args()

    PROOF.mkdir(parents=True, exist_ok=True)
    print(f"[step1] jax devices: {jax.devices()}", flush=True)

    cfg = DailyPipelineConfig(hours=1, dt_s=args.dt, acoustic_substeps=args.acoustic)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=False, run_boundary=False, disable_guards=True
    )
    print(f"[step1] grid (nz,ny,nx)=({case.grid.nz},{case.grid.ny},{case.grid.nx}) "
          f"dt={args.dt} acoustic={args.acoustic} steps={args.steps}", flush=True)
    print(f"[step1] namelist: top_lid={nl.top_lid} w_damping={nl.w_damping} "
          f"damp_opt={nl.damp_opt} zdamp={nl.zdamp} dampcoef={nl.dampcoef} "
          f"diff_6th_opt={nl.diff_6th_opt} use_flux_advection={nl.use_flux_advection} "
          f"force_fp64={nl.force_fp64} disable_guards={nl.disable_guards}", flush=True)

    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=True)
    )
    init_ext = _extrema(carry.state)
    init_origin = _origin_levels(carry.state)
    print(f"[step1] init extrema: {init_ext}", flush=True)
    print(f"[step1] init origin:  {init_origin}", flush=True)

    @jax.jit
    def _step(c, idx):
        return _physics_boundary_step(c, nl, idx, run_radiation=False, debug=False)

    history = []
    first_unphysical = None
    first_nonfinite = None
    blowup_origin = None
    for s in range(1, args.steps + 1):
        carry = _step(carry, jnp.asarray(s, dtype=jnp.int32))
        ext = _extrema(carry.state)
        rec = {"step": s, **ext}
        finite = _finite(ext)
        unphys = _unphysical(ext)
        if (s % args.origin_every == 0) or unphys or not finite:
            rec["origin"] = _origin_levels(carry.state)
        history.append(rec)
        if first_nonfinite is None and not finite:
            first_nonfinite = s
            blowup_origin = rec.get("origin")
        if first_unphysical is None and unphys:
            first_unphysical = s
            if blowup_origin is None:
                blowup_origin = rec.get("origin")
        if (s <= 10) or (s % 20 == 0) or unphys or not finite:
            print(f"[step1] step {s:4d}: theta[{ext['theta_min']:.1f},{ext['theta_max']:.1f}] "
                  f"|w|={ext['w_absmax']:.3f} |u|={ext['u_absmax']:.3f} "
                  f"|v|={ext['v_absmax']:.3f}"
                  + (f"  origin={rec.get('origin')}" if "origin" in rec else ""), flush=True)
        if not finite:
            print(f"[step1] NON-FINITE at step {s} -- stopping", flush=True)
            break

    final = history[-1]
    final_finite = _finite({k: v for k, v in final.items() if k not in ("step", "origin")})
    stable = (
        first_nonfinite is None
        and final["w_absmax"] < 30.0
        and final["theta_min"] >= THETA_LO
        and final["theta_max"] <= THETA_HI
        and final["u_absmax"] < UV_ABS
    )
    verdict = "STABLE" if stable else "UNSTABLE"

    payload = {
        "schema": "dycore_realinit_step1_longrun",
        "schema_version": 1,
        "objective": (
            "Long-run guards-off dycore-only stability on real d02 with the EXACT "
            "validated _build_real_case operational namelist."
        ),
        "run_dir": str(run_dir),
        "config": {
            "dt_s": args.dt,
            "acoustic_substeps": args.acoustic,
            "steps": args.steps,
            "top_lid": bool(nl.top_lid),
            "w_damping": int(nl.w_damping),
            "damp_opt": int(nl.damp_opt),
            "zdamp": float(nl.zdamp),
            "dampcoef": float(nl.dampcoef),
            "diff_6th_opt": int(nl.diff_6th_opt),
            "diff_6th_factor": float(nl.diff_6th_factor),
            "use_flux_advection": bool(nl.use_flux_advection),
            "force_fp64": bool(nl.force_fp64),
            "disable_guards": bool(nl.disable_guards),
            "run_physics": bool(nl.run_physics),
            "run_boundary": bool(nl.run_boundary),
            "epssm": float(nl.epssm),
            "rk_order": int(nl.rk_order),
        },
        "grid": {"nz": int(case.grid.nz), "ny": int(case.grid.ny), "nx": int(case.grid.nx)},
        "init_extrema": init_ext,
        "init_origin": init_origin,
        "steps_run": len(history),
        "first_unphysical_step": first_unphysical,
        "first_nonfinite_step": first_nonfinite,
        "blowup_origin": blowup_origin,
        "final": final,
        "final_all_finite": bool(final_finite),
        "verdict": verdict,
        "history": history,
        "cpu_affinity": sorted(int(c) for c in os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "na",
        "device": str(jax.devices()[0]),
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[step1] VERDICT: {verdict} "
          f"(steps_run={len(history)}, first_unphysical={first_unphysical}, "
          f"first_nonfinite={first_nonfinite})", flush=True)
    print(f"[step1] final: {final}", flush=True)
    print(f"[step1] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
