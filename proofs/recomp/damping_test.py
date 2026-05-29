"""Test whether WRF-standard damping (Rayleigh-w + smdiv divergence) controls
the dry-dycore upper-level w instability on the real d02 case.

Runs dycore-only guards-OFF with several damping configs and reports whether
|w| stays inside the 50 m/s physical envelope across N steps. This isolates
whether the instability is a MISSING-DAMPING bug (WRF real cases always run
smdiv + w_damping/Rayleigh) vs a structural dycore defect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import gpuwrf.integration.d02_replay as dr
from gpuwrf.integration.d02_replay import (
    DEFAULT_REPLAY_RUN_DIR,
    ReplayConfig,
    build_replay_case,
)
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig


def make_acoustic_config_factory(smdiv_coef, rayleigh_coef, rayleigh_top_frac):
    def _factory(grid, n_acoustic, rayleigh_coefficient):
        return AcousticConfig(
            n_substeps=int(n_acoustic),
            dx_m=float(grid.projection.dx_m),
            dy_m=float(grid.projection.dy_m),
            non_hydrostatic=True,
            top_lid=True,
            mu_continuity=True,
            epssm=0.1,
            smdiv=SmdivConfig(enabled=smdiv_coef > 0.0, coefficient=float(smdiv_coef)),
            rayleigh=RayleighConfig(
                enabled=rayleigh_coef > 0.0,
                coefficient=float(rayleigh_coef),
                top_start_fraction=float(rayleigh_top_frac),
            ),
        )
    return _factory


def run_dycore(case, cfg, steps):
    state = case.state
    pp = case.previous_pressure
    hist = []
    first_unphysical = None
    first_nonfinite = None
    for i in range(steps):
        state, pp = dr._dycore_step_adr023(
            state, pp, case.tendencies, case.grid, case.metrics, case.base_state, cfg
        )
        wmax = float(jnp.max(jnp.abs(state.w)))
        tmin = float(jnp.min(state.theta)); tmax = float(jnp.max(state.theta))
        umax = float(jnp.max(jnp.abs(state.u)))
        hist.append({"step": i + 1, "w_absmax": wmax, "theta_min": tmin,
                     "theta_max": tmax, "u_absmax": umax})
        finite = np.isfinite(wmax) and np.isfinite(tmax)
        if first_nonfinite is None and not finite:
            first_nonfinite = i + 1
        if first_unphysical is None and (wmax > 50.0 or tmin < 150.0 or tmax > 550.0 or umax > 150.0 or not finite):
            first_unphysical = i + 1
        if not finite:
            break
    return hist, first_unphysical, first_nonfinite


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--n-acoustic", type=int, default=4)
    ap.add_argument("--out", type=str, default="proofs/recomp/damping_test.json")
    args = ap.parse_args()

    case = build_replay_case(DEFAULT_REPLAY_RUN_DIR, domain="d02")
    cfg = ReplayConfig(dt_s=args.dt, duration_s=args.dt * args.steps, n_acoustic=args.n_acoustic)

    # WRF defaults: smdiv=0.1 (module_big_step_utilities / namelist dynamics),
    # w_damping Rayleigh top layer. We sweep a few coefficients.
    configs = [
        ("baseline_nodamp", 0.0, 0.0, 0.75),
        ("smdiv_0.1", 0.1, 0.0, 0.75),
        ("rayleigh_0.2", 0.0, 0.2, 0.75),
        ("smdiv0.1_rayleigh0.2", 0.1, 0.2, 0.75),
        ("smdiv0.1_rayleigh1.0", 0.1, 1.0, 0.75),
        ("smdiv0.1_rayleigh5.0_top0.66", 0.1, 5.0, 0.66),
    ]
    orig = dr._acoustic_config
    results = []
    try:
        for name, sm, ray, topf in configs:
            dr._acoustic_config = make_acoustic_config_factory(sm, ray, topf)
            hist, fu, fn = run_dycore(case, cfg, args.steps)
            final = hist[-1]
            stable = (fu is None) and (fn is None)
            print(f"[damp] {name:30s}: steps={len(hist)} first_unphysical={fu} "
                  f"first_nonfinite={fn} final |w|={final['w_absmax']:.2f} "
                  f"theta[{final['theta_min']:.1f},{final['theta_max']:.1f}] "
                  f"=> {'STABLE' if stable else 'UNSTABLE'}", flush=True)
            results.append({"config": name, "smdiv": sm, "rayleigh": ray,
                            "rayleigh_top_frac": topf, "steps_run": len(hist),
                            "first_unphysical_step": fu, "first_nonfinite_step": fn,
                            "stable_in_envelope": stable, "final": final,
                            "history": hist})
    finally:
        dr._acoustic_config = orig

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dt_s": args.dt, "n_acoustic": args.n_acoustic,
                               "steps": args.steps, "configs": results}, indent=2))
    print(f"[damp] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
