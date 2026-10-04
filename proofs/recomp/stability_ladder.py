"""Incremental-coupling stability ladder for the coupled real-case d02 forecast.

Builds the pinned Gen2 d02 real case ONCE, then runs a configurable per-step
coupler with the sanitiser guards OFF and records where theta/w/u first leave
physical bounds. Each rung adds one coupling component on top of the dry dycore:

    dycore                      (known stable ~50+ steps guards-off)
    dycore + lateral boundaries
    dycore + boundaries + surface
    dycore + boundaries + surface + MYNN
    dycore + boundaries + surface + MYNN + radiation

The point is to ISOLATE which component triggers the blow-up B4 saw (the
sanitiser pinning theta/w/u within ~30 steps). We run guards-OFF so the raw
instability is visible, and report per-step field extrema. A field is flagged
"unphysical" when theta leaves [150,550] K, |w|>50 m/s, or |u|>150 m/s -- the
same envelope the production sanitiser clips to, so "first unphysical step"
matches "first step the guard would engage".

Usage:
    PYTHONPATH=src python proofs/recomp/stability_ladder.py [--steps N] [--rungs a,b,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gpuwrf.integration.d02_replay as dr
from gpuwrf.integration.d02_replay import (
    DEFAULT_REPLAY_RUN_DIR,
    ReplayConfig,
    _dycore_step_adr023,
    build_replay_case,
)
from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import (
    mynn_adapter,
    rrtmg_adapter,
    surface_adapter,
    thompson_adapter,
)
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig
from gpuwrf.dynamics.damping import RayleighConfig, SmdivConfig


def install_damping(smdiv_coef, rayleigh_coef, rayleigh_top_frac):
    """Monkeypatch d02_replay._acoustic_config to enable WRF-standard damping.

    The dry dycore is unstable guards-off on the real case without upper-level
    Rayleigh w-damping; this hook lets the ladder run each rung on a stabilised
    dynamical core to isolate whether the PHYSICS adds any further instability.
    """
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
    dr._acoustic_config = _factory

THETA_LO, THETA_HI = 150.0, 550.0
W_ABS = 50.0
UV_ABS = 150.0


def _step(state, prev_pp, case, cfg, global_step, rung, *, order_surface_first=True):
    """One coupled timestep, guards OFF, with the components selected by `rung`."""
    next_state, next_pp = _dycore_step_adr023(
        state, prev_pp, case.tendencies, case.grid, case.metrics, case.base_state, cfg
    )
    dt = float(cfg.dt_s)
    if "surface" in rung and order_surface_first:
        next_state = surface_adapter(next_state, dt)
    if "mynn" in rung:
        next_state = mynn_adapter(next_state, dt, case.grid)
    if "surface" in rung and not order_surface_first:
        next_state = surface_adapter(next_state, dt)
    if "radiation" in rung:
        next_state = rrtmg_adapter(next_state, dt, case.grid)
    if "boundary" in rung:
        lead = global_step.astype(jnp.float64) * dt
        next_state = apply_lateral_boundaries(next_state, lead, dt, cfg.boundary_config)
    return next_state, next_pp


def _extrema(state):
    return {
        "theta_min": float(jnp.min(state.theta)),
        "theta_max": float(jnp.max(state.theta)),
        "w_absmax": float(jnp.max(jnp.abs(state.w))),
        "u_absmax": float(jnp.max(jnp.abs(state.u))),
        "v_absmax": float(jnp.max(jnp.abs(state.v))),
        "qv_min": float(jnp.min(state.qv)),
        "qv_max": float(jnp.max(state.qv)),
    }


def _unphysical(ext):
    return (
        ext["theta_min"] < THETA_LO
        or ext["theta_max"] > THETA_HI
        or ext["w_absmax"] > W_ABS
        or ext["u_absmax"] > UV_ABS
        or not np.isfinite(ext["theta_max"])
        or not np.isfinite(ext["w_absmax"])
    )


def run_rung(case, cfg, rung, steps, *, order_surface_first=True):
    state = case.state
    pp = case.previous_pressure
    history = []
    first_unphysical = None
    first_nonfinite = None
    for i in range(steps):
        gstep = jnp.asarray(i, dtype=jnp.int32)
        state, pp = _step(state, pp, case, cfg, gstep, rung, order_surface_first=order_surface_first)
        ext = _extrema(state)
        history.append({"step": i + 1, **ext})
        finite = all(np.isfinite(v) for v in ext.values())
        if first_nonfinite is None and not finite:
            first_nonfinite = i + 1
        if first_unphysical is None and _unphysical(ext):
            first_unphysical = i + 1
        if not finite:
            break
    return {
        "rung": "+".join(rung) if rung else "dycore",
        "components": rung,
        "order_surface_first": order_surface_first,
        "steps_requested": steps,
        "steps_run": len(history),
        "first_unphysical_step": first_unphysical,
        "first_nonfinite_step": first_nonfinite,
        "final": history[-1] if history else None,
        "history": history,
    }


RUNGS = {
    "dycore": [],
    "boundary": ["boundary"],
    "surface": ["boundary", "surface"],
    "mynn": ["boundary", "surface", "mynn"],
    "radiation": ["boundary", "surface", "mynn", "radiation"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--n-acoustic", type=int, default=4)
    ap.add_argument("--run-dir", type=str, default=str(DEFAULT_REPLAY_RUN_DIR))
    ap.add_argument("--rungs", type=str, default="dycore,boundary,surface,mynn,radiation")
    ap.add_argument("--order-mynn-first", action="store_true",
                    help="run mynn before surface (the d02_replay legacy order)")
    ap.add_argument("--smdiv", type=float, default=0.0)
    ap.add_argument("--rayleigh", type=float, default=0.0)
    ap.add_argument("--rayleigh-top-frac", type=float, default=0.75)
    ap.add_argument("--out", type=str, default="proofs/recomp/stability_ladder.json")
    args = ap.parse_args()

    if args.smdiv > 0.0 or args.rayleigh > 0.0:
        install_damping(args.smdiv, args.rayleigh, args.rayleigh_top_frac)
        print(f"[ladder] damping installed: smdiv={args.smdiv} rayleigh={args.rayleigh} "
              f"top_frac={args.rayleigh_top_frac}", flush=True)

    print(f"[ladder] jax devices: {jax.devices()}", flush=True)
    print(f"[ladder] building real case from {args.run_dir} ...", flush=True)
    case = build_replay_case(args.run_dir, domain="d02")
    grid = case.grid
    print(f"[ladder] grid mass_shape=({grid.nz},{grid.ny},{grid.nx}) dt={args.dt}s", flush=True)
    init_ext = _extrema(case.state)
    print(f"[ladder] init extrema: {init_ext}", flush=True)

    cfg = ReplayConfig(dt_s=args.dt, duration_s=args.dt * args.steps, n_acoustic=args.n_acoustic)

    results = {"run_dir": args.run_dir, "dt_s": args.dt, "n_acoustic": args.n_acoustic,
               "steps": args.steps, "init_extrema": init_ext,
               "damping": {"smdiv": args.smdiv, "rayleigh": args.rayleigh,
                           "rayleigh_top_frac": args.rayleigh_top_frac},
               "rungs": []}
    for name in args.rungs.split(","):
        name = name.strip()
        if name not in RUNGS:
            print(f"[ladder] skipping unknown rung {name}", flush=True)
            continue
        print(f"\n[ladder] === rung: {name} ({RUNGS[name]}) ===", flush=True)
        res = run_rung(case, cfg, RUNGS[name], args.steps,
                       order_surface_first=not args.order_mynn_first)
        results["rungs"].append(res)
        print(f"[ladder]   steps_run={res['steps_run']} "
              f"first_unphysical={res['first_unphysical_step']} "
              f"first_nonfinite={res['first_nonfinite_step']}", flush=True)
        if res["final"]:
            f = res["final"]
            print(f"[ladder]   final: theta[{f['theta_min']:.1f},{f['theta_max']:.1f}] "
                  f"|w|={f['w_absmax']:.2f} |u|={f['u_absmax']:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[ladder] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
