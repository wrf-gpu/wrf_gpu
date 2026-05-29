"""F7L Straka stability probe: per-interval max|w|, CFL, diffusion-active audit.

Runs the density-current case in fine time intervals and reports max|w|, min θ',
and (optionally) the acoustic CFL.  Used to characterize the 240 s detonation and
to test WRF-faithful stabilizers (ν=75, acoustic substeps, divergence damping).
"""

from __future__ import annotations

import argparse
import json
import os

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import (
    build_density_current_numpy,
    _build_setup,
    _initial_carry,
    _ready_carry,
    _run_segment,
    _snapshot,
)
from dataclasses import replace as dataclass_replace


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=float, default=300.0)
    p.add_argument("--interval", type=float, default=20.0)
    p.add_argument("--nu", type=float, default=None, help="override const_nu_m2_s")
    p.add_argument("--substeps", type=int, default=None, help="override acoustic_substeps")
    p.add_argument("--dt", type=float, default=None, help="override dt_s")
    p.add_argument("--smdiv", type=float, default=None)
    p.add_argument("--emdiv", type=float, default=None)
    args = p.parse_args()

    case = build_density_current_numpy()
    if args.dt is not None:
        case = dataclass_replace(case, dt_s=float(args.dt))
    setup = _build_setup(case, require_gpu=True)
    nl = setup.namelist
    over = {}
    if args.nu is not None:
        over["const_nu_m2_s"] = float(args.nu)
    if args.substeps is not None:
        over["acoustic_substeps"] = int(args.substeps)
    if args.smdiv is not None:
        over["smdiv"] = float(args.smdiv)
    if args.emdiv is not None:
        over["emdiv"] = float(args.emdiv)
    if over:
        nl = dataclass_replace(nl, **over)

    print(f"DEVICE={setup.device}  nu={nl.const_nu_m2_s}  substeps={nl.acoustic_substeps}  "
          f"dt={case.dt_s}  smdiv={getattr(nl,'smdiv',None)}  emdiv={getattr(nl,'emdiv',None)}  "
          f"damp_opt={nl.damp_opt} dampcoef={nl.dampcoef} zdamp={nl.zdamp} w_damping={nl.w_damping}")
    # acoustic CFL estimate
    c_sound = 340.0
    dt_sound = case.dt_s / float(nl.acoustic_substeps)
    print(f"acoustic CFL = c*dt_sound/dx = {c_sound*dt_sound/case.dx_m:.4f}  (dt_sound={dt_sound:.4f}s)")

    carry = _initial_carry(setup.state)
    _ready_carry(carry).block_until_ready()
    prev = 0
    t = 0.0
    rows = []
    while t < args.end - 1e-9:
        t_next = min(t + args.interval, args.end)
        target = int(round(t_next / case.dt_s))
        carry = _run_segment(carry, nl, start_step=prev + 1, steps=target - prev)
        prev = target
        snap = _snapshot(case, carry.state, float(t_next))
        finite = snap["finite"]
        maxw = snap["max_abs_w_m_s"]
        thmin = snap["theta_prime_min_k"]
        front = snap.get("front_position_m")
        rows.append({"t": t_next, "finite": finite, "maxw": maxw, "thmin": thmin, "front": front})
        print(f"t={t_next:7.1f} finite={finite} maxw={maxw} thmin={thmin} front={front}")
        if not finite:
            break
        t = t_next
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
