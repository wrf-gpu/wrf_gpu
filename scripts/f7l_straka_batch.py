"""F7L Straka stability batch: compare diffusion/damping/CFL settings.

Runs the density-current case to a target time under several WRF-faithful
configurations and logs per-interval max|w|, min θ', front position.  Used to
(a) confirm ν=75 diffusion is actually active (vs ν=0) and (b) find the
WRF-faithful settings that keep Straka finite to 900 s.

Unbuffered (`python -u`) so each interval line flushes immediately.
"""

from __future__ import annotations

import argparse
import sys

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from dataclasses import replace as dataclass_replace

from gpuwrf.ic_generators.idealized import (
    build_density_current_numpy,
    _build_setup,
    _initial_carry,
    _ready_carry,
    _run_segment,
    _snapshot,
)


def run_config(label, *, end, interval, overrides, dt=None):
    case = build_density_current_numpy()
    if dt is not None:
        case = dataclass_replace(case, dt_s=float(dt))
    setup = _build_setup(case, require_gpu=True)
    nl = setup.namelist
    if overrides:
        nl = dataclass_replace(nl, **overrides)
    print(f"\n=== CONFIG [{label}] nu={nl.const_nu_m2_s} substeps={nl.acoustic_substeps} "
          f"dt={case.dt_s} damp_opt={nl.damp_opt} dampcoef={nl.dampcoef} zdamp={nl.zdamp} "
          f"w_damping={nl.w_damping} ===", flush=True)
    carry = _initial_carry(setup.state)
    _ready_carry(carry).block_until_ready()
    prev = 0
    t = 0.0
    last_finite = 0.0
    while t < end - 1e-9:
        t_next = min(t + interval, end)
        target = int(round(t_next / case.dt_s))
        carry = _run_segment(carry, nl, start_step=prev + 1, steps=target - prev)
        prev = target
        snap = _snapshot(case, carry.state, float(t_next))
        print(f"[{label}] t={t_next:7.1f} finite={snap['finite']} "
              f"maxw={snap['max_abs_w_m_s']} thmin={snap['theta_prime_min_k']} "
              f"front={snap.get('front_position_m')}", flush=True)
        if not snap["finite"]:
            print(f"[{label}] FIRST_NONFINITE_AT_t={t_next}", flush=True)
            break
        last_finite = t_next
        t = t_next
    else:
        print(f"[{label}] REACHED_END finite_to={last_finite}", flush=True)
    return last_finite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=float, default=300.0)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--configs", type=str, default="nu0,nu75")
    args = p.parse_args()

    cfgs = {
        "nu0": dict(overrides={"const_nu_m2_s": 0.0}),
        "nu75": dict(overrides={"const_nu_m2_s": 75.0}),
        "nu150": dict(overrides={"const_nu_m2_s": 150.0}),
        "nu75_ss20": dict(overrides={"const_nu_m2_s": 75.0, "acoustic_substeps": 20}),
        "nu75_dt0.05": dict(overrides={"const_nu_m2_s": 75.0}, dt=0.05),
        "nu75_smdiv_hi": dict(overrides={"const_nu_m2_s": 75.0}),  # smdiv handled separately
    }
    for label in args.configs.split(","):
        label = label.strip()
        if label not in cfgs:
            print(f"unknown config {label}", file=sys.stderr)
            continue
        spec = cfgs[label]
        run_config(label, end=args.end, interval=args.interval,
                   overrides=spec.get("overrides"), dt=spec.get("dt"))


if __name__ == "__main__":
    main()
