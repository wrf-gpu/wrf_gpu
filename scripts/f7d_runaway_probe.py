#!/usr/bin/env python
"""F7D diagnostic: trace max|w|, theta' bounds, and finiteness of the
idealized cases on a fine cadence through the historical 80-100 s runaway
window.  No clamps, no masking -- pure observation of the operational dycore.

Run as:  taskset -c 0-3 python scripts/f7d_runaway_probe.py --case warm_bubble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _initial_carry,
    _ready_carry,
    _run_segment,
    build_density_current_numpy,
    build_warm_bubble_numpy,
)


def _stats(carry) -> dict:
    state = carry.state
    w = np.asarray(jax.device_get(state.w[:, 0, :]), dtype=np.float64)
    u = np.asarray(jax.device_get(state.u[:, 0, :]), dtype=np.float64)
    theta = np.asarray(jax.device_get(state.theta[:, 0, :]), dtype=np.float64)
    p = np.asarray(jax.device_get(state.p_perturbation[:, 0, :]), dtype=np.float64)
    mu = np.asarray(jax.device_get(state.mu_total[0, :]), dtype=np.float64)
    finite = bool(
        np.all(np.isfinite(w))
        and np.all(np.isfinite(u))
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(p))
        and np.all(np.isfinite(mu))
    )
    return {
        "finite": finite,
        "max_abs_w": float(np.max(np.abs(w))) if finite else None,
        "max_abs_u": float(np.max(np.abs(u))) if finite else None,
        "theta_prime_min": float(np.min(theta) - 300.0) if finite else None,
        "theta_prime_max": float(np.max(theta) - 300.0) if finite else None,
        "max_abs_p_pert": float(np.max(np.abs(p))) if finite else None,
        "mass_total_pa": float(np.sum(mu)) if finite else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("warm_bubble", "density_current"), required=True)
    parser.add_argument("--end-seconds", type=float, default=200.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy() if args.case == "warm_bubble" else build_density_current_numpy()
    setup = _build_setup(case, require_gpu=True)
    carry = _initial_carry(setup.state)
    _ready_carry(carry).block_until_ready()

    dt = float(case.dt_s)
    stride_steps = max(1, int(round(args.stride_seconds / dt)))
    n_marks = int(round(args.end_seconds / args.stride_seconds))

    trace = [{"second": 0.0, **_stats(carry)}]
    print(f"case={args.case} dt={dt} device={setup.device}")
    print(f"  t=0.0  {trace[0]}")
    step = 0
    for m in range(n_marks):
        carry = _run_segment(carry, setup.namelist, start_step=step + 1, steps=stride_steps)
        step += stride_steps
        second = step * dt
        rec = {"second": round(second, 3), **_stats(carry)}
        trace.append(rec)
        print(f"  t={second:7.1f}  finite={rec['finite']}  max|w|={rec['max_abs_w']}  "
              f"theta'=[{rec['theta_prime_min']},{rec['theta_prime_max']}]  max|p'|={rec['max_abs_p_pert']}")
        if not rec["finite"]:
            print("  -> NON-FINITE; stopping")
            break

    out = args.output or Path("proofs/f7d") / f"runaway_probe_{args.case}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case": args.case, "dt_s": dt, "device": setup.device, "trace": trace}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
