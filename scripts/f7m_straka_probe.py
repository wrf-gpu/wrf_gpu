"""F7M Straka probe — single-compile run to checkpoints (front-region diagnostics).

Unlike f7l_straka_probe (which re-jits each segment on changing start_step and is
slow with the heavier flux-advection graph), this runs ONE jitted fixed-length
scan per checkpoint stride and reuses the compiled function, then reports
max|w|, theta'min, front position, and low-level outflow u at each checkpoint.
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
    _run_segment_jit,
    _snapshot,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--end", type=float, default=300.0)
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    case = build_density_current_numpy()
    setup = _build_setup(case, require_gpu=True)
    nl = setup.namelist
    stride = int(round(args.interval / case.dt_s))  # same step count every segment -> 1 compile

    carry = _initial_carry(setup.state)
    _ready_carry(carry).block_until_ready()
    prev = 0
    t = 0.0
    rows = []
    while t < args.end - 1e-9:
        # fixed-length segment of `stride` steps.  Radiation is off (cadence
        # 999999), so the absolute step index does not affect the dry dynamics;
        # use a constant start_step=1 so the jit compiles ONCE and is reused
        # across all checkpoints (the f7l probe re-jit per segment was the slow
        # path under the heavier flux-advection graph).
        carry = _run_segment_jit(carry, nl, start_step=1, steps=stride)
        _ready_carry(carry).block_until_ready()
        prev += stride
        t = prev * case.dt_s
        snap = _snapshot(case, carry.state, float(t))
        # low-level outflow u (z<=800m), max|u|
        u_mass = np.asarray(snap["u_mass_m_s"], dtype=np.float64)
        zlow = case.z_m <= 800.0
        u_outflow = float(np.max(np.abs(u_mass[zlow, :]))) if np.any(zlow) else None
        row = {
            "t": float(t),
            "finite": bool(snap["finite"]),
            "maxw": snap["max_abs_w_m_s"],
            "thmin": snap["theta_prime_min_k"],
            "front": snap.get("front_position_m"),
            "u_outflow": u_outflow,
            "max_abs_u": snap.get("max_abs_u_m_s"),
        }
        rows.append(row)
        print(f"t={t:7.1f} finite={row['finite']} maxw={row['maxw']} thmin={row['thmin']} "
              f"front={row['front']} u_outflow={u_outflow}", flush=True)
        if not row["finite"]:
            break
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"case": "straka_flux_advection", "dt_s": case.dt_s, "rows": rows}, fh, indent=2)
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
