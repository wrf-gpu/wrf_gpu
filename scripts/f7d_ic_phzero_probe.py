#!/usr/bin/env python
"""F7D IC experiment: inject the warm/cold bubble ONLY through theta', with
ph_perturbation == 0 (let the dycore develop the geopotential dynamically, as
the f7a analytic warm-bubble oracle does and passes).

The current idealized _make_state pre-loads a statically-integrated
ph_perturbation for the perturbed columns; that pre-loaded geopotential is NOT
on the dycore's discrete-balance manifold and excites a horizontally-uniform
growing vertical mode (see proofs/f7d consistent-base vs bubble probes).  This
probe tests whether ph'=0 (pure-theta' buoyancy) removes the runaway.

Run:  taskset -c 0-3 python -u scripts/f7d_ic_phzero_probe.py --case warm_bubble --end-seconds 200
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
    _enforce_operational_precision,
    build_density_current_numpy,
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("warm_bubble", "density_current"), required=True)
    parser.add_argument("--end-seconds", type=float, default=200.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy() if args.case == "warm_bubble" else build_density_current_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    # Zero the pre-loaded geopotential perturbation: ph_total -> base column phi,
    # ph_perturbation -> 0.  Keep theta' (the buoyancy source) intact.
    s = carry.state
    ph_base = s.ph_total - s.ph_perturbation
    carry = carry.replace(state=s.replace(
        ph=ph_base, ph_total=ph_base, ph_perturbation=jnp.zeros_like(s.ph_perturbation),
    ))

    dt = float(case.dt_s)
    stride = max(1, int(round(args.stride_seconds / dt)))
    n_marks = int(round(args.end_seconds / args.stride_seconds))
    z_m = jnp.asarray(case.z_m)

    def diag(c):
        s = c.state
        w = s.w[:, 0, :]
        wmass = 0.5 * (w[:-1, :] + w[1:, :])
        thp = s.theta[:, 0, :] - 300.0
        wpos = jnp.maximum(thp, 0.0); tot = jnp.sum(wpos)
        cz = jnp.where(tot > 1e-9, jnp.sum(wpos * z_m[:, None]) / jnp.maximum(tot, 1e-12), 0.0)
        finite = (jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(thp))).astype(jnp.float64)
        return {"finite": finite, "max_abs_w": jnp.max(jnp.abs(wmass)),
                "thp_min": jnp.min(thp), "thp_max": jnp.max(thp), "pos_cz": cz}

    @jax.jit
    def run(c):
        def body(cc, m):
            steps = m * stride + jnp.arange(stride, dtype=jnp.int32) + 1
            cc, _ = jax.lax.scan(lambda x, st: (_physics_boundary_step(x, namelist, st, run_radiation=False), None), cc, steps)
            return cc, diag(cc)
        return jax.lax.scan(body, c, jnp.arange(n_marks, dtype=jnp.int32))

    print(f"ph'=0 IC probe case={args.case} compiling...", flush=True)
    _f, d = run(carry)
    d = jax.device_get(d)
    trace = []
    for m in range(n_marks):
        sec = (m + 1) * stride * dt
        rec = {"second": round(sec, 2), "finite": bool(float(d["finite"][m]) > 0.5),
               "max_abs_w": float(d["max_abs_w"][m]), "thp_min": float(d["thp_min"][m]),
               "thp_max": float(d["thp_max"][m]), "pos_cz": float(d["pos_cz"][m])}
        trace.append(rec)
        print(f"  t={sec:7.1f} fin={rec['finite']} max|w|={rec['max_abs_w']:.4e} "
              f"thp=[{rec['thp_min']:.3f},{rec['thp_max']:.3f}] cz={rec['pos_cz']:.1f}", flush=True)
    out = Path("proofs/f7d") / f"ic_phzero_{args.case}.json"
    out.write_text(json.dumps({"case": args.case, "ic": "ph_perturbation=0", "trace": trace}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
