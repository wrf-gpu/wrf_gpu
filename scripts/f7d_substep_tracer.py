#!/usr/bin/env python
"""F7D residual-instability tracer.

Runs the warm bubble for a small number of operational steps and records, per
step, the quantities that diagnose WHERE the coherent w runaway comes from:

  - max|w|, max|u|                      (momentum growth)
  - theta' min/max + positive-theta centroid height  (is the bubble transported?)
  - max|p'|, max|al|                    (acoustic pressure response)
  - max|ph'|                            (geopotential response)
  - vertical-velocity-weighted theta flux proxy

A coherent linear w growth with a STATIONARY theta bubble localizes the defect
to the w->theta vertical-transport closure rather than the pressure restoring.

Run:  taskset -c 0-3 python -u scripts/f7d_substep_tracer.py --steps 700
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
    build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("proofs/f7d/substep_trace_warm_bubble.json"))
    args = parser.parse_args(argv)

    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    z_m = jnp.asarray(case.z_m)  # (nz,)
    n_marks = args.steps // args.stride
    stride = args.stride

    def diag(carry):
        s = carry.state
        w = s.w[:, 0, :]
        wmass = 0.5 * (w[:-1, :] + w[1:, :])
        theta = s.theta[:, 0, :]
        thp = theta - 300.0
        p = s.p_perturbation[:, 0, :]
        al = s.al[:, 0, :] if getattr(s, "al", None) is not None else jnp.zeros_like(p)
        php = s.ph_perturbation[:, 0, :]
        wpos = jnp.maximum(thp, 0.0)
        tot = jnp.sum(wpos)
        cz = jnp.where(tot > 1e-9, jnp.sum(wpos * z_m[:, None]) / jnp.maximum(tot, 1e-12), 0.0)
        finite = (jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(p))).astype(jnp.float64)
        return {
            "finite": finite,
            "max_abs_w": jnp.max(jnp.abs(wmass)),
            "max_abs_u": jnp.max(jnp.abs(s.u[:, 0, :])),
            "thp_min": jnp.min(thp),
            "thp_max": jnp.max(thp),
            "pos_theta_centroid_z": cz,
            "max_abs_p": jnp.max(jnp.abs(p)),
            "max_abs_al": jnp.max(jnp.abs(al)),
            "max_abs_php": jnp.max(jnp.abs(php)),
            "max_abs_ww": jnp.max(jnp.abs(s.theta[:, 0, :] * 0.0)) if False else jnp.max(jnp.abs(carry.ww[:, 0, :])),
        }

    @jax.jit
    def run(carry):
        def body(c, m):
            steps = m * stride + jnp.arange(stride, dtype=jnp.int32) + 1
            c, _ = jax.lax.scan(lambda cc, st: (_physics_boundary_step(cc, namelist, st, run_radiation=False), None), c, steps)
            return c, diag(c)
        marks = jnp.arange(n_marks, dtype=jnp.int32)
        final, d = jax.lax.scan(body, carry, marks)
        return final, d, diag(carry)

    print(f"tracer warm_bubble steps={args.steps} stride={stride} compiling...", flush=True)
    _final, d, d0 = run(carry0)
    d = jax.device_get(d)
    d0 = jax.device_get(d0)

    trace = [{"step": 0, "second": 0.0, **{k: float(v) for k, v in d0.items()}}]
    for m in range(n_marks):
        rec = {"step": (m + 1) * stride, "second": round((m + 1) * stride * float(case.dt_s), 2)}
        for k in d:
            rec[k] = float(np.asarray(d[k])[m])
        trace.append(rec)
    for rec in trace:
        fin = rec["finite"] > 0.5
        print(f"  step={rec['step']:5d} t={rec['second']:6.1f} fin={fin} max|w|={rec['max_abs_w']:.3e} "
              f"thp=[{rec['thp_min']:.3f},{rec['thp_max']:.3f}] cz={rec['pos_theta_centroid_z']:.1f} "
              f"max|p'|={rec['max_abs_p']:.3e} max|al|={rec['max_abs_al']:.3e} max|ph'|={rec['max_abs_php']:.3e} max|ww|={rec['max_abs_ww']:.3e}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"case": "warm_bubble", "trace": trace}, indent=2))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
