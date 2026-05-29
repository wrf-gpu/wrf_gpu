#!/usr/bin/env python
"""F7D: null-bubble test.  Run the warm-bubble harness with theta' identically
ZERO (pure base column).  If the horizontally-uniform vertical w mode STILL
grows, the runaway is the idealized-IC base-column discrete-hydrostatic-balance
mismatch -- independent of the bubble and of the MUT/MUTS mass-semantics fix.

Run:  taskset -c 0-3 python -u scripts/f7d_null_bubble_probe.py --steps 700
"""

from __future__ import annotations

import argparse
import dataclasses
from functools import partial

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators import idealized as I
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
    parser.add_argument("--stride", type=int, default=100)
    args = parser.parse_args(argv)

    # Build a fully CONSISTENT zero-perturbation case: theta' == 0 so _make_state
    # integrates ph from theta0 and the IC is self-consistent (no manual override).
    base = build_warm_bubble_numpy()
    case = dataclasses.replace(
        base,
        theta_prime_k=np.zeros_like(base.theta_prime_k),
        theta_k=np.full_like(base.theta_k, I.THETA0_K),
    )
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    n_marks = args.steps // args.stride
    stride = args.stride

    def diag(c):
        w = c.state.w[:, 0, :]
        wmass = 0.5 * (w[:-1, :] + w[1:, :])
        finite = jnp.all(jnp.isfinite(w)).astype(jnp.float64)
        return {"finite": finite, "max_abs_w": jnp.max(jnp.abs(wmass)),
                "thp_max": jnp.max(c.state.theta[:, 0, :]) - 300.0,
                "thp_min": jnp.min(c.state.theta[:, 0, :]) - 300.0}

    @partial(jax.jit, static_argnums=())
    def run(c):
        def body(cc, m):
            steps = m * stride + jnp.arange(stride, dtype=jnp.int32) + 1
            cc, _ = jax.lax.scan(lambda x, st: (_physics_boundary_step(x, namelist, st, run_radiation=False), None), cc, steps)
            return cc, diag(cc)
        return jax.lax.scan(body, c, jnp.arange(n_marks, dtype=jnp.int32))

    print("null-bubble (theta'=0) warm-bubble base column; compiling...", flush=True)
    _final, d = run(carry)
    d = jax.device_get(d)
    for m in range(n_marks):
        sec = (m + 1) * stride * float(case.dt_s)
        print(f"  t={sec:6.1f} fin={float(d['finite'][m])>0.5} max|w|={float(d['max_abs_w'][m]):.4e} "
              f"thp=[{float(d['thp_min'][m]):.2e},{float(d['thp_max'][m]):.2e}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
