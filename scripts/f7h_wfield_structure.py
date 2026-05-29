#!/usr/bin/env python
"""F7H — is the growing w a coherent updraft or a numerical mode?

Run the warm bubble a few hundred steps and dump the w field column profile at
the bubble center vs the domain, plus the theta' field, to distinguish a
physical rising thermal from a grid-scale checkerboard / boundary mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import (
    _build_setup, _enforce_operational_precision, build_warm_bubble_numpy,
)
from gpuwrf.runtime.operational_mode import _physics_boundary_step
from gpuwrf.runtime.operational_state import initial_operational_carry


def main() -> int:
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))
    nx = case.nx
    nz = case.nz
    xc = nx // 2

    from functools import partial

    @partial(jax.jit, static_argnums=(1,))
    def run(carry, n):
        def body(c, st):
            return _physics_boundary_step(c, namelist, st, run_radiation=False), None
        c, _ = jax.lax.scan(body, carry, jnp.arange(1, n + 1, dtype=jnp.int32))
        return c

    for nsteps in (0, 200, 400, 600):
        c = carry if nsteps == 0 else run(carry, nsteps)
        s = c.state
        w = np.asarray(jax.device_get(s.w[:, 0, :]))      # (nz+1, nx)
        th = np.asarray(jax.device_get(s.theta[:, 0, :])) - 300.0
        # w sign pattern across x at the level of max|w|
        kmax = int(np.argmax(np.max(np.abs(w), axis=1)))
        wrow = w[kmax]
        # detect checkerboard: sign flips between adjacent x
        sign_flips = int(np.sum(np.abs(np.diff(np.sign(wrow + 1e-30))) > 1))
        # theta centroid x and z
        thp = np.maximum(th, 0.0)
        tot = thp.sum()
        zc = float((thp * case.z_m[:, None]).sum() / max(tot, 1e-12))
        xcm = float((thp * case.x_m[None, :]).sum() / max(tot, 1e-12))
        print(f"steps={nsteps:4d} t={nsteps*case.dt_s:5.1f}s  max|w|={np.max(np.abs(w)):.3e} @k={kmax} "
              f"w_signflips_in_x={sign_flips}/{nx}  thp_max={th.max():.4f} thp_min={th.min():.4f} "
              f"theta_centroid(z={zc:.1f},x={xcm:.0f})", flush=True)
        # show coarse w column at bubble center
        wcol = w[:, xc]
        idx = np.linspace(0, nz, 9).astype(int)
        print("   w@center(k):", " ".join(f"{w[i, xc]:+.2e}" for i in idx), flush=True)
        # show w across x at kmax (coarse)
        xidx = np.linspace(0, nx - 1, 11).astype(int)
        print("   w@kmax(x):  ", " ".join(f"{wrow[i]:+.2e}" for i in xidx), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
