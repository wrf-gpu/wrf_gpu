#!/usr/bin/env python
"""F7D: locate WHERE the growing w lives (which k-level, which x, what spatial
structure) to distinguish a coherent bubble updraft from a grid-scale 2dx mode
or a boundary-face mode.

Run:  taskset -c 0-3 python -u scripts/f7d_mode_locator.py --steps 600
"""

from __future__ import annotations

import argparse

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
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args(argv)
    case = build_warm_bubble_numpy()
    setup = _build_setup(case, require_gpu=True)
    namelist = setup.namelist
    carry0 = initial_operational_carry(_enforce_operational_precision(setup.state, force_fp64=True))

    from functools import partial

    @partial(jax.jit, static_argnums=(1,))
    def run(carry, n):
        steps = jnp.arange(n, dtype=jnp.int32) + 1
        c, _ = jax.lax.scan(lambda cc, st: (_physics_boundary_step(cc, namelist, st, run_radiation=False), None), carry, steps)
        return c

    print(f"mode locator steps={args.steps} compiling...", flush=True)
    c = run(carry0, int(args.steps))
    w = np.asarray(jax.device_get(c.state.w[:, 0, :]), dtype=np.float64)  # (nz+1, nx)
    nzp1, nx = w.shape
    aw = np.abs(w)
    kmax, xmax = np.unravel_index(np.argmax(aw), aw.shape)
    print(f"max|w|={aw.max():.4e} at k={kmax}/{nzp1-1} x={xmax}/{nx}", flush=True)
    # 2dx checkerboard proxy: ratio of energy at Nyquist (alternating-sign) vs total, per row
    col = w[kmax, :]
    alt_sign = col * ((-1.0) ** np.arange(nx))
    nyq_frac = float(np.abs(np.sum(alt_sign)) / (np.sum(np.abs(col)) + 1e-30))
    # vertical 2dx proxy at xmax
    colz = w[:, xmax]
    alt_signz = colz * ((-1.0) ** np.arange(nzp1))
    nyq_fracz = float(np.abs(np.sum(alt_signz)) / (np.sum(np.abs(colz)) + 1e-30))
    print(f"row k={kmax}: w[x] sample head={np.round(col[:8],3)} tail={np.round(col[-4:],3)}", flush=True)
    print(f"col x={xmax}: w[k] = {np.round(colz,3)}", flush=True)
    print(f"horizontal-coherence(|sum alt|/sum|.|)={nyq_frac:.3f}  vertical-coherence={nyq_fracz:.3f}", flush=True)
    print(f"  (coherence near 1 => smooth/large-scale; near 0 => 2dx checkerboard)", flush=True)
    # where is max along the column vs the bubble center (k~8 for z=2000 dz=250)?
    print(f"per-k max|w|: {np.round(np.max(aw,axis=1),3)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
