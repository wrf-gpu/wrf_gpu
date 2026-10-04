"""B36 REAL4 coupler: the CPU interpret path is exact binary32 (b-diff finding).

XLA's excess-precision rewrite drops the f64->f32->f64 trip between chained
emulated ops; ``lax.reduce_precision`` pins each round.  Without it 5,204 of
24,960 cells differ from NumPy binary32 on this fixture.  The GPU path uses
explicit PTX ``.rn`` instructions and is gated separately on the GPU.

Caveat (review-ring32): XLA:CPU flushes subnormals (FTZ) and reduce_precision
flushes them too, so CPU 'exact' holds on the normal range only; real PROD d01
h12 has 100+ subnormal cells per species.  This fixture is all-normal.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import _couple_scalar_real4_arrays


def test_cpu_real4_coupler_is_exact_binary32():
    rng = np.random.default_rng(0)
    field = (rng.random((12, 40, 52)) * 1e5).astype(np.float32)
    mu = rng.random((40, 52)) * 1e5 + 5e4
    c1 = rng.random(12)
    c2 = rng.random(12) * 100.0
    got = np.asarray(_couple_scalar_real4_arrays(
        jnp.asarray(field), jnp.asarray(mu), jnp.asarray(c1), jnp.asarray(c2)
    ))
    mass = (
        c1.astype(np.float32)[:, None, None] * mu.astype(np.float32)[None]
        + c2.astype(np.float32)[:, None, None]
    )
    np.testing.assert_array_equal(got, field * mass)
