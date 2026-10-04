"""b-diff ring stencil == B36 REAL4 nest-scalar source oracle (CPU interpreter).

The released B36 path couples with couple_scalar_real4, scatters a full-ring
target, applies the relax_bdytend row scatter and adds the spec_bdytend
scatter; _ring_relax_spec_real4 does the same REAL arithmetic in one kernel.
Operands: a small grid with non-trivial fields; boundary records are strips of
one full field per time. (The released full-ring target lets S/N strips win
in corner overlaps while WRF relax_bdytend reads the owning side's array; the
two agree whenever the strips come from one field, as forcedown's do.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import boundary_apply as bd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from v0234_wrf_scalar_boundary_real4_oracle import scalar_boundary_tendency as real4_oracle


@pytest.mark.parametrize("lead", [0.0, 18.0, 27.0])
def test_ring_kernel_matches_released_real4_composition(monkeypatch, lead):
    from gpuwrf.kernels import dyn_ring_relax_fp32 as ring
    rng = np.random.default_rng(7)
    nz, ny, nx, side = 12, 40, 52, 53
    z = np.linspace(0, 1, nz)[:, None, None]
    field = jnp.asarray(1e-4 * (1.0 + 0.3 * z + 0.05 * rng.standard_normal((nz, ny, nx))), jnp.float64)
    mu = jnp.asarray(9.0e4 + 500.0 * rng.standard_normal((ny, nx)), jnp.float64)
    c1h = jnp.asarray(np.linspace(1.0, 0.2, nz), jnp.float64)
    c2h = jnp.asarray(np.linspace(0.0, 4.0e3, nz), jnp.float64)
    # Records are strips of one full coupled field per time (as forcedown
    # produces them), so overlapping corner strips hold the same values.
    records = np.zeros((2, 4, 5, nz, side))
    for t in range(2):
        full = 9.0 * (1.0 + 0.04 * rng.standard_normal((nz, ny, nx)))
        for b in range(5):
            records[t, 0, b, :, :ny] = full[:, :, b]
            records[t, 1, b, :, :ny] = full[:, :, nx - 1 - b]
            records[t, 2, b, :, :nx] = full[:, b, :]
            records[t, 3, b, :, :nx] = full[:, ny - 1 - b, :]
    records = jnp.asarray(records, jnp.float64)
    cfg = bd.BoundaryConfig(update_cadence_s=54.0, force_geopotential=False)
    dt = 18.0

    class _M:  # minimal metrics carrier for couple_scalar_real4
        pass
    metrics = _M()
    metrics.c1h, metrics.c2h = c1h, c2h
    value, rate = bd._scalar_record_value_rate_real4(records, lead, cfg.update_cadence_s)
    got = bd._ring_relax_spec_real4(field, value, rate, mu, metrics, dt, cfg)
    # NumPy binary32 source oracle of relax_bdy_scalar + spec_bdytend (B36's
    # REAL4 gate oracle: every operation separately rounded, no FMA).
    released = real4_oracle(np.asarray(field), np.asarray(mu), np.asarray(c1h), np.asarray(c2h),
                            np.asarray(records), lead_seconds=lead, cadence_s=cfg.update_cadence_s,
                            dt_full=dt, spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    assert got.dtype == jnp.float32 and got.shape == released.shape
    released, got = np.asarray(released, np.float64), np.asarray(got, np.float64)
    scale = np.abs(released).max()
    # Same REAL operations in the same order: bit-identical on the CPU path.
    np.testing.assert_array_equal(got, released)
    # Deletion sensitivity: dropping the spec write or the relax term must fail.
    no_spec = np.asarray(ring.ring_relax_tendency(field, value, None,
        [bd._wrf_relax_weights(b, dt, cfg)[0] / dt for b in range(5)],
        [bd._wrf_relax_weights(b, dt, cfg)[1] / dt for b in range(5)],
        spec=cfg.spec_zone, relax=cfg.relax_zone, mass=(mu, c1h, c2h)), np.float64)
    assert np.max(np.abs(no_spec - released)) > 1e-3 * scale
