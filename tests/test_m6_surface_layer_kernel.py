from __future__ import annotations

from pathlib import Path
import subprocess

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.surface_constants import CP_D, P0_PA, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics


class SurfaceState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _state_from_rows(rows):
    data = np.asarray(rows, dtype=np.float64)
    u, v, t, qv, p, dz, tsk, xland, znt, mavail, old_ust, _old_mol = data.T
    theta = t * (P0_PA / p) ** R_D_OVER_CP
    shape = (data.shape[0], 1)
    return SurfaceState(
        u=jnp.asarray(u.reshape(shape + (1,))),
        v=jnp.asarray(v.reshape(shape + (1,))),
        theta=jnp.asarray(theta.reshape(shape + (1,))),
        qv=jnp.asarray(qv.reshape(shape + (1,))),
        p=jnp.asarray(p.reshape(shape + (1,))),
        dz=jnp.asarray(dz.reshape(shape + (1,))),
        t_skin=jnp.asarray(tsk.reshape(shape)),
        xland=jnp.asarray(xland.reshape(shape)),
        roughness_m=jnp.asarray(znt.reshape(shape)),
        mavail=jnp.asarray(mavail.reshape(shape)),
        ustar=jnp.asarray(old_ust.reshape(shape)),
    )


def test_surface_layer_returns_fp64_finite_fluxes_and_diagnostics():
    rows = [
        [6.0, 2.0, 292.0, 0.008, 95500.0, 80.0, 296.0, 1.0, 0.08, 0.7, 0.0, 0.0],
        [9.0, -1.0, 294.0, 0.010, 100800.0, 60.0, 293.0, 2.0, 0.0015, 1.0, 0.0, 0.0],
    ]
    diag = surface_layer_with_diagnostics(_state_from_rows(rows))

    for value in (
        diag.fluxes.ustar,
        diag.fluxes.theta_flux,
        diag.fluxes.qv_flux,
        diag.fluxes.tau_u,
        diag.fluxes.tau_v,
        diag.fluxes.rhosfc,
        diag.fluxes.fltv,
        diag.u10,
        diag.v10,
        diag.t2,
        diag.q2,
    ):
        assert value.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(value)))
        assert value.shape == (2, 1)

    assert bool(jnp.all(diag.fluxes.ustar > 0.0))
    assert bool(jnp.all(diag.fluxes.rhosfc > 0.0))


def test_surface_layer_first_timestep_uses_wrf_mynn_cold_start_boundary():
    rows = [
        [6.0, 2.0, 292.0, 0.008, 95500.0, 80.0, 296.0, 1.0, 0.08, 0.7, 0.0, 0.0],
    ]
    state = _state_from_rows(rows)

    warm = surface_layer_with_diagnostics(state)
    first = surface_layer_with_diagnostics(state, first_timestep=True)

    qv0 = rows[0][3]
    assert np.allclose(np.asarray(first.qsfc), qv0 / (1.0 + qv0), rtol=0.0, atol=1.0e-14)
    assert abs(float(np.asarray(first.fluxes.qv_flux)[0, 0])) < 1.0e-14
    assert abs(float(np.asarray(warm.fluxes.qv_flux)[0, 0])) > 1.0e-8
    assert float(np.asarray(first.fluxes.ustar)[0, 0]) > float(np.asarray(warm.fluxes.ustar)[0, 0])


def test_surface_layer_matches_wrf_sfclay_harness_when_available(tmp_path):
    """RETIRED (B2 rebuild): this asserted parity with WRF MM5 ``module_sf_sfclay.F``.

    The B2 surface lane was rebuilt on the WRF *revised* surface layer
    (``module_sf_sfclayrev.F`` -> ``sf_sfclayrev_run``, Jimenez et al. 2012), which
    uses Cheng & Brutsaert (2005) integrated similarity functions and a
    bulk-Richardson Newton/secant solve — a different scheme from MM5 ``sfclay``.
    Comparing the rebuilt scheme against the MM5 ``sfclay`` Fortran harness (which
    this test built via ``scripts/wrf_sfclay_harness_build.sh`` and asserted the
    symbol ``module_sf_sfclay_sfclay1d``) is a wrong-oracle comparison and would
    contradict the rebuild by design.

    The authoritative WRF parity for the rebuilt scheme is the WRF *revised*
    surface-layer oracle (``sfclay_pre``/``sfclay_post`` savepoints) compared by
    ``proofs/b2/surface_mynn_parity.py``; standalone algebraic invariants are in
    ``proofs/b2/surface_layer_oracle.py``. The fp64/finite/sign test above still
    guards the rebuilt kernel.
    """

    pytest.skip(
        "MM5 sfclay parity retired: B2 rebuilt on revised sfclayrev. "
        "See proofs/b2/surface_mynn_parity.py (WRF revised-scheme oracle parity)."
    )
