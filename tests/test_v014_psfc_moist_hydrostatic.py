"""V0.14: PSFC = WRF moist hydrostatic ``p_hyd_w(kts)``, not a p extrapolation.

WRF anchor: runtime ``PSFC(i,j) = p8w(i,kts,j)`` (module_surface_driver.F:1988)
where the driver's p8w argument is ``grid%p_hyd_w``
(module_first_rk_step_part1.F:1400), built in ``phy_prep``
(module_big_step_utilities_em.F:4946-4958) as the moist hydrostatic column
integral in the hybrid dry-mass coordinate::

    p_hyd_w(kte) = p_top
    p_hyd_w(k)   = p_hyd_w(k+1) - (1+qtot)*(c1h(k)*MUT+c2h(k))*dnw(k)

with ``qtot`` summed over ALL moist species. Proof against CPU-WRF truth:
proofs/v014/psfc_moist_pressure_state_closure.* (residual RMSE <= 0.18 Pa).
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.runtime.operational_mode import _psfc_from_state


def _reference_p_hyd_w_sfc(p_top, c1h, c2h, dnw, mut, *q_species):
    """Explicit WRF phy_prep downward recurrence (the Fortran loop, verbatim)."""
    nz = dnw.shape[0]
    p_w = np.full(mut.shape, float(p_top), dtype=np.float64)
    for k in range(nz - 1, -1, -1):
        qtot = np.zeros(mut.shape, dtype=np.float64)
        for q in q_species:
            qtot = qtot + q[k]
        p_w = p_w - (1.0 + qtot) * (c1h[k] * mut + c2h[k]) * dnw[k]
    return p_w


def _synthetic_column(nz=4, ny=2, nx=3, seed=7):
    rng = np.random.default_rng(seed)
    dnw = np.full(nz, -1.0 / nz)
    c1h = np.linspace(1.0, 0.9, nz)
    c2h = np.linspace(0.0, 500.0, nz)
    mut = 96000.0 + 1000.0 * rng.random((ny, nx))
    qv = 0.01 * rng.random((nz, ny, nx))
    qc = 0.001 * rng.random((nz, ny, nx))
    qs = 0.0005 * rng.random((nz, ny, nx))
    return dnw, c1h, c2h, mut, qv, qc, qs


def test_psfc_from_state_is_wrf_moist_hydrostatic_integral():
    nz, ny, nx = 4, 2, 3
    dnw, c1h, c2h, mut, qv, qc, qs = _synthetic_column(nz, ny, nx)
    zeros = np.zeros((nz, ny, nx))
    p_top = 5000.0

    state = SimpleNamespace(
        qv=jnp.asarray(qv),
        qc=jnp.asarray(qc),
        qr=jnp.asarray(zeros),
        qi=jnp.asarray(zeros),
        qs=jnp.asarray(qs),
        qg=jnp.asarray(zeros),
        mu_total=jnp.asarray(mut),
        mu_perturbation=jnp.asarray(mut, dtype=jnp.float64),  # fp64 => plain .sum branch
    )
    metrics = SimpleNamespace(
        c1h=jnp.asarray(c1h),
        c2h=jnp.asarray(c2h),
        dnw=jnp.asarray(dnw),
        p_top=jnp.asarray(p_top),
    )
    got = np.asarray(_psfc_from_state(state, metrics))
    want = _reference_p_hyd_w_sfc(p_top, c1h, c2h, dnw, mut, qv, qc, qs)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1.0e-9)
    # the moist load is real: removing moisture must DROP psfc by ~sum(q*dp_dry)
    dry = _reference_p_hyd_w_sfc(p_top, c1h, c2h, dnw, mut)
    assert float(np.min(want - dry)) > 0.0


def test_writer_psfc_fallback_uses_metrics_moist_hydrostatic(tmp_path):
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf

    nz, ny, nx = 4, 3, 5
    dnw, c1h, c2h, mut, qv, qc, qs = _synthetic_column(nz, ny, nx)
    p_top = 5000.0
    zf = np.arange(nz + 1, dtype=np.float64)[:, None, None]
    z3 = np.arange(nz, dtype=np.float64)[:, None, None]
    mub = np.full((ny, nx), 90000.0)
    mu_pert = mut - mub
    state = SimpleNamespace(
        u=np.zeros((nz, ny, nx + 1)),
        v=np.zeros((nz, ny + 1, nx)),
        w=np.zeros((nz + 1, ny, nx)),
        theta=300.0 + np.zeros((nz, ny, nx)),
        qv=qv,
        qc=qc,
        qi=np.zeros((nz, ny, nx)),
        qr=np.zeros((nz, ny, nx)),
        qs=qs,
        p_total=90000.0 - 800.0 * z3 + np.zeros((nz, ny, nx)),
        p_perturbation=np.zeros((nz, ny, nx)),
        ph_total=9.81 * 600.0 * zf + np.zeros((nz + 1, ny, nx)),
        ph_perturbation=np.zeros((nz + 1, ny, nx)),
        mu_total=mut,
        mu_perturbation=mu_pert,
        # NO psfc leaf -> writer must take the metrics-based default
    )
    grid = SimpleNamespace(
        nx=nx,
        ny=ny,
        nz=nz,
        projection=SimpleNamespace(
            kind="lambert", lat_0=28.34, lon_0=-16.12, dx_m=3000.0, dy_m=3000.0
        ),
        vertical=SimpleNamespace(nz=nz, top_pressure_pa=p_top),
        metrics=SimpleNamespace(c1h=c1h, c2h=c2h, dnw=dnw, p_top=np.asarray(p_top)),
    )
    path = tmp_path / "wrfout_d01_test"
    write_wrfout_netcdf(
        state,
        grid,
        None,
        path,
        valid_time=datetime(2026, 5, 1, 19),
        lead_hours=1.0,
        run_start=datetime(2026, 5, 1, 18),
    )
    with Dataset(path) as nc:
        got = np.array(nc["PSFC"][0], dtype=np.float64)
    want = _reference_p_hyd_w_sfc(p_top, c1h, c2h, dnw, mut, qv, qc, qs)
    # wrfout stores float32; compare at float32 resolution of ~100 kPa values
    np.testing.assert_allclose(got, want, rtol=0.0, atol=0.05)
