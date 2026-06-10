"""V0.14 focused regression: the Noah-MP/sfclay column view decouples moist theta.

``coupling.noahmp_surface_hook._build_column_view`` builds the trailing-z view the
Noah-MP coupler hands to the revised surface layer (``surface_layer_with_diagnostics``)
and to ``assemble_noahmp_forcing``. The operational ``State.theta`` is the WRF MOIST
potential temperature ``theta_m = theta_dry*(1 + R_v/R_d*qv)`` (use_theta_m=1), but
the surface layer needs the DRY sensible temperature ``t_phy = theta_dry*(p/p0)^kappa``.
The view must therefore supply a dry ``t_air`` (and a dry ``theta``), mirroring
``physics_couplers._surface_column_view``. Over WATER (where Noah-MP does not run) the
retained sfclay bulk flux otherwise gets a ~+4 K too-warm air temperature.

See proofs/v014/surface_layer_theta_decoupling.* for the WRF-anchored end-to-end proof.
"""

import numpy as np
import jax.numpy as jnp

from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
from gpuwrf.coupling.physics_couplers import WRF_RV_OVER_RD
from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP


class _State:
    """Minimal leading-z duck-typed State for _build_column_view (grid-less path)."""

    pass


def _make_state(nz=2, ny=1, nx=2):
    rng = np.random.default_rng(0)
    p = np.linspace(98000.0, 90000.0, nz)[:, None, None] * np.ones((nz, ny, nx))
    qv = np.array([0.012, 0.015])[:, None, None] * np.ones((nz, ny, nx))  # mixing ratio
    t_dry = np.array([293.0, 290.0])[:, None, None] * np.ones((nz, ny, nx))
    theta_dry = t_dry * (P0_PA / p) ** R_D_OVER_CP
    theta_m = theta_dry * (1.0 + WRF_RV_OVER_RD * qv)  # the stored prognostic (moist)

    s = _State()
    s.theta = jnp.asarray(theta_m)
    s.qv = jnp.asarray(qv)
    s.qc = jnp.zeros((nz, ny, nx))
    s.p = jnp.asarray(p)
    s.u = jnp.asarray(np.full((nz, ny, nx + 1), 3.0))
    s.v = jnp.asarray(np.full((nz, ny + 1, nx), 1.0))
    s.ph = jnp.asarray((np.arange(nz + 1) * 600.0 * 9.81)[:, None, None] * np.ones((nz + 1, ny, nx)))
    for name in ("t_skin", "soil_moisture", "xland", "lakemask", "mavail", "roughness_m", "ustar"):
        setattr(s, name, jnp.asarray(np.full((ny, nx), 1.0)))
    return s, np.asarray(t_dry), np.asarray(qv), np.asarray(p)


def test_build_column_view_supplies_dry_t_air_recovering_t_phy():
    state, t_dry, qv, p = _make_state()
    view = _build_column_view(state)  # grid-less fallback path

    # t_air is the trailing-z dry air temperature; the lowest level recovers WRF t_phy.
    t_air = np.asarray(view.t_air, dtype=np.float64)[..., 0]
    np.testing.assert_allclose(t_air, t_dry[0], rtol=0.0, atol=1e-9)

    # the naive (buggy) moist-theta_m Exner would be warm by exactly (1+Rv/Rd*qv).
    theta_m0 = np.asarray(state.theta, dtype=np.float64)[0]
    naive_t = theta_m0 * (p[0] / P0_PA) ** R_D_OVER_CP
    np.testing.assert_allclose(naive_t / t_air, 1.0 + WRF_RV_OVER_RD * qv[0], rtol=0.0, atol=1e-9)
    assert np.all(naive_t - t_air > 3.0)  # ~+4 K warm bias removed


def test_build_column_view_theta_is_dry_and_grid_less_psfc_rho():
    state, t_dry, qv, p = _make_state()
    view = _build_column_view(state)

    # view.theta is the DRY potential temperature (not the stored moist theta_m).
    theta_view0 = np.asarray(view.theta, dtype=np.float64)[..., 0]
    theta_dry0 = np.asarray(state.theta, dtype=np.float64)[0] / (1.0 + WRF_RV_OVER_RD * qv[0])
    np.testing.assert_allclose(theta_view0, theta_dry0, rtol=0.0, atol=1e-9)

    # grid-less fallback: psfc defaults to the lowest-level air pressure (NOT None, so
    # assemble_noahmp_forcing's _get(...,sfcprs) stays valid); rho stays None.
    assert view.psfc is not None
    np.testing.assert_allclose(np.asarray(view.psfc, dtype=np.float64), p[0], rtol=0.0, atol=1e-6)
    assert view.rho is None
