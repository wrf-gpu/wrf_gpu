"""v0.14 regression: WRF ``hypsometric_opt=2`` LOG-form ``al`` diagnostics.

Root cause (proofs/v014/switzerland_hpg_native_face_fix.json): the JAX runtime
rebuilt the calc_p_rho_phi diagnostics with the LINEAR ``hypsometric_opt=1``
relation while every real WRF case runs the v4 Registry default ``2``
(LOG-pressure-thickness, ``module_big_step_utilities_em.F:1043-1062``).

Analytic oracle: integrate a synthetic total geopotential column from a KNOWN
total specific volume ``alpha_t`` with the exact LOG relation

    ph_tot(k+1) = ph_tot(k) + alpha_t(k) * phm(k) * LOG(pfd(k)/pfu(k))

(``pfu/pfd/phm`` the dry reference pressures of the ``muts`` column).  The
``hypsometric_opt=2`` diagnostics must invert this exactly (machine precision);
the legacy linear form must show the O((dp/p)^2/12) one-signed deviation that
drove the Switzerland d01 h36 mass venting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from gpuwrf.dynamics.acoustic_wrf import diagnose_pressure_al_alt
from gpuwrf.dynamics.core import rk_addtend_dry as rk
from gpuwrf.dynamics.metrics import flat_metrics_for_grid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostic_warm_bubble_vs_slice import build_warm_bubble_3d


def _log_column_inputs():
    grid, state, base, _theta_base, _z_mass = build_warm_bubble_3d()
    metrics = flat_metrics_for_grid(grid)
    mub = np.asarray(base.mub, dtype=np.float64)
    phb = np.asarray(base.phb, dtype=np.float64)
    c3f = np.asarray(metrics.c3f, dtype=np.float64)[:, None, None]
    c4f = np.asarray(metrics.c4f, dtype=np.float64)[:, None, None]
    c3h = np.asarray(metrics.c3h, dtype=np.float64)[:, None, None]
    c4h = np.asarray(metrics.c4h, dtype=np.float64)[:, None, None]
    p_top = float(np.reshape(np.asarray(metrics.p_top), ()))

    nz, ny, nx = state.theta.shape
    # known perturbation dry mass + known target total alpha (smooth, nonzero)
    rng = np.random.default_rng(20260611)
    mu_pert = 50.0 * (1.0 + 0.1 * rng.standard_normal((ny, nx)))
    muts = mub + mu_pert
    alb = np.zeros((nz, ny, nx))
    dnw = np.asarray(metrics.dnw, dtype=np.float64)[:, None, None]
    c1h = np.asarray(metrics.c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(metrics.c2h, dtype=np.float64)[:, None, None]
    alb[:] = -(phb[1:] - phb[:-1]) / (dnw * (c1h * mub[None] + c2h))
    alpha_t = alb * (1.0 + 0.02 * np.sin(np.linspace(0.0, 3.0, nz))[:, None, None])

    pfu = c3f[1:] * muts[None] + c4f[1:] + p_top
    pfd = c3f[:-1] * muts[None] + c4f[:-1] + p_top
    phm = c3h * muts[None] + c4h + p_top
    log_ratio = np.log(pfd / pfu)

    ph_tot = np.zeros((nz + 1, ny, nx))
    ph_tot[0] = phb[0]
    for k in range(nz):
        ph_tot[k + 1] = ph_tot[k] + alpha_t[k] * phm[k] * log_ratio[k]
    ph_pert = ph_tot - phb

    state = state.replace(
        ph_perturbation=jnp.asarray(ph_pert),
        ph_total=jnp.asarray(ph_tot),
        mu_perturbation=jnp.asarray(mu_pert),
        mu_total=jnp.asarray(mub + mu_pert),
    )
    return state, base, metrics, alpha_t, alb


def test_diagnose_hypso2_inverts_log_column_exactly():
    state, base, metrics, alpha_t, alb = _log_column_inputs()
    _p2, al2, alt2 = diagnose_pressure_al_alt(state, base, metrics, hypsometric_opt=2)
    rel2 = np.abs(np.asarray(alt2) - alpha_t) / alpha_t
    assert float(rel2.max()) < 1.0e-12, f"hypso2 must invert the LOG column exactly, got {rel2.max():.3e}"

    # the subtracted base alb must be the LOG form too (WRF real-init integrates
    # PHB from alb with the same relation; native-face proof: live al equals
    # LOG(total) - LOG(base) to ~1.7e-6 rel while a linear-base subtraction
    # corrupts the dominant pb*al HPG face term)
    mub = np.asarray(base.mub, dtype=np.float64)
    phb = np.asarray(base.phb, dtype=np.float64)
    c3f = np.asarray(metrics.c3f, dtype=np.float64)[:, None, None]
    c4f = np.asarray(metrics.c4f, dtype=np.float64)[:, None, None]
    c3h = np.asarray(metrics.c3h, dtype=np.float64)[:, None, None]
    c4h = np.asarray(metrics.c4h, dtype=np.float64)[:, None, None]
    p_top = float(np.reshape(np.asarray(metrics.p_top), ()))
    pfu_b = c3f[1:] * mub[None] + c4f[1:] + p_top
    pfd_b = c3f[:-1] * mub[None] + c4f[:-1] + p_top
    phm_b = c3h * mub[None] + c4h + p_top
    alb_log = (phb[1:] - phb[:-1]) / phm_b / np.log(pfd_b / pfu_b)
    np.testing.assert_allclose(np.asarray(al2), alpha_t - alb_log, rtol=0.0, atol=1.0e-10)

    _p1, al1, alt1 = diagnose_pressure_al_alt(state, base, metrics, hypsometric_opt=1)
    rel1 = np.abs(np.asarray(alt1) - alpha_t) / alpha_t
    # the legacy linear form must show the O((dp/p)^2/12) deviation
    assert float(rel1.max()) > 1.0e-7, f"linear form unexpectedly exact ({rel1.max():.3e})"
    # and the deviation must be one-signed positive (alt_linear > alt_log), the
    # Switzerland signature
    diff = np.asarray(alt1) - np.asarray(alt2)
    assert float(diff.min()) > 0.0


def test_absolute_diagnostics_hypso2_matches_diagnose():
    state, base, metrics, _alpha_t, _alb = _log_column_inputs()
    _p2, al2, _alt2 = diagnose_pressure_al_alt(state, base, metrics, hypsometric_opt=2)
    _ph, _p, al_abs, _alt, _php = rk._absolute_diagnostics(state, metrics, hypsometric_opt=2)
    np.testing.assert_allclose(np.asarray(al_abs), np.asarray(al2), rtol=0.0, atol=1.0e-10)


def test_large_step_pgf_threads_hypsometric_opt():
    state, base, metrics, _alpha_t, _alb = _log_column_inputs()
    # ``al`` only enters the HPG through the (al_l+al_r)*(pb_r-pb_l) face term;
    # the idealized base pressure is horizontally flat, so impose a small
    # horizontal pb variation (real terrain has huge ones) to expose the option.
    nz, ny, nx = state.theta.shape
    pb = np.asarray(state.p_total - state.p_perturbation, dtype=np.float64)
    ramp = 1.0 + 1.0e-3 * np.sin(np.linspace(0.0, 2.0, nx))[None, None, :]
    pb_mod = pb * ramp
    state = state.replace(p_perturbation=state.p_total - jnp.asarray(pb_mod))
    ru1, rv1 = rk.large_step_horizontal_pgf(
        state, metrics, dx_m=400.0, dy_m=400.0, non_hydrostatic=True, top_lid=True, hypsometric_opt=1
    )
    ru2, rv2 = rk.large_step_horizontal_pgf(
        state, metrics, dx_m=400.0, dy_m=400.0, non_hydrostatic=True, top_lid=True, hypsometric_opt=2
    )
    assert bool(np.isfinite(np.asarray(ru2)).all() and np.isfinite(np.asarray(rv2)).all())
    # the option must actually reach the kernel (al enters the pb_al face term)
    assert float(np.max(np.abs(np.asarray(ru1) - np.asarray(ru2)))) > 0.0
