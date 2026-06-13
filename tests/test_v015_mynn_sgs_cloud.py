"""v0.15 MYNN SGS-cloud chain unit tests (mym_condensation CASE(2) + qsq +
DMP shallow-cu overwrite + icloud_bl radiation merge).

The kernels are direct transcriptions of phys/module_bl_mynnedmf.F (pristine
WRF v4); these tests pin the structural/physical contracts the coupled
72 h gates then validate end-to-end.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from gpuwrf.physics.mynn_pbl import (
    MynnPBLColumnState,
    _step_mynn_pbl_impl_with_pblh,
)
from gpuwrf.physics.mynn_sgs_cloud import (
    dmp_shallow_cu_overwrite,
    mym_condensation_cloudpdf2,
)


def _column(nz=24, B=2, rh=0.5, theta0=290.0, lapse=0.003):
    dz = np.full((B, nz), 200.0)
    zmid = np.cumsum(dz, axis=1) - 0.5 * dz
    p = 1.0e5 * np.exp(-zmid / 8000.0)
    theta = theta0 + lapse * zmid
    exner = (p / 1.0e5) ** (2.0 / 7.0)
    t = theta * exner
    es = 610.78 * np.exp(17.27 * (t - 273.15) / (t - 35.85))
    qsat = 0.622 * es / np.maximum(p - es, 1.0)
    qv = np.maximum(rh * qsat, 1.0e-6)
    return dz, p, theta, exner, qv


def test_condensation_dry_column_is_cloud_free():
    dz, p, theta, exner, qv = _column(rh=0.30)
    zeros = np.zeros_like(qv)
    qc_bl, qi_bl, cldfra = mym_condensation_cloudpdf2(
        theta=jnp.asarray(theta), p=jnp.asarray(p), exner=jnp.asarray(exner),
        dz=jnp.asarray(dz), qw=jnp.asarray(qv), qc=jnp.asarray(zeros),
        qi=jnp.asarray(zeros), qs=jnp.asarray(zeros),
        qsq=jnp.asarray(zeros), pblh=jnp.asarray(np.full(qv.shape[0], 500.0)),
    )
    assert float(jnp.max(cldfra)) == 0.0
    assert float(jnp.max(qc_bl)) == 0.0
    assert float(jnp.max(qi_bl)) == 0.0


def test_condensation_saturated_bl_produces_warm_cloud_below_tropopause():
    dz, p, theta, exner, qv = _column(rh=1.00, theta0=288.0, lapse=0.001)
    zeros = np.zeros_like(qv)
    qsq = np.full_like(qv, 1.0e-7)
    qc_bl, qi_bl, cldfra = mym_condensation_cloudpdf2(
        theta=jnp.asarray(theta), p=jnp.asarray(p), exner=jnp.asarray(exner),
        dz=jnp.asarray(dz), qw=jnp.asarray(qv), qc=jnp.asarray(zeros),
        qi=jnp.asarray(zeros), qs=jnp.asarray(zeros),
        qsq=jnp.asarray(qsq), pblh=jnp.asarray(np.full(qv.shape[0], 800.0)),
    )
    cf = np.asarray(cldfra)
    assert cf.max() > 0.5            # saturated air must be cloudy
    assert cf.min() >= 0.0 and cf.max() <= 1.0
    # warm low levels -> liquid, not ice
    assert float(jnp.max(qc_bl[..., :4])) > 0.0
    assert float(jnp.max(qi_bl[..., :4])) == 0.0
    # top model level always zero (WRF kte handling)
    assert float(jnp.max(cldfra[..., -1])) == 0.0


def test_condensation_cold_column_partitions_to_ice():
    dz, p, theta, exner, _ = _column(theta0=180.0, lapse=0.001)  # T << 240 K
    qv = np.full_like(p, 1.0e-4)
    zeros = np.zeros_like(qv)
    qc_bl, qi_bl, cldfra = mym_condensation_cloudpdf2(
        theta=jnp.asarray(theta), p=jnp.asarray(p), exner=jnp.asarray(exner),
        dz=jnp.asarray(dz), qw=jnp.asarray(qv * 50), qc=jnp.asarray(zeros),
        qi=jnp.asarray(zeros), qs=jnp.asarray(zeros),
        qsq=jnp.asarray(zeros), pblh=jnp.asarray(np.full(qv.shape[0], 500.0)),
    )
    assert float(jnp.max(qc_bl)) == 0.0  # liq_frac = 0 below tice=240 K


def test_shallow_cu_overwrite_only_under_threshold_and_plume():
    nz, B = 24, 1
    dz, p, theta, exner, qv = _column(nz=nz, B=B, rh=0.85)
    thl = theta.copy()
    qc_bl = np.zeros((B, nz))
    cldfra = np.zeros((B, nz))
    cldfra[:, 10] = 0.9             # stratus >= cf_thresh -> never overwritten
    edmf_a = np.zeros((B, nz))
    edmf_qc = np.zeros((B, nz))
    edmf_qt = np.zeros((B, nz))
    # condensing plume bracketing mass levels 5..8 and 10..11
    for j in (4, 5, 6, 7, 8, 9, 10, 11):
        edmf_a[:, j] = 0.05
        edmf_qc[:, j] = 2.0e-4
        edmf_qt[:, j] = qv[:, j]
    out_qc, out_cf = dmp_shallow_cu_overwrite(
        qc_bl=jnp.asarray(qc_bl), cldfra_bl=jnp.asarray(cldfra),
        edmf_a=jnp.asarray(edmf_a), edmf_qc=jnp.asarray(edmf_qc),
        edmf_qt=jnp.asarray(edmf_qt), theta=jnp.asarray(theta),
        thl=jnp.asarray(thl), qw=jnp.asarray(qv), p=jnp.asarray(p),
        exner=jnp.asarray(exner), dz=jnp.asarray(dz),
        xland=jnp.asarray(np.full(B, 2.0)),
    )
    out_qc = np.asarray(out_qc)
    out_cf = np.asarray(out_cf)
    # overwritten where plumes condense and stratus < 0.5
    assert out_cf[0, 6] >= 0.01 and out_qc[0, 6] > 0.0
    # protected stratus level keeps its 0.9
    assert out_cf[0, 10] == pytest.approx(0.9)
    # no plume -> untouched
    assert out_cf[0, 15] == 0.0 and out_qc[0, 15] == 0.0
    # WRF mf_cf bounds
    inside = out_cf[(out_cf > 0) & (out_cf != 0.9)]
    assert inside.max() <= 0.8 + 1e-12


def _step_state(nz=20, B=4):
    rng = np.random.default_rng(7)
    dz = np.full((B, nz), 200.0)
    zmid = np.cumsum(dz, axis=1) - 0.5 * dz
    p = 1.0e5 * np.exp(-zmid / 8000.0)
    theta = 289.0 + 0.002 * zmid
    qv = np.maximum(0.0095 * np.exp(-zmid / 2500.0), 1.0e-6)
    u = np.full((B, nz), 6.0)
    v = np.zeros((B, nz))
    w = np.zeros((B, nz))
    tke = np.full((B, nz), 0.2)
    rho = p / (287.0 * theta * (p / 1e5) ** 0.2857)
    z = np.zeros((B, nz))
    return MynnPBLColumnState(*(jnp.asarray(x) for x in (u, v, w, theta, qv, tke, p, rho, dz, z, z, z)))


def test_step_advances_qsq_and_emits_bl_cloud_leaves():
    st = _step_state()
    out, _pblh = _step_mynn_pbl_impl_with_pblh(st, 30.0, False, None, True, 3000.0)
    qsq = np.asarray(out.qsq)
    assert np.isfinite(qsq).all()
    assert (qsq >= 1.0e-17 - 1e-30).all()   # WRF post-solve floor
    assert qsq.max() > 1.0e-17              # production happened somewhere
    assert np.isfinite(np.asarray(out.cldfra_bl)).all()
    assert float(jnp.min(out.cldfra_bl)) >= 0.0
    assert float(jnp.max(out.cldfra_bl)) <= 1.0


def test_sgs_cloud_rollback_flag_restores_dry_path(monkeypatch):
    import gpuwrf.physics.mynn_pbl as mp

    st = _step_state()
    out_on, _ = _step_mynn_pbl_impl_with_pblh(st, 30.0, False, None, True, 3000.0)
    monkeypatch.setattr(mp, "_MYNN_SGS_CLOUD", False)
    out_off, _ = _step_mynn_pbl_impl_with_pblh(st, 30.0, False, None, True, 3000.0)
    # rollback emits NO SGS cloud and passes qsq through
    assert float(jnp.max(out_off.cldfra_bl)) == 0.0
    assert float(jnp.max(out_off.qc_bl)) == 0.0
    assert np.array_equal(np.asarray(out_off.qsq), np.asarray(st.qsq))
    # and the chain is genuinely live when enabled (some state difference)
    assert np.asarray(out_on.qsq).max() > np.asarray(out_off.qsq).max()
