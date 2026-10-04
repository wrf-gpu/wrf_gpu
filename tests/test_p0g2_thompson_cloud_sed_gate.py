"""P0 gate-2 attribution repair: WRF gates Thompson cloud-water sedimentation on ANY(L_qc).

module_mp_thompson.F sets L_qc(k) only from the pre-condensation cloud ((qc1d+qcten*DT) > R1, :3215-3223), clears it
only in the condensation step (:3484-3485), and runs both cloud-sedimentation blocks only if ANY(L_qc) (:3646, :3824).
The gate-2 WRF column (qc = 0 on entry) therefore keeps all freshly condensed level-1 cloud; the ungated port lost
vtc*qc*dt/dz ~= 1.87e-5 kg/kg (observed residual 1.847e-5; p0/gate2/attribution).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import thompson_column as tc


def test_l_qc_any_follows_wrf_set_and_clear_rules():
    rho = jnp.full((1, 3), 1.15)
    br = jnp.asarray([[True, True, False]])
    none_pre = tc._wrf_l_qc_any(jnp.zeros((1, 3)), rho, jnp.asarray([[1.0e-3, 0.0, 0.0]]), br)
    assert not bool(none_pre[0, 0]), "newly condensed cloud must not set L_qc"
    kept = tc._wrf_l_qc_any(jnp.asarray([[0.0, 2.0e-4, 0.0]]), rho, jnp.asarray([[1.0e-3, 2.1e-4, 0.0]]), br)
    assert bool(kept[0, 0])
    cleared = tc._wrf_l_qc_any(jnp.asarray([[0.0, 2.0e-4, 0.0]]), rho, jnp.asarray([[1.0e-3, 0.0, 0.0]]), br)
    assert not bool(cleared[0, 0]), "pre-existing cloud fully evaporated inside the branch clears L_qc"
    assert kept.shape == (1, 1)


def test_l_qc_clear_is_the_branch_rule_not_a_qc_change_rule():
    """Negative control (manager review of e36a30c3a): cloudy level with qc_pre > R1 but qc*rho <= R1 at low rho and
    NO qc change.  WRF :3484-3485 clears L_qc whenever the branch executes and rc = MAX(R1, qc*rho) == R1, so the
    column gate is False; outside the branch L_qc stays set.  A 'qc changed' rule would wrongly keep it."""
    qc_pre = jnp.asarray([[2.0e-12, 0.0]])                 # > R1 as a mixing ratio
    rho = jnp.asarray([[0.4, 0.4]])                        # qc*rho = 8e-13 <= R1
    qc_post = qc_pre                                       # branch ran, rounding left qc unchanged
    in_branch = tc._wrf_l_qc_any(qc_pre, rho, qc_post, jnp.asarray([[True, False]]))
    outside = tc._wrf_l_qc_any(qc_pre, rho, qc_post, jnp.asarray([[False, False]]))
    assert not bool(in_branch[0, 0]) and bool(outside[0, 0])
    changed_rule = bool(jnp.any((qc_pre > tc.R1) & ~((qc_post != qc_pre) & (qc_post * rho <= tc.R1))))
    assert changed_rule is True                            # the superseded rule disagrees with WRF here


def test_condensation_branch_uses_wrf_l_qc_threshold():
    """WRF :3401-3402 evaporation branch needs L_qc (qc > R1), not qc > 0."""
    col = _column(0.0)
    sub = col.replace(qv=col.qv * 0.5, qc=col.qc.at[0, 0].set(5.0e-13))   # subsaturated, qc in (0, R1]
    assert not bool(tc._condensation_branch(sub)[0, 0])
    sub2 = sub.replace(qc=sub.qc.at[0, 0].set(5.0e-6))
    assert bool(tc._condensation_branch(sub2)[0, 0])


def _column(qc1):
    nz = 6
    k = np.arange(nz, dtype=np.float64)
    p = 97614.39 - 700.0 * k
    T = 289.0 - 0.5 * k
    qv = np.full(nz, 0.009)
    qv[0] = 0.0130                                # supersaturated lowest level (gate-2-like)
    rho = tc.density_from_pressure_temperature(p, T, qv)
    z = np.zeros(nz)
    qc = z.copy()
    qc[0] = qc1
    return tc.ThompsonColumnState(
        jnp.asarray(qv[None]), jnp.asarray(qc[None]), jnp.asarray(z[None]), jnp.asarray(z[None]),
        jnp.asarray(z[None]), jnp.asarray(z[None]), jnp.asarray(z[None]), jnp.asarray(z[None]),
        jnp.asarray(T[None]), jnp.asarray(p[None]), jnp.asarray(np.asarray(rho)[None]),
        Ns=jnp.asarray(z[None]), Ng=jnp.asarray(z[None]), dz=jnp.asarray(np.full((1, nz), 50.16)),
        w=jnp.asarray(np.full((1, nz), 0.005)))


def test_cloud_free_entry_keeps_all_condensed_cloud_like_wrf():
    col = _column(0.0)
    out, precip = tc.step_thompson_column_with_precip(col, 18.0)
    water_in = float(col.qv[0, 0] + col.qc[0, 0])
    water_out = float(out.qv[0, 0] + out.qc[0, 0] + out.qr[0, 0])
    assert float(out.qc[0, 0]) > 1.0e-4                 # condensation happened
    assert abs(water_out - water_in) < 1.0e-9           # no cloud-sedimentation sink (WRF ANY(L_qc) = F)
    assert float(np.asarray(precip.get("cloudw", 0.0)).max()) == 0.0


def test_cloudy_entry_still_sediments_cloud():
    col = _column(5.0e-4)
    out, precip = tc.step_thompson_column_with_precip(col, 18.0)
    water_in = float(col.qv[0, 0] + col.qc[0, 0])
    water_out = float(out.qv[0, 0] + out.qc[0, 0] + out.qr[0, 0])
    assert water_in - water_out > 1.0e-6                # L_qc true: the bottom cloud flux leaves the column
