"""Focused tests for the WRF MYNN cold-start level-2 equilibrium init (v0.14).

WRF's first MYNN call (``initflag>0``, not restart) builds the turbulence state
internally via ``mym_initialize`` — a 5-pass level-2 closure equilibrium — not
just the ``5*ust`` taper seed. ``mynn_coldstart_init_columns`` transcribes that
path; these tests pin its WRF-anchored invariants (the field-level oracle gate
against the disposable WRF driver hook lives in
``proofs/v014/mynn_driver_source_output_fix.py``).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, mynn_coldstart_init_columns

QKEMIN_FLOOR = 1.0e-5 ** (2.0 / 3.0)  # level-2 equilibrium qke at the WRF qkemin floor
QKE_CAP = 125.0 ** (2.0 / 3.0)        # WRF MIN(...,125.) cap on b1l*(pdk+pdk)


def _columns(theta_profiles: np.ndarray, ust: np.ndarray) -> MynnPBLColumnState:
    b, nz = theta_profiles.shape
    dz = np.full((b, nz), 50.0)
    p = 100000.0 - np.cumsum(dz, axis=1) * 11.0
    rho = np.full((b, nz), 1.1)
    qv = np.full((b, nz), 0.008)
    zeros = np.zeros((b, nz))
    u = np.full((b, nz), 3.0)
    v = np.full((b, nz), -2.0)
    return MynnPBLColumnState(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(zeros),
        jnp.asarray(theta_profiles), jnp.asarray(qv), jnp.asarray(zeros),
        jnp.asarray(p), jnp.asarray(rho), jnp.asarray(dz),
        jnp.asarray(zeros), jnp.asarray(zeros), jnp.asarray(zeros),
        qc=jnp.asarray(zeros), qi=jnp.asarray(zeros),
    )


def _stable_and_unstable_batch():
    nz = 24
    k = np.arange(nz)
    stable = 290.0 + 0.005 * 50.0 * k                # +0.25 K per level
    unstable = stable.copy()
    unstable[:5] = [294.0, 293.2, 292.4, 291.6, 290.8]  # 3.2 K unstable low layer
    theta = np.stack([stable, unstable, stable, unstable])
    ust = np.array([1.0e-4, 1.0e-4, 0.08, 0.08])
    return theta, ust


def test_stable_columns_sit_at_wrf_qkemin_equilibrium_floor():
    theta, ust = _stable_and_unstable_batch()
    cols = _columns(theta, ust)
    qke, _pblh = mynn_coldstart_init_columns(
        cols, jnp.asarray(ust), 3000.0, jnp.full(ust.shape, 2.0)
    )
    qke = np.asarray(qke)
    # stable interior levels: level-2 production <= 0 -> tmpq clamps at qkemin.
    interior = qke[0, 1:-1]
    np.testing.assert_allclose(interior, QKEMIN_FLOOR, rtol=1e-12)
    # WRF copies the top level from kte-1.
    assert qke[0, -1] == qke[0, -2]


def test_unstable_layer_spins_up_level2_equilibrium_qke():
    theta, ust = _stable_and_unstable_batch()
    cols = _columns(theta, ust)
    qke, _pblh = mynn_coldstart_init_columns(
        cols, jnp.asarray(ust), 3000.0, jnp.full(ust.shape, 2.0)
    )
    qke = np.asarray(qke)
    # the unstable column's low levels must spin far above the floor (this is
    # the order-10 Step-1 MYNN source deficit root cause when missing)...
    assert np.max(qke[1, 1:6]) > 50.0 * QKEMIN_FLOOR
    # ...while remaining under the WRF 125.**(2/3) cap.
    assert np.max(qke) <= QKE_CAP + 1e-12
    # the matching stable column is untouched by the unstable batch member.
    np.testing.assert_allclose(qke[2, 1:-1], qke[0, 1:-1], rtol=1e-12)


def test_surface_value_uses_wrf_ust_floor_and_smoothing():
    theta, ust = _stable_and_unstable_batch()
    cols = _columns(theta, ust)
    qke_lo, _ = mynn_coldstart_init_columns(
        cols, jnp.asarray(ust), 3000.0, jnp.full(ust.shape, 2.0)
    )
    # raising ust below the MAX(ust,0.02) floor must not change the surface qke
    # of the stable column (floor-dominated).
    ust_hi = np.where(ust < 0.02, 0.015, ust)
    qke_hi, _ = mynn_coldstart_init_columns(
        cols, jnp.asarray(ust_hi), 3000.0, jnp.full(ust.shape, 2.0)
    )
    np.testing.assert_allclose(
        np.asarray(qke_lo)[0, 0], np.asarray(qke_hi)[0, 0], rtol=1e-6
    )


def test_d02_replay_gate_keeps_carried_tke():
    from gpuwrf.integration.d02_replay import _wrf_mynn_coldstart_qke

    carried = jnp.full((5, 4, 3), 0.5)
    out, did_seed = _wrf_mynn_coldstart_qke(carried, state=None, grid=None)
    assert not did_seed
    np.testing.assert_array_equal(np.asarray(out), np.asarray(carried))
