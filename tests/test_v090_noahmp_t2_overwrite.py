"""v0.9.0 Noah-MP LSM 2-m T2 overwrite — operational-hook masking confirmation.

Confirms that ``overlay_noahmp_land_diagnostics`` (the operational surface hook
called from ``runtime.operational_mode``) routes the FAITHFUL Noah-MP LSM 2-m
diagnostic ``T2 = FVEG*T2MV + (1-FVEG)*T2MB`` over LAND and keeps the bulk
surface-layer ``T2`` over WATER — the faithful land-T2 overwrite real WRF performs
(module_surface_driver.F:3469-3473). The T2MV/T2MB values themselves are proven
WRF-faithful at savepoint by proofs/v090/noahmp_t2mb_parity.json (11/11 PASS); this
test validates the operational masking/plumbing on top of that.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.coupling.noahmp_surface_hook import overlay_noahmp_land_diagnostics
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters

TABLE_DIR = Path("<USER_HOME>/src/wrf_pristine/WRF/run")
HAVE_TABLES = (TABLE_DIR / "MPTABLE.TBL").exists()
P0_PA = 1.0e5
R_D_OVER_CP = 287.0 / 1004.0


class _State:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def replace(self, **u):
        d = dict(self.__dict__)
        d.update(u)
        return _State(**d)


def _build_operational():
    """Minimal LEADING-z operational State (staggered u/v) with 3 land + 2 water."""
    ny, nx, nz = 1, 5, 1
    t = np.array([292.0, 295.0, 296.0, 294.0, 293.0])
    p = np.array([95500.0, 95000.0, 94800.0, 100800.0, 100500.0])
    qv = np.array([0.008, 0.009, 0.010, 0.010, 0.011])
    tsk = np.array([296.0, 300.0, 301.0, 293.0, 292.5])
    xland = np.array([1.0, 1.0, 1.0, 2.0, 2.0])  # 1 land, 2 water
    znt = np.array([0.08, 0.05, 0.06, 0.0015, 0.0015])
    theta = t * (P0_PA / p) ** R_D_OVER_CP

    def lead(a):
        return jnp.asarray(a.reshape(nz, ny, nx))

    def s2(a):
        return jnp.asarray(a.reshape(ny, nx))

    state = _State(
        u=jnp.asarray(np.full((nz, ny, nx + 1), 6.0)),
        v=jnp.asarray(np.full((nz, ny + 1, nx), 1.0)),
        theta=lead(theta), qv=lead(qv), qc=jnp.zeros((nz, ny, nx)),
        p=lead(p), dz=jnp.full((nz, ny, nx), 80.0),
        # geopotential interfaces (nz+1): 0 and 80 m -> 80 m layer, lowest mass ~40 m.
        ph=jnp.asarray(np.stack([np.zeros((ny, nx)), np.full((ny, nx), 9.80665 * 80.0)])),
        t_skin=s2(tsk), xland=s2(xland), roughness_m=s2(znt),
        soil_moisture=s2(np.array([0.2, 0.2, 0.2, 1.0, 1.0])),
        mavail=s2(np.array([0.7, 0.6, 0.6, 1.0, 1.0])),
        ustar=s2(np.full(nx, 0.1)), lakemask=s2(np.zeros(nx)),
    )

    shape = (ny, nx)
    z = jnp.zeros(shape)
    veg = np.array([5, 9, 10, 5, 9])
    land = NoahMPLandState(
        tslb=jnp.broadcast_to(jnp.asarray(295.0), (NSOIL, ny, nx)),
        smois=jnp.broadcast_to(jnp.asarray(0.20), (NSOIL, ny, nx)),
        sh2o=jnp.broadcast_to(jnp.asarray(0.20), (NSOIL, ny, nx)),
        smcwtd=jnp.broadcast_to(jnp.asarray(0.20), shape),
        isnow=jnp.zeros(shape, dtype=jnp.int32),
        tsno=jnp.broadcast_to(jnp.asarray(273.0), (NSNOW, ny, nx)),
        snice=jnp.broadcast_to(z, (NSNOW, ny, nx)), snliq=jnp.broadcast_to(z, (NSNOW, ny, nx)),
        zsnso=jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 0.0, -0.05, -0.25, -0.7, -1.5]).reshape(NSNOW + NSOIL, 1, 1),
            (NSNOW + NSOIL, ny, nx)),
        snowh=z, sneqv=z, sneqvo=z, tauss=z, albold=jnp.broadcast_to(jnp.asarray(0.2), shape),
        tv=jnp.broadcast_to(jnp.asarray(296.0), shape), tg=jnp.broadcast_to(jnp.asarray(295.0), shape),
        tah=jnp.broadcast_to(jnp.asarray(295.0), shape), eah=jnp.broadcast_to(jnp.asarray(2000.0), shape),
        canliq=z, canice=z, fwet=z, lai=jnp.broadcast_to(jnp.asarray(2.0), shape),
        sai=jnp.broadcast_to(jnp.asarray(0.5), shape),
        cm=jnp.broadcast_to(jnp.asarray(0.01), shape), ch=jnp.broadcast_to(jnp.asarray(0.01), shape),
        t_skin=jnp.broadcast_to(jnp.asarray(295.0), shape), qsfc=jnp.broadcast_to(jnp.asarray(0.01), shape),
        znt=jnp.broadcast_to(jnp.asarray(0.05), shape), emiss=jnp.broadcast_to(jnp.asarray(0.97), shape),
        albedo=jnp.broadcast_to(jnp.asarray(0.2), shape), sfcrunoff=z, udrunoff=z,
    )
    static = NoahMPStatic(
        ivgtyp=jnp.asarray(veg.reshape(shape), dtype=jnp.int32),
        isltyp=jnp.full(shape, 6, dtype=jnp.int32),
        xland=s2(xland), landmask=s2((xland < 1.5).astype(float)), lakemask=z,
        lu_index=jnp.asarray(veg.reshape(shape), dtype=jnp.int32),
        tbot=jnp.broadcast_to(jnp.asarray(290.0), shape),
        dzs=jnp.asarray([0.05, 0.20, 0.45, 0.80]),
        zsoil=jnp.asarray([-0.05, -0.25, -0.7, -1.5]),
        lat=jnp.broadcast_to(jnp.asarray(28.0), shape), dx_m=1000.0,
        parameters=load_noahmp_parameters(TABLE_DIR),
        shdmax=jnp.broadcast_to(jnp.asarray(0.3), shape),
        shdfac=jnp.broadcast_to(jnp.asarray(0.3), shape),
    )

    class _Rad:
        soldn = s2(np.array([600.0, 700.0, 650.0, 500.0, 400.0]))
        lwdn = s2(np.array([350.0, 360.0, 355.0, 340.0, 330.0]))
        cosz = s2(np.array([0.6, 0.7, 0.65, 0.5, 0.4]))

    class _Clock:
        julian = 142.0
        yearlen = 365.0

    return state, land, static, _Rad(), _Clock()


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_lsm_t2_overwrite_land_only():
    state, land, static, rad, clock = _build_operational()
    is_land = np.asarray((state.xland - 1.5) < 0.0).reshape(-1)

    bulk_hfx = jnp.full((1, 5), 100.0)
    bulk_lh = jnp.full((1, 5), 50.0)
    bulk_tsk = state.t_skin
    bulk_t2 = jnp.asarray(np.array([288.0, 289.0, 290.0, 291.0, 292.0]).reshape(1, 5))

    out = overlay_noahmp_land_diagnostics(
        state, land, static, bulk_hfx, bulk_lh, bulk_tsk, 90.0,
        bulk_t2=bulk_t2, radiation=rad, clock=clock,
    )
    assert len(out) == 4, "with bulk_t2 the hook returns (hfx, lh, tsk, t2)"
    _hfx, _lh, _tsk, t2 = out
    t2 = np.asarray(t2).reshape(-1)

    # the Noah-MP LSM T2 the hook should have routed over land
    _lo, nm = noah_mp_step(land, _assemble(state, static, rad, clock), static, 90.0)
    nm_t2 = np.asarray(nm.t2).reshape(-1)

    # WATER columns keep the bulk surface-layer T2 (byte-for-byte).
    bt = np.asarray(bulk_t2).reshape(-1)
    np.testing.assert_allclose(t2[~is_land], bt[~is_land], rtol=0, atol=1e-12)

    # LAND columns are the Noah-MP LSM T2 (the faithful overwrite), NOT bulk_t2.
    np.testing.assert_allclose(t2[is_land], nm_t2[is_land], rtol=0, atol=1e-9)
    assert not np.allclose(t2[is_land], bt[is_land], atol=1e-6), \
        "land T2 must be the Noah-MP LSM value, not the bulk surface-layer T2"
    assert np.all(np.isfinite(t2))

    # the LSM T2 is a physically sane near-surface temperature.
    assert np.all((t2[is_land] > 250.0) & (t2[is_land] < 330.0))


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_legacy_three_tuple_without_bulk_t2():
    """bulk_t2=None keeps the legacy (hfx, lh, tsk) 3-tuple for un-wired callers."""
    state, land, static, rad, clock = _build_operational()
    out = overlay_noahmp_land_diagnostics(
        state, land, static, jnp.full((1, 5), 100.0), jnp.full((1, 5), 50.0),
        state.t_skin, 90.0, radiation=rad, clock=clock,
    )
    assert len(out) == 3


def _assemble(state, static, rad, clock):
    """Reproduce the hook's forcing assembly for the reference Noah-MP T2."""
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
    return assemble_noahmp_forcing(_build_column_view(state), static, rad, clock, 90.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
