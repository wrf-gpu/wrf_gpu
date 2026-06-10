"""Sprint S6a coupler tests — land/water masked blend + ocean-path-unchanged.

Validates ``noahmp_coupler.noahmp_surface_adapter`` (ADR-NOAHMP-INTERFACES.md §4):

  1. LAND-MASKED blend: over land the blended PBL-bottom kinematic flux is rebuilt
     from the Noah-MP HFX/QFX; over water it is byte-for-byte the sfclay flux.
  2. OCEAN PATH UNCHANGED (the key no-regression invariant): the water columns'
     blended theta_flux/qv_flux/ustar/tau equal a sfclay-only run exactly — the
     ``where(is_land,...)`` selection never touches a water column.
  3. CH/CM provenance: sfclay CH/CM are SEEDED into the land carry, but Noah-MP
     RE-DERIVES the authoritative land-tile CH/CM (the returned land ch != seed in
     general; sfclay CH is NOT forced over land).
  4. State write-back: t_skin/roughness_m over land carry the Noah-MP values.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

from pathlib import Path

TABLE_DIR = Path("/home/enric/src/wrf_pristine/WRF/run")
HAVE_TABLES = (TABLE_DIR / "MPTABLE.TBL").exists()


class _State:
    """Minimal column State namespace with a functional ``replace``."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def replace(self, **updates):
        d = dict(self.__dict__)
        d.update(updates)
        return _State(**d)


def _build():
    # 5 columns: 3 land (veg5, veg9, veg10), 2 water. (nx=5 != NSOIL=4 avoids the
    # degenerate nx==NSOIL broadcast collision in the soil-thermo solve.)
    n = 5
    u = np.array([6.0, 7.0, 8.0, 9.0, 5.0])
    v = np.array([2.0, 1.0, 0.0, -1.0, 0.5])
    t = np.array([292.0, 295.0, 296.0, 294.0, 293.0])
    qv = np.array([0.008, 0.009, 0.010, 0.010, 0.011])
    p = np.array([95500.0, 95000.0, 94800.0, 100800.0, 100500.0])
    dz = np.array([80.0, 80.0, 80.0, 60.0, 60.0])
    tsk = np.array([296.0, 300.0, 301.0, 293.0, 292.5])
    xland = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    znt = np.array([0.08, 0.05, 0.06, 0.0015, 0.0015])
    theta = t * (P0_PA / p) ** R_D_OVER_CP
    shape = (1, n)

    def s2(a):
        return jnp.asarray(a.reshape(shape))

    def col(a):
        return jnp.asarray(a.reshape(shape + (1,)))

    state = _State(
        u=col(u), v=col(v), theta=col(theta), qv=col(qv), p=col(p), dz=col(dz),
        t_skin=s2(tsk), xland=s2(xland), roughness_m=s2(znt),
        mavail=s2(np.array([0.7, 0.6, 0.6, 1.0, 1.0])), ustar=s2(np.full(n, 0.1)),
    )

    # land carry over all 5 columns (water columns are masked out by the blend).
    z = jnp.zeros(shape)
    veg = np.array([5, 9, 10, 5, 9])
    soil = np.array([6, 6, 6, 6, 6])
    land = NoahMPLandState(
        tslb=jnp.broadcast_to(jnp.asarray(295.0), (NSOIL, 1, n)),
        smois=jnp.broadcast_to(jnp.asarray(0.20), (NSOIL, 1, n)),
        sh2o=jnp.broadcast_to(jnp.asarray(0.20), (NSOIL, 1, n)),
        smcwtd=jnp.broadcast_to(jnp.asarray(0.20), shape),
        isnow=jnp.zeros(shape, dtype=jnp.int32),
        tsno=jnp.broadcast_to(jnp.asarray(273.0), (NSNOW, 1, n)),
        snice=jnp.broadcast_to(z, (NSNOW, 1, n)), snliq=jnp.broadcast_to(z, (NSNOW, 1, n)),
        zsnso=jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 0.0, -0.05, -0.25, -0.7, -1.5]).reshape(NSNOW + NSOIL, 1, 1),
            (NSNOW + NSOIL, 1, n)),
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
        isltyp=jnp.asarray(soil.reshape(shape), dtype=jnp.int32),
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
def test_ocean_path_unchanged_and_land_blend():
    state, land, static, rad, clock = _build()
    is_land = np.asarray((state.xland - 1.5) < 0.0).reshape(-1)

    # sfclay-only reference (the byte-for-byte ocean baseline).
    diag = surface_layer_with_diagnostics(state)
    sf_ref = diag.fluxes

    state_out, land_out, blended = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90.0)

    # 1. finiteness
    for v in (blended.theta_flux, blended.qv_flux, blended.ustar, blended.fltv):
        assert np.all(np.isfinite(np.asarray(v)))

    # 2. OCEAN PATH UNCHANGED: water columns' blended kinematic flux == sfclay.
    tf_b = np.asarray(blended.theta_flux).reshape(-1)
    qf_b = np.asarray(blended.qv_flux).reshape(-1)
    tf_r = np.asarray(sf_ref.theta_flux).reshape(-1)
    qf_r = np.asarray(sf_ref.qv_flux).reshape(-1)
    water = ~is_land
    np.testing.assert_allclose(tf_b[water], tf_r[water], rtol=0, atol=1e-12)
    np.testing.assert_allclose(qf_b[water], qf_r[water], rtol=0, atol=1e-12)
    # momentum (ustar/tau) is sfclay-owned everywhere (opt_sfc=1) -> identical.
    np.testing.assert_allclose(np.asarray(blended.ustar).reshape(-1),
                               np.asarray(sf_ref.ustar).reshape(-1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(blended.tau_u).reshape(-1),
                               np.asarray(sf_ref.tau_u).reshape(-1), rtol=0, atol=1e-12)

    # 3. LAND blend differs from the sfclay water-baseline on the land columns
    #    (Noah-MP HFX replaced the sfclay HFX there).
    assert not np.allclose(tf_b[is_land], tf_r[is_land], atol=1e-9), \
        "land theta_flux should reflect Noah-MP HFX, not sfclay"

    # 4. CH/CM provenance: Noah-MP re-derived the land-tile CH (not the 0.01 seed).
    ch_land = np.asarray(land_out.ch).reshape(-1)[is_land]
    assert np.all(np.isfinite(ch_land))
    assert not np.allclose(ch_land, 0.01, atol=1e-9), \
        "Noah-MP must re-derive land CH, not keep the sfclay seed verbatim"

    # 5. write-back: land t_skin = Noah-MP TRAD; water t_skin = prescribed SST.
    tsk_out = np.asarray(state_out.t_skin).reshape(-1)
    tsk_in = np.asarray(state.t_skin).reshape(-1)
    np.testing.assert_allclose(tsk_out[water], tsk_in[water], rtol=0, atol=1e-9)
    assert np.all(np.isfinite(tsk_out[is_land]))


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_adapter_runs_end_to_end():
    state, land, static, rad, clock = _build()
    state_out, land_out, blended = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90.0)
    # the advanced land carry + blended fluxes are all finite
    for leaf in jax.tree_util.tree_leaves(land_out):
        assert np.all(np.isfinite(np.asarray(leaf)))
    for leaf in jax.tree_util.tree_leaves(blended):
        assert np.all(np.isfinite(np.asarray(leaf)))


@pytest.mark.skipif(not HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_first_timestep_threads_into_blend_sfclay():
    """v0.14 NoahMP Step-1 closure: ``first_timestep`` must reach the sfclay run
    INSIDE the blend (it previously always ran the warm-call branch, so the WRF
    MYNN surface first-call semantics never engaged on the Noah-MP path)."""
    state, land, static, rad, clock = _build()
    is_land = np.asarray((state.xland - 1.5) < 0.0).reshape(-1)
    water = ~is_land

    # default (no kwarg) == explicit False, bit-for-bit (no-regression invariant).
    _, _, blended_default = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90.0)
    _, _, blended_false = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90.0, first_timestep=False)
    np.testing.assert_array_equal(np.asarray(blended_default.ustar),
                                  np.asarray(blended_false.ustar))
    np.testing.assert_array_equal(np.asarray(blended_default.theta_flux),
                                  np.asarray(blended_false.theta_flux))

    # first_timestep=True engages the first-call branch: water columns must equal
    # a direct first-call sfclay run, and differ from the warm-call blend.
    _, _, blended_first = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90.0, first_timestep=True)
    sf_first = surface_layer_with_diagnostics(state, first_timestep=True).fluxes
    np.testing.assert_allclose(
        np.asarray(blended_first.ustar).reshape(-1)[water],
        np.asarray(sf_first.ustar).reshape(-1)[water], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(blended_first.theta_flux).reshape(-1)[water],
        np.asarray(sf_first.theta_flux).reshape(-1)[water], rtol=0, atol=1e-12)
    assert not np.allclose(np.asarray(blended_first.ustar),
                           np.asarray(blended_false.ustar), atol=1e-12), \
        "first-call branch must change the cold-start sfclay solution"
