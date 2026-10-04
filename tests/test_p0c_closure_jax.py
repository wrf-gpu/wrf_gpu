"""P0 closure T tests (contract P0_CLOSURE_CONTRACT.md rev 2 §2 T; critic 2 A.8 M1-M4, M11, M13).  JAX; run only in
the reviewed L1 package.  Each test names the mutant it kills.

M1   full-step clear: a thin pre-existing cloud fully evaporates in the branch while another level condenses fresh
     cloud -> WRF ANY(L_qc) is False -> no cloud sedimentation.  Kills a branch evaluated on the post-adjustment state.
M2   full-step mixed column: one kept cloudy level + one freshly condensing level -> ANY(L_qc) True -> the cloud flux
     leaves the column.  Kills ALL(L_qc) in place of ANY.
M3   mp=28: the column gate masks mass AND number flux; a cloud-free entry keeps its fresh condensate; the aero branch
     threshold is qc > R1.  Kills dropping cloud_sed_on, masking vtc but not vtnc, and reverting to qc > 0.
M4   per-level density test rc = MAX(R1, qc*rho) > R1 (rho < 1), with rc from the pre-condensation rho stage (K5).
     Kills a mixing-ratio qc > R1 test and rc from the post-adjustment rho.
M11  R2/R3 (+K1) at the Thompson builders: dz = (PH+PHB)/9.81, bottom-face w, T = theta_dry*(p/p0)**(2/7), for mp=8
     and mp=28.  Kills 9.80665, mass-point w and kappa = 287/1004.
M13  the KF and GWDO adapters add no host callback (no step-loop transfer).
"""

from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.physics import thompson_aero_column as tac
from gpuwrf.physics import thompson_column as tc

# Gate-1 helpers by FILE PATH from this test's directory (never a bare ``tests`` package; P0REG r1 incident cf8ca43df).
_LP_SPEC = importlib.util.spec_from_file_location(
    "_p0c_live_path_helpers", pathlib.Path(__file__).resolve().parent / "test_p0_moist_theta_live_path.py")
_lp = importlib.util.module_from_spec(_LP_SPEC)
_LP_SPEC.loader.exec_module(_lp)
_dry, _grid, _state = _lp._dry, _lp._grid, _lp._state

DT = 18.0
CALLBACKS = ("callback", "io_callback", "pure_callback", "debug_callback", "host_callback")


def _levels(nz=6):
    k = np.arange(nz, dtype=np.float64)
    return 97614.39 - 700.0 * k, 289.0 - 0.5 * k


def _column(qv, qc, *, w=0.005, dz=50.16):
    """mp=8 column (1 x nz), warm levels; rho from the WRF EOS form."""
    qv, qc = np.asarray(qv, np.float64), np.asarray(qc, np.float64)
    p, T = _levels(qv.size)
    rho = np.asarray(tc.density_from_pressure_temperature(p, T, qv))
    z = jnp.zeros((1, qv.size), jnp.float64)
    a = lambda x: jnp.asarray(np.asarray(x, np.float64)[None])   # noqa: E731
    return tc.ThompsonColumnState(a(qv), a(qc), z, z, z, z, z, z, a(T), a(p), a(rho), Ns=z, Ng=z,
                                  dz=a(np.full(qv.size, dz)), w=a(np.full(qv.size, w)))


def _aero_column(qv, qc, *, nc=1.0e8, nwfa=1.0e9, w=0.005, dz=50.16):
    """mp=28 column (1 x nz): the mp=8 column plus droplet number and aerosol (per kg)."""
    base = _column(qv, qc, w=w, dz=dz)
    n = base.qv.shape[-1]
    z = jnp.zeros((1, n), jnp.float64)
    return tac.ThompsonAeroColumnState(
        base.qv, base.qc, z, z, z, z, z, z, jnp.where(base.qc > 0.0, nc, 0.0), jnp.full((1, n), nwfa),
        jnp.full((1, n), 1.0e6), base.T, base.p, base.rho, Ns=z, Ng=z, dz=base.dz, w=base.w)


# ------------------------------------------------------------------ M1
def test_m1_full_step_clear_rule_keeps_fresh_cloud_unsedimented():
    """Level 0 condenses fresh cloud (qc_pre = 0: never sets L_qc); level 2 carries a thin cloud (1e-6) in strongly
    subsaturated air that fully evaporates inside the branch (WRF :3484-3485 clears L_qc).  ANY(L_qc) = False, so no
    cloud sedimentation.  Mutant killed: branch evaluated AFTER the adjustment (the evaporated level fails the
    predicate, is not cleared, and the gate turns on -> the level-0 flux leaves the column, cloudw > 0)."""
    col = _column([0.0130, 0.009, 0.004, 0.009, 0.009, 0.009], [0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0])
    out, precip = tc.step_thompson_column_with_precip(col, DT)
    assert float(out.qc[0, 2]) <= tc.R1                    # the thin cloud fully evaporated
    assert float(out.qc[0, 0]) > 1.0e-4                    # fresh condensation at level 0
    assert float(np.asarray(precip["cloudw"]).max()) == 0.0
    water = lambda s: float(s.qv[0, 0] + s.qc[0, 0] + s.qr[0, 0])   # noqa: E731
    assert abs(water(out) - water(col)) < 1.0e-9


# ------------------------------------------------------------------ M2
def test_m2_full_step_mixed_column_sediments_through_any_l_qc():
    """Level 0 keeps a pre-existing cloud (5e-4, supersaturated: L_qc set and kept); level 1 condenses fresh cloud.
    ANY(L_qc) = True, so the cloud flux leaves the column.  Mutant killed: ALL(L_qc) (False here -> cloudw == 0)."""
    col = _column([0.0130, 0.0125, 0.009, 0.009, 0.009, 0.009], [5.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    out, precip = tc.step_thompson_column_with_precip(col, DT)
    assert float(out.qc[0, 1]) > 1.0e-5                    # level 1 condensed fresh cloud
    assert float(np.asarray(precip["cloudw"])[0]) > 0.0
    gate = tc._wrf_l_qc_any(col.qc, col.rho, out.qc, jnp.asarray([[True, True] + [False] * 4]))
    assert bool(gate[0, 0]) and not bool(jnp.all(col.qc > tc.R1))


# ------------------------------------------------------------------ M3 (mp=28)
def test_m3_aero_gate_masks_cloud_mass_and_number_flux():
    """_sed_cloud_water_aero with the column gate off leaves qc AND Nc bit-identical (and cloudw = 0); with it on
    both move.  Mutants killed: dropping cloud_sed_on (qc moves when off) and masking vtc but not vtnc (Nc moves)."""
    col = _aero_column([0.0120, 0.0115, 0.009, 0.009, 0.009, 0.009], [5.0e-4, 3.0e-4, 0.0, 0.0, 0.0, 0.0])
    off_qc, off_nc, off_loss = tac._sed_cloud_water_aero(col, DT, tac.THOMPSON_AERO_TABLES, jnp.asarray([[False]]))
    assert np.array_equal(np.asarray(off_qc), np.asarray(col.qc))
    assert np.array_equal(np.asarray(off_nc), np.asarray(col.Nc))
    assert float(off_loss[0]) == 0.0
    on_qc, on_nc, on_loss = tac._sed_cloud_water_aero(col, DT, tac.THOMPSON_AERO_TABLES, jnp.asarray([[True]]))
    assert not np.array_equal(np.asarray(on_qc), np.asarray(col.qc))
    assert not np.array_equal(np.asarray(on_nc), np.asarray(col.Nc))
    assert float(on_loss[0]) > 0.0


def test_m3_aero_full_step_cloud_free_entry_has_no_cloud_sedimentation():
    """mp=28 full step, cloud-free entry with a supersaturated bottom level: fresh cloud never sets L_qc, so cloudw
    stays 0.  Mutant killed: the body/_sedimentation_aero ignoring cloud_sed_on."""
    col = _aero_column([0.0130, 0.009, 0.009, 0.009, 0.009, 0.009], [0.0] * 6)
    out, precip = tac.step_thompson_aero_column_with_precip(col, DT)
    assert float(out.qc[0, 0]) > 1.0e-5
    assert float(np.asarray(precip["cloudw"]).max()) == 0.0


def test_m3_aero_condensation_branch_needs_qc_above_r1():
    """mp=28 saturation adjustment: qc in (0, R1] in subsaturated air is outside WRF's branch (:3401-3402 needs
    L_qc), so it is untouched.  Mutant killed: the old qc > 0 threshold (it would evaporate the 5e-13)."""
    col = _aero_column([0.004] * 6, [5.0e-13, 0.0, 0.0, 0.0, 0.0, 0.0])
    out, _condensed = tac._saturation_adjustment_aero(col, DT)
    assert float(out.qc[0, 0]) == 5.0e-13


# ------------------------------------------------------------------ M4
def test_m4_fall_speed_tests_the_density_rc_with_the_pre_condensation_rho():
    """WRF :3657 tests rc = MAX(R1, qc*rho) > R1 per level; at rho = 0.5 a qc of 1.5e-12 (> R1 as a mixing ratio)
    gives rc = R1 -> no fall speed, while 3e-12 falls.  With K5 the rc density is the pre-condensation stage: state rho
    1.0 but rho_rc 0.5 must still give zero.  Mutants killed: a mixing-ratio qc > R1 test, and rc from state.rho."""
    col = _column([0.009, 0.009], [1.5e-12, 3.0e-12]).replace(rho=jnp.full((1, 2), 0.5))
    vtc = np.asarray(tc._cloud_water_fall_speed(col))
    assert vtc[0, 0] == 0.0 and vtc[0, 1] > 0.0
    col1 = col.replace(rho=jnp.full((1, 2), 1.0))
    staged = np.asarray(tc._cloud_water_fall_speed(col1, rho_rc=jnp.full((1, 2), 0.5), rho_f=col1.rho))
    assert staged[0, 0] == 0.0 and staged[0, 1] > 0.0
    assert np.asarray(tc._cloud_water_fall_speed(col1))[0, 0] > 0.0        # the post-adjustment rho would fall


# ------------------------------------------------------------------ M11 (+K1)
def _state_with_faces():
    """mp=28-ready State (p0cl1_r1 finding 2): the production preparation thompson_aero_adapter applies before its builder
    (State.ensure_conditional_leaves(mp_physics=28)), then physically valid aerosol numbers (the port has no nbca leaf:
    mp=28 runs with wif_input_opt=1, nbca inert).  The dz/w/T assertions and their mutants do not depend on them."""
    grid = _grid()
    state = _state(grid, 0.012).ensure_conditional_leaves(mp_physics=28)
    w = np.linspace(-0.2, 0.4, grid.nz + 1)[:, None, None] * np.ones((1, grid.ny, grid.nx))
    return grid, state.replace(w=jnp.asarray(w), nwfa=jnp.full_like(state.qc, 1.0e9), nifa=jnp.full_like(state.qc, 1.0e6))


def test_m11_thompson_builders_use_wrf_dz_bottom_face_w_and_rcp():
    """mp=8 and mp=28 builders: dz = (PH+PHB)/9.81 (BSU :4869/:4877), w = the bottom face (MPT :1224), T = dry theta *
    (p/p0)**(2/7) (phy_prep).  The fixture's ph uses 800 m * 9.80665, so dz = 799.72 m, not 800 m.  Mutants killed:
    9.80665 (dz = 800), mass-point w (0.5*(w_k+w_k+1)), kappa = 287/1004."""
    grid, state = _state_with_faces()
    ph = np.asarray(state.ph, np.float64)
    want_dz = np.moveaxis(np.maximum(ph[1:] / 9.81 - ph[:-1] / 9.81, 1.0), 0, -1)
    want_w = np.moveaxis(np.asarray(state.w, np.float64)[:-1], 0, -1)
    want_t = np.moveaxis(_dry(state) * (np.asarray(state.p, np.float64) / 1.0e5) ** (2.0 / 7.0), 0, -1)
    for column in (pc._thompson_column_from_state(state, grid), pc._thompson_aero_column_from_state(state, grid)):
        np.testing.assert_allclose(np.asarray(column.dz), want_dz, rtol=1e-12, atol=0)
        assert np.array_equal(np.asarray(column.w), want_w)
        np.testing.assert_allclose(np.asarray(column.T), want_t, rtol=1e-12, atol=0)
    assert abs(float(want_dz.ravel()[0]) - 800.0) > 0.2                      # discriminates 9.81 vs 9.80665


# ------------------------------------------------------------------ M13
def test_m13_kf_and_gwdo_adapters_add_no_host_callback():
    """The P0-coupled KF and GWDO adapters trace without a host callback (no step-loop host/device transfer)."""
    grid = _grid()
    state = _state(grid, 0.012)
    zeros3 = jnp.zeros((grid.nz, grid.ny, grid.nx))
    zeros2 = jnp.zeros((grid.ny, grid.nx))
    # p0cl1_r1 finding 3: gwdo_adapter requires GWDOStatics (gwdo_columns reads statics.var); a valid flat bundle as in
    # tests/test_gwd_gwdo.py (build_gwdo_statics_from_wrf_fields; var = 0, con = 1, oa1 = 0.4, ol* = 0.3).
    z2, o2 = jnp.zeros((grid.ny, grid.nx)), jnp.ones((grid.ny, grid.nx))
    statics = pc.build_gwdo_statics_from_wrf_fields(
        var2d=z2, con=o2, oa1=jnp.full_like(z2, 0.4), oa2=z2, oa3=z2, oa4=z2, ol1=jnp.full_like(z2, 0.3),
        ol2=jnp.full_like(z2, 0.3), ol3=jnp.full_like(z2, 0.3), ol4=jnp.full_like(z2, 0.3), dx_m=3000.0)
    texts = (
        str(jax.make_jaxpr(lambda s: sa.kf_adapter(s, 54.0, zeros3, zeros2, grid=grid))(state)),
        str(jax.make_jaxpr(lambda s: pc.gwdo_adapter(s, 54.0, statics, grid))(state)),
    )
    for text in texts:
        for primitive in CALLBACKS:
            assert primitive not in text, primitive
