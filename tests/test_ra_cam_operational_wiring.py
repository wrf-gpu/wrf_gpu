"""Operational scan-wiring of WRF CAM radiation (ra_lw_physics = 3 / ra_sw_physics = 3), lane o1-camrad.

Wiring only (kernel parity vs pristine WRF: tests/test_ra_cam_lw_oracle.py, tests/test_ra_cam_sw_oracle.py):

* the suite resolver accepts CAM LW/SW alone and mixed with RRTMG;
* the radiation-slot dispatch routes ra_lw/ra_sw = 3 to CAM: the held RTHRATEN equals the CAM coupler on the same
  state and differs from RRTMG (deleting the dispatch branch falls through to RRTMG and fails these tests);
* the coupler is JIT-traceable and callback-free;
* the Noah-MP surface forcing refresh hands the land model CAM's GLW / SWDOWN (deletion-sensitive).

The small synthetic column grid of tests/test_rrtm_lw_operational_wiring.py is reused (CPU only).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import cam_radiation as camc
from gpuwrf.coupling.physics_couplers import rrtmg_lw_theta_tendency
from gpuwrf.physics import ra_cam
from gpuwrf.physics.ra_cam_common import cam_ozmixm, load_cam_abs_tables
from gpuwrf.runtime import operational_mode
from gpuwrf.runtime.operational_mode import _physics_step_forcing, _resolve_operational_suite
from gpuwrf.runtime.operational_state import initial_operational_carry

from test_rrtm_lw_operational_wiring import TIME_UTC, _grid, _namelist, _state

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid():
    try:
        load_cam_abs_tables()
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"WRF run/CAM_ABS_DATA unavailable: {exc}")
    return _grid()


@pytest.mark.parametrize("ra_lw, ra_sw", [(3, 3), (3, 4), (4, 3), (3, 0), (0, 3)])
def test_cam_resolves_in_operational_suite(grid, ra_lw, ra_sw):
    _resolve_operational_suite(_namelist(grid, ra_sw_physics=ra_sw, ra_lw_physics=ra_lw))


def _step_rthraten(grid, ra_sw, ra_lw):
    nml = _namelist(grid, ra_sw_physics=ra_sw, ra_lw_physics=ra_lw)
    forcing = _physics_step_forcing(initial_operational_carry(_state(grid)), nml, 0.0, run_radiation=True)
    return np.asarray(forcing.carry.rthraten)


def test_lw_dispatch_routes_to_cam(grid):
    rth_cam = _step_rthraten(grid, 0, 3)
    rth_rrtmg = _step_rthraten(grid, 0, 4)
    assert np.all(np.isfinite(rth_cam)) and np.max(np.abs(rth_cam)) > 0.0
    assert not np.allclose(rth_cam, rth_rrtmg, atol=1e-9)
    state = _state(grid)
    direct = np.asarray(camc.cam_lw_theta_tendency(state, grid, time_utc=TIME_UTC, lead_seconds=0.0))
    ref = np.asarray(rrtmg_lw_theta_tendency(state, grid, time_utc=TIME_UTC, lead_seconds=0.0))
    # cooling of the same sign/order as RRTMG on this warm moist column, but a different scheme
    assert np.sign(direct.mean()) == np.sign(ref.mean())
    assert 0.2 < abs(direct.mean() / ref.mean()) < 5.0


def test_sw_dispatch_routes_to_cam(grid):
    rth_cam = _step_rthraten(grid, 3, 0)
    rth_rrtmg = _step_rthraten(grid, 4, 0)
    assert np.all(np.isfinite(rth_cam)) and np.max(np.abs(rth_cam)) > 0.0
    assert not np.allclose(rth_cam, rth_rrtmg, atol=1e-9)


def test_cam_coupler_equals_kernel_on_its_operands(grid):
    state = _state(grid)
    op = camc.cam_operands(state, grid, time_utc=TIME_UTC, lead_seconds=0.0)
    ozm, pin = cam_ozmixm(op.xlat)
    out = ra_cam.camrad_lw(t_phy=op.t_phy, p_phy=op.p_phy, p8w=op.p8w, pi_phy=op.pi_phy, qv=op.qv, qc=op.qc,
                           qi=op.qi, qs=op.qs, cldfra=op.cldfra, xland=op.xland, xice=op.xice, snow=op.snow,
                           emiss=op.emiss, tsk=op.tsk, xlat=op.xlat, coszen=op.coszen, albedo=op.albedo,
                           julian=op.julian, gases=op.gases, ozmixm=ozm, pin=pin, tables=load_cam_abs_tables())
    rth = np.asarray(camc.cam_lw_theta_tendency(state, grid, time_utc=TIME_UTC, lead_seconds=0.0))
    nz, ny, nx = state.theta.shape
    np.testing.assert_array_equal(rth, np.moveaxis(np.asarray(out.rthratenlw).reshape(ny, nx, nz), -1, 0))
    # WRF operands: hydrostatic interface pressures, REAL julian/solcon, CLWRF gases in double
    assert op.p8w.shape == (ny * nx, nz + 1) and np.all(np.diff(np.asarray(op.p8w), axis=1) < 0)
    assert op.gases.co2vmr.dtype == np.float64 and 3.5e-4 < float(op.gases.co2vmr) < 5.0e-4


def test_cam_coupler_is_jit_traceable_and_callback_free(grid):
    state = _state(grid)
    fn = lambda s: camc.cam_lw_theta_tendency(s, grid, time_utc=TIME_UTC)  # noqa: E731
    assert np.all(np.isfinite(np.asarray(jax.jit(fn)(state))))
    text = str(jax.make_jaxpr(fn)(state))
    for token in ("pure_callback", "io_callback", "host_callback"):
        assert token not in text


@pytest.mark.parametrize("ra_lw, ra_sw", [(3, 4), (4, 3)])
def test_land_surface_forcing_uses_cam(grid, ra_lw, ra_sw):
    """_refresh_noahmp_rad: CAM GLW (ra_lw = 3) / SWDOWN = GSW/(1-ALBEDO) (ra_sw = 3) reach the land model."""

    state = _state(grid)
    nml = dataclasses.replace(_namelist(grid, ra_sw_physics=ra_sw, ra_lw_physics=ra_lw), radiation_cadence_steps=1)
    soldn, lwdn, _cosz = operational_mode._refresh_noahmp_rad(state, nml, 0.0, True, None)
    base = operational_mode._refresh_noahmp_rad(state, dataclasses.replace(nml, ra_sw_physics=4, ra_lw_physics=4),
                                                0.0, True, None)
    op = camc.cam_operands(state, grid, time_utc=TIME_UTC, lead_seconds=0.0)
    if ra_lw == 3:
        glw = np.asarray(camc._lw_call(op).glw).reshape(state.t_skin.shape)
        np.testing.assert_allclose(np.asarray(lwdn), glw, rtol=0, atol=0)
        assert not np.allclose(np.asarray(lwdn), np.asarray(base[1]))
        np.testing.assert_array_equal(np.asarray(soldn), np.asarray(base[0]))
    else:
        swdown = np.asarray(camc._sw_call(op, grid).swdown).reshape(state.t_skin.shape)
        np.testing.assert_allclose(np.asarray(soldn), swdown, rtol=0, atol=0)
        np.testing.assert_array_equal(np.asarray(lwdn), np.asarray(base[1]))


def test_mxaerl_matches_wrf_aerosol_init():
    """aerosol_init counts levels with shalf*1e5 + p_top/1000 >= 9e4 (oracle CAM01: 7 on the WN3 grid)."""

    fx = np.load(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "fixtures"
                 / "cam01-compact-v1.npz")
    assert ra_cam.cam_mxaerl(fx["in_znu"], float(fx["in_p_top"])) == int(fx["mxaerl"])


def test_column_tiling_is_value_identical(grid, monkeypatch):
    """GPUWRF_CAM_TILE_COLS tiling is a pure execution-shape change (9 columns in tiles of 4 vs untiled)."""

    state = _state(grid)
    untiled = np.asarray(camc.cam_lw_theta_tendency(state, grid, time_utc=TIME_UTC))
    monkeypatch.setattr(camc, "CAM_TILE_COLS", 4)
    tiled = np.asarray(camc.cam_lw_theta_tendency(state, grid, time_utc=TIME_UTC))
    np.testing.assert_array_equal(tiled, untiled)


# --------------------------------------------------------------------------------------------------------------------- #
# F1 (rv-camrad): WRF cam_abs_freq_s cadence of the held absorptivities, carry + restart                                 #
# --------------------------------------------------------------------------------------------------------------------- #
def test_cam_doabsems_follows_wrf_stepabs():
    """module_radiation_driver.F:1601: STEPABS = nint(cam_abs_freq_s/(dt*STEPRA))*STEPRA; doabsems at itimestep 1 and
    mod(itimestep, STEPABS) == 1 (radiation calls at itimestep = 1 + k*STEPRA)."""

    due = lambda step, freq, held=None: bool(camc.cam_doabsems((step - 1) * 18.0, 18.0, 100, freq, held=held))
    assert [due(s, 21600.0) for s in (1, 101, 201, 1101, 1201, 1301, 2401)] == [True, False, False, False, True,
                                                                                  False, True]
    assert all(due(s, 1800.0) for s in (1, 101, 201))          # cam_abs_freq_s = radt -> every radiation call
    assert all(due(s, 600.0) for s in (1, 101))                # STEPABS rounds to 0 -> every call (WRF MOD(.,0) undefined)
    zero = ra_cam.CamHeld(np.zeros((4, 3, 3), np.float32), np.zeros((4, 2, 4), np.float32), np.zeros((4, 3), np.float32))
    assert due(101, 21600.0, held=zero)                         # camrad restart guard: all-zero held arrays recompute


def _cadence_step(grid, carry, state, lead, freq):
    nml = dataclasses.replace(_namelist(grid, ra_sw_physics=0, ra_lw_physics=3), cam_abs_freq_s=freq)
    return _physics_step_forcing(carry.replace(state=state), nml, lead, run_radiation=True).carry


def test_held_absorptivities_follow_cam_abs_freq_s_in_the_radiation_slot(grid):
    """dt = 10 s, radiation every step: cam_abs_freq_s = 20 s -> STEPABS = 2 -> itimestep 2 REUSES the held REAL
    abstot/absnxt/emstot of itimestep 1 bit for bit although the state changed; cam_abs_freq_s = 10 s recomputes.
    Deleting the cadence (always fresh) or the carry update fails this test."""

    state = _state(grid)
    warmer = state.replace(theta=state.theta + 1.5, qv=state.qv * 1.2)
    nml0 = _namelist(grid, ra_sw_physics=0, ra_lw_physics=3)
    carry0 = operational_mode._initial_carry_for_run(state, nml0)
    assert carry0.cam_abs is not None and float(np.abs(np.asarray(carry0.cam_abs.abstot)).max()) == 0.0
    c1 = _cadence_step(grid, carry0, state, 0.0, 20.0)                 # itimestep 1: fresh
    assert float(np.abs(np.asarray(c1.cam_abs.abstot)).max()) > 0.0 and c1.cam_abs.abstot.dtype == np.float32
    c2 = _cadence_step(grid, c1, warmer, 10.0, 20.0)                   # itimestep 2: held (mod(2, 2) /= 1)
    for got, held in zip(c2.cam_abs, c1.cam_abs):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(held))
    c2_fresh = _cadence_step(grid, c1, warmer, 10.0, 10.0)             # STEPABS = 1: recompute every call
    assert not np.array_equal(np.asarray(c2_fresh.cam_abs.abstot), np.asarray(c1.cam_abs.abstot))
    assert not np.array_equal(np.asarray(c2_fresh.rthraten), np.asarray(c2.rthraten))


def test_cam_abs_restart_round_trip_is_exact(tmp_path):
    """The held REAL absorptivities survive both restart formats (pickle v2 and wrfrst NetCDF) bit for bit."""

    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.io.restart import read_restart, write_restart
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist

    rgrid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(rgrid).items()})
    nml = OperationalNamelist(grid=rgrid, tendencies=None, metrics=rgrid.metrics, dt_s=18, acoustic_substeps=4,
                              ra_lw_physics=3)
    rng = np.random.default_rng(3)
    held = camc.initial_cam_held(state)
    held = ra_cam.CamHeld(*(jnp.asarray(rng.random(a.shape), jnp.float32) for a in held))
    carry = initial_operational_carry(state).replace(cam_abs=held)
    pkl, nc = tmp_path / "cam.pkl", tmp_path / "cam.nc"
    write_restart(carry, nml, rgrid, 3, pkl)
    write_wrfrst_carry(carry, rgrid, {}, nc, valid_time="2026-07-25_18:00:54", run_start="2026-07-25_18:00:00",
                       step_index=3)
    for restored in (read_restart(pkl)[0], read_wrfrst_carry(nc)[0]):
        assert restored.cam_abs._fields == held._fields
        for got, want in zip(restored.cam_abs, held):
            assert np.asarray(got).dtype == np.float32
            assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
