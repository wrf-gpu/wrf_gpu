"""WRF previous-step rain wiring, REAL rates, source ownership and restart."""
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step, seed_precipitation


def test_instantaneous_amounts_feed_total_and_frozen_rates_without_accumulator_cancellation():
    precip = {name: jnp.full((4, 5), value, jnp.float32)
              for name, value in {'rain': .18, 'snow': .09, 'ice': .018, 'graupel': .036}.items()}
    rates = precipitation_from_step(precip, jnp.full((4, 5), .002), 18, spec_zone=1)
    assert all(x.dtype == jnp.float32 for x in rates)
    for name, expected in [('prcpconv', .002), ('prcpnonc', .018), ('prcpsnow', .006), ('prcpgrpl', .002), ('prcphail', 0)]:
        value = np.asarray(getattr(rates, name))
        np.testing.assert_allclose(value[1:-1, 1:-1], expected, atol=2e-9, rtol=0)
        if name != 'prcpconv':
            assert np.all(value[[0, -1]] == 0) and np.all(value[:, [0, -1]] == 0)
        else:
            np.testing.assert_array_equal(value, np.full((4, 5), .002, np.float32))
    # These REAL increments would vanish when differencing a large rounded total.
    assert np.float32(1e8 + .18) - np.float32(1e8) == 0
    assert float(rates.prcpnonc[1, 1]) > 0


def test_surface_hook_passes_held_previous_precipitation_into_the_real_forcing_assembler(monkeypatch):
    from test_noahmp_coupler import _build
    import gpuwrf.coupling.noahmp_surface_hook as hook
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
    state, land, static, rad, clock = _build()
    # A real column view avoids unrelated operational geometry in this wiring test.
    view = SimpleNamespace(**state.__dict__)
    view.psfc = jnp.full((1, 5), 95000.)
    view.qc = jnp.zeros((1, 5, 1))
    view.replace = lambda **updates: SimpleNamespace(**dict(view.__dict__, **updates))
    monkeypatch.setattr(hook, '_build_column_view', lambda *a, **kw: view)
    rates = seed_precipitation(SimpleNamespace(t_skin=jnp.zeros((1, 5))))
    rates = rates._replace(prcpnonc=jnp.full((1, 5), .001), prcpconv=jnp.full((1, 5), .0002))
    seen = []
    def adapter(column, old_land, config, **kw):
        forcing = assemble_noahmp_forcing(column, config, rad, clock, 18)
        seen.append(forcing)
        raise RuntimeError('forcing captured')
    monkeypatch.setattr(hook, 'noahmp_surface_adapter', adapter)
    with pytest.raises(RuntimeError, match='forcing captured'):
        hook.noahmp_surface_step(state, land, static, 18, precipitation=rates)
    np.testing.assert_array_equal(seen[0].prcpnonc, rates.prcpnonc)
    np.testing.assert_array_equal(seen[0].prcpconv, rates.prcpconv)


def test_post_rk_mp_retains_actual_precipitation_side_channel(monkeypatch):
    import gpuwrf.runtime.operational_mode as mode
    from test_m7_netcdf_writer import synthetic_case
    state, _, _ = synthetic_case()
    amounts = {name: jnp.full((4, 5), value) for name, value in {'rain': .001, 'snow': .002, 'ice': .003, 'graupel': .004}.items()}
    calls = []
    def adapter(before, dt, **kwargs):
        calls.append(kwargs.get('return_precipitation'))
        return (before, amounts) if kwargs.get('return_precipitation') else before
    monkeypatch.setattr(mode, 'thompson_adapter', adapter)
    monkeypatch.setattr(mode, '_microphysics_interior_only', lambda before, after, nml: after)
    actual, precip = mode._apply_post_rk_microphysics(state, SimpleNamespace(run_physics=True, mp_physics=8, dt_s=18),
                                                     return_precipitation=True)
    assert actual is state and precip is amounts and calls == [True]


@pytest.mark.parametrize('temperature,raw_snow,expect_snow', [(278., .001, False), (270., 0., True)])
def test_energy_receives_ground_jordan_snow_instead_of_raw_microphysics(monkeypatch, temperature, raw_snow, expect_snow):
    from test_noahmp_coupler import _build
    import gpuwrf.physics.noahmp.noahmp_driver as driver
    from gpuwrf.physics.noahmp_coupler import assemble_noahmp_forcing
    state, land, static, rad, clock = _build()
    forcing = assemble_noahmp_forcing(state, static, rad, clock, 18)._replace(
        sfctmp=jnp.full((1, 5), temperature), prcpnonc=jnp.full((1, 5), .001),
        prcpsnow=jnp.full((1, 5), raw_snow))
    captured = []
    def capture(old_land, energy_forcing, *args, **kwargs):
        captured.append(energy_forcing.prcpsnow)
        raise RuntimeError('radiation input captured')
    monkeypatch.setattr(driver, 'radiation_twostream', capture)
    with pytest.raises(RuntimeError, match='radiation input captured'):
        driver.noah_mp_step(land, forcing, static, 18)
    snow = np.asarray(captured[0])
    if expect_snow:
        assert np.all(snow > 0) and np.all(snow <= .001)
    else:
        np.testing.assert_array_equal(snow, 0)


def test_precipitation_restart_is_exact_and_old_active_land_schema_fails_closed(tmp_path):
    import pickle
    from netCDF4 import Dataset
    from test_writer_land_history import _land
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.io.restart import read_restart, write_restart
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})
    nml = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18, acoustic_substeps=4)
    precipitation = seed_precipitation(state)._replace(prcpnonc=jnp.full(state.t_skin.shape, .003, jnp.float32))
    carry = initial_operational_carry(state).replace(noahmp_land=_land(state.t_skin.shape), noahmp_precipitation=precipitation)
    pkl, nc = tmp_path / 'precip.pkl', tmp_path / 'precip.nc'
    write_restart(carry, nml, grid, 3, pkl)
    write_wrfrst_carry(carry, grid, {}, nc, valid_time='2026-07-25_18:00:54', run_start='2026-07-25_18:00:00', step_index=3)
    for restored in (read_restart(pkl)[0], read_wrfrst_carry(nc)[0]):
        assert restored.noahmp_precipitation._fields == precipitation._fields
        for actual, expected in zip(restored.noahmp_precipitation, precipitation):
            assert np.asarray(actual).dtype == np.float32
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    with pkl.open('rb') as stream:
        payload = pickle.load(stream)
    payload['format_version'] = 4
    with pkl.open('wb') as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match='precipitation.*E78'):
        read_restart(pkl)
    with Dataset(nc, 'r+') as ds:
        ds.GPUWRF_WRFRST_SCHEMA_VERSION = 'v0.25-wrfrst-netcdf-4'
    with pytest.raises(ValueError, match='schema'):
        read_wrfrst_carry(nc)


def test_operational_init_seeds_previous_precipitation_independently_of_output_mode(monkeypatch):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.runtime.operational_mode import OperationalNamelist, _initial_carry_for_run
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})
    nml = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, dt_s=18,
        acoustic_substeps=4, use_noahmp=True, run_physics=True, mp_physics=8,
        ra_sw_physics=0, ra_lw_physics=0, cu_physics=0, bl_pbl_physics=0)
    monkeypatch.setenv('GPUWRF_FULL_WRFOUT_VARIABLES', '0')
    carry = _initial_carry_for_run(state, nml)
    assert carry.land_history is None
    assert carry.noahmp_precipitation is not None
    assert carry.noahmp_precipitation._fields == seed_precipitation(state)._fields
    for rate in carry.noahmp_precipitation:
        assert rate.dtype == jnp.float32 and rate.shape == state.t_skin.shape
        np.testing.assert_array_equal(rate, 0)
