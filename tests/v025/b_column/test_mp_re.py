"""MP_RE producer lifetime, mediation boundaries, restart and disabled path.

Independent pristine radiation gate: mp_re_probe.py on RE01's 186 real columns.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import rrtmg_mp_re as R
from gpuwrf.physics.thompson_column import ThompsonColumnState
from gpuwrf.io.restart import _validate_state_field_order, _state_fields


def _real_state():
    from mp_re_probe import load
    fx = load()
    cols = np.array([0, 31, 78, 114])
    ncol, nz = len(cols), fx["in_t"].shape[1]
    grid = SimpleNamespace(nx=ncol, ny=1, nz=nz)
    state = State(**{k: jnp.zeros(shape, DEFAULT_DTYPES.dtype_for(k)) for k, shape in _state_field_shapes(grid).items()})
    a = lambda name: jnp.asarray(fx[name][cols], jnp.float32)[None]
    mass = lambda name: jnp.moveaxis(a(name), -1, 0)
    state = state.replace(p=mass("in_p"), theta=mass("in_th"), qv=mass("in_qv"),
                          qc=mass("in_qc"), qi=mass("in_qi"), qs=mass("in_qs"), Ni=mass("in_ni"),
                          xland=a("in_xland"), t_skin=a("in_tsk"))
    out = ThompsonColumnState(*(a("in_" + name) for name in ("qv", "qc", "qr", "qi", "qs", "qg", "ni")),
                              jnp.zeros_like(a("in_ni")), a("in_t"), a("in_p"), a("in_rho"))
    return state, out


def test_previous_mp_radii_reach_both_radiation_columns(monkeypatch):
    monkeypatch.setattr(R, "_MP_RE", True)
    before, mp = _real_state()
    before = before.ensure_conditional_leaves(mp_physics=8)
    for name, bg in (("re_cloud", 2.49e-6), ("re_ice", 4.99e-6), ("re_snow", 9.99e-6)):
        np.testing.assert_array_equal(getattr(before, name), jnp.full_like(before.qc, bg))
    # End of step1: shared native/reference Thompson writeback calls the producer.
    after = C._state_from_thompson_output(before, mp)
    expected = R.calc_thompson_effective_radii(mp.T, mp.p, mp.qv, mp.qc, mp.qi, mp.Ni, mp.qs)
    # Transport/physics can change next step's hydrometeors before radiation.
    step2 = after.replace(qc=jnp.zeros_like(after.qc), qi=jnp.zeros_like(after.qi), qs=jnp.zeros_like(after.qs))
    sw, lw, *_ = C._rrtmg_column_inputs(step2, None, time_utc="2026-02-28_00:00:00")
    for name, radius in zip(("re_cloud", "re_ice", "re_snow"), expected):
        np.testing.assert_array_equal(getattr(sw, name), radius)
        np.testing.assert_array_equal(getattr(lw, name), radius)
        assert getattr(after, name).dtype == jnp.float32
    assert np.any(np.asarray(after.re_cloud) != np.float32(R.RE_CLOUD_BG))
    assert np.any(np.asarray(after.re_snow) != np.float32(R.RE_SNOW_BG))


def test_disabled_path_keeps_absent_leaves_and_fixed_operator(monkeypatch):
    monkeypatch.setattr(R, "_MP_RE", False)
    state, mp = _real_state()
    state = state.ensure_conditional_leaves(mp_physics=8)
    assert state.re_cloud is state.re_ice is state.re_snow is None
    after = C._state_from_thompson_output(state, mp)
    sw, lw, *_ = C._rrtmg_column_inputs(after, None, time_utc="2026-02-28_00:00:00")
    assert sw.re_cloud is lw.re_cloud is None


def test_enabled_missing_saved_radii_fails_closed(monkeypatch):
    monkeypatch.setattr(R, "_MP_RE", True)
    state, _ = _real_state()
    with pytest.raises(ValueError, match="held Thompson radii"):
        C._rrtmg_column_inputs(state, None, time_utc="2026-02-28_00:00:00")


def test_use_mp_re_zero_disables_candidate_and_static_config_roundtrips(monkeypatch):
    monkeypatch.setattr(R, "_MP_RE", True)
    state, _ = _real_state()
    assert state.ensure_conditional_leaves(mp_physics=8, use_mp_re=0).re_cloud is None
    sw, lw, *_ = C._rrtmg_column_inputs(state, None, use_mp_re=0, time_utc="2026-02-28_00:00:00")
    assert sw.re_cloud is lw.re_cloud is None


def test_restart_payload_keeps_exact_held_radii_and_refuses_missing(monkeypatch):
    monkeypatch.setattr(R, "_MP_RE", True)
    state, mp = _real_state()
    state = C._state_from_thompson_output(state.ensure_conditional_leaves(mp_physics=8), mp)
    fields = _state_fields(state)
    _validate_state_field_order(tuple(fields))
    restored = State.tree_unflatten(None, tuple(jnp.asarray(fields[n]) if n in fields else None for n in State.__slots__))
    for name in ("re_cloud", "re_ice", "re_snow"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(state, name))
    with pytest.raises(ValueError, match="missing held Thompson"):
        _validate_state_field_order(tuple(n for n in fields if not n.startswith("re_")))
    monkeypatch.setattr(R, "_MP_RE", False)
    _validate_state_field_order(tuple(n for n in fields if not n.startswith("re_")))


def test_radiation_fallbacks_and_large_snow_mass_reduction():
    bg = jnp.array([[R.RE_CLOUD_BG] * 3], jnp.float32)
    r = R.prepare_radiation_radii(jnp.array([[250.25, 250.25, 250.25]]), jnp.array([[1., 1., 0.]]),
                                 jnp.array([[2., 1., 2.]]), bg, jnp.full_like(bg, R.RE_ICE_BG),
                                 jnp.array([[9.99e-6, 130e-6, 260e-6]], jnp.float32))
    np.testing.assert_array_equal(r.liquid_um, [[10.5, 7.5, 2.5]])
    np.testing.assert_array_equal(r.snow_um, [[10., 130., 130.]])
    np.testing.assert_allclose(r.snow_mass_factor, [[.99, .99, .25]], rtol=0, atol=1e-7)
    assert r.ice_um[0, 0] == r.ice_um[0, 1] and r.ice_um[0, 2] == 5


def test_radii_are_restart_only_and_do_not_change_history_schema(tmp_path, monkeypatch):
    from datetime import datetime
    import importlib.util
    from pathlib import Path
    from netCDF4 import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout, FULL_WRFOUT_VARIABLES
    from gpuwrf.validation.tier2 import make_ideal_grid
    spec = importlib.util.spec_from_file_location("mp_re_writer_fixture", Path(__file__).parents[2] / "test_m7_netcdf_writer.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    monkeypatch.setattr(R, "_MP_RE", True)
    before, mp = _real_state()
    state = C._state_from_thompson_output(before.ensure_conditional_leaves(mp_physics=8), mp)
    grid = make_ideal_grid(*state.theta.shape)
    names = ("RE_CLOUD", "RE_ICE", "RE_SNOW", "RE_CLOUD_GSFC", "RE_ICE_GSFC", "RE_SNOW_GSFC")
    inventory = []
    for label, carry in (("on", state), ("off", state.replace(re_cloud=None, re_ice=None, re_snow=None))):
        path = tmp_path / (label + ".nc")
        prepared = prepare_wrfout_payload(
            carry, grid, None, path, domain="d02", domain_authority=fixture.writer_authority(grid),
            valid_time=datetime(2026, 2, 28, 1), lead_hours=1, run_start=datetime(2026, 2, 28), full_variable_set=True,
        )
        write_prepared_wrfout(prepared, expected_domain="d02", expected_domain_authority=fixture.writer_authority(grid))
        with Dataset(path) as dataset:
            inventory.append(set(dataset.variables))
            assert not set(names).intersection(dataset.variables)
    assert inventory[0] == inventory[1] == set(FULL_WRFOUT_VARIABLES)
    # The source authority explicitly gives these diagnostics restart IO only.
    registry = (Path(__file__).parents[3] / "data/wrf_pristine/WRF/Registry/Registry.EM_COMMON").read_text()
    for name in ("re_cloud", "re_ice", "re_snow"):
        row = next(line.split() for line in registry.splitlines() if line.startswith("state") and name in line.split()[2:3])
        assert row[7] == "r", row


def test_real32_carry_lock_and_static_option(monkeypatch):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.validation.tier2 import make_ideal_grid
    monkeypatch.setattr(R, "_MP_RE", True)
    state, mp = _real_state()
    state = C._state_from_thompson_output(state.ensure_conditional_leaves(mp_physics=8), mp)
    grid = make_ideal_grid(*state.theta.shape)
    from gpuwrf.contracts import state as state_contract
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices()[0])
    nml = op.OperationalNamelist.from_grid(grid, use_mp_re=0, force_fp64=True)
    children, aux = nml.tree_flatten()
    assert op.OperationalNamelist.tree_unflatten(aux, children).use_mp_re == 0
    out = op._enforce_operational_precision(state, force_fp64=True)
    for name in ("re_cloud", "re_ice", "re_snow"):
        assert getattr(out, name).dtype == jnp.float32
        np.testing.assert_array_equal(getattr(out, name), getattr(state, name))


@pytest.mark.parametrize("use_mp_re", [0, 1])
def test_startup_land_radiation_uses_the_static_radius_selection(monkeypatch, use_mp_re):
    from gpuwrf.runtime import operational_mode as op
    monkeypatch.setattr(R, "_MP_RE", True)
    state, _ = _real_state()
    state = state.ensure_conditional_leaves(mp_physics=8, use_mp_re=use_mp_re)
    seen = []
    def diagnostics(s, grid, **kwargs):
        assert kwargs["use_mp_re"] == use_mp_re
        sw, lw, *_ = C._rrtmg_column_inputs(s, grid, time_utc=kwargs["time_utc"], use_mp_re=kwargs["use_mp_re"])
        seen.extend((sw, lw))
        z = jnp.zeros_like(s.t_skin)
        return SimpleNamespace(swnorm=z, glw=z, coszen=z)
    monkeypatch.setattr(op, "rrtmg_radiation_diagnostics", diagnostics)
    nml = SimpleNamespace(grid=None, time_utc="2026-02-28_00:00:00", radiation_static=None,
                          topo_shading=0, slope_rad=0, topo_shadow_length_m=25000.,
                          mp_physics=8, use_mp_re=use_mp_re, ra_sw_physics=4, ra_lw_physics=4)
    op.noahmp_initial_rad(state, nml)
    assert len(seen) == 2
    for column in seen:
        if use_mp_re:
            np.testing.assert_array_equal(column.re_cloud, jnp.full_like(column.qc, 2.49e-6, dtype=jnp.float32))
        else:
            assert column.re_cloud is None
