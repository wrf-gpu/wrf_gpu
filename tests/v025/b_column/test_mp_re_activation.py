"""MP_RE resolved activation and restart regressions for the RC default."""
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.contracts.precision import MP_RE_DIAGNOSTIC_LEAVES
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.io.restart import read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.physics import rrtmg_mp_re as R
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.validation.tier2 import make_ideal_grid
from test_mp_re import _real_state


def _case(monkeypatch, **selection):
    monkeypatch.setattr(R, "_MP_RE", True)
    state, column = _real_state()
    grid = make_ideal_grid(*state.theta.shape)
    nml = op.OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics,
        bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0, run_boundary=False,
        time_utc="2026-02-28_00:00:00", **selection)
    state = op._operational_scan_state(state, nml)
    if nml.mp_physics == 8:
        state = C._state_from_thompson_output(state, column)
    return grid, nml, initial_operational_carry(state)


def _exact_finite(a, b):
    aa, ta = jax.tree.flatten(a)
    bb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(aa, bb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("selection", [{"use_mp_re": 0}, {"mp_physics": 6},
    {"ra_sw_physics": 1}, {"ra_lw_physics": 1}, {"ra_sw_physics": 0}, {"mp_physics": 28}])
def test_flag_on_inactive_selection_roundtrips_all_restart_formats(monkeypatch, tmp_path, selection):
    grid, nml, carry = _case(monkeypatch, **selection)
    assert not R.mp_re_active(nml)
    assert all(getattr(carry.state, name) is None for name in MP_RE_DIAGNOSTIC_LEAVES)
    path = tmp_path / "inactive.pkl"
    write_restart(carry, nml, grid, 2, path)
    _exact_finite(carry, read_restart(path)[0])
    path = tmp_path / "inactive.nc"
    write_wrfrst_carry(carry, grid, nml, path, valid_time="2026-02-28_00:01:48", run_start="2026-02-28_00:00:00", step_index=2)
    with Dataset(path) as ds:
        assert json.loads(ds.GPUWRF_MP_RE_CONFIG) == R.mp_re_config(nml)
    _exact_finite(carry, read_wrfrst_carry(path)[0])
    identity = {"mp_re_config": {"d01": R.mp_re_config(nml)}}
    store = RestartStore(tmp_path / "store", 10**8, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, identity)
    _exact_finite(carry, store.read(generation)[0]["carries"]["d01"])


def test_dudhia_sw_runtime_lw_handoff_uses_no_mp_radii(monkeypatch):
    grid, nml, carry = _case(monkeypatch, ra_sw_physics=1, ra_lw_physics=4)
    seen = []
    def lw(s, *args, **kwargs):
        assert kwargs["use_mp_re"] == 0
        sw_column, lw_column, *_ = C._rrtmg_column_inputs(s, None, time_utc=nml.time_utc,
                                                       use_mp_re=kwargs["use_mp_re"])
        seen.append((sw_column.re_cloud, lw_column.re_cloud))
        return jnp.zeros_like(s.theta)
    monkeypatch.setattr(op, "rrtmg_lw_theta_tendency", lw)
    monkeypatch.setattr(op, "dudhia_sw_theta_tendency", lambda s, *a, **k: jnp.zeros_like(s.theta))
    op._physics_step_forcing(carry, nml, jnp.float64(0), run_radiation=True)
    assert seen == [(None, None)]


@pytest.mark.parametrize("missing", MP_RE_DIAGNOSTIC_LEAVES)
def test_active_selection_rejects_each_missing_radius_in_both_readers(monkeypatch, tmp_path, missing):
    grid, nml, carry = _case(monkeypatch)
    assert R.mp_re_active(nml)
    assert all(getattr(carry.state, n).dtype == jnp.float32 for n in MP_RE_DIAGNOSTIC_LEAVES)
    path = tmp_path / "active.pkl"
    write_restart(carry, nml, grid, 2, path)
    _exact_finite(carry, read_restart(path)[0])
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["carry"]["state_field_order"].remove(missing)
    del payload["carry"]["state_fields"][missing]
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="missing held Thompson.*E78"):
        read_restart(path)
    path = tmp_path / "active.nc"
    write_wrfrst_carry(carry, grid, nml, path, valid_time="2026-02-28_00:01:48", run_start="2026-02-28_00:00:00", step_index=2)
    with Dataset(path, "a") as ds:
        order = json.loads(ds.GPUWRF_STATE_FIELD_ORDER)
        order.remove(missing)
        ds.GPUWRF_STATE_FIELD_ORDER = json.dumps(order)
    with pytest.raises(ValueError, match="missing held Thompson.*E78"):
        read_wrfrst_carry(path)


def test_inactive_selection_drops_previously_active_radius_carry(monkeypatch):
    grid, nml, carry = _case(monkeypatch)
    state = op._operational_scan_state(carry.state, replace(nml, ra_sw_physics=1))
    assert all(getattr(state, n) is None for n in MP_RE_DIAGNOSTIC_LEAVES)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_roundtrip_gate_refuses_nonfinite_values(monkeypatch, bad):
    _, _, carry = _case(monkeypatch)
    changed = carry.replace(state=carry.state.replace(theta=carry.state.theta.at[0, 0, 0].set(bad)))
    with pytest.raises(AssertionError):
        _exact_finite(changed, changed)
