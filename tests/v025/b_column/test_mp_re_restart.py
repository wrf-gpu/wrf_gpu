"""BP90 + MP_RE restart composition, with actual post-MP radius operands."""
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts import state as state_contract
from gpuwrf.contracts.precision import STATE_FIELD_ORDER
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.io.restart import read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.physics import rrtmg_mp_re as R
from gpuwrf.validation.tier2 import make_ideal_grid
from test_mp_re import _real_state


def _combined(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "1")
    monkeypatch.setattr(R, "_MP_RE", True)
    state, column = _real_state()
    state = C._state_from_thompson_output(state.ensure_conditional_leaves(mp_physics=8), column)
    fixture = json.loads((Path(__file__).parents[1] / "b_phys/fixtures/mynn_sfc_wspd_day_wrf.json").read_text())
    wspd = jnp.asarray(np.asarray(fixture["baseline"]["wrfwind"]).reshape(-1)[:state.xland.size], jnp.float32).reshape(state.xland.shape)
    state = state.replace(sfc_wspd=wspd)
    grid = make_ideal_grid(*state.theta.shape)
    return grid, op.OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics), initial_operational_carry(state)


def _exact(a, b):
    aa, ta = jax.tree.flatten(a)
    bb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(aa, bb, strict=True):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype
        assert x.tobytes() == y.tobytes()


def test_bp90_tail_and_both_flags_roundtrip_in_all_restart_formats(monkeypatch, tmp_path):
    grid, nml, carry = _combined(monkeypatch)
    assert state_contract.State.__slots__[-4:] == ("sfc_wspd", "re_cloud", "re_ice", "re_snow")
    assert STATE_FIELD_ORDER[-4:] == state_contract.State.__slots__[-4:]
    widened = op._enforce_operational_precision(carry.state, force_fp64=True)
    for name in state_contract.State.__slots__[-4:]:
        assert getattr(widened, name).dtype == jnp.float32
        np.testing.assert_array_equal(getattr(widened, name), getattr(carry.state, name))
    path = tmp_path / "both.pkl"
    write_restart(carry, nml, grid, 2, path)
    _exact(carry, read_restart(path)[0])
    path = tmp_path / "both.nc"
    write_wrfrst_carry(carry, grid, {}, path, valid_time="2026-02-28_00:01:48", run_start="2026-02-28_00:00:00", step_index=2)
    _exact(carry, read_wrfrst_carry(path)[0])
    store = RestartStore(tmp_path / "store", 10**8, 2, 0)
    generation, _ = store.save({"d01": carry}, {"d01": 2}, {"d01": 54.}, {"both_flags": True})
    restored = store.read(generation)[0]["carries"]["d01"]
    _exact(carry, restored)
    # The saved radii must be the non-background previous-MP output, not zeros.
    assert np.any(np.asarray(restored.state.re_cloud) != np.float32(2.49e-6))


@pytest.mark.parametrize("missing,message", [("sfc_wspd", "sfc_wspd.*E78"), ("re_cloud", "missing held Thompson.*E78")])
def test_each_e78_guard_survives_the_composition(monkeypatch, tmp_path, missing, message):
    grid, nml, carry = _combined(monkeypatch)
    path = tmp_path / "missing.pkl"
    write_restart(carry, nml, grid, 2, path)
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    payload["carry"]["state_field_order"].remove(missing)
    del payload["carry"]["state_fields"][missing]
    with path.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match=message):
        read_restart(path)
    path = tmp_path / "missing.nc"
    write_wrfrst_carry(carry, grid, {}, path, valid_time="2026-02-28_00:01:48", run_start="2026-02-28_00:00:00", step_index=2)
    from netCDF4 import Dataset
    with Dataset(path, "a") as ds:
        order = json.loads(ds.GPUWRF_STATE_FIELD_ORDER)
        order.remove(missing)
        ds.GPUWRF_STATE_FIELD_ORDER = json.dumps(order)
    with pytest.raises(ValueError, match=message):
        read_wrfrst_carry(path)
