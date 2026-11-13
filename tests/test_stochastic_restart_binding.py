"""Restart admission binds stochastic presence and controls per domain (critic FIX v50-stoch finding 5)."""
from dataclasses import replace
from datetime import datetime

import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.stochastic_config import bind_stochastic, restart_controls
from gpuwrf.physics.stochastic_driver import initial_stochastic
from test_stochastic_restart import _grid

CONTROLS = dict(skebs=1, sppt=1, spp=1, nens=1, **{"lengthscale_" + k: 6000. for k in ("sppt", "spp_conv", "spp_pbl", "spp_lsm")})


def _setup(**change):
    from test_noahmp_checkpoint_v2 import _state, _namelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid = _grid()
    config = bind_stochastic({"stoch": {**CONTROLS, **change}}, "d01")
    namelist = replace(_namelist(grid), stochastic_config=config)
    carry = initial_operational_carry(_state(grid)).replace(
        stochastic=initial_stochastic(config, grid, 18, datetime(2023, 1, 15)))
    return grid, namelist, carry


def test_netcdf_restart_refuses_changed_member_controls(tmp_path):
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    grid, namelist, carry = _setup()
    path = tmp_path / "wrfrst"
    write_wrfrst_carry(carry, grid, namelist, path, valid_time="2023-01-15_00:00:36",
                       run_start="2023-01-15_00:00:00", step_index=2)
    read_wrfrst_carry(path, namelist=namelist)
    for change in ({"nens": 2}, {"iseed_sppt": 99}, {"gridpt_stddev_sppt": .3}):
        _, other, _ = _setup(**change)
        with pytest.raises(ValueError, match="controls.*differ.*E78"):
            read_wrfrst_carry(path, namelist=other)
    with Dataset(path, "a") as ds:
        ds.delncattr("GPUWRF_STOCHASTIC_CONTROLS")
    with pytest.raises(ValueError, match="controls digest.*E78"):
        read_wrfrst_carry(path, namelist=namelist)


def test_native_store_refuses_active_domain_without_stochastic_carry(tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore
    _, namelist, carry = _setup()
    controls = restart_controls(namelist.stochastic_config)
    identity = {"stochastic_config": {"d01": controls, "d02": controls}}
    store = RestartStore(tmp_path / "store", 10**8, 2, 0)
    with pytest.raises(ValueError, match="d02: restart stochastic carry presence.*E78"):
        store.save({"d01": carry, "d02": carry.replace(stochastic=None)}, {"d01": 2, "d02": 6},
                   {"d01": 18., "d02": 6.}, identity)
    generation, _ = store.save({"d01": carry, "d02": carry}, {"d01": 2, "d02": 6}, {"d01": 18., "d02": 6.}, identity)
    store.read(generation, device=False)
    # A store with a stochastic group but no authenticated controls is refused at save.
    with pytest.raises(ValueError, match="lacks authenticated controls.*E78"):
        RestartStore(tmp_path / "other", 10**8, 2, 0).save({"d01": carry}, {"d01": 2}, {"d01": 18.}, {})


def test_delta1_restart_bypasses_are_refused(tmp_path):
    """Critic Delta 1 finding 1: missing seed (native), requested OFF + stripped marker/digest (wrfrst), pickle controls."""
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state, write_checkpoint
    from gpuwrf.runtime.restart_store import RestartStore
    grid, namelist, carry = _setup()
    controls = restart_controls(namelist.stochastic_config)
    identity = {"stochastic_config": {"d01": controls, "d02": controls}}
    bad = carry.stochastic._replace(seeds=(None, *carry.stochastic.seeds[1:]))
    with pytest.raises(ValueError, match="d02: .*RNG stream.*E78"):
        RestartStore(tmp_path / "store", 10**8, 2, 0).save({"d01": carry, "d02": carry.replace(stochastic=bad)},
                                                           {"d01": 2, "d02": 6}, {"d01": 18., "d02": 6.}, identity)
    path = tmp_path / "wrfrst"
    write_wrfrst_carry(carry, grid, namelist, path, valid_time="2023-01-15_00:00:36",
                       run_start="2023-01-15_00:00:00", step_index=2)
    with pytest.raises(ValueError, match="stochastic physics off.*E78"):
        read_wrfrst_carry(path, namelist=replace(namelist, stochastic_config=None))
    with Dataset(path, "a") as ds:
        ds.delncattr("GPUWRF_STOCHASTIC_CONTROLS"); ds.delncattr("GPUWRF_STOCHASTIC_SCHEMA_VERSION")
    with pytest.raises(ValueError, match="without its schema marker.*E78"):
        read_wrfrst_carry(path, namelist=namelist)
    pickle_path = tmp_path / "checkpoint"
    write_checkpoint(carry.state, namelist, grid, 0, pickle_path, runtime_state=carry)
    read_checkpoint_with_runtime_state(pickle_path, namelist=namelist)
    _, other, _ = _setup(nens=2, iseed_sppt=99)
    with pytest.raises(ValueError, match="checkpoint stochastic controls.*E78"):
        read_checkpoint_with_runtime_state(pickle_path, namelist=other)


def test_delta2_native_store_checks_root_child_layout_from_recorded_controls(tmp_path):
    """Critic Delta 2 finding 2: a parent FFT/RNG carry stored under recorded d02 child controls must be refused."""
    from gpuwrf.runtime.restart_store import RestartStore
    grid, namelist, carry = _setup()
    child_config = bind_stochastic({"stoch": CONTROLS}, "d02")
    assert child_config.root is False and namelist.stochastic_config.root is True
    child_carry = carry.replace(stochastic=initial_stochastic(child_config, grid, 6, datetime(2023, 1, 15)))
    identity = {"stochastic_config": {"d01": restart_controls(namelist.stochastic_config),
                                      "d02": restart_controls(child_config)}}
    with pytest.raises(ValueError, match="d02: .*root/child coefficient layout differs.*E78"):
        RestartStore(tmp_path / "bad", 10**8, 2, 0).save({"d01": carry, "d02": carry}, {"d01": 2, "d02": 6},
                                                         {"d01": 18., "d02": 6.}, identity)
    with pytest.raises(ValueError, match="d01: .*root/child coefficient layout differs.*E78"):
        RestartStore(tmp_path / "swap", 10**8, 2, 0).save({"d01": child_carry, "d02": child_carry},
                                                          {"d01": 2, "d02": 6}, {"d01": 18., "d02": 6.}, identity)
    store = RestartStore(tmp_path / "good", 10**8, 2, 0)
    generation, _ = store.save({"d01": carry, "d02": child_carry}, {"d01": 2, "d02": 6}, {"d01": 18., "d02": 6.},
                               identity)
    store.read(generation, expected_identity=identity, device=False)
