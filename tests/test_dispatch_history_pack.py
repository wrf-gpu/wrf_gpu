"""History accumulators ride the hot carry as ONE leaf per family (dispatch, b-diff BD73 host gap).

The 54 land-history fields and 20 energy sums were 74 carry leaves per domain;
every executable parameter costs host work per execute. ``PackedFields`` keeps
name access (Mapping) for the writer and restart while the carry holds one
stacked array per family. Values are the stacked slices: bitwise unchanged.
"""
from types import SimpleNamespace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import history_accumulators as ha
from gpuwrf.runtime.history_accumulators import (
    ENERGY_ACCUMULATORS, LAND_FLUX_FIELDS, PackedFields, accumulate_energy, as_packed, seed_history)


def _grid_state():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    grid = GridSpec.canary_3km_template()
    return grid, State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})


def test_seed_is_one_leaf_per_family_in_canonical_order():
    land, energy = seed_history(SimpleNamespace(t_skin=jnp.zeros((3, 4))))
    assert isinstance(land, PackedFields) and isinstance(energy, PackedFields)
    assert land.names == LAND_FLUX_FIELDS and energy.names == ENERGY_ACCUMULATORS
    assert len(jax.tree.leaves(land)) == len(jax.tree.leaves(energy)) == 1
    assert land.data.shape == (len(LAND_FLUX_FIELDS), 3, 4) and land.data.dtype == jnp.float32
    assert not np.any(np.asarray(land.data)) and not np.any(np.asarray(energy.data))


def test_operational_init_carry_holds_two_history_leaves(monkeypatch):
    from gpuwrf.runtime.operational_mode import OperationalNamelist, _initial_carry_for_run
    grid, state = _grid_state()
    nml = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, dt_s=18,
        acoustic_substeps=4, use_noahmp=True, run_physics=False,
        ra_sw_physics=0, ra_lw_physics=0, cu_physics=0, mp_physics=0, bl_pbl_physics=0)
    monkeypatch.setenv("GPUWRF_FULL_WRFOUT_VARIABLES", "1")
    carry = _initial_carry_for_run(state, nml)
    assert len(jax.tree.leaves((carry.land_history, carry.energy_accumulators))) == 2  # was 74


def test_mapping_access_and_pytree_maps_keep_names():
    packed = PackedFields.pack({"A": jnp.full((2,), 1.0), "B": jnp.full((2,), 2.0)})
    assert list(packed) == ["A", "B"] and len(packed) == 2 and "A" in packed and "Z" not in packed
    with pytest.raises(KeyError):
        packed["Z"]
    assert dict(packed).keys() == {"A", "B"}
    merged = {"C": 0}
    merged.update(packed)
    np.testing.assert_array_equal(merged["B"], [2.0, 2.0])
    host = jax.device_get(packed)
    assert isinstance(host, PackedFields) and host.names == ("A", "B")
    doubled = jax.tree.map(lambda x: x * 2, packed)
    assert doubled.names == packed.names and float(doubled["A"][0]) == 2.0


def test_energy_accumulation_is_bitwise_packed_vs_per_name():
    rng = np.random.default_rng(7)
    old = {name: jnp.asarray(rng.normal(size=(3, 4)) * 1e6, jnp.float32) for name in ENERGY_ACCUMULATORS}
    rad = SimpleNamespace(**{attr: jnp.asarray(rng.normal(size=(3, 4)) * 300, jnp.float32)
                             for attr in ha.RADIATION_SOURCES.values()})
    surface = {attr: jnp.asarray(rng.normal(size=(3, 4)) * 200, jnp.float32) for attr in ha.SURFACE_SOURCES.values()}
    surface["land_history"] = {"SNOM_INCREMENT": jnp.asarray(rng.random((3, 4)), jnp.float32)}
    per_name = jax.jit(lambda o: accumulate_energy(o, rad, surface, 18.))(old)
    packed = jax.jit(lambda o: PackedFields.pack(accumulate_energy(o, rad, surface, 18.)))(PackedFields.pack(old))
    assert packed.names == ENERGY_ACCUMULATORS
    for name in ENERGY_ACCUMULATORS:
        assert np.asarray(packed[name]).tobytes() == np.asarray(per_name[name]).tobytes(), name


def test_restart_roundtrip_is_exact_and_legacy_dicts_upgrade_in_canonical_order(tmp_path):
    from gpuwrf.io.restart import read_restart, write_restart
    from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    grid, state = _grid_state()
    land, energy = seed_history(state)
    land = PackedFields(land.names, land.data + jnp.arange(len(land))[:, None, None] + .125)
    energy = PackedFields(energy.names, energy.data + (jnp.arange(len(energy))[:, None, None] + 1) * 1e5)
    nml = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18, acoustic_substeps=4)
    legacy = {"land_history": {n: land[n] for n in sorted(land)}, "energy_accumulators": dict(energy)}
    for label, fields in (("packed", {"land_history": land, "energy_accumulators": energy}), ("legacy", legacy)):
        carry = initial_operational_carry(state).replace(**fields)
        pkl, nc = tmp_path / f"{label}.pkl", tmp_path / f"{label}.nc"
        write_restart(carry, nml, grid, 7, pkl)
        write_wrfrst_carry(carry, grid, {}, nc, valid_time="2026-07-25_18:02:06",
                           run_start="2026-07-25_18:00:00", step_index=7)
        for restored in (read_restart(pkl)[0], read_wrfrst_carry(nc)[0]):
            for name, expected in (("land_history", land), ("energy_accumulators", energy)):
                got = getattr(restored, name)
                assert isinstance(got, PackedFields) and got.names == expected.names, (label, name)
                assert np.asarray(got.data).tobytes() == np.asarray(expected.data).tobytes(), (label, name)


def test_as_packed_passes_through_packed_and_none():
    land, _ = seed_history(SimpleNamespace(t_skin=jnp.zeros((2, 2))))
    assert as_packed(land, LAND_FLUX_FIELDS) is land and as_packed(None, LAND_FLUX_FIELDS) is None


def test_noah_step_writes_packed_families():
    from gpuwrf.runtime import operational_mode as om
    source = inspect.getsource(om)
    assert "land_history=PackedFields.pack(" in source and "energy_accumulators=PackedFields.pack(accumulate_energy(" in source
