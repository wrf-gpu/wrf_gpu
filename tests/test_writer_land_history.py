"""Deletion-sensitive land output and literal WRF per-DT accumulator gates."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.io.land_history import LAND_HISTORY_FIELDS, LAND_LEAVES, land_history_diagnostics, load_land_history_inputs
from gpuwrf.io.land_history_metadata import LAND_HISTORY_METADATA
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, WRFOUT_VARIABLE_SPECS
from gpuwrf.runtime.history_accumulators import RADIATION_SOURCES, SURFACE_SOURCES, SNOW_ACCUMULATORS, accumulate_energy, seed_history
from test_m7_netcdf_writer import synthetic_case, writer_authority

CASE = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1")
CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
TABLE = Path("<USER_HOME>/src/wrf_pristine/WRF/run")


@pytest.mark.parametrize("domain", ["d01", "d02", "d03"])
def test_static_land_values_match_real_cpu_wrf_inventory(domain):
    source = CASE / f"wrfinput_{domain}"
    truth = CPU / f"wrfout_{domain}_2026-02-28_00:00:00"
    if not source.exists() or not truth.exists():
        pytest.skip("WN3 CPU-WRF inventory is not mounted")
    fields = load_land_history_inputs(source, table_dir=TABLE)
    names = ("ISLTYP", "IVGTYP", "SHDMAX", "SHDMIN", "SHDAVG", "SNOALB", "VAR", "CON",
             "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4", "ALBBCK", "ALBEDO", "EMISS")
    with Dataset(truth) as ds:
        for name in names:
            assert name in fields and name in ds.variables, name
            expected = np.asarray(ds[name][0])
            actual = fields[name].astype(expected.dtype)
            np.testing.assert_array_equal(actual, expected, err_msg=name)
            # Pins WRF's physical name and units against an independent inventory.
            assert WRFOUT_VARIABLE_SPECS[name].units == ds[name].units, name
        for name in ("LFMASS", "STMASS", "RTMASS", "WOOD", "STBLCP", "FASTCP"):
            np.testing.assert_array_equal(fields[name], np.asarray(ds[name][0]), err_msg=name)
        for name in ("LAI", "XSAI"):
            np.testing.assert_array_equal(fields[name + "_INIT"], np.asarray(ds[name][0]), err_msg=name)


def _land(shape=(4, 5), offset=0):
    ny, nx = shape
    values = {}
    for n, name in enumerate(NoahMPLandState.__slots__):
        depth = 4 if name in {"tslb", "smois", "sh2o"} else 3 if name in {"tsno", "snice", "snliq"} else 7 if name == "zsnso" else None
        dims = shape if depth is None else (depth, ny, nx)
        values[name] = np.arange(np.prod(dims), dtype=np.float32).reshape(dims) * .01 + n + offset
    values["isnow"] = np.zeros(shape, dtype=np.int32)
    return NoahMPLandState(**values)


def test_land_payload_reads_evolved_carry_and_preserves_initialized_water(tmp_path):
    state, grid, nml = synthetic_case()
    initial, evolved = _land(), _land(offset=10)
    inputs = {"LANDMASK": state.landmask, "ALBBCK": np.full((4, 5), .08, np.float32),
              "ALBEDO": np.full((4, 5), .08, np.float32), "EMISS": np.full((4, 5), .98, np.float32),
              "ISLTYP": np.full((4, 5), 6, np.int32)}
    diag, host_land = land_history_diagnostics(evolved, initial, inputs, own_step=67)
    prepared = prepare_wrfout_payload(state, grid, nml, tmp_path / "land.nc", domain="d02",
        domain_authority=writer_authority(grid), valid_time=datetime(2026, 5, 25, 19),
        run_start=datetime(2026, 5, 25, 18), lead_hours=1,
        diagnostics=diag, land_state=host_land, full_variable_set=True)
    mask = state.landmask > .5
    for name, attr in LAND_LEAVES.items():
        old, new = np.asarray(getattr(initial, attr)), np.asarray(getattr(evolved, attr))
        if name in {"ALBEDO", "EMISS"}:
            old = inputs[name]
        if name == "TSLB":
            old = old.copy(); old[0] = new[0]
        np.testing.assert_array_equal(prepared.fields[name], np.where(mask, new, old).astype(prepared.fields[name].dtype))
    np.testing.assert_array_equal(prepared.fields["ISLTYP"], inputs["ISLTYP"])
    np.testing.assert_array_equal(prepared.fields["SFROFF"], np.where(mask, evolved.sfcrunoff * 1000, 0))
    np.testing.assert_array_equal(prepared.fields["CANWAT"], prepared.fields["CANLIQ"] + prepared.fields["CANICE"])


def test_nested_callback_passes_land_state_to_payload(monkeypatch, tmp_path):
    import gpuwrf.integration.nested_pipeline as nested
    state, grid, nml = synthetic_case()
    land = _land()
    writer = nested._PerDomainWrfoutWriter.__new__(nested._PerDomainWrfoutWriter)
    writer.bundles = {"d02": SimpleNamespace(grid=grid, namelist=SimpleNamespace(use_noahmp=False))}
    writer.run_start = datetime(2026, 5, 25, 18)
    writer.writer_diagnostics = {}
    writer._initial_surface_fields = lambda name: None
    writer._surface_diagnostics_for_output = lambda *a, **kw: {}
    writer._merge_output_diagnostics = lambda *a: {}
    writer._variable_subset = None; writer._full_variable_set = True
    writer.domain_authorities = {"d02": writer_authority(grid)}
    writer._async_writer = None; writer.census_io_ledger = None
    observed = []
    monkeypatch.setattr(nested, "assert_state_finite_at_boundary", lambda *a, **kw: None)
    def capture(*a, **kw):
        observed.append(kw.get("land_state"))
        return SimpleNamespace(target=tmp_path / "land.nc")
    monkeypatch.setattr(nested, "prepare_wrfout_payload", capture)
    monkeypatch.setattr(nested, "write_prepared_wrfout", lambda *a, **kw: None)
    writer._materialize_and_submit(name="d02", own_step=1, carry=SimpleNamespace(state=state, noahmp_land=land),
        valid_time=datetime(2026, 5, 25, 19), lead_seconds=3600, lead_hours=1, path=tmp_path / "land.nc")
    assert observed == [land]


def test_energy_accumulates_each_dt_including_held_radiation_and_signed_fluxes():
    _, sums = seed_history(SimpleNamespace(t_skin=jnp.zeros((2, 3))))
    assert all(np.all(np.asarray(x) == 0) for x in sums.values())
    rad = SimpleNamespace(**{a: jnp.full((2, 3), i + 1.) for i, a in enumerate(RADIATION_SOURCES.values())})
    surface = {"hfx": -2., "lh": 3., "grdflx": -4., "land_history": {"SNOM_INCREMENT": .5}}
    step = jax.jit(lambda old: accumulate_energy(old, rad, surface, 18.))
    # Literal WRF driver: AC = AC + held_flux*DT on all three non-radiation steps.
    for _ in range(3):
        sums = step(sums)
    assert float(sums["ACHFX"][0, 0]) == -108.
    assert float(sums["ACLHF"][0, 0]) == 162.
    assert float(sums["ACGRDFLX"][0, 0]) == -216.
    assert float(sums["ACSNOM"][0, 0]) == 1.5  # Already-integrated mm; never multiply ponding by DT twice.
    for i, name in enumerate(RADIATION_SOURCES):
        np.testing.assert_array_equal(sums[name], np.full((2, 3), (i + 1.) * 54, np.float32))
    assert all(x.dtype == jnp.float32 for x in sums.values())


def test_history_container_restart_exact_and_old_schema_rejected(tmp_path, monkeypatch):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.io.restart import write_restart, read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})
    history, sums = seed_history(state)
    history = {n: v + i + .125 for i, (n, v) in enumerate(history.items())}
    sums = {n: v + (i + 1) * 1e5 for i, (n, v) in enumerate(sums.items())}
    carry = initial_operational_carry(state).replace(land_history=history, energy_accumulators=sums)
    nml = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18, acoustic_substeps=4)
    pkl = tmp_path / "restart.pkl"; nc = tmp_path / "restart.nc"
    write_restart(carry, nml, grid, 7, pkl)
    write_wrfrst_carry(carry, grid, {}, nc, valid_time="2026-07-25_18:02:06", run_start="2026-07-25_18:00:00", step_index=7)
    for restored in (read_restart(pkl)[0], read_wrfrst_carry(nc)[0]):
        for field in ("land_history", "energy_accumulators"):
            assert set(getattr(restored, field)) == set(getattr(carry, field))
            for n, value in getattr(carry, field).items():
                assert np.asarray(getattr(restored, field)[n]).tobytes() == np.asarray(value).tobytes()
    with Dataset(nc, "r+") as ds:
        ds.GPUWRF_WRFRST_SCHEMA_VERSION = "v0.25-wrfrst-netcdf-3"
    with pytest.raises(ValueError, match="schema"):
        read_wrfrst_carry(nc)


def test_land_history_is_retained_from_the_actual_noah_step(monkeypatch):
    from test_noahmp_coupler import _build
    import gpuwrf.physics.noahmp.noahmp_driver as driver
    from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
    from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS
    state, land, static, rad, clock = _build()
    captured = []
    original = driver.noahmp_energy_canopy
    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append((args[3], kwargs["rad_extras"], result[1], result[2]))
        return result
    monkeypatch.setattr(driver, "noahmp_energy_canopy", spy)
    _, new_land, _, fields = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90, land_history=True)
    h = fields["land_history"]
    assert set(h) == set(LAND_FLUX_FIELDS)
    r, extras, energy, et = captured[0]
    is_land = np.asarray(state.xland < 1.5)
    for name, value in (("SAV", r.sav), ("SAG", r.sag), ("GRDFLX", energy.ssoil),
                        ("ECAN", et.ecan), ("EDIR", et.edir), ("ETRAN", et.etran),
                        ("CANHS", energy.canhs), ("TRAD", energy.trad),
                        ("APAR", r.parsun * extras["laisun"] + r.parsha * extras["laisha"])):
        np.testing.assert_array_equal(h[name], np.where(is_land, value, 0), err_msg=name)
    assert np.max(np.asarray(h["SOILENERGY"])) > 0
    assert np.max(np.asarray(h["SAV"])) > 0
    for value in h.values():
        assert np.all(np.isfinite(value))
    baseline = noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=90)[1]
    for a, b in zip(jax.tree.leaves(new_land), jax.tree.leaves(baseline), strict=True):
        np.testing.assert_array_equal(a, b)


def test_bare_land_uses_wrf_skipped_canopy_outputs():
    from test_noahmp_coupler import _build
    from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
    state, land, static, rad, clock = _build()
    static = static.replace(ivgtyp=static.ivgtyp.at[0, 0].set(16))
    _, _, _, fields = noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock,
                                            dt=90, land_history=True)
    h = fields["land_history"]
    for name in ("FVEG", "TR", "EVC", "SHC", "IRC", "SHG", "IRG", "EVG", "GHV", "CHLEAF", "CHUC", "CHV2", "RSSUN", "RSSHA", "T2V", "Q2V"):
        assert float(h[name][0, 0]) == 0, name
    for name, value in h.items():
        assert np.all(np.isfinite(value)), name


def test_all_land_and_accumulation_metadata_matches_cpu_inventory():
    truth = CPU / "wrfout_d01_2026-02-28_00:00:00"
    if not truth.exists():
        pytest.skip("WN3 CPU-WRF inventory is not mounted")
    with Dataset(truth) as ds:
        for name, (description, units) in LAND_HISTORY_METADATA.items():
            assert description == ds[name].description, name
            assert units == ds[name].units, name
            spec = WRFOUT_VARIABLE_SPECS[name]
            assert spec.description == description and spec.units == units, name
            assert spec.dimensions == ds[name].dimensions, name
        assert {n for n in ds.variables if n.startswith("AC")} <= set(LAND_HISTORY_METADATA)


def test_full_history_is_seeded_on_the_actual_operational_init_path(monkeypatch):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run, OperationalNamelist
    from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS
    grid = GridSpec.canary_3km_template()
    state = State(**{name: jnp.ones(shape) for name, shape in _state_field_shapes(grid).items()})
    nml = OperationalNamelist(grid=grid, metrics=grid.metrics, tendencies=None, dt_s=18,
        acoustic_substeps=4, use_noahmp=True, run_physics=False,
        ra_sw_physics=0, ra_lw_physics=0, cu_physics=0, mp_physics=0, bl_pbl_physics=0)
    monkeypatch.setenv("GPUWRF_FULL_WRFOUT_VARIABLES", "1")
    carry = _initial_carry_for_run(state, nml)
    assert set(carry.land_history) == set(LAND_FLUX_FIELDS)
    assert set(carry.energy_accumulators) == {*RADIATION_SOURCES, *SURFACE_SOURCES, *SNOW_ACCUMULATORS}
    for value in (*carry.land_history.values(), *carry.energy_accumulators.values()):
        assert value.dtype == jnp.float32 and np.all(np.asarray(value) == 0)
    monkeypatch.setenv("GPUWRF_FULL_WRFOUT_VARIABLES", "0")
    legacy = _initial_carry_for_run(state, nml)
    assert legacy.land_history is legacy.energy_accumulators is None
