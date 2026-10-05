"""Host history parity against original WRF, with unchanged carried bytes."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.integration.daily_pipeline import _merge_output_diagnostics
from gpuwrf.io.land_history import LAND_LEAVES, land_history_diagnostics, load_land_history_inputs
from gpuwrf.io.lower_boundary import load_lower_boundary
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority, prepare_wrfout_payload, write_prepared_wrfout
from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS

TABLE = Path('<USER_HOME>/src/wrf_pristine/WRF/run')
WN3 = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1')
WN3_CPU = Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
WN3_GPU = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/W9/finalb_r3/20260227_18z_a1/wrfout')
SWISS = Path(__file__).resolve().parents[1]/'examples/switzerland_d01'
SWISS_CPU = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
SWISS_GPU = Path('<USER_HOME>/wrf_gpu2_lanes/writer-land/WL22/swiss/wrfout')


def _read(path, names):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        return {name: np.asarray(ds[name][0]) for name in names}


def _bytes(tree):
    return [(x.dtype.str, x.shape, x.tobytes()) for x in jax.tree.leaves(jax.device_get(tree))]


@pytest.mark.parametrize('domain', ['d01', 'd02', 'd03'])
def test_real_wn3_water_history_t0_and_first_solve_phase(domain, tmp_path, monkeypatch):
    cpu = sorted(WN3_CPU.glob(f'wrfout_{domain}_*'))
    gpu = sorted(WN3_GPU.glob(f'wrfout_{domain}_*'))
    if len(cpu) < 25 or len(gpu) < 25:
        pytest.skip('original WN3 CPU-WRF and measured FINAL-b frames not mounted')
    run = Gen2Run(WN3)
    grid = run.grid(domain)
    authority = bind_wrfout_domain_authority(domain, grid, grid)
    initial, static, _ = build_noahmp_land_state(WN3, domain, table_dir=TABLE)
    inputs = load_land_history_inputs(WN3/f'wrfinput_{domain}', table_dir=TABLE,
                                      parameters=static.parameters)
    water = np.asarray(inputs['LANDMASK']) < .5
    assert int(water.sum()) > 1000
    start = datetime.strptime(cpu[0].name[-19:], '%Y-%m-%d_%H:%M:%S')
    with Dataset(cpu[0]) as ds:
        dt = float(ds.DT)
    boundary = load_lower_boundary(WN3, run.namelist, domain, run_start=start,
                                   dt_s=dt, shape=(grid.ny, grid.nx))
    assert boundary is not None
    writer = object.__new__(pipeline._PerDomainWrfoutWriter)
    writer.run_start = start
    writer.bundles = {domain: SimpleNamespace(grid=grid,
        namelist=SimpleNamespace(use_noahmp=True, lower_boundary=boundary))}
    writer._land_history_inputs = {domain: inputs}
    writer._land_history_initial = {domain: jax.device_get(initial)}
    writer.writer_diagnostics = {}
    writer._initial_surface_fields = lambda name: {}
    writer._merge_output_diagnostics = _merge_output_diagnostics
    writer.domain_authorities = {domain: authority}
    writer._variable_subset = ('TSLB',)
    writer._full_variable_set = True
    writer._async_writer = None
    writer.census_io_ledger = None
    monkeypatch.setattr(pipeline, 'assert_state_finite_at_boundary', lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, '_noahmp_surface_diagnostics_for_output', lambda *a, **kw: {})
    # Literal first-call phase from pristine driver697-705: the reset follows
    # SST_UPDATE, so that one step still has273.16 even with prescribed SST.
    first, _ = land_history_diagnostics(initial, initial, inputs, own_step=1,
                                        water_sst=_read(gpu[0], ['TSK'])['TSK'])
    np.testing.assert_array_equal(first['TSLB'][:, water],
        np.full((4, int(water.sum())), 273.16, dtype=initial.tslb.dtype))
    # Real model operands from the released run; original CPU-WRF remains the
    # oracle. Land columns and the top ocean layer pass through without edits.
    for hour in (0, 1, 6, 24):
        operands = _read(gpu[hour], ['TSLB', 'TSK'])
        real, sst = operands['TSLB'], operands['TSK']
        carry = initial.replace(tslb=jnp.asarray(real, dtype=initial.tslb.dtype))
        before = _bytes((carry, initial))
        valid = datetime.strptime(gpu[hour].name[-19:], '%Y-%m-%d_%H:%M:%S')
        target = tmp_path/gpu[hour].name
        seconds = (valid-start).total_seconds()
        writer._materialize_and_submit(name=domain, own_step=round(seconds/dt),
            carry=SimpleNamespace(state=SimpleNamespace(t_skin=jnp.asarray(sst)), noahmp_land=carry),
            valid_time=valid, lead_seconds=seconds, lead_hours=hour, path=target)
        actual = _read(target, ['TSLB'])['TSLB']
        expected = _read(cpu[hour], ['TSLB'])['TSLB']
        np.testing.assert_array_equal(actual[:, water], expected[:, water])
        np.testing.assert_array_equal(actual[:, ~water], real[:, ~water])
        np.testing.assert_array_equal(actual[0, water], sst[water])
        assert before == _bytes((carry, initial))


def test_real_swiss_initialized_statics_and_glacier_missing_value_phase(tmp_path):
    cpu = sorted(SWISS_CPU.glob('wrfout_d01_*'))
    gpu = sorted(SWISS_GPU.glob('wrfout_d01_*'))
    if len(cpu) < 3 or len(gpu) < 3:
        pytest.skip('original Swiss CPU-WRF and captured public CLI producer frames not mounted')
    names = ('WA', 'WT', 'ZWT', 'GRAIN', 'GDD', 'BGAP', 'WGAP', 'WSLAKE',
             'NEE', 'GPP', 'NPP', 'APAR', 'PSN')
    initial, static, _ = build_noahmp_land_state(SWISS, 'd01', table_dir=TABLE)
    inputs = load_land_history_inputs(SWISS/'wrfinput_d01', table_dir=TABLE,
                                      parameters=static.parameters)
    glacier = np.asarray(static.ivgtyp) == int(static.parameters.isice)
    assert int(glacier.sum()) == 22
    grid = Gen2Run(SWISS).grid('d01')
    authority = bind_wrfout_domain_authority('d01', grid, grid)
    start = datetime(2023, 1, 15)
    for hour in (0, 1, 2):
        # Actual saved GPU operands have zero where WRF emits glacier missing
        # values; do not seed the oracle's missing-value pattern into the input.
        with Dataset(gpu[hour]) as ds:
            ds.set_auto_mask(False)
            updates = {attr: jnp.asarray(ds[name][0], dtype=getattr(initial, attr).dtype)
                       for name, attr in LAND_LEAVES.items()}
            history = {name: np.asarray(ds[name][0]) for name in LAND_FLUX_FIELDS
                       if name in ds.variables}
        carry = initial.replace(**updates)
        before = _bytes((carry, initial, history))
        fields, host = land_history_diagnostics(carry, initial, inputs,
                                               own_step=hour*200, history=history)
        target = tmp_path/gpu[hour].name
        prepared = prepare_wrfout_payload({}, grid, None, target, domain='d01',
            domain_authority=authority, valid_time=datetime(2023, 1, 15, hour),
            run_start=start, lead_hours=hour, diagnostics=fields, land_state=host,
            variable_subset=names, full_variable_set=True)
        write_prepared_wrfout(prepared, expected_domain='d01', expected_domain_authority=authority)
        with Dataset(target) as actual, Dataset(cpu[hour]) as oracle:
            actual.set_auto_mask(False); oracle.set_auto_mask(False)
            for name in names:
                np.testing.assert_array_equal(actual[name][0], oracle[name][0], err_msg=f'{hour}:{name}')
                for attr in ('units', 'description', 'dimensions'):
                    assert getattr(actual[name], attr) == getattr(oracle[name], attr), (name, attr)
        if hour:
            for name in ('APAR', 'PSN'):
                np.testing.assert_array_equal(fields[name][~glacier], history[name][~glacier])
        assert before == _bytes((carry, initial, history))
        assert _bytes(host) == _bytes(carry)


@pytest.mark.parametrize('domain', ['d01', 'd02'])
def test_real_prod_water_without_sst_update_matches_first_call_reset(domain):
    source = Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
    cpu_dir = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')
    cpu = sorted(cpu_dir.glob(f'wrfout_{domain}_*'))
    if len(cpu) < 25 or not source.exists():
        pytest.skip('original PROD CPU-WRF input/history not mounted')
    initial, static, _ = build_noahmp_land_state(source, domain, table_dir=TABLE)
    inputs = load_land_history_inputs(source/f'wrfinput_{domain}', table_dir=TABLE,
                                      parameters=static.parameters)
    water = inputs['LANDMASK'] < .5
    before = _bytes(initial)
    for hour in (0, 1, 6, 24):
        fields, host = land_history_diagnostics(initial, initial, inputs,
                                                own_step=0 if hour == 0 else hour)
        expected = _read(cpu[hour], ['TSLB'])['TSLB']
        np.testing.assert_array_equal(fields['TSLB'][:, water].astype(expected.dtype), expected[:, water])
        assert _bytes(host) == before == _bytes(initial)
