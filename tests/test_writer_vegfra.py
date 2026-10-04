"""VEGFRA is prescribed percent, not Noah-MP's distinct FVEG output."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.integration.daily_pipeline import _merge_output_diagnostics
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_history import load_land_history_inputs
from gpuwrf.io.lower_boundary import load_lower_boundary
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority

TABLE = Path('<USER_HOME>/src/wrf_pristine/WRF/run')
PROD = Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
D5 = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')
WN3 = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1')
WN3_CPU = Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
SWISS = Path(__file__).resolve().parents[1] / 'examples/switzerland_d01'
SWISS_CPU = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
CASES = [(PROD, D5, domain) for domain in ('d01', 'd02')]
CASES += [(WN3, WN3_CPU, domain) for domain in ('d01', 'd02', 'd03')]
CASES += [(SWISS, SWISS_CPU, 'd01')]


@pytest.fixture(scope='module', params=CASES,
                ids=['prod-d01', 'prod-d02', 'wn3-d01', 'wn3-d02', 'wn3-d03', 'swiss-d01'])
def real_case(request):
    source, cpu, domain = request.param
    frames = sorted(cpu.glob(f'wrfout_{domain}_*'))
    if not (source / f'wrfinput_{domain}').exists() or len(frames) < 7:
        pytest.skip('real WRF inputs and CPU history inventory are not mounted')
    run = Gen2Run(source)
    grid = run.grid(domain)
    land, static, _ = build_noahmp_land_state(source, domain, table_dir=TABLE)
    inputs = load_land_history_inputs(source / f'wrfinput_{domain}', table_dir=TABLE,
                                      parameters=static.parameters)
    start = datetime.strptime(frames[0].name[-19:], '%Y-%m-%d_%H:%M:%S')
    with Dataset(frames[0]) as ds:
        dt = float(ds.DT)
    boundary = load_lower_boundary(source, run.namelist, domain, run_start=start,
                                   dt_s=dt, shape=(grid.ny, grid.nx))
    return domain, grid, land, inputs, start, dt, boundary, frames


@pytest.mark.parametrize('hour', [0, 1, 6])
def test_shared_writer_vegfra_is_exact_real_cpu_wrf_percent(real_case, hour, tmp_path, monkeypatch):
    domain, grid, land, inputs, start, dt, boundary, frames = real_case
    valid = datetime.strptime(frames[hour].name[-19:], '%Y-%m-%d_%H:%M:%S')
    seconds = (valid - start).total_seconds()
    writer = object.__new__(pipeline._PerDomainWrfoutWriter)
    writer.run_start = start
    writer.bundles = {domain: SimpleNamespace(grid=grid, namelist=SimpleNamespace(
        use_noahmp=True, lower_boundary=boundary))}
    writer._land_history_inputs = {domain: inputs}
    writer._land_history_initial = {domain: jax.device_get(land)}
    writer.writer_diagnostics = {}
    writer._initial_surface_fields = lambda name: {}
    writer._merge_output_diagnostics = _merge_output_diagnostics
    writer.domain_authorities = {domain: bind_wrfout_domain_authority(domain, grid, grid)}
    writer._variable_subset = ('VEGFRA',)
    writer._full_variable_set = True
    writer._async_writer = None
    writer.census_io_ledger = None
    # No atmosphere/surface re-solve is needed for this prescribed field. Keep
    # the actual land mapping, aux4 selection, payload and NetCDF writer intact.
    monkeypatch.setattr(pipeline, 'assert_state_finite_at_boundary', lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, '_noahmp_surface_diagnostics_for_output', lambda *a, **kw: {})
    target = tmp_path / frames[hour].name
    writer._materialize_and_submit(name=domain, own_step=round(seconds / dt),
        carry=SimpleNamespace(state=SimpleNamespace(), noahmp_land=land),
        valid_time=valid, lead_seconds=seconds, lead_hours=seconds / 3600, path=target)
    with Dataset(target) as actual, Dataset(frames[hour]) as cpu:
        np.testing.assert_array_equal(actual['VEGFRA'][0], cpu['VEGFRA'][0])
        assert actual['VEGFRA'].units == cpu['VEGFRA'].units
        assert actual['VEGFRA'].description == cpu['VEGFRA'].description
        assert actual['VEGFRA'].dimensions == cpu['VEGFRA'].dimensions
        # The pristine Noah driver reads VEGFRA/100, and writes FVEG separately.
        # This real-frame control rejects substituting the phenology diagnostic.
        assert np.any(cpu['VEGFRA'][0] != 100 * cpu['FVEG'][0])
