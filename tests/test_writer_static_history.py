"""Real CPU-WRF static-history oracles: WN3 0227 and PROD, t0/h1/h6."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io import wrfout_writer as output
from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested

WN3 = Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
PROD = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')
DOMAINS = [(WN3, 'd01'), (WN3, 'd02'), (WN3, 'd03'), (PROD, 'd01'), (PROD, 'd02')]
CONSTANTS = ('T00', 'TLP_STRAT', 'WATER_DEPTH', 'GOT_VAR_SSO', 'DX2D', 'AREA2D', 'MAX_MSFTX', 'MAX_MSFTY', 'CROPCAT', 'SSTSK', 'SST_INPUT')
INITIAL = ('TH2', 'UST', 'COSZEN')

def read(path, names):
    with Dataset(path) as ds:
        result = {}
        for n in names:
            if n in ds.variables:
                v = ds[n]
                result[n] = np.asarray(v[0] if v.dimensions and v.dimensions[0] == 'Time' else v[:]).copy()
        return result

@pytest.fixture(params=DOMAINS, scope='module', ids=lambda x: ('WN3' if x[0] == WN3 else 'PROD') + '-' + x[1])
def real_case(request):
    base, domain = request.param
    if not (base / f'wrfinput_{domain}').is_file():
        pytest.skip('original CPU-WRF oracle/input is not mounted')
    run = Gen2Run(base)
    source = run.grid(domain)
    grid = source.as_grid_spec()
    statics, _ = daily._load_static_latlon_writer_diagnostics(run, domain, grid=grid)
    # Probe the exact constructor source, including scalar/integer passthrough.
    assert all(n in statics for n in daily.WRITER_INPUT_HISTORY_FIELDS)
    writer = object.__new__(nested._PerDomainWrfoutWriter)
    writer.input_dir = base
    writer.bundles = {domain: SimpleNamespace(grid=grid)}
    initial = writer._initial_surface_fields(domain)
    frames = sorted(base.glob(f'wrfout_{domain}_*'))
    assert len(frames) > 6
    return base, domain, grid, source, statics, initial, [frames[i] for i in (0, 1, 6)]

def prepare(real_case, frame, tmp_path):
    base, domain, grid, source, statics, initial, paths = real_case
    # Every state value is read from the real CPU frame. No model is advanced.
    names = ('T', 'THM', 'QVAPOR', 'QCLOUD', 'QICE', 'QRAIN', 'U', 'V', 'W', 'P', 'PB', 'PH', 'PHB', 'MU', 'MUB', 'T2', 'TSK', 'PSFC', 'Q2', 'U10', 'V10', 'HFX', 'LH', 'UST', 'SST', 'LU_INDEX', 'LANDMASK')
    raw = read(paths[frame], names)
    state = dict(raw)
    if frame == 0:
        # t0 CPU history has already received WRF's UST seed. The port's real
        # input omits UST, so do not smuggle that expected history value into
        # both the writer state and diagnostics before testing the new seed.
        state.pop('UST')
    state.update(theta=raw.get('THM', raw['T']) + np.float32(300), p_total=raw['P'] + raw['PB'], ph_total=raw['PH'] + raw['PHB'], mu_total=raw['MU'] + raw['MUB'])
    diagnostics = dict(statics)
    diagnostics.update(read(base / f'wrfinput_{domain}', ('MAPFAC_MX', 'MAPFAC_MY', 'MAPFAC_M')))
    diagnostics.update({n: raw[n] for n in ('T2', 'TSK', 'PSFC', 'Q2', 'U10', 'V10', 'HFX', 'LH', 'UST') if n in raw})
    if frame == 0:
        diagnostics.pop('UST')
        diagnostics.update(initial)
    start = datetime.strptime(paths[0].name[11:], '%Y-%m-%d_%H:%M:%S')
    valid = datetime.strptime(paths[frame].name[11:], '%Y-%m-%d_%H:%M:%S')
    lead = (valid - start).total_seconds() / 3600
    before = {n: a.tobytes() for n, a in state.items()}
    payload = output.prepare_wrfout_payload(state, grid, SimpleNamespace(sst_skin=False, use_adaptive_time_step=False), tmp_path / 'history.nc', domain=domain, domain_authority=output.bind_wrfout_domain_authority(domain, source, grid), run_start=start, valid_time=valid, lead_hours=lead, diagnostics=diagnostics, full_variable_set=True)
    assert before == {n: a.tobytes() for n, a in state.items()}, 'output mutated state'
    return payload

@pytest.mark.parametrize('frame', [0, 1, 2], ids=['t0', 'h1', 'h6'])
def test_history_constants_equal_original_cpu_wrf(real_case, frame, tmp_path):
    payload = prepare(real_case, frame, tmp_path)
    wanted = (*CONSTANTS, *INITIAL) if frame == 0 else CONSTANTS
    truth = read(real_case[-1][frame], wanted)
    for n in wanted:
        assert np.isfinite(truth[n]).all() and np.isfinite(payload.fields[n]).all(), n
        np.testing.assert_array_equal(payload.fields[n], truth[n], err_msg=n)
        assert payload.fields[n].dtype == truth[n].dtype, n
    # Persist the same payload, so the oracle also exercises physical dims/types.
    output.write_prepared_wrfout(payload, expected_domain=real_case[1], expected_domain_authority=output.bind_wrfout_domain_authority(real_case[1], real_case[3], real_case[2]))
    with Dataset(payload.target) as ds, Dataset(real_case[-1][frame]) as cpu:
        for n in wanted:
            assert ds[n].dimensions == cpu[n].dimensions, n
            assert ds[n].dtype == cpu[n].dtype, n
            np.testing.assert_array_equal(np.asarray(ds[n][0]), truth[n], err_msg=n)

def test_later_frames_keep_computed_th2_and_coszen(real_case, tmp_path):
    from gpuwrf.coupling.physics_couplers import _compute_coszen
    payload = prepare(real_case, 1, tmp_path)
    raw = read(real_case[-1][1], ('T2', 'PSFC'))
    expect = raw['T2'].astype(np.float64) * (output.P0_PA / np.maximum(raw['PSFC'].astype(np.float64), 1)) ** output.R_D_OVER_CP
    np.testing.assert_array_equal(payload.fields['TH2'], expect.astype(np.float32))
    latlon = read(real_case[-1][0], ('XLAT', 'XLONG'))
    expected = _compute_coszen(latlon['XLAT'].astype(np.float64), latlon['XLONG'].astype(np.float64), payload.run_start_dt, lead_seconds=payload.lead_hours*3600)
    np.testing.assert_array_equal(payload.fields['COSZEN'], np.asarray(expected, dtype=np.float32))
    np.testing.assert_array_equal(payload.fields['UST'], read(real_case[-1][1], ('UST',))['UST'])

def test_enabled_sst_skin_uses_named_source(real_case):
    raw = read(real_case[0] / f'wrfinput_{real_case[1]}', ('SST',))['SST']
    for name in ('SST_INPUT', 'SSTSK'):
        actual = output._full_source_value(name, raw.shape, dtype=np.float32, diagnostics={name: raw}, state=None, land_state=None, grid=None, namelist={'sst_skin': True})
        np.testing.assert_array_equal(actual, raw)
