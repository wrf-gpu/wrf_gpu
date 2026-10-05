"""WN3 native namelist binding and CLI cadence reporting (no forecast)."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from gpuwrf import cli
from gpuwrf.integration import nested_pipeline as npl
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.namelist_check import collect_namelist_warnings
from gpuwrf.runtime.operational_mode import OperationalNamelist


def test_wn3_cli_reports_applied_stepcu_per_domain(tmp_path, monkeypatch, capsys):
    # Exact native WN3 20260227_18z_a1 namelist; tiny binding test, no synthetic
    # physics/State or GPU/fidelity claim. No lane-specific evidence paths.
    fixture = Path(__file__).parent / 'fixtures/wn3_20260227_cadence.namelist'
    (tmp_path/'namelist.input').write_bytes(fixture.read_bytes())
    run = Gen2Run(tmp_path)
    names = ('d01','d02','d03')
    monkeypatch.setattr(run, 'grid', lambda name: SimpleNamespace(
        parent_id=npl._domain_int(run,'domains','parent_id',name,1),
        parent_grid_ratio=npl._domain_int(run,'domains','parent_grid_ratio',name,1)))
    dts = npl._dt_by_domain(run, names)
    # Avoid model allocation while exercising the actual _make_namelist binding
    # and dataclass replacement. Only the expensive from_grid allocation is stubbed.
    monkeypatch.setattr(OperationalNamelist, 'from_grid',
                        lambda grid, **kwargs: OperationalNamelist(grid=grid, **kwargs))
    domains = {}
    for name in names:
        grid = SimpleNamespace(projection=SimpleNamespace(dx_m=9000.,dy_m=9000.))
        nl = npl._make_namelist(grid=grid, tendencies=None, metrics=None,
            dt_s=dts[name], parent_dt_s=None, run_start=datetime(2026,2,27,18,tzinfo=timezone.utc),
            radiation_static=None, cu_physics=npl._domain_cu_physics(run,name),
            cudt_minutes=npl._domain_float(run,'physics','cudt',name,0.))
        domains[name] = {'namelist': {'cu_physics':nl.cu_physics,
                                    **npl._cumulus_cadence_metadata(nl)}}
    assert dts == {'d01':54., 'd02':18., 'd03':6.}
    assert [domains[d]['namelist']['cumulus_cadence_steps'] for d in names] == [6,17,50]
    assert [domains[d]['namelist']['cu_physics'] for d in names] == [1,0,0]
    assert all(domains[d]['namelist']['cudt_minutes']==5. for d in names)
    assert not any('cudt' in w for w in collect_namelist_warnings(run.namelist))

    def execute(config):
        assert config.max_dom==3 and config.input_dir==tmp_path
        return {'verdict':'PIPELINE_GREEN', 'metadata':{'domains':domains}}
    monkeypatch.setattr(npl,'execute_nested_pipeline',execute)
    # The CPU test already imported JAX; allocator bootstrap/re-exec is a
    # separate CLI test concern. Keep this process on the declared CPU backend.
    monkeypatch.setenv('XLA_PYTHON_CLIENT_ALLOCATOR','cuda_async')
    rc = cli.main(['run','--input-dir',str(tmp_path),'--output-dir',str(tmp_path/'out'),
                   '--domains-from-namelist','--hours','1','--force-gpu-run'])
    assert rc==0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload['metadata']['domains']==domains
    assert 'cudt=5 cadence not honored' not in captured.err


def test_run_proof_attests_every_resolved_scalar_control(tmp_path, monkeypatch):
    """alisios §3.6: the run proof (metadata.domains[d].namelist_resolved) carries the EXECUTED options, incl. microphysics,
    PBL, surface layer, radiation and diffusion/damping controls (v0.3.0 proofs: only a 14-control summary)."""
    fixture = Path(__file__).parent / 'fixtures/wn3_20260227_cadence.namelist'
    (tmp_path/'namelist.input').write_bytes(fixture.read_bytes())
    run = Gen2Run(tmp_path)
    monkeypatch.setattr(run, 'grid', lambda name: SimpleNamespace(
        parent_id=npl._domain_int(run,'domains','parent_id',name,1),
        parent_grid_ratio=npl._domain_int(run,'domains','parent_grid_ratio',name,1)))
    monkeypatch.setattr(OperationalNamelist, 'from_grid',
                        lambda grid, **kwargs: OperationalNamelist(grid=grid, **kwargs))
    grid = SimpleNamespace(projection=SimpleNamespace(dx_m=9000.,dy_m=9000.), nz=44, ny=70, nx=120)
    nl = npl._make_namelist(grid=grid, tendencies=None, metrics=None, dt_s=54., parent_dt_s=None,
        run_start=datetime(2026,2,27,18,tzinfo=timezone.utc), radiation_static=None,
        cu_physics=npl._domain_cu_physics(run,'d01'), cudt_minutes=npl._domain_float(run,'physics','cudt','d01',0.))
    opts = npl._resolved_namelist_options(nl, grid)
    json.dumps(opts)  # JSON-safe scalars only
    for key in ('mp_physics', 'bl_pbl_physics', 'sf_sfclay_physics', 'sf_surface_physics', 'ra_lw_physics', 'ra_sw_physics',
                'cu_physics', 'diff_opt', 'km_opt', 'damp_opt', 'diff_6th_opt', 'dt_s', 'radiation_cadence_steps'):
        assert key in opts, key
    assert opts['dt_s'] == 54. and opts['cu_physics'] == 1 and opts['mass_shape'] == [44, 70, 120]
    assert len(opts) > 35
    src = Path(npl.__file__).read_text()  # deletion-sensitive: the proof dict must carry the call
    assert '"namelist_resolved": _resolved_namelist_options(namelist, case.grid)' in src
