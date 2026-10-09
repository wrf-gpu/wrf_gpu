"""CPU mechanism checks; real PROD initialization is a separate GPU gate."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from gpuwrf.integration.init_kernels import InitKernelCache
from gpuwrf.integration.init_kernels import radiation_init_key
from gpuwrf.runtime import aot_executable as ax


@partial(jax.jit, static_argnames=("debug", "with_clear_sky", "column_tile_cols"))
def _kernel(x, *, debug=False, with_clear_sky=False, column_tile_cols=None,
            topography=None):
    # Plain adds avoid the known CPU-AOT fused-symbol loader limitation.
    y = x + (x if topography is None else topography)
    return {"flux": y, "clear": y if with_clear_sky else None}


def test_roundtrip_keeps_static_arguments_out_of_dynamic_abi(tmp_path):
    x = jnp.arange(4.0)
    topo = jnp.ones_like(x)
    kwargs = dict(debug=False, with_clear_sky=True, column_tile_cols=4, topography=topo)
    first = InitKernelCache(tmp_path)
    expected = _kernel(x, **kwargs)
    out = first(_kernel, x, **kwargs)
    np.testing.assert_array_equal(out["flux"], expected["flux"])
    assert first.events[-1]["status"]["aot_written"]
    _kernel.clear_cache()
    fresh = InitKernelCache(tmp_path)
    loaded = fresh(_kernel, x, **kwargs)
    assert fresh.events[-1]["hit"]
    for key in expected:
        assert np.asarray(loaded[key]).tobytes() == np.asarray(expected[key]).tobytes()
    # Dynamic values change without a new program; the new input is consumed.
    changed = fresh(_kernel, x + 2, **kwargs)
    np.testing.assert_array_equal(changed["flux"], x + 2 + topo)


def test_static_branch_and_shape_get_distinct_exact_hlo_addresses(tmp_path):
    cache = InitKernelCache(tmp_path)
    for clear, shape in [(False, 4), (True, 4), (True, 5)]:
        cache(_kernel, jnp.arange(float(shape)), with_clear_sky=clear)
    assert len({e["hlo_sha256"] for e in cache.events}) == 3
    assert all(e["status"]["aot_written"] for e in cache.events)


def test_corrupt_blob_falls_back_to_original_compilation(tmp_path):
    cache = InitKernelCache(tmp_path)
    x = jnp.arange(4.0)
    expected = cache(_kernel, x)
    from pathlib import Path
    Path(cache.events[-1]["status"]["aot_path"]).write_bytes(b"invalid")
    fresh = InitKernelCache(tmp_path)
    out = fresh(_kernel, x)
    assert not fresh.events[-1]["hit"]
    assert np.asarray(out["flux"]).tobytes() == np.asarray(expected["flux"]).tobytes()


def _native_columns():
    """Actual CPU-WRF PROD cells; a key/HLO test, not a physics oracle."""
    from netCDF4 import Dataset
    from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
    from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
    path = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/'
                'alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/'
                'cpu_case/wrfout_d01_2026-07-26_12:00:00')
    if not path.is_file():
        pytest.skip('CPU-WRF PROD snapshot unavailable: '+str(path))
    with Dataset(path) as ds:
        def surface(name):
            return jnp.asarray(np.asarray(ds.variables[name][0,20,40:42],dtype=np.float64))
        def col(name):
            return jnp.asarray(np.asarray(ds.variables[name][0,:,20,40:42],dtype=np.float64).T)
        pressure = col('P')+col('PB')
        qv = col('QVAPOR')
        temp = (col('T')+300.)*(pressure/100000.)**(287./1004.5)
        dz = jnp.diff((col('PH')+col('PHB'))/9.81,axis=-1)
        rho = pressure/(287.*temp*(1.+.61*qv))
        common = [temp,pressure,qv,col('QCLOUD'),col('QICE'),col('QSNOW'),col('QGRAUP'),col('CLDFRA')]
        sw = RRTMGSWColumnState(*common,surface('ALBEDO'),surface('COSZEN'),dz,rho)
        lw = RRTMGLWColumnState(*common,surface('TSK'),surface('EMISS'),dz,rho,
                              top_pressure_pa=float(ds.variables['P_TOP'][0]))
    return [(solve_rrtmg_sw_column,sw),(solve_rrtmg_lw_column,lw)]


def test_real_column_metadata_keys_match_hlo_identity(monkeypatch, cpu_pallas_interpret):
    from gpuwrf.integration import init_kernels as ik
    from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time
    from datetime import datetime, timezone
    gases = clwrf_ssp245_gases_for_time(datetime(2026,7,26,12,tzinfo=timezone.utc))
    groups = {}
    for fn,state in _native_columns():
        fields = ['co2_vmr','n2o_vmr','ch4_vmr']
        if hasattr(state,'cfc11_vmr'):
            fields += ['cfc11_vmr','cfc12_vmr']
        dynamic_gas = state.replace(**{n:jnp.asarray(getattr(gases,n),dtype=jnp.float64) for n in fields})
        changed_gas = dynamic_gas.replace(T=state.T+1.,co2_vmr=dynamic_gas.co2_vmr+1.e-6)
        variants = [(state,{}),(state,{'debug':False,'with_clear_sky':False,'column_tile_cols':None}),
                    (dynamic_gas,{}),(changed_gas,{}),
                    (state,{'with_clear_sky':True}), (state,{'column_tile_cols':1})]
        if hasattr(state,'top_pressure_pa'):
            variants.append((state.replace(top_pressure_pa=7000.),{}))
        for operand,kw in variants:
            key = radiation_init_key(fn,(operand,),kw)
            assert key is not None
            hlo = ax.hlo_sha256_from_lowered(fn.lower(operand,**kw))
            if key in groups:
                assert groups[key]==hlo, 'one metadata key addressed two programs'
            groups[key]=hlo
        a = radiation_init_key(fn,(state,),{})
        assert a==radiation_init_key(fn,(state,),{'debug':False,'with_clear_sky':False,'column_tile_cols':None})
        assert radiation_init_key(fn,(dynamic_gas,),{})==radiation_init_key(fn,(changed_gas,),{})
        assert a!=radiation_init_key(fn,(dynamic_gas,),{})
        assert a!=radiation_init_key(fn,(state,),{'with_clear_sky':True})
        assert a!=radiation_init_key(fn,(state,),{'column_tile_cols':1})
        # The legacy solver requires a consistent precision across its state
        # and tables; an unsupported mixed dtype must still miss the old key.
        assert a!=radiation_init_key(fn,(jax.tree.map(lambda x:x.astype(jnp.float32),state),),{})
        # Source, constants/tables and target/compile options cannot weaken to an
        # old key; use the authoritative key builder instead of duplicating it.
        with monkeypatch.context() as m:
            m.setattr(ik,'_table_digest',lambda:'changed-table-content')
            assert a!=radiation_init_key(fn,(state,),{})
        with monkeypatch.context() as m:
            m.setattr(ik.ck,'source_fingerprint_hash',lambda:'changed-source')
            assert a!=radiation_init_key(fn,(state,),{})
        with monkeypatch.context() as m:
            m.setattr(ik.ck,'exec_env_hash',lambda dev=None:'changed-target-or-options')
            assert a!=radiation_init_key(fn,(state,),{})


def test_metadata_hit_never_lowers_original_kernel(tmp_path, monkeypatch):
    from gpuwrf.integration import init_kernels as ik
    fn,state = _native_columns()[0]
    def refuse_lower(*args,**kwargs):
        raise AssertionError('metadata hit must not lower')
    monkeypatch.setattr(fn,'lower',refuse_lower)
    def loaded(name,cache_dir,**kwargs):
        assert kwargs['cheap_key']==radiation_init_key(fn,(state,),{})
        return lambda operand:operand.T, {'hlo_sha256':'guarded-blob-hlo'}
    monkeypatch.setattr(ik.aotp,'load_domain_blob',loaded)
    cache = InitKernelCache(tmp_path)
    assert cache(fn,state) is state.T
    assert cache.events[-1]['hit'] and cache.events[-1]['lower_s']==0.0


def test_verify_mode_quarantines_wrong_hlo_before_execution(tmp_path, monkeypatch):
    from gpuwrf.integration import init_kernels as ik
    from types import SimpleNamespace
    fn,state = _native_columns()[0]
    monkeypatch.setenv('GPUWRF_AOT_VERIFY','1')
    rejected_executions = []
    quarantined = []
    def bad(*args,**kwargs):
        rejected_executions.append(True)
        raise AssertionError('wrong-HLO callable executed')
    def load(name,cache_dir,**kw):
        return (bad,{'hlo_sha256':'different-program'}) if 'cheap_key' in kw else (None,{})
    lowered = SimpleNamespace(as_text=lambda:'original-program', compile=lambda:lambda operand:operand.T)
    monkeypatch.setattr(fn,'lower',lambda *a,**kw:lowered)
    monkeypatch.setattr(ik.aotp,'load_domain_blob',load)
    monkeypatch.setattr(ik.aotp,'quarantine_cheap_key',lambda *a,**kw:quarantined.append((a,kw)))
    monkeypatch.setattr(ik.aotp,'_serialize_domain_blob',lambda *a,**kw:{'aot_written':False})
    assert InitKernelCache(tmp_path)(fn,state) is state.T
    assert quarantined and not rejected_executions


def test_metadata_roundtrip_uses_actual_loader_schema(tmp_path, monkeypatch):
    """Exercise the real exporter AND loader, rather than mocking a cache hit."""
    from gpuwrf.integration import init_kernels as ik
    key = ik.ck.canonical_digest(('cpu-metadata-roundtrip',))
    monkeypatch.setattr(ik,'radiation_init_key',lambda *a,**kw:key)
    x = jnp.arange(4.)
    cold = InitKernelCache(tmp_path)
    expected = cold(_kernel,x)
    assert cold.events[-1]['status']['aot_written']
    _kernel.clear_cache()
    # A fresh eager wrapper must reach the actual metadata loader without
    # lowering. This would have caught the rejected custom schema on SI12.
    monkeypatch.setattr(_kernel,'lower',lambda *a,**kw:pytest.fail('warm metadata load lowered'))
    warm = InitKernelCache(tmp_path)
    out = warm(_kernel,x)
    assert warm.events[-1]['lower_s']==0 and warm.events[-1]['hit']
    assert np.asarray(out['flux']).tobytes()==np.asarray(expected['flux']).tobytes()


def test_resolved_radiation_flags_survive_live_env_normalization(tmp_path):
    """Regression for RI02: same live environment, different imported HLOs."""
    import subprocess,sys,os,json
    import gpuwrf
    # Check availability in the parent, so missing external WRF evidence skips
    # with a reason instead of making the helper subprocess fail obscurely.
    _native_columns()
    helper = Path(__file__).parent/'helpers/init_import_flags_probe.py'
    results = []
    for imported in ['0','1']:
        dest = tmp_path/('import'+imported+'.json')
        env = dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0')
        source_root = str(Path(gpuwrf.__file__).resolve().parents[1])
        env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
        # The probe changes one imported switch. Unrelated native kernels are
        # GPU-only and must not be inherited from a release-defaults caller.
        for flag in ('GPUWRF_RRTMG_SW_FUSED_QUADRATURE',
                     'GPUWRF_RRTMG_LW_FUSED_TRANSFER',
                     'GPUWRF_RRTMG_LW_BAND_SUMS', 'GPUWRF_MCICA_KISS_KERNEL'):
            env[flag] = '0'
        env.pop('GPUWRF_AOT_VERIFY',None)
        subprocess.run([sys.executable,str(helper),'GPUWRF_MCICA_JUMPAHEAD',imported,'0',str(dest)],
                       env=env,check=True,capture_output=True,text=True,timeout=90)
        results.append(json.loads(dest.read_text()))
    for a,b in zip(results[0]['records'],results[1]['records'],strict=True):
        assert a['hlo']!=b['hlo'], 'injection did not exercise a different program'
        assert a['key']!=b['key'], 'different imported programs shared a metadata key'


def test_every_resolved_radiation_switch_changes_key(monkeypatch):
    from gpuwrf.integration import init_kernels as ik
    from gpuwrf.physics import rrtmg_sw,rrtmg_lw
    from gpuwrf.kernels import rad_mcica
    switches = [(rrtmg_sw,n) for n in ['_SW_COLUMN_TILING','_SW_COLUMN_TILE_COLS','_MCICA_JUMPAHEAD','_FUSED_QUADRATURE']]
    switches += [(rrtmg_lw,n) for n in ['_LW_COLUMN_TILING','_LW_COLUMN_TILE_COLS','_LW_CLOUD_OPTICS_FP32','_MCICA_JUMPAHEAD','_FUSED_TRANSFER','_CLEAR_SKY_COLUMN_CLOUD']]
    switches += [(rad_mcica,'_LEGACY_FP64')]
    columns = _native_columns()
    for module,name in switches:
        for fn,state in columns:
            before = radiation_init_key(fn,(state,),{})
            value = getattr(module,name)
            with monkeypatch.context() as m:
                m.setattr(module,name,not value if isinstance(value,bool) else value+1)
                assert before!=radiation_init_key(fn,(state,),{}),(module.__name__,name,fn.__name__)
        # Respect JAX's import-time capture rule when changing module globals.
        rrtmg_sw.solve_rrtmg_sw_column.clear_cache()
        rrtmg_lw.solve_rrtmg_lw_column.clear_cache()
