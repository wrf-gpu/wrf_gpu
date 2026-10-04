"""Bit gates for init-only native loops calling the SAME scalar libm pointers."""
from pathlib import Path
import numpy as np
import pytest
import jax
from netCDF4 import Dataset
from gpuwrf.integration import d02_replay as rp
from gpuwrf.integration import _init_libm_arrays as loops


@pytest.fixture
def implementations(tmp_path, monkeypatch):
    monkeypatch.setenv('GPUWRF_JAX_CACHE_DIR',str(tmp_path/'native'))
    original_load = loops.load
    original_load.cache_clear()
    old = rp._WRFInitLibm32(native_arrays=False)
    new = rp._WRFInitLibm32()
    if old._libm is None or loops.load() is None:
        pytest.skip('glibc float32 libm or local C compiler unavailable')
    yield old,new
    original_load.cache_clear()


def test_scalar_primitive_bytes_and_fallback(implementations, monkeypatch):
    old,new = implementations
    bits = np.array([0,0x80000000,1,0x80000001,0x3f800000,0x7f7fffff,
                     0x7f800000,0xff800000,0x7fc00000,0x7fa12345],dtype=np.uint32)
    x = bits.view(np.float32)
    for name in ['exp','log','sqrt_via_pow']:
        assert getattr(old,name)(x).tobytes()==getattr(new,name)(x).tobytes(),name
    for exponent in [np.float32(.5),np.float32(287./1004.),np.float32(-1.)]:
        assert old.pow(x,exponent).tobytes()==new.pow(x,exponent).tobytes()
    # Failure/unavailability falls back to the original per-float implementation.
    monkeypatch.setattr(loops,'load',lambda:None)
    finite = np.linspace(.01,5,99,dtype=np.float32)
    assert old.log(finite).tobytes()==new.log(finite).tobytes()


@pytest.mark.parametrize('domain',['d01','d02'])
def test_all_libm_native_helpers_on_wrf_inputs_byte_exact(domain, implementations, monkeypatch):
    old,new = implementations
    root = Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
    path = root/('wrfinput_'+domain)
    if not path.is_file():
        pytest.skip('native WRF fixture unavailable: '+str(path))
    run = rp.Gen2Run(root)
    metrics = rp.load_wrfinput_metrics(path)
    grid = run.grid(domain).as_grid_spec()
    with Dataset(path) as ds:
        fields={n:np.asarray(ds.variables[n][0]) for n in ['HGT','PH','MU','T','W','U','V']}
    results=[]
    for implementation in [old,new]:
        monkeypatch.setattr(rp,'_WRF_INIT_LIBM32',implementation)
        base = rp._wrf_start_domain_base_from_hgt(run,domain,hgt=fields['HGT'],metrics=metrics)
        pert = rp._wrf_live_nest_start_domain_perturb_init(run,domain=domain,grid=grid,metrics=metrics,
            ph_perturbation=fields['PH'],mu_perturbation=fields['MU'],theta_full=fields['T']+300.,
            w=fields['W'],u=fields['U'],v=fields['V'],ht_fine=fields['HGT'],
            base_pb=base[0],base_mub=base[1],base_phb=base[2])
        jax.block_until_ready(base+pert[:3])
        results.append([np.asarray(x) for x in base+pert[:3]])
    for a,b in zip(*results,strict=True):
        assert (a.shape,a.dtype,a.tobytes())==(b.shape,b.dtype,b.tobytes())
