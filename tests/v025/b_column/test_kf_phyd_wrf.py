"""KF pressure delivery and independent Exner against original warm WRF."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import cumulus_kf as kf
from gpuwrf.kernels import phys_kf_column as K

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
FLAG='GPUWRF_KF_PHYD_WRF'

def finite(a,b):
    assert np.isfinite(a).all(), 'nonfinite candidate'
    assert np.isfinite(b).all(), 'nonfinite reference'

@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('candidate',[True,False])
def test_finite_gate_rejects_both_roles(value,candidate):
    a,b=np.ones(2),np.ones(2)
    (a if candidate else b)[0]=value
    with pytest.raises(AssertionError):finite(a,b)

def test_pi_phy_uses_temperature_tendency_before_pressure_conversion(monkeypatch):
    monkeypatch.setenv('GPUWRF_KF_COLUMN_FP32','1')
    original=K.kf_column
    monkeypatch.setattr(K,'kf_column',lambda *a,**kw:original(*a,**kw,interpret=True))
    kf.kf_eta_para.clear_cache()
    d=json.loads((ROOT/'proofs/v060/savepoints/kf_case_2.json').read_text())
    a=[jnp.asarray(d['columns'][n],jnp.float32) for n in ['T','QV','P','DZ','RHO','W0AVG','U','V']]
    pi=jnp.full_like(a[0],.8)
    result=kf.step_kf_column(*a,d['scalars']['DT'],d['scalars']['DX'],pi_phy=pi)
    raw=original(*a,d['scalars']['DT'],d['scalars']['DX'],len(a[0]),interpret=True)
    actual=np.asarray(result.diagnostics.cumulus['rthcuten']);expected=np.asarray(raw['DTDT']/pi)
    finite(actual,expected)
    assert np.any(expected!=0)
    np.testing.assert_array_equal(actual,expected)
    kf.kf_eta_para.clear_cache()

def test_live_flag_splits_trace_key(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash
    monkeypatch.setenv(FLAG,'0');off=global_trace_env_hash()
    monkeypatch.setenv(FLAG,'1');assert global_trace_env_hash()!=off

def test_no_trigger_preserves_expired_warm_nca(monkeypatch):
    monkeypatch.setenv('GPUWRF_KF_COLUMN_FP32','1')
    original=K.kf_column
    monkeypatch.setattr(K,'kf_column',lambda *a,**kw:original(*a,**kw,interpret=True))
    kf.kf_eta_para.clear_cache()
    d=json.loads((ROOT/'proofs/v060/savepoints/kf_case_5.json').read_text())
    a=[jnp.asarray(d['columns'][n],jnp.float32) for n in ['T','QV','P','DZ','RHO','W0AVG','U','V']]
    # Original tau48 countdown: 30 - DT54 = -24 after final-step consumption.
    result=kf.step_kf_column(*a,d['scalars']['DT'],d['scalars']['DX'],
        nca=jnp.float32(-24),pi_phy=jnp.ones_like(a[0]))
    assert int(result.diagnostics.cumulus['ishall'])==2
    assert np.isfinite(np.asarray(result.carry.cumulus['nca']))
    assert float(result.carry.cumulus['nca'])==-24.
    kf.kf_eta_para.clear_cache()

@pytest.mark.parametrize('lead',[12,36,48])
def test_original_warm_cps_bound_and_held_rate_carry(lead,tmp_path):
    fixture=os.environ.get('GPUWRF_KF_ORIGINAL_FIXTURE')
    if fixture is None:
        pytest.skip('requires registered original P0227 warm KF reference package')
    fixture=Path(fixture)
    pin=json.loads((HERE/'kf_warm_reference.json').read_text())
    case=pin['cases'][str(lead)]
    for filename,key in [(f'tau{lead}.npz','input_sha256'),(f'tau{lead}_actual_ref.bin','reference_sha256')]:
        assert hashlib.sha256((fixture/filename).read_bytes()).hexdigest()==case[key]
    assert hashlib.sha256((fixture/'warm_oracle.f90').read_bytes()).hexdigest()==pin['wrapper_sha256']
    for source,expected in pin['original_reference_source_SHA256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',GPUWRF_FAST_DEFAULTS='0',
             GPUWRF_KF_PHYD_WRF='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(HERE/'replay_kf_warm.py'),'--fixture',str(fixture),
                           '--output',str(tmp_path),'--lead',str(lead)],env=env,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-2000:]
    actual=np.load(tmp_path/f'tau{lead}_port.npz')['records'].astype(np.float64)
    ref=np.fromfile(fixture/f'tau{lead}_actual_ref.bin','<f4').reshape(8400,8,317).astype(np.float64)
    finite(actual,ref)
    active=np.any(ref[:,:,:264]!=0,axis=(1,2))
    assert int(active.sum())==case['active_reference']
    bad=np.zeros((8400,8),bool)
    for i in range(6):
        aa=actual[:,:,i*44:(i+1)*44];rr=ref[:,:,i*44:(i+1)*44]
        error=np.max(np.abs(aa-rr),axis=-1)
        scale=np.maximum(np.max(np.abs(rr),axis=-1),1e-9)
        bad|=(error>1e-9)&(error>.002*scale)
    bad|=np.abs(actual[:,:,308]-ref[:,:,308])>1e-6
    bad|=np.abs(actual[:,:,309]-ref[:,:,309])>np.maximum(.0001,.003*np.abs(ref[:,:,309]))
    failures=int((bad.any(axis=1)&active).sum())
    assert failures<=case['maximum_failed_active'],(lead,failures,case['maximum_failed_active'])
    # When both operators did not trigger a new cloud, the countdown must be
    # retained exactly. Trigger mismatches remain in the frozen numeric gate.
    inactive=(ref[:,:,311]==2)&(actual[:,:,308]<=0)
    np.testing.assert_allclose(actual[:,:,308][inactive],ref[:,:,308][inactive],rtol=0,atol=1e-6)
    # Existing source STEPCU/NCA advance/last-step consumption must still agree.
    np.testing.assert_array_equal(actual[:,:,315:],ref[:,:,315:])
