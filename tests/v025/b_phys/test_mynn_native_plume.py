"""Native plume gate against the frozen independent WRF active-column oracle."""
import importlib.util
from pathlib import Path
import numpy as np


def test_active_native_plume_matches_wrf(monkeypatch):
    from gpuwrf.physics import mynn_edmf
    from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
    path=Path(__file__).resolve().parents[2]/'test_mynn_edmf_oracle.py'
    spec=importlib.util.spec_from_file_location('b_phys_mynn_oracle',path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    monkeypatch.setattr(mynn_edmf,'dmp_mf_columns',dmp_mf_columns_native)
    import json
    c=json.loads(Path(oracle.COL).read_text())
    oracle.test_dmp_mf_matches_wrf_oracle(c)
    result=oracle._dmp_result(c)
    assert all(str(x.dtype)=='float32' for x in result.values())
    assert all(np.isfinite(np.asarray(x)).all() for x in result.values())


def test_native_hook_uses_existing_coupler_seam(monkeypatch):
    monkeypatch.setenv('GPUWRF_EDMF_FUSED_PLUME','1')
    monkeypatch.setenv('GPUWRF_MYNN_FP32_PLUME','1')
    path=Path(__file__).resolve().parents[2]/'test_mynn_edmf_oracle.py'
    spec=importlib.util.spec_from_file_location('b_phys_mynn_hook_oracle',path)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    import json
    column=json.loads(Path(oracle.COL).read_text())
    oracle.test_dmp_mf_matches_wrf_oracle(column)


def test_limiter_active_native_plume_matches_pristine_wrf():
    import json
    from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
    fixture = json.loads((Path(__file__).parent / 'fixtures/limiter_active_wrf.json').read_text())
    assert .01 <= fixture['adjustment'] < 1.0
    inputs = {key: np.asarray(value) for key, value in fixture['inputs'].items()}
    result = dmp_mf_columns_native(**inputs)
    assert float(result['active'][0]) == 1.0
    for name, values in fixture['expected'].items():
        actual = np.asarray(result[name][0])
        expected = np.asarray(values)
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        error = np.max(np.abs(actual - expected)) / max(1e-30, np.max(np.abs(expected)))
        assert error <= fixture['tolerance'], (name, error)
