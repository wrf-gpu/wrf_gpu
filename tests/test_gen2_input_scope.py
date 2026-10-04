"""Read-only init scopes preserve data and release handles on every exit."""
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.io import gen2_accessor as ga


def test_scope_nested_exception_and_lazy_read_after_close(tmp_path, monkeypatch):
    path = tmp_path / 'wrfinput_d01'
    with Dataset(path, 'w') as ds:
        ds.createDimension('Time', 1)
        ds.createDimension('x', 3)
        ds.createVariable('T', 'f4', ('Time', 'x'))[:] = [[1, 2, 3]]
        ds.createVariable('Z', 'i4', ('x',))[:] = [4, 5, 6]
    opened = []
    def counted(*args, **kwargs):
        ds = Dataset(*args, **kwargs)
        opened.append(ds)
        return ds
    monkeypatch.setattr(ga, 'Dataset', counted)
    run = ga.Gen2Run(tmp_path)
    with pytest.raises(ValueError, match='test exit'):
        with run.input_read_scope('d01'):
            assert run.wrfinput_variables('d01') == ['T', 'Z']
            lazy = run.load_wrfinput('d01', 'T')
            with run.input_read_scope('d01'):
                np.testing.assert_array_equal(run._read_variable(path, 'Z', None), [4, 5, 6])
            assert opened[0].isopen()
            with pytest.raises(KeyError):
                run.load_wrfinput('d01', 'missing')
            assert len(opened) == 1
            raise ValueError('test exit')
    assert not run._input_handles
    assert not opened[0].isopen()
    # A lazy array outliving the scope reopens the input instead of retaining
    # a dead handle; the ordinary device cache still returns the same object.
    first = lazy.materialize()
    np.testing.assert_array_equal(first, [1, 2, 3])
    assert lazy.materialize() is first
    assert len(opened) == 2
    assert not opened[-1].isopen()


@pytest.mark.parametrize('domain', ['d01', 'd02'])
def test_actual_prod_input_arrays_byte_exact(domain):
    root = Path('<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725')
    path = root / ('wrfinput_' + domain)
    if not path.is_file():
        pytest.skip('native PROD WRF input unavailable: ' + str(path))
    fields = 'P PB PH PHB MU MUB T QVAPOR U V W TSK XLAND TSLB SMOIS'.split()
    before = ga.Gen2Run(root)
    expected = {n: before._read_variable(path, n, 0) for n in fields}
    after = ga.Gen2Run(root)
    with after.input_read_scope(domain):
        for name, x in expected.items():
            y = after._read_variable(path, name, 0)
            assert (y.dtype, y.shape, y.tobytes()) == (x.dtype, x.shape, x.tobytes())
    assert not after._input_handles
