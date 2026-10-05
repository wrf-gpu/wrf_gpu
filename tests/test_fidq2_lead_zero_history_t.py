"""fid-q2 R01 (F3): WRF history "T" at lead zero is th_phy_m_t0 = the wrfinput dry theta.

WRF's nest ``adjust_tempqv`` changes only t_2 (THM) and QV; th_phy_m_t0 is re-derived from them at the
first phy_prep, so CPU-WRF writes the input T at lead zero (WN3 nests: == wrfinput T bitwise) while
THM/QVAPOR carry the adjustment.  Later frames keep the model T.
"""
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_initial_history_wrf_t0 import _writer  # type: ignore  # noqa: E402


def _add_input_t(path, nz, ny, nx, value):
    with Dataset(path, "a") as ds:
        ds.createDimension("bottom_top", nz)
        ds.createVariable("T", "f4", ("Time", "bottom_top", "south_north", "west_east"))[0] = value


def test_lead_zero_t_is_the_wrfinput_dry_theta_not_the_model_t(tmp_path):
    def no_solve(*args, **kwargs):
        raise AssertionError("t0 must not re-solve physics")

    writer, state = _writer(tmp_path, no_solve)
    nz, ny, nx = np.asarray(state.theta).shape
    rng = np.random.default_rng(7)
    t_in = (5.0 + rng.standard_normal((nz, ny, nx))).astype(np.float32)
    _add_input_t(tmp_path / "wrfinput_d01", nz, ny, nx, t_in)
    result = writer("d01", 0, state)
    with Dataset(result["wrfout"]) as ds:
        written = np.asarray(ds.variables["T"][0])
    assert np.array_equal(written, t_in)
    assert not np.allclose(t_in, np.asarray(state.theta) - 300.0, atol=1e-3)  # non-vacuous


def test_lead_zero_without_input_t_keeps_the_model_t(tmp_path):
    def no_solve(*args, **kwargs):
        raise AssertionError("t0 must not re-solve physics")

    writer, state = _writer(tmp_path, no_solve)
    assert "T" not in writer._initial_surface_fields("d01")
