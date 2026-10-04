"""Root (standalone) init applies WRF start_em set_w_surface(fill_w_flag) like the live nest (CPU).

WRF start_em.F runs set_w_surface for every domain at the start of a non-restart run whose
input surface W is identically ~0 (real.exe wrfinput).  Truth: pristine serial WRF d01 t=0
history frame (written after start_domain) -- evidence tests/v025/b_core/root_init_w_evidence.json.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
PRISTINE_T0 = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase2b/BC44_relax/pristine_d01_nodump/"
                   "wrfout_d01_2026-07-26_00:00:00")


@pytest.fixture(scope="module")
def root_case():
    if not (RUN_DIR / "wrfinput_d01").exists():
        pytest.skip("PROD s0 case unavailable")
    from gpuwrf.contracts import state as state_contract
    original = state_contract._gpu_device
    state_contract._gpu_device = lambda: jax.devices()[0]
    try:
        from gpuwrf.integration.d02_replay import build_replay_case
        yield build_replay_case(RUN_DIR, domain="d01", standalone=True)
    finally:
        state_contract._gpu_device = original


def test_root_standalone_init_sets_kinematic_w(root_case):
    from netCDF4 import Dataset
    from gpuwrf.integration.d02_replay import _wrf_set_w_surface
    assert jax.devices()[0].platform == "cpu"
    w = np.asarray(root_case.state.w)
    with Dataset(RUN_DIR / "wrfinput_d01") as ds:
        assert float(np.abs(ds["W"][0]).max()) == 0.0  # real.exe leaves W at zero
    assert float(np.abs(w[0]).max()) > 1e-3  # gate opened: kinematic surface w present
    expected, opened, _ = _wrf_set_w_surface(np.zeros_like(w), root_case.state.u, root_case.state.v,
                                             grid=root_case.grid, metrics=root_case.metrics)
    assert opened
    np.testing.assert_array_equal(w, expected.astype(w.dtype))
    if PRISTINE_T0.exists():
        with Dataset(PRISTINE_T0) as ds:
            np.testing.assert_array_equal(w.astype(np.float64), np.asarray(ds["W"][0], np.float64))


def test_set_w_surface_gate_keeps_nonzero_input(root_case):
    from gpuwrf.integration.d02_replay import _wrf_set_w_surface
    w_in = np.full(np.asarray(root_case.state.w).shape, 0.01, np.float32)
    out, opened, surface_max = _wrf_set_w_surface(w_in, root_case.state.u, root_case.state.v,
                                                  grid=root_case.grid, metrics=root_case.metrics)
    assert not opened and surface_max == pytest.approx(0.01)
    np.testing.assert_array_equal(out, w_in)
