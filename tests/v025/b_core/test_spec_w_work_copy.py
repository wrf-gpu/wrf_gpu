"""GPUWRF_SPEC_W_WORK_COPY (b-core B44 residual, default off): specified-domain w ring = WRF zero_grad_bdy (CPU).

WRF solve_em.F:1599 calls zero_grad_bdy(grid%w_2, 'w') on the small-step WORK w every acoustic substep for specified
domains. Truth: the UNCHANGED pristine share/module_bc.F zero_grad_bdy (binary32 ctypes adapter, zero_grad_oracle.py)
on synthetic fields (spec_zone 1/2/5) and on real PROD d01 stage operands (east ring-0 terrain step j=47, i=119).
The default helper copies the interior PHYSICAL W instead (deletion-sensitive: it must differ from WRF).
"""
import importlib.util
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.core import acoustic


def _finite_equal(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert np.isfinite(actual).all(), "nonfinite candidate"
    assert np.isfinite(desired).all(), "nonfinite reference"
    np.testing.assert_array_equal(actual, desired)


def _finite_close(actual, desired, **kwargs):
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert np.isfinite(actual).all(), "nonfinite candidate"
    assert np.isfinite(desired).all(), "nonfinite reference"
    np.testing.assert_allclose(actual, desired, **kwargs)


FC = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
PROD_D01 = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/fixtures/d01.npz")


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    import os
    if not Path(os.environ.get("FC", FC)).exists() and shutil.which("gfortran") is None:
        pytest.skip("gfortran unavailable for the pristine zero_grad_bdy oracle")
    spec = importlib.util.spec_from_file_location("b_core_zero_grad_oracle", Path(__file__).with_name("zero_grad_oracle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.build(tmp_path_factory.mktemp("zero_grad_oracle"))


def _operands(seed, nkw=6, ny=13, nx=17):
    rng = np.random.default_rng(seed)
    f32 = lambda a: np.asarray(a, np.float32)
    return dict(
        w_work=f32(rng.standard_normal((nkw, ny, nx))),
        w_save=f32(0.3 * rng.standard_normal((nkw, ny, nx))),
        mut=f32(9.0e4 + 2.0e3 * rng.random((ny, nx))),
        muts=f32(9.0e4 + 2.0e3 * rng.random((ny, nx))),
        c1f=f32(np.linspace(1.0, 0.0, nkw)), c2f=f32(np.linspace(0.0, 5.0e3, nkw)),
        msfty=f32(1.0 + 0.02 * rng.random((ny, nx))),
    )


def _port(monkeypatch, enabled, ops, spec_zone):
    monkeypatch.setattr(acoustic, "_SPEC_W_WORK_COPY", enabled)
    args = {k: jnp.asarray(v) for k, v in ops.items()}
    work = args.pop("w_work")
    return np.asarray(acoustic._specified_w_zero_grad_work(work, spec_zone=spec_zone, **args))


@pytest.mark.parametrize("spec_zone", [1, 2, 5])
def test_work_copy_bitwise_pristine_synthetic(oracle, monkeypatch, spec_zone):
    module, library = oracle
    ops = _operands(spec_zone)
    want = module.evaluate(library, ops["w_work"], spec_zone=spec_zone)
    assert not np.array_equal(want, ops["w_work"])
    _finite_equal(_port(monkeypatch, True, ops, spec_zone), want)
    # Default (physical-W projection) is NOT WRF's copy: the gate sees the difference on every ring.
    off = _port(monkeypatch, False, ops, spec_zone)
    assert np.isfinite(off).all(), "nonfinite OFF candidate"
    assert not np.array_equal(off, want)
    _finite_equal(off[:, spec_zone:-spec_zone, spec_zone:-spec_zone],
                                  want[:, spec_zone:-spec_zone, spec_zone:-spec_zone])  # interior untouched


def test_work_copy_bitwise_pristine_real_prod_d01(oracle, monkeypatch):
    if not PROD_D01.exists():
        pytest.skip("PROD d01 stage fixture unavailable")
    module, library = oracle
    with np.load(PROD_D01) as d:
        g = lambda k: np.asarray(d["state_" + k], np.float32)
        ops = dict(w_work=g("w"), w_save=g("w_save"), mut=g("mut"), muts=g("muts"),
                   c1f=g("c1f"), c2f=g("c2f"), msfty=g("msfty"))
        hgt = g("ht")
    want = module.evaluate(library, ops["w_work"], spec_zone=1)
    on = _port(monkeypatch, True, ops, 1)
    _finite_equal(on, want)
    off = _port(monkeypatch, False, ops, 1)
    assert np.isfinite(off).all(), "nonfinite OFF candidate"
    assert float(np.abs(off - want).max()) > 0.0
    assert float(hgt[47, 119] - hgt[47, 118]) > 10.0  # the B44 east face: ring-0 terrain step 0 -> 14 m
    assert on[0, 47, 119] == want[0, 47, 119] and off[0, 47, 119] != want[0, 47, 119]


def test_flag_is_cheap_keyed_and_default_off():
    import os
    from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
    assert ("gpuwrf.dynamics.core.acoustic", "_SPEC_W_WORK_COPY") in IMPORT_TIME_ENV_CONSTANTS
    if os.environ.get("GPUWRF_SPEC_W_WORK_COPY", "0") != "1":
        assert acoustic._SPEC_W_WORK_COPY is False

@pytest.mark.parametrize("side", ["candidate", "reference", "both"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_numeric_gate_rejects_nonfinite(side, bad):
    actual, desired = np.ones((2, 3)), np.ones((2, 3))
    if side in ("candidate", "both"):
        actual[0, 0] = bad
    if side in ("reference", "both"):
        desired[0, 0] = bad
    for compare in (_finite_equal, _finite_close):
        with pytest.raises(AssertionError, match="nonfinite"):
            compare(actual, desired)
