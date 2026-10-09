"""GPUWRF_W_DAMP_STAGE (b-core BD85, default off): WRF ``w_damp`` once per RK stage on the stage ``rw_tend`` (CPU).

WRF rk_tendency (module_em.F:738) calls w_damp(rw_tend, ..., u_2, v_2, grid%ww, grid%w_2, mut, c1f, c2f, rdnw, ...,
grid%dt) once per RK stage; the native fp32 acoustic never applied it (ledger BD85). Truth: the UNCHANGED pristine
dyn_em/module_big_step_utilities_em.F w_damp (binary32 ctypes adapter, w_damp_oracle.py) on synthetic fields with
activated columns (incl. signed zeros of w) and on REAL PROD d01 stage operands (rw_tend, w, mut, metrics); the
fixture stores no omega, so its real coupled vertical-momentum field x0.02 serves as a real-structured omega that puts
the vertical CFL into the active range (real 0227/PROD CFL is <= 0.25: unscaled operands would be vacuous, E102). The dispatch test traces
one step and proves the helper runs once per RK stage with WRF's operands, and that the legacy in-acoustic copy is off.
"""
import importlib.util
import inspect
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.core import advance_w as aw

FC = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
PROD_D01 = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/fixtures/d01.npz")


def _finite_equal(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert np.isfinite(actual).all(), "nonfinite candidate"
    assert np.isfinite(desired).all(), "nonfinite reference"
    np.testing.assert_array_equal(actual, desired)


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    import os
    if not Path(os.environ.get("FC", FC)).exists() and shutil.which("gfortran") is None:
        pytest.skip("gfortran unavailable for the pristine w_damp oracle")
    spec = importlib.util.spec_from_file_location("b_core_w_damp_oracle", Path(__file__).with_name("w_damp_oracle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.build(tmp_path_factory.mktemp("w_damp_oracle"))


def _synthetic(seed, nz=12, ny=9, nx=11):
    rng = np.random.default_rng(seed)
    f32 = lambda a: np.asarray(a, np.float32)
    c1f = f32(np.linspace(1.0, 0.0, nz + 1)); c2f = f32(np.linspace(0.0, 4.0e3, nz + 1))
    rdnw = f32(1.0 / np.linspace(-0.02, -0.06, nz))
    mut = f32(9.0e4 + 3.0e3 * rng.random((ny, nx)))
    mass = c1f[:, None, None] * mut[None] + c2f[:, None, None]
    cfl_target = rng.uniform(0.0, 3.0, (nz + 1, ny, nx))      # many faces above w_beta = 1
    ww = f32(cfl_target * mass / (np.abs(np.concatenate([rdnw, rdnw[-1:]]))[:, None, None] * 18.0)
             * rng.choice([-1.0, 1.0], (nz + 1, ny, nx)))
    w = f32(rng.standard_normal((nz + 1, ny, nx)))
    w[3, 2, 2], w[4, 3, 3] = np.float32(0.0), np.float32(-0.0)   # Fortran SIGN(1.,+-0.) = +-1
    rw = f32(50.0 * rng.standard_normal((nz + 1, ny, nx)))
    return dict(rw_tend=rw, ww=ww, w=w, mut=mut, c1f=c1f, c2f=c2f, rdnw=rdnw)


def _port(ops, dt, **kw):
    return np.asarray(aw.w_damp_rw_tend_wrf(jnp.asarray(ops["rw_tend"]), ww=jnp.asarray(ops["ww"]),
                                            w=jnp.asarray(ops["w"]), mut=jnp.asarray(ops["mut"]),
                                            c1f=jnp.asarray(ops["c1f"]), c2f=jnp.asarray(ops["c2f"]),
                                            rdnw=jnp.asarray(ops["rdnw"]), dt=dt, **kw))


def test_w_damp_bitwise_pristine_synthetic(oracle):
    module, library = oracle
    ops = _synthetic(1)
    want, max_cfl = module.evaluate(library, ops["rw_tend"], ops["ww"], ops["w"], ops["mut"],
                                    ops["c1f"], ops["c2f"], ops["rdnw"], dt=18.0)
    assert max_cfl > 2.0 and not np.array_equal(want, ops["rw_tend"])   # damping active (non-vacuous)
    got = _port(ops, 18.0)
    _finite_equal(got, want)
    assert got[3, 2, 2] != ops["rw_tend"][3, 2, 2] or got[4, 3, 3] != ops["rw_tend"][4, 3, 3]
    # The fixture sees each WRF detail: full dt (not the small step), the activation threshold, signed zeros.
    assert not np.array_equal(_port(ops, 18.0 / 4.0), want)
    assert not np.array_equal(_port(ops, 18.0, w_damp_on=0.0), want)
    legacy = np.asarray(aw.w_damp_vertical_cfl(jnp.asarray(ops["rw_tend"]), ww=jnp.asarray(ops["ww"]),
                                               w=jnp.asarray(ops["w"]), mut=jnp.asarray(ops["mut"]),
                                               c1f=jnp.asarray(ops["c1f"]), c2f=jnp.asarray(ops["c2f"]),
                                               rdnw=jnp.asarray(ops["rdnw"]), dt=18.0))
    assert not np.array_equal(legacy, want)   # jnp.sign(+-0)=0 drops WRF's +-1 at signed-zero faces


def test_w_damp_bitwise_pristine_real_prod_d01(oracle):
    if not PROD_D01.exists():
        pytest.skip("PROD d01 stage fixture unavailable")
    module, library = oracle
    with np.load(PROD_D01) as d:
        g = lambda k: np.asarray(d["state_" + k], np.float32)
        ops = dict(rw_tend=g("rw_tend_pg_buoy"), ww=g("w") * np.float32(0.02), w=g("w_save"), mut=g("mut"),
                   c1f=g("c1f"), c2f=g("c2f"), rdnw=g("rdnw"))
    want, max_cfl = module.evaluate(library, ops["rw_tend"], ops["ww"], ops["w"], ops["mut"],
                                    ops["c1f"], ops["c2f"], ops["rdnw"], dt=54.0)
    assert max_cfl > 1.0 and not np.array_equal(want, ops["rw_tend"])
    _finite_equal(_port(ops, 54.0), want)


def test_stage_dispatch_once_per_rk_stage_with_wrf_operands(monkeypatch):
    from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
    from gpuwrf.runtime import operational_mode as om
    setup = build_warm_bubble_setup(require_gpu=False)
    assert int(setup.namelist.w_damping) == 1
    carry = om.initial_operational_carry(setup.state)
    calls = []

    def spy(rw_tend, **kw):
        calls.append(kw["dt"])
        return aw.w_damp_rw_tend_wrf(rw_tend, **kw)

    trace = lambda: jax.make_jaxpr(lambda c, k: om._physics_boundary_step(c, setup.namelist, k, run_radiation=False))(
        carry, jnp.asarray(0, jnp.int32))
    monkeypatch.setattr(om, "_w_damp_rw_tend_wrf", spy)
    monkeypatch.setattr(om, "_W_DAMP_STAGE", False)
    trace()
    assert calls == []
    monkeypatch.setattr(om, "_W_DAMP_STAGE", True)
    trace()
    assert calls == [float(setup.namelist.dt_s)] * int(setup.namelist.rk_order)   # once per stage, FULL dt


def test_call_site_wiring_and_no_double_application():
    from gpuwrf.runtime import operational_mode as om
    source = inspect.getsource(om._acoustic_core_state_from_prep)
    call = source.index("_w_damp_rw_tend_wrf(")
    assert source[call:call + 160].replace("\n", " ").split("dt=")[0].count("ww=prep.ww_save, w=state.w, mut=prep.mut") == 1
    assert "w_damping=0 if _W_DAMP_STAGE else int(namelist.w_damping)" in inspect.getsource(om)


def test_flag_is_cheap_keyed_and_default_off():
    import os
    from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
    assert ("gpuwrf.dynamics.core.advance_w", "W_DAMP_STAGE") in IMPORT_TIME_ENV_CONSTANTS
    if os.environ.get("GPUWRF_W_DAMP_STAGE", "0") != "1":
        assert aw.W_DAMP_STAGE is False


@pytest.mark.parametrize("side", ["candidate", "reference", "both"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_numeric_gate_rejects_nonfinite(side, bad):
    actual, desired = np.ones((2, 3)), np.ones((2, 3))
    if side in ("candidate", "both"):
        actual[0, 0] = bad
    if side in ("reference", "both"):
        desired[0, 0] = bad
    with pytest.raises(AssertionError, match="nonfinite"):
        _finite_equal(actual, desired)
