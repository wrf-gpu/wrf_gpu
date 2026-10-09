"""GPUWRF_W_SURFACE_RESET (b-core LL01 Finding 5, default off): WRF's end-of-step set_w_surface (CPU).

WRF solve_em.F:4818-4834 resets w(k=1) from the final u/v after every step (fill_w_flag=.false.).
Truth: the UNCHANGED pristine module_bc_em.F set_w_surface (binary32 ctypes adapter, w_surface_oracle.py)
on real 0227 CPU-WRF fields and on real-structured synthetic fields (msftx != msfty, both lateral modes);
CPU-WRF's own history W(k=0) equals it to <= 2 binary32 ulp. The step test proves the reset runs at the
end of _physics_boundary_step on the final u/v (deletion-sensitive) and touches face 0 only.
"""
import dataclasses
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.core import w_surface_reset as wsr


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


RUN_0227 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
FC = Path("<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    import os
    if not Path(os.environ.get("FC", FC)).exists() and shutil.which("gfortran") is None:
        pytest.skip("gfortran unavailable for the pristine set_w_surface oracle")
    import importlib.util
    spec = importlib.util.spec_from_file_location("b_core_w_surface_oracle", Path(__file__).with_name("w_surface_oracle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.build(tmp_path_factory.mktemp("wsurf_oracle"))


def _synthetic(seed, ny=37, nx=29, nz=6):
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ht = (1800.0 * np.exp(-((x - 11.0) ** 2 + (y - 20.0) ** 2) / 60.0) + 40.0 * rng.random((ny, nx))).astype(np.float32)
    u = (8.0 + 3.0 * rng.standard_normal((nz, ny, nx + 1))).astype(np.float32)
    v = (-5.0 + 3.0 * rng.standard_normal((nz, ny + 1, nx))).astype(np.float32)
    msftx = (1.0 + 0.02 * rng.random((ny, nx))).astype(np.float32)
    msfty = (1.0 + 0.02 * rng.random((ny, nx))).astype(np.float32)  # != msftx: a swap must be visible
    return u, v, ht, msftx, msfty, (1.6318, -0.8475, 0.2157)


def _port(u, v, ht, msftx, msfty, cf, dx, dy, periodic_x=False, periodic_y=False):
    return np.asarray(wsr.kinematic_surface_w(u, v, ht, msftx, msfty, *cf, dx=dx, dy=dy,
                                              periodic_x=periodic_x, periodic_y=periodic_y, dtype=np.float32))


@pytest.mark.parametrize("periodic_x", [False, True])
def test_kinematic_bitwise_pristine_synthetic(oracle, periodic_x):
    module, library = oracle
    u, v, ht, msftx, msfty, cf = _synthetic(3)
    want = module.evaluate(library, u, v, ht, msftx, msfty, cf, dx=1000.0, dy=1000.0, periodic_x=periodic_x)
    got = _port(u, v, ht, msftx, msfty, cf, 1000.0, 1000.0, periodic_x)
    _finite_equal(got, want)
    assert float(np.abs(want).max()) > 1.0
    # The fixture sees each WRF detail: map-factor pairing, edge clamp vs wrap, dx vs dy.
    assert not np.array_equal(_port(u, v, ht, msfty, msftx, cf, 1000.0, 1000.0, periodic_x), want)
    assert not np.array_equal(_port(u, v, ht, msftx, msfty, cf, 1000.0, 1000.0, not periodic_x), want)
    assert not np.array_equal(_port(u, v, ht, msftx, msfty, cf, 1000.0, 1001.0, periodic_x), want)


@pytest.mark.parametrize("periodic_x,periodic_y", [(False, True), (True, True)])
def test_kinematic_bitwise_pristine_periodic_y(oracle, periodic_x, periodic_y):
    # WRF set_w_surface periodic_y limits (jm1_limit = jds-1, jp1_limit = jde) = the periodic halo row.
    module, library = oracle
    u, v, ht, msftx, msfty, cf = _synthetic(5)
    want = module.evaluate(library, u, v, ht, msftx, msfty, cf, dx=1000.0, dy=1000.0,
                           periodic_x=periodic_x, periodic_y=periodic_y)
    _finite_equal(_port(u, v, ht, msftx, msfty, cf, 1000.0, 1000.0, periodic_x, periodic_y), want)
    assert not np.array_equal(_port(u, v, ht, msftx, msfty, cf, 1000.0, 1000.0, periodic_x, False), want)


@pytest.mark.parametrize("domain", ["d01", "d02", "d03"])
def test_kinematic_bitwise_pristine_real_0227(oracle, domain):
    from netCDF4 import Dataset
    path = RUN_0227 / f"wrfout_{domain}_2026-02-28_12:00:00"
    if not path.exists():
        pytest.skip("0227 CPU-WRF history unavailable")
    module, library = oracle
    with Dataset(path) as ds:
        get = lambda name: np.asarray(ds[name][0], np.float32)
        u, v, w, ht, mx, my = (get(n) for n in ("U", "V", "W", "HGT", "MAPFAC_MX", "MAPFAC_MY"))
        cf = [float(ds[n][0]) for n in ("CF1", "CF2", "CF3")]
        dx, dy = float(ds.DX), float(ds.DY)
    want = module.evaluate(library, u, v, ht, mx, my, cf, dx=dx, dy=dy)
    got = _port(u, v, ht, mx, my, cf, dx, dy)
    _finite_equal(got, want)
    # CPU-WRF itself ends the step with this value: its history W(k=0) agrees at its own build's binary32
    # round-off (measured max 9.5e-7 / rms <= 5.2e-8 m/s; the v0.3.1 GPU twin missed by 0.16 / 4e-3).
    assert np.isfinite(w[0]).all(), "nonfinite CPU-WRF history W"
    diff = got.astype(np.float64) - w[0]
    assert float(np.abs(diff).max()) <= 2.0e-6 and float(np.sqrt(np.mean(diff ** 2))) <= 1.0e-7
    assert float(np.abs(w[0]).max()) > 0.5


def _hill_case():
    from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
    from gpuwrf.runtime.operational_mode import initial_operational_carry
    setup = build_warm_bubble_setup(require_gpu=False)
    grid = setup.namelist.grid
    nx = grid.terrain_height.shape[-1]
    hill = 300.0 * np.exp(-((np.arange(nx) - nx / 2.0) / 6.0) ** 2)
    terrain = jnp.broadcast_to(jnp.asarray(hill, grid.terrain_height.dtype), grid.terrain_height.shape)
    namelist = dataclasses.replace(setup.namelist, grid=dataclasses.replace(grid, terrain_height=terrain))
    state = setup.state.replace(u=setup.state.u + jnp.asarray(6.0, setup.state.u.dtype))
    return initial_operational_carry(state), namelist


def _step(monkeypatch, enabled, carry, namelist):
    from gpuwrf.runtime import operational_mode as om
    monkeypatch.setattr(om, "_W_SURFACE_RESET", enabled)
    fn = jax.jit(lambda c, s: om._physics_boundary_step(c, namelist, s, run_radiation=False))  # fresh trace
    return fn(carry, jnp.asarray(0, jnp.int32)).state


def test_step_end_resets_surface_w_from_final_winds(monkeypatch):
    from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags
    carry, namelist = _hill_case()
    on = _step(monkeypatch, True, carry, namelist)
    off = _step(monkeypatch, False, carry, namelist)
    grid, metrics = namelist.grid, namelist.metrics
    want = wsr.kinematic_surface_w(on.u, on.v, grid.terrain_height, metrics.msftx, metrics.msfty,
                                   metrics.cf1, metrics.cf2, metrics.cf3,
                                   dx=float(grid.projection.dx_m), dy=float(grid.projection.dy_m),
                                   periodic_x=_acoustic_lateral_bc_flags(namelist)[0], dtype=on.w.dtype)
    for value in (on.w, off.w, want):
        assert np.isfinite(np.asarray(value)).all(), "nonfinite step W"
    assert float(jnp.abs(want).max()) > 1e-2
    # Compiled step vs eager formula: XLA:CPU may contract multiply-adds inside the fused step (E107).
    _finite_close(np.asarray(on.w[0]), np.asarray(want), rtol=1e-9, atol=1e-20)
    assert float(jnp.abs(off.w[0] - want).max()) > 1e-2  # deletion-sensitive
    # End of step: nothing else changes (u/v/theta/ph and every w face above the surface).
    for name in ("u", "v", "theta", "ph", "mu"):
        _finite_equal(np.asarray(getattr(on, name)), np.asarray(getattr(off, name)))
    _finite_equal(np.asarray(on.w[1:]), np.asarray(off.w[1:]))


def test_flag_is_cheap_keyed_and_default_off():
    from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
    assert ("gpuwrf.dynamics.core.w_surface_reset", "ENABLED") in IMPORT_TIME_ENV_CONSTANTS
    import os
    if os.environ.get("GPUWRF_W_SURFACE_RESET", "0") != "1":
        assert wsr.ENABLED is False


def test_reset_follows_the_boundary_pass_and_precedes_precision():
    # WRF order: RK stages -> microphysics -> lateral boundary updates -> set_w_surface (solve_em.F:4818).
    import inspect
    from gpuwrf.runtime import operational_mode as om
    source = inspect.getsource(om._physics_boundary_step_with_limiter_diagnostics)
    reset = source.index("_reset_surface_w(")
    assert source.rindex("apply_lateral_boundaries(") < reset < source.rindex("_enforce_operational_precision(")
    assert source[reset:].startswith("_reset_surface_w(\n            next_state, namelist.grid, namelist.metrics,")


def test_diagnostics_harness_mirrors_the_reset():
    # diagnostics/comprehensive_harness.py mirrors the operational step: same reset, same place.
    import inspect
    from gpuwrf.diagnostics import comprehensive_harness as ch
    source = inspect.getsource(ch.instrumented_physics_boundary_step)
    reset = source.index("_reset_surface_w(")
    assert "if _W_SURFACE_RESET:" in source
    assert source.rindex("apply_lateral_boundaries(") < reset < source.rindex("_enforce_operational_precision(")

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
