"""BP55 (S2): native KF keeps WRF REAL through kernel, carry and adapter.

module_cu_kfeta.F declares no DOUBLE: W0AVG, NCA, the held R*CUTEN/PRATEC
rates and the KF_eta_CPS running-mean recurrence are REAL in WRF. With
GPUWRF_KF_COLUMN_FP32=1 the port must not promote any of them to fp64.
"""
from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import cumulus_kf as kf

ROOT = Path(__file__).resolve().parents[3]
ARITH = {"add", "sub", "mul", "div", "max", "min", "pow", "integer_pow", "exp", "log", "floor",
         "neg", "sqrt", "rsqrt", "abs", "lt", "gt", "le", "ge", "eq", "ne"}


@pytest.fixture
def native_kf(monkeypatch):
    """Native KF path; the Pallas column runs in interpret mode on the CPU backend."""
    from gpuwrf.kernels import phys_kf_column as kc

    monkeypatch.setenv("GPUWRF_KF_COLUMN_FP32", "1")
    monkeypatch.setattr(kc, "kf_column", functools.partial(kc.kf_column, interpret=True))
    kf.kf_eta_para.clear_cache()   # the flag is read at trace time
    yield
    kf.kf_eta_para.clear_cache()


def _f64_arith(jaxpr, found):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in ARITH and any(
                getattr(v.aval, "dtype", None) == jnp.float64 for v in (*eqn.invars, *eqn.outvars)):
            found.append((eqn.primitive.name, tuple(eqn.outvars[0].aval.shape),
                          jax._src.source_info_util.summarize(eqn.source_info)))
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                if isinstance(sub, jax.extend.core.ClosedJaxpr):
                    _f64_arith(sub.jaxpr, found)
                elif isinstance(sub, jax.extend.core.Jaxpr):
                    _f64_arith(sub, found)
    return found


def test_update_w0avg_real_is_the_wrf_real_recurrence():
    rng = np.random.default_rng(55)
    kx = 44
    w = rng.normal(0.0, 2.0, kx + 1).astype(np.float32)
    w0avg = rng.normal(0.0, 1.0, kx).astype(np.float32)
    for stepcu in (1, 6):
        got = jax.jit(lambda a, b, s=stepcu: kf.update_w0avg(a, b, 54.0, stepcu=s, dtype=jnp.float32))(w0avg, w)
        assert got.dtype == jnp.float32
        # module_cu_kfeta.F KF_eta_CPS, all REAL: W0=0.5*(w(k)+w(k+1)); W0AVG=(W0AVG*(TST-1.)+W0)/TST
        tst = np.float32(stepcu * 2)
        w0 = np.float32(0.5) * (w[:kx] + w[1:])
        ref = (w0avg * (tst - np.float32(1.0)) + w0) / tst
        # XLA may contract the FMA / take the reciprocal; GPU f32 divide is div.full (<= 2 ulp, E84).
        np.testing.assert_array_max_ulp(np.asarray(got), ref, maxulp=2)


def test_native_step_kf_column_is_real_and_passes_frozen_wrf_oracle(native_kf):
    spec = importlib.util.spec_from_file_location("frozen_kf_oracle", ROOT / "tests/test_kf_cumulus_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    keys = ("T", "QV", "P", "DZ", "RHO", "W0AVG", "U", "V")
    for cid in range(1, 6):
        data = json.loads((ROOT / f"proofs/v060/savepoints/kf_case_{cid}.json").read_text())
        s, c = data["scalars"], data["columns"]
        cols = [jnp.asarray(c[key], jnp.float32) for key in keys]
        def step(*a):
            res = kf.step_kf_column(*a, s["DT"], s["DX"], nca=jnp.float32(-100.0), stepcu=5)
            return res.tendency.state_tendencies, res.diagnostics.cumulus, res.carry.cumulus

        tend, diag, carry = jax.jit(step)(*cols)
        dtypes = {name: value.dtype for name, value in (*tend.items(), *diag.items(), *carry.items())}
        assert {name for name, dtype in dtypes.items() if dtype != jnp.float32} == {"ishall"}, dtypes
        for key in oracle.TEND_FIELDS:
            ok, mad, mrd = oracle._check_field(diag[key.lower()], c[key])
            assert ok, (cid, key, mad, mrd)
        assert int(diag["ishall"]) == int(round(s["SHALL"]))
        assert abs(float(carry["nca"]) - s["NCA"]) <= 1e-6
        assert abs(float(diag["raincv"]) - s["RAINCV"]) <= max(oracle.RAINCV_REL * abs(s["RAINCV"]), oracle.RAINCV_ABS)


def test_native_kf_cadence_step_has_no_fp64_arithmetic(native_kf):
    from gpuwrf.diagnostics import census
    from gpuwrf.kernels.dyn_carry_fp32 import real_state
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("kf_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    grid = fixture._grid(ny=2, nx=2, nz=8)
    state = real_state(op._enforce_operational_precision(fixture._state(grid), force_fp64=True))
    from gpuwrf.coupling.scan_adapters import initial_kf_carry

    w0avg, nca = initial_kf_carry(state)
    assert w0avg.dtype == nca.dtype == jnp.float32
    rates = tuple(jnp.zeros_like(state.theta, dtype=jnp.float32) for _ in range(6)) + (
        jnp.zeros_like(state.t_skin, dtype=jnp.float32),)
    carry = initial_operational_carry(state).replace(
        cumulus_carry=(w0avg, nca), cumulus_tendencies=rates, census=census.initial_census())
    nml = SimpleNamespace(dt_s=54.0, cumulus_cadence_steps=6, cudt_minutes=0.0, grid=grid)
    step = lambda st, cy: op._kf_cadence_step(st, cy, nml, jnp.asarray(6))  # noqa: E731
    found = _f64_arith(jax.make_jaxpr(step)(state, carry).jaxpr, [])
    # The only fp64 arithmetic left is the RAINC accumulator add: State.rainc_acc is still
    # FP64-locked (ADR-007); ADR-038 makes the accumulators REAL (b-carry, CARRY_REAL_ALL).
    # With a REAL accumulator the KF step has no fp64 arithmetic at all.
    assert [(prim, shape) for prim, shape, _ in found] == [("add", state.rainc_acc.shape)], found
    line = int(found[0][2].split(":")[1])
    assert "rainc_acc" in Path(op.__file__).read_text().splitlines()[line - 1], found
    real_acc = state.replace(_cast=False, rainc_acc=state.rainc_acc.astype(jnp.float32))
    assert not _f64_arith(jax.make_jaxpr(step)(real_acc, carry).jaxpr, [])
    _, out = jax.eval_shape(step, state, carry)
    assert all(leaf.dtype == jnp.float32 for leaf in (*out.cumulus_carry, *out.cumulus_tendencies))
