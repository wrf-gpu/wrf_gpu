"""GPUWRF_MYNN_FLTV_WRF: WRF's MYNN derives flt/flqv/fltv inside its driver from HFX/QFX/TSK
(module_bl_mynnedmf.F:859-876); the port's surface producers built FLTV as
``(1+EP1*qv)*theta_flux + EP1*theta_air*qv_flux`` (+0.62 % max on real WN3 land columns, compile LL01).

Fixture ``fixtures/mynn_fltv_wrf.json``: 96 real WN3 columns (compile LL01 caller_seam_inputs) and the
outputs of the VERBATIM pristine WRF REAL4 statements compiled with WRF's own constants modules
(b-diff BD91/fltv_oracle.py) -- an independent oracle, not a self-compare (E98).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import gpuwrf.physics.noahmp_coupler as nc
import gpuwrf.physics.surface_layer as sl
from gpuwrf.physics.surface_constants import CP_D, EP1

FLAG = "GPUWRF_MYNN_FLTV_WRF"
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "mynn_fltv_wrf.json").read_text())


def _finite_or_fail(label, *arrays):
    """21:53Z rule: reject non-finite candidate/reference values before any max/ratio."""
    for i, a in enumerate(arrays):
        a = np.asarray(a)
        if not np.all(np.isfinite(a)):
            raise AssertionError(f"{label}: operand {i} has {int((~np.isfinite(a)).sum())} non-finite values")


def _max_ulp(got, ref):
    _finite_or_fail("ulp", got, ref)
    got = np.asarray(got, np.float32); ref = np.asarray(ref, np.float32)
    return float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)) / np.spacing(np.abs(ref)).astype(np.float64)))


def _inputs():
    f = {k: jnp.asarray(np.asarray(v, np.float32)) for k, v in FIXTURE["inputs"].items()}
    return f, {k: np.asarray(v, np.float32) for k, v in FIXTURE["wrf"].items()}


def test_helper_matches_pristine_wrf_real4_statements():
    f, wrf = _inputs()
    out = sl.mynn_driver_surface_fluxes(f["hfx"], f["qfx"], f["rho"], f["qv"], f["tsk"], f["p"])
    assert out.fltv.dtype == jnp.float32 and out.flt.dtype == jnp.float32
    for name in ("fltv", "flt", "flqv"):
        got = np.asarray(getattr(out, name))
        assert _max_ulp(got, wrf[name]) <= 1.0, name
        # Exact on CPU (96/96, BD91). A 1-ulp tolerance would hide WRF's REAL p608 = r_v/r_d - 1.
        # (the f64-rounded EP1 differs by 1 ulp and moves 1/96 columns; BD91 mutant M5).
        np.testing.assert_array_equal(got, wrf[name], err_msg=name)


def test_legacy_producer_formula_misses_wrf():
    """The fixture discriminates: the pre-flag producer expression is ~0.5 % off (deletion sensitivity)."""
    f, wrf = _inputs()
    qx = jnp.maximum(f["qv"], 0.0)
    theta_flux = f["hfx"] / (f["rho"] * (CP_D * (1.0 + 0.84 * qx)))
    qv_flux = f["qfx"] / f["rho"]
    theta_m = f["theta"] * (1.0 + jnp.float32(461.6 / 287.0) * f["qv"])
    legacy = np.asarray((1.0 + EP1 * qx) * theta_flux + EP1 * theta_m * qv_flux, np.float32)
    _finite_or_fail("legacy", legacy, wrf["fltv"])
    rel = np.abs(legacy.astype(np.float64) / wrf["fltv"] - 1.0)
    assert rel.max() > 3e-3 and rel.mean() > 2e-3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_comparison_rejects_non_finite(bad, side):
    a = np.ones(4, np.float32); b = np.ones(4, np.float32)
    (a if side == "candidate" else b)[2] = bad
    with pytest.raises(AssertionError, match="non-finite"):
        _max_ulp(a, b)


def _coupler_build():
    spec = importlib.util.spec_from_file_location("_noahmp_coupler_tests", Path(__file__).parents[2] / "test_noahmp_coupler.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not mod.HAVE_TABLES:
        pytest.skip("pristine WRF MPTABLE not available")
    return mod._build()


class _Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *args):
        out = self.fn(*args)
        self.calls.append((args, out))
        return out


def test_surface_layer_producer_dispatch(monkeypatch):
    state, *_ = _coupler_build()
    monkeypatch.delenv(FLAG, raising=False)
    off = sl.surface_layer_with_diagnostics(state)
    spy = _Spy(sl.mynn_driver_surface_fluxes); monkeypatch.setattr(sl, "mynn_driver_surface_fluxes", spy)
    sl.surface_layer_with_diagnostics(state)
    assert spy.calls == []  # OFF: helper not traced
    monkeypatch.setenv(FLAG, "1")
    on = sl.surface_layer_with_diagnostics(state)
    assert len(spy.calls) == 1
    (_, hfx_in, qfx_in, rho_in, _, tsk_in, p_in), ret = (None, *spy.calls[0][0]), spy.calls[0][1]
    for got, want in ((on.fluxes.fltv, ret.fltv), (on.fluxes.theta_flux, ret.flt), (on.fluxes.qv_flux, ret.flqv)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(hfx_in), np.asarray(on.hfx))  # the producer's own HFX/QFX feed it
    np.testing.assert_array_equal(np.asarray(rho_in), np.asarray(on.fluxes.rhosfc))
    np.testing.assert_array_equal(np.asarray(p_in), np.asarray(state.p)[..., 0])
    np.testing.assert_array_equal(np.asarray(tsk_in), np.asarray(state.t_skin).reshape(np.asarray(tsk_in).shape))
    assert not np.array_equal(np.asarray(on.fluxes.fltv), np.asarray(off.fluxes.fltv))
    for name in ("ustar", "tau_u", "tau_v", "rhosfc"):  # flag touches only the three MYNN handles
        np.testing.assert_array_equal(np.asarray(getattr(on.fluxes, name)), np.asarray(getattr(off.fluxes, name)))
    np.testing.assert_array_equal(np.asarray(on.hfx), np.asarray(off.hfx))


def test_noahmp_coupler_producer_dispatch(monkeypatch):
    state, land, static, rad, clock = _coupler_build()
    spy = _Spy(nc.mynn_driver_surface_fluxes); monkeypatch.setattr(nc, "mynn_driver_surface_fluxes", spy)
    monkeypatch.delenv(FLAG, raising=False)
    _, _, off = nc.noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=90.0)
    assert spy.calls == []
    monkeypatch.setenv(FLAG, "1")
    _, _, on = nc.noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=90.0)
    assert len(spy.calls) == 1  # the coupler's own call (the inner sfclay uses surface_layer's binding)
    args, ret = spy.calls[0]
    for got, want in ((on.fltv, ret.fltv), (on.theta_flux, ret.flt), (on.qv_flux, ret.flqv)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(args[5]).reshape(-1), np.asarray(state.p)[..., 0].reshape(-1).astype(np.asarray(args[5]).dtype))
    _finite_or_fail("coupler fltv", on.fltv, off.fltv)  # 21:53Z rule: before any ratio
    rel = np.abs(np.asarray(on.fltv, np.float64) / np.asarray(off.fltv, np.float64) - 1.0)
    assert 0.0 < rel.max() < 0.05
    for name in ("ustar", "tau_u", "tau_v", "rhosfc"):
        np.testing.assert_array_equal(np.asarray(getattr(on, name)), np.asarray(getattr(off, name)))
