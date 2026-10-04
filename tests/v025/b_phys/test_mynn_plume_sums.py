"""BP58: NUP plume sums folded into the plume kernel == plume arrays + XLA assembly.

Same F90:6363-6491 products; only the summation order over the 8 plumes (and
the limiter factor applied after instead of before the plume sum) differ, so
the bound is REAL rounding relative to each field's scale.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _batch(seed, n=37):
    """Real limiter-active WRF column tiled with perturbations; a third of the
    columns get negative surface buoyancy flux (inactive), so tiles mix states."""
    fixture = json.loads((HERE / "fixtures/limiter_active_wrf.json").read_text())
    rng = np.random.default_rng(seed)
    out = {}
    for key, value in fixture["inputs"].items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = arr
            continue
        tiled = np.repeat(arr, n, axis=0).astype(np.float32)
        if key in ("thl", "th", "thv"):
            tiled = tiled + rng.normal(0.0, 0.3, tiled.shape).astype(np.float32)
        if key in ("sqv", "sqw"):
            tiled = tiled * (1.0 + rng.normal(0.0, 0.02, tiled.shape)).astype(np.float32)
        if key in ("flt", "fltv"):
            tiled = tiled * rng.uniform(0.2, 2.0, tiled.shape).astype(np.float32)
            tiled[::3] = -np.abs(tiled[::3])
        out[key] = tiled
    return out


@pytest.mark.parametrize("seed", range(2))
def test_plume_sums_match_plume_arrays(monkeypatch, seed):
    from gpuwrf.kernels import phys_mynn_plume as P

    inputs = _batch(seed)
    monkeypatch.setenv("GPUWRF_MYNN_PLUME_SUMS", "0")
    ref = {k: np.asarray(v) for k, v in P.dmp_mf_columns_native(**inputs).items()}
    monkeypatch.setenv("GPUWRF_MYNN_PLUME_SUMS", "1")
    calls = []
    real = P.fused_plume_sums
    monkeypatch.setattr(P, "fused_plume_sums", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = {k: np.asarray(v) for k, v in P.dmp_mf_columns_native(**inputs).items()}
    assert calls == [1]                                   # the sums path ran (E114)
    assert set(got) == set(ref)
    np.testing.assert_array_equal(got["active"], ref["active"])
    assert 0 < ref["active"].sum() < len(ref["active"])   # mixed active/inactive tiles
    for key in ref:
        a, b = got[key], ref[key]
        assert a.dtype == b.dtype == np.float32 and a.shape == b.shape, key
        scale = max(float(np.max(np.abs(b))), 1e-30)
        assert float(np.max(np.abs(a - b))) <= 2e-6 * scale, (key, float(np.max(np.abs(a - b))), scale)


def test_plume_sums_pass_frozen_wrf_oracles(monkeypatch):
    from gpuwrf.kernels import phys_mynn_plume as P

    monkeypatch.setenv("GPUWRF_MYNN_PLUME_SUMS", "1")
    calls = []
    real = P.fused_plume_sums
    monkeypatch.setattr(P, "fused_plume_sums", lambda *a, **k: calls.append(1) or real(*a, **k))
    spec = importlib.util.spec_from_file_location("plume_native_tests", HERE / "test_mynn_native_plume.py")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.test_active_native_plume_matches_wrf(monkeypatch)
    tests.test_limiter_active_native_plume_matches_pristine_wrf()
    assert len(calls) >= 2
