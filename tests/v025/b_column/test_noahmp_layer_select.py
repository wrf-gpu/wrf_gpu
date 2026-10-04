"""GPUWRF_NOAHMP_LAYER_SELECT (native REAL Noah-MP): select-based layer updates + unrolled ROSR12.

The flag only re-expresses operations: static-layer ``.at[k].set/add`` as an
elementwise select and the ROSR12 scans as unrolled recurrences with the scan
bodies' order. Gates: helpers and recurrences bitwise equal to the retained
operations; the NS01 pristine-WRF REAL4 snow gate passes with the flag and gives
exactly the flag-off errors; the flag is inert off the native REAL path.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_noahmp_layers as L


def _rng_layers(shape=(4, 5, 7), seed=0):
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32)


@pytest.mark.parametrize("k", [0, 1, 3, -1])
def test_set_add_layer_equal_at_ops(k, monkeypatch):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_SELECT", "1")
    x, v = _rng_layers(), _rng_layers((5, 7), 1)
    assert L.layers_enabled()
    for got, want in ((L.set_layer(x, k, v), x.at[k].set(v)), (L.add_layer(x, k, v), x.at[k].add(v)),
                      (L.set_layer(x, k, 0.0), x.at[k].set(0.0)),
                      (L.add_layer(x, k, jnp.float64(0.3)), x.at[k].add(jnp.float64(0.3)))):
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_rosr12_unrolled_equals_scan_eager():
    from gpuwrf.physics.noahmp import soil_thermo as S
    rows = tuple(_rng_layers((6, 5, 7), s) for s in range(4))
    carry = (_rng_layers((5, 7), 7), _rng_layers((5, 7), 8))
    with jax.disable_jit():  # op-by-op: identical arithmetic must give identical bits
        a = S._rosr12_fwd_scan(carry, rows)
        b = L.rosr12_forward(carry, rows)
        c = S._rosr12_bwd_scan(carry[0], (rows[0], rows[1]))
        d = L.rosr12_backward(carry[0], (rows[0], rows[1]))
    for x, y in zip(jax.tree.leaves((a, c)), jax.tree.leaves((b, d))):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _ns01():
    path = Path(__file__).resolve().parents[1] / "b_noahmp" / "snow_real_gate.py"
    spec = importlib.util.spec_from_file_location("ns01_snow_real_gate_ls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ns01_snow_gate_passes_with_flag_and_matches_flag_off(monkeypatch):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_SELECT", flag)
        jax.clear_caches()  # noahmp_snow is jitted; flags are read at trace time
        out[flag] = _ns01().run_gate()
    jax.clear_caches()
    assert out["1"]["all_pass"] and out["1"]["n"] == 14
    assert json.dumps(out["0"], sort_keys=True) == json.dumps(out["1"], sort_keys=True)


def test_flag_is_inert_off_the_native_real_path(monkeypatch):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "0")
    monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_SELECT", "1")
    assert not L.layers_enabled()
