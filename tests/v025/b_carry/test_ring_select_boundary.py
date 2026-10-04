"""GPUWRF_SPEC_RING_SELECT in boundary_apply: one select/gather must equal the original write loops bit-for-bit."""
from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp

from gpuwrf.coupling import boundary_apply as ba


def _hlo_ops(text):
    """(op, result rank) for every StableHLO op statement (attributes like #stablehlo.scatter<> excluded)."""
    import re
    out = []
    for line in text.splitlines():
        m = re.search(r'= "?stablehlo\.([a-z_]+)"?[ (]', line)
        if m:
            types = re.findall(r"tensor<([^>]*)>", line)
            rank = len(types[-1].split("x")) - 1 if types else 0
            out.append((m.group(1), rank))
    return out


def _both(monkeypatch, fn):
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "0")
    ref = np.asarray(fn())
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "1")
    got = np.asarray(fn())
    return ref, got


def _same(ref, got):
    assert ref.dtype == got.dtype and ref.shape == got.shape
    assert np.array_equal(ref.view(np.uint8), got.view(np.uint8))


@pytest.mark.parametrize("z,y,x,bw,pad,dtype", [
    (3, 9, 12, 1, 13, np.float32), (3, 9, 12, 5, 13, np.float64), (2, 7, 7, 4, 8, np.float32),
    (44, 117, 268, 5, 269, np.float32), (1, 117, 267, 5, 269, np.float64)])
def test_full_ring_target_gather(monkeypatch, z, y, x, bw, pad, dtype):
    rng = np.random.default_rng(z * 100 + bw)
    leaf = jnp.asarray(rng.standard_normal((4, bw, z + 1, pad)).astype(np.float64))
    ref, got = _both(monkeypatch, lambda: ba._full_ring_target_from_leaf(leaf, z, y, x, dtype))
    _same(ref, got)
    assert np.count_nonzero(ref) > 0 and np.count_nonzero(ref[:, bw:y - bw, bw:x - bw]) == 0


def test_full_ring_target_legacy_leaf_untouched(monkeypatch):
    leaf = jnp.asarray(np.random.default_rng(3).standard_normal((4, 3, 13)).astype(np.float32))
    ref, got = _both(monkeypatch, lambda: ba._full_ring_target_from_leaf(leaf, 3, 9, 12, np.float32))
    _same(ref, got)


@pytest.mark.parametrize("spec_zone", [1, 2])
def test_ph_inloop_select(monkeypatch, spec_zone):
    rng = np.random.default_rng(spec_zone)
    shp = (45, 70, 120)
    ph_work, leaf, save = (jnp.asarray(rng.standard_normal(shp).astype(np.float32)) for _ in range(3))
    cfg = SimpleNamespace(spec_zone=spec_zone)
    one = jnp.ones((70, 120), jnp.float32); c = jnp.ones((45,), jnp.float32)
    ref, got = _both(monkeypatch, lambda: ba.spec_bdyupdate_ph_inloop(ph_work, leaf, save, one, one, c, c, 18.0, cfg))
    _same(ref, got)
    assert not np.array_equal(ref, np.asarray(ph_work))


@pytest.mark.parametrize("spec_zone", [1, 2])
def test_ph_tendency_inloop_select(monkeypatch, spec_zone):
    rng = np.random.default_rng(10 + spec_zone)
    shp = (45, 117, 267)
    adv, before, tend, save = (jnp.asarray(rng.standard_normal(shp).astype(np.float32)) for _ in range(4))
    mu_tend = jnp.asarray(rng.standard_normal(shp[1:]).astype(np.float32))
    muts = jnp.asarray(9.0e4 + rng.standard_normal(shp[1:]).astype(np.float32))
    c1 = jnp.asarray(rng.uniform(0.1, 1.0, 45).astype(np.float32)); c2 = jnp.asarray(rng.uniform(0, 100, 45).astype(np.float32))
    cfg = SimpleNamespace(spec_zone=spec_zone)
    ref, got = _both(monkeypatch, lambda: ba.spec_bdyupdate_ph_tendency_inloop(
        adv, before, tend, save, mu_tend, muts, c1, c2, 6.0, cfg))
    _same(ref, got)
    assert not np.array_equal(ref, np.asarray(adv))


@pytest.mark.parametrize("which", ["inloop", "tendency"])
def test_ph_select_keeps_storage_dtype_under_boundary_fp32(monkeypatch, which):
    """b-diff 7f0ab3b9e: REAL math, caller's storage dtype on return -- the select must not skip that cast."""
    monkeypatch.setattr(ba, "_NATIVE_BOUNDARY_FP32", True)
    rng = np.random.default_rng(21)
    shp = (45, 70, 120)
    f64 = lambda: jnp.asarray(rng.standard_normal(shp))
    cfg = SimpleNamespace(spec_zone=1)
    one = jnp.ones(shp[1:], jnp.float64); c = jnp.ones((45,), jnp.float64)
    a, b_, t, s = f64(), f64(), f64(), f64()
    if which == "inloop":
        call = lambda: ba.spec_bdyupdate_ph_inloop(a, b_, s, one, one, c, c, 18.0, cfg)
    else:
        call = lambda: ba.spec_bdyupdate_ph_tendency_inloop(a, b_, t, s, one * 0.0, one * 9.0e4, c, c, 6.0, cfg)
    ref, got = _both(monkeypatch, call)
    assert ref.dtype == np.float64
    _same(ref, got)


def _prims(fn):
    """[(op, result rank)] of the lowered StableHLO (nested jits are inlined there)."""
    import jax
    jax.clear_caches()  # the env flag is read at trace time, not part of the jit cache key
    return _hlo_ops(jax.jit(lambda: fn()).lower().as_text())


def test_ring_select_on_path_is_taken(monkeypatch):
    """review-ringsel advisory (a): ON must really dispatch (spy) and emit no per-row scatter."""
    from gpuwrf.kernels import ring_select
    calls = {"select": 0, "gather": 0}
    orig_sel, orig_gat = ring_select.select_ring, ring_select.full_ring_target
    monkeypatch.setattr(ring_select, "select_ring", lambda *a, **k: (calls.__setitem__("select", calls["select"] + 1), orig_sel(*a, **k))[1])
    monkeypatch.setattr(ring_select, "full_ring_target", lambda *a, **k: (calls.__setitem__("gather", calls["gather"] + 1), orig_gat(*a, **k))[1])
    rng = np.random.default_rng(5)
    leaf = jnp.asarray(rng.standard_normal((4, 5, 4, 13)))
    ph = jnp.asarray(rng.standard_normal((4, 9, 12)).astype(np.float32))
    cfg = SimpleNamespace(spec_zone=1)
    one = jnp.ones((9, 12), jnp.float32); c = jnp.ones((4,), jnp.float32)
    ring_fn = lambda: ba._full_ring_target_from_leaf(leaf, 3, 9, 12, np.float32)
    ph_fn = lambda: ba.spec_bdyupdate_ph_inloop(ph, ph, ph, one, one, c, c, 18.0, cfg)
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "1")
    on_ring, on_ph = _prims(ring_fn), _prims(ph_fn)
    assert calls["gather"] >= 1 and calls["select"] >= 1
    ring_names, ph_names = [o for o, _ in on_ring], [o for o, _ in on_ph]
    assert ring_names.count("gather") == 1 and "scatter" not in ring_names, on_ring   # one gather, no write chain
    assert ph_names.count("scatter") == 1 and "dynamic_update_slice" not in ph_names, on_ph
    assert not [o for o, r in on_ph if o == "select" and r >= 2], on_ph               # no field-sized select
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "0")
    off = [o for o, _ in _prims(ring_fn) + _prims(ph_fn)]
    assert off.count("scatter") + off.count("dynamic_update_slice") >= 4, off
