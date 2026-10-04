"""CPU gates for the AOT call fast path (dispatch D05 arm F).

A loaded AOT callable checks the call contract once per call signature and
passes device-resident leaves to the executable without ``device_put``; results
and the contract's rejections are unchanged.

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_dispatch_aot_fastcall.py``
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import itertools

from gpuwrf.runtime import aot_executable as aotx
from tests._jax_cache_isolation import private_jax_cache

pytestmark = pytest.mark.filterwarnings("ignore")


_SALT = itertools.count(1)


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch, tmp_path):
    """Private persistent cache per test (as tests/test_aot_executable.py).

    A persistent-cache HIT makes CPU ``aotx.serialize`` drop the native objects
    (B12d), so every compiled program below is also salted to be unique."""
    with private_jax_cache(monkeypatch, tmp_path / "jit"):
        yield


def _loaded():
    salt = float(next(_SALT)) * 1e-30  # unique HLO per call: never a persistent-cache hit

    def fn(c):
        return {"y": c["a"] + c["b"] + salt, "z": c["a"] * 2.0}

    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,)), "dead": jnp.ones((3,))}
    compiled = jax.jit(fn).lower(carry).compile()
    blob, meta = aotx.serialize(compiled)
    return aotx.load(blob, meta), carry, compiled(carry)


def _equal(out, ref):
    return all(np.array_equal(np.asarray(out[k]), np.asarray(ref[k])) for k in ref)


def test_contract_checked_once_per_signature(monkeypatch):
    f, carry, ref = _loaded()
    calls = []
    real = aotx._check_call_contract
    monkeypatch.setattr(aotx, "_check_call_contract", lambda *a: (calls.append(1), real(*a))[1])
    for _ in range(4):
        assert _equal(f(carry), ref)
    assert len(calls) == 1, "a repeated call signature must not re-run the contract"


def test_changed_shape_after_cached_signature_still_rejected():
    f, carry, ref = _loaded()
    assert _equal(f(carry), ref)
    bad = dict(carry)
    bad["a"] = jnp.arange(5.0)
    with pytest.raises(RuntimeError, match="AOT input aval mismatch"):
        f(bad)


def test_committed_leaves_skip_device_put_others_transfer(monkeypatch):
    f, carry, ref = _loaded()
    dev = jax.devices()[0]
    committed = {k: jax.device_put(v, dev) for k, v in carry.items()}
    assert all(v._committed for v in committed.values())
    puts = []
    real = jax.device_put
    monkeypatch.setattr(aotx.jax, "device_put", lambda x, *a, **k: (puts.append(type(x)), real(x, *a, **k))[1])
    assert _equal(f(committed), ref)
    assert puts == [], "leaves committed to the executable device must be passed without device_put"
    host = {k: np.asarray(v) for k, v in carry.items()}
    assert _equal(f(host), ref)
    assert puts and all(t is np.ndarray for t in puts), "host leaves must still be transferred"


def test_uncommitted_resident_leaves_still_device_put(monkeypatch):
    f, carry, ref = _loaded()
    assert not any(v._committed for v in carry.values())
    puts = []
    real = jax.device_put
    monkeypatch.setattr(aotx.jax, "device_put", lambda x, *a, **k: (puts.append(type(x)), real(x, *a, **k))[1])
    assert _equal(f(carry), ref)
    assert len(puts) == 2, "uncommitted leaves keep the device_put path (2 kept leaves)"


def test_committed_on_requires_device_and_commitment():
    dev = jax.devices()[0]
    assert aotx._committed_on(jax.device_put(jnp.ones(3), dev), dev)
    assert not aotx._committed_on(jnp.ones(3), dev)
    assert not aotx._committed_on(np.ones(3), dev)


def test_changed_dtype_after_cached_signature_still_rejected():
    # Same shape AND same byte size: PJRT's size check cannot catch it, only the
    # contract can -> the memo key must carry the dtype (review-fastcall M2 mutant).
    f, carry, ref = _loaded()
    assert _equal(f(carry), ref)
    bad = dict(carry)
    same_size_int = jnp.int64 if carry["a"].dtype == jnp.float64 else jnp.int32
    bad["a"] = jax.lax.bitcast_convert_type(carry["a"], same_size_int)
    with pytest.raises(RuntimeError, match="AOT input aval mismatch"):
        f(bad)


def test_committed_on_rejects_non_default_memory_kinds():
    dev = jax.devices()[0]
    kinds = [m.kind for m in dev.addressable_memories() if m.kind != dev.default_memory().kind]
    if not kinds:
        pytest.skip("backend exposes no non-default memory kind")
    for kind in kinds:
        x = jax.device_put(jnp.ones(3), jax.sharding.SingleDeviceSharding(dev, memory_kind=kind))
        assert x._committed and not aotx._committed_on(x, dev), kind


def test_committed_on_rejects_other_device():
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import jax, jax.numpy as jnp\n"
        "from gpuwrf.runtime import aot_executable as aotx\n"
        "d0, d1 = jax.devices()[:2]\n"
        "x = jax.device_put(jnp.ones(3), d1)\n"
        "assert x._committed and aotx._committed_on(x, d1)\n"
        "assert not aotx._committed_on(x, d0)\n"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        "PYTHONPATH": os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p),
    }
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=120)


def _put_spy(monkeypatch):
    puts = []
    real = jax.device_put
    monkeypatch.setattr(aotx.jax, "device_put", lambda x, *a, **k: (puts.append(x), real(x, *a, **k))[1])
    return puts


def test_same_uncommitted_array_object_is_put_once(monkeypatch):
    f, carry, ref = _loaded()
    puts = _put_spy(monkeypatch)
    for _ in range(3):
        assert _equal(f(carry), ref)
    assert len(puts) == 2, "two kept uncommitted leaves: one device_put each, then reused"


def test_new_uncommitted_object_is_put_again(monkeypatch):
    f, carry, ref = _loaded()
    puts = _put_spy(monkeypatch)
    assert _equal(f(carry), ref)
    fresh = {k: v + 0.0 for k, v in carry.items()}
    assert _equal(f(fresh), ref)
    assert len(puts) == 4, "different array objects must be transferred again"


def test_host_leaves_are_put_every_call(monkeypatch):
    f, carry, ref = _loaded()
    host = {k: np.asarray(v) for k, v in carry.items()}
    puts = _put_spy(monkeypatch)
    assert _equal(f(host), ref)
    assert _equal(f(host), ref)
    assert len(puts) == 4 and all(isinstance(x, np.ndarray) for x in puts)


def test_committed_leaf_drops_cached_copy(monkeypatch):
    f, carry, ref = _loaded()
    dev = jax.devices()[0]
    committed = {k: jax.device_put(v, dev) for k, v in carry.items()}
    puts = _put_spy(monkeypatch)
    assert _equal(f(carry), ref)  # caches both uncommitted kept leaves
    assert len(f.put_cache) == 2
    assert _equal(f(committed), ref)  # committed leaves: entries dropped
    assert len(f.put_cache) == 0
    assert _equal(f(carry), ref)  # same objects again: must be re-put
    assert len(puts) == 4
