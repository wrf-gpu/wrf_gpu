"""BP61: lane-wise McICA KISS kernel == integer-exact jump-ahead stream, draw for draw."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import rad_mcica as M


def _pressures(seed, shape):
    rng = np.random.default_rng(seed)
    surface = rng.uniform(70000.0, 103000.0, shape[:-1] + (1,))
    return jnp.asarray(surface * np.linspace(1.0, 0.02, shape[-1]), jnp.float32)


@pytest.mark.parametrize("first,ng,shape", [(151, 140, (3, 37, 57)), (2, 224, (131, 47)), (151, 140, (1, 4, 6))])
def test_kiss_lanes_equal_jump_ahead_stream(monkeypatch, first, ng, shape):
    p = _pressures(first + ng, shape)
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "0")
    ref = np.asarray(jax.jit(lambda q: M._random_values(q, first=first, ng=ng, legacy_fp64=False))(p))
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "1")
    calls = []
    real = M.kiss_random_lanes
    monkeypatch.setattr(M, "kiss_random_lanes", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = np.asarray(jax.jit(lambda q: M._random_values(q, first=first, ng=ng, legacy_fp64=False))(p))
    assert calls == [1]
    assert got.dtype == ref.dtype == np.float32 and got.shape == ref.shape == shape + (ng,)
    np.testing.assert_array_equal(got, ref)


def test_masks_follow_the_kernel_stream(monkeypatch):
    p = _pressures(5, (2, 9, 57))
    cloud = jnp.asarray(np.random.default_rng(6).uniform(0, 1, (2, 9, 57)), jnp.float32)
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "0")
    ref = np.asarray(jax.jit(lambda q, c: M.lw_cloud_mask(q, c))(p, cloud))
    monkeypatch.setenv("GPUWRF_MCICA_KISS_KERNEL", "1")
    got = np.asarray(jax.jit(lambda q, c: M.lw_cloud_mask(q, c))(p, cloud))
    np.testing.assert_array_equal(got, ref)
    assert 0 < ref.mean() < 1
