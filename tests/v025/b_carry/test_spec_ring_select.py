"""GPUWRF_SPEC_RING_SELECT: masked-select ring pin must be bit-identical to the DUS loop."""
import numpy as np
import pytest
import jax.numpy as jnp

from gpuwrf.dynamics.core import acoustic


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


@pytest.mark.parametrize("shape", [(44, 70, 121), (45, 118, 267), (70, 120)])
@pytest.mark.parametrize("spec_zone", [1, 2, 4])
@pytest.mark.parametrize("tdtype", [np.float32, np.float64])
def test_ring_select_matches_loop(monkeypatch, shape, spec_zone, tdtype):
    rng = np.random.default_rng(spec_zone + len(shape))
    field = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
    target = jnp.asarray(rng.standard_normal(shape).astype(tdtype) * 1e3)
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "0")
    ref = np.asarray(acoustic._pin_spec_ring_wrf_owned(field, target, spec_zone=spec_zone))
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "1")
    got = np.asarray(acoustic._pin_spec_ring_wrf_owned(field, target, spec_zone=spec_zone))
    assert got.dtype == ref.dtype
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32))
    # deletion-sensitivity: the ring really changed something
    assert not np.array_equal(ref, np.asarray(field))


def test_uv_tan_select_matches_loop(monkeypatch):
    from types import SimpleNamespace
    from gpuwrf.kernels import dyn_acoustic_fp32 as k
    rng = np.random.default_rng(7)
    u = jnp.asarray(rng.standard_normal((44, 70, 121)).astype(np.float32))
    v = jnp.asarray(rng.standard_normal((44, 71, 120)).astype(np.float32))
    ut = jnp.asarray(rng.standard_normal(u.shape).astype(np.float32))
    vt = jnp.asarray(rng.standard_normal(v.shape).astype(np.float32))

    class S(SimpleNamespace):
        def replace(self, **kw):
            return S(**{**self.__dict__, **kw})
    for sz in (1, 2):
        before = S(u=u, v=v, u_work_bdy=None, v_work_bdy=None, u_spec_tan_target=ut, v_spec_tan_target=vt,
                   ru_m=jnp.zeros_like(u), rv_m=jnp.zeros_like(v))
        cfg = SimpleNamespace(spec_zone=sz)
        out = {}
        for flag in ("0", "1"):
            monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", flag)
            r = k._uv_boundaries(before, before, cfg)
            out[flag] = (np.asarray(r.u), np.asarray(r.v))
        for a, b in zip(out["0"], out["1"]):
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_pin_spec_ring_on_path_is_one_ring_scatter(monkeypatch):
    """review-ringsel advisory (a) + G07: ON is ONE in-place ring scatter (no field-sized select), OFF the 4-write chain."""
    import jax
    field = jnp.zeros((4, 9, 12), jnp.float32); target = jnp.ones((4, 9, 12), jnp.float32)
    def ops():
        jax.clear_caches()  # the env flag is read at trace time, not part of the jit cache key
        return _hlo_ops(jax.jit(lambda: acoustic._pin_spec_ring_wrf_owned(field, target, spec_zone=1)).lower().as_text())
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "1")
    on = ops()
    monkeypatch.setenv("GPUWRF_SPEC_RING_SELECT", "0")
    off = ops()
    names_on = [o for o, _ in on]
    assert names_on.count("scatter") == 1 and "dynamic_update_slice" not in names_on, on
    assert not [o for o, r in on if o == "select" and r >= 2], on
    names_off = [o for o, _ in off]
    assert names_off.count("scatter") + names_off.count("dynamic_update_slice") >= 4, off
