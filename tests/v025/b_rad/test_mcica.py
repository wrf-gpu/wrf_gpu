"""Independent uint32 recurrence, reference masks and frozen WRF oracles."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import os

from gpuwrf.kernels.rad_mcica import kiss_jump_stream, lw_cloud_mask, sw_cloud_mask
from gpuwrf.kernels import rad_mcica
from gpuwrf.physics import rrtmg_lw, rrtmg_sw

if os.environ.get('B_RAD_REQUIRE_GPU') == '1':
    assert jax.devices()[0].platform == 'gpu', 'GPU RNG suite cannot silently use CPU'


def _serial_words(seeds, steps):
    s1, s2, s3, s4 = [s.copy() for s in seeds]
    out = []
    for _ in range(steps):
        s1 = s1 * np.uint32(69069) + np.uint32(1327217885)
        s2 ^= s2 << np.uint32(13)
        s2 ^= s2 >> np.uint32(17)
        s2 ^= s2 << np.uint32(5)
        s3 = np.uint32(18000) * (s3 & np.uint32(65535)) + (s3 >> np.uint32(16))
        s4 = np.uint32(30903) * (s4 & np.uint32(65535)) + (s4 >> np.uint32(16))
        out.append(s1 + s2 + (s3 << np.uint32(16)) + s4)
    return np.stack(out)


@pytest.mark.parametrize("first,count", [(1, 50), (2, 112 * 45), (151, 140 * 57)])
def test_stream_integer_exact(first, count):
    rng = np.random.default_rng(20261001)
    edges = np.array([0, 1, 65535, 65536, 999999999], dtype=np.uint32)
    seeds = tuple(np.concatenate((edges, rng.integers(0, 10**9, 27, dtype=np.uint32))) for _ in range(4))
    expected = _serial_words(seeds, first + count - 1)[first - 1:]
    fn = jax.jit(lambda *s: kiss_jump_stream(s, first=first, count=count))
    actual = fn(*(jnp.asarray(s) for s in seeds))
    np.testing.assert_array_equal(actual, expected)
    assert "stablehlo.while" not in fn.lower(*(jnp.asarray(s) for s in seeds)).as_text()


@pytest.mark.parametrize("shape", [(7, 17), (2, 3, 45), (4, 57)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_reference_masks_bitwise(shape, dtype, monkeypatch):
    # This comparison targets the serial legacy conversion, independent of
    # the product flags selected for the full-solver oracle tests below.
    monkeypatch.setattr(rrtmg_lw, '_MCICA_JUMPAHEAD', False)
    monkeypatch.setattr(rrtmg_sw, '_MCICA_JUMPAHEAD', False)
    rng = np.random.default_rng(107)
    p = jnp.asarray(rng.uniform(100, 101500, shape), dtype=dtype)
    cf = jnp.asarray(rng.choice([0, 1e-25, 1e-20, 0.2, 0.5, 0.999999, 1], size=shape), dtype=dtype)
    lw_ref = jax.jit(lambda p, c: rrtmg_lw._lw_mcica_random_cloud_mask(p, c, output_dtype=dtype))(p, cf)
    lw_new = jax.jit(lambda p, c: lw_cloud_mask(p, c, output_dtype=dtype, legacy_fp64=True))(p, cf)
    np.testing.assert_array_equal(lw_new, lw_ref)
    gm = jnp.asarray([[g < cnt for g in range(12)] for cnt in rrtmg_sw._SW_GPOINT_COUNTS], dtype=jnp.float64)
    sw_ref = jax.jit(rrtmg_sw._mcica_random_overlap_mask)(p, cf, gm)
    sw_new = jax.jit(lambda p, c, g: sw_cloud_mask(p, c, g, legacy_fp64=True))(p, cf, gm)
    np.testing.assert_array_equal(sw_new, sw_ref)


@pytest.fixture
def fresh_solver_caches():
    # The monkeypatched helper is captured at trace time. Clear both on entry
    # and teardown so this test cannot alter later tests' flag-off executables.
    functions = (rrtmg_lw.solve_rrtmg_lw_column, rrtmg_sw.solve_rrtmg_sw_column)
    for fn in functions:
        fn.clear_cache()
    yield
    for fn in functions:
        fn.clear_cache()


@pytest.mark.parametrize("legacy", [False, True])
def test_wrf_oracles_with_jumpahead(monkeypatch, tmp_path, legacy, fresh_solver_caches):
    # Exercise the actual product dispatch. Override only the kernel's legacy
    # conversion setting to cover both modes in one test process.
    from gpuwrf.validation.rrtmg_intermediate_oracles import run_intermediate_validation
    from gpuwrf.validation.tier1_rrtmg import run_tier1_lw, run_tier1_sw

    monkeypatch.setattr(rrtmg_lw, "_MCICA_JUMPAHEAD", True)
    monkeypatch.setattr(rrtmg_sw, "_MCICA_JUMPAHEAD", True)
    monkeypatch.setattr(rad_mcica, "lw_cloud_mask",
                        lambda p, c, **kw: lw_cloud_mask(p, c, legacy_fp64=legacy, **kw))
    monkeypatch.setattr(rad_mcica, "sw_cloud_mask",
                        lambda p, c, g: sw_cloud_mask(p, c, g, legacy_fp64=legacy))
    assert run_tier1_lw(tmp_path / "lw.json")["pass"]
    assert run_tier1_sw(tmp_path / "sw.json")["pass"]
    if jax.devices()[0].platform == 'cpu':
        # The intermediate runner deliberately uses CPU. GPU arms validate
        # full heating/flux above; the CPU arms retain every intermediate gate.
        record = run_intermediate_validation(tmp_path / "intermediate.json", tmp_path / "bands.json")
        assert record["lw_cldprmc"]["pass"]
        assert record["lw_rtrnmc"]["pass"]
        assert record["pass"]


def test_native_mask_has_no_fp64_hlo():
    p = jnp.full((2, 17), 60123.45, dtype=jnp.float32)
    c = jnp.full(p.shape, 0.5, dtype=jnp.float32)
    gm = jnp.ones((14, 12), dtype=jnp.float32)
    for fn, args in ((lw_cloud_mask, (p, c)), (sw_cloud_mask, (p, c, gm))):
        text = jax.jit(fn).lower(*args).as_text()
        assert "f64" not in text


def test_flag_off_retains_serial_scan(monkeypatch):
    p = jnp.full((2, 17), 60123.45, dtype=jnp.float32)
    c = jnp.full(p.shape, 0.5, dtype=jnp.float32)
    gm = jnp.ones((14, 12), dtype=jnp.float32)
    for module, fn, args in (
        (rrtmg_lw, rrtmg_lw._lw_mcica_random_cloud_mask, (p, c)),
        (rrtmg_sw, rrtmg_sw._mcica_random_overlap_mask, (p, c, gm)),
    ):
        wrapped = jax.jit(fn)
        wrapped.clear_cache()
        monkeypatch.setattr(module, '_MCICA_JUMPAHEAD', False)
        assert 'stablehlo.while' in wrapped.lower(*args).as_text()
        monkeypatch.setattr(module, '_MCICA_JUMPAHEAD', True)
        wrapped.clear_cache()
        assert 'stablehlo.while' not in wrapped.lower(*args).as_text()
        wrapped.clear_cache()


def test_gas_leaf_states_support_lower_compile():
    from gpuwrf.validation.tier1_rrtmg import load_lw_fixture_state, load_sw_fixture_state

    sw, _ = load_sw_fixture_state()
    lw, _ = load_lw_fixture_state()
    gases = dict(co2_vmr=400e-6, n2o_vmr=0.3e-6, ch4_vmr=1.8e-6)
    sw = sw.replace(**gases)
    lw = lw.replace(**gases, cfc11_vmr=0.2e-9, cfc12_vmr=0.5e-9)
    for state in (sw, lw):
        # .lower constructs ArgInfo leaves, which must not pass through array
        # conversion in the pytree unflattener (the SW full-call failure).
        result = jax.jit(lambda s: s).lower(state).compile()(state)
        for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(result), strict=True):
            np.testing.assert_array_equal(actual, expected)
