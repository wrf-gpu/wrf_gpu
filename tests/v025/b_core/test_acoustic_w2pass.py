"""A1 two-pass advance_w (GPUWRF_ACOUSTIC_W2PASS) and A2 blocked mass kernel (GPUWRF_ACOUSTIC_MASS_BLOCK), CPU interpret.

Same per-element expressions; the merged level loops may contract/round differently, so the
gate is a few fp32 ulps of the field scale.  Fidelity vs pristine WRF (D7 oracle, both flags):
tests/v025/b_core/acoustic_w2pass_evidence.json.
"""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState
from gpuwrf.kernels import dyn_acoustic_fp32 as da

FIXTURES = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1/fixtures")
EPS32 = float(np.finfo(np.float32).eps)


def _fixture(domain):
    if not (FIXTURES / f"{domain}.npz").exists():
        pytest.skip("acoustic fixtures unavailable")
    data = np.load(FIXTURES / f"{domain}.npz")
    meta = json.loads((FIXTURES / f"{domain}.json").read_text())
    state = AcousticCoreState(**{key[6:]: jnp.asarray(data[key], jnp.float32)
                                 for key in data.files if key.startswith("state_")})
    if state.mu_work is None:
        state = state.replace(mu_work=jnp.asarray(data["state_muts"] - data["state_mut"], jnp.float32))
    return state, AcousticCoreConfig(**meta["config"])


def _w_phase(monkeypatch, flag, state, coef, cfg):
    monkeypatch.setenv("GPUWRF_ACOUSTIC_W2PASS", flag)
    jax.clear_caches()
    fn = lambda s: da._w_phase(s, coef, cfg, interpret=True)  # noqa: E731
    outs = [e for e in jax.make_jaxpr(fn)(state).jaxpr.eqns if e.primitive.name == "pallas_call"]
    result = jax.jit(fn)(state)
    jax.clear_caches()
    return result, len(outs[0].outvars)


def test_two_pass_w_phase_matches_fourteen_pass(monkeypatch):
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("JAX_PALLAS_USE_MOSAIC_GPU", "false")
    state, cfg = _fixture("d01")
    assert cfg.specified or cfg.nested  # ring path exercised
    coef = jax.jit(lambda s: da.calc_coef_fp32(s, cfg, interpret=True))(state)
    ref, n_ref = _w_phase(monkeypatch, "0", state, coef, cfg)
    new, n_new = _w_phase(monkeypatch, "1", state, coef, cfg)
    assert (n_ref, n_new) == (7, 4)  # wdwn/rw_eff/wtop scratch planes gone
    nz, ny, nx = state.theta.shape
    ring = np.ones((ny, nx), bool)
    ring[1:-1, 1:-1] = False
    for name in ("w", "ph", "t_2ave"):
        a = np.asarray(getattr(ref, name), np.float64)
        b = np.asarray(getattr(new, name), np.float64)
        assert np.isfinite(b).all(), name
        assert np.max(np.abs(b - a)) <= 16 * EPS32 * np.max(np.abs(a)), name
        # specified/nested ring cells keep their inputs exactly (masked final stores)
        np.testing.assert_array_equal(b[:, ring], np.asarray(getattr(state, name), np.float64)[:, ring])
    np.testing.assert_array_equal(np.asarray(new.t_2ave), np.asarray(ref.t_2ave))


def test_mass_block_substep_is_bitwise(monkeypatch):
    """A2: GPUWRF_ACOUSTIC_MASS_BLOCK walks the mass-kernel levels in trips of 4 (loads before
    stores); same expressions and order -> the whole substep is bitwise (GPU: BC49 n_diff 0)."""
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("JAX_PALLAS_USE_MOSAIC_GPU", "false")
    monkeypatch.setenv("GPUWRF_ACOUSTIC_W2PASS", "0")
    state, cfg = _fixture("d01")
    coef = jax.jit(lambda s: da.calc_coef_fp32(s, cfg, interpret=True))(state)
    calls = []
    original = da._level_blocks

    def spy(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(da, "_level_blocks", spy)
    outs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("GPUWRF_ACOUSTIC_MASS_BLOCK", flag)
        jax.clear_caches()
        out, guards = jax.jit(lambda s: da.acoustic_substep_fp32(
            s, coefficients=coef, cfg=cfg, interpret=True, return_guard_events=True))(state)
        outs[flag] = ({k: np.asarray(v) for k, v in out.to_dict().items() if v is not None}, np.asarray(guards))
    jax.clear_caches()
    nz = state.theta.shape[0]
    assert calls == [nz, nz]  # blocked reduce + blocked level loop ran (flag 1 only)
    np.testing.assert_array_equal(outs["0"][1], outs["1"][1])
    assert outs["0"][0].keys() == outs["1"][0].keys()
    for name, value in outs["0"][0].items():
        np.testing.assert_array_equal(outs["1"][0][name], value, err_msg=name)


@pytest.mark.parametrize("n", range(1, 14))
def test_level_blocks_tail_matches_sequential(n):
    """_level_blocks visits levels 0..n-1 in order including the n % 4 tail (PROD/WN3 nz = 44 never
    runs the tail); exact, order-sensitive integer recurrence (review-s2small advisory)."""
    load = lambda k: jnp.asarray(k, jnp.int32) * 7 + 3  # noqa: E731
    step = lambda k, d, c: (c * 31 + d + 1000 * (jnp.asarray(k, jnp.int32) + 1)) % 1000003  # noqa: E731
    blocked = int(jax.jit(lambda: da._level_blocks(n, load, step, jnp.int32(0)))())
    seq = 0
    for k in range(n):
        seq = (seq * 31 + (k * 7 + 3) + 1000 * (k + 1)) % 1000003
    assert blocked == seq


def test_uv_masked_substep_is_bitwise(monkeypatch):
    """GPUWRF_ACOUSTIC_UV_MASKED: the uv kernel loads p for the bottom/top/middle dpn branches only on
    the lanes that select them (same values there) -> substep bitwise; the traced kernel must differ."""
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("JAX_PALLAS_USE_MOSAIC_GPU", "false")
    state, cfg = _fixture("d01")
    coef = jax.jit(lambda s: da.calc_coef_fp32(s, cfg, interpret=True))(state)
    outs, texts = {}, {}
    for flag in ("0", "1"):
        monkeypatch.setenv("GPUWRF_ACOUSTIC_UV_MASKED", flag)
        jax.clear_caches()
        fn = lambda s: da.acoustic_substep_fp32(s, coefficients=coef, cfg=cfg, interpret=True,  # noqa: E731
                                               return_guard_events=True)
        texts[flag] = str(jax.make_jaxpr(fn)(state))
        out, guards = jax.jit(fn)(state)
        outs[flag] = ({k: np.asarray(v) for k, v in out.to_dict().items() if v is not None}, np.asarray(guards))
    jax.clear_caches()
    assert texts["0"] != texts["1"]  # masked path dispatched
    np.testing.assert_array_equal(outs["0"][1], outs["1"][1])
    for name, value in outs["0"][0].items():
        np.testing.assert_array_equal(outs["1"][0][name], value, err_msg=name)


def test_split_w_phase_matches_two_pass(monkeypatch):
    """A1c (GPUWRF_ACOUSTIC_W2PASS=3): k-parallel pointwise kernels (t_2ave/rhs, explicit w update)
    + a column kernel with only the recurrences; same expressions as the 2-pass kernel
    (GPU: bitwise target, BC52). CPU interpret may contract differently -> few-ulp gate."""
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("JAX_PALLAS_USE_MOSAIC_GPU", "false")
    state, cfg = _fixture("d01")
    assert cfg.specified or cfg.nested
    coef = jax.jit(lambda s: da.calc_coef_fp32(s, cfg, interpret=True))(state)
    ref, n_ref = _w_phase(monkeypatch, "1", state, coef, cfg)
    monkeypatch.setenv("GPUWRF_ACOUSTIC_W2PASS", "3")
    jax.clear_caches()
    fn = lambda s: da._w_phase(s, coef, cfg, interpret=True)  # noqa: E731
    calls = [e for e in jax.make_jaxpr(fn)(state).jaxpr.eqns if e.primitive.name == "pallas_call"]
    new = jax.jit(fn)(state)
    jax.clear_caches()
    assert len(calls) == 3  # P1, P2, R
    nz, ny, nx = state.theta.shape
    ring = np.ones((ny, nx), bool)
    ring[1:-1, 1:-1] = False
    for name in ("w", "ph", "t_2ave"):
        a = np.asarray(getattr(ref, name), np.float64)
        b = np.asarray(getattr(new, name), np.float64)
        assert np.isfinite(b).all(), name
        assert np.max(np.abs(b - a)) <= 16 * EPS32 * np.max(np.abs(a)), name
        np.testing.assert_array_equal(b[:, ring], np.asarray(getattr(state, name), np.float64)[:, ring])


def _recur_inputs(nz, ncol, seed=0):
    from gpuwrf.kernels import fused_vertical_implicit as vi
    rng = np.random.default_rng(seed)
    z = nz + 1
    f = lambda *s: jnp.asarray(rng.uniform(0.5, 1.5, s).astype(np.float32))  # noqa: E731
    phb = np.cumsum(rng.uniform(900.0, 1100.0, (z, ncol)), 0).astype(np.float32)
    ins = [f(z, ncol), f(z, ncol) * 0.1, f(z, ncol), f(z, ncol) * 0.1, f(z, ncol), f(z, ncol), f(z, ncol),
           jnp.asarray(phb), f(z, ncol), f(z, ncol), f(z), f(z), f(ncol), f(ncol), f(ncol)]
    s = np.zeros(vi._N_SLOTS, np.float32)
    s[vi._DTS], s[vi._G], s[vi._EPSSM] = 2.0, 9.81, 0.1
    s[vi._HDEPTH], s[vi._HALF_PI], s[vi._DAMPMAG] = 15000.0, np.pi / 2, 3.6
    return ins + [jnp.asarray(s)]


@pytest.mark.parametrize("block", (2, 3))
def test_w_recur_blocked_kernel_is_bitwise(block):
    """GPUWRF_ACOUSTIC_W_RECUR_SL kernel = _advance_w_kernel_recur in trips of `block` levels (loads
    before stores); same expressions and level order -> bitwise for damp3/keep in all combinations,
    incl. the (nz - 1) % block tail (nz = 44)."""
    from jax.experimental import pallas as pl
    from gpuwrf.kernels import fused_vertical_implicit as vi
    nz, ncol = 44, 2 * vi.TX
    ins = _recur_inputs(nz, ncol)

    def run(fn, damp3, keep, **kw):
        def body(*refs):
            kp = (pl.program_id(0) * vi.TX + jnp.arange(vi.TX)) % 7 != 0 if keep else None
            fn(*refs, nz=nz, damp3=damp3, keep=kp, **kw)
        return pl.pallas_call(body, grid=(ncol // vi.TX,), interpret=True,
                              out_shape=(jax.ShapeDtypeStruct((nz + 1, ncol), jnp.float32),) * 2)(*ins)

    for damp3 in (False, True):
        for keep in (False, True):
            ref = run(vi._advance_w_kernel_recur, damp3, keep)
            new = run(vi._advance_w_kernel_recur_sl, damp3, keep, block=block)
            for a, b in zip(ref, new):
                assert np.isfinite(np.asarray(b)).all()
                np.testing.assert_array_equal(np.asarray(b), np.asarray(a), err_msg=f"{damp3=} {keep=}")


def test_w_recur_blocked_substep_is_bitwise(monkeypatch):
    """GPUWRF_ACOUSTIC_W_RECUR_SL=1 on the split w phase (W2PASS=3, LW11 acoustic set): the whole
    substep is bitwise on the real d01 fixture (CPU interpret; d02 nested ring: same, v031/A1)."""
    from gpuwrf.kernels import fused_vertical_implicit as vi
    assert jax.devices()[0].platform == "cpu"
    for key, value in dict(JAX_PALLAS_USE_MOSAIC_GPU="false", GPUWRF_ACOUSTIC_W2PASS="3",
                           GPUWRF_ACOUSTIC_MASS_BLOCK="1", GPUWRF_ACOUSTIC_UV_MASKED="1").items():
        monkeypatch.setenv(key, value)
    state, cfg = _fixture("d01")
    assert cfg.damp_opt == 3 and state.w_save is not None  # damp3 branch exercised
    coef = jax.jit(lambda s: da.calc_coef_fp32(s, cfg, interpret=True))(state)
    calls = []
    original = vi._advance_w_kernel_recur_sl

    def spy(*args, **kwargs):
        calls.append(kwargs["nz"])
        return original(*args, **kwargs)

    monkeypatch.setattr(vi, "_advance_w_kernel_recur_sl", spy)
    outs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("GPUWRF_ACOUSTIC_W_RECUR_SL", flag)
        jax.clear_caches()
        out, guards = jax.jit(lambda s: da.acoustic_substep_fp32(
            s, coefficients=coef, cfg=cfg, interpret=True, return_guard_events=True))(state)
        outs[flag] = ({k: np.asarray(v) for k, v in out.to_dict().items() if v is not None}, np.asarray(guards))
    jax.clear_caches()
    assert calls == [state.theta.shape[0]]  # blocked recurrence dispatched (flag 1 only)
    np.testing.assert_array_equal(outs["0"][1], outs["1"][1])
    assert outs["0"][0].keys() == outs["1"][0].keys()
    for name, value in outs["0"][0].items():
        np.testing.assert_array_equal(outs["1"][0][name], value, err_msg=name)
