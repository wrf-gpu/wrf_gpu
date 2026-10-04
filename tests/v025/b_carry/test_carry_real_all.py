"""GPUWRF_CARRY_REAL_ALL helpers (CPU): flag gating, REAL seeding/restores, exact PBL removal."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import dyn_carry_fp32 as dc


class _FakeState:
    __slots__ = ("theta", "ustar", "rain_acc", "lu_index", "flag", "absent")

    def __init__(self, **values):
        for name in self.__slots__:
            object.__setattr__(self, name, values.get(name))

    def replace(self, *, _cast=True, **updates):
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return _FakeState(**values)


def _state():
    return _FakeState(
        theta=jnp.ones((2, 3), jnp.float32), ustar=jnp.ones((3,), jnp.float64),
        rain_acc=jnp.ones((3,), jnp.float64), lu_index=jnp.ones((3,), jnp.float64),
        flag=jnp.ones((3,), jnp.int32), absent=None,
    )


@pytest.fixture
def carry_on(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_CARRY_FP32", "1")
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", "1")


def test_real_all_needs_the_carry_flag(monkeypatch):
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", "1")
    monkeypatch.delenv("GPUWRF_DYN_CARRY_FP32", raising=False)
    assert not dc.real_all_enabled() and dc.real_dtype(jnp.float64) == jnp.float64
    monkeypatch.setenv("GPUWRF_DYN_CARRY_FP32", "1")
    assert dc.real_all_enabled() and dc.real_dtype(jnp.float64) == jnp.float32


def test_real_state_off_keeps_surface_leaves_wide(monkeypatch):
    monkeypatch.setenv("GPUWRF_DYN_CARRY_FP32", "1")
    monkeypatch.delenv("GPUWRF_CARRY_REAL_ALL", raising=False)
    out = dc.real_state(_state())
    assert out.ustar.dtype == jnp.float64 and out.rain_acc.dtype == jnp.float64


def test_real_state_all_narrows_every_float_leaf(carry_on):
    out = dc.real_state(_state())
    assert {n: str(getattr(out, n).dtype) for n in ("theta", "ustar", "rain_acc", "lu_index", "flag")} == {
        "theta": "float32", "ustar": "float32", "rain_acc": "float32", "lu_index": "float32", "flag": "int32"}
    assert out.absent is None


def test_like_casts_to_reference_and_keeps_structure():
    ref = (jnp.zeros(2, jnp.float32), (jnp.zeros(2, jnp.float32), jnp.zeros(2, jnp.int32)))
    val = (jnp.ones(2, jnp.float64), (jnp.ones(2, jnp.float64), jnp.ones(2, jnp.int32)))
    out = dc.like(val, ref)
    assert jax.tree.structure(out) == jax.tree.structure(ref)
    assert [x.dtype for x in jax.tree.leaves(out)] == [x.dtype for x in jax.tree.leaves(ref)]


def test_real_carry_seeds_only_the_listed_leaves(carry_on):
    class _Carry:
        def __init__(self, **v):
            self.__dict__.update(v)

        def replace(self, **u):
            return _Carry(**{**self.__dict__, **u})

    carry = _Carry(noahmp_rad=(jnp.ones(2, jnp.float64),) * 3, cumulus_carry=None,
                   cumulus_tendencies=(jnp.ones(2, jnp.float64),) * 7,
                   radiation_diagnostics=None, rthraten=jnp.ones(2, jnp.float64))
    out = dc.real_carry(carry)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves((out.noahmp_rad, out.cumulus_tendencies)))
    assert out.rthraten.dtype == jnp.float64 and out.cumulus_carry is None


def test_pbl_removal_real_form_equals_wide_form():
    """entry + (next - after) in REAL == round32(next - (after - entry)) in f64.

    theta: the post-PBL change (KF heating) keeps next and after within a factor 2,
    so next - after is exact (Sterbenz) and both forms round the same exact value once.
    u/v: nothing changes them after the PBL call (next == after) -> both give entry.
    """
    rng = np.random.default_rng(7)
    entry = rng.uniform(250.0, 500.0, 200_000).astype(np.float32)
    after = (entry + rng.normal(0.0, 0.5, entry.size)).astype(np.float32)
    nxt = (after + rng.normal(0.0, 0.05, entry.size)).astype(np.float32)
    wide = (nxt.astype(np.float64) - (after.astype(np.float64) - entry.astype(np.float64))).astype(np.float32)
    real = entry + (nxt - after)
    assert np.array_equal(wide.view(np.uint32), real.view(np.uint32))
    u_entry = rng.normal(0.0, 10.0, entry.size).astype(np.float32)
    u_after = (u_entry + rng.normal(0.0, 1.0, entry.size)).astype(np.float32)
    wide_u = (u_after.astype(np.float64) - (u_after.astype(np.float64) - u_entry.astype(np.float64))).astype(np.float32)
    real_u = u_entry + (u_after - u_after)
    assert np.array_equal(wide_u.view(np.uint32), u_entry.view(np.uint32))
    assert np.array_equal(real_u.view(np.uint32), u_entry.view(np.uint32))
