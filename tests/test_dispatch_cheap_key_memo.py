"""CPU gates for the per-runtime AOT cheap-key memo (dispatch D02).

The memo must return exactly the full ``aot_cheap_key.cheap_key`` for every
call, skip the full computation on a repeated call signature, and miss (recompute)
whenever anything the full key reads per call changes.

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_dispatch_cheap_key_memo.py``
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import jax.numpy as jnp

from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime.domain_tree import _CheapKeyMemo
from gpuwrf.runtime.operational_mode import _advance_chunk_fori

pytestmark = pytest.mark.filterwarnings("ignore")


def _build_call():
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
    from gpuwrf.runtime.operational_mode import (
        OperationalNamelist,
        _initial_carry_for_run,
        build_clock_base,
    )

    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    state = State(
        **{
            f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
            for f, s in shapes.items()
        }
    )
    sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tend = Tendencies(
        **{
            k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
            for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
        }
    )
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=tend,
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=6,
        time_utc="2024-09-01_00:00:00",
    )
    return _initial_carry_for_run(state, namelist), namelist, build_clock_base(namelist)


@pytest.fixture(scope="module")
def call():
    return _build_call()


def _full(carry, namelist, start, clock_base, n_steps=1, cadence=1):
    return ck.cheap_key(
        _advance_chunk_fori,
        (carry, namelist, start, clock_base),
        {"n_steps": n_steps, "cadence": cadence},
        namelist,
    )


@pytest.fixture()
def spy(monkeypatch):
    calls = []
    real = ck.cheap_key

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ck, "cheap_key", counted)
    return calls


def test_memo_hit_equals_full_key_and_skips_recompute(call, spy):
    carry, namelist, clock_base = call
    start = jnp.asarray(2, jnp.int32)
    memo = _CheapKeyMemo()
    first = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    assert first is not None and len(spy) == 1
    for step in (3, 4, 5):
        again = memo.key(
            carry, namelist, jnp.asarray(step, jnp.int32), clock_base, n_steps=1, cadence=1
        )
        assert again == first
    assert len(spy) == 1, "memo hit must not recompute the full key"
    assert first == _full(carry, namelist, start, clock_base)


def test_memo_misses_on_leaf_dtype_change(call, spy):
    carry, namelist, clock_base = call
    memo = _CheapKeyMemo()
    k32 = memo.key(carry, namelist, jnp.asarray(2, jnp.int32), clock_base, n_steps=1, cadence=1)
    start64 = jnp.asarray(2, jnp.int64)
    k64 = memo.key(carry, namelist, start64, clock_base, n_steps=1, cadence=1)
    assert len(spy) == 2
    assert k64 == _full(carry, namelist, start64, clock_base)
    assert k64 != k32


def test_memo_misses_on_gpuwrf_env_change(call, spy, monkeypatch):
    carry, namelist, clock_base = call
    start = jnp.asarray(2, jnp.int32)
    memo = _CheapKeyMemo()
    base = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    monkeypatch.setenv("GPUWRF_DISPATCH_MEMO_TEST_KNOB", "1")
    changed = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    assert len(spy) == 2
    assert changed == _full(carry, namelist, start, clock_base)
    assert changed != base


def test_memo_keys_namelist_by_identity(call, spy):
    carry, namelist, clock_base = call
    start = jnp.asarray(2, jnp.int32)
    memo = _CheapKeyMemo()
    base = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    twin = dataclasses.replace(namelist)
    assert twin is not namelist
    again = memo.key(carry, twin, start, clock_base, n_steps=1, cadence=1)
    assert len(spy) == 2, "a different namelist object must recompute"
    assert again == _full(carry, twin, start, clock_base) == base


def test_memo_n_steps_is_signature_but_not_key(call, spy):
    carry, namelist, clock_base = call
    start = jnp.asarray(2, jnp.int32)
    memo = _CheapKeyMemo()
    one = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    three = memo.key(carry, namelist, start, clock_base, n_steps=3, cadence=1)
    assert len(spy) == 2
    assert one == three == _full(carry, namelist, start, clock_base, n_steps=3)


def test_memo_misses_on_import_time_module_constant_flip(call, spy, monkeypatch):
    module_path, attr = ck.IMPORT_TIME_ENV_CONSTANTS[0]
    import importlib

    module = importlib.import_module(module_path)
    carry, namelist, clock_base = call
    start = jnp.asarray(2, jnp.int32)
    memo = _CheapKeyMemo()
    base = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    value = getattr(module, attr)
    flipped = (not value) if isinstance(value, bool) else (value + 1 if isinstance(value, (int, float)) else f"{value}x")
    monkeypatch.setattr(module, attr, flipped)
    changed = memo.key(carry, namelist, start, clock_base, n_steps=1, cadence=1)
    assert len(spy) == 2, "a resolved module-constant flip must recompute"
    assert changed == _full(carry, namelist, start, clock_base)
    assert changed != base
