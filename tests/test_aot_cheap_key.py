"""CPU IDENTITY-PROOF for the cheap-key AOT manifest -- the falsification engine.

The cheap-key manifest (``gpuwrf.runtime.aot_cheap_key``) lets the eager Step-C
loop locate a serialized ``_advance_chunk_fori`` executable by a metadata-only
hash WITHOUT lowering (the ~30-54 min wall the AOT preview otherwise still paid).

THE LOAD-BEARING RISK = KEY COMPLETENESS: if ``cheap_key`` misses ANY input baked
into the lowered HLO, two distinct executables can share a key -> wrong blob loads
-> SILENT WRONG RESULT. This test PROVES, on CPU, with NO GPU, that the map
``cheap_key -> hlo_sha256`` is a function (INJECTIVE: one key never maps to two
HLOs) across a determinant matrix, and that a deliberately-incomplete key is
CAUGHT (collision detected). It lowers the REAL ``_advance_chunk_fori`` (feasible
on CPU in ~5 s; we compare HLO *text*, never execute -- the XLA:CPU machine-type
SIGILL warning is benign because we never run the program).

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_aot_cheap_key.py``

Each lowering is ~190 MB of HLO text and a few seconds, so the matrix is kept
focused and every distinct config is lowered AT MOST ONCE (module-scoped cache).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gpuwrf.runtime import aot_cheap_key as ck

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Fixture builders (real GridSpec + namelist + carry; CPU-constructible).
# --------------------------------------------------------------------------- #
def _build_call(nl_kwargs=None, time_utc="2024-09-01_00:00:00"):
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
    fields = {
        f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
        for f, s in shapes.items()
    }
    state = State(**fields)
    sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tend = Tendencies(
        **{
            k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
            for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
        }
    )
    kw = dict(
        grid=grid,
        tendencies=tend,
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=6,
        time_utc=time_utc,
    )
    if nl_kwargs:
        kw.update(nl_kwargs)
    namelist = OperationalNamelist(**kw)
    carry = _initial_carry_for_run(state, namelist)
    clock_base = build_clock_base(namelist)
    return carry, namelist, clock_base


def _real_hlo_sha256(carry, namelist, clock_base, *, n_steps=1, cadence=1):
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    lowered = _advance_chunk_fori.lower(
        carry,
        namelist,
        jnp.asarray(1, dtype=jnp.int32),
        clock_base,
        n_steps=int(n_steps),
        cadence=int(cadence),
    )
    return hashlib.sha256(lowered.as_text().encode("utf-8")).hexdigest()


def _cheap_key(carry, namelist, clock_base, *, n_steps=1, cadence=1):
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    return ck.cheap_key(
        _advance_chunk_fori,
        (carry, namelist, jnp.asarray(1, dtype=jnp.int32), clock_base),
        {"n_steps": int(n_steps), "cadence": int(cadence)},
        namelist,
    )


# A trace-time GPUWRF_* env knob is read INSIDE the jitted body, but JAX caches a
# lowering in-process by its (avals, statics) key -- which (by design, the bug we
# fix) does NOT include the env. So a second in-process ``.lower()`` after an env
# change returns the STALE cached HLO. To observe the env's true effect on the
# HLO (and on the cheap_key), the env-axis cells must lower in a FRESH process.
def _lower_in_subprocess(env_overrides: dict[str, str], nl_kwargs=None, call_kwargs=None):
    """Return (cheap_key, hlo_sha256) computed in a FRESH process under ``env``.

    Fresh process => empty JAX lowering cache => the env knobs are read at trace
    time and their true effect on both the HLO and the cheap_key is observed."""
    import json
    import subprocess
    import sys
    import textwrap

    payload = {
        "env": env_overrides,
        "nl_kwargs": nl_kwargs or {},
        "call_kwargs": call_kwargs or {},
    }
    code = textwrap.dedent(
        """
        import os, sys, json, hashlib
        import numpy as np, jax, jax.numpy as jnp
        from gpuwrf.contracts.grid import GridSpec
        from gpuwrf.contracts.precision import DEFAULT_DTYPES
        from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
        from gpuwrf.runtime.operational_mode import (
            OperationalNamelist, _initial_carry_for_run, build_clock_base,
            _advance_chunk_fori,
        )
        from gpuwrf.runtime import aot_cheap_key as ck
        spec = json.loads(sys.argv[1])
        nl_kwargs = dict(spec["nl_kwargs"])
        call_kwargs = dict(spec["call_kwargs"])
        time_utc = nl_kwargs.pop("time_utc", "2024-09-01_00:00:00")
        grid = GridSpec.canary_3km_template()
        shapes = _state_field_shapes(grid)
        fields = {f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
                  for f, s in shapes.items()}
        state = State(**fields)
        sk = {"p":"p_total","ph":"ph_total","mu":"mu_total"}
        tend = Tendencies(**{k: jnp.zeros(shapes[sk.get(k,k)],
                             dtype=DEFAULT_DTYPES.dtype_for(k))
                             for k in ("u","v","w","theta","qv","p","ph","mu")})
        kw = dict(grid=grid, tendencies=tend, metrics=grid.metrics,
                  dt_s=10.0, acoustic_substeps=6, time_utc=time_utc)
        kw.update(nl_kwargs)
        nl = OperationalNamelist(**kw)
        carry = _initial_carry_for_run(state, nl)
        cb = build_clock_base(nl)
        n_steps = int(call_kwargs.get("n_steps", 1))
        cadence = int(call_kwargs.get("cadence", 1))
        key = ck.cheap_key(_advance_chunk_fori,
                           (carry, nl, jnp.asarray(1, jnp.int32), cb),
                           {"n_steps": n_steps, "cadence": cadence}, nl)
        # An env-BLIND key (component 5 / global_trace_env_hash omitted) for the
        # collision-detection proof.
        incomplete = ck.canonical_digest((
            ck.KEY_SCHEMA,
            ck.version_fingerprint_hash(),
            ck.fn_identity_hash(_advance_chunk_fori),
            ck.static_config_hash(nl),
            ck.carry_aval_hash(
                (carry, nl, jnp.asarray(1, jnp.int32), cb),
                {"n_steps": n_steps, "cadence": cadence}),
        ))
        low = _advance_chunk_fori.lower(carry, nl, jnp.asarray(1, jnp.int32), cb,
                                        n_steps=n_steps, cadence=cadence)
        hlo = hashlib.sha256(low.as_text().encode("utf-8")).hexdigest()
        print(json.dumps({"key": key, "hlo": hlo, "incomplete_key": incomplete}))
        """
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    for k, v in env_overrides.items():
        env[k] = v
    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return out["key"], out["hlo"], out["incomplete_key"]


# The determinant matrix. Each cell is (name, build_kwargs_or_env, call_kwargs).
# `env` cells set a GPUWRF_* knob for the lowering (read at trace time); the
# fixture restores the env afterwards. Cells are chosen so SOME share an HLO
# (date, n_steps, an inert option under a zero state) and SOME differ (dt_s,
# epssm, the env knobs) -- both classes are needed to exercise injectivity AND
# the no-over-fragment direction.
_MATRIX = [
    ("base", {}, {}, {}),
    # date varies -> key + HLO MUST be IDENTICAL to base (#114 date-blindness).
    ("date2", {"time_utc": "2025-12-25_06:00:00"}, {}, {}),
    # n_steps / cadence vary -> traced int32 -> key + HLO IDENTICAL to base.
    ("nsteps9", {}, {"n_steps": 9, "cadence": 3}, {}),
    # dt_s changes a baked scalar -> key + HLO MUST DIFFER.
    ("dt18", {"dt_s": 18.0}, {}, {}),
    # epssm changes a baked acoustic coefficient -> key + HLO MUST DIFFER.
    ("epssm", {"epssm": 0.3}, {}, {}),
    # acoustic_substeps changes the loop bound -> key + HLO MUST DIFFER.
    ("subs10", {"acoustic_substeps": 10}, {}, {}),
    # NOTE: the ENV-AXIS cells (GPUWRF_MOIST_CQW / GPUWRF_ACOUSTIC_UNROLL) are NOT
    # in this in-process matrix -- JAX caches a lowering by avals+statics (which by
    # design excludes the env), so a second in-process lower after an env change
    # returns the STALE HLO. They are exercised via FRESH SUBPROCESSES in
    # test_env_axis_* below (which is also the more faithful trace-time scenario).
]


@pytest.fixture(scope="module")
def matrix_results():
    """Lower every matrix cell once; return [(name, cheap_key, hlo_sha256), ...]."""
    results = []
    for name, nl_kwargs, call_kwargs, _env in _MATRIX:
        nl_kwargs = dict(nl_kwargs)
        time_utc = nl_kwargs.pop("time_utc", "2024-09-01_00:00:00")
        carry, namelist, clock_base = _build_call(nl_kwargs, time_utc=time_utc)
        key = _cheap_key(carry, namelist, clock_base, **call_kwargs)
        hlo = _real_hlo_sha256(carry, namelist, clock_base, **call_kwargs)
        assert key is not None, f"cheap_key returned None for {name}"
        assert hlo is not None, f"lowering produced no HLO for {name}"
        results.append((name, key, hlo))
    return results


# --------------------------------------------------------------------------- #
# (1) INJECTIVITY -- catches a MISSED determinant = the silent-wrong bug.
# --------------------------------------------------------------------------- #
def test_cheap_key_is_injective_over_hlo(matrix_results):
    """One cheap_key must never map to two different HLOs (no silent wrong load)."""
    by_key: dict[str, set[str]] = {}
    names_by_key: dict[str, list[str]] = {}
    for name, key, hlo in matrix_results:
        by_key.setdefault(key, set()).add(hlo)
        names_by_key.setdefault(key, []).append(name)
    collisions = {
        key: (names_by_key[key], sorted(h[:12] for h in hlos))
        for key, hlos in by_key.items()
        if len(hlos) > 1
    }
    assert not collisions, (
        "CHEAP_KEY COLLISION (a determinant is missing from the key): "
        f"{collisions}"
    )


# --------------------------------------------------------------------------- #
# (2) date / n_steps / cadence are NOT determinants -> key AND HLO invariant.
# --------------------------------------------------------------------------- #
def test_date_and_nsteps_are_key_and_hlo_invariant(matrix_results):
    """base, date2, nsteps9 must all share ONE key AND ONE HLO."""
    by_name = {name: (key, hlo) for name, key, hlo in matrix_results}
    base_key, base_hlo = by_name["base"]
    for name in ("date2", "nsteps9"):
        key, hlo = by_name[name]
        assert hlo == base_hlo, (
            f"{name}: HLO unexpectedly differs from base -- the matrix assumption "
            "that this axis is HLO-inert is wrong"
        )
        assert key == base_key, (
            f"{name}: cheap_key OVER-FRAGMENTS on an HLO-inert axis "
            "(would silently return the 30-min cost as 'warm but slow')"
        )


# --------------------------------------------------------------------------- #
# (3) HLO-changing config + env knobs -> key MUST differ (completeness).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["dt18", "epssm", "subs10"])
def test_hlo_changing_config_axes_change_the_key(matrix_results, name):
    """Each namelist cell that changes the HLO MUST change the cheap_key."""
    by_name = {n: (key, hlo) for n, key, hlo in matrix_results}
    base_key, base_hlo = by_name["base"]
    key, hlo = by_name[name]
    assert hlo != base_hlo, (
        f"{name}: precondition -- this cell was expected to change the HLO but did "
        "not; the test cannot prove the key responds to it"
    )
    assert key != base_key, (
        f"{name}: HLO changed but cheap_key did NOT -> SILENT WRONG LOAD risk "
        "(a determinant is missing from the key)"
    )


# --------------------------------------------------------------------------- #
# (3b) ENV-AXIS (the load-bearing completeness cells) via FRESH SUBPROCESSES.
#      These trace-time GPUWRF_* knobs branch the HLO but live OUTSIDE the
#      namelist/carry; the cheap_key MUST respond to them (else silent wrong load).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "env",
    [
        {"GPUWRF_MOIST_CQW": "0"},
        {"GPUWRF_ACOUSTIC_UNROLL": "4"},
    ],
)
def test_env_axis_changes_both_hlo_and_key(env):
    """A trace-time env knob that changes the HLO MUST change the cheap_key."""
    base_key, base_hlo, _ = _lower_in_subprocess({})
    env_key, env_hlo, _ = _lower_in_subprocess(env)
    assert env_hlo != base_hlo, (
        f"precondition: {env} must change the HLO (trace-time env determinant)"
    )
    assert env_key != base_key, (
        f"{env}: HLO changed but cheap_key did NOT -> SILENT WRONG LOAD risk "
        "(component 5 / global_trace_env_hash is missing this knob)"
    )


# --------------------------------------------------------------------------- #
# (4) Cross-process stability -- proves _StaticHolder is content-hashed, not id().
# --------------------------------------------------------------------------- #
def test_cheap_key_is_process_stable():
    """The same config must yield the same cheap_key in TWO fresh processes.

    If any component used ``id()``/``hash(str)`` (PYTHONHASHSEED-salted), a fresh
    run would compute a different key and miss every prewarmed blob (100% fallback,
    zero warm win). We compute the key under two DIFFERENT PYTHONHASHSEEDs in
    separate processes and require an exact match -- the strongest salt check.
    """
    key_a, _, _ = _lower_in_subprocess({"PYTHONHASHSEED": "0"})
    key_b, _, _ = _lower_in_subprocess({"PYTHONHASHSEED": "12345"})
    assert key_a and key_b
    assert key_a == key_b, (
        "cheap_key is NOT process-stable (a component uses id()/salted hash): "
        f"seed0={key_a[:16]} seed12345={key_b[:16]}"
    )


# --------------------------------------------------------------------------- #
# (4a) GPU-LOCK-ENV INVARIANCE -- the v0.21.0 cross-process WARM-LOAD blocker.
#
# scripts/with_gpu_lock.sh exports GPUWRF_GPU_LOCK_{HELD,FD,FILE,HOLDER_FILE,
# LABEL,TOKEN}; LABEL and TOKEN are UNIQUE per invocation. The fail-SAFE env
# auto-discovery (global_trace_env_hash) folded ALL GPUWRF_* vars into the key, so
# every lock-wrapped process computed a DIFFERENT cheap_key -> the warm run looked
# under a key the cold run never wrote -> "fallback:missing" for an artifact that
# DID exist (just under a different filename). The CPU suite was not lock-wrapped,
# so it was process-stable and MISSED this -- this test injects the lock env and
# would have caught it. (TWO fresh processes, each with a different lock token.)
# --------------------------------------------------------------------------- #
def test_gpu_lock_env_does_not_fragment_cheap_key():
    """The per-invocation GPU-lock bookkeeping env MUST NOT change the cheap_key."""
    lock_a = {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_FD": "9",
        "GPUWRF_GPU_LOCK_FILE": "/tmp/wrf_gpu2_gpu.lock",
        "GPUWRF_GPU_LOCK_HOLDER_FILE": "/tmp/wrf_gpu2_gpu.lock.holder",
        "GPUWRF_GPU_LOCK_LABEL": "coldA",
        "GPUWRF_GPU_LOCK_TOKEN": "<LOCK_TOKEN_A>",
    }
    lock_b = dict(lock_a)
    lock_b["GPUWRF_GPU_LOCK_LABEL"] = "warmB"  # different invocation
    lock_b["GPUWRF_GPU_LOCK_TOKEN"] = "<LOCK_TOKEN_B>"  # unique token
    key_none, _, _ = _lower_in_subprocess({})  # no lock env (the canonical key)
    key_a, _, _ = _lower_in_subprocess(lock_a)
    key_b, _, _ = _lower_in_subprocess(lock_b)
    assert key_a == key_b, (
        "GPU-lock bookkeeping env fragments the cheap_key (per-invocation TOKEN/"
        f"LABEL leaks in): cold={key_a[:16]} warm={key_b[:16]} -> cross-process "
        "WARM LOAD breaks (fallback:missing)"
    )
    assert key_a == key_none, (
        "the GPU-lock env must be inert vs no-lock too (it is pure infra "
        f"bookkeeping): lock={key_a[:16]} no-lock={key_none[:16]}"
    )


def test_aot_prewarm_env_does_not_fragment_cheap_key():
    """The AOT prewarm orchestration knob is not a lowered-HLO determinant."""

    key_none, _, _ = _lower_in_subprocess({})
    key_off, _, _ = _lower_in_subprocess({"GPUWRF_NESTED_AOT_PREWARM": "0"})
    key_on, _, _ = _lower_in_subprocess({"GPUWRF_NESTED_AOT_PREWARM": "1"})
    assert key_none == key_off == key_on, (
        "GPUWRF_NESTED_AOT_PREWARM leaked into cheap_key despite being an AOT "
        f"orchestration knob: none={key_none[:16]} off={key_off[:16]} on={key_on[:16]}"
    )


def test_gpu_lock_env_prefix_is_denylisted():
    """The lock bookkeeping vars are excluded by name AND by inert prefix."""
    import os as _os

    saved = {k: _os.environ.get(k) for k in (
        "GPUWRF_GPU_LOCK_TOKEN", "GPUWRF_GPU_LOCK_SOMETHING_NEW",
    )}
    try:
        base = ck.global_trace_env_hash()
        # A NEW (not individually denylisted) var under the inert prefix must also
        # be excluded by the prefix guard, so a future lock var cannot re-break it.
        _os.environ["GPUWRF_GPU_LOCK_TOKEN"] = "<LOCK_TOKEN_C>"
        _os.environ["GPUWRF_GPU_LOCK_SOMETHING_NEW"] = "future-bookkeeping"
        assert ck.global_trace_env_hash() == base, (
            "a GPUWRF_GPU_LOCK_* var leaked into global_trace_env_hash (prefix "
            "guard not applied)"
        )
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


# --------------------------------------------------------------------------- #
# (4b) PREWARM-vs-EAGER carry REPRESENTATION INVARIANCE.
#
# The parallel-prewarm worker keys a domain off a ShapeDtypeStruct carry
# (aot_precompile._to_shape_dtype_tree -- picklable, no device buffers) while the
# eager warm loop keys off the CONCRETE runtime carry. _to_shape_dtype_tree lowers
# to the IDENTICAL HLO (documented invariant), so the cheap_key MUST match too --
# else the prewarm-serialized blob is unloadable by the eager loop and the WHOLE
# parallel-prewarm wall-win is defeated. The MetaTy strings (sharding/committed/
# is_jax_array) differ between the two representations, so they must NOT be hashed
# raw; carry_aval_hash hashes a canonical placement class instead.
# --------------------------------------------------------------------------- #
def test_prewarm_shape_carry_and_concrete_carry_share_cheap_key():
    """ShapeDtypeStruct (prewarm) and concrete (eager) carries -> SAME cheap_key."""
    from gpuwrf.runtime.aot_precompile import _to_shape_dtype_tree
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base

    carry, namelist, clock_base = _build_call()
    start = jnp.asarray(1, jnp.int32)
    cadence = int(namelist.radiation_cadence_steps)

    k_concrete = ck.cheap_key(
        _advance_chunk_fori, (carry, namelist, start, clock_base),
        {"n_steps": 1, "cadence": cadence}, namelist,
    )
    carry_shapes = _to_shape_dtype_tree(carry)
    clock_shapes = _to_shape_dtype_tree(build_clock_base(namelist))
    k_shape = ck.cheap_key(
        _advance_chunk_fori, (carry_shapes, namelist, start, clock_shapes),
        {"n_steps": 1, "cadence": cadence}, namelist,
    )
    assert k_concrete and k_shape
    assert k_concrete == k_shape, (
        "prewarm shape-dtype carry and concrete carry compute DIFFERENT cheap_keys "
        f"(concrete={k_concrete[:16]} shape={k_shape[:16]}) -> the prewarm-written "
        "blob is unloadable by the eager loop; the MetaTy placement fields must be "
        "canonicalized (placement_class), not hashed raw"
    )


def test_carry_aval_hash_is_representation_invariant():
    """carry_aval_hash matches for ShapeDtypeStruct vs concrete carry."""
    from gpuwrf.runtime.aot_precompile import _to_shape_dtype_tree
    from gpuwrf.runtime.operational_mode import build_clock_base

    carry, namelist, clock_base = _build_call()
    start = jnp.asarray(1, jnp.int32)
    args_c = (carry, namelist, start, clock_base)
    args_s = (
        _to_shape_dtype_tree(carry), namelist, start,
        _to_shape_dtype_tree(build_clock_base(namelist)),
    )
    kw = {"n_steps": 1, "cadence": 1}
    assert ck.carry_aval_hash(args_c, kw) == ck.carry_aval_hash(args_s, kw)


def test_placement_class_still_distinguishes_real_placement_changes():
    """The canonicalization keeps the GPT-critic SAFETY: a genuinely different
    placement (committed array / non-single-device sharding) is NOT collapsed."""
    default_abstract = ck._placement_class({"committed": None, "sharding": None})
    default_concrete = ck._placement_class(
        {"committed": False, "sharding": "SingleDeviceSharding(device=CpuDevice(id=0))"}
    )
    committed = ck._placement_class(
        {"committed": True, "sharding": "SingleDeviceSharding(device=CpuDevice(id=0))"}
    )
    sharded = ck._placement_class(
        {"committed": False, "sharding": "NamedSharding(mesh=m, spec=PartitionSpec('x'))"}
    )
    # The abstract (prewarm) and uncommitted single-device (eager) cases MUST agree.
    assert default_abstract == default_concrete == "default"
    # A genuinely different placement MUST be distinguished (no silent reuse).
    assert committed != "default" and committed != default_concrete
    assert sharded != "default" and sharded != committed


# --------------------------------------------------------------------------- #
# (5) COLLISION DETECTION -- a deliberately-incomplete key IS caught.
#
# This is the adversarial proof: build a key that OMITS the trace-time env
# component (component 5). Under it, two configs that differ ONLY in an env knob
# (which DOES change the HLO) collide -> same key, different HLO. The injectivity
# check MUST flag it. This proves the test has teeth: it would FAIL if a real
# determinant were dropped, so the PASS in test (1) is meaningful.
# --------------------------------------------------------------------------- #
def test_incomplete_key_collision_is_detected():
    """An env-blind key collides on the env axis -> the collision IS caught.

    Two FRESH-PROCESS lowerings that differ only in GPUWRF_ACOUSTIC_UNROLL have
    DIFFERENT HLOs (it changes the acoustic-loop unroll STRUCTURE) while leaving the
    namelist AND the carry avals UNCHANGED -- so it is captured ONLY by component 5
    (global_trace_env_hash). The env-BLIND key (component 5 omitted) therefore gives
    the two lowerings the SAME key. We assert (a) the HLOs really differ (the knob
    is a genuine determinant captured by no other component), (b) the incomplete key
    COLLIDES (the exact silent-wrong bug), and (c) the FULL key does NOT collide
    (the real key closes the gap). This proves the injectivity test has teeth: it
    would FAIL if component 5 were dropped.
    """
    full_on, hlo_on, incomplete_on = _lower_in_subprocess({})
    full_off, hlo_off, incomplete_off = _lower_in_subprocess(
        {"GPUWRF_ACOUSTIC_UNROLL": "4"}
    )

    # The env knob is a genuine HLO determinant captured by NO other component.
    assert hlo_on != hlo_off, (
        "precondition: GPUWRF_ACOUSTIC_UNROLL must change the HLO for this test to "
        "mean anything"
    )
    # The INCOMPLETE (env-blind) key COLLIDES -> exactly the silent-wrong bug.
    assert incomplete_on == incomplete_off, (
        "the deliberately-incomplete (env-blind) key did NOT collide; the "
        "collision-detection test is not exercising the env gap it claims to"
    )
    # The FULL key does NOT collide -> the real key closes the gap.
    assert full_on != full_off, (
        "the FULL cheap_key collided on the env axis -- component 5 "
        "(global_trace_env_hash) is not capturing GPUWRF_ACOUSTIC_UNROLL"
    )


# =========================================================================== #
# GPT-CRITIC SAFETY FIXES (KEY_SCHEMA v2) -- the P0/P1/P2 closures.
# =========================================================================== #
import dataclasses  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402


# --------------------------------------------------------------------------- #
# A tiny REAL jit + serialize harness (no GPU State needed) reused by the P0-1
# collision-overwrite test. Two DIFFERENT compiled programs are forced under ONE
# cheap_key to construct a real two-HLO same-key collision on disk.
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _MiniNamelist:
    radiation_cadence_steps: int
    scale: object
    time_utc: object = datetime(2024, 9, 1, tzinfo=timezone.utc)
    noahmp_julian: float = 1.0
    noahmp_yearlen: float = 365.0

    def tree_flatten(self):
        return (self.scale,), (
            int(self.radiation_cadence_steps),
            self.time_utc,
            float(self.noahmp_julian),
            float(self.noahmp_yearlen),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        cadence, time_utc, julian, yearlen = aux
        return cls(
            radiation_cadence_steps=cadence,
            scale=children[0],
            time_utc=time_utc,
            noahmp_julian=julian,
            noahmp_yearlen=yearlen,
        )


def _mini_advance():
    from gpuwrf.runtime import domain_tree as dt  # build_clock_base lives here

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {"y": carry["a"] + carry["b"]}

    return advance_like, dt


def _compile_mini(mult: int):
    """Compile a small jit that lowers to a DIFFERENT HLO per ``mult`` (a*mult)."""
    from gpuwrf.runtime import domain_tree as dt

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {"y": carry["a"] * mult + carry["b"]}

    namelist = _MiniNamelist(radiation_cadence_steps=7, scale=jnp.asarray(1, jnp.int32))
    clock_base = dt.build_clock_base(namelist)
    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,))}
    lowered = advance_like.lower(
        carry, namelist, jnp.asarray(1, jnp.int32), clock_base, n_steps=1, cadence=7
    )
    from gpuwrf.runtime import aot_executable as aotx

    hlo = aotx.hlo_sha256_from_lowered(lowered)
    compiled = lowered.compile()
    return compiled, hlo


def _compile_mini_with_lowered(mult: int):
    """Like :func:`_compile_mini` but ALSO returns the ``lowered`` object.

    The HLO-digest fix needs the lowered StableHLO to assert the persisted
    ``meta.hlo_sha256`` equals ``hlo_sha256_from_lowered(lowered)`` and to drive
    the no-``hlo_sha256``-arg serialize path (the cross-process GPU bug)."""
    from gpuwrf.runtime import aot_executable as aotx
    from gpuwrf.runtime import domain_tree as dt

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {"y": carry["a"] * mult + carry["b"]}

    namelist = _MiniNamelist(radiation_cadence_steps=7, scale=jnp.asarray(1, jnp.int32))
    clock_base = dt.build_clock_base(namelist)
    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,))}
    lowered = advance_like.lower(
        carry, namelist, jnp.asarray(1, jnp.int32), clock_base, n_steps=1, cadence=7
    )
    hlo = aotx.hlo_sha256_from_lowered(lowered)
    compiled = lowered.compile()
    return compiled, lowered, hlo


@pytest.fixture()
def _cache(monkeypatch, tmp_path):
    from gpuwrf.runtime import compile_cache as cc

    cache_dir = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("GPUWRF_JAX_CACHE", raising=False)
    cc.configure_compilation_cache()
    return cache_dir


# --------------------------------------------------------------------------- #
# (P0-1) Collision-overwrite is FAIL-CLOSED: a REAL two-HLO same-cheap_key
#        collision does NOT overwrite k_<cheap_key>; the load fails OPEN.
# --------------------------------------------------------------------------- #
def test_p0_1_collision_does_not_overwrite_cheap_key_blob_and_load_fails_open(_cache):
    """Two distinct HLOs forced under ONE cheap_key -> quarantine, no overwrite.

    Constructs the exact silent-wrong scenario: program A is serialized under
    cheap_key K; then program B (a DIFFERENT HLO) is serialized under the SAME K.
    The write-guard must (a) NOT overwrite A's blob with B, (b) quarantine K, and
    (c) make the subsequent load of K fail OPEN (return None) -- never silently
    serve B (or a half-overwritten blob)."""
    from gpuwrf.runtime import aot_precompile as aotp

    compiled_a, hlo_a = _compile_mini(mult=2)
    compiled_b, hlo_b = _compile_mini(mult=5)
    assert hlo_a and hlo_b and hlo_a != hlo_b, "preconditions: A and B are distinct HLOs"

    forged_key = "collision" + "0" * 56  # one shared cheap_key for both programs

    # 1) Write program A under the forged cheap_key (clean, first write).
    sa = aotp._serialize_domain_blob(
        "d01", compiled_a, str(_cache),
        hlo_sha256=hlo_a, cheap_key=forged_key, key_schema=ck.KEY_SCHEMA,
    )
    assert sa["aot_written"] is True and sa.get("cheap_key") == forged_key, sa
    blob_path, meta_path = aotp._aot_blob_paths(
        "d01", str(_cache), cheap_key=forged_key
    )
    a_blob_sha = sa["blob_sha256"]
    assert blob_path.is_file()

    # 2) Write program B under the SAME forged cheap_key -> COLLISION.
    sb = aotp._serialize_domain_blob(
        "d01", compiled_b, str(_cache),
        hlo_sha256=hlo_b, cheap_key=forged_key, key_schema=ck.KEY_SCHEMA,
    )
    # The write must be marked quarantined and must NOT have written the cheap-key
    # address (cheap_key in the status is None -> only the hlo-addressed fallback).
    assert sb.get("cheap_key_quarantined") is True, sb
    assert sb.get("cheap_key") is None, sb

    # 3) The cheap-key blob/meta are REMOVED (poisoned), and the key is quarantined.
    assert aotp.cheap_key_is_quarantined("d01", forged_key, str(_cache))
    assert not blob_path.is_file(), "ambiguous cheap-key blob was NOT removed"
    assert not meta_path.is_file(), "ambiguous cheap-key meta was NOT removed"

    # 4) The load of the quarantined cheap_key FAILS OPEN (no silent wrong blob).
    call, status = aotp.load_domain_blob(
        "d01", str(_cache), cheap_key=forged_key, return_status=True
    )
    assert call is None
    assert status["source"] == "fallback:cheap-key-quarantined", status

    # 5) Program B's HLO-addressed fallback DID land (keyed by exact HLO -> safe).
    b_blob_path, _ = aotp._aot_blob_paths("d01", str(_cache), hlo_sha256=hlo_b)
    assert b_blob_path.is_file(), "the hlo-addressed fallback for B should exist"
    # And A's exact-HLO fallback is untouched/distinct from B's.
    assert a_blob_sha != sb["blob_sha256"]


def test_p0_1_quarantine_blocks_future_cheap_key_writes(_cache):
    """Once quarantined, even a re-write of the matching program skips the cheap key."""
    from gpuwrf.runtime import aot_precompile as aotp

    compiled_a, hlo_a = _compile_mini(mult=3)
    forged_key = "poison" + "0" * 58
    aotp.quarantine_cheap_key("d01", forged_key, str(_cache), reason="test")
    assert aotp.cheap_key_is_quarantined("d01", forged_key, str(_cache))
    s = aotp._serialize_domain_blob(
        "d01", compiled_a, str(_cache),
        hlo_sha256=hlo_a, cheap_key=forged_key, key_schema=ck.KEY_SCHEMA,
    )
    # Wrote (the hlo fallback) but NOT under the quarantined cheap key.
    assert s["aot_written"] is True
    assert s.get("cheap_key") is None, s
    assert s.get("cheap_key_quarantined") is True, s
    blob_path, _ = aotp._aot_blob_paths("d01", str(_cache), cheap_key=forged_key)
    assert not blob_path.is_file()


# --------------------------------------------------------------------------- #
# (P1-3) Load metadata enforcement: a cheap-key load requires the full contract.
# --------------------------------------------------------------------------- #
def test_p1_3_cheap_key_load_requires_metadata_contract(_cache):
    """meta.cheap_key/key_schema/hlo_sha256/blob_sha256 must all positively match."""
    import pickle

    from gpuwrf.runtime import aot_executable as aotx
    from gpuwrf.runtime import aot_precompile as aotp

    compiled, hlo = _compile_mini(mult=2)
    blob, meta = aotx.serialize(
        compiled, hlo_sha256=hlo, cheap_key="goodkey" + "0" * 57, key_schema=ck.KEY_SCHEMA
    )
    good_key = "goodkey" + "0" * 57

    def _write(m):
        bp, mp = aotp._aot_blob_paths("d01", str(_cache), cheap_key=good_key)
        aotp._atomic_write_bytes(bp, blob)
        aotp._atomic_write_bytes(mp, pickle.dumps(m))
        return bp, mp

    # (a) wrong key_schema -> reject.
    _write(dataclasses.replace(meta, key_schema="GPUWRF-AOTKEY-OLD"))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st

    # (b) missing hlo_sha256 -> reject.
    _write(dataclasses.replace(meta, hlo_sha256=None))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st

    # (c) missing blob_sha256 -> reject (cannot attest integrity).
    _write(dataclasses.replace(meta, blob_sha256=None))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st

    # (d) meta.cheap_key None (not recorded) -> reject (was silently ACCEPTED before).
    _write(dataclasses.replace(meta, cheap_key=None))
    call, st = aotp.load_domain_blob("d01", str(_cache), cheap_key=good_key, return_status=True)
    assert call is None and st["source"] == "fallback:cheap-key-meta-mismatch", st


# --------------------------------------------------------------------------- #
# (P0-2) Transitive source identity: a source-fingerprint change -> different key.
# --------------------------------------------------------------------------- #
def test_p0_2_source_fingerprint_change_changes_program_key(monkeypatch):
    """If source_fingerprint_hash changes, program_key + cheap_key change too.

    Proves a TRACED-CALLEE source edit (modeled here by perturbing the source
    fingerprint, which a real edit to dynamics/physics would do) invalidates the
    key even with __version__ unbumped -- closing the silent stale-blob reuse."""
    carry, namelist, clock_base = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    args = (carry, namelist, jnp.asarray(1, jnp.int32), clock_base)
    kw = {"n_steps": 1, "cadence": 1}
    pk0 = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek0 = ck.exec_key(_advance_chunk_fori, args, kw, namelist)

    # Model a source edit: clear the lru_cache and force a different fingerprint.
    ck.source_fingerprint_hash.cache_clear()
    monkeypatch.setattr(
        ck, "source_fingerprint_hash", lambda: "EDITED-SOURCE-FINGERPRINT"
    )
    pk1 = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek1 = ck.exec_key(_advance_chunk_fori, args, kw, namelist)
    assert pk0 != pk1, "program_key did NOT respond to a source-fingerprint change"
    assert ek0 != ek1, "exec_key did NOT respond to a source-fingerprint change"


def test_p0_2_source_fingerprint_is_nonempty_and_stable():
    """The real source fingerprint resolves to a stable 64-hex digest."""
    ck.source_fingerprint_hash.cache_clear()
    a = ck.source_fingerprint_hash()
    b = ck.source_fingerprint_hash()
    assert a == b and len(a) == 64, a


# --------------------------------------------------------------------------- #
# (P0-2 SCOPE) source_fingerprint is scoped to TRACE-REACHABLE modules only.
#
# THE v0.21.0 9-NEST cold->warm BLOCKER: source_fingerprint_hash content-hashed
# the WHOLE src/gpuwrf tree, so a concurrent UNCOMMITTED edit to an ORCHESTRATION
# module (domain_tree.py -- the de-fuse-flip agent's diff that landed between the
# cold compile and the warm load) shifted ALL 9 domains' cheap_keys while the
# lowered _advance_chunk_fori HLO was byte-identical -> the warm process looked
# under keys the cold process never wrote -> 100% fallback:missing, full ~60-min
# re-lower, headline defeated. The 3-nest passed only because its cold+warm ran in
# ONE unedited window (NOT a max_dom bug). Fix: scope the digest to the STATIC
# IMPORT CLOSURE of the traced body (gpuwrf.runtime.operational_mode), a provable
# SUPERSET of the trace-reachable set -> an orchestration/IO edit is invariant,
# but a genuine traced-callee (dynamics/physics/coupling) edit still shifts it.
# --------------------------------------------------------------------------- #
def _patch_one_source_file_byte(rel_path: str):
    """Context manager: append a harmless comment to one src file, then restore.

    Models a real working-tree edit (the exact failure mode: a concurrent agent
    edits a .py between the cold and warm runs). Returns a contextmanager."""
    import contextlib
    from pathlib import Path

    @contextlib.contextmanager
    def _ctx():
        src_root = Path(ck.__file__).resolve().parents[2]  # .../src
        target = src_root / rel_path
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n# regression-test transient edit\n")
            ck.source_fingerprint_hash.cache_clear()
            yield
        finally:
            target.write_bytes(original)
            ck.source_fingerprint_hash.cache_clear()

    return _ctx()


def test_source_fingerprint_invariant_to_orchestration_edit():
    """An edit to a NON-traced orchestration module must NOT shift the fingerprint.

    domain_tree.py / nested_pipeline.py / cli.py / aot_*.py import the traced body
    but are NOT imported BY it -> outside the import closure -> cannot change the
    lowered HLO -> the cache key MUST be invariant. This is the exact 9-nest
    cold->warm blocker: a domain_tree.py edit between cold and warm must no longer
    desync the keys. Would have FAILED before the scope fix (whole-tree digest)."""
    ck.source_fingerprint_hash.cache_clear()
    base = ck.source_fingerprint_hash()
    for orchestration in (
        "gpuwrf/runtime/domain_tree.py",
        "gpuwrf/integration/nested_pipeline.py",
        "gpuwrf/cli.py",
        "gpuwrf/runtime/aot_precompile.py",
    ):
        with _patch_one_source_file_byte(orchestration):
            shifted = ck.source_fingerprint_hash()
        assert shifted == base, (
            f"source_fingerprint shifted on an HLO-IRRELEVANT edit to {orchestration} "
            f"(the 9-nest cold->warm blocker): base={base[:16]} edited={shifted[:16]}. "
            "The orchestration module must be OUTSIDE the trace-import closure."
        )


@pytest.mark.parametrize(
    "traced_module",
    [
        "gpuwrf/runtime/operational_mode.py",  # the traced body itself
        "gpuwrf/dynamics/core/acoustic.py",  # a deep traced dycore callee
        "gpuwrf/physics/__init__.py",  # a traced physics module
        "gpuwrf/coupling/physics_couplers.py",  # a traced coupling callee
        "gpuwrf/contracts/state.py",  # a traced contract (carry struct)
    ],
)
def test_source_fingerprint_responds_to_traced_callee_edit(traced_module):
    """An edit to a TRACE-REACHABLE module MUST shift the fingerprint (SAFETY).

    The safety side of the scope fix: a real source edit to a module that CAN
    change the lowered _advance_chunk_fori HLO must invalidate the cache key, so a
    stale blob is never silently reused. Under-scoping here = SILENT WRONG RESULT;
    this positive control guards against an over-narrow closure."""
    ck.source_fingerprint_hash.cache_clear()
    base = ck.source_fingerprint_hash()
    with _patch_one_source_file_byte(traced_module):
        shifted = ck.source_fingerprint_hash()
    assert shifted != base, (
        f"source_fingerprint did NOT respond to an edit of the trace-reachable "
        f"module {traced_module} -- it is missing from the import closure "
        "(under-scoped = silent stale-blob reuse risk)."
    )


def test_trace_closure_includes_traced_excludes_orchestration():
    """The import closure structurally contains every traced subtree and NO
    orchestration/IO module -- a direct assertion on the scope set itself."""
    from pathlib import Path

    pkg_root = Path(ck.__file__).resolve().parent.parent  # .../src/gpuwrf
    files = ck._trace_reachable_source_files(pkg_root)
    assert files is not None and files, "import closure could not be computed"
    rels = {f.relative_to(pkg_root).as_posix() for f in files}
    # MUST contain traced callees (HLO-affecting source).
    for needle in (
        "runtime/operational_mode.py",
        "contracts/state.py",
        "contracts/grid.py",
    ):
        assert any(r.endswith(needle) for r in rels), f"closure missing traced {needle}"
    assert any(r.startswith("dynamics/") for r in rels), "closure missing dynamics/"
    assert any(r.startswith("physics/") for r in rels), "closure missing physics/"
    assert any(r.startswith("coupling/") for r in rels), "closure missing coupling/"
    # MUST NOT contain orchestration/IO (the bug source).
    for forbidden in (
        "runtime/domain_tree.py",
        "integration/nested_pipeline.py",
        "cli.py",
        "runtime/aot_precompile.py",
        "runtime/aot_cheap_key.py",
    ):
        assert not any(r.endswith(forbidden) for r in rels), (
            f"orchestration/IO module {forbidden} is INSIDE the closure -- it would "
            "re-introduce the 9-nest cold->warm key churn"
        )
    # The closure is a strict subset of the whole tree (proves it actually narrows).
    whole = {
        p.relative_to(pkg_root).as_posix()
        for p in pkg_root.rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert rels < whole, "closure is not a strict subset of the whole tree"


def test_cheap_key_stable_across_process_and_concurrent_orchestration_edit():
    """END-TO-END regression for the 9-nest blocker, on CPU, two fresh processes.

    Process A (cold): computes the cheap_key with a clean tree.
    Process B (warm): computes it under a DIFFERENT PYTHONHASHSEED *and* with an
    HLO-irrelevant orchestration edit applied (domain_tree.py) -- exactly the
    cold->warm window that broke the GPU 9-nest. The keys MUST match. Before the
    scope fix B's key would differ (whole-tree digest shifted by the edit), so this
    test would have CAUGHT the blocker."""
    import contextlib
    from pathlib import Path

    key_a, hlo_a, _ = _lower_in_subprocess({"PYTHONHASHSEED": "0"})

    src_root = Path(ck.__file__).resolve().parents[2]
    orchestration = src_root / "gpuwrf/runtime/domain_tree.py"
    original = orchestration.read_bytes()
    try:
        orchestration.write_bytes(original + b"\n# concurrent de-fuse-flip edit\n")
        key_b, hlo_b, _ = _lower_in_subprocess({"PYTHONHASHSEED": "31337"})
    finally:
        orchestration.write_bytes(original)

    assert key_a and key_b
    assert hlo_a == hlo_b, (
        "the orchestration edit changed the lowered HLO -- test fixture is wrong "
        f"(should be HLO-irrelevant): {hlo_a[:12]} vs {hlo_b[:12]}"
    )
    assert key_a == key_b, (
        "cheap_key is NOT stable across a cold->warm window with a concurrent "
        f"HLO-irrelevant orchestration edit (the 9-nest blocker): cold={key_a[:16]} "
        f"warm={key_b[:16]}. source_fingerprint must be scoped to trace-reachable "
        "modules only."
    )


# --------------------------------------------------------------------------- #
# (P1-4) program_key vs exec_key: same HLO program, different compile env ->
#        SAME program_key but DIFFERENT exec_key (the blob address splits).
# --------------------------------------------------------------------------- #
def test_p1_4_exec_key_splits_on_compile_options_but_program_key_does_not():
    """A change in an exec-only determinant (XLA_FLAGS) must move exec_key only."""
    carry, namelist, clock_base = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    args = (carry, namelist, jnp.asarray(1, jnp.int32), clock_base)
    kw = {"n_steps": 1, "cadence": 1}

    pk_base = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek_base = ck.exec_key(_advance_chunk_fori, args, kw, namelist)

    # exec_env_hash reads XLA_FLAGS live; mutate it and recompute (program is inert).
    old = os.environ.get("XLA_FLAGS")
    os.environ["XLA_FLAGS"] = (old or "") + " --xla_force_host_platform_device_count=3"
    try:
        pk_flag = ck.program_key(_advance_chunk_fori, args, kw, namelist)
        ek_flag = ck.exec_key(_advance_chunk_fori, args, kw, namelist)
    finally:
        if old is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = old

    assert pk_flag == pk_base, "program_key MUST be invariant to XLA_FLAGS (exec-only)"
    assert ek_flag != ek_base, "exec_key MUST split on XLA_FLAGS (blob-address determinant)"


# --------------------------------------------------------------------------- #
# (P1-5) Resolved import-time env constants: the key captures the RESOLVED module
#        constant, so an env mutation AFTER import cannot desync it from the HLO.
# --------------------------------------------------------------------------- #
def test_p1_5_module_const_env_hash_is_import_resolved_not_live_env(monkeypatch):
    """module_const_env_hash hashes the RESOLVED constant, not live os.environ.

    The module constant (boundary_apply.NORMAL_BDY_RELAX_STRENGTH) is resolved at
    import; mutating GPUWRF_NORMAL_BDY_RELAX_STRENGTH in this (already-imported)
    process must NOT change the hash -- proving the key tracks what the HLO baked,
    not the live env (the desync the GPT critic flagged)."""
    h0 = ck.module_const_env_hash()
    monkeypatch.setenv("GPUWRF_NORMAL_BDY_RELAX_STRENGTH", "999.0")
    h1 = ck.module_const_env_hash()
    assert h0 == h1, (
        "module_const_env_hash changed on a POST-import env mutation -- it must hash "
        "the resolved import-time constant, not live os.environ"
    )


def test_p1_5_module_const_env_changes_when_resolved_constant_changes(monkeypatch):
    """If the resolved module constant differs, the hash differs (positive control)."""
    import gpuwrf.coupling.boundary_apply as bdy

    real = bdy.NORMAL_BDY_RELAX_STRENGTH
    h0 = ck.module_const_env_hash()
    monkeypatch.setattr(bdy, "NORMAL_BDY_RELAX_STRENGTH", real + 7.0)
    h1 = ck.module_const_env_hash()
    assert h0 != h1, "hash did not respond to a changed resolved module constant"


# --------------------------------------------------------------------------- #
# (P1-5 CI scanner) Flag trace-reachable import-time env-derived module constants
#        that are NOT covered by IMPORT_TIME_ENV_CONSTANTS.
# --------------------------------------------------------------------------- #
def test_no_uncovered_import_time_env_constants():
    """CI scan: every module-level constant resolved from env at IMPORT in a
    trace-reachable module must be in IMPORT_TIME_ENV_CONSTANTS (so the key hashes
    its resolved value). A NEW one trips this test -> add it to the registry.

    Detection: a module-level (column-0) assignment whose RHS contains an
    ``os.environ.get(``/``os.getenv(`` call. Functions/methods (indented env reads)
    are NOT import-time constants -- they resolve at TRACE time and are covered by
    ``global_trace_env_hash`` (live env) instead."""
    import ast

    src_root = Path(__file__).resolve().parent.parent / "src"
    covered = {
        f"{m.replace('.', '/')}.py::{a}" for (m, a) in ck.IMPORT_TIME_ENV_CONSTANTS
    }
    offenders: list[str] = []
    for rel in ck.TRACE_REACHABLE_ENV_SCAN_ROOTS:
        root = src_root / rel
        if not root.is_dir():
            continue
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                tree = ast.parse(py.read_text(), filename=str(py))
            except SyntaxError:
                continue
            for node in tree.body:  # MODULE-LEVEL statements only (import-time)
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if value is None:
                    continue
                reads_env = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and (
                        (
                            c.func.attr == "get"
                            and isinstance(c.func.value, ast.Attribute)
                            and c.func.value.attr == "environ"
                        )
                        or c.func.attr == "getenv"
                    )
                    for c in ast.walk(value)
                )
                if not reads_env:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for t in targets:
                    if isinstance(t, ast.Name):
                        rel_mod = py.relative_to(src_root).as_posix()
                        ident = f"{rel_mod}::{t.id}"
                        if ident not in covered:
                            offenders.append(ident)
    assert not offenders, (
        "UNCOVERED import-time env-derived module constants in trace-reachable code "
        "(add each to aot_cheap_key.IMPORT_TIME_ENV_CONSTANTS so the cheap_key hashes "
        f"its RESOLVED value): {sorted(offenders)}"
    )


# --------------------------------------------------------------------------- #
# (P2) Determinant matrix expansion: GridSpec array content + MetaTy.
# --------------------------------------------------------------------------- #
def test_p2_gridspec_array_content_is_content_hashed():
    """GridSpec STATIC-aux array CONTENT (eta_levels / terrain_height) is hashed.

    Proves the static config is CONTENT-hashed (not by salted __hash__/id), so a
    GridSpec static-array content edit cannot share a key with the original. (The
    GridSpec's TRACED children -- e.g. ``vertical`` arrays passed as device leaves
    -- are correctly handled by carry_aval_hash's shape/dtype instead, so we
    perturb the AUX arrays here.)"""
    import dataclasses as _dc

    from gpuwrf.contracts.grid import GridSpec

    g0 = GridSpec.canary_3km_template()
    d0 = ck.canonical_digest(g0)

    # (a) top-level eta_levels (a baked static array) content change.
    e = np.asarray(g0.eta_levels)
    assert e.size > 3
    e2 = e.copy()
    e2[1] = float(e2[1]) * 0.999 + 1e-4
    g_eta = _dc.replace(g0, eta_levels=jnp.asarray(e2, dtype=e.dtype))
    assert ck.canonical_digest(g_eta) != d0, "eta_levels content not hashed"

    # (b) terrain_height (a baked static field) content change.
    if g0.terrain_height is not None:
        th = np.asarray(g0.terrain_height)
        if th.size:
            th2 = th.copy()
            th2.flat[0] = float(th2.flat[0]) + 1.0
            g_th = _dc.replace(g0, terrain_height=jnp.asarray(th2, dtype=th.dtype))
            assert ck.canonical_digest(g_th) != d0, "terrain_height content not hashed"

    # Content equality holds for value-identical copies (no id()/salt leakage).
    g_same = _dc.replace(g0, eta_levels=jnp.asarray(e.copy(), dtype=e.dtype))
    assert ck.canonical_digest(g_same) == d0, "value-identical GridSpec digest differs"


def test_p2_carry_metaty_fields_present_in_aval_record():
    """carry_aval_hash leaf records carry the MetaTy fields (sharding/committed/...)."""
    carry, namelist, clock_base = _build_call()
    leaf = jnp.arange(4.0)
    rec = ck._leaf_metaty_record(leaf)
    for field in ("shape", "dtype", "weak_type", "sharding", "committed", "is_jax_array"):
        assert field in rec, f"MetaTy field {field} missing from leaf record"


def test_p2_clock_base_none_fails_open():
    """clock_base=None (date-static path) must NOT crash; cheap_key fails open or keys.

    The contract requires a non-None clock_base; a None must fail OPEN (return None),
    never raise, so the caller compiles. We assert no exception escapes and the
    result is either a valid key or None (both are fail-safe)."""
    carry, namelist, _cb = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    key = ck.cheap_key(
        _advance_chunk_fori,
        (carry, namelist, jnp.asarray(1, jnp.int32), None),
        {"n_steps": 1, "cadence": 1},
        namelist,
    )
    assert key is None or (isinstance(key, str) and len(key) == 64)


def test_p2_program_key_and_exec_key_are_distinct_and_stable():
    """program_key != exec_key (different folds) and both are stable 64-hex digests."""
    carry, namelist, clock_base = _build_call()
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    args = (carry, namelist, jnp.asarray(1, jnp.int32), clock_base)
    kw = {"n_steps": 1, "cadence": 1}
    pk = ck.program_key(_advance_chunk_fori, args, kw, namelist)
    ek = ck.exec_key(_advance_chunk_fori, args, kw, namelist)
    assert len(pk) == 64 and len(ek) == 64 and pk != ek
    # cheap_key IS exec_key.
    assert ck.cheap_key(_advance_chunk_fori, args, kw, namelist) == ek


# --------------------------------------------------------------------------- #
# CROSS-PROCESS HLO-DIGEST FIX: the persisted meta.hlo_sha256 MUST be the
# lower-only StableHLO digest (== hlo_sha256_from_lowered(lowered)), so a fresh
# warm process loads the cheap-key blob and verify-mode confirms it. Regression
# guard for the GPU bug where the on-disk meta.hlo_sha256 was None (the compiled
# fallback returns None / a DIFFERENT digest) -> fallback:cheap-key-meta-mismatch
# + verify-error.
# --------------------------------------------------------------------------- #
def test_serialize_persists_lowered_hlo_digest_in_meta(_cache):
    """``aot_executable.serialize`` writes the LOWERED digest to ``meta.hlo_sha256``.

    Even when the explicit ``hlo_sha256`` arg is None, passing ``lowered=`` makes
    the persisted digest equal ``hlo_sha256_from_lowered(lowered)`` -- NOT the
    compiled-HLO fallback (which is None on GPU / a non-matching digest on CPU)."""
    from gpuwrf.runtime import aot_executable as aotx

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=2)
    assert hlo, "precondition: lowered HLO digest is available on this backend"

    # (a) explicit hlo arg omitted, but lowered provided -> persisted == lowered digest.
    _blob, meta = aotx.serialize(compiled, lowered=lowered)
    assert meta.hlo_sha256 == hlo, (
        "serialize(lowered=...) must persist the lowered StableHLO digest, "
        f"got {meta.hlo_sha256!r} expected {hlo!r}"
    )

    # (b) the misleading compiled fallback is a DIFFERENT (or None) digest -> proves
    #     the lowered source is what we must persist, never the compiled one.
    compiled_digest = aotx._compiled_hlo_sha256(compiled)
    assert compiled_digest != hlo, (
        "compiled-HLO digest unexpectedly equals the lowered digest; the fix's "
        "premise (they differ) would be moot"
    )


def test_serialize_domain_blob_threads_lowered_so_on_disk_meta_has_hlo(_cache):
    """``_serialize_domain_blob(..., lowered=...)`` -> on-disk meta has a real hlo.

    This is the cold-write contract: the blob is written under the cheap_key, and
    the PICKLED meta on disk carries a nonempty ``hlo_sha256`` == the lowered
    digest. Without the fix the on-disk meta.hlo_sha256 was None."""
    import pickle

    from gpuwrf.runtime import aot_precompile as aotp

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=3)
    cheap = "hlofix" + "0" * 58

    # Simulate the EXACT cold-write GPU bug condition: the caller's hlo_sha256 came
    # back None on this backend, but we thread the lowered object through.
    status = aotp._serialize_domain_blob(
        "d01",
        compiled,
        str(_cache),
        hlo_sha256=None,
        lowered=lowered,
        cheap_key=cheap,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] is True, status
    assert status.get("cheap_key") == cheap, status
    assert status.get("hlo_sha256") == hlo, status

    # Read the ON-DISK meta back (a fresh process only ever sees this).
    _blob_path, meta_path = aotp._aot_blob_paths("d01", str(_cache), cheap_key=cheap)
    with open(meta_path, "rb") as fh:
        on_disk_meta = pickle.load(fh)
    assert on_disk_meta.hlo_sha256 == hlo, (
        "on-disk meta.hlo_sha256 must equal the lowered digest; "
        f"got {on_disk_meta.hlo_sha256!r}"
    )


def test_fresh_process_cheap_key_load_succeeds_after_lowered_serialize(_cache):
    """A fresh-process-style cheap-key load + verify-style cross-check both PASS.

    ``load_domain_blob`` keeps NO in-memory state -- it reads the pickled meta from
    disk every call, so calling it after the serialize models a fresh warm process.
    With the fix the P1-3 metadata-enforcement path accepts the blob (loaded, not
    ``fallback:cheap-key-meta-mismatch``) and the verify-style cross-check
    (``meta.hlo_sha256 == hlo_sha256_from_lowered(live_lowered)``) confirms -- no
    verify-error."""
    from gpuwrf.runtime import aot_executable as aotx
    from gpuwrf.runtime import aot_precompile as aotp

    compiled, lowered, hlo = _compile_mini_with_lowered(mult=4)
    cheap = "warmload" + "0" * 56

    # Cold write (hlo_sha256 arg None on this backend; lowered threaded through).
    status = aotp._serialize_domain_blob(
        "d02",
        compiled,
        str(_cache),
        hlo_sha256=None,
        lowered=lowered,
        cheap_key=cheap,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] and status.get("cheap_key") == cheap, status

    # Fresh-process-style warm load via the metadata-enforcement (P1-3) path.
    call, ld = aotp.load_domain_blob(
        "d02", str(_cache), cheap_key=cheap, return_status=True
    )
    assert call is not None, f"warm cheap-key load failed: {ld}"
    assert ld["source"] == "aot_blob", ld
    assert ld["source"] != "fallback:cheap-key-meta-mismatch", ld
    assert ld["meta_hlo_sha256"] == hlo, ld

    # Verify-mode style cross-check: the recorded meta hlo MUST match the digest a
    # fresh process recomputes by re-lowering the SAME program (so verify confirms,
    # never `missing HLO digest for verify`).
    live_hlo = aotx.hlo_sha256_from_lowered(lowered)
    assert live_hlo and ld["meta_hlo_sha256"] == live_hlo, (
        "verify-mode cross-check would fail: meta hlo "
        f"{ld['meta_hlo_sha256']!r} != live lowered hlo {live_hlo!r}"
    )
