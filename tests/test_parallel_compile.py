"""vNext cross-domain PARALLEL COMPILE tests (CPU-only, numerically inert).

The de-fuse nest path compiles N independent per-domain ``_advance_chunk_fori``
modules SEQUENTIALLY (~Sum(N) ~50 min cold). This change pre-compiles those N
modules CONCURRENTLY in SPAWNED CHILD PROCESSES, each warming exactly one
domain's ``<key>-cache`` entry in the ONE shared version-keyed JAX cache, made
concurrent-write-safe by the FileLock + atomic-writer in ``compile_cache.py``.
The main run then warm-hits all N and runs the UNCHANGED eager de-fuse numerics.

These tests exercise the MECHANISM (spawn + locked cache + warm hit + fail-open +
the gate's no-op/active logic) with tiny synthetic graphs, because the real WRF
``State`` is GPU-only (conftest skips real-State builds on CPU). The full GPU
9-nest cold-wall A/B is run by the manager (see the handoff).

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_parallel_compile.py``
"""

from __future__ import annotations

import multiprocessing
import os
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

import pytest

from gpuwrf.runtime import aot_precompile as aot
from gpuwrf.runtime import compile_cache as cc
from gpuwrf.runtime import domain_tree as dt

from tests import _parallel_compile_workers as W


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch, tmp_path):
    """Point the cache at a private tmp dir + engage the lock for each test."""
    cache_dir = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("GPUWRF_JAX_CACHE_LOCK", "1")
    monkeypatch.delenv("GPUWRF_JAX_CACHE", raising=False)
    cc.configure_compilation_cache()
    yield cache_dir


def _spawn_compile(cache_dir, whichs):
    ctx = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=min(4, len(whichs)), mp_context=ctx) as pool:
        futs = [pool.submit(W.compile_distinct_graph, (str(cache_dir), w)) for w in whichs]
        for f in as_completed(futs):
            results.append(f.result())
    return results


# --------------------------------------------------------------------------- #
# (1) N spawn workers compile N distinct graphs -> N distinct entries + warm hit
# --------------------------------------------------------------------------- #
def test_spawn_workers_write_distinct_entries_and_main_warm_hits(_isolate_cache):
    cache_dir = _isolate_cache
    whichs = [0, 1, 2, 3]
    results = _spawn_compile(cache_dir, whichs)

    assert all(r.get("error") is None for r in results), [r for r in results if r.get("error")]
    # Each distinct graph wrote at least one new entry into the shared cache (XLA
    # may also cache shared sub-computations, so total >= N, all decompressible).
    assert all(r.get("wrote") for r in results)
    entries = cc.cache_entry_count(cache_dir)
    assert entries >= len(whichs)
    # The cross-process FileLock engaged (its .lockfile exists in the shared dir).
    assert (cache_dir / ".lockfile").exists()

    # MAIN PROCESS warm-hit: re-lowering+compiling the SAME graphs adds NO entry.
    import jax
    import jax.numpy as jnp

    before = cc.cache_entry_count(cache_dir)
    for w in whichs:
        const = float(w + 1)

        @jax.jit
        def fn(x, _c=const, _w=w):
            return x * _c + jnp.asarray(_w, dtype=jnp.float64)

        fn(jnp.ones((w + 2,), dtype=jnp.float64)).block_until_ready()
    after = cc.cache_entry_count(cache_dir)
    assert after == before, "main process re-compiled instead of warm-hitting"


def test_seeded_extra_carry_subtree_is_spec_and_warm_hit(_isolate_cache):
    """A post-init Noah-MP-like carry subtree must be part of the prewarm spec.

    This is the CPU reproducer for the stop-ship risk: ``None`` vs a seeded land /
    radiation subtree is a different pytree, so the prewarm spec must come from the
    already-seeded runtime carry, not from rebuilding the generic initial carry.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax._src import compilation_cache as jcc

    # Keep this test independent from JAX's process-global persistent-cache object.
    jcc.reset_cache()
    cc.configure_compilation_cache()

    class _FakeHierarchy:
        def children(self, _name):
            return ()

    class _FakeNamelist:
        radiation_cadence_steps = 7
        time_utc = None
        noahmp_julian = 12.0
        noahmp_yearlen = 366.0

    seeded = {
        "state": np.ones((4,), dtype=np.float64),
        "scratch": np.zeros((4,), dtype=np.float64),
        "noahmp": (
            np.full((2,), 3.0, dtype=np.float64),
            (
                np.full((2,), 4.0, dtype=np.float64),
                np.full((2,), 5.0, dtype=np.float64),
                np.full((2,), 6.0, dtype=np.float64),
            ),
        ),
    }
    unseeded = dict(seeded)
    unseeded["noahmp"] = None
    tree = SimpleNamespace(
        domains={
            "d01": SimpleNamespace(
                namelist=_FakeNamelist(),
                state=object(),
            )
        },
        edges={},
        hierarchy=_FakeHierarchy(),
    )

    spec = aot.build_defused_specs(tree, carries={"d01": seeded})[0]
    expected = aot._to_shape_dtype_tree(seeded)
    assert jax.tree_util.tree_structure(spec.carry) == jax.tree_util.tree_structure(expected)
    assert jax.tree_util.tree_structure(spec.carry) != jax.tree_util.tree_structure(unseeded)

    @jax.jit
    def graph(carry):
        land, (soldn, lwdn, cosz) = carry["noahmp"]
        return (
            jnp.sum(carry["state"])
            + jnp.sum(carry["scratch"])
            + jnp.sum(land)
            + jnp.sum(soldn)
            + jnp.sum(lwdn)
            + jnp.sum(cosz)
        )

    before = cc.cache_entry_count(_isolate_cache)
    graph.lower(spec.carry).compile()
    after_spec = cc.cache_entry_count(_isolate_cache)
    assert after_spec > before
    graph.lower(seeded).compile()
    after_runtime = cc.cache_entry_count(_isolate_cache)
    assert after_runtime == after_spec, "runtime seeded carry did not warm-hit spec"


# --------------------------------------------------------------------------- #
# (2) a crashed worker triggers the driver's fail-open (sequential fallback)
# --------------------------------------------------------------------------- #
def test_crashed_worker_is_fail_open(_isolate_cache):
    cache_dir = _isolate_cache
    ctx = multiprocessing.get_context("spawn")
    results = []
    errored = False
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        futs = [
            pool.submit(W.compile_distinct_graph, (str(cache_dir), 0)),
            pool.submit(W.crash_midway, (str(cache_dir), 1)),
        ]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except BaseException:  # noqa: BLE001 - a broken worker surfaces here
                errored = True
    # The crashing worker raised a BrokenProcessPool-style error. The real driver
    # catches exactly this (it wraps fut.result() in try/except) and falls back to
    # a sequential cold compile -- the run is never wrong, only slower. The
    # load-bearing claim here is that a hard worker crash SURFACES as a catchable
    # error (so the driver can fail open), not that it silently corrupts state.
    assert errored, "expected the crashed worker to surface an error to the pool"


def test_driver_reports_error_on_worker_failure_but_does_not_raise(monkeypatch, _isolate_cache):
    """``prewarm_defused_nest`` is fail-open: a bad spec is captured, not raised."""
    # Force the driver to use a tree whose specs cannot be built (None tree) and
    # confirm it returns a report with an error instead of raising.
    rep = aot.prewarm_defused_nest(object())  # not a DomainTree -> build fails
    assert isinstance(rep, dict)
    assert rep["error"] is not None
    assert rep["warm_all"] is False


# --------------------------------------------------------------------------- #
# (Step A) parent-side warm-hit verification is DEFAULT-OFF (the free cold win)
#
# The parent verification re-lowers all N huge _advance_chunk_fori modules
# SEQUENTIALLY (~30 min, NET-NEGATIVE for wall). It must be OFF by default; the
# eager loop warm-hits each domain anyway. These tests stub the heavy compile
# machinery (specs/pool/verify) so they run on the CPU box in milliseconds and
# assert the CONTROL FLOW: verify=False skips the re-lower entirely, verify=True
# runs it. The real GPU 9-nest A/B is run by the manager.
# --------------------------------------------------------------------------- #
def _stub_prewarm_internals(monkeypatch, *, verify_calls, worker_error=None, verify_miss=False):
    """Stub build_defused_specs / the spawn pool / parent verify for a fast test.

    ``verify_calls`` is a mutable list the stubbed ``_verify_parent_warm_hits``
    appends to, so a test can assert whether (and how) the parent re-lower ran.
    """
    # Two trivial picklable specs (the pool is also stubbed, so picklability of
    # the worker call is irrelevant here -- we replace the whole pool path).
    fake_specs = [
        SimpleNamespace(name="d01"),
        SimpleNamespace(name="d02"),
    ]
    monkeypatch.setattr(aot, "build_defused_specs", lambda tree, carries=None: fake_specs)

    # Stub the per-domain worker so no real (slow) compile + no real spawn runs:
    # the driver only calls pool.submit(_compile_one_domain_worker, spec). Replace
    # the ProcessPoolExecutor with an in-process fake that just runs the worker.
    def _fake_worker(spec):
        return {
            "name": spec.name,
            "compile_seconds": 0.0,
            "wrote_entry": True,
            "error": worker_error,
        }

    monkeypatch.setattr(aot, "_compile_one_domain_worker", _fake_worker)

    class _FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakePool:
        def __init__(self, *a, **k):
            self._futs = {}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, spec):
            fut = _FakeFuture(fn(spec))
            self._futs[fut] = spec.name
            return fut

    monkeypatch.setattr(aot, "ProcessPoolExecutor", lambda *a, **k: _FakePool())
    monkeypatch.setattr(aot, "as_completed", lambda futures: list(futures))

    def _fake_verify(specs):
        verify_calls.append([s.name for s in specs])
        return [
            {
                "name": s.name,
                "entries_before": 1,
                "entries_after": 1 if not verify_miss else 2,
                "entries_delta": 0 if not verify_miss else 1,
                "warm_hit": not verify_miss,
                "verify_seconds": 0.0,
                "error": None,
            }
            for s in specs
        ]

    monkeypatch.setattr(aot, "_verify_parent_warm_hits", _fake_verify)
    return fake_specs


def test_prewarm_verify_default_off_skips_parent_relower(monkeypatch, _isolate_cache):
    """verify defaults OFF: the parent re-lower is NOT called and warm_all is the
    workers-only signal (all workers OK => True, no parent exact-key proof)."""
    verify_calls: list = []
    _stub_prewarm_internals(monkeypatch, verify_calls=verify_calls)

    rep = aot.prewarm_defused_nest(object())  # verify defaults to False

    assert verify_calls == [], "parent re-lower must NOT run when verify is off"
    assert rep["parent_verification"] == []
    assert rep["error"] is None
    assert rep["warm_all"] is True  # all workers reported success
    assert len(rep["domains"]) == 2


def test_prewarm_verify_default_off_warm_all_false_on_worker_error(monkeypatch, _isolate_cache):
    """verify OFF + a worker error => warm_all False + error captured, no re-lower."""
    verify_calls: list = []
    _stub_prewarm_internals(monkeypatch, verify_calls=verify_calls, worker_error="boom")

    rep = aot.prewarm_defused_nest(object())

    assert verify_calls == []
    assert rep["warm_all"] is False
    assert rep["error"] is not None and "error" in rep["error"]


def test_prewarm_verify_on_runs_parent_relower(monkeypatch, _isolate_cache):
    """verify=True runs the parent exact-key check and reports its result."""
    verify_calls: list = []
    _stub_prewarm_internals(monkeypatch, verify_calls=verify_calls)

    rep = aot.prewarm_defused_nest(object(), verify=True)

    assert verify_calls == [["d01", "d02"]], "parent re-lower must run when verify=True"
    assert len(rep["parent_verification"]) == 2
    assert all(r["warm_hit"] for r in rep["parent_verification"])
    assert rep["warm_all"] is True
    assert rep["error"] is None


def test_prewarm_verify_on_detects_silent_no_speedup(monkeypatch, _isolate_cache):
    """verify=True must still catch the silent-no-speedup (entries_delta != 0)."""
    verify_calls: list = []
    _stub_prewarm_internals(monkeypatch, verify_calls=verify_calls, verify_miss=True)

    rep = aot.prewarm_defused_nest(object(), verify=True)

    assert verify_calls == [["d01", "d02"]]
    assert rep["warm_all"] is False
    assert rep["error"] is not None and "warm-hit verification failed" in rep["error"]


# --------------------------------------------------------------------------- #
# (Step A, gate) GPUWRF_NESTED_PARALLEL_VERIFY threads verify into the driver
# --------------------------------------------------------------------------- #
def test_verify_env_default_off_and_threaded(monkeypatch, _isolate_cache):
    """The gate parses GPUWRF_NESTED_PARALLEL_VERIFY (default OFF) and threads it."""
    assert dt._nested_parallel_verify_enabled() is False  # unset

    captured = {}

    def _fake_prewarm(tree, *, carries=None, max_workers=None, verify=False):
        captured["verify"] = verify
        return {"workers": 1, "domains": [], "parent_verification": [],
                "wall_seconds": 0.0, "warm_all": True, "error": None}

    monkeypatch.setattr(
        "gpuwrf.runtime.aot_precompile.prewarm_defused_nest", _fake_prewarm
    )
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", "1")
    # Parallel prewarm is now opt-in (v0.21.0 spawn-safety): set it explicitly so
    # the gate fires the (faked) driver instead of the sequential no-spawn skip.
    monkeypatch.setenv("GPUWRF_NESTED_PARALLEL_COMPILE", "1")

    # default OFF
    monkeypatch.delenv("GPUWRF_NESTED_PARALLEL_VERIFY", raising=False)
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    assert captured["verify"] is False
    assert st["verify"] is False
    assert st["active"] is True

    # =1 re-enables
    monkeypatch.setenv("GPUWRF_NESTED_PARALLEL_VERIFY", "1")
    assert dt._nested_parallel_verify_enabled() is True
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    assert captured["verify"] is True
    assert st["verify"] is True

    # junk / 0 / false stay OFF
    for val in ("0", "false", "no", "off", "bogus", ""):
        monkeypatch.setenv("GPUWRF_NESTED_PARALLEL_VERIFY", val)
        assert dt._nested_parallel_verify_enabled() is False, val


# --------------------------------------------------------------------------- #
# (3) the gate: no-op when fused (default), active when de-fused, =0 disables
# --------------------------------------------------------------------------- #
class _FakeHierarchy:
    def __init__(self, children):
        self._children = children

    def children(self, name):
        return self._children.get(name, ())


class _FakeTree:
    """Minimal stand-in so the gate logic can be exercised without a real State."""

    def __init__(self):
        self.domains = {"d01": object(), "d02": object()}
        self.edges = {}
        self.hierarchy = _FakeHierarchy({"d01": ("d02",), "d02": ()})


def test_gate_is_noop_on_default_fused(monkeypatch):
    """No env flags => fused default, so the de-fuse prewarm gate is inactive."""
    monkeypatch.delenv("GPUWRF_NESTED_DEFUSE_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    monkeypatch.delenv("GPUWRF_NESTED_PARALLEL_COMPILE", raising=False)
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    assert st["active"] is False
    assert st["source"] == "skip:fused-default"


def test_gate_is_noop_when_fused_forced(monkeypatch):
    """Explicit GPUWRF_NESTED_FUSE=1 keeps the fused cascade => prewarm no-op."""
    monkeypatch.delenv("GPUWRF_NESTED_DEFUSE_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    monkeypatch.setenv("GPUWRF_NESTED_FUSE", "1")
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    assert st["active"] is False
    assert st["source"] == "skip:fused-default"


def test_gate_opt_out_disables(monkeypatch):
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", "1")
    monkeypatch.setenv("GPUWRF_NESTED_PARALLEL_COMPILE", "0")
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    assert st["active"] is False
    assert st["source"] == "skip:GPUWRF_NESTED_PARALLEL_COMPILE=0"


def test_gate_active_when_defused_and_parallel_explicit_fails_open(monkeypatch):
    """De-fused + parallel EXPLICITLY on => the gate FIRES the driver; with a fake
    tree the driver's spec build fails and the gate captures it (fail-open, no
    raise). Parallel spawn is now opt-in, so the test sets the worker count."""
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", "1")
    monkeypatch.setenv("GPUWRF_NESTED_PARALLEL_COMPILE", "2")
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    # It attempted the de-fuse prewarm (active), then the driver reported an
    # error building specs from the fake tree -> captured, not raised.
    assert st["attempted"] is True
    assert st["active"] is True
    assert st["source"] == "GPUWRF_NESTED_PARALLEL_COMPILE=2"
    # error surfaced from the driver report (fake tree has no real namelist).
    assert st["error"] is not None


def test_gate_sequential_nospawn_when_defused_parallel_unset(monkeypatch):
    """Spawn-safety: de-fused (explicit) but PARALLEL_COMPILE unset =>
    the gate takes the SEQUENTIAL no-spawn sub-mode (does NOT fire the spawning
    driver, so it never spawn-recurses an unguarded entry point)."""
    monkeypatch.setenv("GPUWRF_NESTED_DEFUSE_COMPILE", "1")
    monkeypatch.delenv("GPUWRF_NESTED_PARALLEL_COMPILE", raising=False)
    st = dt.maybe_prewarm_defused_nest(_FakeTree())
    assert st["attempted"] is True
    assert st["active"] is False
    assert st["source"] == "skip:defuse-sequential-no-parallel"
    assert st["error"] is None


# --------------------------------------------------------------------------- #
# (4) default worker count is RAM/CPU-derived and capped at 4
# --------------------------------------------------------------------------- #
def test_default_parallel_workers_capped_at_4():
    assert aot.default_parallel_workers(9) <= 4
    assert aot.default_parallel_workers(1) == 1
    assert aot.default_parallel_workers(0) == 1  # clamped to >=1


def test_nested_precompile_report_roundtrips(monkeypatch):
    monkeypatch.delenv("GPUWRF_NESTED_DEFUSE_COMPILE", raising=False)
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    dt.maybe_prewarm_defused_nest(_FakeTree())
    rep = dt.nested_precompile_report()
    assert "attempted" in rep and "active" in rep and "source" in rep


def test_env_help_documents_parallel_compile_knob():
    assert "GPUWRF_NESTED_PARALLEL_COMPILE" in dt.nested_defuse_env_help()
    assert "GPUWRF_NESTED_PARALLEL_VERIFY" in dt.nested_defuse_env_help()
    assert "GPUWRF_JAX_CACHE_LOCK" in cc.cache_env_help()
