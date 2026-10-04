"""Cache identity and lifecycle checks for overlapping AOT load with init."""
import threading
import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime import aot_executable as ax
from gpuwrf.runtime import aot_precompile as ap


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWRF_JAX_CACHE", "1")
    monkeypatch.setenv("XLA_FLAGS", "")
    value = jnp.arange(8, dtype=jnp.float64)
    lowered = jax.jit(lambda x: x * 2 + 1).lower(value)
    compiled = lowered.compile()
    for name, key in (("d01", "a" * 64), ("d02", "b" * 64)):
        status = ap._serialize_domain_blob(
            name, compiled, tmp_path, lowered=lowered,
            hlo_sha256=ax.hlo_sha256_from_lowered(lowered),
            cheap_key=key, key_schema=ck.KEY_SCHEMA,
        )
        assert status["aot_written"], status
    return tmp_path, value, jax.devices()[0]


def test_prefetch_two_loads_overlap_and_are_consumed_once(cache, monkeypatch):
    directory, value, dev = cache
    original = ap._load_domain_blob
    barrier = threading.Barrier(3)

    def simultaneous(*args, **kwargs):
        barrier.wait(timeout=10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ap, "_load_domain_blob", simultaneous)
    with ap.prefetch_domain_blobs({"d01": "a" * 64, "d02": "b" * 64}, directory, dev=dev) as prefetch:
        barrier.wait(timeout=10)
        for name, key in (("d01", "a" * 64), ("d02", "b" * 64)):
            call, status = ap.load_domain_blob(name, directory, cheap_key=key, dev=dev, return_status=True)
            assert status["loaded"] and status["prefetched"]
            np.testing.assert_array_equal(call(value), np.arange(8) * 2 + 1)
            assert status["prefetch_wait_s"] >= 0
            assert status["prefetch_load_s"] >= 0
        assert not prefetch.futures  # Retired after both first runtime lookups.
    assert not ap._prefetched_loads


def test_foreign_key_does_not_consume_prefetched_executable(cache):
    directory, value, dev = cache
    with ap.prefetch_domain_blobs({"d01": "a" * 64}, directory, dev=dev) as prefetch:
        prefetch.futures["d01"].result(timeout=10)
        assert ap.load_domain_blob("d01", directory, cheap_key="c" * 64, dev=dev) is None
        call, status = ap.load_domain_blob("d01", directory, cheap_key="a" * 64, dev=dev, return_status=True)
        assert not status.get("prefetched")  # The earlier mismatch retired its hint.
        assert not prefetch.futures
        np.testing.assert_array_equal(call(value), np.arange(8) * 2 + 1)


@pytest.mark.parametrize("change", ["schema", "fingerprint", "quarantine"])
def test_current_guards_reject_prefetch_when_policy_changes(cache, monkeypatch, change):
    directory, value, dev = cache
    with ap.prefetch_domain_blobs({"d01": "a" * 64}, directory, dev=dev) as prefetch:
        prefetch.futures["d01"].result(timeout=10)
        if change == "schema":
            monkeypatch.setattr(ck, "KEY_SCHEMA", "different-current-schema")
        elif change == "fingerprint":
            monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
        else:
            monkeypatch.setattr(ap, "cheap_key_is_quarantined", lambda *a, **k: True)
        call, status = ap.load_domain_blob("d01", directory, cheap_key="a" * 64, dev=dev, return_status=True)
        assert call is None and not status["loaded"]
        assert status["source"].startswith("fallback:")


def test_prefetch_close_removes_unused_requests(cache):
    directory, value, dev = cache
    prefetch = ap.prefetch_domain_blobs({"d01": "a" * 64}, directory, dev=dev)
    prefetch.close()
    assert not ap._prefetched_loads


def test_partial_constructor_failure_joins_and_removes_its_first_request(cache, monkeypatch):
    directory, value, dev = cache
    original = ap._aot_blob_paths

    def fail_second(name, *args, **kwargs):
        if name == "d02":
            raise ValueError("second request failed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ap, "_aot_blob_paths", fail_second)
    with pytest.raises(ValueError, match="second request failed"):
        ap.prefetch_domain_blobs({"d01": "a" * 64, "d02": "b" * 64}, directory, dev=dev)
    assert not ap._prefetched_loads
    assert not any(thread.name.startswith("wrf-aot") for thread in threading.enumerate())



def test_close_releases_unclaimed_executable_even_while_handle_survives(cache):
    directory, value, dev = cache
    prefetch = ap.prefetch_domain_blobs({"d01": "a"*64}, directory, dev=dev)
    reference = weakref.ref(prefetch.futures["d01"].result(timeout=10)[0])
    assert reference() is not None
    prefetch.close()
    gc.collect()
    assert reference() is None
    assert not prefetch.futures and not prefetch._slots
    assert not ap._prefetched_loads and not ap._active_prefetches


def test_first_runtime_mismatches_retire_all_unclaimed_executables(cache):
    directory, value, dev = cache
    prefetch = ap.prefetch_domain_blobs({"d01": "a"*64, "d02": "b"*64}, directory, dev=dev)
    references = [weakref.ref(prefetch.futures[name].result(timeout=10)[0])
                  for name in ("d01", "d02")]
    # The real keys miss these artifacts. Nothing invokes either speculative call.
    assert ap.load_domain_blob("d01", directory, cheap_key="c"*64, dev=dev) is None
    assert set(prefetch.futures) == {"d02"}
    assert ap.load_domain_blob("d02", directory, cheap_key="d"*64, dev=dev) is None
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not prefetch.futures and not ap._prefetched_loads
    assert not ap._active_prefetches
    assert not any(t.name.startswith("wrf-aot") for t in threading.enumerate())


def test_last_cheap_key_records_runtime_use_never_speculation(cache):
    directory, value, dev = cache
    pointer = ap.aot_dir(directory) / "d01" / ap._LAST_CHEAP_KEY
    assert pointer.read_text() == "a" * 64  # capture records its key
    hlo = ax.hlo_sha256_from_lowered(jax.jit(lambda x: x * 2 + 1).lower(value))
    assert ap.alias_cheap_key_to_hlo("d01", "c" * 64, hlo, directory)["aliased"]
    assert pointer.read_text() == "c" * 64  # alias records its key
    with ap.prefetch_domain_blobs({"d01": "a" * 64}, directory, dev=dev) as prefetch:
        prefetch.futures["d01"].result(timeout=10)
        assert pointer.read_text() == "c" * 64  # speculative load: no record
        call, status = ap.load_domain_blob("d01", directory, cheap_key="a" * 64, dev=dev, return_status=True)
        assert status["prefetched"] and call is not None
    assert pointer.read_text() == "a" * 64  # runtime load records
    assert ap.load_domain_blob("d01", directory, cheap_key="e" * 64, dev=dev) is None
    assert pointer.read_text() == "a" * 64  # a miss never records
