"""A hard-linked cheap-key alias reuses one native load, with its own ABI guards."""
from dataclasses import replace
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime import aot_executable as ax
from gpuwrf.runtime import aot_precompile as ap
from tests._jax_cache_isolation import private_jax_cache


@pytest.fixture
def aliases(tmp_path, monkeypatch):
    with private_jax_cache(monkeypatch, tmp_path / "jit"):
        monkeypatch.setenv("XLA_FLAGS", "")
        value = jnp.arange(8, dtype=jnp.float64)
        low = jax.jit(lambda x: x * 2 + 1).lower(value)
        key, alias = "a" * 64, "b" * 64
        hlo = ax.hlo_sha256_from_lowered(low)
        status = ap._serialize_domain_blob("d01", low.compile(), tmp_path,
            lowered=low, hlo_sha256=hlo, cheap_key=key, key_schema=ck.KEY_SCHEMA)
        assert status["aot_written"], status
        assert ap.alias_cheap_key_to_hlo("d01", alias, hlo, tmp_path)["aliased"]
        yield tmp_path, value, jax.devices()[0], key, alias


def test_alias_consumes_one_native_load_and_binds_requested_contract(aliases, monkeypatch):
    directory, value, dev, key, alias = aliases
    original = ax.load
    loads = []
    monkeypatch.setattr(ax, "load", lambda *a, **kw: (loads.append(1), original(*a, **kw))[1])
    with ap.prefetch_domain_blobs({"d01": key}, directory, dev=dev) as prefetch:
        source = prefetch.futures["d01"].result(timeout=10)[0]
        call, status = ap.load_domain_blob("d01", directory, cheap_key=alias,
                                         dev=dev, return_status=True)
        assert status["prefetched"] and status["prefetched_alias_from"] == key
        assert status["cheap_key"] == call.meta.cheap_key == alias
        assert call.loaded_executable is source.loaded_executable
        assert loads == [1]
        np.testing.assert_array_equal(call(value), np.arange(8) * 2 + 1)
        with pytest.raises(RuntimeError, match="dtype"):
            call(value.astype(jnp.int64))  # same bytes, wrong dtype (E108)
    assert not ap._prefetched_loads and not ap._active_prefetches


@pytest.mark.parametrize("change", ["key", "schema", "hlo", "sha", "target",
    "kept", "avals", "out_tree", "format", "in_leaves", "corrupt_meta",
    "copy_blob", "changed_blob", "quarantine"])
def test_changed_alias_falls_back_to_ordinary_loader(aliases, monkeypatch, change):
    directory, value, dev, key, alias = aliases
    blob_path, meta_path = ap._aot_blob_paths("d01", directory, cheap_key=alias)
    with ap.prefetch_domain_blobs({"d01": key}, directory, dev=dev) as prefetch:
        prefetch.futures["d01"].result(timeout=10)
        meta = pickle.loads(meta_path.read_bytes())
        edits = {"key": {"cheap_key": "c" * 64}, "schema": {"key_schema": "bad"},
            "hlo": {"hlo_sha256": "f" * 64}, "sha": {"blob_sha256": "f" * 64},
            "target": {"fingerprint": {}}, "kept": {"kept_var_idx": ()},
            "avals": {"in_avals": ()}, "out_tree": {"out_tree": jax.tree.structure(())},
            "format": {"executable_format": "bad"},
            "in_leaves": {"in_tree": jax.tree.structure(((1, 2), {}))}}
        if change in edits:
            meta_path.write_bytes(pickle.dumps(replace(meta, **edits[change])))
        elif change == "corrupt_meta":
            meta_path.write_bytes(b"not pickle")
        elif change == "copy_blob":
            blob = blob_path.read_bytes()
            blob_path.unlink()
            blob_path.write_bytes(blob)  # hashes match, inode no longer attested
        elif change == "changed_blob":
            blob_path.write_bytes(b"changed")
        elif change == "quarantine":
            original = ap.cheap_key_is_quarantined
            monkeypatch.setattr(ap, "cheap_key_is_quarantined",
                lambda n, k, c=None: k == alias or original(n, k, c))
        _, status = ap.load_domain_blob("d01", directory, cheap_key=alias,
                                       dev=dev, return_status=True)
        assert not status.get("prefetched")
        assert not prefetch.futures
