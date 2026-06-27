"""AOT executable serialize/load tests (CPU-only, identity-preserving).

The de-fuse compile-wall fix serializes each domain's compiled
``_advance_chunk_fori`` PJRT executable ONCE (in the parallel-prewarm worker) and
DESERIALIZES + executes it in a fresh process WITHOUT re-lowering. The GPU
kill-gate (``proofs/v021/parallel_compile/aot_killgate/``) PROVED byte-identity on
the real bigswiss ``d01`` module (78 leaves, 0 diffs, deserialize 8.54 s).

These tests exercise the MECHANISM on tiny synthetic graphs (the real WRF State is
GPU-only): serialize/load round-trip, the ``kept_var_idx`` const-drop filter, the
fingerprint-mismatch fallback, the version-keyed AOT cache layout in
``aot_precompile``, and the env gate. They deliberately use NON-fused graphs --
the XLA:CPU AOT loader trips a ``broadcast_multiply_fusion not found`` quirk on
some fused CPU executables that does NOT occur on the GPU (where the kill-gate
proved the path). The 9-nest GPU end-to-end is run by the manager.

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_aot_executable.py``
"""

from __future__ import annotations

import dataclasses
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gpuwrf.runtime import aot_executable as aotx
from gpuwrf.runtime import aot_precompile as aot
from gpuwrf.runtime import compile_cache as cc
from gpuwrf.runtime import domain_tree as dt


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch, tmp_path):
    """Point the cache at a private tmp dir for each test (AOT dir lives under it)."""
    cache_dir = tmp_path / "jit"
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("GPUWRF_JAX_CACHE", raising=False)
    cc.configure_compilation_cache()
    yield cache_dir


def _compile_dropping_graph(n_dead: int = 7):
    """Compile a NON-fused graph whose XLA drops ``n_dead`` of its inputs.

    Returns ``(compiled, args, out_ref)``. The carry has 2 live leaves + ``n_dead``
    unused ('dead*') leaves, so kept_var_idx drops exactly ``n_dead`` (mirrors the
    real 130->123 = 7-dropped case)."""
    def fn(c):
        # NON-fused: plain adds (avoids the CPU-AOT fusion-table quirk).
        return {"y": c["a"] + c["b"], "z": c["a"] + c["a"]}

    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,))}
    for k in range(n_dead):
        carry[f"dead{k}"] = jnp.ones((k + 2,))
    args = (carry,)
    compiled = jax.jit(fn).lower(*args).compile()
    return compiled, args, compiled(*args)


# --------------------------------------------------------------------------- #
# (1) serialize/load round-trip is byte-identical + drops the right buffers
# --------------------------------------------------------------------------- #
def test_serialize_load_roundtrip_byte_identical():
    compiled, args, out_ref = _compile_dropping_graph(n_dead=7)
    blob, meta = aotx.serialize(compiled)

    # kept_var_idx dropped exactly the 7 dead leaves (2 live kept of 9 naive).
    in_flat, _ = jax.tree_util.tree_flatten((args, {}))
    assert meta.kept_var_idx is not None
    assert len(in_flat) == 9
    assert len(meta.kept_var_idx) == 2  # 9 naive - 7 dropped
    assert meta.hlo_sha256 is not None and len(meta.hlo_sha256) == 64
    assert meta.in_avals
    assert {rec["dtype"] for rec in meta.in_avals} >= {"float64"}

    f = aotx.load(blob, meta)
    out_rt = f(*args)
    assert jax.tree_util.tree_structure(out_rt) == jax.tree_util.tree_structure(out_ref)
    for k in out_ref:
        assert np.array_equal(np.asarray(out_rt[k]), np.asarray(out_ref[k]))


def test_load_supplies_exactly_kept_buffers():
    """The loaded callable must supply len(kept_var_idx) buffers, not the naive flat."""
    compiled, args, out_ref = _compile_dropping_graph(n_dead=5)
    blob, meta = aotx.serialize(compiled)
    le = jax.devices()[0].client.deserialize_executable(
        blob, __import__("jaxlib._jax", fromlist=["DeviceList"]).DeviceList((jax.devices()[0],)), None
    )
    in_flat, _ = jax.tree_util.tree_flatten((args, {}))
    # naive flatten = 7 leaves; executable expects only kept (2)
    assert len(in_flat) == 7
    assert len(meta.kept_var_idx) == 2
    selected = [in_flat[i] for i in meta.kept_var_idx]
    bufs = [jax.device_put(np.asarray(x), jax.devices()[0]) for x in selected]
    out_bufs = le.execute(bufs)  # must NOT raise a buffer-count error
    out = jax.tree_util.tree_unflatten(meta.out_tree, out_bufs)
    for k in out_ref:
        assert np.array_equal(np.asarray(out[k]), np.asarray(out_ref[k]))


# --------------------------------------------------------------------------- #
# (2) fingerprint mismatch => load raises (caller falls back to compile)
# --------------------------------------------------------------------------- #
def test_fingerprint_mismatch_refuses_load():
    compiled, args, _ = _compile_dropping_graph(n_dead=1)
    blob, meta = aotx.serialize(compiled)
    bad = dataclasses.replace(
        meta, fingerprint={**meta.fingerprint, "device_kind": "WRONG_TARGET"}
    )
    assert aotx.fingerprint_matches(meta.fingerprint) is True
    assert aotx.fingerprint_matches(bad.fingerprint) is False
    with pytest.raises(RuntimeError):
        aotx.load(blob, bad, check_fingerprint=True)
    # check_fingerprint=False bypasses the guard (still loads + runs).
    f = aotx.load(blob, bad, check_fingerprint=False)
    assert f(*args) is not None


def test_in_tree_mismatch_raises():
    """Calling with a different arg structure than compiled => raise (fallback)."""
    compiled, args, _ = _compile_dropping_graph(n_dead=1)
    blob, meta = aotx.serialize(compiled)
    f = aotx.load(blob, meta)
    with pytest.raises(RuntimeError):
        # Wrong structure: extra positional arg the executable was not compiled for.
        f(*args, jnp.ones((3,)))


def test_load_rejects_runtime_aval_mismatch():
    """A matching HLO guard still rejects wrong runtime buffer shape/dtype."""
    compiled, args, _ = _compile_dropping_graph(n_dead=1)
    blob, meta = aotx.serialize(compiled)
    f = aotx.load(blob, meta)
    bad = dict(args[0])
    bad["a"] = jnp.arange(5.0)
    with pytest.raises(RuntimeError, match="AOT input aval mismatch"):
        f(bad)


# --------------------------------------------------------------------------- #
# (3) aot_precompile: version-keyed AOT dir + serialize-to-disk + load-from-disk
# --------------------------------------------------------------------------- #
def test_aot_dir_is_version_keyed(_isolate_cache):
    d = aot.aot_dir(_isolate_cache)
    assert d is not None
    assert d.parent.name == "aot"
    assert d.name == cc.version_cache_tag()


def test_serialize_then_load_domain_blob_roundtrip(_isolate_cache):
    """Step B writes blob+meta to the version-keyed dir; Step C load reads them."""
    compiled, args, out_ref = _compile_dropping_graph(n_dead=4)
    status = aot._serialize_domain_blob("d01", compiled, str(_isolate_cache))
    assert status["aot_written"] is True, status
    assert status["aot_blob_bytes"] > 0
    assert status["hlo_sha256"] is not None
    blob_path, meta_path = aot._aot_blob_paths(
        "d01", str(_isolate_cache), hlo_sha256=status["hlo_sha256"]
    )
    assert blob_path.is_file() and meta_path.is_file()
    # no stray .tmp left behind by the atomic writer
    assert not list(blob_path.parent.glob("*.tmp"))

    # Step C: load it back as a drop-in callable and byte-compare.
    f = aot.load_domain_blob(
        "d01", str(_isolate_cache), hlo_sha256=status["hlo_sha256"]
    )
    assert f is not None
    out_rt = f(*args)
    for k in out_ref:
        assert np.array_equal(np.asarray(out_rt[k]), np.asarray(out_ref[k]))


def test_load_domain_blob_missing_returns_none(_isolate_cache):
    """No blob on disk => fail-open None (caller compiles)."""
    assert aot.load_domain_blob("nonexistent", str(_isolate_cache)) is None


def test_load_domain_blob_status_reports_missing(_isolate_cache):
    """Step-C diagnostics expose why a domain fell back without raising."""
    call, status = aot.load_domain_blob(
        "nonexistent", str(_isolate_cache), return_status=True
    )
    assert call is None
    assert status["loaded"] is False
    assert status["source"] == "fallback:missing"
    assert "missing AOT artifact" in status["error"]


def test_load_domain_blob_fingerprint_mismatch_returns_none(_isolate_cache):
    """A blob whose saved fingerprint mismatches the live target => None (fallback)."""
    compiled, _, _ = _compile_dropping_graph(n_dead=1)
    blob, meta = aotx.serialize(compiled)
    bad = dataclasses.replace(
        meta, fingerprint={**meta.fingerprint, "jaxlib_version": "0.0.0-WRONG"}
    )
    blob_path, meta_path = aot._aot_blob_paths("d01", str(_isolate_cache))
    aot._atomic_write_bytes(blob_path, blob)
    aot._atomic_write_bytes(meta_path, pickle.dumps(bad))
    assert aot.load_domain_blob("d01", str(_isolate_cache)) is None


def test_load_domain_blob_corrupt_meta_returns_none(_isolate_cache):
    """Corrupt meta => fail-open None, never raises."""
    blob_path, meta_path = aot._aot_blob_paths("d01", str(_isolate_cache))
    aot._atomic_write_bytes(blob_path, b"not-a-real-blob")
    aot._atomic_write_bytes(meta_path, b"not-a-pickle")
    assert aot.load_domain_blob("d01", str(_isolate_cache)) is None


# --------------------------------------------------------------------------- #
# (4) the env gate: GPUWRF_NESTED_AOT is DEFAULT-ON (vNext cheap-key manifest);
#     explicit 0/off/false disables it.
# --------------------------------------------------------------------------- #
def test_aot_env_gate_default_on(monkeypatch):
    # Unset -> default ON for both the prewarm-serialize gate and the eager gate.
    monkeypatch.delenv("GPUWRF_NESTED_AOT", raising=False)
    assert aot._aot_enabled() is True
    assert dt._nested_aot_enabled() is True
    # Explicit off-like values (incl. an explicitly-empty string) DISABLE.
    for val in ("0", "false", "no", "off", "", "junk"):
        monkeypatch.setenv("GPUWRF_NESTED_AOT", val)
        assert aot._aot_enabled() is False, val
        assert dt._nested_aot_enabled() is False, val
    for val in ("1", "true", "yes", "on", "ON", "True"):
        monkeypatch.setenv("GPUWRF_NESTED_AOT", val)
        assert aot._aot_enabled() is True, val
        assert dt._nested_aot_enabled() is True, val


def test_aot_verify_gate_default_off(monkeypatch):
    """Verify-mode (lower-once) is OFF unless GPUWRF_AOT_VERIFY is truthy."""
    monkeypatch.delenv("GPUWRF_AOT_VERIFY", raising=False)
    assert dt._nested_aot_verify_enabled() is False
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv("GPUWRF_AOT_VERIFY", val)
        assert dt._nested_aot_verify_enabled() is True, val
    for val in ("0", "false", "off", ""):
        monkeypatch.setenv("GPUWRF_AOT_VERIFY", val)
        assert dt._nested_aot_verify_enabled() is False, val


def test_env_help_documents_aot_knob():
    assert "GPUWRF_NESTED_AOT" in dt.nested_defuse_env_help()


def test_nested_aot_report_shape():
    rep = dt.nested_aot_report()
    assert "enabled" in rep and "domains" in rep
    assert isinstance(rep["domains"], dict)


# --------------------------------------------------------------------------- #
# (5) Step C dispatch: the advance closure uses the AOT call when loaded, and
#     transparently falls back to the jitted _advance_chunk when not.
# --------------------------------------------------------------------------- #
from datetime import datetime, timezone
from types import SimpleNamespace


class _FakeNamelist:
    radiation_cadence_steps = 7
    time_utc = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)
    noahmp_julian = 15.0
    noahmp_yearlen = 365.0


@dataclasses.dataclass(frozen=True)
class _ArrayAux:
    value: object

    def __getitem__(self, idx):
        return self.value[idx]

    def __eq__(self, other):
        return isinstance(other, _ArrayAux) and np.array_equal(
            np.asarray(self.value), np.asarray(other.value)
        )

    def __hash__(self):
        arr = np.asarray(self.value)
        return hash((arr.shape, str(arr.dtype), arr.tobytes()))


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _AotNamelist:
    radiation_cadence_steps: int
    scale: object
    eta_levels: object
    time_utc: object = datetime(2023, 1, 15, 0, tzinfo=timezone.utc)
    noahmp_julian: float = 15.0
    noahmp_yearlen: float = 365.0

    def tree_flatten(self):
        return (self.scale,), (
            int(self.radiation_cadence_steps),
            self.time_utc,
            float(self.noahmp_julian),
            float(self.noahmp_yearlen),
            self.eta_levels,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        cadence, time_utc, julian, yearlen, eta_levels = aux
        return cls(
            radiation_cadence_steps=cadence,
            scale=children[0],
            eta_levels=eta_levels,
            time_utc=time_utc,
            noahmp_julian=julian,
            noahmp_yearlen=yearlen,
        )


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _FusedCarry:
    state: object

    def replace(self, *, state):
        return _FusedCarry(state=state)

    def tree_flatten(self):
        return (self.state,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        (state,) = children
        return cls(state=state)


def _fake_tree(namelist=None):
    return SimpleNamespace(
        domains={"d01": SimpleNamespace(namelist=namelist or _FakeNamelist())},
    )


def _aot_namelist() -> _AotNamelist:
    return _AotNamelist(
        radiation_cadence_steps=7,
        scale=jnp.asarray(9, dtype=jnp.int32),
        eta_levels=_ArrayAux(jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float64)),
    )


def _aot_carry(leading: int = 1):
    return {
        "state": jnp.arange(leading * 4, dtype=jnp.int32).reshape((leading, 4)),
        "extra": {"soil": jnp.ones((leading, 2), dtype=jnp.float64)},
    }


def _make_aot_advance_like():
    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {
            "state": carry["state"],
            "extra": carry["extra"],
            "seen": (
                start,
                n_steps,
                cadence,
                namelist.scale,
                clock_base.rad_julian,
                jnp.asarray(namelist.eta_levels[0], dtype=jnp.float64),
            ),
        }

    return advance_like


def _cheap_key_for_variant(advance_like, carry, namelist, clock_base):
    """cheap_key for the exact (carry, namelist, n_steps, cadence) Step-C call."""
    from gpuwrf.runtime import aot_cheap_key as ck

    return ck.cheap_key(
        advance_like,
        (carry, namelist, jnp.asarray(1, dtype=jnp.int32), clock_base),
        {"n_steps": int(5), "cadence": int(7)},
        namelist,
    )


def _fused_like_aux(
    *,
    child_weights=(jnp.asarray([1.0, 2.0], dtype=jnp.float64),),
    child_ratios=(3,),
    child_cadences=(7,),
):
    parent = _aot_namelist()
    children = tuple(_aot_namelist() for _ in child_weights)
    return dt._build_fused_aux_namelist(
        parent_namelist=parent,
        parent_cadence=7,
        child_names=tuple(f"d{i + 2:02d}" for i in range(len(child_weights))),
        child_namelists=children,
        child_weights=tuple(child_weights),
        child_bdy_widths=tuple(5 for _ in child_weights),
        child_ratios=tuple(int(v) for v in child_ratios),
        child_cadences=tuple(int(v) for v in child_cadences),
    )


def _make_fused_like_jit():
    @jax.jit
    def fused_like(parent_carry, child_carries, parent_start, child_starts, parent_cb, child_cbs):
        del child_starts, parent_cb, child_cbs
        return parent_carry, tuple(child_carries)

    return fused_like


def _fused_like_key(aux, child_count=1):
    fused_like = _make_fused_like_jit()
    parent = _FusedCarry(jnp.asarray([1.0], dtype=jnp.float64))
    children = tuple(
        _FusedCarry(jnp.asarray([float(i)], dtype=jnp.float64))
        for i in range(child_count)
    )
    parent_cb = dt.build_clock_base(_aot_namelist())
    child_cbs = tuple(dt.build_clock_base(_aot_namelist()) for _ in range(child_count))
    return dt._fused_cascade_cheap_key(
        fused_like,
        aux,
        parent,
        children,
        1,
        tuple(range(1, child_count + 1)),
        parent_cb,
        child_cbs,
    )


def test_aval_signature_default_keeps_v0211_leaf_aval_key(monkeypatch):
    """Default fused cache key stays v0.21.1-compatible for release bit identity."""

    a = jnp.ones((2,), dtype=jnp.float32)
    b = jnp.zeros((2,), dtype=jnp.float32)

    tuple_state = ((a, b),)
    dict_state = ({"a": a, "b": b},)
    fewer_leaves_state = ((a,),)

    monkeypatch.delenv("GPUWRF_AOT_STRICT_AVAL_SIGNATURE", raising=False)
    tuple_sig = dt._aval_signature(tuple_state)
    dict_sig = dt._aval_signature(dict_state)
    fewer_leaves_sig = dt._aval_signature(fewer_leaves_state)

    assert tuple_sig == dict_sig
    assert tuple_sig != fewer_leaves_sig


def test_aval_signature_strict_mode_includes_treedef_and_leaf_count(monkeypatch):
    """Opt-in strict mode splits identical aval leaves under different pytrees."""

    a = jnp.ones((2,), dtype=jnp.float32)
    b = jnp.zeros((2,), dtype=jnp.float32)

    tuple_state = ((a, b),)
    same_tuple_state = (
        (
            jnp.full((2,), 3.0, dtype=jnp.float32),
            jnp.full((2,), 4.0, dtype=jnp.float32),
        ),
    )
    dict_state = ({"a": a, "b": b},)
    fewer_leaves_state = ((a,),)

    monkeypatch.setenv("GPUWRF_AOT_STRICT_AVAL_SIGNATURE", "1")
    tuple_sig = dt._aval_signature(tuple_state)
    dict_sig = dt._aval_signature(dict_state)
    fewer_leaves_sig = dt._aval_signature(fewer_leaves_state)
    assert tuple_sig == dt._aval_signature(same_tuple_state)
    assert tuple_sig[1] == dict_sig[1]
    assert tuple_sig[0] != dict_sig[0]
    assert tuple_sig != dict_sig
    assert tuple_sig != fewer_leaves_sig
    assert tuple_sig[0][0] == "treedef"
    assert tuple_sig[0][2] == 2
    assert fewer_leaves_sig[0][2] == 1


def _lower_compile_serialize_variant(name, carry, namelist, cache_dir, advance_like):
    from gpuwrf.runtime import aot_cheap_key as ck

    clock_base = dt.build_clock_base(namelist)
    lowered = advance_like.lower(
        carry,
        namelist,
        jnp.asarray(1, dtype=jnp.int32),
        clock_base,
        n_steps=int(5),
        cadence=int(7),
    )
    hlo = aotx.hlo_sha256_from_lowered(lowered)
    assert hlo is not None
    cheap_key = _cheap_key_for_variant(advance_like, carry, namelist, clock_base)
    assert cheap_key is not None
    compiled = lowered.compile()
    # Serialize under the cheap-key address so the eager warm path (which looks up
    # by cheap_key WITHOUT lowering) finds it.
    status = aot._serialize_domain_blob(
        name,
        compiled,
        str(cache_dir),
        hlo_sha256=hlo,
        cheap_key=cheap_key,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] is True, status
    assert status["hlo_sha256"] == hlo
    assert status["cheap_key"] == cheap_key
    return hlo, status


def test_fused_cheap_key_folds_edge_geometry():
    """Changing fused closure edge geometry must change the metadata-only key."""
    base = _fused_like_key(_fused_like_aux())
    assert base is not None

    changed_weight = _fused_like_key(
        _fused_like_aux(child_weights=(jnp.asarray([1.0, 3.0], dtype=jnp.float64),))
    )
    changed_ratio = _fused_like_key(_fused_like_aux(child_ratios=(5,)))
    changed_cadence = _fused_like_key(_fused_like_aux(child_cadences=(11,)))
    changed_count = _fused_like_key(
        _fused_like_aux(
            child_weights=(
                jnp.asarray([1.0, 2.0], dtype=jnp.float64),
                jnp.asarray([4.0, 5.0], dtype=jnp.float64),
            ),
            child_ratios=(3, 3),
            child_cadences=(7, 7),
        ),
        child_count=2,
    )

    assert changed_weight != base
    assert changed_ratio != base
    assert changed_cadence != base
    assert changed_count != base


def test_fused_cascade_uses_loaded_aot_blob(monkeypatch):
    """Fused AOT warm path calls the loaded fused/<parent> executable."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.delenv("GPUWRF_AOT_VERIFY", raising=False)
    parent_nl = _aot_namelist()
    child_nl = _aot_namelist()
    parent = _FusedCarry(jnp.asarray([1.0], dtype=jnp.float64))
    child = _FusedCarry(jnp.asarray([2.0], dtype=jnp.float64))
    seen = {}

    def fake_aot_call(parent_carry, child_carries, parent_start, child_starts, parent_cb, child_cbs):
        seen["called"] = True
        seen["parent_start"] = int(parent_start)
        seen["child_starts"] = tuple(int(v) for v in child_starts)
        del parent_cb, child_cbs
        return parent_carry.replace(state=parent_carry.state + 10), child_carries

    def fake_load(name, *args, **kwargs):
        assert name == "fused/d02"
        assert kwargs.get("cheap_key")
        assert kwargs.get("return_status") is True
        return fake_aot_call, {
            "name": name,
            "loaded": True,
            "source": "aot_blob",
            "cheap_key": kwargs["cheap_key"],
            "meta_hlo_sha256": "abcd" * 16,
        }

    monkeypatch.setattr(aot, "load_domain_blob", fake_load)
    program = dt._build_fused_cascade_program(
        parent_name="d02",
        parent_namelist=parent_nl,
        parent_cadence=7,
        child_names=("d03",),
        child_namelists=(child_nl,),
        child_weights=(jnp.asarray([1.0], dtype=jnp.float64),),
        child_bdy_widths=(5,),
        child_ratios=(3,),
        child_cadences=(7,),
    )

    out_parent, out_children = program(parent, (child,), 4, (12,))
    assert seen == {"called": True, "parent_start": 4, "child_starts": (12,)}
    assert np.array_equal(np.asarray(out_parent.state), np.asarray(parent.state + 10))
    assert len(out_children) == 1
    assert np.array_equal(np.asarray(out_children[0].state), np.asarray(child.state))
    assert dt.nested_aot_report()["domains"]["fused/d02"]["loaded"] is True


def test_fused_cascade_captures_aot_blob_on_load_miss(monkeypatch):
    """Cold fused miss compiles the fused executable and serializes fused/<parent>."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.delenv("GPUWRF_AOT_VERIFY", raising=False)
    parent_nl = _aot_namelist()
    child_nl = _aot_namelist()
    parent = _FusedCarry(jnp.asarray([1.0], dtype=jnp.float64))
    child = _FusedCarry(jnp.asarray([2.0], dtype=jnp.float64))

    def fake_advance(carry, namelist, start, clock_base, *, n_steps, cadence):
        del clock_base
        inc = jnp.asarray(start, dtype=jnp.float64)
        inc = inc + jnp.asarray(n_steps + cadence, dtype=jnp.float64)
        inc = inc + jnp.asarray(namelist.scale, dtype=jnp.float64)
        return carry.replace(state=carry.state + inc)

    def fake_force(child_state, parent_state, weights, *, bdy_width):
        return child_state + parent_state + jnp.sum(weights) + float(bdy_width)

    monkeypatch.setattr(dt, "_advance_chunk", fake_advance)
    monkeypatch.setattr(dt, "build_child_boundary_package", fake_force)
    monkeypatch.setattr(
        aot,
        "load_domain_blob",
        lambda name, *args, **kwargs: (
            None,
            {
                "name": name,
                "loaded": False,
                "source": "fallback:missing",
                "cheap_key": kwargs.get("cheap_key"),
            },
        ),
    )
    captured = {}

    def fake_serialize(name, compiled, cache_dir, **kwargs):
        del compiled, cache_dir
        captured.update({"name": name, **kwargs})
        return {
            "aot_written": True,
            "aot_blob_bytes": 123,
            "aot_path": "/tmp/fused.xlaexec",
            "hlo_sha256": kwargs.get("hlo_sha256"),
            "cheap_key": kwargs.get("cheap_key"),
        }

    monkeypatch.setattr(aot, "_serialize_domain_blob", fake_serialize)
    program = dt._build_fused_cascade_program(
        parent_name="d02",
        parent_namelist=parent_nl,
        parent_cadence=7,
        child_names=("d03",),
        child_namelists=(child_nl,),
        child_weights=(jnp.asarray([1.0], dtype=jnp.float64),),
        child_bdy_widths=(5,),
        child_ratios=(3,),
        child_cadences=(7,),
    )

    out_parent, out_children = program(parent, (child,), 4, (12,))
    assert captured["name"] == "fused/d02"
    assert captured["cheap_key"]
    assert captured["hlo_sha256"]
    assert captured["key_schema"]
    assert np.asarray(out_parent.state).shape == (1,)
    assert len(out_children) == 1


def test_fused_cascade_cached_calls_are_keyed_by_aval_signature(monkeypatch, capsys):
    """Alternating fused carry shapes compile once per shape and reuse both."""

    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.delenv("GPUWRF_AOT_VERIFY", raising=False)
    parent_nl = _aot_namelist()
    child_nl = _aot_namelist()
    parent = _FusedCarry(jnp.asarray([1.0], dtype=jnp.float64))
    child_shape1 = _FusedCarry(jnp.ones((1, 2), dtype=jnp.float64))
    child_shape2 = _FusedCarry(jnp.ones((2, 2), dtype=jnp.float64))

    def fake_advance(carry, namelist, start, clock_base, *, n_steps, cadence):
        del namelist, clock_base, n_steps, cadence
        inc = jnp.asarray(start, dtype=carry.state.dtype)
        return carry.replace(state=carry.state + inc)

    def fake_force(child_state, parent_state, weights, *, bdy_width):
        del parent_state, weights, bdy_width
        return child_state

    loads: list[str | None] = []
    serializes: list[str | None] = []

    def fake_load(name, *args, **kwargs):
        del args
        loads.append(kwargs.get("cheap_key"))
        return None, {
            "name": name,
            "loaded": False,
            "source": "fallback:missing",
            "cheap_key": kwargs.get("cheap_key"),
        }

    def fake_serialize(name, compiled, cache_dir, **kwargs):
        del name, compiled, cache_dir
        serializes.append(kwargs.get("cheap_key"))
        return {
            "aot_written": True,
            "aot_blob_bytes": 123,
            "aot_path": "/tmp/fused-shape.xlaexec",
            "hlo_sha256": kwargs.get("hlo_sha256"),
            "cheap_key": kwargs.get("cheap_key"),
        }

    monkeypatch.setattr(dt, "_advance_chunk", fake_advance)
    monkeypatch.setattr(dt, "build_child_boundary_package", fake_force)
    monkeypatch.setattr(aot, "load_domain_blob", fake_load)
    monkeypatch.setattr(aot, "_serialize_domain_blob", fake_serialize)

    program = dt._build_fused_cascade_program(
        parent_name="d02",
        parent_namelist=parent_nl,
        parent_cadence=7,
        child_names=("d03",),
        child_namelists=(child_nl,),
        child_weights=(jnp.asarray([1.0], dtype=jnp.float64),),
        child_bdy_widths=(5,),
        child_ratios=(1,),
        child_cadences=(7,),
    )

    out_a1, _ = program(parent, (child_shape1,), 4, (12,))
    out_b, _ = program(parent, (child_shape2,), 5, (13,))
    out_a2, _ = program(parent, (child_shape1,), 6, (14,))

    assert np.asarray(out_a1.state).shape == (1,)
    assert np.asarray(out_b.state).shape == (1,)
    assert np.asarray(out_a2.state).shape == (1,)
    assert len(loads) == 2
    assert len(serializes) == 2
    assert serializes[0] != serializes[1]
    assert "fallback:fused-cached-call-error" not in capsys.readouterr().err


def test_advance_falls_back_to_jit_when_aot_off(monkeypatch):
    """AOT off => the advance closure calls the jitted _advance_chunk, NOT a blob."""
    # AOT is DEFAULT-ON now; this test exercises the explicit-OFF path.
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "0")
    seen = {}

    def _fake_chunk(carry, namelist, start, clock_base, *, n_steps, cadence):
        seen["path"] = "jit"
        seen["n_steps"] = int(n_steps)
        return ("JIT_RESULT", carry)

    monkeypatch.setattr(dt, "_advance_chunk", _fake_chunk)
    advance = dt._operational_advance_factory(_fake_tree())
    out = advance("d01", "CARRY", start_step=1, n_steps=3)
    assert seen["path"] == "jit"
    assert seen["n_steps"] == 3
    assert out == ("JIT_RESULT", "CARRY")
    assert dt.nested_aot_report()["enabled"] is False


def test_advance_uses_aot_call_when_loaded(monkeypatch):
    """AOT on + a load returns a callable => the advance closure uses it (drop-in)."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    advance_like = _make_aot_advance_like()
    namelist = _aot_namelist()
    carry = _aot_carry(1)

    def _fake_chunk(*a, **k):  # must NOT be called when the blob loads
        raise AssertionError("jitted path used despite a loaded AOT blob")

    monkeypatch.setattr(dt, "_advance_chunk", _fake_chunk)
    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)

    calls = {"n": 0}

    def _fake_aot_call(carry, namelist, start, clock_base, *, n_steps, cadence):
        calls["n"] += 1
        # n_steps/cadence are Python-int runtime kwargs, matching the worker's
        # `.lower(..., n_steps=int, cadence=int)` call shape exactly.
        assert isinstance(n_steps, int) and isinstance(cadence, int)
        return ("AOT_RESULT", carry)

    monkeypatch.setattr(
        aot, "load_domain_blob", lambda name, *a, **k: _fake_aot_call
    )
    advance = dt._operational_advance_factory(_fake_tree(namelist))
    out = advance("d01", carry, start_step=2, n_steps=5)
    assert out == ("AOT_RESULT", carry)
    # memoised: a second advance does not re-load
    advance("d01", carry, start_step=3, n_steps=5)
    assert calls["n"] == 2
    rep = dt.nested_aot_report()
    assert rep["enabled"] is True
    assert rep["domains"]["d01"]["loaded"] is True


def test_advance_captures_variant_when_aot_load_returns_none(monkeypatch, _isolate_cache):
    """AOT on + load miss => compile current variant and serialize it."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    advance_like = _make_aot_advance_like()
    namelist = _aot_namelist()
    carry = _aot_carry(1)

    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)
    monkeypatch.setattr(aot, "load_domain_blob", lambda name, *a, **k: None)
    advance = dt._operational_advance_factory(_fake_tree(namelist))
    out = advance("d01", carry, start_step=1, n_steps=2)

    assert np.array_equal(np.asarray(out["state"]), np.asarray(carry["state"]))
    status = dt.nested_aot_report()["domains"]["d01"]
    assert status["loaded"] is False
    assert status["source"] == "fallback:jit-compiled+aot-captured"
    assert status["aot_written"] is True
    assert Path(status["aot_path"]).is_file()


def test_advance_falls_back_when_aot_call_raises(monkeypatch):
    """A loaded blob whose execute raises => compile/capture the current variant."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    advance_like = _make_aot_advance_like()
    namelist = _aot_namelist()
    carry = _aot_carry(1)

    def _boom(*a, **k):
        raise RuntimeError("execute exploded")

    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)
    monkeypatch.setattr(aot, "load_domain_blob", lambda name, *a, **k: _boom)
    advance = dt._operational_advance_factory(_fake_tree(namelist))
    out = advance("d01", carry, start_step=1, n_steps=2)

    assert np.array_equal(np.asarray(out["state"]), np.asarray(carry["state"]))
    assert "jit-compiled" in dt.nested_aot_report()["domains"]["d01"]["source"]


def test_advance_loads_serialized_blob_through_real_step_c_path(
    monkeypatch, _isolate_cache, capsys
):
    """CPU repro for Step C: eager advance loads a worker-shaped AOT blob.

    The carry includes an extra nested subtree to mirror the Noah-MP-style
    post-seeding carry expansion that must remain part of the serialized in_tree.
    """
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    advance_like = _make_aot_advance_like()
    eta_levels_compile = _ArrayAux(jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float64))
    eta_levels_eager = _ArrayAux(jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float64))
    namelist_compile = _AotNamelist(
        radiation_cadence_steps=7,
        scale=jnp.asarray(9, dtype=jnp.int32),
        eta_levels=eta_levels_compile,
    )
    namelist_eager = _AotNamelist(
        radiation_cadence_steps=7,
        scale=jnp.asarray(9, dtype=jnp.int32),
        eta_levels=eta_levels_eager,
    )
    clock_base = dt.build_clock_base(namelist_compile)
    carry = {
        "state": jnp.arange(3, dtype=jnp.int32),
        "extra": {"soil": jnp.ones((2,), dtype=jnp.float64)},
    }

    lowered = advance_like.lower(
        carry,
        namelist_compile,
        jnp.asarray(1, dtype=jnp.int32),
        clock_base,
        n_steps=int(5),
        cadence=int(7),
    )
    hlo = aotx.hlo_sha256_from_lowered(lowered)
    assert hlo is not None
    from gpuwrf.runtime import aot_cheap_key as ck

    cheap_key = _cheap_key_for_variant(
        advance_like, carry, namelist_compile, clock_base
    )
    assert cheap_key is not None
    compiled = lowered.compile()
    status = aot._serialize_domain_blob(
        "d01",
        compiled,
        str(_isolate_cache),
        hlo_sha256=hlo,
        cheap_key=cheap_key,
        key_schema=ck.KEY_SCHEMA,
    )
    assert status["aot_written"] is True, status

    def _fake_chunk(*a, **k):
        raise AssertionError("jitted fallback used despite a loadable AOT blob")

    monkeypatch.setattr(dt, "_advance_chunk", _fake_chunk)
    monkeypatch.setattr(dt, "_advance_chunk_fori", _make_aot_advance_like())
    advance = dt._operational_advance_factory(_fake_tree(namelist_eager))
    out = advance("d01", carry, start_step=2, n_steps=5)

    assert np.array_equal(np.asarray(out["state"]), np.asarray(carry["state"]))
    assert np.array_equal(
        np.asarray(out["extra"]["soil"]), np.asarray(carry["extra"]["soil"])
    )
    assert int(np.asarray(out["seen"][1])) == 5
    assert int(np.asarray(out["seen"][2])) == 7
    assert float(np.asarray(out["seen"][5])) == 1.0
    rep = dt.nested_aot_report()
    assert rep["domains"]["d01"]["loaded"] is True
    assert rep["domains"]["d01"]["source"] == "aot_blob"
    assert "domain=d01 loaded=true source=aot_blob" in capsys.readouterr().err


def test_step_c_captures_and_warm_loads_multiple_shape_variants(
    monkeypatch, _isolate_cache, capsys
):
    """Cold captures a second carry-shape variant; warm loads both variants."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    namelist_compile = _aot_namelist()
    # Separate but value-identical static aux object: still exercises the 28cde291
    # static metadata relaxation while adding the new shape-variant behavior.
    namelist_eager = _aot_namelist()
    carry1 = _aot_carry(1)
    carry2 = _aot_carry(2)
    advance_like = _make_aot_advance_like()
    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)

    hlo1, status1 = _lower_compile_serialize_variant(
        "d01", carry1, namelist_compile, _isolate_cache, advance_like
    )
    assert Path(status1["aot_path"]).is_file()
    monkeypatch.setattr(dt, "_advance_chunk_fori", _make_aot_advance_like())

    def _fake_chunk(*a, **k):
        raise AssertionError("legacy _advance_chunk fallback used")

    monkeypatch.setattr(dt, "_advance_chunk", _fake_chunk)

    # Cold eager run: shape-1 loads the prewarmed blob; shape-2 has no blob yet,
    # so Step C compiles exactly that lowered variant and serializes it.
    advance = dt._operational_advance_factory(_fake_tree(namelist_eager))
    out1 = advance("d01", carry1, start_step=1, n_steps=5)
    assert np.array_equal(np.asarray(out1["state"]), np.asarray(carry1["state"]))
    first = dt.nested_aot_report()["domains"]["d01"]
    assert first["loaded"] is True
    assert first["hlo_sha256"] == hlo1

    out2 = advance("d01", carry2, start_step=2, n_steps=5)
    assert np.array_equal(np.asarray(out2["state"]), np.asarray(carry2["state"]))
    second = dt.nested_aot_report()["domains"]["d01"]
    assert second["loaded"] is False
    assert second["source"] == "fallback:jit-compiled+aot-captured"
    assert second["aot_written"] is True
    hlo2 = second["hlo_sha256"]
    assert hlo2 and hlo2 != hlo1
    assert Path(second["aot_path"]).is_file()
    variant_blobs = sorted((aot.aot_dir(_isolate_cache) / "d01").glob("*.xlaexec"))
    assert len(variant_blobs) == 2

    # Fresh Step-C factory: no in-memory compiled callables are carried over.
    # Both shape variants should AOT-load from disk; serialize must not be called.
    def _no_serialize(*a, **k):
        raise AssertionError("warm run serialized despite existing variant blob")

    monkeypatch.setattr(aot, "_serialize_domain_blob", _no_serialize)
    capsys.readouterr()
    monkeypatch.setattr(dt, "_advance_chunk_fori", _make_aot_advance_like())
    warm = dt._operational_advance_factory(_fake_tree(namelist_eager))
    warm1 = warm("d01", carry1, start_step=3, n_steps=5)
    warm2 = warm("d01", carry2, start_step=4, n_steps=5)
    assert np.array_equal(np.asarray(warm1["state"]), np.asarray(carry1["state"]))
    assert np.array_equal(np.asarray(warm2["state"]), np.asarray(carry2["state"]))
    err = capsys.readouterr().err
    assert err.count("domain=d01 loaded=true source=aot_blob") == 2
    assert f"hlo={hlo1[:12]}" in err
    assert f"hlo={hlo2[:12]}" in err


# --------------------------------------------------------------------------- #
# (6) VERIFY MODE (GPUWRF_AOT_VERIFY): lower-once confirm; fail-CLOSED on a
#     cheap_key collision (loaded blob's HLO != recorded digest).
# --------------------------------------------------------------------------- #
def test_verify_mode_confirms_matching_blob(monkeypatch, _isolate_cache, capsys):
    """verify=on + a blob whose recorded HLO matches the live lower => loads it."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.setenv("GPUWRF_AOT_VERIFY", "1")
    advance_like = _make_aot_advance_like()
    namelist = _aot_namelist()
    carry = _aot_carry(1)

    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)
    # Serialize a correct cheap-key-addressed blob with the REAL hlo digest.
    _lower_compile_serialize_variant("d01", carry, namelist, _isolate_cache, advance_like)

    def _fake_chunk(*a, **k):
        raise AssertionError("jitted fallback used despite a verified AOT blob")

    monkeypatch.setattr(dt, "_advance_chunk", _fake_chunk)
    monkeypatch.setattr(dt, "_advance_chunk_fori", _make_aot_advance_like())
    advance = dt._operational_advance_factory(_fake_tree(namelist))
    out = advance("d01", carry, start_step=2, n_steps=5)
    assert np.array_equal(np.asarray(out["state"]), np.asarray(carry["state"]))
    rep = dt.nested_aot_report()
    assert rep["verify"] is True
    assert rep["domains"]["d01"]["loaded"] is True
    assert rep["domains"]["d01"]["source"] == "aot_blob"


def test_verify_mode_fails_closed_on_hlo_mismatch(monkeypatch, _isolate_cache):
    """verify=on + a blob whose recorded HLO is WRONG => fail-CLOSED, compile fresh.

    Simulates a cheap_key collision (the silent-wrong bug): the loaded blob's meta
    carries a bogus hlo_sha256, so the lower-once verify detects the mismatch and
    discards the blob, compiling the correct variant instead.
    """
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.setenv("GPUWRF_AOT_VERIFY", "1")
    advance_like = _make_aot_advance_like()
    namelist = _aot_namelist()
    carry = _aot_carry(1)

    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)
    clock_base = dt.build_clock_base(namelist)
    cheap_key = _cheap_key_for_variant(advance_like, carry, namelist, clock_base)
    lowered = advance_like.lower(
        carry, namelist, jnp.asarray(1, dtype=jnp.int32), clock_base,
        n_steps=int(5), cadence=int(7),
    )
    compiled = lowered.compile()
    # Serialize with a deliberately WRONG hlo digest so verify must reject it.
    from gpuwrf.runtime import aot_cheap_key as ckmod

    status = aot._serialize_domain_blob(
        "d01", compiled, str(_isolate_cache),
        hlo_sha256="deadbeef" * 8, cheap_key=cheap_key, key_schema=ckmod.KEY_SCHEMA,
    )
    assert status["aot_written"] is True, status

    compiled_fresh = {"n": 0}
    real_chunk = dt._advance_chunk

    def _count_chunk(*a, **k):
        compiled_fresh["n"] += 1
        return real_chunk(*a, **k)

    monkeypatch.setattr(dt, "_advance_chunk", _count_chunk)
    monkeypatch.setattr(dt, "_advance_chunk_fori", _make_aot_advance_like())
    advance = dt._operational_advance_factory(_fake_tree(namelist))
    out = advance("d01", carry, start_step=2, n_steps=5)
    # Result is still correct (compiled the right variant), and the blob was NOT used.
    assert np.array_equal(np.asarray(out["state"]), np.asarray(carry["state"]))
    src = dt.nested_aot_report()["domains"]["d01"]["source"]
    assert "verify-MISMATCH" in src or "jit-compiled" in src, src


def test_load_rejects_corrupt_blob_integrity(monkeypatch, _isolate_cache):
    """A blob whose bytes don't match meta.blob_sha256 => load returns None."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    advance_like = _make_aot_advance_like()
    namelist = _aot_namelist()
    carry = _aot_carry(1)
    monkeypatch.setattr(dt, "_advance_chunk_fori", advance_like)
    _hlo, status = _lower_compile_serialize_variant(
        "d01", carry, namelist, _isolate_cache, advance_like
    )
    cheap_key = status["cheap_key"]
    # Corrupt the on-disk blob bytes (append garbage) -> blob_sha256 mismatch.
    blob_path = Path(status["aot_path"])
    with open(blob_path, "ab") as fh:
        fh.write(b"CORRUPTION")
    call, load_status = aot.load_domain_blob(
        "d01", str(_isolate_cache), cheap_key=cheap_key, return_status=True
    )
    assert call is None
    assert load_status["source"] == "fallback:blob-integrity", load_status
