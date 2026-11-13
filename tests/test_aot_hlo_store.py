"""Content-addressed AOT store keyed by the lowered program (v0.5.0 item 0.d). CPU only.

Covers the pre-registered gate of lane v50-detab and the critic's FIX list:

* OFF: no key work, same files, same cheap key whether or not the switches exist;
  an unusable store root is an unset store;
* key: source path/line differences do not change it (StableHLO locations and the
  Triton IR embedded by Pallas); one constant element does (E224 mutant); so do
  the actual compile options (per call, embedded in the lowering, parsed from
  ``XLA_FLAGS`` inline or from a flags file), the device assignment, the driver,
  the toolchain files and an explicit autotune results file; cases whose compile
  inputs are not bound are refused;
* store: tampered, redirected, stale, corrupt or foreign entries are never served,
  a refusal is never installed under the cheap key, a corrupt entry is repaired;
* ON: two builds with different cheap keys, cache roots and source locations run
  one binary through the real eager, fused, init and threaded paths; a build that
  loses a publication runs the winner.

Run: ``JAX_PLATFORMS=cpu python -m pytest tests/test_aot_hlo_store.py``
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime import aot_executable as aotx
from gpuwrf.runtime import aot_hlo_store as store
from gpuwrf.runtime import aot_precompile as aot
from gpuwrf.runtime import domain_tree as dt
from tests._jax_cache_isolation import private_jax_cache
from tests.test_aot_executable import (
    _FusedCarry,
    _aot_carry,
    _aot_namelist,
    _cheap_key_for_variant,
    _fake_tree,
    _make_aot_advance_like,
)

REPO = Path(__file__).resolve().parents[1]
FAST_MATH = {"xla_cpu_enable_fast_math": True, "xla_cpu_fast_math_honor_nans": False,
             "xla_cpu_fast_math_honor_infs": False, "xla_cpu_fast_math_honor_division": False,
             "xla_cpu_fast_math_honor_functions": False}
ONES, NINES = np.ones((4,)), np.full((4,), 9.0)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    for name in (store.STORE_ENV, store.SEEDS_ENV, store.AUTOTUNE_ENV, "XLA_FLAGS", "GPUWRF_AOT_VERIFY"):
        monkeypatch.delenv(name, raising=False)
    for name, fresh in (("_KEY_STATE", {}), ("_WINNER_CALLS", {}), ("_EVENTS", []), ("CONFIG_ERRORS", []),
                        ("_INIT_KERNELS", []), ("_LOCKS_HELD", {})):
        monkeypatch.setattr(store, name, fresh)  # process-wide by design: one process per test
    with private_jax_cache(monkeypatch, tmp_path / "jit") as cache_dir:
        yield cache_dir


def _module(tmp_path: Path, name: str, source: str, *, pad: int = 0, sub: str = "a"):
    """Import ``source`` from its own directory, shifted down by ``pad`` lines."""
    directory = tmp_path / sub / "gpuwrf_like"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text("\n" * pad + textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location(f"{name}_{sub}_{pad}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PLAIN = """
    import jax.numpy as jnp
    import numpy as np

    TABLE = np.arange(4096, dtype=np.float32) * 0.5

    def step(x):
        y = x[0] + jnp.asarray(TABLE)  # the whole table stays a constant of the program
        return {"y": y, "z": x + x}
"""


def _lower_plain(module):
    return jax.jit(module.step).lower(jnp.arange(8, dtype=jnp.float32))


def _leaves(tree):
    return [np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(tree)]


# --------------------------------------------------------------------------- #
# Key: program
# --------------------------------------------------------------------------- #
def test_key_ignores_source_location_metadata(tmp_path):
    low_a = _lower_plain(_module(tmp_path, "plain", _PLAIN, pad=0, sub="a"))
    low_b = _lower_plain(_module(tmp_path, "plain", _PLAIN, pad=37, sub="other_checkout"))
    # Teeth: the two lowerings do carry different source locations.
    assert low_a.as_text(debug_info=True) != low_b.as_text(debug_info=True)
    key_a, key_b = store.content_key(low_a), store.content_key(low_b)
    assert key_a is not None and key_a.key == key_b.key
    assert key_a.program_sha256 == key_b.program_sha256
    # Same lowering, second evaluation: the key is a pure function of the program.
    assert store.content_key(low_a).key == key_a.key


def test_constant_mutant_changes_key(tmp_path):
    """E224: one changed element of a large constant must change the key."""
    base = _module(tmp_path, "plain", _PLAIN, sub="a")
    mutant = _module(tmp_path, "plain", _PLAIN, sub="b")
    mutant.TABLE = base.TABLE.copy()
    mutant.TABLE[3000] = np.nextafter(mutant.TABLE[3000], np.float32(np.inf))  # one element, 1 ULP
    low_a, low_b = _lower_plain(base), _lower_plain(mutant)
    # Teeth: the elided HLO text rendering cannot see the difference.
    assert low_a.compiler_ir("hlo").as_hlo_text() == low_b.compiler_ir("hlo").as_hlo_text()
    assert store.content_key(low_a).program_sha256 != store.content_key(low_b).program_sha256
    assert "{...}" in low_a.compiler_ir("hlo").as_hlo_text()
    # The lowered-text digest of the existing exact-HLO address does see it.
    assert aotx.hlo_sha256_from_lowered(low_a) != aotx.hlo_sha256_from_lowered(low_b)
    same = _module(tmp_path, "plain", _PLAIN, sub="c")
    assert store.content_key(_lower_plain(same)).key == store.content_key(low_a).key


def test_key_separates_shape_and_dtype(tmp_path):
    module = _module(tmp_path, "plain", _PLAIN)
    keys = {
        store.content_key(jax.jit(module.step).lower(jnp.arange(n, dtype=dtype))).key
        for n in (8, 9) for dtype in (jnp.float32, jnp.float64)
    }
    assert len(keys) == 4


_PALLAS = """
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    SCALE = 2.0

    def kern(x_ref, o_ref):
        o_ref[...] = x_ref[...] * SCALE + 1.0

    def step(x):
        return pl.pallas_call(
            kern, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype), grid=(1,), name="kern",
            compiler_params=plt.CompilerParams(num_warps=1, num_stages=1))(x)
"""


def _lower_pallas(module, monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    try:
        return jax.jit(module.step).trace(jnp.ones((64,), jnp.float32)).lower(
            lowering_platforms=("cuda",))
    except Exception as exc:  # noqa: BLE001 - jaxlib without the Triton lowering
        pytest.skip(f"native Pallas lowering unavailable on this jaxlib: {exc}")


def _triton_ir(lowered) -> list[bytes]:
    from jax._src.lib.mlir import ir

    calls = store._walk_custom_calls(lowered.compiler_ir("stablehlo").operation, [])
    return [ir.StringAttr(ir.DictAttr(op.attributes["mhlo.backend_config"])["ir"]).value_bytes
            for op in calls]


def test_pallas_line_shift_keeps_key_and_kernel_change_breaks_it(tmp_path, monkeypatch):
    """E58/E109: Triton IR embeds source locations; the key must not."""
    low_a = _lower_pallas(_module(tmp_path, "kernel", _PALLAS, pad=0, sub="a"), monkeypatch)
    low_b = _lower_pallas(_module(tmp_path, "kernel", _PALLAS, pad=11, sub="elsewhere"), monkeypatch)
    changed = _module(tmp_path, "kernel", _PALLAS.replace("SCALE = 2.0", "SCALE = 3.0"), sub="c")
    low_c = _lower_pallas(changed, monkeypatch)
    ir_a, ir_b = _triton_ir(low_a), _triton_ir(low_b)
    assert len(ir_a) == 1 and ir_a != ir_b, "teeth: embedded Triton IR must differ by location"

    info: dict = {}
    sha_a = store.canonical_program_sha256(low_a, info)
    assert info["triton_calls"] == 1 and info["triton_kernels"] == 1 and info["triton_raw_kept"] == 0
    assert sha_a == store.canonical_program_sha256(low_b)
    assert sha_a != store.canonical_program_sha256(low_c)
    # The lowering itself is left untouched (the key works on a clone).
    assert _triton_ir(low_a) == ir_a

    # No Triton dialect: the raw IR stays in the key -> location-sensitive, a miss.
    def unavailable(raw):
        raise RuntimeError("no triton dialect")

    monkeypatch.setattr(store, "_canonical_triton_ir", unavailable)
    info = {}
    raw_a = store.canonical_program_sha256(low_a, info)
    assert info["triton_raw_kept"] == 1
    assert raw_a != store.canonical_program_sha256(low_b)
    assert raw_a == store.canonical_program_sha256(low_a)


# --------------------------------------------------------------------------- #
# Key: every effective compile input, or a refusal (critic finding 1)
# --------------------------------------------------------------------------- #
def _mul0(x):
    return {"flux": x * 0.0}  # inf * 0: NaN in IEEE, 0 under fast math


def test_key_binds_the_options_the_compile_really_uses(tmp_path):
    """The replica of JAX's CompileOptions equals what XLA is handed, byte for byte."""
    from jax._src import compiler

    seen = {}
    real = compiler.compile_or_get_cached

    def spy(backend, computation, devices, compile_options, *rest, **kwargs):
        seen["raw"] = bytes(compile_options.SerializeAsString())
        return real(backend, computation, devices, compile_options, *rest, **kwargs)

    x = jnp.asarray([np.inf, 2.0], jnp.float32)
    for jitted, per_call in ((jax.jit(_mul0), None), (jax.jit(_mul0, compiler_options=FAST_MATH), None),
                             (jax.jit(_mul0), {"xla_cpu_enable_fast_math": True})):
        lowered = jitted.lower(x)
        options, device = store.effective_compile_options(lowered, per_call)
        compiler.compile_or_get_cached = spy
        try:
            lowered.compile(per_call)
        finally:
            compiler.compile_or_get_cached = real
        assert bytes(options.SerializeAsString()) == seen["raw"]
        assert device.platform == "cpu"


def test_options_embedded_in_the_lowering_split_the_key_and_the_binary(monkeypatch, tmp_path):
    """Critic 1: a strict call must not receive the fast-math binary of the same program."""
    from gpuwrf.integration.init_kernels import InitKernelCache

    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    x = jnp.asarray([np.inf, 2.0], jnp.float32)
    fast, strict = jax.jit(_mul0, compiler_options=FAST_MATH), jax.jit(_mul0)
    low_fast, low_strict = fast.lower(x), strict.lower(x)
    assert store.canonical_program_sha256(low_fast) == store.canonical_program_sha256(low_strict)
    key_fast, key_strict = store.content_key(low_fast), store.content_key(low_strict)
    assert key_fast.key != key_strict.key
    assert key_fast.components["compile_options_sha256"] != key_strict.components["compile_options_sha256"]
    assert store.content_key(low_strict, compile_options=FAST_MATH).key != key_strict.key

    first = InitKernelCache(tmp_path / "fast_build")
    out_fast = np.asarray(first(fast, x)["flux"])
    assert first.events[-1]["status"]["content_store"]["state"] == "published"
    second = InitKernelCache(tmp_path / "strict_build")
    out_strict = np.asarray(second(strict, x)["flux"])
    lookup = second.events[-1]["status"]["content_store_lookup"]
    assert lookup["state"] == "miss" and not second.events[-1]["hit"], lookup
    assert out_fast[0] == 0.0 and np.isnan(out_strict[0]), (out_fast, out_strict)
    assert len(list((tmp_path / "store").rglob("*.json"))) == 2


def test_location_fields_do_not_move_the_options_digest_and_code_fields_do(tmp_path):
    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    options, _device = store.effective_compile_options(lowered)
    base = store.canonical_compile_options(options)

    def edited(**fields):
        clone = copy.deepcopy(options)
        for name, value in fields.items():
            setattr(clone.executable_build_options.debug_options, name, value)
        return clone

    moved = edited(xla_gpu_per_fusion_autotune_cache_dir="/bundles/research/autotune",
                   xla_dump_to="/tmp/dump", xla_dump_hlo_as_text=True,
                   xla_gpu_kernel_cache_file="/c/k", xla_gpu_cuda_data_dir="/opt/cuda")
    assert bytes(moved.SerializeAsString()) != bytes(options.SerializeAsString())  # teeth
    assert store.canonical_compile_options(moved) == base
    assert store.canonical_compile_options(edited(xla_cpu_enable_fast_math=True)) != base
    assert store.canonical_compile_options(edited(xla_backend_optimization_level=1)) != base
    assert store.canonical_compile_options(edited(xla_disable_hlo_passes="algsimp")) != base
    assert store.canonical_compile_options(options) == base  # the original is not modified
    assert not options.executable_build_options.debug_options.xla_dump_to


def test_device_assignment_is_part_of_the_options_digest():
    """Critic 1: a reversed two-device assignment must not key alike."""
    from jax._src import compiler

    forward = compiler.get_compile_options(num_replicas=1, num_partitions=2,
                                           device_assignment=np.array([[0, 1]]))
    reverse = compiler.get_compile_options(num_replicas=1, num_partitions=2,
                                           device_assignment=np.array([[1, 0]]))
    assert store.canonical_compile_options(forward) != store.canonical_compile_options(reverse)
    single = compiler.get_compile_options(num_replicas=1, num_partitions=1,
                                          device_assignment=np.array([[0]]))
    other = compiler.get_compile_options(num_replicas=1, num_partitions=1,
                                         device_assignment=np.array([[1]]))
    assert store.canonical_compile_options(single) != store.canonical_compile_options(other)


def test_explicit_autotune_inputs_are_bound_by_content_or_refused(tmp_path):
    from jax._src.lib import xla_client as xc

    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    options, _device = store.effective_compile_options(lowered)
    assert store._explicit_autotune(options) == {"load_results_sha256": None}
    results = tmp_path / "results.pb"
    with_file = copy.deepcopy(options)
    with_file.executable_build_options.debug_options.xla_gpu_load_autotune_results_from = str(results)
    with pytest.raises(store.Refused, match="autotune-results-unreadable"):
        store._explicit_autotune(with_file)
    results.write_bytes(b"decisions-a")
    digest_a = store._explicit_autotune(with_file)["load_results_sha256"]
    results.write_bytes(b"decisions-b")
    assert store._explicit_autotune(with_file)["load_results_sha256"] not in (None, digest_a)
    # The path itself is a location; the content is what was bound above.
    assert store.canonical_compile_options(with_file) == store.canonical_compile_options(options)
    pinned = copy.deepcopy(options)
    pinned.executable_build_options.debug_options.xla_gpu_experimental_autotune_cache_mode = (
        xc.AutotuneCacheMode.READ)
    with pytest.raises(store.Refused, match="autotune-cache-read-mode"):
        store._explicit_autotune(pinned)


def test_driver_and_toolchain_identity_are_resolved_or_refused(monkeypatch, tmp_path):
    """Critic 1: bind the actual driver and compiler files, not only wheel versions."""
    assert set(store._driver_identity("cpu")) == {"cpu_features"}
    version = tmp_path / "nvidia_version"
    monkeypatch.setattr(store, "_NVIDIA_VERSION_FILES", (str(tmp_path / "absent"), str(version)))
    with pytest.raises(store.Refused, match="driver-unresolved"):
        store._driver_identity("cuda")
    version.write_text("595.99.02\n")
    assert store._driver_identity("cuda") == {"nvidia_kernel_module": "595.99.02"}

    with pytest.raises(store.Refused, match="toolchain-unresolved"):
        store._toolchain_identity("cuda", "")
    cuda = tmp_path / "cuda"
    (cuda / "nvvm" / "lib64").mkdir(parents=True)
    with pytest.raises(store.Refused, match="no ptxas"):
        store._toolchain_identity("cuda", str(cuda))
    (cuda / "bin").mkdir()
    (cuda / "bin" / "ptxas").write_bytes(b"ptxas 13.2.78")
    (cuda / "nvvm" / "lib64" / "libnvvm.so.4").write_bytes(b"nvvm")
    first = store._toolchain_identity("cuda", str(cuda))
    assert set(first["files"]) == {"bin/ptxas", "nvvm/lib64/libnvvm.so.4"}
    (cuda / "bin" / "ptxas").write_bytes(b"ptxas 13.2.79 ")  # other size: other file
    assert store._toolchain_identity("cuda", str(cuda))["files"]["bin/ptxas"] != first["files"]["bin/ptxas"]

    # The identities are key components; an unsupported backend has no key.
    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    base = store.content_key(lowered)
    monkeypatch.setattr(store, "_driver_identity", lambda platform: {"nvidia_kernel_module": "1"})
    changed = store.content_key(lowered)
    assert changed.key != base.key and changed.program_sha256 == base.program_sha256
    monkeypatch.setattr(store, "_toolchain_identity", lambda platform, path: {"files": {"bin/ptxas": "x"}})
    assert store.content_key(lowered).key not in (base.key, changed.key)

    def unresolved(platform):
        raise store.Refused("driver-unresolved")

    monkeypatch.setattr(store, "_driver_identity", unresolved)
    assert store.key_or_refusal(lowered) == (None, "driver-unresolved")

    class Client:
        platform_version = "rocm 6"

    class Device:
        platform, device_kind, id, client = "gpu", "AMD Radeon", 0, Client()

    options, _device = store.effective_compile_options(lowered)
    with pytest.raises(store.Refused, match="backend-unsupported"):
        store._environment(options, Device())


def test_target_and_export_switches_are_key_components(monkeypatch):
    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    base = store.content_key(lowered)
    assert base.components["schema"] == store.KEY_SCHEMA
    assert "xla_flags" not in base.components["target"]
    assert base.components["target"]["jaxlib_version"] and base.components["target"]["device_id"] == 0
    monkeypatch.setenv("GPUWRF_AOT_SLIM_MODULE", "0")
    assert store.content_key(lowered).key != base.key
    monkeypatch.delenv("GPUWRF_AOT_SLIM_MODULE")
    live = dict(aotx.target_fingerprint(), jaxlib_version="0.0.0")
    monkeypatch.setattr(aotx, "target_fingerprint", lambda dev=None: dict(live))
    assert store.content_key(lowered).key != base.key


def test_unbound_lowerings_are_refused(monkeypatch):
    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    assert store.key_or_refusal(lowered)[1] is None
    lowering = lowered._lowering
    monkeypatch.setattr(lowering, "const_args", [np.ones(3)], raising=False)
    assert store.key_or_refusal(lowered) == (None, "hoisted-constants")
    monkeypatch.setattr(lowering, "const_args", [], raising=False)
    monkeypatch.setitem(lowering.compile_args, "host_callbacks", [object()])
    assert store.key_or_refusal(lowered) == (None, "host-callbacks")
    monkeypatch.setitem(lowering.compile_args, "host_callbacks", [])
    monkeypatch.setattr(lowering, "_device_list", None)
    assert store.key_or_refusal(lowered) == (None, "device-assignment")


_CHILD = r"""
import json, os, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from gpuwrf.integration.init_kernels import InitKernelCache
from gpuwrf.runtime import aot_hlo_store as store

mode, cache_dir = sys.argv[1], sys.argv[2]
out = {"mode": mode}
if mode == "mul0":
    @jax.jit
    def kernel(x):
        return {"flux": x * 0.0}
    x = jnp.asarray([np.inf, 2.0], jnp.float32)
elif mode == "hoist":
    table = jnp.full((4096,), float(sys.argv[3]), jnp.float32)  # a captured device array
    @jax.jit
    def kernel(x):
        return {"flux": x[0] * 0.0 + table}
    x = jnp.arange(4, dtype=jnp.float32)
elif mode == "multidev":
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(np.array(jax.devices()[:2]), ("i",))
    @jax.jit
    def kernel(x):
        return {"flux": x + 1.0}
    x = jax.device_put(jnp.arange(4, dtype=jnp.float32), NamedSharding(mesh, P("i")))
lowered = kernel.lower(x)
key, why = store.key_or_refusal(lowered)
out.update(key=key.key if key else None, why=why,
           program=store.canonical_program_sha256(lowered),
           const_args=len(lowered._lowering.const_args),
           options=key.components["compile_options_sha256"] if key else None)
if mode != "multidev":
    cache = InitKernelCache(Path(cache_dir))
    flux = np.asarray(cache(kernel, x)["flux"]).ravel()
    status = cache.events[-1]["status"]
    out.update(flux0=repr(float(flux[0])), hit=bool(cache.events[-1]["hit"]),
               lookup=(status.get("content_store_lookup") or {}).get("state"),
               publish=(status.get("content_store") or {}).get("state"),
               events=[(e["name"], e["key"], e["state"], e.get("shared")) for e in store.report()["events"]])
print("RESULT " + json.dumps(out))
sys.stdout.flush()
os._exit(0)
"""


def _child(tmp_path: Path, label: str, *args: str, **env_overrides: str) -> dict:
    """One fresh process (XLA parses its flags once per process)."""
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPUWRF_", "XLA_", "JAX_"))}
    env.update(JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1", GPUWRF_FAST_DEFAULTS="0",
               PYTHONPATH=f"{REPO / 'src'}{os.pathsep}{REPO}",
               GPUWRF_JAX_CACHE_DIR=str(tmp_path / f"jit_{label}"),
               GPUWRF_AOT_HLO_STORE=str(tmp_path / "store"), **env_overrides)
    proc = subprocess.run([sys.executable, str(script), args[0], str(tmp_path / f"aot_{label}"), *args[1:]],
                          env=env, capture_output=True, text=True, timeout=600)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, (proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:])
    return json.loads(lines[-1][len("RESULT "):])


def test_flags_file_contents_are_bound_through_the_parsed_options(tmp_path):
    """Critic 1: two processes, ONE flags-file path, different contents."""
    flags = tmp_path / "xla_flags.txt"
    fast = " ".join(f"--{name}={str(value).lower()}" for name, value in FAST_MATH.items())
    flags.write_text(fast + "\n")
    first = _child(tmp_path, "fast", "mul0", XLA_FLAGS=str(flags))
    flags.write_text("--xla_cpu_enable_fast_math=false\n")
    second = _child(tmp_path, "strict", "mul0", XLA_FLAGS=str(flags))
    assert first["program"] == second["program"]
    assert first["key"] != second["key"] and first["options"] != second["options"]
    assert (first["publish"], second["lookup"], second["hit"]) == ("published", "miss", False)
    assert first["flux0"] == "0.0" and second["flux0"] == "nan", (first, second)
    # The same flags inline are the same compile: one key, one binary, by content not by path.
    third = _child(tmp_path, "inline", "mul0", XLA_FLAGS=fast)
    assert third["key"] == first["key"] and third["lookup"] == "hit" and third["flux0"] == "0.0"
    # A bundle-specific autotune cache path is a location: same key, shared binary.
    release = _child(tmp_path, "release", "mul0",
                     XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=/bundles/release/autotune")
    research = _child(tmp_path, "research", "mul0",
                      XLA_FLAGS="--xla_gpu_per_fusion_autotune_cache_dir=/bundles/research/autotune")
    assert release["key"] == research["key"] and research["lookup"] == "hit"
    assert research["flux0"] == "nan"


def test_hoisted_constants_and_multi_device_programs_are_refused(tmp_path):
    """Critic 1: constants outside the module, or a device list, are not in any blob."""
    one = _child(tmp_path, "one", "hoist", "1", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="1")
    nine = _child(tmp_path, "nine", "hoist", "9", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="1")
    assert one["const_args"] == 1 and one["program"] == nine["program"], "teeth: same module text"
    for run in (one, nine):
        assert run["key"] is None and run["why"] == "hoisted-constants"
        assert run["lookup"] == "refused:hoisted-constants" and run["publish"] is None
        # Delta 1.2: the refusal is an event (unshared work must not vanish from a gate).
        assert run["events"] == [["init_kernel", None, "lookup:refused:hoisted-constants", False]]
    assert (one["flux0"], nine["flux0"]) == ("1.0", "9.0")
    assert not list((tmp_path / "store").rglob("*.json"))
    multi = _child(tmp_path, "multi", "multidev", XLA_FLAGS="--xla_force_host_platform_device_count=2")
    assert multi["key"] is None and multi["why"] == "device-assignment"


def test_switch_names_are_cheap_key_inert(tmp_path, monkeypatch):
    advance_like = _make_aot_advance_like()
    namelist, carry = _aot_namelist(), _aot_carry(1)
    clock_base = dt.build_clock_base(namelist)
    off = _cheap_key_for_variant(advance_like, carry, namelist, clock_base)
    env_hash = ck.global_trace_env_hash()
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setenv(store.SEEDS_ENV, str(tmp_path / "seed"))
    monkeypatch.setenv(store.AUTOTUNE_ENV, "1")
    assert ck.global_trace_env_hash() == env_hash
    assert _cheap_key_for_variant(advance_like, carry, namelist, clock_base) == off
    assert store.enabled()
    monkeypatch.setenv(store.STORE_ENV, "0")
    monkeypatch.delenv(store.SEEDS_ENV)
    assert not store.enabled() and store.store_roots() == (None, ())


def test_unusable_store_root_is_an_unset_store(monkeypatch, tmp_path):
    """Critic 5: a root that cannot be parsed must not abort initialization."""
    from tests.test_init_kernel_cache import _kernel
    from gpuwrf.integration.init_kernels import InitKernelCache

    monkeypatch.setenv(store.STORE_ENV, "~no_such_user_v50_detab/store")
    monkeypatch.setenv(store.SEEDS_ENV, f"relative/seed{os.pathsep}~no_such_user_v50_detab/seed")
    assert store.store_roots() == (None, ()) and not store.enabled()
    assert len(store.CONFIG_ERRORS) == 3 and "no_such_user_v50_detab" in store.CONFIG_ERRORS[0]
    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    assert store.prepare("d01", lowered, "f" * 64) == (None, None)
    assert not aot._content_store_enabled() and not store.autotune_seed_enabled()
    _kernel.clear_cache()
    cache = InitKernelCache(tmp_path / "aot")
    x = jnp.arange(4.0)
    out = cache(_kernel, x)
    np.testing.assert_array_equal(out["flux"], x + x)
    assert cache.events[-1]["status"]["aot_written"] and "content_store" not in cache.events[-1]["status"]
    assert store.report()["config_errors"] == store.CONFIG_ERRORS

    def broken(*a, **k):
        raise RuntimeError("store exploded")

    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(store, "key_or_refusal", broken)
    key, status = store.prepare("d01", lowered, "f" * 64)
    assert key is None and status["state"].startswith("error:RuntimeError")


# --------------------------------------------------------------------------- #
# Store entries
# --------------------------------------------------------------------------- #
def _dropping_fn(c):
    return {"y": c["a"] + c["b"], "z": c["a"] + c["a"]}


def _dropping_args(n_dead=3):
    carry = {"a": jnp.arange(4.0), "b": jnp.ones((4,))}
    for k in range(n_dead):
        carry[f"dead{k}"] = jnp.ones((k + 2,))
    return (carry,)


def _publish(monkeypatch, root, cache_dir, fn=_dropping_fn, args=None, **extra):
    monkeypatch.setenv(store.STORE_ENV, str(root))
    args = _dropping_args(3) if args is None else args
    lowered = jax.jit(fn).lower(*args)
    hlo = aotx.hlo_sha256_from_lowered(lowered)
    key = store.content_key(lowered)
    compiled = lowered.compile()
    status = aot._serialize_domain_blob("d01", compiled, str(cache_dir), hlo_sha256=hlo,
                                        lowered=lowered, content_key=key, **extra)
    assert status["aot_written"] and status["content_store"]["state"] == "published", status
    return args, lowered, hlo, key, compiled, status


def _record_file(root, key):
    return store._record_path(root, key.key)


def _rewrite_record(root, key, **changes):
    """Edit a record the way a careless or stale writer would (no new attestation)."""
    path = _record_file(root, key)
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record, sort_keys=True))
    return record


def _load(cache_dir, **address):
    return aot.load_domain_blob("d01", str(cache_dir), **address)


def test_meta_from_lowering_equals_meta_from_compiled(monkeypatch, tmp_path, _isolate):
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, tmp_path / "store", _isolate)
    _blob, meta = aotx.serialize(compiled, hlo_sha256=hlo)
    record, _path, _root = store.lookup(key)
    own = store.meta_from_lowered(lowered, hlo_sha256=hlo, blob_sha256=record["blob_sha256"],
                                  record=record)
    assert own.kept_var_idx == meta.kept_var_idx and len(own.kept_var_idx) == 2
    assert own.in_avals == meta.in_avals
    assert own.in_tree == meta.in_tree and own.out_tree == meta.out_tree
    assert own.fingerprint == meta.fingerprint and own.hlo_sha256 == meta.hlo_sha256
    assert own.blob_sha256 == meta.blob_sha256 == status["blob_sha256"]


def test_publish_is_first_writer_wins_and_shares_the_verified_inode(monkeypatch, tmp_path, _isolate):
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    record, blob_path, found_root = store.lookup(key)
    assert found_root == root and record["key"] == key.key
    assert record["components"] == key.components and record["attestation"] == store._attestation(record)
    assert record["kept_in_avals"] == [{"shape": [4], "dtype": "float64"}] * 2
    assert blob_path.parent.name == key.key, "a blob lives under the key it implements"
    local_blob, _ = aot._aot_blob_paths("d01", str(_isolate), hlo_sha256=hlo)
    assert local_blob.read_bytes() == blob_path.read_bytes()
    assert local_blob.stat().st_ino == blob_path.stat().st_ino  # same fs: no second copy
    assert status["content_store"]["local_verified"] is True
    before = (blob_path.read_bytes(), _record_file(root, key).read_bytes())
    blob, meta = aotx.serialize(compiled, hlo_sha256=hlo)
    assert store.publish(key, blob, meta)["state"] == "exists-same"
    other = store.publish(key, blob + b"\0", meta)
    assert other["state"] == "exists-differs" and other["blob_sha256"] == record["blob_sha256"]
    assert other["winner"]["data"] == before[0]
    assert (blob_path.read_bytes(), _record_file(root, key).read_bytes()) == before
    assert not [p for p in root.rglob("*") if p.name.endswith((".tmp", ".link"))]


def test_entry_that_appears_during_publish_is_kept(monkeypatch, tmp_path, _isolate):
    """Two publishers pass the existence scan together: the first record stays."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    first_record = _record_file(root, key).read_bytes()
    blob, meta = aotx.serialize(compiled, hlo_sha256=hlo)
    real_read, calls = store._read_record, {"n": 0}

    def late(root_, key_):
        calls["n"] += 1
        return None if calls["n"] == 1 else real_read(root_, key_)  # not visible at the first scan

    monkeypatch.setattr(store, "_read_record", late)
    out = store.publish(key, blob + b"\0", meta)
    assert out["state"] == "exists-differs" and out["blob_sha256"] == status["blob_sha256"], out
    assert _record_file(root, key).read_bytes() == first_record
    target = tmp_path / "plain.bin"
    assert store._write_new(target, b"first") is True and store._write_new(target, b"second") is False
    assert target.read_bytes() == b"first" and [p.name for p in tmp_path.glob("*plain*")] == ["plain.bin"]


@pytest.mark.parametrize("tamper,expect", [
    ("components", "refused:record-key"), ("foreign-key", "refused:record-key"),
    ("schema", "refused:record-schema"), ("truncated-record", "refused:record-unreadable"),
    ("blob-pointer", "refused:record-attestation"), ("contract-unattested", "refused:record-attestation"),
    ("contract-attested", "refused:call-contract"), ("blob-bytes", "refused:blob-integrity"),
    ("blob-size", "refused:blob-integrity"), ("blob-missing", "refused:blob-missing"),
])
def test_tampered_or_stale_entries_are_never_served(monkeypatch, tmp_path, _isolate, tamper, expect):
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    record_path = _record_file(root, key)
    record = json.loads(record_path.read_text())
    blob_path = store._blob_path(root, key.key, record["blob_sha256"])
    consumer = tmp_path / "consumer"
    if tamper == "components":  # a record claiming another environment under this key
        record["components"]["compile_options_sha256"] = "0" * 64
    elif tamper == "foreign-key":  # a valid record of another program copied to this address
        record["key"] = "0" * 64
    elif tamper == "schema":
        record["schema"] = "GPUWRF-AOT-HLOKEY-v1"
    elif tamper == "blob-pointer":  # pointer moved without a new attestation
        record["blob_bytes"] += 1
    elif tamper.startswith("contract"):
        record["kept_in_avals"][0]["dtype"] = "float32"
        if tamper == "contract-attested":
            record["attestation"] = store._attestation(record)
    if tamper == "truncated-record":
        record_path.write_text(record_path.read_text()[:40])
    elif tamper == "blob-bytes":
        data = bytearray(blob_path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        blob_path.unlink()  # break the link to the producer cache before editing
        blob_path.write_bytes(bytes(data))
    elif tamper == "blob-size":
        data = blob_path.read_bytes()
        blob_path.unlink()
        blob_path.write_bytes(data[:-1])
    elif tamper == "blob-missing":
        blob_path.unlink()
    else:
        record_path.write_text(json.dumps(record, sort_keys=True))
    out = store.materialize("d01", key, lowered, hlo_sha256=hlo, cache_dir=str(consumer))
    assert out["state"].startswith(expect), out
    assert store._KEY_STATE[key.key]["state"] == "refused"
    assert store._KEY_STATE[key.key]["blocking"] is (tamper == "contract-attested")
    assert not list(consumer.rglob("*.xlaexec")) and not list(consumer.rglob("*.meta")), \
        "nothing may reach the consumer cache"


def test_other_program_or_environment_misses(monkeypatch, tmp_path, _isolate):
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    consumer = str(tmp_path / "consumer")
    hit = store.materialize("d01", key, lowered, hlo_sha256=hlo, cache_dir=consumer)
    assert hit["state"] == "hit" and hit["executed_blob_sha256"] == status["blob_sha256"]
    other = jax.jit(lambda c: {"y": c["a"] - c["b"], "z": c["a"] + c["a"]}).lower(*args)
    other_key = store.content_key(other)
    assert other_key.key != key.key
    assert store.materialize("d01", other_key, other, hlo_sha256="f" * 64, cache_dir=consumer)["state"] == "miss"
    flagged = store.content_key(lowered, compile_options={"xla_cpu_enable_fast_math": True})
    assert flagged.program_sha256 == key.program_sha256 and flagged.key != key.key
    assert store.lookup(flagged) is None


def test_redirected_seed_record_is_not_served_and_never_reaches_the_cheap_key(monkeypatch, tmp_path, _isolate):
    """Critic 2: a seed record pointed at a valid same-ABI blob of ANOTHER program."""
    root = tmp_path / "store"
    x = jnp.arange(4.0)
    plus1, plus9 = (lambda v: {"y": v + ONES}), (lambda v: {"y": v + NINES})
    _a, low1, hlo1, key1, _c1, st1 = _publish(monkeypatch, root, tmp_path / "p1", plus1, (x,))
    _b, _low9, _hlo9, key9, _c9, st9 = _publish(monkeypatch, root, tmp_path / "p9", plus9, (x,))
    blob9 = store._blob_path(root, key9.key, st9["blob_sha256"]).read_bytes()
    monkeypatch.delenv(store.STORE_ENV)
    monkeypatch.setenv(store.SEEDS_ENV, str(root))

    def consume(label):
        """prepare -> (refused) -> compile -> serialize under a cheap key -> next-process load."""
        store._KEY_STATE.clear()
        cache = tmp_path / label
        key, lookup = store.prepare("d01", low1, hlo1, cache_dir=str(cache))
        assert key.key == key1.key
        ser = aot._serialize_domain_blob("d01", low1.compile(), str(cache), hlo_sha256=hlo1,
                                         lowered=low1, cheap_key="c" * 64, key_schema=ck.KEY_SCHEMA,
                                         content_key=key)
        assert store.take_winner(key) is None, "a refused entry must not be handed out as a winner"
        loaded = _load(cache, cheap_key="c" * 64)
        return lookup, ser, float(np.asarray(loaded(x)["y"])[0])

    # (a) the critic's edit: digest and length of the x+1 record now name the x+9 blob.
    _rewrite_record(root, key1, blob_sha256=st9["blob_sha256"], blob_bytes=len(blob9))
    lookup, ser, value = consume("redirected")
    assert lookup["state"] == "refused:record-attestation", lookup
    assert ser["content_store"]["state"] == "read-only" and ser["blob_sha256"] != st9["blob_sha256"]
    assert value == 1.0

    # (b) a pointer with a NEW attestation still cannot leave the key's own blob directory:
    #     the x+9 bytes are not stored under the x+1 key.
    forged = _rewrite_record(root, key1)
    forged["attestation"] = store._attestation(forged)
    _record_file(root, key1).write_text(json.dumps(forged, sort_keys=True))
    lookup, ser, value = consume("forged_pointer")
    assert lookup["state"].startswith("refused:blob-missing") and value == 1.0

    # (c) a valid, attested entry whose call contract differs is refused, and the refusal
    #     survives the fallback compile: the own binary, not the refused one, gets the cheap key.
    target = store._blob_path(root, key1.key, st9["blob_sha256"])
    target.write_bytes(blob9)
    forged["kept_in_avals"] = [{"shape": [4], "dtype": "float32"}]
    forged["attestation"] = store._attestation(forged)
    _record_file(root, key1).write_text(json.dumps(forged, sort_keys=True))
    lookup, ser, value = consume("contract")
    assert lookup["state"] == "refused:call-contract", lookup
    assert ser["content_store"]["state"] == "exists-refused", ser["content_store"]
    assert ser["content_store"]["shared"] is False and "adopted" not in ser["content_store"]
    assert ser["blob_sha256"] != st9["blob_sha256"]
    assert value == 1.0, "the refused x+9 blob was installed under the cheap key"
    assert st1["blob_sha256"] != st9["blob_sha256"]


def test_corrupt_store_blob_is_repaired_not_relinked(monkeypatch, tmp_path, _isolate):
    """Critic 3: `exists-same` must never hard-link a corrupt blob back into the cache."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    blob_path = store._blob_path(root, key.key, status["blob_sha256"])
    good = blob_path.read_bytes()
    corrupt = bytearray(good)
    corrupt[len(corrupt) // 2] ^= 0xFF
    blob_path.unlink()
    blob_path.write_bytes(bytes(corrupt))  # one flipped byte, same length

    consumer = tmp_path / "consumer"
    key2, lookup = store.prepare("d01", lowered, hlo, cache_dir=str(consumer))
    assert lookup["state"] == "refused:blob-integrity"
    # The same executable serializes to the recorded bytes again (a second cold compile would not).
    ser = aot._serialize_domain_blob("d01", compiled, str(consumer), hlo_sha256=hlo,
                                     lowered=lowered, cheap_key="c" * 64, key_schema=ck.KEY_SCHEMA,
                                     content_key=key2)
    assert ser["content_store"]["state"] == "repaired" and ser["content_store"]["local_verified"]
    assert blob_path.read_bytes() == good, "the store blob was not repaired"
    for address in ({"hlo_sha256": hlo}, {"cheap_key": "c" * 64}):
        local, _meta = aot._aot_blob_paths("d01", str(consumer), **address)
        assert hashlib.sha256(local.read_bytes()).hexdigest() == status["blob_sha256"]
        assert _leaves(_load(consumer, **address)(*args)) == _leaves(_dropping_fn(*args))

    # A corrupt entry that these bytes cannot repair is set aside and replaced by a valid one.
    blob_path.unlink()
    blob_path.write_bytes(bytes(corrupt))
    blob, meta = aotx.serialize(compiled, hlo_sha256=hlo)
    other = blob + b"\0"
    out = store.publish(key, other, meta)
    assert out["state"] == "published" and out["blob_sha256"] == hashlib.sha256(other).hexdigest()
    assert any("set aside" in note for note in out["notes"])
    assert len(list(_record_file(root, key).parent.glob("*.invalid.*"))) == 1
    assert store.lookup(key)[0]["blob_sha256"] == out["blob_sha256"]


def test_threaded_job_reports_ready_only_for_a_verified_alias(monkeypatch, tmp_path, _isolate):
    """Critic 3: a truncated local blob with intact metadata must not be reported as ready."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    with private_jax_cache(monkeypatch, tmp_path / "cache_b") as cache_b:
        assert store.materialize("d01", key, lowered, hlo_sha256=hlo)["state"] == "hit"
        local_blob, local_meta = aot._aot_blob_paths("d01", str(cache_b), hlo_sha256=hlo)
        data = local_blob.read_bytes()
        local_blob.unlink()
        local_blob.write_bytes(data[: len(data) // 2])  # truncated, metadata intact
        assert pickle.loads(local_meta.read_bytes()).blob_sha256 == status["blob_sha256"]

        content_key, ready = aot._content_store_job("d01", "b" * 64, lowered, None)
        assert ready is not None and ready["status"]["content_store_lookup"]["state"] == "hit"
        for address in ({"hlo_sha256": hlo}, {"cheap_key": "b" * 64}):
            path, _ = aot._aot_blob_paths("d01", str(cache_b), **address)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == status["blob_sha256"]
        assert _leaves(_load(cache_b, cheap_key="b" * 64)(*args)) == _leaves(_dropping_fn(*args))

        # The alias itself is checked: a link that does not hold the verified bytes is not ready.
        real_alias = aot.alias_cheap_key_to_hlo

        def bad_alias(name, cheap_key, hlo_sha256, cache_dir=None):
            out = real_alias(name, cheap_key, hlo_sha256, cache_dir)
            blob_file, _ = aot._aot_blob_paths(name, cache_dir, cheap_key=cheap_key)
            payload = blob_file.read_bytes()
            blob_file.unlink()
            blob_file.write_bytes(payload[:-7])
            return out

        monkeypatch.setattr(aot, "alias_cheap_key_to_hlo", bad_alias)
        content_key, ready = aot._content_store_job("d01", "e" * 64, lowered, None)
        assert content_key is not None and ready is None
        assert not aot._aot_blob_paths("d01", str(cache_b), cheap_key="e" * 64)[0].exists()


def test_seed_root_is_read_only(monkeypatch, tmp_path, _isolate):
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    blob, meta = aotx.serialize(compiled, hlo_sha256=hlo)
    record, blob_path, _ = store.lookup(key)
    monkeypatch.delenv(store.STORE_ENV)
    monkeypatch.setenv(store.SEEDS_ENV, f"{tmp_path / 'absent'}{os.pathsep}{root}")
    snapshot = {p: (p.stat().st_ino, p.stat().st_nlink, p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}
    assert store.publish(key, blob + b"\0", meta)["state"] == "exists-differs"
    other = jax.jit(lambda c: {"y": c["a"] - c["b"]}).lower(*args)
    assert store.publish(store.content_key(other), blob, meta)["state"] == "read-only"
    consumer = tmp_path / "consumer"
    out = store.materialize("d01", key, lowered, hlo_sha256=hlo, cache_dir=str(consumer))
    assert out["state"] == "hit" and out["placed"] == "copy"
    local = next(consumer.rglob("*.xlaexec"))
    assert local.stat().st_ino != blob_path.stat().st_ino
    assert snapshot == {p: (p.stat().st_ino, p.stat().st_nlink, p.stat().st_mtime_ns)
                        for p in root.rglob("*") if p.is_file()}
    # An invalid seed entry is skipped, never repaired or moved.
    _rewrite_record(root, key, blob_bytes=1)
    store._KEY_STATE.clear()
    assert store.publish(key, blob, meta)["state"] == "read-only"
    assert snapshot.keys() == {p for p in root.rglob("*") if p.is_file()}


def test_installed_blob_is_the_hashed_inode_or_a_copy_of_the_hashed_bytes(monkeypatch, tmp_path, _isolate):
    """Critic 3: the path may be replaced between hashing and linking."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    record, blob_path, _ = store.lookup(key)
    good = blob_path.read_bytes()
    real_link = os.link

    def swapping_link(source, destination, *a, **k):
        if str(source) == str(blob_path):  # the store path is swapped after it was hashed
            evil = blob_path.with_suffix(".evil")
            evil.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
            os.replace(evil, blob_path)
        return real_link(source, destination, *a, **k)

    monkeypatch.setattr(os, "link", swapping_link)
    destination = tmp_path / "installed" / "blob.xlaexec"
    out = store._install(blob_path, record, destination, link=True)
    assert out["placed"] == "copy", "a link to an inode that was not hashed was kept"
    assert destination.read_bytes() == good


# --------------------------------------------------------------------------- #
# Through the real miss paths
# --------------------------------------------------------------------------- #
class _CountingAdvance:
    """``_advance_chunk_fori`` stand-in that counts lowerings, compiles and own-binary calls."""

    def __init__(self, fn, *, forbid_own_call=False):
        self.fn, self.lowerings, self.compiles, self.own_calls = fn, 0, 0, 0
        self.forbid_own_call = forbid_own_call

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def lower(self, *args, **kwargs):
        self.lowerings += 1
        outer = self
        lowered = self.fn.lower(*args, **kwargs)

        class Compiled:
            def __init__(self, compiled):
                self._compiled = compiled

            def __getattr__(self, name):
                return getattr(self._compiled, name)

            def __call__(self, *a, **k):
                outer.own_calls += 1
                assert not outer.forbid_own_call, "the losing build executed its own binary"
                return self._compiled(*a, **k)

        class Proxy:
            def __getattr__(self, name):
                return getattr(lowered, name)

            def compile(self, *a, **k):
                outer.compiles += 1
                return Compiled(lowered.compile(*a, **k))

        return Proxy()


_ADVANCE = """
    import jax
    import jax.numpy as jnp

    @jax.jit
    def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
        return {
            "state": carry["state"] + start,
            "extra": carry["extra"],
            "seen": (n_steps, cadence, namelist.scale, clock_base.rad_julian,
                     jnp.asarray(namelist.eta_levels[0], dtype=jnp.float64)),
        }
"""


def _run_build(monkeypatch, tmp_path, label, cheap_key, *, pad, calls=1, forbid_own_call=False,
               leading=1):
    """One 'build': own cache root, cheap key and source location."""
    module = _module(tmp_path, "advance", _ADVANCE, pad=pad, sub=label)
    counting = _CountingAdvance(module.advance_like, forbid_own_call=forbid_own_call)
    monkeypatch.setattr(ck, "cheap_key", lambda *a, **k: cheap_key)
    monkeypatch.setattr(dt, "_advance_chunk_fori", counting)
    namelist, carry = _aot_namelist(), _aot_carry(leading)
    outs, reports = [], []
    with private_jax_cache(monkeypatch, tmp_path / f"cache_{label}") as cache_dir:
        for _process in range(calls):  # each factory models a fresh process
            advance = dt._operational_advance_factory(_fake_tree(namelist))
            outs.append(advance("d01", carry, start_step=2, n_steps=5))
            reports.append(dt.nested_aot_report())
        blobs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in Path(cache_dir).rglob("*.xlaexec")}
    return outs, counting, reports if calls > 1 else reports[0], blobs


def test_two_builds_resolve_to_one_binary_through_the_eager_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    out_a, count_a, report_a, blobs_a = _run_build(monkeypatch, tmp_path, "release", "a" * 64, pad=0)
    assert (count_a.lowerings, count_a.compiles) == (1, 1)
    assert report_a["content_store"]["last_lookup"]["d01"]["state"] == "miss"
    published = report_a["domains"]["d01"]["content_store"]
    assert published["state"] == "published" and published["shared"] and published["local_verified"]

    out_b, count_b, (report_b, report_b2), blobs_b = _run_build(
        monkeypatch, tmp_path, "research", "b" * 64, pad=23, calls=2)
    assert count_b.compiles == 0, "same program was compiled again"
    assert count_b.lowerings == 1, "the second process must load by cheap key without lowering"
    hit = report_b["content_store"]["last_lookup"]["d01"]
    assert hit["state"] == "hit" and hit["key"] == report_a["content_store"]["last_lookup"]["d01"]["key"]
    assert hit["blob_sha256"] == published["blob_sha256"]
    assert report_b["domains"]["d01"]["cheap_key_alias"]["aliased"] is True
    assert report_b2["domains"]["d01"]["address"] == "cheap_key"
    assert "content_store" not in report_b2, "a cheap-key hit must not touch the store"
    assert set(blobs_b.values()) == set(blobs_a.values()) and len(set(blobs_a.values())) == 1
    assert f"k_{'b' * 64}.xlaexec" in blobs_b
    for out in out_b:
        assert _leaves(out) == _leaves(out_a[0])

    # Critic 6: the process log attests every variant with the binary it runs and the backend.
    events = store.report()["events"]
    assert [e["state"] for e in events] == ["lookup:miss", "publish:published", "lookup:hit"]
    assert {e["executed_blob_sha256"] for e in events[1:]} == {published["blob_sha256"]}
    assert all(e["backend"] == "cpu:cpu" and e["name"] == "d01" for e in events)
    _out, _count, report_c, _blobs = _run_build(monkeypatch, tmp_path, "variant", "d" * 64, pad=0, leading=2)
    events = store.report()["events"]
    assert len(events) == 5 and events[-1]["state"] == "publish:published"
    assert events[-1]["key"] != events[0]["key"], "a second shape variant of d01 has its own events"
    assert report_c["content_store"]["events"] == events


def test_store_off_is_the_previous_behaviour(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")

    def explode(*a, **k):
        raise AssertionError("content key computed with the store off")

    for name in ("key_or_refusal", "content_key", "canonical_program_sha256", "effective_compile_options"):
        monkeypatch.setattr(store, name, explode)
    out_a, count_a, report_a, blobs_a = _run_build(monkeypatch, tmp_path, "release", "a" * 64, pad=0)
    out_b, count_b, report_b, blobs_b = _run_build(monkeypatch, tmp_path, "research", "b" * 64, pad=23)
    assert (count_a.compiles, count_b.compiles) == (1, 1)  # separate roots: each build compiles
    assert (count_a.own_calls, count_b.own_calls) == (1, 1)
    assert "content_store" not in report_a and "content_store" not in report_b
    assert "content_store" not in report_a["domains"]["d01"]
    assert sorted(report_a) == ["cache_hit_count", "domains", "enabled", "load_count",
                                "load_wall_seconds", "verify"]
    assert len(blobs_a) == 2 and len(blobs_b) == 2  # exact-HLO address + cheap-key alias only
    assert not (tmp_path / "store").exists() and store.report() == {"events": [], "config_errors": [], "init_kernels": []}
    assert _leaves(out_a[0]) == _leaves(out_b[0])


def _other_bytes_serialize(monkeypatch):
    """Make this process's own compile serialize to other bytes, as a second cold
    compile of the same program does (E99)."""
    real_serialize = aotx.serialize

    def other_bytes(*a, **k):
        blob, meta = real_serialize(*a, **k)
        blob = blob + b"\0" * 4
        return blob, dataclasses.replace(meta, blob_sha256=aotx.blob_sha256(blob))

    monkeypatch.setattr(aotx, "serialize", other_bytes)


def _concurrent_miss(monkeypatch):
    """The lookup ran before the other build published: it saw nothing."""
    monkeypatch.setattr(store, "materialize", lambda name, key, lowered, **k: {
        "state": "miss", "key": key.key, "program_sha256": key.program_sha256})


def test_build_that_loses_the_publication_runs_the_winner(monkeypatch, tmp_path):
    """Critic 4: two concurrent misses must not run two binaries."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    out_a, _count_a, report_a, blobs_a = _run_build(monkeypatch, tmp_path, "winner", "a" * 64, pad=0)
    winner_sha = report_a["domains"]["d01"]["content_store"]["blob_sha256"]

    _concurrent_miss(monkeypatch)
    _other_bytes_serialize(monkeypatch)  # the loser's own cold compile is another binary
    out_b, count_b, report_b, blobs_b = _run_build(
        monkeypatch, tmp_path, "loser", "b" * 64, pad=5, forbid_own_call=True)
    assert count_b.compiles == 1 and count_b.own_calls == 0
    lost = report_b["domains"]["d01"]["content_store"]
    assert lost["state"] == "exists-differs" and lost["adopted"] is True and lost["shared"] is True
    assert lost["executed_blob_sha256"] == winner_sha != lost["own_blob_sha256"]
    assert set(blobs_b.values()) == {winner_sha}, "the local addresses must hold the winner"
    assert _leaves(out_b[0]) == _leaves(out_a[0])
    assert store.report()["events"][-1]["executed_blob_sha256"] == winner_sha
    assert not store._WINNER_CALLS, "the winner executable was handed over exactly once"

    # A winner that cannot be loaded here is refused: the own compile runs and is flagged unshared.
    store._KEY_STATE.clear()
    monkeypatch.setattr(aotx, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no load")))
    out_c, count_c, report_c, blobs_c = _run_build(monkeypatch, tmp_path, "unloadable", "e" * 64, pad=9)
    kept = report_c["domains"]["d01"]["content_store"]
    assert kept["state"] == "exists-refused" and kept["adopted"] is False and kept["shared"] is False
    assert count_c.own_calls == 1 and kept["executed_blob_sha256"] == kept["own_blob_sha256"]
    assert winner_sha not in blobs_c.values()


def test_stored_binary_that_failed_here_is_not_reinstalled(monkeypatch, tmp_path, _isolate):
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    record, blob_path, _ = store.lookup(key)
    consumer = tmp_path / "consumer"
    assert store.materialize("d01", key, lowered, hlo_sha256=hlo, cache_dir=str(consumer))["state"] == "hit"
    # The placed executable did not run (modelled: the caller compiled anyway) and the
    # fallback compile produced other bytes.
    _other_bytes_serialize(monkeypatch)
    out = aot._serialize_domain_blob("d01", compiled, str(consumer), hlo_sha256=hlo,
                                     lowered=lowered, content_key=key)
    assert out["content_store"]["state"] == "exists-refused" and not out["content_store"]["shared"]
    assert "adopted" not in out["content_store"] and store.take_winner(key) is None
    assert out["blob_sha256"] != record["blob_sha256"]
    local_blob, _ = aot._aot_blob_paths("d01", str(consumer), hlo_sha256=hlo)
    assert aotx.blob_sha256(local_blob.read_bytes()) == out["blob_sha256"]
    assert store.lookup(key)[0]["blob_sha256"] == record["blob_sha256"]


def test_winner_with_another_call_contract_is_not_adopted(monkeypatch, tmp_path, _isolate):
    """A publisher that never looked the entry up still checks it before running it."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    record = json.loads(_record_file(root, key).read_text())
    record["kept_in_avals"][0]["dtype"] = "float32"
    record["attestation"] = store._attestation(record)
    _record_file(root, key).write_text(json.dumps(record, sort_keys=True))
    _other_bytes_serialize(monkeypatch)
    out = aot._serialize_domain_blob("d01", compiled, str(tmp_path / "consumer"), hlo_sha256=hlo,
                                     lowered=lowered, content_key=key)
    assert out["content_store"]["state"] == "exists-refused" and out["content_store"]["adopted"] is False
    assert "call-contract" in out["content_store"]["adopt_error"]
    assert store.take_winner(key) is None and store._KEY_STATE[key.key]["state"] == "refused"
    assert out["aot_written"] and out["blob_sha256"] != record["blob_sha256"]


def test_serialize_reports_written_only_for_a_verified_install(monkeypatch, tmp_path, _isolate):
    """Critic 3: the cheap-key address is a link made after hashing; hash what is installed."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    real = aot._atomic_link_or_copy

    def torn(source, destination, data):
        mode = real(source, destination, data)
        payload = destination.read_bytes()
        destination.unlink()
        destination.write_bytes(payload[:-3])
        return mode

    monkeypatch.setattr(aot, "_atomic_link_or_copy", torn)
    consumer = tmp_path / "consumer"
    out = aot._serialize_domain_blob("d01", compiled, str(consumer), hlo_sha256=hlo, lowered=lowered,
                                     cheap_key="c" * 64, key_schema=ck.KEY_SCHEMA, content_key=key)
    assert out["content_store"]["local_verified"] is False and out["aot_written"] is False
    assert "verification" in out["aot_error"]
    assert not aot._aot_blob_paths("d01", str(consumer), cheap_key="c" * 64)[0].exists()
    assert _load(consumer, cheap_key="c" * 64) is None
    assert _leaves(_load(consumer, hlo_sha256=hlo)(*args)) == _leaves(_dropping_fn(*args))


def test_different_constant_never_shares_a_binary(monkeypatch, tmp_path):
    """Two builds that differ in one baked constant keep separate executables."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    source = """
        import jax
        import jax.numpy as jnp
        import numpy as np

        TABLE = np.linspace(0.0, 1.0, 2048)

        @jax.jit
        def advance_like(carry, namelist, start, clock_base, *, n_steps, cadence):
            bias = jnp.asarray(TABLE)[{index}]
            return {{"state": carry["state"], "extra": {{"soil": carry["extra"]["soil"] + bias}}}}
    """
    results = {}
    for label, edit in (("base", None), ("mutant", 1000)):
        module = _module(tmp_path, "advance", source.format(index=1000), sub=label)
        if edit is not None:
            module.TABLE = module.TABLE.copy()
            module.TABLE[edit] += 1e-9
        counting = _CountingAdvance(module.advance_like)
        monkeypatch.setattr(ck, "cheap_key", lambda *a, _l=label, **k: _l.ljust(64, "0"))
        monkeypatch.setattr(dt, "_advance_chunk_fori", counting)
        namelist, carry = _aot_namelist(), _aot_carry(1)
        with private_jax_cache(monkeypatch, tmp_path / f"cache_{label}"):
            advance = dt._operational_advance_factory(_fake_tree(namelist))
            out = advance("d01", carry, start_step=2, n_steps=5)
            results[label] = (np.asarray(out["extra"]["soil"]), counting.compiles,
                              dt.nested_aot_report()["content_store"]["last_lookup"]["d01"])
    assert results["base"][1] == 1 and results["mutant"][1] == 1
    assert results["mutant"][2]["state"] == "miss"
    assert results["base"][2]["key"] != results["mutant"][2]["key"]
    assert not np.array_equal(results["base"][0], results["mutant"][0])
    assert len(list((tmp_path / "store").rglob("*.json"))) == 2


def test_two_builds_resolve_to_one_binary_through_the_fused_path(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    parent_nl, child_nl = _aot_namelist(), _aot_namelist()
    parent = _FusedCarry(jnp.asarray([1.0], dtype=jnp.float64))
    child = _FusedCarry(jnp.asarray([2.0], dtype=jnp.float64))

    def fake_advance(carry, namelist, start, clock_base, *, n_steps, cadence):
        del clock_base
        inc = jnp.asarray(start + n_steps + cadence, dtype=carry.state.dtype)
        return carry.replace(state=carry.state + inc + jnp.asarray(namelist.scale, dtype=carry.state.dtype))

    monkeypatch.setattr(dt, "_advance_chunk", fake_advance)
    monkeypatch.setattr(dt, "build_child_boundary_package",
                        lambda c, p, w, *, bdy_width: c + p + jnp.sum(w) + float(bdy_width))
    active_key = ["fused-a" + "0" * 57]
    monkeypatch.setattr(dt, "_fused_cascade_cheap_key", lambda *a, **k: active_key[0])
    real_serialize, serializes = aot._serialize_domain_blob, {"n": 0}

    def counting_serialize(*args, **kwargs):
        serializes["n"] += 1
        return real_serialize(*args, **kwargs)

    monkeypatch.setattr(aot, "_serialize_domain_blob", counting_serialize)

    def run():
        program = dt._build_fused_cascade_program(
            parent_name="d02", parent_namelist=parent_nl, parent_cadence=7,
            child_names=("d03",), child_namelists=(child_nl,),
            child_weights=(jnp.asarray([1.0], dtype=jnp.float64),),
            child_bdy_widths=(5,), child_ratios=(3,), child_cadences=(7,))
        out_parent, out_children = program(parent, (child,), 4, (12,))
        return np.asarray(out_parent.state).tobytes(), np.asarray(out_children[0].state).tobytes()

    with private_jax_cache(monkeypatch, tmp_path / "cache_release"):
        first = run()
        published = dt.nested_aot_report()["domains"]["fused/d02"]["content_store"]
    assert serializes["n"] == 1 and published["state"] == "published"

    active_key[0] = "fused-b" + "0" * 57
    with private_jax_cache(monkeypatch, tmp_path / "cache_research") as cache_dir:
        second = run()
        report = dt.nested_aot_report()
        third = run()  # next process: cheap key, no lowering
        assert dt.nested_aot_report()["domains"]["fused/d02"]["address"] == "cheap_key"
        local = {hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(cache_dir).rglob("*.xlaexec")}
    assert serializes["n"] == 1, "same fused program was compiled again"
    assert report["content_store"]["last_lookup"]["fused/d02"]["state"] == "hit"
    assert report["domains"]["fused/d02"]["cheap_key_alias"]["aliased"] is True
    assert local == {published["blob_sha256"]}
    assert first == second == third

    # Critic 4 on the fused path: a concurrent miss with another own binary runs the winner.
    active_key[0] = "fused-c" + "0" * 57
    store._KEY_STATE.clear()
    _concurrent_miss(monkeypatch)
    _other_bytes_serialize(monkeypatch)
    with private_jax_cache(monkeypatch, tmp_path / "cache_loser") as cache_dir:
        fourth = run()
        lost = dt.nested_aot_report()["domains"]["fused/d02"]["content_store"]
        local = {hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(cache_dir).rglob("*.xlaexec")}
    assert lost["state"] == "exists-differs" and lost["adopted"] is True
    assert lost["executed_blob_sha256"] == published["blob_sha256"] and local == {published["blob_sha256"]}
    assert fourth == first and not store._WINNER_CALLS


# --------------------------------------------------------------------------- #
# Init kernels, threaded prewarm, autotune pool
# --------------------------------------------------------------------------- #
def test_init_kernel_cache_shares_one_binary_between_cache_roots(monkeypatch, tmp_path):
    from tests.test_init_kernel_cache import _kernel
    from gpuwrf.integration.init_kernels import InitKernelCache

    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    x, topo = jnp.arange(4.0), jnp.ones((4,))
    kwargs = dict(debug=False, with_clear_sky=True, column_tile_cols=4, topography=topo)
    _kernel.clear_cache()
    first = InitKernelCache(tmp_path / "release")
    expected = first(_kernel, x, **kwargs)
    status = first.events[-1]["status"]
    assert not first.events[-1]["hit"] and status["content_store"]["state"] == "published"
    assert status["content_store_lookup"]["state"] == "miss"

    _kernel.clear_cache()
    compiles, ran = {"n": 0}, {"own": 0}
    real_lower = _kernel.lower

    def counting_lower(*a, **k):
        lowered = real_lower(*a, **k)

        class Own:
            def __init__(self, compiled):
                self._compiled = compiled

            def __getattr__(self, name):
                return getattr(self._compiled, name)

            def __call__(self, *ca, **ck_):
                ran["own"] += 1
                return self._compiled(*ca, **ck_)

        class Proxy:
            def __getattr__(self, name):
                return getattr(lowered, name)

            def compile(self, *ca, **ck_):
                compiles["n"] += 1
                return Own(lowered.compile(*ca, **ck_))

        return Proxy()

    monkeypatch.setattr(_kernel, "lower", counting_lower, raising=False)
    second = InitKernelCache(tmp_path / "research")
    loaded = second(_kernel, x, **kwargs)
    event = second.events[-1]
    assert event["hit"] and compiles["n"] == 0, event
    assert event["status"]["content_store_lookup"]["state"] == "hit"
    assert event["status"]["content_store_lookup"]["blob_sha256"] == status["blob_sha256"]
    for name in expected:
        assert np.asarray(loaded[name]).tobytes() == np.asarray(expected[name]).tobytes()
    # Static keywords stay out of the call contract derived from the lowering.
    changed = second(_kernel, x + 2, **kwargs)
    np.testing.assert_array_equal(changed["flux"], x + 2 + topo)
    # Critic 6: init-kernel lookups and publications are in the same process log.
    events = [e for e in store.report()["events"] if e["name"] == "init__kernel"]
    assert [e["state"] for e in events] == ["lookup:miss", "publish:published", "lookup:hit",
                                            "lookup:hit-already-local"]  # one per lowering
    assert {e["executed_blob_sha256"] for e in events[1:]} == {status["blob_sha256"]}

    # Critic 4 for init kernels: a compile that loses the publication is not what runs.
    _kernel.clear_cache()
    store._KEY_STATE.clear()
    _concurrent_miss(monkeypatch)
    _other_bytes_serialize(monkeypatch)
    third = InitKernelCache(tmp_path / "loser")
    out = third(_kernel, x, **kwargs)
    lost = third.events[-1]["status"]["content_store"]
    assert compiles["n"] == 1 and ran["own"] == 0, "the losing init compile was executed"
    assert lost["state"] == "exists-differs" and lost["adopted"]
    assert lost["executed_blob_sha256"] == status["blob_sha256"]
    for name in expected:
        assert np.asarray(out[name]).tobytes() == np.asarray(expected[name]).tobytes()


def test_threaded_prewarm_job_reuses_a_stored_executable(monkeypatch, tmp_path, _isolate):
    """De-fused threaded prewarm: a store hit skips the compile and aliases the cheap key."""
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setattr("gpuwrf.runtime.operational_mode.step_compile_options", lambda base=None: base)
    args = _dropping_args(2)
    compiles = {"n": 0}

    def job(cheap_key):
        def finish(compiled, lowered, content_key=None):
            compiles["n"] += 1
            status = aot._serialize_domain_blob(
                "d01", compiled, None, lowered=lowered, hlo_sha256=aotx.hlo_sha256_from_lowered(lowered),
                cheap_key=cheap_key, key_schema=ck.KEY_SCHEMA,
                **({"content_key": content_key} if content_key is not None else {}))
            return {"name": "d01", "cheap_key": cheap_key, "status": status}

        return {"name": "d01", "lower": lambda: jax.jit(_dropping_fn).lower(*args), "finish": finish,
                "store": functools.partial(aot._content_store_job, "d01", cheap_key)}

    with private_jax_cache(monkeypatch, tmp_path / "cache_a"):
        results, failures = aot._compile_lowered_threads([job("a" * 64)])
    assert not failures and compiles["n"] == 1
    assert results[0]["status"]["content_store"]["state"] == "published"
    with private_jax_cache(monkeypatch, tmp_path / "cache_b") as cache_b:
        results, failures = aot._compile_lowered_threads([job("b" * 64)])
        assert not failures and compiles["n"] == 1, "stored executable was compiled again"
        status = results[0]["status"]
        assert status["content_store_lookup"]["state"] == "hit" and status["cheap_key_alias"]["aliased"]
        call = _load(cache_b, cheap_key="b" * 64)
        assert call is not None
        assert _leaves(call(*args)) == _leaves(_dropping_fn(*args))
    # Store off: the job dict has no hook and the finish callback keeps its old signature.
    monkeypatch.delenv(store.STORE_ENV)
    assert not aot._content_store_enabled()


def test_autotune_pool_copies_missing_decisions_only(monkeypatch, tmp_path, _isolate):
    low = _lower_plain(_module(tmp_path, "plain", _PLAIN))
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    assert not store.autotune_seed_enabled()
    monkeypatch.setenv(store.AUTOTUNE_ENV, "1")
    assert store.autotune_seed_enabled()
    key = store.content_key(low)
    live = store._live_autotune_dir()
    assert live == Path(_isolate) / "xla_gpu_per_fusion_autotune_cache_dir"
    live.mkdir(parents=True, exist_ok=True)
    first, second = "A" * 43 + ".textproto", "B" * 43 + ".textproto"
    (live / first).write_text('other { name: "NATIVE_EMITTER" }')
    (live / "not-a-decision.txt").write_text("x")
    (live / ("C" * 43 + ".textproto")).write_bytes(b"x" * (64 * 1024 + 1))
    exported = store.autotune_export(key)
    pool = Path(exported["pool"])
    assert exported["exported"] == 1 and [p.name for p in pool.iterdir()] == [first]

    # A second cache root with its own, different decision for the same fusion.
    other = tmp_path / "other" / "xla_gpu_per_fusion_autotune_cache_dir"
    other.mkdir(parents=True)
    (other / second).write_text('other { name: "BLOCK_LEVEL_EMITTER" }')
    monkeypatch.setattr(store, "_live_autotune_dir", lambda: other)
    assert store.autotune_import(key)["imported"] == 1
    assert (other / first).read_text() == 'other { name: "NATIVE_EMITTER" }'
    (other / first).write_text('other { name: "BLOCK_LEVEL_EMITTER" }')  # a later local decision
    assert store.autotune_export(key)["exported"] == 1  # only the new name
    assert (pool / first).read_text() == 'other { name: "NATIVE_EMITTER" }', "first decision is kept"
    assert store.autotune_import(key)["imported"] == 0
    # Another compile environment has its own pool; a flags file is scoped by its content.
    flags = tmp_path / "flags.txt"
    flags.write_text("--xla_gpu_autotune_level=0 --xla_dump_to=/tmp/x\n")
    monkeypatch.setenv("XLA_FLAGS", str(flags))
    assert store.canonical_xla_flags() == ["--xla_gpu_autotune_level=0"]
    assert store._autotune_pool(tmp_path / "store", key.components) != pool
    monkeypatch.delenv(store.AUTOTUNE_ENV)
    monkeypatch.delenv("XLA_FLAGS")
    _key, status = store.prepare("d01", low, "f" * 64, cache_dir=str(tmp_path / "c"))
    assert status["state"] == "miss" and "autotune_import" not in status


# --------------------------------------------------------------------------- #
# Delta 1: serialized recovery/publication, evidence for refused and unshared work
# --------------------------------------------------------------------------- #
def _two_other_binaries(compiled, hlo):
    """Two distinct, loadable serializations of one executable (as two cold compiles give)."""
    blob, meta = aotx.serialize(compiled, hlo_sha256=hlo)
    variants = []
    for tag in (1, 2):
        data = blob + aotx._pack(9001, 0, tag)  # an unknown proto field: still loads
        variants.append((data, dataclasses.replace(meta, blob_sha256=aotx.blob_sha256(data))))
    return variants


def _corrupt_store_blob(root, key, status):
    blob_path = store._blob_path(root, key.key, status["blob_sha256"])
    blob_path.unlink()
    blob_path.write_bytes(b"corrupt")
    return blob_path


def test_recovery_cannot_replace_a_valid_winner(monkeypatch, tmp_path, _isolate):
    """Critic Delta 1.1 (recovery_race.py): P2 recovers between P1's failed read and its set-aside."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    (b1, m1), (b2, m2) = _two_other_binaries(compiled, hlo)
    assert aotx.load(b1, m1) is not None and aotx.load(b2, m2) is not None
    _corrupt_store_blob(root, key, status)
    real, competing = store._set_aside, {}

    def interleave(root_, key_, reason):
        monkeypatch.setattr(store, "_set_aside", real)
        competing.update(store.publish(key, b2, m2))  # the other publisher completes its recovery here
        return real(root_, key_, reason)

    monkeypatch.setattr(store, "_set_aside", interleave)
    p1 = store.publish(key, b1, m1)
    assert competing["state"] == "published" and competing["blob_sha256"] == m2.blob_sha256
    assert p1["state"] == "exists-differs" and p1["blob_sha256"] == m2.blob_sha256, p1
    assert p1["winner"]["data"] == b2
    assert store.lookup(key)[0]["blob_sha256"] == m2.blob_sha256, "the valid winner was replaced"
    assert len(list(_record_file(root, key).parent.glob("*.invalid.*"))) == 1  # only the corrupt entry

    # A stale observation alone never renames: the entry is verified again at the rename.
    assert store._set_aside(root, key.key, "blob-integrity seen earlier") is False
    assert store.lookup(key)[0]["blob_sha256"] == m2.blob_sha256


def test_publications_of_one_key_are_serialized_by_a_lock(monkeypatch, tmp_path, _isolate):
    """Delta 1.1: the window between the re-check and the rename is closed by the per-key lock."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    (b1, m1), (b2, m2) = _two_other_binaries(compiled, hlo)
    _corrupt_store_blob(root, key, status)
    real_check = store._still_invalid
    inside, release, results = threading.Event(), threading.Event(), {}

    def paused_check(root_, key_):
        verdict = real_check(root_, key_)
        if threading.current_thread().name == "P1" and not inside.is_set():
            inside.set()            # P1 has re-checked "still invalid" and is about to rename...
            release.wait(timeout=30)
        return verdict

    monkeypatch.setattr(store, "_still_invalid", paused_check)
    p1 = threading.Thread(name="P1", target=lambda: results.update(p1=store.publish(key, b1, m1)))
    p2 = threading.Thread(name="P2", target=lambda: results.update(p2=store.publish(key, b2, m2)))
    p1.start()
    assert inside.wait(timeout=30)
    p2.start()
    p2.join(timeout=1.0)
    assert p2.is_alive(), "the second publisher was not held back while the first one recovers"
    release.set()
    p1.join(timeout=60)
    p2.join(timeout=60)
    states = sorted(results[name]["state"] for name in ("p1", "p2"))
    assert states == ["exists-differs", "published"], results
    assert results["p1"]["state"] == "published" and results["p2"]["blob_sha256"] == m1.blob_sha256
    assert store.lookup(key)[0]["blob_sha256"] == m1.blob_sha256
    assert not store._LOCKS_HELD

    # A lock that cannot be taken leaves the publisher unshared, never unsafe.
    import fcntl

    holder = os.open(store._lock_path(root, key.key), os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        monkeypatch.setattr(store, "_LOCK_TIMEOUT_S", 0.3)
        started = time.monotonic()
        out = store.publish(key, b2, m2)
        assert out["state"].startswith("refused:publication-lock") and time.monotonic() - started < 5
    finally:
        os.close(holder)
    assert store.publish(key, b2, m2)["state"] == "exists-differs"  # the lock is free again


def test_integrity_refusal_does_not_block_a_later_valid_winner(monkeypatch, tmp_path, _isolate):
    """A failure observed at lookup is not carried into the publication: the entry is verified again."""
    root = tmp_path / "store"
    args, lowered, hlo, key, compiled, status = _publish(monkeypatch, root, _isolate)
    (b1, m1), (b2, m2) = _two_other_binaries(compiled, hlo)
    _corrupt_store_blob(root, key, status)
    consumer = tmp_path / "consumer"
    key2, lookup = store.prepare("d01", lowered, hlo, cache_dir=str(consumer))
    assert lookup["state"] == "refused:blob-integrity" and store._KEY_STATE[key.key]["blocking"] is False
    assert store.publish(key, b2, m2)["state"] == "published"  # another process recovers meanwhile
    _other_bytes_serialize(monkeypatch)
    out = aot._serialize_domain_blob("d01", compiled, str(consumer), hlo_sha256=hlo, lowered=lowered,
                                     content_key=key2)
    assert out["content_store"]["state"] == "exists-differs" and out["content_store"]["adopted"] is True
    assert out["content_store"]["executed_blob_sha256"] == m2.blob_sha256 == out["blob_sha256"]
    assert store.take_winner(key2) is not None


def test_refused_and_unshared_work_is_logged(monkeypatch, tmp_path):
    """Critic Delta 1.2: refusals and local exact-HLO hits must appear in the evidence."""
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "1")
    # A cache that already holds this program at its own exact-HLO address (compiled with the store off).
    out_off, _count, _report, blobs_off = _run_build(monkeypatch, tmp_path, "layer", "a" * 64, pad=0)
    assert store.report()["events"] == []
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    out_on, count_on, report_on, _blobs = _run_build(monkeypatch, tmp_path, "layer", "b" * 64, pad=0)
    assert count_on.compiles == 0 and _leaves(out_on[0]) == _leaves(out_off[0])
    events = store.report()["events"]
    assert [(e["name"], e["state"], e["shared"] if "shared" in e else None) for e in events] == [
        ("d01", "lookup:miss", None), ("d01", "local:exact-hlo-hit", False)], events
    assert events[1]["key"] == events[0]["key"] and events[1]["backend"] == "cpu:cpu"

    # A refused lowering is an event too, with no key.
    monkeypatch.setattr(store, "key_or_refusal", lambda *a, **k: (None, "hoisted-constants"))
    lowered = jax.jit(_mul0).lower(jnp.ones((2,), jnp.float32))
    assert store.prepare("fused/d02", lowered, "f" * 64)[1]["state"] == "refused:hoisted-constants"
    last = store.report()["events"][-1]
    assert (last["name"], last["key"], last["state"], last["shared"]) == (
        "fused/d02", None, "lookup:refused:hoisted-constants", False)
    assert last["backend"] == "cpu:cpu"


def test_init_kernel_calls_are_logged_for_the_warm_gate(monkeypatch, tmp_path):
    from tests.test_init_kernel_cache import _kernel
    from gpuwrf.integration.init_kernels import InitKernelCache

    x = jnp.arange(4.0)
    _kernel.clear_cache()
    InitKernelCache(tmp_path / "off")(_kernel, x)
    assert store.report()["init_kernels"] == []  # store off: nothing is recorded
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    _kernel.clear_cache()
    InitKernelCache(tmp_path / "on")(_kernel, x)
    InitKernelCache(tmp_path / "on")(_kernel, x)
    assert store.report()["init_kernels"] == [
        {"kernel": "init__kernel", "hit": False, "lowered": True, "cheap_key": False, "backend": "cpu:cpu"},
        {"kernel": "init__kernel", "hit": True, "lowered": True, "cheap_key": False, "backend": "cpu:cpu"}]
