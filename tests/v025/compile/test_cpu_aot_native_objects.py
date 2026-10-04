"""Actual cold/cache-hit exports; these are cache integrity tests, not WRF oracles."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_CHILD = r'''
import json, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from gpuwrf.runtime import aot_executable as ax, aot_precompile as ap
spec = json.loads(sys.argv[1])
device = jax.devices()[spec.get("device", 0)]
x = jax.device_put(jnp.arange(8, dtype=jnp.float64), device)
args = ({"x": x, "unused": x + 1},)
@jax.jit
def fn(carry):
    return {"y": jnp.sin(carry["x"]) + carry["x"] * 3.0}
lowered = fn.lower(*args)
compiled = lowered.compile(compiler_options=spec.get("options"), device_assignment=(device,))
reference = np.asarray(compiled(*args)["y"])
raw = bytes(compiled._executable.xla_executable.serialize())
original_options = ax._cpu_compile_options(raw).SerializeAsString()
original_fusions = compiled._executable.xla_executable.hlo_modules()[0].to_string().count(" fusion(")
try:
    ax._check_cpu_native_objects(raw, "cpu")
    raw_complete = True
except ax.AotSerializeError:
    raw_complete = False
if spec["mode"] == "write":
    status = ap._serialize_domain_blob("native", compiled, spec["aot"],
        lowered=lowered, cheap_key="native" + "0" * 58,
        key_schema=__import__("gpuwrf.runtime.aot_cheap_key", fromlist=["KEY_SCHEMA"]).KEY_SCHEMA)
    assert status["aot_written"], status
    stored = Path(status["aot_path"]).read_bytes()
    restored_options = ax._cpu_compile_options(stored)
    original_parsed = ax._cpu_compile_options(raw)
    assert restored_options.env_option_overrides == original_parsed.env_option_overrides
    assert restored_options.device_assignment.serialize() == original_parsed.device_assignment.serialize()
    if not raw_complete:
        assert status["aot_recovery"] == "fresh-cpu-compile", status
        assert status["aot_recovery_compile_seconds"] > 0
        assert status["aot_recovery_seconds"] >= status["aot_recovery_compile_seconds"]
        assert "native objects" in status["aot_recovery_reason"]
        assert status["aot_recovery_compile_options_sha256"] == ax.blob_sha256(original_options)
        assert status["aot_recovery_layouts_verified"] is True
    # Without a lowering, an incomplete export must be refused, not published.
    if not raw_complete:
        try:
            ax.serialize(compiled)
        except ax.AotSerializeError:
            pass
        else:
            raise AssertionError("accepted incomplete CPU export without lowering")
else:
    call, status = ap.load_domain_blob("native", spec["aot"],
        cheap_key="native" + "0" * 58, dev=device, return_status=True)
    assert call is not None, status
    np.testing.assert_array_equal(np.asarray(call(*args)["y"]), reference)
    loaded_fusions = call.loaded_executable.hlo_modules()[0].to_string().count(" fusion(")
    assert loaded_fusions == original_fusions, (loaded_fusions, original_fusions)
    # Re-exporting a loaded CPU executable reproduces the native-object loss.
    broken = bytes(call.loaded_executable.serialize())
    try:
        ax.load(broken, call.meta, dev=device)
    except ax.AotSerializeError as exc:
        assert "native objects" in str(exc)
    else:
        raise AssertionError("accepted incomplete CPU payload on load")
print(json.dumps({"raw_complete": raw_complete, "status": status, "fusions": original_fusions}))
'''


@pytest.mark.parametrize("options,device", [(None, 0), ({"xla_disable_hlo_passes": "fusion"}, 1)],
                         ids=["default", "override-device1"])
def test_cold_and_cache_hit_exports_survive_fresh_process(tmp_path, options, device):
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               GPUWRF_JAX_CACHE="1", GPUWRF_JAX_CACHE_DIR=str(tmp_path / "jit"),
               JAX_COMPILATION_CACHE_DIR=str(tmp_path / "jit"),
               JAX_ENABLE_COMPILATION_CACHE="true",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=2")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
    records = []
    for mode, directory in (("write", "cold"), ("write", "warm"),
                            ("read", "cold"), ("read", "warm")):
        spec = {"mode": mode, "aot": str(tmp_path / directory), "options": options, "device": device}
        proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(spec)],
                              env=env, capture_output=True, text=True, timeout=90)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert records[0]["raw_complete"]
    assert not records[1]["raw_complete"], "cache-hit mutation did not reproduce"
    assert all(r["status"]["loaded"] for r in records[2:])
    if options:
        assert all(r["fusions"] == 4 for r in records), records


def test_truncated_cpu_payload_rejected_before_deserialization():
    from gpuwrf.runtime import aot_executable as ax
    for blob in (b"", b"\x80", b"\x7fshort", b"\x00\x0a\x09x"):
        with pytest.raises(ax.AotSerializeError):
            ax._check_cpu_native_objects(blob, "cpu")


def test_worker_includes_recovery_compile_cost(monkeypatch):
    from gpuwrf.runtime import aot_precompile as ap, aot_cheap_key as ck
    result = ap.PrecompileResult("d01", 2.0, None, False)
    monkeypatch.setattr(ap, "precompile", lambda *a, **k: (object(), result))
    monkeypatch.setattr(ap, "_aot_enabled", lambda: True)
    monkeypatch.setattr(ck, "cheap_key", lambda *a, **k: None)
    monkeypatch.setattr(ap, "_serialize_domain_blob", lambda *a, **k: {
        "aot_written": True, "aot_recovery": "fresh-cpu-compile",
        "aot_recovery_compile_seconds": 0.125,
    })
    spec = ap.DomainCompileSpec("d01", None, None, None, 1, 1)
    measured = ap._compile_one_domain_worker(spec)
    assert measured["error"] is None, measured
    assert measured["compile_seconds"] == 2.125
    assert measured["aot_recovery_compile_seconds"] == 0.125
