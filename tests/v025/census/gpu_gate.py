"""Run transparency gates on GPU without v025's CPU-only pytest conftest.

Must be launched under with_gpu_lock, pinned to census cores, with timeout.
This is an instrumentation regression check; PROD timings live in bench receipts.
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import time

if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
    raise SystemExit("GPU gate requires scripts/with_gpu_lock.sh --label census")

import jax
import jax.numpy as jnp
import pytest

if jax.devices()[0].platform != "gpu":
    raise SystemExit("GPU gate requires GPU backend")

spec = importlib.util.spec_from_file_location("census_tests", Path(__file__).with_name("test_census.py"))
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
started = time.perf_counter()
with pytest.MonkeyPatch.context() as patch:
    test.test_injected_real_guard_path_is_bit_identical_and_counts_every_site(patch)
test.test_disabled_has_no_leaves_or_device_work_and_uint64_does_not_wrap()
for dtype in (jnp.float32, jnp.float64):
    test.test_runtime_fault_operands_count_on_device_without_uint32_wrap(dtype)
test.test_native_mass_sink_is_resident_cumulative_and_free_when_disabled()
test.test_real_acoustic_loop_counts_executed_trips_and_preserves_fields()
for predicate in (False, True):
    with pytest.MonkeyPatch.context() as patch:
        test.test_radiation_counts_are_inside_executed_branch(patch, predicate)
result = {"tests_passed": 8, "elapsed_s": time.perf_counter() - started,
          "device": str(jax.devices()[0]), "field_gate": "bit-identical",
          "scope": "instrumentation regression; no WRF fidelity or PROD overhead claim"}
root = Path(__file__).resolve().parents[3]
result["source_sha256"] = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in ("src/gpuwrf/runtime/operational_mode.py", "src/gpuwrf/runtime/operational_state.py",
                 "src/gpuwrf/diagnostics/census.py", "tests/v025/census/test_census.py")
}
print(json.dumps(result))
Path(os.environ["CENSUS_GPU_RESULT"]).write_text(json.dumps(result, indent=2) + "\n")
