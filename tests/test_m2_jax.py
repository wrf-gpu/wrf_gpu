from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jax
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "jax"
SCRATCH = ROOT / "data" / "scratch" / "m2-jax"

# The M2 JAX bakeoff runs the pipeline in a venv and asserts default_backend=="gpu"
# with a CudaDevice (RTX 5090). It cannot pass on a CPU-only checkout; it is a
# GPU-benchmark test of a legacy subsystem untouched by the operational pipeline.
# (The committed static-artifact check still runs.)
requires_gpu_toolchain = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="M2 JAX bakeoff requires a JAX GPU (CUDA) backend on the RTX 5090",
)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


@requires_gpu_toolchain
def test_jax_pipeline_artifacts_are_valid() -> None:
    run(["bash", "scripts/m2_run_jax.sh"])
    backend = json.loads((SCRATCH / "jax_backend.json").read_text())
    assert backend["default_backend"] == "gpu"
    assert any("CudaDevice(id=0" in device or device.lower() in {"gpu:0", "cuda:0"} for device in backend["devices"])

    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True

    for name in ("stencil_profile.json", "column_profile.json"):
        profile = json.loads((ARTIFACT_DIR / name).read_text())
        assert profile["backend"] == "jax"
        assert profile["hardware"] == "RTX 5090 32GB"
        assert profile["jax_backend"] == "gpu"
        assert 1 <= profile["kernel_launches"] <= 5
        assert profile["wall_time_s"] >= 0.0
        assert profile["host_device_transfer_bytes"] > 0
        assert isinstance(profile["occupancy_pct"], float)
        assert profile["registers_per_thread"] > 0
        if name == "column_profile.json":
            assert profile["local_memory_bytes"] == 0
        assert profile["achieved_bandwidth_gbps"] >= 0.0
        assert profile["achieved_bandwidth_method"] == "fallback-derived"
        assert profile["profiler_limitation"]
        assert profile["warmup_pattern"]
        assert profile["hlo_kernel_ops"]
        for artifact in profile["artifact_paths"]:
            assert (ROOT / artifact).exists(), artifact


def test_jax_static_artifacts_exist() -> None:
    for name in (
        "stencil_profile.json",
        "column_profile.json",
        "correctness.json",
        "maintainability.md",
        "agent_success.json",
    ):
        assert (ARTIFACT_DIR / name).exists()

    maintainability_words = (ARTIFACT_DIR / "maintainability.md").read_text().split()
    assert len(maintainability_words) <= 300
    agent_success = json.loads((ARTIFACT_DIR / "agent_success.json").read_text())
    assert agent_success["candidate"] == "jax"
    assert agent_success["backend_used"] == "jax"
