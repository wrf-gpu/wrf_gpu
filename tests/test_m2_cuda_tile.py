from __future__ import annotations

import json
import subprocess
from pathlib import Path

import jax
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "cuda_tile"
BENCH = ROOT / "data" / "scratch" / "cuda_tile" / "bench"

# The M2 backend bakeoff compiles a CUDA/sm_120 kernel (nvcc/cmake/cuobjdump) and
# benchmarks it on the RTX 5090. These build+run tests cannot pass on a CPU-only
# checkout without the CUDA toolchain or a GPU; they are GPU-benchmark tests of a
# legacy subsystem untouched by the operational pipeline. (The committed
# static-artifact checks below still run.)
requires_gpu_toolchain = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="M2 cuda_tile bakeoff requires a GPU + CUDA toolchain (nvcc/cmake/cuobjdump)",
)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


@requires_gpu_toolchain
def test_cuda_tile_build_succeeds_and_targets_sm120() -> None:
    run(["bash", "src/gpuwrf/backends/cuda_tile/build.sh"])
    assert BENCH.exists()
    sass = run(["cuobjdump", "--dump-sass", str(BENCH)]).stdout
    assert "arch = sm_120" in sass


@requires_gpu_toolchain
def test_cuda_tile_pipeline_artifacts_are_valid() -> None:
    run(["bash", "scripts/m2_run_cuda_tile.sh"])
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True

    for name in ("stencil_profile.json", "column_profile.json"):
        profile = json.loads((ARTIFACT_DIR / name).read_text())
        assert profile["backend"] == "cuda-tile"
        assert profile["hardware"] == "RTX 5090 32GB"
        assert profile["kernel_launches"] >= 1
        assert profile["wall_time_s"] >= 0.0
        assert profile["host_device_transfer_bytes"] > 0
        assert isinstance(profile["occupancy_pct"], float)
        assert profile["registers_per_thread"] > 0
        assert profile["local_memory_bytes"] == 0
        assert profile["achieved_bandwidth_gbps"] >= 0.0
        assert profile["artifact_paths"]
        for artifact in profile["artifact_paths"]:
            assert (ROOT / artifact).exists(), artifact
