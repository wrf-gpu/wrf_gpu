from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "cuda_tile"
BENCH = ROOT / "data" / "scratch" / "cuda_tile" / "bench"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_cuda_tile_build_succeeds_and_targets_sm120() -> None:
    run(["bash", "src/gpuwrf/backends/cuda_tile/build.sh"])
    assert BENCH.exists()
    sass = run(["cuobjdump", "--dump-sass", str(BENCH)]).stdout
    assert "arch = sm_120" in sass


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
