from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "cupy_or_numba"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_cupy_pipeline_artifacts_are_valid() -> None:
    run(["bash", "scripts/m2_run_cupy.sh"])
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True

    for name in ("stencil_profile.json", "column_profile.json"):
        profile = json.loads((ARTIFACT_DIR / name).read_text())
        assert profile["backend"] == "cupy"
        assert profile["hardware"] == "RTX 5090 32GB"
        assert profile["kernel_launches"] >= 1
        assert profile["kernel_launches"] <= 5
        assert profile["wall_time_s"] >= 0.0
        assert profile["host_device_transfer_bytes"] > 0
        assert isinstance(profile["occupancy_pct"], float)
        assert profile["registers_per_thread"] > 0
        if name == "column_profile.json":
            assert profile["local_memory_bytes"] == 0
        assert profile["achieved_bandwidth_gbps"] >= 0.0
        assert profile["achieved_bandwidth_method"] == "fallback-derived"
        assert profile["profiler_limitation"]
        assert profile["artifact_paths"]
        for artifact in profile["artifact_paths"]:
            assert (ROOT / artifact).exists(), artifact


def test_cupy_static_artifacts_exist() -> None:
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
    assert agent_success["candidate"] == "cupy_or_numba"
    assert agent_success["backend_used"] == "cupy"
