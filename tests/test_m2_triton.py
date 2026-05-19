from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "triton"
SCRATCH = ROOT / "data" / "scratch" / "m2-triton"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_triton_pipeline_artifacts_are_valid() -> None:
    run(["bash", "scripts/m2_run_triton.sh"])
    backend = json.loads((SCRATCH / "triton_backend.json").read_text())
    assert backend["cuda_available"] is True
    assert backend["torch_cuda"]
    assert backend["triton_version"] == "3.7.0"

    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True

    for name in ("stencil_profile.json", "column_profile.json"):
        profile = json.loads((ARTIFACT_DIR / name).read_text())
        assert profile["backend"] == "triton"
        assert profile["hardware"] == "RTX 5090 32GB"
        assert profile["triton_version"] == "3.7.0"
        assert profile["torch_cuda"]
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
        assert profile["artifact_paths"]
        for artifact in profile["artifact_paths"]:
            path = Path(artifact)
            assert (path if path.is_absolute() else ROOT / path).exists(), artifact


def test_triton_static_artifacts_exist() -> None:
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
    assert agent_success["candidate"] == "triton"
    assert agent_success["backend_used"] == "triton"
