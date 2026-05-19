from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "kokkos"
BENCH = ROOT / "data" / "scratch" / "kokkos" / "bench"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


def test_kokkos_build_succeeds_and_targets_blackwell() -> None:
    run(["bash", "src/gpuwrf/backends/kokkos/build.sh"])
    assert BENCH.exists()
    usage = run([str(BENCH)]).stdout
    assert "usage: bench" in usage
    sass = run(["cuobjdump", "--dump-sass", str(BENCH)]).stdout
    if "arch = sm_120" not in sass:
        config = run([str(BENCH), "config"]).stdout
        assert "runtime_compute_capability=12.0" in config


def test_kokkos_pipeline_artifacts_are_valid() -> None:
    run(["bash", "scripts/m2_run_kokkos.sh"])
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True

    for name in ("stencil_profile.json", "column_profile.json"):
        profile = json.loads((ARTIFACT_DIR / name).read_text())
        assert profile["backend"] == "kokkos"
        assert profile["hardware"] == "RTX 5090 32GB"
        assert profile["kokkos_execution_space"] == "Cuda"
        assert profile["runtime_compute_capability"] == "12.0"
        assert 1 <= profile["kernel_launches"] <= 5
        assert profile["wall_time_s"] >= 0.0
        assert profile["host_device_transfer_bytes"] > 0
        assert isinstance(profile["occupancy_pct"], float)
        assert profile["registers_per_thread"] > 0
        assert profile["local_memory_bytes"] == 0
        assert profile["achieved_bandwidth_gbps"] >= 0.0
        assert profile["achieved_bandwidth_method"] == "fallback-derived"
        assert profile["artifact_paths"]
        for artifact in profile["artifact_paths"]:
            assert (ROOT / artifact).exists(), artifact


def test_kokkos_static_artifacts_exist() -> None:
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
    assert agent_success["candidate"] == "kokkos"
    assert agent_success["backend_used"] == "kokkos"
