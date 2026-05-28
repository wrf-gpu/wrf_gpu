"""Tester-added edge cases for the M2 cuda_tile bakeoff candidate.

Sprint: 2026-05-19-m2-cuda-tile-stencil-column
Owner: tester/sonnet (Claude Opus 4.7).

These tests do NOT re-run the GPU pipeline (they consume the worker's already-
produced artifacts on disk).  They focus on schema rigor, internal consistency
of the profile JSON numbers (so a future worker can't paste fabricated values
past the test), CC target verification of the produced binary, and the bench
binary's argument-parsing / error-path behavior.

If the bench binary or the worker artifacts do not exist (e.g. running on a
CI host without the toolchain), every test in this module is skipped rather
than failed - the canonical assertions still live in
``tests/test_m2_cuda_tile.py``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "cuda_tile"
SCRATCH = ROOT / "data" / "scratch" / "cuda_tile"
PROFILER_DIR = ROOT / "data" / "profiler_artifacts" / "cuda_tile"
BENCH = SCRATCH / "bench"
STENCIL_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-column-thermo-v1.npz"

PROFILE_REQUIRED_KEYS = {
    "benchmark",
    "backend",
    "hardware",
    "case",
    "wall_time_s",
    "kernel_launches",
    "host_device_transfer_bytes",
    "occupancy_pct",
    "registers_per_thread",
    "local_memory_bytes",
    "achieved_bandwidth_gbps",
    "artifact_paths",
}


def _require_artifacts() -> None:
    if not (ARTIFACT_DIR / "stencil_profile.json").exists():
        pytest.skip("cuda_tile artifacts not present; run scripts/m2_run_cuda_tile.sh first")


def _require_bench() -> None:
    if not BENCH.exists():
        pytest.skip(f"cuda_tile bench binary not built at {BENCH}")


def _load_profile(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text())


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_schema_keys_and_types(profile_name: str) -> None:
    _require_artifacts()
    profile = _load_profile(profile_name)
    missing = PROFILE_REQUIRED_KEYS - profile.keys()
    assert not missing, f"{profile_name} missing required schema keys: {sorted(missing)}"
    assert profile["backend"] == "cuda-tile"
    assert profile["hardware"] == "RTX 5090 32GB"
    assert isinstance(profile["wall_time_s"], float)
    assert isinstance(profile["kernel_launches"], int) and not isinstance(profile["kernel_launches"], bool)
    assert isinstance(profile["host_device_transfer_bytes"], int) and not isinstance(profile["host_device_transfer_bytes"], bool)
    assert isinstance(profile["occupancy_pct"], float)
    assert isinstance(profile["registers_per_thread"], int) and not isinstance(profile["registers_per_thread"], bool)
    assert isinstance(profile["local_memory_bytes"], int) and not isinstance(profile["local_memory_bytes"], bool)
    assert isinstance(profile["achieved_bandwidth_gbps"], float)
    assert isinstance(profile["artifact_paths"], list)
    assert all(isinstance(p, str) and p for p in profile["artifact_paths"])


@pytest.mark.parametrize(
    "profile_name,benchmark,case,reg_limit",
    [
        ("stencil_profile.json", "m2_stencil", "analytic-stencil-3d-advdiff-v1", 64),
        ("column_profile.json", "m2_column", "analytic-column-thermo-v1", 128),
    ],
)
def test_profile_sanity_bounds_match_contract(profile_name: str, benchmark: str, case: str, reg_limit: int) -> None:
    """Contract Performance Metrics sanity bounds (Performance Metrics section)."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    assert profile["benchmark"] == benchmark
    assert profile["case"] == case
    assert 0.0 <= profile["wall_time_s"] <= 5.0
    assert 1 <= profile["kernel_launches"] <= 10
    assert profile["host_device_transfer_bytes"] > 0
    assert profile["registers_per_thread"] <= reg_limit
    occupancy_floor = 25.0 if benchmark == "m2_stencil" else 20.0
    assert profile["occupancy_pct"] >= occupancy_floor


def test_column_profile_has_zero_local_memory() -> None:
    """Contract AC #7: column kernel must have local_memory_bytes == 0 (no spills)."""

    _require_artifacts()
    profile = _load_profile("column_profile.json")
    assert profile["local_memory_bytes"] == 0, (
        "column kernel reports register spilling; AC #7 requires zero local memory "
        "or an explicit maintainability waiver"
    )


def test_stencil_profile_has_zero_local_memory() -> None:
    """Same invariant on the stencil kernel - guards against silent spills."""

    _require_artifacts()
    profile = _load_profile("stencil_profile.json")
    assert profile["local_memory_bytes"] == 0


def test_achieved_bandwidth_is_consistent_with_transfer_and_wall() -> None:
    """Profile's bandwidth must equal host_device_transfer_bytes / wall_time_s / 1e9.

    Catches a worker who pastes a hand-rolled bandwidth number that doesn't agree
    with the other reported quantities.
    """

    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        if profile["wall_time_s"] <= 0:
            continue
        expected = profile["host_device_transfer_bytes"] / profile["wall_time_s"] / 1.0e9
        assert math.isclose(profile["achieved_bandwidth_gbps"], expected, rel_tol=1e-3, abs_tol=1e-6), (
            f"{name}: achieved_bandwidth_gbps={profile['achieved_bandwidth_gbps']} "
            f"inconsistent with transfer/wall computation {expected}"
        )


def test_profile_artifact_paths_are_relative_and_exist() -> None:
    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        for p in profile["artifact_paths"]:
            assert not Path(p).is_absolute(), f"{name}: artifact_paths must be relative ({p!r})"
            assert (ROOT / p).exists(), f"{name}: missing referenced artifact {p}"


def test_correctness_json_passes_both_problems() -> None:
    _require_artifacts()
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True
    assert correctness["stencil"]["fixture_id"] == "analytic-stencil-3d-advdiff-v1"
    assert correctness["column"]["fixture_id"] == "analytic-column-thermo-v1"
    for problem in ("stencil", "column"):
        for var in correctness[problem]["variables"]:
            assert var["pass"] is True, f"{problem}/{var['name']} reported failure"


def test_agent_success_log_is_well_formed() -> None:
    _require_artifacts()
    raw = json.loads((ARTIFACT_DIR / "agent_success.json").read_text())
    assert raw["sprint_count"] == 1
    assert raw["escalation_events"] == 0
    assert "build_attempts" in raw
    assert "runtime_failures" in raw


def test_maintainability_markdown_is_within_budget() -> None:
    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text()
    words = re.findall(r"\S+", text)
    assert len(words) <= 300, f"maintainability.md word count {len(words)} > 300"
    assert "build" in text.lower() or "Makefile" in text
    assert "debug" in text.lower() or "cuda-gdb" in text or "compute-sanitizer" in text


def test_resource_usage_matches_profile_register_counts() -> None:
    """The profile JSON's register/local numbers must come from cuobjdump output,
    not a hand-edited estimate.  Re-parse the on-disk resource_usage.txt and
    confirm the JSON values agree.
    """

    _require_artifacts()
    usage_text_path = SCRATCH / "resource_usage.txt"
    if not usage_text_path.exists():
        pytest.skip("resource_usage.txt not present; cuda_tile pipeline likely not run")
    text = usage_text_path.read_text(errors="replace")

    def parse(marker: str) -> tuple[int, int]:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if marker in line:
                for detail in lines[i + 1 : i + 5]:
                    m = re.search(r"REG:(\d+).*LOCAL:(\d+)", detail)
                    if m:
                        return int(m.group(1)), int(m.group(2))
        raise AssertionError(f"resource_usage.txt missing marker {marker!r}")

    s_reg, s_local = parse("stencil_advdiff_kernel")
    c_reg, c_local = parse("column_thermo_kernel")
    stencil = _load_profile("stencil_profile.json")
    column = _load_profile("column_profile.json")
    assert stencil["registers_per_thread"] == s_reg
    assert stencil["local_memory_bytes"] == s_local
    assert column["registers_per_thread"] == c_reg
    assert column["local_memory_bytes"] == c_local


def test_bench_binary_targets_only_sm120() -> None:
    """The built binary must contain sm_120 SASS and nothing lower; a silent
    fallback to e.g. sm_90 would make the bakeoff numbers meaningless for the
    target hardware."""

    _require_bench()
    if shutil.which("cuobjdump") is None:
        pytest.skip("cuobjdump not on PATH")
    sass = subprocess.run(
        ["cuobjdump", "--dump-sass", str(BENCH)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    archs = set(re.findall(r"^arch = (sm_\d+)", sass, flags=re.MULTILINE))
    assert archs == {"sm_120"}, f"expected only sm_120 SASS, got {archs}"


def test_bench_reports_usage_on_missing_args() -> None:
    _require_bench()
    res = subprocess.run([str(BENCH)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode != 0
    assert "usage:" in res.stderr.lower() or "usage:" in res.stdout.lower()


def test_bench_rejects_unknown_problem() -> None:
    _require_bench()
    res = subprocess.run(
        [str(BENCH), "not_a_problem", "--input", str(STENCIL_FIXTURE), "--output", "/tmp/x.npz"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    assert "unknown problem" in res.stderr.lower()


def test_bench_rejects_missing_required_flag() -> None:
    _require_bench()
    res = subprocess.run(
        [str(BENCH), "stencil"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    err = res.stderr.lower() + res.stdout.lower()
    assert "input" in err or "required" in err


def test_bench_rejects_missing_input_file(tmp_path: Path) -> None:
    _require_bench()
    missing = tmp_path / "does_not_exist.npz"
    res = subprocess.run(
        [str(BENCH), "stencil", "--input", str(missing), "--output", str(tmp_path / "out.npz")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    assert "cannot open" in res.stderr.lower() or "no such" in res.stderr.lower()


def test_bench_rejects_malformed_npz(tmp_path: Path) -> None:
    _require_bench()
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"not a zip file at all\n")
    res = subprocess.run(
        [str(BENCH), "stencil", "--input", str(garbage), "--output", str(tmp_path / "out.npz")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    assert res.stderr  # non-empty diagnostic


def test_bench_detects_wrong_fixture_for_problem(tmp_path: Path) -> None:
    """Passing the column fixture to the stencil problem must fail loudly on
    a missing required array (rather than silently producing garbage output).
    """

    _require_bench()
    if not COLUMN_FIXTURE.exists() or not STENCIL_FIXTURE.exists():
        pytest.skip("fixtures missing")
    res = subprocess.run(
        [str(BENCH), "stencil", "--input", str(COLUMN_FIXTURE), "--output", str(tmp_path / "out.npz")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    assert "phi_initial" in res.stderr or "missing" in res.stderr.lower()


def test_run_json_matches_profile_numbers() -> None:
    """The profile JSON values must agree with the bench's per-run JSON dump.

    Guards against post-hoc tampering of the profile JSON.
    """

    _require_artifacts()
    for kind in ("stencil", "column"):
        run_path = SCRATCH / f"{kind}_run.json"
        if not run_path.exists():
            pytest.skip(f"{kind}_run.json absent")
        run = json.loads(run_path.read_text())
        profile = _load_profile(f"{kind}_profile.json")
        assert profile["kernel_launches"] == run["kernel_launches"]
        assert profile["host_device_transfer_bytes"] == run["host_device_transfer_bytes"]
        assert math.isclose(profile["wall_time_s"], run["wall_time_s"], rel_tol=1e-9)
        assert math.isclose(profile["occupancy_pct"], run["theoretical_occupancy_pct"], rel_tol=1e-6)


def test_profiler_limitation_field_only_when_ncu_report_missing() -> None:
    """If the profile records a profiler_limitation, the .ncu-rep MUST be absent;
    if it's absent, the profile must explain why.  Catches both honest reporting
    failures and overclaiming."""

    _require_artifacts()
    for kind in ("stencil", "column"):
        profile = _load_profile(f"{kind}_profile.json")
        rep = PROFILER_DIR / f"{kind}.ncu-rep"
        has_limitation = "profiler_limitation" in profile
        if rep.exists():
            assert not has_limitation, (
                f"{kind}_profile.json declares profiler_limitation but {rep} exists"
            )
        else:
            assert has_limitation, (
                f"{kind}_profile.json has no profiler_limitation but {rep} is missing"
            )
            assert profile["profiler_limitation"].strip()
