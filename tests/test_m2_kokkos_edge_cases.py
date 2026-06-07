"""Tester-added edge cases for the M2 Kokkos bakeoff candidate.

Sprint: 2026-05-19-m2-kokkos-stencil-column
Owner: tester/sonnet (Claude Opus 4.7) - cross-AI verification of the
gpt-5.5 worker output for ADR-001 input.

These tests consume the worker's already-produced artifacts on disk plus
the built bench binary; they do not re-run scripts/m2_run_kokkos.sh (that
canonical happy-path lives in tests/test_m2_kokkos.py). Coverage focuses
on:

- Profile JSON schema rigor and internal numeric consistency (so a
  future worker can't paste fabricated values past the test).
- Contract sanity bounds (registers <=64 stencil / <=128 column,
  local_memory_bytes==0, kernel_launches<=5, occupancy floors).
- Bench binary behavior on malformed / missing / wrong-fixture input.
- Build idempotency and that the bench is linked against the Kokkos
  install (not a host-only stale build).
- Kokkos-specific evidence: CUDA execution space, CudaSpace allocations
  in the resource dump, kokkos_config.txt agreement, build.log presence.

If the bench binary or artifacts are missing, tests skip cleanly.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import jax
import pytest


# These two checks need GPU-build outputs: the committed kokkos profile json
# references un-vendored Nsight Compute (ncu) profiler dumps, and the build-log
# check needs the cmake/CUDA build to have run (CPU run here has no cmake). Both
# are GPU/CUDA-toolchain artifacts of a legacy bakeoff subsystem.
requires_gpu_toolchain = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="needs GPU + CUDA toolchain build outputs (cmake build log / Nsight ncu artifacts)",
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "kokkos"
SCRATCH = ROOT / "data" / "scratch" / "kokkos"
KOKKOS_INSTALL = ROOT / "data" / "scratch" / "kokkos-install"
PROFILER_DIR = ROOT / "data" / "profiler_artifacts" / "kokkos"
BENCH = SCRATCH / "bench"
STENCIL_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-column-thermo-v1.npz"
BUILD_SH = ROOT / "src" / "gpuwrf" / "backends" / "kokkos" / "build.sh"

PROFILE_REQUIRED_KEYS = {
    "achieved_bandwidth_gbps",
    "achieved_bandwidth_method",
    "artifact_paths",
    "backend",
    "benchmark",
    "case",
    "hardware",
    "host_device_transfer_bytes",
    "kernel_launches",
    "kokkos_execution_space",
    "kokkos_version",
    "local_memory_bytes",
    "occupancy_pct",
    "profiler_limitation",
    "registers_per_thread",
    "runtime_compute_capability",
    "wall_time_s",
}


def _require_artifacts() -> None:
    if not (ARTIFACT_DIR / "stencil_profile.json").exists():
        pytest.skip("kokkos artifacts not present; run scripts/m2_run_kokkos.sh first")


def _require_bench() -> None:
    if not BENCH.exists():
        pytest.skip(f"kokkos bench binary not built at {BENCH}")


def _load_profile(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text())


# --------------------------------------------------------------------------- #
# Profile JSON schema rigor and consistency                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_schema_keys_and_types(profile_name: str) -> None:
    _require_artifacts()
    profile = _load_profile(profile_name)
    missing = PROFILE_REQUIRED_KEYS - profile.keys()
    assert not missing, f"{profile_name} missing required schema keys: {sorted(missing)}"
    assert profile["backend"] == "kokkos"
    assert profile["hardware"] == "RTX 5090 32GB"
    assert isinstance(profile["wall_time_s"], float)
    assert isinstance(profile["kernel_launches"], int) and not isinstance(profile["kernel_launches"], bool)
    assert isinstance(profile["host_device_transfer_bytes"], int) and not isinstance(
        profile["host_device_transfer_bytes"], bool
    )
    assert isinstance(profile["occupancy_pct"], float)
    assert isinstance(profile["registers_per_thread"], int) and not isinstance(
        profile["registers_per_thread"], bool
    )
    assert isinstance(profile["local_memory_bytes"], int) and not isinstance(profile["local_memory_bytes"], bool)
    assert isinstance(profile["achieved_bandwidth_gbps"], float)
    assert isinstance(profile["artifact_paths"], list)
    assert all(isinstance(p, str) and p for p in profile["artifact_paths"])
    assert profile["kokkos_execution_space"] == "Cuda", (
        "kokkos_execution_space must be Cuda; Serial/OpenMP would invalidate the bakeoff row"
    )
    assert profile["runtime_compute_capability"] == "12.0"
    assert profile["kokkos_version"] == 40701
    assert profile["achieved_bandwidth_method"] == "fallback-derived"


@pytest.mark.parametrize(
    "profile_name,benchmark,case,reg_limit,occupancy_floor",
    [
        ("stencil_profile.json", "m2_stencil", "analytic-stencil-3d-advdiff-v1", 64, 25.0),
        ("column_profile.json", "m2_column", "analytic-column-thermo-v1", 128, 20.0),
    ],
)
def test_profile_sanity_bounds_match_contract(
    profile_name: str, benchmark: str, case: str, reg_limit: int, occupancy_floor: float
) -> None:
    """Contract Acceptance Criteria 7-9 plus Performance Metrics sanity bounds."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    assert profile["benchmark"] == benchmark
    assert profile["case"] == case
    assert 0.0 < profile["wall_time_s"] <= 5.0
    assert 1 <= profile["kernel_launches"] <= 5, "AC #7: <=5 kernel launches"
    assert profile["host_device_transfer_bytes"] > 0
    assert profile["registers_per_thread"] > 0
    assert profile["registers_per_thread"] <= reg_limit, (
        f"AC #9: registers_per_thread {profile['registers_per_thread']} > {reg_limit}"
    )
    assert profile["occupancy_pct"] >= occupancy_floor
    assert profile["occupancy_pct"] <= 100.0


def test_column_profile_has_zero_local_memory() -> None:
    """Contract AC #8: column kernel must have local_memory_bytes == 0 (no spills)."""

    _require_artifacts()
    profile = _load_profile("column_profile.json")
    assert profile["local_memory_bytes"] == 0, (
        "column kernel reports register spilling; AC #8 requires zero local memory"
    )


def test_stencil_profile_has_zero_local_memory() -> None:
    """Guard against silent stencil-kernel spills (same invariant)."""

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


@requires_gpu_toolchain
def test_profile_artifact_paths_are_relative_and_exist() -> None:
    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        for p in profile["artifact_paths"]:
            assert not Path(p).is_absolute(), f"{name}: artifact_paths must be relative ({p!r})"
            assert (ROOT / p).exists(), f"{name}: missing referenced artifact {p}"


def test_run_json_matches_profile_numbers() -> None:
    """Profile JSON values must agree with the bench's per-run JSON dump.

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
        assert profile["kokkos_version"] == run["kokkos_version"]
        assert profile["kokkos_execution_space"] == run["kokkos_execution_space"]
        assert profile["runtime_compute_capability"] == run["runtime_compute_capability"]


def test_profiler_limitation_field_only_when_ncu_report_missing() -> None:
    """If the profile records a profiler_limitation, the .ncu-rep MUST be absent;
    if .ncu-rep is present, no limitation should be declared. Catches both honest
    reporting failures and overclaiming."""

    _require_artifacts()
    for kind in ("stencil", "column"):
        profile = _load_profile(f"{kind}_profile.json")
        rep = PROFILER_DIR / f"{kind}.ncu-rep"
        has_limitation = "profiler_limitation" in profile and profile["profiler_limitation"].strip()
        if rep.exists() and rep.stat().st_size > 0:
            assert not has_limitation, (
                f"{kind}_profile.json declares profiler_limitation but {rep} exists"
            )
        else:
            assert has_limitation, (
                f"{kind}_profile.json has no profiler_limitation but {rep} is missing/empty"
            )


def test_resource_usage_matches_profile_register_counts() -> None:
    """Profile JSON register/local numbers must come from cuobjdump output,
    not a hand-edited estimate.  Re-parse the on-disk resource_usage.txt and
    confirm the JSON values agree."""

    _require_artifacts()
    usage_text_path = SCRATCH / "resource_usage.txt"
    if not usage_text_path.exists():
        pytest.skip("resource_usage.txt not present; kokkos pipeline likely not run")
    text = usage_text_path.read_text(errors="replace")
    lines = text.splitlines()

    def parse(marker: str) -> tuple[int, int]:
        for i, line in enumerate(lines):
            if marker in line:
                for detail in lines[i + 1 : i + 5]:
                    m = re.search(r"REG:(\d+).*LOCAL:(\d+)", detail)
                    if m:
                        return int(m.group(1)), int(m.group(2))
        raise AssertionError(f"resource_usage.txt missing marker {marker!r}")

    s_reg, s_local = parse("StencilAdvdiffKernel")
    c_reg, c_local = parse("ColumnThermoKernel")
    stencil = _load_profile("stencil_profile.json")
    column = _load_profile("column_profile.json")
    assert stencil["registers_per_thread"] == s_reg
    assert stencil["local_memory_bytes"] == s_local
    assert column["registers_per_thread"] == c_reg
    assert column["local_memory_bytes"] == c_local


def test_resource_usage_kernels_target_cuda_space() -> None:
    """Handoff requirement: verify View allocations are CUDA-space (not Host).

    The Kokkos mangled name for each parallel_for embeds the execution-space
    type; for the stencil/column kernels it must contain `Kokkos::Cuda`
    (mangled as 'NS_4CudaE...' or similar).  HostSpace / Serial would show
    a different substring.
    """

    _require_artifacts()
    usage_text_path = SCRATCH / "resource_usage.txt"
    if not usage_text_path.exists():
        pytest.skip("resource_usage.txt not present")
    text = usage_text_path.read_text(errors="replace")
    lines = text.splitlines()

    def find_kernel_line(marker: str) -> str:
        for line in lines:
            if marker in line and "Function" in line:
                return line
        raise AssertionError(f"no Function line matched {marker}")

    stencil_line = find_kernel_line("StencilAdvdiffKernel")
    column_line = find_kernel_line("ColumnThermoKernel")
    # Kokkos::Cuda mangles as '4Cuda' (length-prefixed name).
    assert "4Cuda" in stencil_line, (
        f"stencil kernel symbol does not advertise Kokkos::Cuda execution space: {stencil_line}"
    )
    assert "4Cuda" in column_line, (
        f"column kernel symbol does not advertise Kokkos::Cuda execution space: {column_line}"
    )
    # Defensive: confirm Serial host space is not the executor.
    assert "6Serial" not in stencil_line.split("StencilAdvdiffKernel")[-1]
    assert "6Serial" not in column_line.split("ColumnThermoKernel")[-1]


# --------------------------------------------------------------------------- #
# Correctness, maintainability, agent-success                                 #
# --------------------------------------------------------------------------- #


def test_correctness_json_passes_both_problems() -> None:
    _require_artifacts()
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["backend"] == "kokkos"
    assert correctness["stencil"]["pass"] is True
    assert correctness["column"]["pass"] is True
    assert correctness["stencil"]["fixture_id"] == "analytic-stencil-3d-advdiff-v1"
    assert correctness["column"]["fixture_id"] == "analytic-column-thermo-v1"
    assert correctness["stencil"]["tier"] == 1
    assert correctness["column"]["tier"] == 1
    for problem in ("stencil", "column"):
        for var in correctness[problem]["variables"]:
            assert var["pass"] is True, f"{problem}/{var['name']} reported failure"


def test_agent_success_log_is_well_formed() -> None:
    _require_artifacts()
    raw = json.loads((ARTIFACT_DIR / "agent_success.json").read_text())
    assert raw["candidate"] == "kokkos"
    assert raw["backend_used"] == "kokkos"
    assert isinstance(raw["sprint_count"], int) and raw["sprint_count"] >= 1
    assert isinstance(raw["reviewer_rejections"], int) and raw["reviewer_rejections"] >= 0
    assert isinstance(raw["escalation_events"], int) and raw["escalation_events"] >= 0
    assert "build_attempts" in raw
    assert "runtime_failures" in raw
    assert isinstance(raw["build_attempts"], int) and raw["build_attempts"] >= 1
    assert isinstance(raw["notes"], list)


def test_maintainability_markdown_is_within_budget_and_covers_required_topics() -> None:
    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text()
    words = re.findall(r"\S+", text)
    assert len(words) <= 300, f"maintainability.md word count {len(words)} > 300"
    lower = text.lower()
    # Contract AC #11 topics: build complexity, error legibility, debugger, agent friction.
    assert "build" in lower or "cmake" in lower
    assert "error" in lower or "diagnostic" in lower or "legibility" in lower
    assert "debug" in lower or "cuobjdump" in lower or "cuda-gdb" in lower or "sass" in lower
    assert "agent" in lower or "iteration" in lower or "friction" in lower


# --------------------------------------------------------------------------- #
# Bench binary behavior                                                       #
# --------------------------------------------------------------------------- #


def test_bench_binary_targets_blackwell() -> None:
    """AC #3: bench SASS or runtime CC must indicate Blackwell (sm_120 / CC 12.0).

    Per the contract, if only PTX is embedded the runtime compute-capability
    check is the acceptable fallback. This test honors that fallback.
    """

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
    if archs:
        # If any SASS is embedded, it must be sm_120 (no silent downgrade).
        assert "sm_120" in archs, f"expected sm_120 SASS, got {archs}"
        # Reject any earlier arch leaking in alongside.
        earlier = {a for a in archs if a != "sm_120"}
        assert not earlier, f"unexpected non-blackwell SASS targets: {earlier}"
    else:
        # PTX-only fallback per AC #3: confirm runtime CC matches.
        config = subprocess.run(
            [str(BENCH), "config"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        assert "runtime_compute_capability=12.0" in config


def test_bench_reports_usage_on_missing_args() -> None:
    _require_bench()
    res = subprocess.run([str(BENCH)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0  # host.cpp prints usage and exits 0 when argc < 2
    out = (res.stdout + res.stderr).lower()
    assert "usage:" in out


def test_bench_rejects_unknown_problem(tmp_path: Path) -> None:
    _require_bench()
    res = subprocess.run(
        [str(BENCH), "not_a_problem", "--input", str(STENCIL_FIXTURE), "--output", str(tmp_path / "x.npz")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    err = (res.stderr + res.stdout).lower()
    assert "unknown problem" in err or "not_a_problem" in err


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
    err = (res.stderr + res.stdout).lower()
    assert "input" in err or "required" in err


def test_bench_rejects_flag_without_value(tmp_path: Path) -> None:
    """`bench stencil --input` (no value) must fail loudly."""

    _require_bench()
    res = subprocess.run(
        [str(BENCH), "stencil", "--input"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0


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
    err = res.stderr.lower()
    assert "cannot open" in err or "no such" in err or "input" in err


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


def test_bench_rejects_truncated_zip(tmp_path: Path) -> None:
    """A near-empty file (<22 bytes, below EOCD size) should be rejected, not crash."""

    _require_bench()
    short = tmp_path / "short.npz"
    short.write_bytes(b"PK\x05\x06")  # EOCD magic without the rest
    res = subprocess.run(
        [str(BENCH), "stencil", "--input", str(short), "--output", str(tmp_path / "out.npz")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode != 0
    assert res.stderr


def test_bench_detects_wrong_fixture_for_problem(tmp_path: Path) -> None:
    """Passing the column fixture to the stencil problem must fail loudly on
    a missing required array rather than silently produce garbage output."""

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
    err = res.stderr.lower() + res.stdout.lower()
    assert "phi_initial" in err or "missing" in err or "shape" in err


def test_bench_stencil_output_matches_reference(tmp_path: Path) -> None:
    """End-to-end: run the bench on a fresh temp output path and confirm the
    candidate matches the reference NPZ. This guards against a future worker
    silently shipping stale output files in the artifact directory."""

    _require_bench()
    if not STENCIL_FIXTURE.exists():
        pytest.skip("stencil fixture missing")
    out = tmp_path / "stencil_out.npz"
    res = subprocess.run(
        [str(BENCH), "stencil", "--input", str(STENCIL_FIXTURE), "--output", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists() and out.stat().st_size > 0

    cmp = subprocess.run(
        [
            "python",
            "-m",
            "gpuwrf.validation.compare_fixture",
            "--manifest",
            str(ROOT / "fixtures" / "manifests" / "analytic-stencil-3d-advdiff-v1.yaml"),
            "--candidate",
            str(out),
            "--reference",
            str(STENCIL_FIXTURE),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert cmp.returncode == 0, cmp.stderr
    parsed = json.loads(cmp.stdout)
    assert parsed["pass"] is True


def test_bench_column_output_matches_reference(tmp_path: Path) -> None:
    """End-to-end column kernel: same shape as the stencil test."""

    _require_bench()
    if not COLUMN_FIXTURE.exists():
        pytest.skip("column fixture missing")
    out = tmp_path / "column_out.npz"
    res = subprocess.run(
        [str(BENCH), "column", "--input", str(COLUMN_FIXTURE), "--output", str(out)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists() and out.stat().st_size > 0

    cmp = subprocess.run(
        [
            "python",
            "-m",
            "gpuwrf.validation.compare_fixture",
            "--manifest",
            str(ROOT / "fixtures" / "manifests" / "analytic-column-thermo-v1.yaml"),
            "--candidate",
            str(out),
            "--reference",
            str(COLUMN_FIXTURE),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert cmp.returncode == 0, cmp.stderr
    parsed = json.loads(cmp.stdout)
    assert parsed["pass"] is True


def test_bench_run_is_reproducible_bitwise(tmp_path: Path) -> None:
    """Two back-to-back stencil runs of the bench on the same input must
    produce identical output bytes; the bench is fully deterministic with no
    nondeterministic atomics or RNG. Catches silent introduction of warp-level
    nondeterministic reductions or atomicAdd in a future refactor."""

    _require_bench()
    if not STENCIL_FIXTURE.exists():
        pytest.skip("stencil fixture missing")
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    for path in (a, b):
        res = subprocess.run(
            [str(BENCH), "stencil", "--input", str(STENCIL_FIXTURE), "--output", str(path)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert res.returncode == 0
    assert a.read_bytes() == b.read_bytes(), "bench is not bitwise reproducible between runs"


def test_bench_config_advertises_blackwell120() -> None:
    """`bench config` must advertise CC 12.0, Cuda exec space, BLACKWELL120
    arch.  Guards against a build that silently fell back to Serial-only."""

    _require_bench()
    res = subprocess.run(
        [str(BENCH), "config"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "kokkos_execution_space=Cuda" in out
    assert "runtime_compute_capability=12.0" in out
    assert "BLACKWELL120" in out or "Capability: 12.0" in out
    assert "KOKKOS_ENABLE_CUDA: yes" in out


# --------------------------------------------------------------------------- #
# Build idempotency / install presence                                        #
# --------------------------------------------------------------------------- #


def test_kokkos_install_present_and_recognized() -> None:
    """The Kokkos install tree must exist; the bench must be linked against it,
    not built against a stray system Kokkos.  Both `KokkosConfig.cmake` and the
    `nvcc_wrapper` shim are required."""

    _require_bench()
    assert (KOKKOS_INSTALL / "lib" / "cmake" / "Kokkos" / "KokkosConfig.cmake").exists(), (
        "Kokkos install missing or incomplete; build.sh must produce KokkosConfig.cmake"
    )
    wrapper = KOKKOS_INSTALL / "bin" / "nvcc_wrapper"
    assert wrapper.exists() and (wrapper.stat().st_mode & 0o111), "nvcc_wrapper missing or not executable"


@requires_gpu_toolchain
def test_build_log_recorded() -> None:
    """scripts/m2_run_kokkos.sh tees the build log; absence implies the
    pipeline was not run end-to-end."""

    _require_artifacts()
    log = SCRATCH / "build.log"
    assert log.exists(), "data/scratch/kokkos/build.log missing"
    text = log.read_text(errors="replace")
    assert "Built target bench" in text or "bench" in text.lower()


def test_build_is_idempotent_on_rerun() -> None:
    """Re-running build.sh when the install and bench both already exist must
    NOT re-clone the Kokkos source or re-install Kokkos. AC #1: idempotent."""

    _require_bench()
    if not BUILD_SH.exists():
        pytest.skip("build.sh missing")
    # If Kokkos source dir or install is absent, the first re-run would re-clone;
    # only run this test when caches are warm.
    src = ROOT / "data" / "scratch" / "kokkos-src"
    if not src.exists() or not KOKKOS_INSTALL.exists():
        pytest.skip("kokkos caches not warm; skipping idempotency timing test")
    src_mtime = src.stat().st_mtime
    install_mtime = KOKKOS_INSTALL.stat().st_mtime
    res = subprocess.run(
        ["bash", str(BUILD_SH)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr
    # Source clone must not have been touched.
    assert abs(src.stat().st_mtime - src_mtime) < 1.0, (
        "kokkos source directory mtime changed; build.sh re-cloned despite cache"
    )
    # Install directory's own mtime should not have advanced (no re-install).
    assert KOKKOS_INSTALL.stat().st_mtime <= install_mtime + 1.0, (
        "kokkos install directory was re-installed; AC #1 idempotency violated"
    )


# --------------------------------------------------------------------------- #
# Deliberate-bug capture (maintainability evidence)                           #
# --------------------------------------------------------------------------- #


def test_deliberate_bug_capture_present_and_diagnostic() -> None:
    """maintainability.md cites data/scratch/kokkos/deliberate_bug_stderr.txt.
    The file must exist and contain a real compiler diagnostic, not be empty."""

    bug_stderr = SCRATCH / "deliberate_bug_stderr.txt"
    bug_exit = SCRATCH / "deliberate_bug_exit.txt"
    if not bug_stderr.exists():
        pytest.skip("deliberate_bug_stderr.txt not present")
    text = bug_stderr.read_text(errors="replace")
    assert "error" in text.lower(), "deliberate_bug_stderr.txt has no compiler error"
    # The worker report names this exact undefined-identifier diagnostic.
    assert "phi_nxt" in text or "undefined" in text.lower() or "identifier" in text.lower()
    if bug_exit.exists():
        rc = bug_exit.read_text(errors="replace").strip()
        assert rc != "0", "deliberate bug exit code was 0; build did not actually fail"
