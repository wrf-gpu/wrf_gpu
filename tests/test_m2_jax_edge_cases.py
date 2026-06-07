"""Tester-added edge cases for the M2 JAX bakeoff candidate.

Sprint: 2026-05-19-m2-jax-stencil-column
Owner: tester/sonnet (Claude Opus 4.7) - cross-AI verification of the
gpt-5.5 worker output for ADR-001 input.

These tests consume the worker's already-produced artifacts on disk plus
the captured XLA dump tree; they do not re-run `scripts/m2_run_jax.sh`
(the canonical happy path lives in `tests/test_m2_jax.py`). Coverage
focuses on:

- Profile JSON schema rigor and internal numeric consistency (so a
  future worker cannot paste fabricated numbers past the test).
- Contract sanity bounds (registers <=64 stencil / <=128 column,
  `local_memory_bytes == 0`, `kernel_launches <= 5`, occupancy floors).
- JAX-specific evidence: `jax_backend == "gpu"`, `jax_devices` contains a
  CUDA device, `jax_version == "0.10.0"` per the M2-S1 pin, and the
  warmup pattern documents compile-time exclusion.
- Cross-checks against the XLA dump tree: ptxas reports `0 bytes spill
  stores, 0 bytes spill loads`, the thunk_sequence shows exactly one
  `kKernel` (column may add a `kCopy`), and the compiled HLO contains a
  single fusion.
- Bench CLI behaviour: missing/malformed inputs, wrong-fixture-for-
  problem, unknown --problem value, and end-to-end reproduction against
  the reference fixtures including bitwise reproducibility.
- Maintainability/agent_success hygiene: budget enforcement, required
  topics, no Pallas/Triton claims that would violate the contract.

If the venv or the XLA dump artifacts are absent the tests skip cleanly.
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


# The tests decorated below run the bench inside the m2-jax CUDA venv (asserting
# its jax default_backend=="gpu") or cross-check GPU-compiled HLO / un-vendored GPU
# profiler artifacts. They cannot pass on a CPU-only checkout (the venv's jax is
# CPU there); they are GPU-benchmark tests of a legacy bakeoff subsystem untouched
# by the operational pipeline.
requires_gpu_toolchain = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="M2 jax bakeoff edge cases require a JAX GPU (CUDA) backend in the bench venv",
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "jax"
SCRATCH = ROOT / "data" / "scratch" / "m2-jax"
VENV = ROOT / "data" / "scratch" / "m2-jax-venv"
VENV_PY = VENV / "bin" / "python"
PROFILER_DIR = ROOT / "data" / "profiler_artifacts" / "jax"
STENCIL_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-column-thermo-v1.npz"
STENCIL_MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-stencil-3d-advdiff-v1.yaml"
COLUMN_MANIFEST = ROOT / "fixtures" / "manifests" / "analytic-column-thermo-v1.yaml"

PROFILE_REQUIRED_KEYS = {
    "achieved_bandwidth_gbps",
    "achieved_bandwidth_method",
    "artifact_paths",
    "backend",
    "benchmark",
    "case",
    "hardware",
    "hlo_kernel_ops",
    "host_device_transfer_bytes",
    "jax_backend",
    "jax_devices",
    "jax_version",
    "kernel_launches",
    "local_memory_bytes",
    "occupancy_pct",
    "profiler_limitation",
    "registers_per_thread",
    "wall_time_s",
    "warmup_pattern",
}


def _require_artifacts() -> None:
    if not (ARTIFACT_DIR / "stencil_profile.json").exists():
        pytest.skip("jax artifacts not present; run scripts/m2_run_jax.sh first")


def _require_venv() -> None:
    if not VENV_PY.exists():
        pytest.skip(f"jax venv missing at {VENV_PY}")


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
    assert profile["backend"] == "jax"
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
    assert isinstance(profile["local_memory_bytes"], int) and not isinstance(
        profile["local_memory_bytes"], bool
    )
    assert isinstance(profile["achieved_bandwidth_gbps"], float)
    assert isinstance(profile["artifact_paths"], list) and profile["artifact_paths"]
    assert all(isinstance(p, str) and p for p in profile["artifact_paths"])
    assert isinstance(profile["hlo_kernel_ops"], list) and profile["hlo_kernel_ops"]
    assert profile["achieved_bandwidth_method"] == "fallback-derived"


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_declares_pinned_jax_runtime(profile_name: str) -> None:
    """AC #1 says jax[cuda13]==0.10.0; profile must reflect that pin and a
    real GPU device. Catches a worker who fell back to the CPU backend."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    assert profile["jax_version"] == "0.10.0", (
        f"{profile_name}: jax_version={profile['jax_version']} != contract pin 0.10.0"
    )
    assert profile["jax_backend"] == "gpu", (
        f"{profile_name}: jax_backend={profile['jax_backend']} (CPU fallback would invalidate the bakeoff row)"
    )
    devices = profile["jax_devices"]
    assert isinstance(devices, list) and devices, f"{profile_name}: jax_devices empty"
    assert any(("cuda" in str(d).lower()) or ("gpu" in str(d).lower()) for d in devices), (
        f"{profile_name}: no CUDA/GPU device in jax_devices={devices}"
    )


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_warmup_pattern_documents_compile_exclusion(profile_name: str) -> None:
    """Contract AC #5 + Risk #3/#4: compile time must be excluded; the
    `warmup_pattern` field must call out a post-compile warmup and a
    multi-run median (or equivalent)."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    pattern = profile.get("warmup_pattern", "")
    assert isinstance(pattern, str) and pattern
    lower = pattern.lower()
    assert "compile" in lower or "warmup" in lower, (
        f"{profile_name}: warmup_pattern does not mention compile/warmup: {pattern!r}"
    )
    assert "median" in lower or "min" in lower, (
        f"{profile_name}: warmup_pattern does not document a multi-run statistic: {pattern!r}"
    )
    # Wall time on these tiny fixtures should be well below a typical XLA
    # compile cost (>=100ms); >1s would indicate compile is being counted.
    assert 0.0 < profile["wall_time_s"] < 0.5, (
        f"{profile_name}: wall_time_s={profile['wall_time_s']}s implausibly large "
        "for a single jit invocation on a 32x16x8 stencil / 40-cell column; "
        "compile time may be leaking into the measurement"
    )


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
    """Contract Performance Metrics: <=5 launches, registers <=64/128,
    occupancy >= 25/20%, local_memory_bytes == 0 on the column kernel."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    assert profile["benchmark"] == benchmark
    assert profile["case"] == case
    assert 1 <= profile["kernel_launches"] <= 5, (
        f"{profile_name}: kernel_launches={profile['kernel_launches']} outside [1,5]"
    )
    assert profile["host_device_transfer_bytes"] > 0
    assert profile["registers_per_thread"] > 0
    assert profile["registers_per_thread"] <= reg_limit, (
        f"{profile_name}: registers_per_thread {profile['registers_per_thread']} > {reg_limit}"
    )
    assert occupancy_floor <= profile["occupancy_pct"] <= 100.0


def test_column_profile_has_zero_local_memory() -> None:
    """Contract AC #14 (the ADR-001 signal): column kernel must have
    local_memory_bytes == 0; XLA spilling here would mean JAX cannot win
    Problem 2 without dropping into Pallas (M2-S6 territory)."""

    _require_artifacts()
    profile = _load_profile("column_profile.json")
    assert profile["local_memory_bytes"] == 0, (
        "column kernel reports register spilling; AC #14 requires zero local memory or an explicit "
        "ADR-001 carve-out in maintainability.md"
    )


def test_stencil_profile_has_zero_local_memory() -> None:
    """Guard against silent stencil-kernel spills (same invariant). The
    contract does not literally require this on stencil, but a spill here
    would still be load-bearing evidence for ADR-001."""

    _require_artifacts()
    profile = _load_profile("stencil_profile.json")
    assert profile["local_memory_bytes"] == 0


def test_achieved_bandwidth_is_consistent_with_transfer_and_wall() -> None:
    """Profile bandwidth must equal `host_device_transfer_bytes /
    wall_time_s / 1e9`. Catches a worker pasting a hand-rolled bandwidth
    that does not agree with the other reported quantities."""

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
    """Profile artifact_paths must be repo-relative and resolve on disk."""

    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        for p in profile["artifact_paths"]:
            assert not Path(p).is_absolute(), f"{name}: artifact_paths must be relative ({p!r})"
            assert (ROOT / p).exists(), f"{name}: missing referenced artifact {p}"


def test_run_json_matches_profile_numbers() -> None:
    """Profile JSON values must agree with the bench's per-run JSON dump.
    Guards against post-hoc tampering of the profile JSON."""

    _require_artifacts()
    for kind in ("stencil", "column"):
        run_path = SCRATCH / f"{kind}_run.json"
        if not run_path.exists():
            pytest.skip(f"{kind}_run.json absent")
        run = json.loads(run_path.read_text())
        profile = _load_profile(f"{kind}_profile.json")
        assert profile["kernel_launches"] == run["kernel_launches"]
        assert profile["host_device_transfer_bytes"] == run["host_device_transfer_bytes"]
        assert profile["registers_per_thread"] == run["registers_per_thread"]
        assert profile["local_memory_bytes"] == run["local_memory_bytes"]
        assert math.isclose(profile["occupancy_pct"], run["occupancy_pct"], rel_tol=1e-9)
        assert math.isclose(profile["wall_time_s"], run["wall_time_s"], rel_tol=1e-6)


def test_profiler_limitation_is_documented_when_ncu_failed() -> None:
    """`profiler_limitation` must be populated whenever ncu produced no
    valid .ncu-rep (ERR_NVGPUCTRPERM on this workstation)."""

    _require_artifacts()
    for kind in ("stencil", "column"):
        profile = _load_profile(f"{kind}_profile.json")
        rep = PROFILER_DIR / f"{kind}.ncu-rep"
        ncu_exit_path = PROFILER_DIR / f"{kind}_ncu_exit.txt"
        ncu_rc = ncu_exit_path.read_text().strip() if ncu_exit_path.exists() else "missing"
        rep_ok = rep.exists() and rep.stat().st_size > 0
        has_limitation = bool(profile.get("profiler_limitation", "").strip())
        if rep_ok and ncu_rc == "0":
            assert not has_limitation, (
                f"{kind}: ncu rep exists with rc=0 but profiler_limitation is set"
            )
        else:
            assert has_limitation, (
                f"{kind}: ncu rep missing/failed (rc={ncu_rc}) but no profiler_limitation declared"
            )


# --------------------------------------------------------------------------- #
# Cross-check against XLA dump tree (independent of bench self-report)         #
# --------------------------------------------------------------------------- #


def _xla_dump_dir(problem: str) -> Path:
    return PROFILER_DIR / f"{problem}_xla_dump"


def _ptxas_text(problem: str) -> str:
    path = PROFILER_DIR / f"{problem}_cuobjdump_resource_usage.txt"
    if not path.exists():
        pytest.skip(f"{path} absent")
    return path.read_text(errors="replace")


@pytest.mark.parametrize("problem", ["stencil", "column"])
def test_ptxas_reports_no_spills(problem: str) -> None:
    """ptxas log captured during the run must say `0 bytes spill stores,
    0 bytes spill loads`. This is the independent, tool-level evidence
    behind `local_memory_bytes == 0`."""

    _require_artifacts()
    text = _ptxas_text(problem)
    assert re.search(r"0\s+bytes\s+spill\s+stores", text), (
        f"{problem}: ptxas log does not assert zero spill stores"
    )
    assert re.search(r"0\s+bytes\s+spill\s+loads", text), (
        f"{problem}: ptxas log does not assert zero spill loads"
    )
    assert re.search(r"\bLOCAL:0\b", text), (
        f"{problem}: cuobjdump resource summary does not show LOCAL:0"
    )


@pytest.mark.parametrize("problem", ["stencil", "column"])
def test_cuobjdump_register_count_matches_profile(problem: str) -> None:
    """The REG:N reported by cuobjdump on the XLA-dumped cubin must equal
    `registers_per_thread` in the profile JSON. Catches a worker pasting
    a hand-rolled register count."""

    _require_artifacts()
    text = _ptxas_text(problem)
    match = re.search(r"REG:(\d+)", text)
    assert match, f"{problem}: cuobjdump output has no REG:N line"
    cubin_regs = int(match.group(1))
    profile = _load_profile(f"{problem}_profile.json")
    assert profile["registers_per_thread"] == cubin_regs, (
        f"{problem}: profile registers_per_thread={profile['registers_per_thread']} "
        f"!= cuobjdump REG:{cubin_regs}"
    )


@pytest.mark.parametrize("problem", ["stencil", "column"])
def test_xla_thunk_sequence_has_one_compute_kernel(problem: str) -> None:
    """The XLA dump's `thunk_sequence.txt` must record exactly one
    `kKernel` thunk; copy/memcpy thunks are allowed. This is the
    independent confirmation behind `kernel_launches`."""

    _require_artifacts()
    dump = _xla_dump_dir(problem)
    matches = list(dump.rglob("*thunk_sequence.txt"))
    if not matches:
        pytest.skip(f"{problem}: no thunk_sequence.txt in XLA dump")
    text = matches[0].read_text(errors="replace")
    kkernels = re.findall(r"\bkKernel\b", text)
    assert len(kkernels) == 1, (
        f"{problem}: thunk_sequence shows {len(kkernels)} kKernel thunks; "
        "contract expects 1 (XLA fusion target)"
    )
    # All thunks must be either kernels or simple copies; no unfused
    # custom-calls / collective ops on a single-GPU bakeoff.
    forbidden = re.findall(r"kCustomCall|kAllReduce|kCollectivePermute|kAllToAll", text)
    assert not forbidden, f"{problem}: thunk_sequence contains forbidden ops {forbidden}"


@pytest.mark.parametrize("problem", ["stencil", "column"])
def test_compiled_hlo_has_single_fusion_op(problem: str) -> None:
    """The compiled HLO text captured by the bench must contain exactly
    one `fusion(` call site. Multiple fusions would suggest XLA could not
    merge the program into one mega-fusion, which is the contract's
    primary stencil-quality question."""

    _require_artifacts()
    hlo_path = PROFILER_DIR / f"{problem}_compiled_hlo.txt"
    if not hlo_path.exists():
        pytest.skip(f"{hlo_path} absent")
    text = hlo_path.read_text(errors="replace")
    fusion_count = len(re.findall(r"\bfusion\(", text))
    assert fusion_count == 1, (
        f"{problem}: compiled HLO has {fusion_count} fusion ops; expected exactly 1"
    )
    # The bakeoff is single-GPU, single-device: no SPMD/sharding ops.
    assert "all-reduce" not in text, f"{problem}: HLO contains all-reduce"
    assert "collective-permute" not in text, f"{problem}: HLO contains collective-permute"


@requires_gpu_toolchain
def test_compiled_hlo_uses_fp64() -> None:
    """Contract Non-Goal: no mixed precision. Both compiled HLOs must
    output f64; the column kernel must process f64 inputs throughout."""

    _require_artifacts()
    stencil_hlo = (PROFILER_DIR / "stencil_compiled_hlo.txt").read_text(errors="replace")
    column_hlo = (PROFILER_DIR / "column_compiled_hlo.txt").read_text(errors="replace")
    # Output element type of the root tuple/array must be f64.
    assert re.search(r"->\s*f64\[", stencil_hlo), "stencil HLO does not return f64"
    assert re.search(r"->\s*\(?f64\[", column_hlo), "column HLO does not return f64 (tuple element)"
    # f16/bf16 must not appear anywhere on the column compute path
    # (input/output dtypes are explicit f64 per the contract).
    assert "f16" not in column_hlo and "bf16" not in column_hlo, (
        "column HLO mentions reduced-precision types; contract forbids mixed precision"
    )


# --------------------------------------------------------------------------- #
# Backend snapshot & deliberate bug evidence                                  #
# --------------------------------------------------------------------------- #


def test_jax_backend_snapshot_records_gpu_with_cuda_device() -> None:
    """`data/scratch/m2-jax/jax_backend.json` is the bench's witness that
    the script saw a GPU backend; tester confirms it agrees with the
    profile JSON's claim."""

    snap = SCRATCH / "jax_backend.json"
    if not snap.exists():
        pytest.skip(f"{snap} absent")
    data = json.loads(snap.read_text())
    assert data["default_backend"] == "gpu"
    assert data["jax_version"] == "0.10.0"
    assert any(("cuda" in str(d).lower()) or ("gpu" in str(d).lower()) for d in data["devices"]), (
        f"jax_backend.json devices={data['devices']} has no CUDA/GPU entry"
    )


def test_deliberate_jax_bug_capture_present_and_diagnostic() -> None:
    """maintainability.md cites the captured shape-mismatch traceback;
    the file must exist and contain a real broadcast/shape error."""

    bug_path = PROFILER_DIR / "deliberate_jax_bug.txt"
    if not bug_path.exists():
        pytest.skip(f"{bug_path} absent")
    text = bug_path.read_text(errors="replace")
    lower = text.lower()
    assert "shape" in lower or "broadcast" in lower or "incompatible" in lower, (
        f"deliberate_jax_bug.txt does not contain a shape/broadcast diagnostic: {text[:200]}"
    )
    # The unsupported deliberate program should not have silently succeeded.
    assert "unexpected" not in lower, (
        "deliberate_jax_bug.txt indicates the invalid program ran successfully"
    )


def test_xla_flags_dump_path_was_active_during_run() -> None:
    """The runner exports XLA_FLAGS=--xla_dump_to=...; the jax_backend
    snapshot recorded the active env. Confirm the dump directory exists
    and contains the expected per-problem subtree."""

    snap = SCRATCH / "jax_backend.json"
    if not snap.exists():
        pytest.skip(f"{snap} absent")
    data = json.loads(snap.read_text())
    flags = data.get("xla_flags", "")
    assert "--xla_dump_to=" in flags, f"jax_backend.json xla_flags missing dump flag: {flags!r}"
    for problem in ("stencil", "column"):
        cubin = sorted(_xla_dump_dir(problem).rglob("*.cubin"))
        ptx = sorted(_xla_dump_dir(problem).rglob("*.ptx"))
        assert cubin or ptx, f"{problem}: XLA dump dir has no cubin/ptx artifact"


# --------------------------------------------------------------------------- #
# Correctness, maintainability, agent_success                                 #
# --------------------------------------------------------------------------- #


def test_correctness_json_passes_both_problems() -> None:
    _require_artifacts()
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["backend"] == "jax"
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
    assert raw["candidate"] == "jax"
    assert raw["backend_used"] == "jax"
    assert isinstance(raw["sprint_count"], int) and raw["sprint_count"] >= 1
    assert isinstance(raw["reviewer_rejections"], int) and raw["reviewer_rejections"] >= 0
    assert isinstance(raw["escalation_events"], int) and raw["escalation_events"] >= 0
    assert isinstance(raw["build_attempts"], int) and raw["build_attempts"] >= 1
    assert isinstance(raw["runtime_failures"], int) and raw["runtime_failures"] >= 0
    assert isinstance(raw["notes"], list)


def test_maintainability_markdown_within_budget_and_covers_topics() -> None:
    """Contract AC #7: <=300 words and must cover install complexity,
    error legibility on a deliberate bug, debugger story, and agent-
    iteration friction."""

    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text()
    words = re.findall(r"\S+", text)
    assert len(words) <= 300, f"maintainability.md word count {len(words)} > 300"
    lower = text.lower()
    assert "install" in lower or "venv" in lower or "wheel" in lower
    assert "error" in lower or "diagnostic" in lower or "legibility" in lower
    assert "debug" in lower or "jax.debug" in lower or "disable_jit" in lower
    assert "agent" in lower or "iteration" in lower or "friction" in lower


def test_maintainability_does_not_claim_pallas_or_mixed_precision() -> None:
    """Contract Non-Goals: no Pallas/Triton drop-down, no mixed precision.
    A maintainability narrative that quietly cites either is a contract
    violation."""

    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text().lower()
    assert "pallas" not in text, "maintainability.md mentions Pallas (forbidden by contract)"
    assert "mixed precision" not in text, (
        "maintainability.md mentions mixed precision (forbidden by contract)"
    )


# --------------------------------------------------------------------------- #
# Bench CLI behaviour (run inside the worker venv so jax is importable)        #
# --------------------------------------------------------------------------- #


def _run_bench(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [str(VENV_PY), "-m", "gpuwrf.backends.jax.bench", *args]
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(ROOT / "src"),
    }
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


@requires_gpu_toolchain
def test_bench_rejects_unknown_problem(tmp_path: Path) -> None:
    _require_venv()
    res = _run_bench(["--problem", "not_a_problem"])
    assert res.returncode != 0
    assert "invalid choice" in (res.stdout + res.stderr).lower() or "not_a_problem" in (
        res.stdout + res.stderr
    )


@requires_gpu_toolchain
def test_bench_rejects_missing_stencil_input(tmp_path: Path) -> None:
    _require_venv()
    missing = tmp_path / "does_not_exist.npz"
    res = _run_bench([
        "--problem", "stencil",
        "--stencil-fixture", str(missing),
        "--column-fixture", str(COLUMN_FIXTURE),
        "--scratch", str(tmp_path / "scratch"),
        "--artifact-dir", str(tmp_path / "art"),
        "--profiler-dir", str(tmp_path / "prof"),
    ])
    assert res.returncode != 0
    err = (res.stderr + res.stdout).lower()
    assert "no such" in err or "cannot" in err or "errno 2" in err or "stencil_fixture" in err or "filenotfound" in err


def test_bench_rejects_malformed_npz(tmp_path: Path) -> None:
    _require_venv()
    garbage = tmp_path / "garbage.npz"
    garbage.write_bytes(b"this is not a zip file at all\n")
    res = _run_bench([
        "--problem", "stencil",
        "--stencil-fixture", str(garbage),
        "--column-fixture", str(COLUMN_FIXTURE),
        "--scratch", str(tmp_path / "scratch"),
        "--artifact-dir", str(tmp_path / "art"),
        "--profiler-dir", str(tmp_path / "prof"),
    ])
    assert res.returncode != 0
    assert res.stderr or res.stdout, "no diagnostic emitted on malformed npz"


@requires_gpu_toolchain
def test_bench_detects_wrong_fixture_for_problem(tmp_path: Path) -> None:
    """Passing the column fixture as the stencil input must fail loudly
    (missing required array) rather than silently produce garbage."""

    _require_venv()
    if not COLUMN_FIXTURE.exists() or not STENCIL_FIXTURE.exists():
        pytest.skip("fixtures missing")
    res = _run_bench([
        "--problem", "stencil",
        "--stencil-fixture", str(COLUMN_FIXTURE),
        "--column-fixture", str(COLUMN_FIXTURE),
        "--scratch", str(tmp_path / "scratch"),
        "--artifact-dir", str(tmp_path / "art"),
        "--profiler-dir", str(tmp_path / "prof"),
    ])
    assert res.returncode != 0
    err = (res.stderr + res.stdout).lower()
    assert (
        "phi_initial" in err
        or "u_face" in err
        or "missing" in err
        or "keyerror" in err
        or "not a file in the archive" in err
    )


@requires_gpu_toolchain
def test_bench_end_to_end_stencil_reproduces_reference(tmp_path: Path) -> None:
    """End-to-end: invoke the bench on a fresh temp scratch and confirm
    its candidate matches the reference fixture. Catches a worker
    silently shipping stale npz files in the artifact directory."""

    _require_venv()
    if not STENCIL_FIXTURE.exists():
        pytest.skip("stencil fixture missing")
    scratch = tmp_path / "scratch"
    res = _run_bench([
        "--problem", "stencil",
        "--stencil-fixture", str(STENCIL_FIXTURE),
        "--column-fixture", str(COLUMN_FIXTURE),
        "--scratch", str(scratch),
        "--artifact-dir", str(tmp_path / "art"),
        "--profiler-dir", str(tmp_path / "prof"),
    ])
    assert res.returncode == 0, res.stderr or res.stdout
    out = scratch / "stencil_out.npz"
    assert out.exists() and out.stat().st_size > 0
    cmp = subprocess.run(
        [
            "python", "-m", "gpuwrf.validation.compare_fixture",
            "--manifest", str(STENCIL_MANIFEST),
            "--candidate", str(out),
            "--reference", str(STENCIL_FIXTURE),
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert cmp.returncode == 0, cmp.stderr
    assert json.loads(cmp.stdout)["pass"] is True


@requires_gpu_toolchain
def test_bench_end_to_end_column_reproduces_reference(tmp_path: Path) -> None:
    """Same shape as the stencil end-to-end test, for the column kernel."""

    _require_venv()
    if not COLUMN_FIXTURE.exists():
        pytest.skip("column fixture missing")
    scratch = tmp_path / "scratch"
    res = _run_bench([
        "--problem", "column",
        "--stencil-fixture", str(STENCIL_FIXTURE),
        "--column-fixture", str(COLUMN_FIXTURE),
        "--scratch", str(scratch),
        "--artifact-dir", str(tmp_path / "art"),
        "--profiler-dir", str(tmp_path / "prof"),
    ])
    assert res.returncode == 0, res.stderr or res.stdout
    out = scratch / "column_out.npz"
    assert out.exists() and out.stat().st_size > 0
    cmp = subprocess.run(
        [
            "python", "-m", "gpuwrf.validation.compare_fixture",
            "--manifest", str(COLUMN_MANIFEST),
            "--candidate", str(out),
            "--reference", str(COLUMN_FIXTURE),
        ],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert cmp.returncode == 0, cmp.stderr
    assert json.loads(cmp.stdout)["pass"] is True


@requires_gpu_toolchain
def test_bench_stencil_is_bitwise_reproducible(tmp_path: Path) -> None:
    """Two back-to-back stencil runs of the bench on the same input must
    produce byte-identical output. Catches the silent introduction of
    nondeterministic reductions or atomics in a future refactor."""

    _require_venv()
    if not STENCIL_FIXTURE.exists():
        pytest.skip("stencil fixture missing")
    digests = []
    for tag in ("a", "b"):
        scratch = tmp_path / tag
        res = _run_bench([
            "--problem", "stencil",
            "--stencil-fixture", str(STENCIL_FIXTURE),
            "--column-fixture", str(COLUMN_FIXTURE),
            "--scratch", str(scratch),
            "--artifact-dir", str(tmp_path / f"art_{tag}"),
            "--profiler-dir", str(tmp_path / f"prof_{tag}"),
        ])
        assert res.returncode == 0, res.stderr
        digests.append((scratch / "stencil_out.npz").read_bytes())
    assert digests[0] == digests[1], "stencil bench output differs between back-to-back runs"


# --------------------------------------------------------------------------- #
# Venv idempotency / pin enforcement                                          #
# --------------------------------------------------------------------------- #


@requires_gpu_toolchain
def test_venv_python_is_resolvable_and_pinned_jax() -> None:
    """AC #1/#2 demand a working jax==0.10.0 venv that the runner reuses."""

    _require_venv()
    res = subprocess.run(
        [str(VENV_PY), "-c", "import jax; print(jax.__version__); print(jax.default_backend())"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.strip().splitlines()
    assert out[0] == "0.10.0", f"venv jax version is {out[0]} != 0.10.0"
    assert out[1] == "gpu", f"venv jax default_backend is {out[1]} != gpu"


def test_pip_freeze_pins_match_contract() -> None:
    """`data/scratch/m2-jax/pip_freeze.txt` must record the
    jax==0.10.0/jaxlib==0.10.0 pins from the M2-S1 scout decision."""

    pip_freeze = SCRATCH / "pip_freeze.txt"
    if not pip_freeze.exists():
        pytest.skip("pip_freeze.txt absent")
    text = pip_freeze.read_text(errors="replace")
    assert re.search(r"^jax==0\.10\.0\s*$", text, flags=re.MULTILINE), (
        "pip_freeze.txt does not pin jax==0.10.0"
    )
    assert re.search(r"^jaxlib==0\.10\.0\s*$", text, flags=re.MULTILINE), (
        "pip_freeze.txt does not pin jaxlib==0.10.0"
    )
    # Forbidden non-goals: no Pallas/Triton extras or precision toggles.
    assert "triton" not in text.lower(), "pip_freeze includes triton (contract Non-Goal violation)"
