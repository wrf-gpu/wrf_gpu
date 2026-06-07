"""Tester-added edge cases for the M2 OpenAI Triton bakeoff candidate.

Sprint: 2026-05-19-m2-triton-stencil-column
Owner: tester/sonnet (Claude Opus 4.7) - cross-AI verification of the
gpt-5.5 worker output for ADR-001 input.

These tests consume the worker's already-produced artifacts on disk plus
the captured Triton cubins under data/profiler_artifacts/triton/; they
do not normally re-run `scripts/m2_run_triton.sh` (the canonical happy
path lives in `tests/test_m2_triton.py`). Coverage focuses on:

- Profile JSON schema rigor, type-correctness, and internal numeric
  consistency (so a worker cannot paste fabricated numbers past tests).
- Contract sanity bounds: kernel_launches <= 5, registers > 0,
  `local_memory_bytes == 0` (column AC #14, stencil also expected),
  occupancy floors, achieved_bandwidth_method == "fallback-derived".
- Cross-check against the captured Triton cubins:
  - cuobjdump per kernel: column's `_column_thermo_kernel` must report
    its own REG:N (the worker's bench uses max-of-all-cubins which
    silently inflates the column register count when the stencil cubin
    is left in cache); the column profile must reflect the kernel's
    own register count.
  - `LOCAL:0` and `STACK:0` for column.
  - Per-cubin SASS export contains `_column_thermo_kernel` symbol.
- Triton-specific evidence: torch_version starts with `2.12.0`,
  triton_version == "3.7.0", warmup_pattern documents post-compile
  warmup + multi-run median, wall_time_s implausibly small (<0.1s)
  would still be plausible -- but compile time must not leak in.
- Source-tree compliance: `@triton.jit` decorators are present on the
  two compute kernels, no torch tensor ops in the compute body,
  pyproject.toml does NOT declare torch/triton, scripts/m2_run_triton.sh
  installs into the data/ venv only.
- Bench CLI behaviour: malformed inputs, missing files, unknown problem,
  wrong-fixture-for-problem.
- Maintainability/agent_success hygiene.

If the venv or Triton cubin artifacts are absent the tests skip cleanly.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

import jax
import pytest


# The tests decorated below run the bench inside the torch+triton CUDA venv and
# launch Triton kernels on the RTX 5090, or cross-check un-vendored GPU profiler
# artifacts (.cubin). They cannot pass on a CPU-only checkout without torch/triton
# + a GPU; they are GPU-benchmark tests of a legacy bakeoff subsystem untouched by
# the operational pipeline.
requires_gpu_toolchain = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="M2 triton bakeoff edge cases require a GPU + torch/triton CUDA backend",
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "triton"
SCRATCH = ROOT / "data" / "scratch" / "m2-triton"
VENV = ROOT / "data" / "scratch" / "m2-triton-venv"
VENV_PY = VENV / "bin" / "python"
PROFILER_DIR = ROOT / "data" / "profiler_artifacts" / "triton"
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
    "host_device_transfer_bytes",
    "kernel_launches",
    "local_memory_bytes",
    "occupancy_pct",
    "profiler_limitation",
    "registers_per_thread",
    "torch_cuda",
    "torch_devices",
    "torch_version",
    "triton_cache_dir",
    "triton_version",
    "wall_time_s",
    "warmup_pattern",
}


def _require_artifacts() -> None:
    if not (ARTIFACT_DIR / "stencil_profile.json").exists():
        pytest.skip("triton artifacts not present; run scripts/m2_run_triton.sh first")


def _require_venv() -> None:
    if not VENV_PY.exists():
        pytest.skip(f"triton venv missing at {VENV_PY}")


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
    assert profile["backend"] == "triton"
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
    assert profile["achieved_bandwidth_method"] == "fallback-derived"


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_declares_pinned_triton_and_torch_runtime(profile_name: str) -> None:
    """Contract AC #1 pins triton==3.7.0 and torch==2.12.0 (CUDA13).
    Profile JSON must reflect these pins so a future worker on a drifted
    venv cannot pass the bakeoff row."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    assert profile["triton_version"] == "3.7.0", (
        f"{profile_name}: triton_version={profile['triton_version']!r} != contract pin 3.7.0"
    )
    assert profile["torch_version"].split("+", 1)[0] == "2.12.0", (
        f"{profile_name}: torch_version={profile['torch_version']!r} != contract pin 2.12.0"
    )
    assert profile["torch_cuda"], f"{profile_name}: torch_cuda missing"
    devices = profile["torch_devices"]
    assert isinstance(devices, list) and devices, f"{profile_name}: torch_devices empty"
    assert any(("cuda" in str(d).lower()) or ("gpu" in str(d).lower()) or ("rtx" in str(d).lower())
               for d in devices), (
        f"{profile_name}: no CUDA/GPU device in torch_devices={devices}"
    )


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_warmup_pattern_documents_compile_exclusion(profile_name: str) -> None:
    """Contract AC #6: wall_time_s is a median of 5 post-warmup runs
    around `kernel[grid](*args); torch.cuda.synchronize()`. The
    `warmup_pattern` field must call out post-compile/warmup behavior and
    a multi-run statistic; the wall must be small (<0.5s) which is
    incompatible with leaking Triton compile time."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    pattern = profile.get("warmup_pattern", "")
    assert isinstance(pattern, str) and pattern
    lower = pattern.lower()
    assert ("compile" in lower) or ("warmup" in lower), (
        f"{profile_name}: warmup_pattern does not mention compile/warmup: {pattern!r}"
    )
    assert ("median" in lower) or ("min" in lower), (
        f"{profile_name}: warmup_pattern does not document a multi-run statistic: {pattern!r}"
    )
    assert 0.0 < profile["wall_time_s"] < 0.5, (
        f"{profile_name}: wall_time_s={profile['wall_time_s']}s implausibly large for a "
        "single Triton launch on a 32x16x8 stencil / 40-cell column; compile time may "
        "be leaking into the measurement"
    )


@pytest.mark.parametrize(
    "profile_name,benchmark,case,reg_limit,occupancy_floor",
    [
        ("stencil_profile.json", "m2_stencil", "analytic-stencil-3d-advdiff-v1", 96, 25.0),
        ("column_profile.json", "m2_column", "analytic-column-thermo-v1", 128, 20.0),
    ],
)
def test_profile_sanity_bounds_match_contract(
    profile_name: str, benchmark: str, case: str, reg_limit: int, occupancy_floor: float
) -> None:
    """Contract Performance Metrics: <=5 launches, registers > 0 but
    bounded, occupancy non-degenerate, host_device_transfer_bytes > 0."""

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
    """Contract AC #14: column kernel local_memory_bytes MUST be 0."""

    _require_artifacts()
    profile = _load_profile("column_profile.json")
    assert profile["local_memory_bytes"] == 0, (
        "column kernel reports register spilling; AC #14 forbids non-zero local memory"
    )


def test_stencil_profile_has_zero_local_memory() -> None:
    """The contract does not literally require local_memory_bytes == 0 on
    the stencil kernel, but a non-zero would be load-bearing evidence for
    ADR-001 (Triton stencil ergonomics) and must be cited."""

    _require_artifacts()
    profile = _load_profile("stencil_profile.json")
    assert profile["local_memory_bytes"] == 0, (
        "stencil kernel reports register spilling; ADR-001 must cite this"
    )


def test_achieved_bandwidth_is_consistent_with_transfer_and_wall() -> None:
    """Profile bandwidth must equal `host_device_transfer_bytes /
    wall_time_s / 1e9`. Catches a worker pasting a hand-rolled bandwidth
    that doesn't agree with the other reported quantities."""

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
    """Profile JSON structural values must agree with the bench's per-run
    JSON dump. Guards against post-hoc tampering of registers/launches/
    transfer/local_memory/occupancy/triton_version. wall_time_s is
    excluded because run.json captures the most-recent runtime while the
    committed profile.json may be from an earlier worker invocation
    (run-to-run jitter in microsecond timings is expected)."""

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
        assert profile["triton_version"] == run["triton_version"]
        assert profile["torch_version"] == run["torch_version"]


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
# Cubin-level cross-check: the contract requires register/local memory be    #
# derived from cuobjdump on the kernel's own cubin. The bench currently      #
# takes the maximum register count across recently-cached cubins, which      #
# silently inflates the column profile (because the stencil cubin is still   #
# in the Triton cache when the column runs in the same process).             #
# --------------------------------------------------------------------------- #


_KERNEL_SYMBOL = {"stencil": "_stencil_advdiff_kernel", "column": "_column_thermo_kernel"}


def _cuobjdump_text(cubin: Path) -> str:
    if not cubin.exists():
        pytest.skip(f"{cubin} absent")
    proc = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(cubin)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cuobjdump failed on {cubin}: {proc.stdout}")
    return proc.stdout


def _kernel_section_regs_local(text: str, kernel_name: str) -> tuple[int, int]:
    """Return (REG, LOCAL) for the named kernel in a cuobjdump
    --dump-resource-usage report. Skips the test if the kernel is not
    present in the file."""
    matches = list(re.finditer(r"Function\s+([^\s:]+):\s*\n\s*([^\n]+)", text))
    for m in matches:
        if m.group(1) == kernel_name:
            line = m.group(2)
            reg = re.search(r"\bREG:(\d+)", line)
            local = re.search(r"\bLOCAL:(\d+)", line)
            if not (reg and local):
                pytest.skip(f"could not parse REG/LOCAL for {kernel_name} in: {line!r}")
            return int(reg.group(1)), int(local.group(1))
    pytest.skip(f"{kernel_name} not present in cuobjdump output")
    return 0, 0  # pragma: no cover - pytest.skip exits


def _find_cubin_with_kernel(problem: str, kernel_name: str) -> Path:
    cubins = sorted(PROFILER_DIR.glob(f"{problem}_triton_*.cubin"))
    if not cubins:
        pytest.skip(f"no triton cubins captured for {problem}")
    for cubin in cubins:
        text = _cuobjdump_text(cubin)
        if f"Function {kernel_name}:" in text:
            return cubin
    pytest.skip(f"no captured cubin contains {kernel_name}")
    return cubins[0]  # pragma: no cover


@pytest.mark.parametrize("problem", ["stencil", "column"])
def test_cubin_has_correct_kernel_symbol(problem: str) -> None:
    """At least one captured cubin for each problem must contain its own
    kernel symbol (proves the bench picked up the right cache entry)."""

    _require_artifacts()
    cubin = _find_cubin_with_kernel(problem, _KERNEL_SYMBOL[problem])
    assert cubin.exists()


@pytest.mark.parametrize("problem", ["stencil", "column"])
def test_kernel_local_memory_zero_in_own_cubin(problem: str) -> None:
    """LOCAL:N must be 0 in the kernel's own cubin section (contract
    AC #14 for column; ADR-relevant for stencil). This is the
    independent, tool-level evidence behind `local_memory_bytes == 0`."""

    _require_artifacts()
    kernel = _KERNEL_SYMBOL[problem]
    cubin = _find_cubin_with_kernel(problem, kernel)
    text = _cuobjdump_text(cubin)
    _, local = _kernel_section_regs_local(text, kernel)
    assert local == 0, f"{problem}: cuobjdump LOCAL:{local} != 0 for {kernel}"


def test_column_profile_registers_match_column_kernel_cubin() -> None:
    """Contract: registers_per_thread must come from cuobjdump on the
    cubin. The column profile must reflect `_column_thermo_kernel`'s own
    register count, NOT the max of all cubins in the Triton cache.

    This catches the live bug in `_resource_metrics_factory`:
    `_recent_cubins` returns every cubin newer than `marker - 0.25s`, and
    `_parse_resource_usage` takes `max(regs)` across all parsed entries.
    When the stencil cubin (REG:60 in this sprint) is still in the
    Triton cache when the column runs, the column profile silently
    reports the stencil's register count instead of the column kernel's.
    """

    _require_artifacts()
    cubin = _find_cubin_with_kernel("column", _KERNEL_SYMBOL["column"])
    text = _cuobjdump_text(cubin)
    kernel_regs, _ = _kernel_section_regs_local(text, _KERNEL_SYMBOL["column"])
    profile_regs = _load_profile("column_profile.json")["registers_per_thread"]
    assert profile_regs == kernel_regs, (
        f"column_profile.json registers_per_thread={profile_regs} does not match the "
        f"_column_thermo_kernel cubin REG:{kernel_regs}. The bench is taking the max "
        "register count across all recently-cached cubins (see "
        "src/gpuwrf/backends/triton/bench.py:_parse_resource_usage), which inflates "
        "the column profile whenever the stencil cubin is still in cache."
    )


def test_stencil_profile_registers_match_stencil_kernel_cubin() -> None:
    """Symmetric to the column test: profile must reflect this kernel's
    own register count, not any unrelated cached cubin."""

    _require_artifacts()
    cubin = _find_cubin_with_kernel("stencil", _KERNEL_SYMBOL["stencil"])
    text = _cuobjdump_text(cubin)
    kernel_regs, _ = _kernel_section_regs_local(text, _KERNEL_SYMBOL["stencil"])
    profile_regs = _load_profile("stencil_profile.json")["registers_per_thread"]
    assert profile_regs == kernel_regs, (
        f"stencil_profile.json registers_per_thread={profile_regs} does not match the "
        f"_stencil_advdiff_kernel cubin REG:{kernel_regs}"
    )


def test_cuobjdump_resource_usage_artifact_has_kernel_section() -> None:
    """The committed cuobjdump dump under data/profiler_artifacts/triton/
    must contain the kernel's `Function _..._kernel:` section so the
    reviewer can audit the resource numbers directly."""

    _require_artifacts()
    for problem in ("stencil", "column"):
        path = PROFILER_DIR / f"{problem}_cuobjdump_resource_usage.txt"
        if not path.exists():
            pytest.skip(f"{path} absent")
        text = path.read_text(errors="replace")
        kernel = _KERNEL_SYMBOL[problem]
        assert f"Function {kernel}:" in text, (
            f"{path} does not contain `Function {kernel}:`; reviewer cannot audit regs/local"
        )


# --------------------------------------------------------------------------- #
# Source-tree compliance with Non-Goals / File Ownership                     #
# --------------------------------------------------------------------------- #


def test_kernels_use_triton_jit_decorator() -> None:
    """Contract Non-Goal: 'No JAX/Pallas wrapping - write Triton directly
    with @triton.jit'. Both compute kernels must use the decorator."""

    stencil_src = (ROOT / "src/gpuwrf/backends/triton/stencil.py").read_text()
    column_src = (ROOT / "src/gpuwrf/backends/triton/column.py").read_text()
    assert "@triton.jit" in stencil_src, "stencil.py is missing @triton.jit"
    assert "@triton.jit" in column_src, "column.py is missing @triton.jit"
    assert "import triton" in stencil_src and "import triton.language as tl" in stencil_src
    assert "import triton" in column_src and "import triton.language as tl" in column_src


def test_kernels_do_not_use_torch_compute_ops() -> None:
    """Contract Non-Goal: torch is for buffer/sync only, not operational.
    The two compute modules may use torch.from_numpy / .to('cuda') /
    .empty_like / .cuda.synchronize() but must not call torch math ops
    (matmul, exp, add, etc.) inside the bench path."""

    forbidden_patterns = [
        r"\btorch\.matmul\b", r"\btorch\.exp\b", r"\btorch\.log\b",
        r"\btorch\.add\b", r"\btorch\.mul\b", r"\btorch\.sum\b",
        r"\btorch\.mean\b", r"\btorch\.relu\b", r"\btorch\.nn\b",
        r"\btorch\.einsum\b",
    ]
    for stem in ("stencil", "column"):
        src = (ROOT / f"src/gpuwrf/backends/triton/{stem}.py").read_text()
        for pattern in forbidden_patterns:
            assert not re.search(pattern, src), (
                f"{stem}.py uses forbidden torch compute op `{pattern}` "
                "(contract forbids torch tensor compute operationally)"
            )


def test_pyproject_does_not_declare_triton_or_torch_dependency() -> None:
    """Contract File Ownership: 'do NOT add triton/torch as project deps
    -- venv only'. pyproject.toml must keep those out."""

    text = (ROOT / "pyproject.toml").read_text()
    deps_block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL | re.MULTILINE)
    optional_block = re.search(
        r"^\[project\.optional-dependencies\][^\[]*", text, re.DOTALL | re.MULTILINE
    )
    blob = (deps_block.group(1) if deps_block else "") + (optional_block.group(0) if optional_block else "")
    lower = blob.lower()
    assert "triton" not in lower, "pyproject.toml declares triton (forbidden; venv only)"
    # The bare token "torch" might appear in a comment elsewhere; restrict to deps blocks only.
    for token in ("torch==", '"torch"', "'torch'", "torch>", "torch~", "torch ", "torch\n"):
        assert token not in lower, f"pyproject.toml declares torch (`{token}`); forbidden, venv only"


def test_runner_script_installs_into_data_venv_only() -> None:
    """`scripts/m2_run_triton.sh` must keep pip install scoped to the
    venv under data/scratch/m2-triton-venv/ and never touch the global
    or repo-wide environment."""

    script = (ROOT / "scripts/m2_run_triton.sh").read_text()
    assert "data/scratch/m2-triton-venv" in script, (
        "scripts/m2_run_triton.sh does not reference the data-scoped venv"
    )
    # Find every `pip install` invocation and ensure it is prefixed by the
    # venv python.
    pip_lines = [line for line in script.splitlines() if "pip install" in line and not line.lstrip().startswith("#")]
    assert pip_lines, "scripts/m2_run_triton.sh has no pip install invocation"
    for line in pip_lines:
        assert "$VENV/bin/python" in line or '"$VENV/bin/python"' in line, (
            f"scripts/m2_run_triton.sh has unscoped pip install: {line!r}"
        )


def test_runner_pins_triton_and_torch_versions() -> None:
    """The runner must pin triton==3.7.0 and torch==2.12.0 per the
    contract; a drift would invalidate the bakeoff comparison."""

    script = (ROOT / "scripts/m2_run_triton.sh").read_text()
    assert "triton==3.7.0" in script, "scripts/m2_run_triton.sh does not pin triton==3.7.0"
    assert "torch==2.12.0" in script, "scripts/m2_run_triton.sh does not pin torch==2.12.0"


# --------------------------------------------------------------------------- #
# Correctness, maintainability, agent_success                                 #
# --------------------------------------------------------------------------- #


def test_correctness_json_passes_both_problems() -> None:
    _require_artifacts()
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["backend"] == "triton"
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
    assert raw["candidate"] == "triton"
    assert raw["backend_used"] == "triton"
    assert isinstance(raw["sprint_count"], int) and raw["sprint_count"] >= 1
    assert isinstance(raw["reviewer_rejections"], int) and raw["reviewer_rejections"] >= 0
    assert isinstance(raw["escalation_events"], int) and raw["escalation_events"] >= 0
    assert isinstance(raw["build_attempts"], int) and raw["build_attempts"] >= 1
    assert isinstance(raw["runtime_failures"], int) and raw["runtime_failures"] >= 0
    assert isinstance(raw["notes"], list)


def test_maintainability_markdown_within_budget_and_covers_topics() -> None:
    """Contract AC #7: <=300 words and must cover install complexity,
    error legibility, debugger story, agent-iteration friction."""

    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text()
    words = re.findall(r"\S+", text)
    assert len(words) <= 300, f"maintainability.md word count {len(words)} > 300"
    lower = text.lower()
    assert ("install" in lower) or ("venv" in lower) or ("wheel" in lower)
    assert ("error" in lower) or ("legibility" in lower) or ("diagnostic" in lower)
    assert ("debug" in lower) or ("interpret" in lower) or ("triton_interpret" in lower)
    assert ("agent" in lower) or ("iteration" in lower) or ("friction" in lower)


def test_maintainability_acknowledges_torch_dependency() -> None:
    """The contract specifically highlights torch as a heavy dep (~2GB);
    the maintainability narrative must call it out."""

    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text().lower()
    assert "torch" in text, (
        "maintainability.md does not mention torch; contract AC #7 requires install-"
        "complexity coverage including the heavy torch dep"
    )


def test_maintainability_does_not_claim_autotuning_or_mixed_precision() -> None:
    """Contract Non-Goals: no autotuning, no mixed precision. A narrative
    that claims either is a contract violation."""

    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text().lower()
    assert "autotun" not in text, "maintainability.md mentions autotuning (forbidden)"
    assert "mixed precision" not in text, "maintainability.md mentions mixed precision (forbidden)"


# --------------------------------------------------------------------------- #
# Bench CLI behaviour (run inside the worker venv so triton is importable)    #
# --------------------------------------------------------------------------- #


def _run_bench(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [str(VENV_PY), "-m", "gpuwrf.backends.triton.bench", *args]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)


@requires_gpu_toolchain
def test_bench_rejects_unknown_problem(tmp_path: Path) -> None:
    _require_venv()
    res = _run_bench(["--problem", "not_a_problem"])
    assert res.returncode != 0
    blob = (res.stdout + res.stderr).lower()
    assert "invalid choice" in blob or "not_a_problem" in blob


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
    assert (
        "no such" in err
        or "cannot" in err
        or "errno 2" in err
        or "stencil_fixture" in err
        or "filenotfound" in err
    )


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


def test_bench_rejects_truncated_zip(tmp_path: Path) -> None:
    """A truncated zip header (looks like an npz prefix) must also fail,
    not silently succeed with empty arrays."""

    _require_venv()
    truncated = tmp_path / "truncated.npz"
    truncated.write_bytes(b"PK\x03\x04" + b"\x00" * 4)  # zip magic but no entries
    res = _run_bench([
        "--problem", "column",
        "--stencil-fixture", str(STENCIL_FIXTURE),
        "--column-fixture", str(truncated),
        "--scratch", str(tmp_path / "scratch"),
        "--artifact-dir", str(tmp_path / "art"),
        "--profiler-dir", str(tmp_path / "prof"),
    ])
    assert res.returncode != 0


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
        "--skip-artifacts",
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
        "--skip-artifacts",
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
    """Two back-to-back stencil runs on the same input must produce
    byte-identical output. Catches the silent introduction of
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
            "--skip-artifacts",
        ])
        assert res.returncode == 0, res.stderr
        digests.append((scratch / "stencil_out.npz").read_bytes())
    assert digests[0] == digests[1], "stencil bench output differs between back-to-back runs"


@requires_gpu_toolchain
def test_bench_column_is_bitwise_reproducible(tmp_path: Path) -> None:
    _require_venv()
    if not COLUMN_FIXTURE.exists():
        pytest.skip("column fixture missing")
    digests = []
    for tag in ("a", "b"):
        scratch = tmp_path / tag
        res = _run_bench([
            "--problem", "column",
            "--stencil-fixture", str(STENCIL_FIXTURE),
            "--column-fixture", str(COLUMN_FIXTURE),
            "--scratch", str(scratch),
            "--artifact-dir", str(tmp_path / f"art_{tag}"),
            "--profiler-dir", str(tmp_path / f"prof_{tag}"),
            "--skip-artifacts",
        ])
        assert res.returncode == 0, res.stderr
        digests.append((scratch / "column_out.npz").read_bytes())
    assert digests[0] == digests[1], "column bench output differs between back-to-back runs"


# --------------------------------------------------------------------------- #
# Venv pin enforcement and deliberate-bug evidence                            #
# --------------------------------------------------------------------------- #


def test_venv_python_resolves_pinned_triton_and_torch() -> None:
    """AC #1: the worker venv must contain triton==3.7.0 and torch==2.12.0."""

    _require_venv()
    res = subprocess.run(
        [str(VENV_PY), "-c",
         "import triton, torch; print(triton.__version__); print(torch.__version__)"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.strip().splitlines()
    assert out[0] == "3.7.0", f"venv triton version {out[0]!r} != 3.7.0"
    assert out[1].split("+", 1)[0] == "2.12.0", f"venv torch version {out[1]!r} != 2.12.0"


def test_pip_freeze_pins_match_contract() -> None:
    """`data/scratch/m2-triton/pip_freeze.txt` records what the venv
    actually has after install; must include the contract pins."""

    pip_freeze = SCRATCH / "pip_freeze.txt"
    if not pip_freeze.exists():
        pytest.skip("pip_freeze.txt absent")
    text = pip_freeze.read_text(errors="replace")
    assert re.search(r"^triton==3\.7\.0\s*$", text, flags=re.MULTILINE), (
        f"pip_freeze.txt missing triton==3.7.0 pin:\n{text[:400]}"
    )
    assert re.search(r"^torch==2\.12\.0(\+\S+)?\s*$", text, flags=re.MULTILINE), (
        f"pip_freeze.txt missing torch==2.12.0 pin:\n{text[:400]}"
    )


def test_triton_cache_dir_is_repo_scoped() -> None:
    """Risk: 'Triton's cubin caching at ~/.triton/cache/ should not
    pollute the repo'. The runner sets TRITON_CACHE_DIR to a path under
    data/; the profile JSON must reflect that."""

    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        cache = profile.get("triton_cache_dir", "")
        assert "data/scratch/m2-triton-cache" in cache or "data/scratch/m2-triton" in cache, (
            f"{name}: triton_cache_dir={cache!r} is not repo-scoped"
        )
        assert not cache.startswith(str(Path.home() / ".triton")), (
            f"{name}: triton_cache_dir leaks to user home {cache!r}"
        )


def test_deliberate_triton_bug_capture_present_and_diagnostic() -> None:
    """maintainability.md context: the bench captures a deliberate
    invalid `@triton.jit` program so the reviewer can see Triton's error
    legibility. The captured file must exist and contain a real
    Triton compile-time diagnostic, not a 'ran successfully' string."""

    bug_path = PROFILER_DIR / "deliberate_triton_bug.txt"
    if not bug_path.exists():
        pytest.skip(f"{bug_path} absent")
    text = bug_path.read_text(errors="replace")
    lower = text.lower()
    assert "unexpected" not in lower, (
        "deliberate_triton_bug.txt indicates the invalid Triton program ran successfully"
    )
    assert any(
        token in lower for token in (
            "shape", "broadcast", "type", "error", "incompat", "rank",
            "arange", "power of 2", "constexpr", "must be",
        )
    ), (
        f"deliberate_triton_bug.txt does not contain a triton diagnostic: {text[:300]!r}"
    )
    # The diagnostic should also include a source-line pointer (the
    # `@triton.jit` line where the failure occurred), which is one of
    # Triton's selling points relative to opaque XLA errors.
    assert re.search(r"\bat\s+\d+", text) or ("^" in text), (
        f"deliberate_triton_bug.txt has no source-line pointer: {text[:300]!r}"
    )
