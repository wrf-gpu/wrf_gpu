"""Tester-added edge cases for the M2 cupy_or_numba bakeoff candidate.

Sprint: 2026-05-19-m2-cupy-stencil-column
Owner: tester/sonnet (Claude Opus 4.7) — cross-AI verification of gpt-5.5 worker.

Two flavours of test live here:

1. Artifact-only tests (no GPU required): they read the worker's already-
   produced JSON / Markdown files and verify schema rigor, internal
   consistency of the numeric fields, the cross-AI invariants from the
   sprint contract (real raw-CUDA source strings, venv pin, column
   ``local_memory_bytes == 0``), and the deliberate-kernel-bug capture.

2. GPU-execution tests: they re-compile the RawKernels via the sprint
   venv (so kernel attributes are read by the tester process, not just
   trusted from the worker's JSON), check kernel reproducibility, and
   exercise malformed-input / missing-file paths.

All GPU-execution tests skip cleanly when CuPy is unimportable, when
the sprint venv is missing, or when the artifact set is incomplete -
the canonical happy-path assertions still live in tests/test_m2_cupy.py.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "m2" / "cupy_or_numba"
SCRATCH = ROOT / "data" / "scratch" / "m2-cupy"
PROFILER_DIR = ROOT / "data" / "profiler_artifacts" / "cupy_or_numba"
VENV_PY = ROOT / "data" / "scratch" / "m2-cupy-venv" / "bin" / "python"
STENCIL_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-stencil-3d-advdiff-v1.npz"
COLUMN_FIXTURE = ROOT / "fixtures" / "samples" / "analytic-column-thermo-v1.npz"

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
    "wall_time_s",
}


def _require_artifacts() -> None:
    if not (ARTIFACT_DIR / "stencil_profile.json").exists():
        pytest.skip("cupy_or_numba artifacts not present; run scripts/m2_run_cupy.sh first")


def _require_venv_python() -> Path:
    if not VENV_PY.exists():
        pytest.skip(f"sprint venv missing at {VENV_PY}; run scripts/m2_run_cupy.sh first")
    return VENV_PY


def _load_profile(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text())


# --------------------------------------------------------------------------- #
# 1. Artifact-only tests (no GPU)                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile_name", ["stencil_profile.json", "column_profile.json"])
def test_profile_schema_keys_and_types(profile_name: str) -> None:
    _require_artifacts()
    profile = _load_profile(profile_name)
    missing = PROFILE_REQUIRED_KEYS - profile.keys()
    assert not missing, f"{profile_name} missing required schema keys: {sorted(missing)}"
    assert profile["backend"] == "cupy"
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
    assert isinstance(profile["achieved_bandwidth_method"], str) and profile["achieved_bandwidth_method"]
    assert isinstance(profile["profiler_limitation"], str) and profile["profiler_limitation"]
    assert isinstance(profile["artifact_paths"], list)
    assert all(isinstance(p, str) and p for p in profile["artifact_paths"])


@pytest.mark.parametrize(
    "profile_name,benchmark,case,reg_limit",
    [
        ("stencil_profile.json", "m2_stencil", "analytic-stencil-3d-advdiff-v1", 64),
        ("column_profile.json", "m2_column", "analytic-column-thermo-v1", 128),
    ],
)
def test_profile_sanity_bounds_match_contract(
    profile_name: str, benchmark: str, case: str, reg_limit: int
) -> None:
    """Sprint contract Performance Metrics sanity bounds (per-problem)."""

    _require_artifacts()
    profile = _load_profile(profile_name)
    assert profile["benchmark"] == benchmark
    assert profile["case"] == case
    assert 0.0 <= profile["wall_time_s"] <= 5.0
    assert 1 <= profile["kernel_launches"] <= 5, (
        f"{profile_name}: kernel_launches={profile['kernel_launches']} "
        "violates contract Performance Metrics bound (≤5)"
    )
    assert profile["host_device_transfer_bytes"] > 0
    assert profile["registers_per_thread"] > 0
    assert profile["registers_per_thread"] <= reg_limit, (
        f"{profile_name}: registers_per_thread={profile['registers_per_thread']} "
        f"exceeds sanity bound {reg_limit}"
    )
    assert 0.0 <= profile["occupancy_pct"] <= 100.0


def test_column_profile_has_zero_local_memory() -> None:
    """Contract AC #13: column kernel must have ``local_memory_bytes == 0``."""

    _require_artifacts()
    profile = _load_profile("column_profile.json")
    assert profile["local_memory_bytes"] == 0, (
        "column kernel reports register spilling; AC #13 requires zero local memory"
    )


def test_stencil_local_memory_within_reason() -> None:
    """Stencil is allowed non-zero local memory (the contract only requires
    zero for the column kernel) but if it explodes that signals a regression."""

    _require_artifacts()
    profile = _load_profile("stencil_profile.json")
    assert profile["local_memory_bytes"] <= 1024, (
        f"stencil local_memory_bytes={profile['local_memory_bytes']} looks like "
        "register spilling; investigate"
    )


def test_achieved_bandwidth_is_consistent_with_transfer_and_wall() -> None:
    """Bandwidth must equal host_device_transfer_bytes / wall_time_s / 1e9.

    Catches fabricated bandwidth numbers that disagree with the other reported
    quantities — important on the fallback-derived path.
    """

    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        assert profile["achieved_bandwidth_method"] == "fallback-derived"
        if profile["wall_time_s"] <= 0.0:
            continue
        expected = profile["host_device_transfer_bytes"] / profile["wall_time_s"] / 1.0e9
        assert math.isclose(
            profile["achieved_bandwidth_gbps"], expected, rel_tol=1e-3, abs_tol=1e-6
        ), (
            f"{name}: achieved_bandwidth_gbps={profile['achieved_bandwidth_gbps']} "
            f"inconsistent with transfer/wall computation {expected}"
        )


def test_host_device_transfer_meets_fixture_byte_floor() -> None:
    """The reported H2D+D2H bytes must at least cover the fixture inputs/outputs
    that the candidate has to move.  A worker who paints a tiny synthetic number
    here would slip past the schema test but be caught by this floor.
    """

    _require_artifacts()
    if not STENCIL_FIXTURE.exists() or not COLUMN_FIXTURE.exists():
        pytest.skip("fixture samples not present")

    stencil_arrays = np.load(STENCIL_FIXTURE)
    stencil_floor = (
        stencil_arrays["phi_initial"].nbytes
        + stencil_arrays["u_face"].nbytes
        + stencil_arrays["v_face"].nbytes
        + stencil_arrays["w_face"].nbytes
        + stencil_arrays["phi_initial"].nbytes  # phi_next == same shape/dtype
    )
    column_arrays = np.load(COLUMN_FIXTURE)
    col_inp = (
        column_arrays["temperature_initial"].nbytes
        + column_arrays["qv_initial"].nbytes
        + column_arrays["pressure_initial"].nbytes
        + column_arrays["saturation_qv"].nbytes
    )
    column_floor = col_inp + 4 * column_arrays["temperature_initial"].nbytes

    stencil_profile = _load_profile("stencil_profile.json")
    column_profile = _load_profile("column_profile.json")
    assert stencil_profile["host_device_transfer_bytes"] >= stencil_floor, (
        f"stencil transfer={stencil_profile['host_device_transfer_bytes']} < floor {stencil_floor}"
    )
    assert column_profile["host_device_transfer_bytes"] >= column_floor, (
        f"column transfer={column_profile['host_device_transfer_bytes']} < floor {column_floor}"
    )


def test_profile_artifact_paths_are_relative_and_exist() -> None:
    _require_artifacts()
    for name in ("stencil_profile.json", "column_profile.json"):
        profile = _load_profile(name)
        assert profile["artifact_paths"], f"{name}: artifact_paths is empty"
        for p in profile["artifact_paths"]:
            assert not Path(p).is_absolute(), f"{name}: artifact_paths must be relative ({p!r})"
            assert (ROOT / p).exists(), f"{name}: missing referenced artifact {p}"


def test_correctness_json_passes_both_problems() -> None:
    _require_artifacts()
    correctness = json.loads((ARTIFACT_DIR / "correctness.json").read_text())
    assert correctness["pass"] is True
    assert correctness["backend"] == "cupy"
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
    assert raw["candidate"] == "cupy_or_numba"
    assert raw["backend_used"] == "cupy"
    assert raw["sprint_count"] == 1
    assert raw["escalation_events"] == 0
    assert "fallbacks_used" in raw
    assert isinstance(raw["fallbacks_used"], list)


def test_maintainability_markdown_is_within_budget_and_covers_topics() -> None:
    _require_artifacts()
    text = (ARTIFACT_DIR / "maintainability.md").read_text()
    words = re.findall(r"\S+", text)
    assert len(words) <= 300, f"maintainability.md word count {len(words)} > 300"
    # Contract AC #6 — install complexity, error legibility, debugger story, agent friction.
    lowered = text.lower()
    assert "venv" in lowered or "pip" in lowered, "maintainability.md: install section missing"
    assert "error" in lowered or "nvrtc" in lowered or "compile" in lowered, (
        "maintainability.md: error-legibility section missing"
    )
    assert (
        "cuda-gdb" in lowered
        or "nsight" in lowered
        or "profiler" in lowered
        or "debug" in lowered
    ), "maintainability.md: debugger/profiler section missing"


def test_deliberate_kernel_bug_capture_has_compile_error() -> None:
    """Contract AC #6(b): worker captured a deliberate kernel bug; the capture must
    look like a real NVRTC compile error, not a hand-written success placeholder.
    """

    _require_artifacts()
    bug_path = PROFILER_DIR / "deliberate_kernel_bug.txt"
    assert bug_path.exists(), f"{bug_path} missing — worker did not capture the deliberate bug"
    text = bug_path.read_text()
    assert "unexpected" not in text.lower(), (
        "deliberate kernel bug compiled successfully — the bug capture is invalid"
    )
    assert "error" in text.lower(), "deliberate kernel bug capture has no error line"


# --------------------------------------------------------------------------- #
# 2. Cross-AI source / packaging invariants                                    #
# --------------------------------------------------------------------------- #


def test_kernel_source_strings_are_real_raw_cuda() -> None:
    """Cross-AI verification (a): kernels must be real raw CUDA C, not
    idiomatic CuPy NumPy-style ops.  Worker promised raw kernels in the
    contract Non-Goals; this guards against a future regression that
    swaps them for ``cp.einsum`` or similar.

    Reads the source files as text so the test can run without CuPy
    installed on the host Python (the kernels live in modules that
    import cupy at module top).
    """

    cupy_pkg = ROOT / "src" / "gpuwrf" / "backends" / "cupy"
    for module_name in ("stencil.py", "column.py"):
        text = (cupy_pkg / module_name).read_text()
        assert "cp.RawKernel" in text, f"{module_name}: not using cp.RawKernel"
        assert "__global__" in text, f"{module_name} kernel source missing __global__"
        assert 'extern "C"' in text, f"{module_name} kernel source missing extern \"C\" linkage"
        assert "threadIdx" in text, f"{module_name} kernel source missing threadIdx"


def test_no_idiomatic_cupy_numpy_ops_in_kernel_modules() -> None:
    """Cross-AI verification (a) — paranoia level 2.  Scan the kernel modules
    for tell-tale idiomatic CuPy NumPy-style calls (``cp.matmul``, ``cp.einsum``,
    ``cp.ElementwiseKernel``, ``cp.ReductionKernel``).  Trivial helpers like
    ``cp.asarray`` and ``cp.empty_like`` are fine — those are necessary
    host-to-device staging.
    """

    forbidden = (
        "cp.matmul",
        "cp.einsum",
        "cp.tensordot",
        "cp.fft.",
        "cp.ElementwiseKernel",
        "cp.ReductionKernel",
        "cupy.ElementwiseKernel",
        "cupy.ReductionKernel",
    )
    for module_name in ("stencil.py", "column.py"):
        text = (ROOT / "src" / "gpuwrf" / "backends" / "cupy" / module_name).read_text()
        for needle in forbidden:
            assert needle not in text, (
                f"{module_name} uses idiomatic CuPy op {needle!r}; contract requires raw kernels"
            )


def test_venv_python_is_pinned_to_cupy_cuda13x_14_0_1() -> None:
    """Cross-AI verification (c): the sprint venv has cupy-cuda13x exactly 14.0.1.
    Reads the .dist-info from disk so the assertion isn't proxied through the
    worker's reporting.
    """

    py = _require_venv_python()
    venv = py.parent.parent
    site_packages = list(venv.glob("lib/python*/site-packages"))
    assert site_packages, f"no site-packages under {venv}"
    dist_info = list(site_packages[0].glob("cupy_cuda13x-*.dist-info"))
    assert dist_info, "cupy-cuda13x not installed in sprint venv"
    # Dist-info dir name is ``<project>-<version>.dist-info``; strip suffix first.
    versions = sorted(p.name.removesuffix(".dist-info").split("-", 1)[1] for p in dist_info)
    assert versions == ["14.0.1"], (
        f"sprint venv has cupy-cuda13x versions {versions}; contract pins ==14.0.1"
    )
    # And there should be no other cupy distribution side by side.
    other_cupy = [
        p.name for p in site_packages[0].glob("cupy*.dist-info")
        if "cupy_cuda13x" not in p.name and "cupy_backends" not in p.name
    ]
    assert not other_cupy, f"unexpected cupy distributions in sprint venv: {other_cupy}"


# --------------------------------------------------------------------------- #
# 3. GPU-execution tests (require sprint venv)                                 #
# --------------------------------------------------------------------------- #


def _venv_run(script: str) -> tuple[int, str, str]:
    py = _require_venv_python()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [str(py), "-c", script],
        env=env,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_kernels_compile_and_report_matching_attributes() -> None:
    """Cross-AI verification (b): independently re-compile both RawKernels from
    the sprint venv and read ``Function.attributes``.  The kernel attributes
    seen here must exactly equal what the worker's profile JSON reports — that
    proves the JSON wasn't hand-edited.
    """

    _require_artifacts()
    script = r"""
import json, sys
sys.path.insert(0, 'src')
from gpuwrf.backends.cupy.stencil import _compile_kernel as cs
from gpuwrf.backends.cupy.column import _compile_kernel as cc
out = {}
for label, fn in (('stencil', cs), ('column', cc)):
    k = fn()
    attrs = dict(k.attributes)
    out[label] = {
        'num_regs': int(attrs.get('num_regs', -1)),
        'local_size_bytes': int(attrs.get('local_size_bytes', -1)),
        'max_threads_per_block': int(attrs.get('max_threads_per_block', -1)),
    }
print(json.dumps(out))
"""
    rc, stdout, stderr = _venv_run(script)
    assert rc == 0, f"venv kernel-attribute probe failed: {stderr}"
    observed = json.loads(stdout.strip().splitlines()[-1])

    stencil = _load_profile("stencil_profile.json")
    column = _load_profile("column_profile.json")
    assert observed["stencil"]["num_regs"] == stencil["registers_per_thread"], (
        f"stencil regs from venv {observed['stencil']['num_regs']} != "
        f"profile JSON {stencil['registers_per_thread']}"
    )
    assert observed["stencil"]["local_size_bytes"] == stencil["local_memory_bytes"]
    assert observed["column"]["num_regs"] == column["registers_per_thread"]
    assert observed["column"]["local_size_bytes"] == column["local_memory_bytes"]
    # Independent confirmation of contract AC #13.
    assert observed["column"]["local_size_bytes"] == 0


def test_kernel_runs_are_reproducible() -> None:
    """Same input twice → bitwise-identical NPZ output.  The kernels are
    deterministic; a regression that introduces stream-of-launches non-determinism
    (e.g. atomicAdd) would show up here.
    """

    _require_artifacts()
    if not STENCIL_FIXTURE.exists() or not COLUMN_FIXTURE.exists():
        pytest.skip("fixture samples not present")

    with tempfile.TemporaryDirectory(dir=str(ROOT / "data" / "scratch")) as tmp:
        tmp = Path(tmp)
        script = rf"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from gpuwrf.backends.cupy.stencil import run_stencil
from gpuwrf.backends.cupy.column import run_column
run_stencil(Path({str(STENCIL_FIXTURE)!r}), Path({str(tmp / 's1.npz')!r}))
run_stencil(Path({str(STENCIL_FIXTURE)!r}), Path({str(tmp / 's2.npz')!r}))
run_column(Path({str(COLUMN_FIXTURE)!r}), Path({str(tmp / 'c1.npz')!r}))
run_column(Path({str(COLUMN_FIXTURE)!r}), Path({str(tmp / 'c2.npz')!r}))
"""
        rc, _, stderr = _venv_run(script)
        assert rc == 0, f"venv reproducibility probe failed: {stderr}"
        s1 = np.load(tmp / "s1.npz")["phi_next"]
        s2 = np.load(tmp / "s2.npz")["phi_next"]
        c1 = np.load(tmp / "c1.npz")["temperature_next"]
        c2 = np.load(tmp / "c2.npz")["temperature_next"]
        assert np.array_equal(s1, s2), "stencil output is non-deterministic between runs"
        assert np.array_equal(c1, c2), "column output is non-deterministic between runs"


def test_run_stencil_rejects_missing_input_key() -> None:
    """Malformed-input edge case: NPZ missing a required key should raise
    KeyError, not silently produce a bogus profile.
    """

    _require_artifacts()
    if not STENCIL_FIXTURE.exists():
        pytest.skip("stencil fixture missing")
    with tempfile.TemporaryDirectory(dir=str(ROOT / "data" / "scratch")) as tmp:
        tmp = Path(tmp)
        bad = tmp / "bad_stencil.npz"
        arrays = dict(np.load(STENCIL_FIXTURE))
        del arrays["phi_initial"]
        np.savez(bad, **arrays)
        script = rf"""
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from gpuwrf.backends.cupy.stencil import run_stencil
try:
    run_stencil(Path({str(bad)!r}), Path({str(tmp / 'out.npz')!r}))
    print('UNEXPECTED_OK')
except KeyError:
    print('KEYERROR')
except Exception as e:
    print('OTHER', type(e).__name__, e)
"""
        rc, stdout, _ = _venv_run(script)
        assert rc == 0
        assert "KEYERROR" in stdout, f"expected KeyError, got {stdout!r}"


def test_run_column_rejects_missing_input_key() -> None:
    _require_artifacts()
    if not COLUMN_FIXTURE.exists():
        pytest.skip("column fixture missing")
    with tempfile.TemporaryDirectory(dir=str(ROOT / "data" / "scratch")) as tmp:
        tmp = Path(tmp)
        bad = tmp / "bad_column.npz"
        arrays = dict(np.load(COLUMN_FIXTURE))
        del arrays["saturation_qv"]
        np.savez(bad, **arrays)
        script = rf"""
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from gpuwrf.backends.cupy.column import run_column
try:
    run_column(Path({str(bad)!r}), Path({str(tmp / 'out.npz')!r}))
    print('UNEXPECTED_OK')
except KeyError:
    print('KEYERROR')
except Exception as e:
    print('OTHER', type(e).__name__, e)
"""
        rc, stdout, _ = _venv_run(script)
        assert rc == 0
        assert "KEYERROR" in stdout, f"expected KeyError, got {stdout!r}"


def test_run_stencil_raises_filenotfound_for_missing_fixture() -> None:
    _require_artifacts()
    with tempfile.TemporaryDirectory(dir=str(ROOT / "data" / "scratch")) as tmp:
        tmp = Path(tmp)
        script = rf"""
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from gpuwrf.backends.cupy.stencil import run_stencil
try:
    run_stencil(Path('/tmp/does_not_exist_tester_probe_xyz.npz'),
                Path({str(tmp / 'out.npz')!r}))
    print('UNEXPECTED_OK')
except FileNotFoundError:
    print('FNF')
except Exception as e:
    print('OTHER', type(e).__name__, e)
"""
        rc, stdout, _ = _venv_run(script)
        assert rc == 0
        assert "FNF" in stdout, f"expected FileNotFoundError, got {stdout!r}"


def test_bench_cli_problem_flag_is_respected() -> None:
    """Run the bench CLI with --problem column --skip-artifacts in a fresh
    scratch dir.  Only column outputs should appear; stencil_run.json must not
    be created (would mean --problem is ignored).
    """

    _require_artifacts()
    with tempfile.TemporaryDirectory(dir=str(ROOT / "data" / "scratch")) as tmp:
        tmp = Path(tmp)
        scratch = tmp / "scratch"
        prof = tmp / "prof"
        art = tmp / "art"
        py = _require_venv_python()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [
                str(py),
                "-m",
                "gpuwrf.backends.cupy.bench",
                "--problem",
                "column",
                "--scratch",
                str(scratch),
                "--artifact-dir",
                str(art),
                "--profiler-dir",
                str(prof),
                "--column-fixture",
                str(COLUMN_FIXTURE),
                "--skip-artifacts",
            ],
            env=env,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 0, f"bench CLI failed: {proc.stderr}"
        assert (scratch / "column_run.json").exists(), "column_run.json not produced"
        assert not (scratch / "stencil_run.json").exists(), (
            "stencil_run.json produced despite --problem column"
        )
        # --skip-artifacts must really skip the artifact writes.
        assert not (art / "column_profile.json").exists(), (
            "--skip-artifacts did not suppress column_profile.json"
        )


# --------------------------------------------------------------------------- #
# 4. Run script hygiene                                                        #
# --------------------------------------------------------------------------- #


def test_run_script_pins_cupy_cuda13x_exactly() -> None:
    """Static read of the install line: must pin ``cupy-cuda13x==14.0.1``."""

    text = (ROOT / "scripts" / "m2_run_cupy.sh").read_text()
    assert "cupy-cuda13x==14.0.1" in text, (
        "scripts/m2_run_cupy.sh does not pin cupy-cuda13x==14.0.1 (contract AC #1)"
    )
    # And it must not pin any other cupy distribution variant alongside.
    assert "cupy-cuda12x" not in text
    assert "cupy-cuda11x" not in text
