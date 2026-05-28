"""Regression test for sprint 2026-05-25-m6b-ladder-hygiene-cleanup Stage 7.

Asserts the two committed Fortran patches (``solve_em.F.patch`` and
``module_small_step_em.F.patch``) apply cleanly against canonical WRF source
via ``patch -p1 --dry-run`` (RC=0).

Future sprints (M6B4/B5/B6 hook-ABI follow-up) extending the patches MUST keep
this test green. Re-introduction of malformed hunks (bare ``@@`` markers,
wrong hunk counts, cross-file pollution) will fail the test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from gpuwrf.paths import reference_path


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "external" / "wrf_savepoint_patch"
SOLVE_EM_PATCH = PATCH_ROOT / "solve_em.F.patch"
MODULE_SMALL_STEP_PATCH = PATCH_ROOT / "module_small_step_em.F.patch"
CANONICAL_WRF = Path(os.environ.get("WRF_GPU_WRF_SOURCE", str(reference_path("artifacts", "wrf_gpu_src", "WRF"))))
SOLVE_EM_SRC = CANONICAL_WRF / "dyn_em" / "solve_em.F"
MODULE_SMALL_STEP_SRC = CANONICAL_WRF / "dyn_em" / "module_small_step_em.F"


def _build_minimal_tree(tmp_path: Path) -> Path:
    """Make a temporary copy of just the two patched WRF files."""

    if not SOLVE_EM_SRC.exists() or not MODULE_SMALL_STEP_SRC.exists():
        pytest.skip(
            "Canonical WRF source not available at expected path "
            f"{CANONICAL_WRF}; skipping patch-apply regression"
        )
    target = tmp_path / "wrf_canonical"
    (target / "dyn_em").mkdir(parents=True)
    shutil.copy(SOLVE_EM_SRC, target / "dyn_em" / "solve_em.F")
    shutil.copy(MODULE_SMALL_STEP_SRC, target / "dyn_em" / "module_small_step_em.F")
    return target


def _patch_dry_run(workdir: Path, patch_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["patch", "-p1", "--dry-run"],
        stdin=patch_file.open("rb"),
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_solve_em_patch_dry_run_returns_zero(tmp_path: Path) -> None:
    """`solve_em.F.patch` must apply cleanly against canonical WRF source."""

    assert SOLVE_EM_PATCH.exists(), f"missing patch file: {SOLVE_EM_PATCH}"
    workdir = _build_minimal_tree(tmp_path)
    result = _patch_dry_run(workdir, SOLVE_EM_PATCH)
    assert result.returncode == 0, (
        f"patch RC={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "FAILED" not in result.stdout, (
        f"patch reported FAILED hunks: stdout={result.stdout!r}"
    )


def test_module_small_step_em_patch_dry_run_returns_zero(tmp_path: Path) -> None:
    """`module_small_step_em.F.patch` must apply cleanly against canonical WRF."""

    assert MODULE_SMALL_STEP_PATCH.exists(), (
        f"missing patch file: {MODULE_SMALL_STEP_PATCH}"
    )
    workdir = _build_minimal_tree(tmp_path)
    result = _patch_dry_run(workdir, MODULE_SMALL_STEP_PATCH)
    assert result.returncode == 0, (
        f"patch RC={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "FAILED" not in result.stdout, (
        f"patch reported FAILED hunks: stdout={result.stdout!r}"
    )


def test_both_patches_apply_in_sequence(tmp_path: Path) -> None:
    """The two patches must apply in sequence (no cross-file collision)."""

    workdir = _build_minimal_tree(tmp_path)
    r1 = subprocess.run(
        ["patch", "-p1"],
        stdin=SOLVE_EM_PATCH.open("rb"),
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r1.returncode == 0, f"solve_em.F.patch failed: {r1.stdout!r} {r1.stderr!r}"

    r2 = subprocess.run(
        ["patch", "-p1"],
        stdin=MODULE_SMALL_STEP_PATCH.open("rb"),
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0, (
        f"module_small_step_em.F.patch failed: {r2.stdout!r} {r2.stderr!r}"
    )

    # At least 6 hook CALL sites must end up in solve_em.F
    # (calc_coef_w pre/post x2 = 4; advance_mu_t pre/post = 2).
    solve_em_out = (workdir / "dyn_em" / "solve_em.F").read_text()
    assert solve_em_out.count("CALL sp_") >= 6
    # And at least 4 in module_small_step_em.F (tridiag fwd/back pre/post).
    module_out = (workdir / "dyn_em" / "module_small_step_em.F").read_text()
    assert module_out.count("CALL sp_") >= 4


def test_no_bare_at_markers_in_patches() -> None:
    """Future regression: no hunk headers without offsets."""

    for patch_file in (SOLVE_EM_PATCH, MODULE_SMALL_STEP_PATCH):
        for lineno, line in enumerate(patch_file.read_text().splitlines(), start=1):
            if line.startswith("@@"):
                # Reject `@@` markers with no offsets (e.g. the M6B3-era
                # bare `@@\n` that broke patch parsing).
                assert line.endswith("@@") or " @@" in line, (
                    f"{patch_file.name}:{lineno}: header has trailing content: {line!r}"
                )
                # Header must contain both -N,M and +N,M offsets.
                assert " -" in line and " +" in line, (
                    f"{patch_file.name}:{lineno}: missing offsets in hunk header: {line!r}"
                )
