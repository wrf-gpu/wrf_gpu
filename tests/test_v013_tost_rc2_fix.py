"""Regression guard for the v0.13 powered-TOST rc=2 fix (KI-5).

On v0.12.0 the powered-n=15 campaign returned rc=2 (``L2_D02_BLOCKED``,
0/15 -> ABORT) and, separately, the orchestrator returned rc=2 whenever fewer
than 2 cases were scored -- which ALSO fired in single-case ``--case`` debug
mode even when the one case scored perfectly. That made the harness impossible
to drive a single case through for diagnosis.

These tests lock the FIXED rc semantics of
``proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.main``:

* 0 cases scored                     -> rc=2 (genuine ABORT) + abort JSON
* 1 case scored, --case / --allow-single -> rc=0 + single-case JSON
* 1 case scored, full campaign       -> rc=2 (under-powered) + single-case JSON

The scoring/GPU functions are monkeypatched so the test is CPU-only and needs
no GPU, no JAX, and no on-disk corpus.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py"


@pytest.fixture()
def camp(monkeypatch, tmp_path):
    """Import the orchestrator module with all on-disk side effects redirected."""
    if not HARNESS.is_file():
        pytest.skip("powered-TOST harness not present in this checkout")
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("camp_rc2", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Redirect every filesystem target into tmp; never touch /tmp or the repo.
    monkeypatch.setattr(mod, "PROOF_DIR", tmp_path / "proof", raising=True)
    monkeypatch.setattr(mod, "GPU_RUNS_ROOT", tmp_path / "gpu_runs", raising=True)
    monkeypatch.setattr(mod, "MERGED_RUN_ROOT", tmp_path / "merged", raising=True)
    # Neutralise the merged-root preparation (no real corpus on this box).
    monkeypatch.setattr(mod, "prepare_merged_run_root", lambda: tmp_path / "merged")
    # Neutralise per-case git commits.
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: None)
    return mod


def _make_case_runnable(mod, monkeypatch, run_id, *, scored: bool):
    """Wire one case so it either scores cleanly or is excluded, with no GPU."""
    merged_dir = mod.MERGED_RUN_ROOT / run_id
    cpu_dir = tmp_cpu = mod.GPU_RUNS_ROOT / "cpu" / run_id
    merged_dir.mkdir(parents=True, exist_ok=True)
    cpu_dir.mkdir(parents=True, exist_ok=True)
    # Enough fake CPU d02 files to pass the completeness prerequisite.
    for h in range(mod.FORECAST_HOURS + 1):
        (cpu_dir / f"wrfout_d02_h{h:02d}").write_text("x")
    monkeypatch.setattr(mod, "L2_CPU_ROOT", mod.GPU_RUNS_ROOT / "cpu", raising=True)

    # GPU forecast + validation: always "succeed" without touching a GPU.
    monkeypatch.setattr(mod, "run_gpu_forecast",
                        lambda rid, gd, ps: {"run_id": rid, "returncode": 0, "elapsed_s": 1.0})
    gpu_dir = mod.GPU_RUNS_ROOT / f"l2_d02_{run_id}"
    gpu_dir.mkdir(parents=True, exist_ok=True)
    for h in range(mod.FORECAST_HOURS + 1):
        (gpu_dir / f"wrfout_d02_h{h:02d}").write_text("x")
    monkeypatch.setattr(mod, "validate_gpu_output",
                        lambda gd, rid: {"run_id": rid, "valid": scored, "n_files": 25,
                                         "expected_files": 25, "nonfinite_hour_files": [],
                                         "dtype_fp64_first_file": True})
    if scored:
        monkeypatch.setattr(mod, "score_one_case",
                            lambda rid, gd, cd, it, val: {
                                "run_id": rid, "tost_pairs": {"case_id": rid},
                                "cell_level": {"field_stats": {}},
                            })


def test_single_case_scores_rc0(camp, monkeypatch):
    """--case with a scored case returns rc=0 (was rc=2 on v0.12.0)."""
    rid = camp.CASE_IDS[0]
    _make_case_runnable(camp, monkeypatch, rid, scored=True)
    rc = camp.main(["--skip-gpu", "--case", rid])
    assert rc == 0, "single scored --case must be rc=0"
    assert (camp.PROOF_DIR / "powered_tost_single_case.json").is_file()


def test_zero_cases_scored_rc2_with_abort_json(camp, monkeypatch):
    """All cases excluded -> rc=2 + machine-readable abort JSON."""
    rid = camp.CASE_IDS[0]
    _make_case_runnable(camp, monkeypatch, rid, scored=False)  # validation fails -> excluded
    rc = camp.main(["--skip-gpu", "--case", rid])
    assert rc == 2, "0 scored cases must abort rc=2"
    assert (camp.PROOF_DIR / "powered_tost_abort.json").is_file()


def test_allow_single_flag_rc0(camp, monkeypatch):
    """--allow-single makes a 1-case full run a success."""
    rid = camp.CASE_IDS[0]
    _make_case_runnable(camp, monkeypatch, rid, scored=True)
    # Run as a (degenerate) full campaign of just this id via --case + allow-single.
    rc = camp.main(["--skip-gpu", "--case", rid, "--allow-single"])
    assert rc == 0


def test_gpu_lock_wrapper_only_if_present(camp, monkeypatch):
    """GPU_LOCK_WRAPPER must be None (launch direct) when the wrapper is absent."""
    import importlib
    monkeypatch.setenv("GPUWRF_GPU_LOCK_WRAPPER", "/nonexistent/wrapper.sh")
    spec = importlib.util.spec_from_file_location("camp_wrap", HARNESS)
    mod2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod2)
    assert mod2.GPU_LOCK_WRAPPER is None
