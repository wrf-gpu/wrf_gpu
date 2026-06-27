"""v0.22 F2 missing-scheme bundle gate.

The four requested schemes are large. This test locks the honest one-pass
outcome: New-Tiedtke and RUC have local oracle evidence and remain
reference-only; NSSL mp=18 and Morrison aerosol mp=40 have no local
single-column oracle artifacts in this worktree and therefore stay fail-closed.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from gpuwrf.io.namelist_check import (  # noqa: E402
    NotOperationallyWiredError,
    UnsupportedSchemeError,
    validate_namelist,
    validate_operational_namelist,
)
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROOF_SCRIPT = REPO_ROOT / "proofs/v022/f2_missing_scheme_bundle_oracle_check.py"


def _proof_module():
    spec = importlib.util.spec_from_file_location("f2_missing_scheme_bundle_oracle_check", PROOF_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f2_bundle_oracle_gate_report_is_honest() -> None:
    report = _proof_module().build_report()
    assert report["gate_pass"] is True
    assert report["full_bundle_landed"] is False

    schemes = report["schemes"]
    assert schemes["new_tiedtke"]["coverage"] == "reference_only_oracle_present"
    assert schemes["new_tiedtke"]["oracle"]["file_count"] == 5
    assert schemes["new_tiedtke"]["oracle"]["nontrivial"] is True

    assert schemes["ruc_lsm"]["coverage"] == "reference_only_oracle_present"
    assert schemes["ruc_lsm"]["oracle"]["all_green"] is True
    assert schemes["ruc_lsm"]["oracle"]["raw_savepoint_present"] is False

    for scheme_id in ("nssl_2mom", "morrison_aero"):
        entry = schemes[scheme_id]
        assert entry["coverage"] == "fail_closed_oracle_absent"
        assert entry["catalog_status"] == "recognized_fail_closed"
        assert entry["accepted_by_reference_validator"] is False
        assert entry["scan_wired"] is False
        assert "ORACLE-ABSENT" in entry["catalog_reason"]


def test_f2_catalog_and_scan_path_statuses() -> None:
    from gpuwrf.runtime.operational_mode import _SCAN_UNWIRED_REASON, _SCAN_WIRED_OPTIONS

    assert classify_scheme("cu_physics", 16).status is SupportStatus.REFERENCE_ONLY
    assert classify_scheme("sf_surface_physics", 3).status is SupportStatus.REFERENCE_ONLY
    assert classify_scheme("mp_physics", 18).status is SupportStatus.RECOGNIZED_FAIL_CLOSED
    assert classify_scheme("mp_physics", 40).status is SupportStatus.RECOGNIZED_FAIL_CLOSED

    for key, code in (
        ("cu_physics", 16),
        ("sf_surface_physics", 3),
        ("mp_physics", 18),
        ("mp_physics", 40),
    ):
        assert code not in _SCAN_WIRED_OPTIONS.get(key, ())
        assert _SCAN_UNWIRED_REASON[f"{key}={code}"]


def test_f2_namelist_gate_accepts_only_reference_or_operational_paths() -> None:
    validate_namelist({"physics": {"cu_physics": [16]}})
    validate_namelist({"physics": {"sf_surface_physics": [3]}})

    with pytest.raises(NotOperationallyWiredError):
        validate_operational_namelist({"physics": {"cu_physics": [16]}})
    with pytest.raises(NotOperationallyWiredError):
        validate_operational_namelist({"physics": {"sf_surface_physics": [3]}})

    for mp in (18, 40):
        with pytest.raises(UnsupportedSchemeError):
            validate_namelist({"physics": {"mp_physics": [mp]}})
        with pytest.raises(UnsupportedSchemeError):
            validate_operational_namelist({"physics": {"mp_physics": [mp]}})
