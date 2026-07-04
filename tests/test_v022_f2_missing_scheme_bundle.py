"""v0.22 F2 missing-scheme bundle gate.

The four requested schemes are large. This test locks the honest v0.23 F2
state: New-Tiedtke, RUC, NSSL mp=18 and Morrison-aerosol mp=40 ALL have real
local single-column WRF oracle evidence now (the mp18/mp40 oracles were built
in v0.23 F2 at proofs/v022/f2_oracles/) and are reference-only: namelist-
accepted for single-column comparison, fail-closed in the operational scan.
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
    # v0.23 F2: cu16 graduated -- faithful fp64 kernel (machine precision vs
    # the 5 oracle savepoints) + ntiedtke_adapter scan wiring.
    assert schemes["new_tiedtke"]["coverage"] == "implemented_scan_wired"
    assert schemes["new_tiedtke"]["catalog_status"] == "implemented"
    assert schemes["new_tiedtke"]["scan_wired"] is True
    assert schemes["new_tiedtke"]["oracle"]["file_count"] == 5
    assert schemes["new_tiedtke"]["oracle"]["nontrivial"] is True

    assert schemes["ruc_lsm"]["coverage"] == "reference_only_oracle_present"
    assert schemes["ruc_lsm"]["oracle"]["all_green"] is True
    # v0.23 F2: the raw v017 fp64 oracle savepoint was restored from history
    # (a922b3b5) alongside the integrated v018 RUC port, so it is present now.
    assert schemes["ruc_lsm"]["oracle"]["raw_savepoint_present"] is True

    # v0.23 F2: real single-column oracles landed for both MP schemes
    # (fp32 + fp64 x 6 regimes each), flipping them to reference-only.
    for scheme_id in ("nssl_2mom", "morrison_aero"):
        entry = schemes[scheme_id]
        assert entry["coverage"] == "reference_only_oracle_present"
        assert entry["catalog_status"] == "reference_only"
        assert entry["accepted_by_reference_validator"] is True
        assert entry["scan_wired"] is False
        assert entry["oracle"]["file_count"] == 12
        assert entry["oracle"]["all_finite"] is True


def test_f2_catalog_and_scan_path_statuses() -> None:
    from gpuwrf.runtime.operational_mode import _SCAN_UNWIRED_REASON, _SCAN_WIRED_OPTIONS

    assert classify_scheme("cu_physics", 16).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("sf_surface_physics", 3).status is SupportStatus.REFERENCE_ONLY
    assert classify_scheme("mp_physics", 18).status is SupportStatus.REFERENCE_ONLY
    assert classify_scheme("mp_physics", 40).status is SupportStatus.REFERENCE_ONLY

    assert 16 in _SCAN_WIRED_OPTIONS["cu_physics"]
    assert "cu_physics=16" not in _SCAN_UNWIRED_REASON
    for key, code in (
        ("sf_surface_physics", 3),
        ("mp_physics", 18),
        ("mp_physics", 40),
    ):
        assert code not in _SCAN_WIRED_OPTIONS.get(key, ())
        assert _SCAN_UNWIRED_REASON[f"{key}={code}"]


def test_f2_namelist_gate_accepts_only_reference_or_operational_paths() -> None:
    validate_namelist({"physics": {"cu_physics": [16]}})
    validate_namelist({"physics": {"sf_surface_physics": [3]}})

    # cu16 is operationally wired in v0.23 F2 (like cu=6 it additionally
    # requires active flux-form moisture advection at runtime).
    validate_operational_namelist({"physics": {"cu_physics": [16]}})
    with pytest.raises(NotOperationallyWiredError):
        validate_operational_namelist({"physics": {"sf_surface_physics": [3]}})

    for mp in (18, 40):
        validate_namelist({"physics": {"mp_physics": [mp]}})
        with pytest.raises(NotOperationallyWiredError):
            validate_operational_namelist({"physics": {"mp_physics": [mp]}})
