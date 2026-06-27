"""v0.18 microphysics-family batch honesty gates.

The v0.18 MP-family sprint asked for one batched pass over the open WRF-v4
microphysics tail. This test keeps the result explicit: already-wired schemes
stay implemented, WSM7/WDM7 remain untouched in this sprint, and every
previously open MP scheme has either an exact oracle-backed reference endpoint
or a source-backed irrelevance/no-op proof.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from gpuwrf.contracts.physics_registry import ACCEPTED_MP_PHYSICS
from gpuwrf.config.paths import wrf_root
from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_namelist
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme


WRF_ROOT = wrf_root()
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "proofs/v018/mp_endpoint_manifest.json"
STATUS_PATH = REPO_ROOT / "proofs/v018/mp_family_status.json"

OPERATIONAL_MP = {0, 1, 2, 3, 4, 6, 8, 10, 13, 14, 16, 24, 26, 28, 97}

REQUESTED_FAIL_CLOSED: dict[int, tuple[str, ...]] = {
    5: ("Ferrier", "module_mp_fer_hires.F", "qt/CWM", "NOT YET IMPLEMENTED"),
    95: ("Ferrier", "module_mp_etanew.F", "qt/CWM", "NOT YET IMPLEMENTED"),
    96: ("MadWRF", "CASE (MADWRF_MP)", "no-op", "NOT YET IMPLEMENTED"),
    7: ("Goddard 4-ice", "module_mp_gsfcgce_4ice_nuwrf.F", "qh", "NOT YET IMPLEMENTED"),
    38: ("Thompson graupel-hail", "module_mp_thompson.F", "qvolg", "NOT YET IMPLEMENTED"),
    9: ("Milbrandt-Yau", "module_mp_milbrandt2mom.F", "qnh", "NOT YET IMPLEMENTED"),
    11: ("CAM 5.1", "module_mp_cammgmp_driver.F", "CAM-specific", "NOT YET IMPLEMENTED"),
    17: ("NSSL", "module_mp_nssl_2mom.F", "qvolg", "NOT YET IMPLEMENTED"),
    18: ("NSSL", "module_mp_nssl_2mom.F", "CCN", "NOT YET IMPLEMENTED"),
    19: ("NSSL", "module_mp_nssl_2mom.F", "qvolg", "NOT YET IMPLEMENTED"),
    21: ("NSSL", "module_mp_nssl_2mom.F", "qvolg", "NOT YET IMPLEMENTED"),
    22: ("NSSL", "module_mp_nssl_2mom.F", "qvolg", "NOT YET IMPLEMENTED"),
    27: ("UDM", "module_mp_udm.F", "qnn/qnc/qnr", "NOT YET IMPLEMENTED"),
    29: ("RCON", "module_mp_rcon.F", "cloudnc", "NOT YET IMPLEMENTED"),
    30: ("HUJI fast", "module_mp_fast_sbm.F", "bin-state", "NOT YET IMPLEMENTED"),
    32: ("HUJI full", "module_mp_full_sbm.F", "bin-state", "NOT YET IMPLEMENTED"),
    40: ("Morrison aerosol", "module_mp_morr_two_moment_aero.F", "CCN", "NOT YET IMPLEMENTED"),
    50: ("P3", "module_mp_p3.F", "qir/qib", "NOT YET IMPLEMENTED"),
    51: ("P3", "module_mp_p3.F", "qnc/qir/qib", "NOT YET IMPLEMENTED"),
    52: ("P3", "module_mp_p3.F", "qi2", "NOT YET IMPLEMENTED"),
    53: ("P3", "module_mp_p3.F", "qzi", "NOT YET IMPLEMENTED"),
    55: ("Jensen-ISHMAEL", "module_mp_jensen_ishmael.F", "qi2/qi3", "NOT YET IMPLEMENTED"),
    56: ("NTU", "module_mp_ntu.F", "i3m", "NOT YET IMPLEMENTED"),
}

REFERENCE_WITH_ORACLE = {5, 7, 9, 18, 27, 29, 38, 40, 50, 51, 52, 53, 56, 95}
PROVEN_IRRELEVANT = {11, 17, 19, 21, 22, 30, 32, 55, 96}
STILL_OPEN: set[int] = set()
STANDALONE_ORACLE = {5, 7, 38, 95}
FULL_WRF_ORACLE = {9, 18, 27, 29, 40, 50, 51, 52, 53, 56}

SOURCE_BY_CODE = {
    5: "phys/module_mp_fer_hires.F",
    95: "phys/module_mp_etanew.F",
    7: "phys/module_mp_gsfcgce_4ice_nuwrf.F",
    38: "phys/module_mp_thompson.F",
    9: "phys/module_mp_milbrandt2mom.F",
    11: "phys/module_mp_cammgmp_driver.F",
    17: "phys/module_mp_nssl_2mom.F",
    18: "phys/module_mp_nssl_2mom.F",
    19: "phys/module_mp_nssl_2mom.F",
    21: "phys/module_mp_nssl_2mom.F",
    22: "phys/module_mp_nssl_2mom.F",
    27: "phys/module_mp_udm.F",
    29: "phys/module_mp_rcon.F",
    30: "phys/module_mp_fast_sbm.F",
    32: "phys/module_mp_full_sbm.F",
    40: "phys/module_mp_morr_two_moment_aero.F",
    50: "phys/module_mp_p3.F",
    51: "phys/module_mp_p3.F",
    52: "phys/module_mp_p3.F",
    53: "phys/module_mp_p3.F",
    55: "phys/module_mp_jensen_ishmael.F",
    56: "phys/module_mp_ntu.F",
}

REGISTRY_PACKAGES = {
    5: "fer_mp_hires",
    95: "etampnew",
    96: "madwrf_mp",
    7: "nuwrf4icescheme",
    38: "thompsongh",
    9: "milbrandt2mom",
    11: "cammgmpscheme",
    18: "nssl_2mom",
    27: "udmscheme",
    29: "rcon_mp_scheme",
    40: "morr_tm_aero",
    50: "p3_1category",
    51: "p3_1category_nc",
    52: "p3_2category",
    53: "p3_1cat_3mom",
    55: "jensen_ishmael",
    56: "ntu",
}


def test_v018_mp_operational_set_is_preserved() -> None:
    """The batch does not clobber existing scan-wired microphysics schemes."""

    assert set(ACCEPTED_MP_PHYSICS) == OPERATIONAL_MP
    for mp in sorted(OPERATIONAL_MP):
        assert classify_scheme("mp_physics", mp).status is SupportStatus.IMPLEMENTED

    # Contract scope: WSM7/WDM7 were skipped in this sprint, and remain the
    # already-existing operational hail schemes from trunk.
    for mp in (24, 26):
        support = classify_scheme("mp_physics", mp)
        assert support.status is SupportStatus.IMPLEMENTED
        assert support.reason == "Operationally wired into the GPU scan."


@pytest.mark.parametrize("mp, expected_tokens", sorted(REQUESTED_FAIL_CLOSED.items()))
def test_v018_open_mp_family_fails_closed_with_named_reason(
    mp: int, expected_tokens: tuple[str, ...]
) -> None:
    support = classify_scheme("mp_physics", mp)
    assert support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
    assert mp not in ACCEPTED_MP_PHYSICS
    for token in expected_tokens:
        assert token in support.reason
    assert "mp_physics=0/1/2/3/4/6/8/10/13/14/16/24/26/28/97" in support.alternative


def test_v018_open_mp_family_is_rejected_at_namelist_layer() -> None:
    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"physics": {"mp_physics": sorted(REQUESTED_FAIL_CLOSED)}})

    rejected = {selection.value for selection in excinfo.value.selections if selection.key == "mp_physics"}
    assert rejected == set(REQUESTED_FAIL_CLOSED)
    message = str(excinfo.value)
    assert "Ferrier" in message
    assert "RCON" in message
    assert "Madwrf" in message


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def _status() -> dict[str, Any]:
    return json.loads(STATUS_PATH.read_text())


def _manifest_entries() -> dict[int, dict[str, Any]]:
    return {int(k): v for k, v in _manifest()["schemes"].items()}


def _repo_or_abs(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _walk_numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_walk_numbers(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_walk_numbers(item))
        return out
    return []


def test_v018_endpoint_manifest_covers_every_requested_mp_code() -> None:
    manifest = _manifest()
    entries = _manifest_entries()
    expected = OPERATIONAL_MP | set(REQUESTED_FAIL_CLOSED)
    assert set(entries) == expected
    assert REFERENCE_WITH_ORACLE | PROVEN_IRRELEVANT | STILL_OPEN == set(REQUESTED_FAIL_CLOSED)
    assert REFERENCE_WITH_ORACLE.isdisjoint(PROVEN_IRRELEVANT)
    assert REFERENCE_WITH_ORACLE.isdisjoint(STILL_OPEN)
    assert PROVEN_IRRELEVANT.isdisjoint(STILL_OPEN)
    assert manifest["full_ship_gate"] is True
    assert manifest["still_open"] == []
    assert all(entry["endpoint"] != "still_open_fail_closed" for entry in entries.values())


def test_v018_endpoint_manifest_matches_catalog_status() -> None:
    entries = _manifest_entries()
    for mp, entry in entries.items():
        support = classify_scheme("mp_physics", mp)
        assert support.status.value == entry["catalog_status"], mp

        if entry["endpoint"] == "operational":
            assert entry["bar_met"] is True
            assert mp in OPERATIONAL_MP
            assert support.status is SupportStatus.IMPLEMENTED
            continue

        assert mp not in ACCEPTED_MP_PHYSICS
        assert support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
        if entry["endpoint"] == "ref_with_oracle_fail_closed":
            assert entry["bar_met"] is True
            assert "REFERENCE-WITH-REAL-ORACLE" in support.reason
        elif entry["endpoint"] == "proven_irrelevant_fail_closed":
            assert entry["bar_met"] is True
            assert entry["irrelevance_basis"]
            assert "PROVEN" in support.reason
        else:
            raise AssertionError(f"unexpected endpoint for mp={mp}: {entry['endpoint']}")


def test_v018_closer_status_has_no_still_open_schemes() -> None:
    status = _status()
    assert status["full_ship_gate"] is True
    assert status["still_open"] == []
    assert {int(mp) for mp in status["schemes"]} == FULL_WRF_ORACLE | {5}
    for mp_text, entry in status["schemes"].items():
        mp = int(mp_text)
        assert entry["endpoint"] == "ref_with_oracle_fail_closed"
        assert entry["bar_met"] is True
        assert _repo_or_abs(entry["oracle_path"]).exists(), mp


def test_v018_exact_oracle_artifacts_are_module_specific_and_nontrivial() -> None:
    entries = _manifest_entries()
    assert {entries[mp]["endpoint"] for mp in REFERENCE_WITH_ORACLE} == {
        "ref_with_oracle_fail_closed"
    }

    expected_modules = {
        5: "module_mp_fer_hires.F",
        7: "module_mp_gsfcgce_4ice_nuwrf.F",
        38: "module_mp_thompson.F",
        95: "module_mp_etanew.F",
    }
    expected_case_counts = {5: 4, 7: 6, 38: 1, 95: 6}
    expected_globs = {
        5: "ferrier_mp5_case_*.json",
        7: "goddard4ice_case_*.json",
        38: "thompgh_case_*.json",
        95: "ferrier_case_*.json",
    }

    for mp in sorted(STANDALONE_ORACLE):
        entry = entries[mp]
        for path_text in entry["evidence_paths"] + entry["source_checksums"]:
            assert _repo_or_abs(path_text).exists(), f"mp={mp} missing {path_text}"

        checksum_text = "\n".join(_repo_or_abs(path).read_text() for path in entry["source_checksums"])
        assert expected_modules[mp] in checksum_text

        all_cases: list[Path] = []
        for directory_text in entry["savepoint_dirs"]:
            directory = _repo_or_abs(directory_text)
            assert directory.is_dir()
            cases = sorted(directory.glob(expected_globs[mp]))
            assert len(cases) == expected_case_counts[mp]
            all_cases.extend(cases)

        assert all_cases
        for case_path in all_cases:
            data = json.loads(case_path.read_text())
            numbers = _walk_numbers(data)
            assert numbers
            assert all(math.isfinite(number) for number in numbers)

    mp5_driver = (REPO_ROOT / "proofs/v018/mp_oracles/ferrier_hires/oracle/ferrier_oracle_driver.f90").read_text()
    assert "use module_mp_fer_hires" in mp5_driver
    assert "call FER_HIRES" in mp5_driver
    mp5_cases = sorted(
        (REPO_ROOT / "proofs/v018/mp_oracles/ferrier_hires/savepoints").glob(
            "ferrier_mp5_case_*.json"
        )
    )
    assert all(json.loads(path.read_text())["metadata"]["scheme"] == "mp5" for path in mp5_cases)
    assert any(json.loads(path.read_text())["scalars"]["RAINNCV"] > 0.0 for path in mp5_cases)

    # Exact-module rule: the MP95 oracle is real ETAMP_NEW evidence, and the MP5
    # endpoint has its own FER_HIRES artifact root.
    ferrier_checksums = "\n".join(
        _repo_or_abs(path).read_text() for path in entries[95]["source_checksums"]
    )
    assert "module_mp_etanew.F" in ferrier_checksums
    assert "module_mp_fer_hires.F" not in ferrier_checksums
    assert entries[5]["endpoint"] == "ref_with_oracle_fail_closed"
    assert entries[5]["exact_module"] == "phys/module_mp_fer_hires.F:FER_HIRES"
    assert entries[5]["artifact_root"] != entries[95]["artifact_root"]

    goddard_case = json.loads(
        (REPO_ROOT / "proofs/v018/mp_oracles/goddard4ice/savepoints_fp64/goddard4ice_case_4.json").read_text()
    )
    assert goddard_case["scalars"]["HAILNCV"] > 1.0e-6

    thompgh_case = json.loads(
        (REPO_ROOT / "proofs/v018/mp_oracles/thompgh/savepoints_fp32/thompgh_case_1.json").read_text()
    )
    assert thompgh_case["scalars"]["RAINNCV"] > 0.0


def test_v018_active_full_wrf_oracles_are_exact_and_nontrivial() -> None:
    entries = _manifest_entries()
    expected_modules = {
        9: "phys/module_mp_milbrandt2mom.F",
        18: "phys/module_mp_nssl_2mom.F",
        27: "phys/module_mp_udm.F",
        29: "phys/module_mp_rcon.F",
        40: "phys/module_mp_morr_two_moment_aero.F",
        50: "phys/module_mp_p3.F",
        51: "phys/module_mp_p3.F",
        52: "phys/module_mp_p3.F",
        53: "phys/module_mp_p3.F",
        56: "phys/module_mp_ntu.F",
    }
    for mp in sorted(FULL_WRF_ORACLE):
        entry = entries[mp]
        summary_paths = entry["oracle_summaries"]
        assert len(summary_paths) == 1
        summary_path = _repo_or_abs(summary_paths[0])
        assert summary_path.exists(), mp
        data = json.loads(summary_path.read_text())

        assert data["oracle_type"] == "pristine_wrf_full_model_active_driver"
        assert data["endpoint"] == "ref_with_oracle_fail_closed"
        assert data["bar_met"] is True
        assert data["mp_physics"] == mp
        assert data["exact_module"] == expected_modules[mp]
        assert data["wrf_success"] is True
        assert data["history_times"] == ["0001-01-01_00:00:00", "0001-01-01_00:01:00"]
        assert data["nontrivial_seeded_microphysics_fields"]

        checksum_paths = " ".join(item["path"] for item in data["source_checksums"])
        assert expected_modules[mp] in checksum_paths
        assert "phys/module_microphysics_driver.F" in checksum_paths

        variable_numbers = _walk_numbers(data["variables"])
        assert variable_numbers
        assert all(math.isfinite(number) for number in variable_numbers)
        assert any(
            variable["delta"]["max_abs"] > 0.0
            for name, variable in data["variables"].items()
            if name not in {"T", "QVAPOR"}
        )


def test_v018_proven_irrelevant_entries_have_named_local_evidence() -> None:
    entries = _manifest_entries()
    for mp in sorted(PROVEN_IRRELEVANT):
        entry = entries[mp]
        assert entry["endpoint"] == "proven_irrelevant_fail_closed"
        assert entry["irrelevance_basis"]
        if WRF_ROOT.exists():
            for path_text in entry["evidence_paths"]:
                assert _repo_or_abs(path_text).exists(), f"mp={mp} missing {path_text}"

    assert "mp_physics=18" in entries[17]["irrelevance_basis"]
    assert entries[18]["exact_module"] == "phys/module_mp_nssl_2mom.F"
    for mp in (17, 19, 21, 22):
        assert entries[mp]["exact_module"].startswith("MP18 alias via phys/module_mp_nssl_2mom.F")
    assert "BUILD_SBM_FAST" in entries[30]["irrelevance_basis"]
    assert "wrf_debug" in entries[96]["irrelevance_basis"]


def test_v018_no_still_open_entries_or_messages_remain() -> None:
    entries = _manifest_entries()
    assert STILL_OPEN == set()
    for mp, entry in entries.items():
        assert entry["endpoint"] != "still_open_fail_closed", mp
    for mp in sorted(REQUESTED_FAIL_CLOSED):
        assert "STILL OPEN" not in classify_scheme("mp_physics", mp).reason


@pytest.mark.skipif(not WRF_ROOT.exists(), reason="pristine WRF source tree is not available")
def test_v018_fail_closed_mp_reasons_are_source_backed() -> None:
    for mp, relative_path in SOURCE_BY_CODE.items():
        path = WRF_ROOT / relative_path
        assert path.exists(), f"mp={mp} missing WRF source {path}"
        assert path.name in classify_scheme("mp_physics", mp).reason

    registry = (WRF_ROOT / "Registry/Registry.EM_COMMON").read_text()
    for mp, package in REGISTRY_PACKAGES.items():
        assert package in registry
        assert f"mp_physics=={mp}" in registry


@pytest.mark.skipif(not WRF_ROOT.exists(), reason="pristine WRF source tree is not available")
def test_madwrf_mp96_driver_case_is_noop() -> None:
    driver = (WRF_ROOT / "phys/module_microphysics_driver.F").read_text()
    case_start = driver.index("CASE (MADWRF_MP)")
    case_end = driver.index("CASE DEFAULT", case_start)
    body = driver[case_start:case_end]
    body_lines = [
        line.split("!")[0].strip()
        for line in body.splitlines()[1:]
        if line.split("!")[0].strip()
    ]
    normalized = re.sub(r"\s+", " ", " ".join(body_lines))
    assert normalized == "CALL wrf_debug ( 100 , 'microphysics_driver: case MADWRF_MP')"
