"""v0.18 RADIATION (RA) tail ship-gate: reference-only real-WRF oracles + class-(c).

Scope (the radiation long-tail beyond the operational RRTMG SW/LW + Dudhia/GSFC SW
+ classic-RRTM LW + Held-Suarez 31):

  * ra_lw/sw_physics = 3 (CAM), 5 (new Goddard NUWRF), 7 (FLG/UCLA), 99 (GFDL-Eta)
    are REFERENCE_ONLY. Each is a large (8k-15k LOC) WRF radiation module that is
    not faithfully portable to a traceable JAX kernel in-scope without becoming a
    self-compare/happy-path, so it is namelist-accepted (for a reference
    comparison) and FAIL-CLOSES in the operational GPU scan -- backed by a REAL
    physics-pristine WRF *exact-driver* oracle: a WRFGPU2_ORACLE-instrumented
    wrf.exe run with ra_lw_physics=ra_sw_physics=N, whose radiation_driver
    dispatches the exact upstream-identical source module (CAMRAD / goddardrad /
    RAD_FLG / ETARA) and writes the radiation tendency + flux history this savepoint records
    (proofs/v018/savepoints/ra_tail_wrf/raN_wrf_real.json). NOT a JAX self-compare.

  * ra_lw/sw_physics = 14 (RRTMG-K / KIAPS) and 24 (fast RRTMG, GPU/MIC) are
    class-(c) computationally-UNAVAILABLE: they are compiled OUT of standard WRF
    itself. Their source modules (phys/module_ra_rrtmg_{lwk,swk,lwf,swf}.F) are
    bare `#if( BUILD_RRTMK != 1)` / `#if( BUILD_RRTMG_FAST != 1)` dummy stubs, the
    pristine configure.wrf sets -DBUILD_RRTMK=0 / -DBUILD_RRTMG_FAST=0, and the
    radiation_driver CASEs are BUILD-gated -- so selecting 14/24 in unmodified WRF
    hits the driver's default abort. There is no real oracle to build; they fail
    closed at the namelist layer with a source-cited reason.

This module proves the ship-gate, with NO silent gaps:
  1. each reference-only oracle savepoint EXISTS, is finite, non-trivial
     (longwave AND shortwave both actually fired), and attributes its exact module;
  2. the source checksum + raw manifest files record upstream-identical radiation
     modules and the numerically-inert dump-only instrumentation;
  3. every oracle path the operational scan CITES in its fail-close reason exists
     on disk (no dangling reference);
  4. 3/5/7/99 classify REFERENCE_ONLY, pass the namelist layer, and fail closed in
     the operational scan (NotOperationallyWiredError);
  5. 14/24 classify RECOGNIZED_FAIL_CLOSED, carry the compiled-out source citation,
     and fail closed at the namelist layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gpuwrf.io.namelist_check import (
    NotOperationallyWiredError,
    UnsupportedSchemeError,
    validate_namelist,
    validate_operational_namelist,
)
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAVE_DIR = _REPO_ROOT / "proofs" / "v018" / "savepoints" / "ra_tail_wrf"
_RA_KEYS = ("ra_lw_physics", "ra_sw_physics")
_REF_ONLY_CODES = (3, 5, 7, 99)
_COMPILED_OUT_CODES = (14, 24)
_CORE_FIELDS = ("RTHRATLW", "RTHRATSW", "GLW", "OLR", "SWDOWN")
_EXACT_MODULE = {
    3: "module_ra_cam.F",
    5: "module_ra_goddard.F",
    7: "module_ra_flg.F",
    99: "module_ra_gfdleta.F",
}


def _iter_numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_iter_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_iter_numbers(v))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out


def _isfinite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _load_savepoint(code: int) -> dict:
    path = _SAVE_DIR / f"ra{code}_wrf_real.json"
    assert path.is_file(), (
        f"missing v0.18 RA tail real-WRF oracle savepoint {path} -- the operational "
        f"scan cites it as the fail-close evidence for ra_lw/sw_physics={code}; "
        f"regenerate via proofs/v018/oracle/ra_tail_wrf/run_parallel.sh"
    )
    return json.loads(path.read_text())


@pytest.mark.parametrize("code", _REF_ONLY_CODES)
def test_ra_tail_oracle_is_real_and_nontrivial(code: int) -> None:
    """The exact-driver real-WRF oracle exists, is finite, and exercised LW AND SW."""

    data = _load_savepoint(code)
    assert data["schema"] == "wrf-v018-ra-tail-real-wrf-column-savepoint-v1"
    assert data["scheme"] == code
    assert data["ra_lw_physics"] == code and data["ra_sw_physics"] == code

    # Exact-module provenance: the savepoint must attribute the precise WRF module
    # the radiation_driver dispatches for this code (exact-module-oracle rule).
    modules = " ".join(data.get("source_modules", []))
    assert _EXACT_MODULE[code] in modules, (
        f"ra{code} oracle does not attribute its exact module {_EXACT_MODULE[code]}"
    )
    assert "physics-pristine" in data.get("exact_module_rule", "")
    assert "upstream-identical radiation module" in data.get("exact_module_rule", "")

    # Non-trivial: both the longwave and shortwave radiation paths actually fired
    # (not an all-night SW-zero degenerate run). This is the load-bearing "the
    # scheme really computed radiation" evidence.
    assert data["lw_nonzero"] is True, f"ra{code}: longwave never fired in the oracle"
    assert data["sw_nonzero"] is True, f"ra{code}: shortwave never fired in the oracle"
    assert data["nontrivial"] is True

    # Every recorded number is finite (no NaN/Inf leaking into the oracle).
    flat = _iter_numbers(data)
    assert flat, f"ra{code} oracle carries no numeric payload"
    assert all(_isfinite(v) for v in flat), f"non-finite value in ra{code} oracle"

    # The core radiation tendency + flux fields are present with finite maxima.
    for field in _CORE_FIELDS:
        assert field in data["max_abs"], f"ra{code}: core field {field} absent"
        assert _isfinite(data["max_abs"][field]), f"ra{code}: {field} max non-finite"


def test_ra_tail_source_checksums_record_physics_pristine_wrf() -> None:
    """The oracle records upstream-identical modules plus dump-only instrumentation."""

    cks = _SAVE_DIR / "wrf_source_checksums.txt"
    raw = _SAVE_DIR / "raw_hash_manifest.txt"
    assert cks.is_file(), f"missing WRF source checksum file {cks}"
    assert raw.is_file(), f"missing raw provenance/hash manifest {raw}"
    text = cks.read_text()
    for module in (
        "module_radiation_driver.F",
        "module_ra_cam.F",
        "module_ra_goddard.F",
        "module_ra_flg.F",
        "module_ra_gfdleta.F",
    ):
        assert module in text, f"{module} not recorded in {cks.name}"
    # Each payload line carries a 64-hex sha256 (or an explicit MISSING marker).
    payload = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert payload, "checksum file has no payload lines"
    for line in payload:
        token = line.split()[0]
        assert token == "MISSING" or re.fullmatch(r"[0-9a-f]{64}", token), (
            f"checksum line is neither a sha256 nor MISSING: {line!r}"
        )

    manifest = raw.read_text()
    for expected in (
        "numerically-inert dump-only",
        "phys/module_wrfgpu2_oracle.F",
        "phys/module_ra_cam.F",
        "phys/module_ra_goddard.F",
        "phys/module_ra_flg.F",
        "phys/module_ra_gfdleta.F",
        "main/wrf.exe",
        "ra3/wrfout_d01_2026-04-28_18:00:00",
        "ra5/wrfout_d01_2026-04-28_18:00:00",
        "ra7/wrfout_d01_2026-04-28_18:00:00",
        "ra99/wrfout_d01_2026-04-28_18:00:00",
    ):
        assert expected in manifest, f"{expected} not documented in raw manifest"


def test_operational_scan_reasons_cite_existing_oracle_paths() -> None:
    """Every proofs/* oracle path the RA operational fail-close reasons cite exists.

    This closes the dangling-reference gap: the operational scan must never name an
    evidence file that is not actually on disk.
    """

    from gpuwrf.runtime import operational_mode

    reasons = operational_mode._SCAN_UNWIRED_REASON
    cited: set[str] = set()
    for tag, reason in reasons.items():
        if not tag.startswith(("ra_lw_physics=", "ra_sw_physics=")):
            continue
        for match in re.findall(r"proofs/\S+?\.json", reason):
            cited.add(match.rstrip(").,"))
    assert cited, "operational scan cites no RA tail oracle paths"
    for rel in sorted(cited):
        assert (_REPO_ROOT / rel).is_file(), (
            f"operational scan fail-close reason cites a missing oracle: {rel}"
        )


@pytest.mark.parametrize("code", _REF_ONLY_CODES)
@pytest.mark.parametrize("key", _RA_KEYS)
def test_ra_tail_reference_only_and_fails_closed_operationally(key: str, code: int) -> None:
    """3/5/7/99 are REFERENCE_ONLY: namelist-accepted, operationally fail-closed."""

    assert classify_scheme(key, code).status is SupportStatus.REFERENCE_ONLY
    # Accepted at the namelist layer (selectable for a reference comparison).
    validate_namelist({"physics": {key: [code]}})
    # But the operational forecast scan refuses it (never a silent wrong scheme).
    with pytest.raises(NotOperationallyWiredError):
        validate_operational_namelist({"physics": {key: [code]}})


@pytest.mark.parametrize("code", _COMPILED_OUT_CODES)
@pytest.mark.parametrize("key", _RA_KEYS)
def test_ra_compiled_out_schemes_fail_closed_with_source_citation(key: str, code: int) -> None:
    """14/24 are class-(c) compiled-out of standard WRF; fail closed with a citation."""

    support = classify_scheme(key, code)
    assert support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
    # The catalog honesty API carries the precise compiled-out reason + BUILD flag.
    assert "compiled OUT" in support.reason
    flag = "BUILD_RRTMK" if code == 14 else "BUILD_RRTMG_FAST"
    assert flag in support.reason
    # Fail closed at the namelist layer; the user-facing message flags compiled-out
    # rather than implying a fillable port gap.
    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"physics": {key: [code]}})
    assert "compiled-out" in str(excinfo.value)
