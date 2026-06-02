from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.init.static_geog import (
    STATIC_GEOG_GROUPS,
    load_static_geog,
    static_geog_field_names,
    static_geog_specs,
)


ROOT = Path(__file__).resolve().parents[2]
WPS_ROOT = Path(os.environ.get("GPUWRF_WPS_CASES_ROOT", "/mnt/data/canairy_meteo/runs/wps_cases"))
PROOF_PATH = ROOT / "proofs" / "v030" / "s2_static_geog_report.json"
DOMAINS = ("d01", "d02", "d03")
CATEGORY_FIELDS = ("LU_INDEX", "LANDMASK", "SCT_DOM", "SCB_DOM")


def _case_dirs() -> list[Path]:
    cases = [
        path
        for path in sorted(WPS_ROOT.glob("*"))
        if all((path / "l3" / f"geo_em.{domain}.nc").exists() for domain in DOMAINS)
        and all(list((path / "l3").glob(f"met_em.{domain}.*.nc")) for domain in DOMAINS)
    ]
    if len(cases) < 3:
        pytest.skip(f"need at least three WPS cases with geo_em/met_em d01-d03 under {WPS_ROOT}")
    return cases[:3]


def _first_met_em(case: Path, domain: str) -> Path:
    return sorted((case / "l3").glob(f"met_em.{domain}.*.nc"))[0]


def _max_abs(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(candidate, np.float64) - np.asarray(reference, np.float64))))


def _max_rel(candidate: np.ndarray, reference: np.ndarray) -> float:
    reference = np.asarray(reference, np.float64)
    diff = np.abs(np.asarray(candidate, np.float64) - reference)
    return float(np.max(diff / np.maximum(np.abs(reference), 1.0e-12)))


def _empty_check(spec_name: str) -> dict[str, Any]:
    return {
        "field": spec_name,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "worst_case": "",
        "worst_domain": "",
        "passes": True,
    }


def _update_check(
    check: dict[str, Any],
    *,
    candidate: np.ndarray,
    reference: np.ndarray,
    case: str,
    domain: str,
    parity_tol: float,
    rel_tol: float,
) -> None:
    max_abs = _max_abs(candidate, reference)
    max_rel = _max_rel(candidate, reference)
    if max_abs > check["max_abs"]:
        check["max_abs"] = max_abs
        check["max_rel_at_worst_abs"] = max_rel
        check["worst_case"] = case
        check["worst_domain"] = domain
    if max_rel > check["max_rel"]:
        check["max_rel"] = max_rel
    check["passes"] = bool(check["passes"] and (max_abs <= parity_tol or (rel_tol > 0.0 and max_rel <= rel_tol)))


def _projection_report_item(data: Any, geo: Dataset, case: str, domain: str) -> dict[str, dict[str, Any]]:
    derived = data.lambert_grid.derive_fields()
    checks: dict[str, dict[str, Any]] = {}
    for spec in static_geog_specs():
        if spec.group not in ("coord", "mapfac"):
            continue
        check = _empty_check(spec.name)
        _update_check(
            check,
            candidate=derived[spec.name],
            reference=geo.variables[spec.name][0],
            case=case,
            domain=domain,
            parity_tol=spec.parity_tol,
            rel_tol=spec.rel_tol,
        )
        checks[spec.name] = check
    return checks


def _merge_checks(target: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]) -> None:
    for name, item in source.items():
        if name not in target:
            target[name] = item
            continue
        existing = target[name]
        if item["max_abs"] > existing["max_abs"]:
            existing["max_abs"] = item["max_abs"]
            existing["max_rel_at_worst_abs"] = item.get("max_rel_at_worst_abs", item["max_rel"])
            existing["worst_case"] = item["worst_case"]
            existing["worst_domain"] = item["worst_domain"]
        existing["max_rel"] = max(existing["max_rel"], item["max_rel"])
        existing["passes"] = bool(existing["passes"] and item["passes"])


def _update_non_s2_alias(
    checks: dict[str, dict[str, Any]],
    key: str,
    *,
    case: str,
    schema_field: str,
    geo_em_candidate: str,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> None:
    max_abs = _max_abs(candidate, reference)
    item = checks.setdefault(
        key,
        {
            "case": "",
            "schema_field": schema_field,
            "schema_source": "aifs_grib",
            "geo_em_candidate": geo_em_candidate,
            "max_abs": 0.0,
            "s2_owned": False,
        },
    )
    if max_abs >= item["max_abs"]:
        item["case"] = case
        item["max_abs"] = max_abs


def test_static_geog_loader_extracts_schema_static_subset_and_writes_report() -> None:
    specs = static_geog_specs()
    assert all(spec.source == "geo_em" and spec.group in STATIC_GEOG_GROUPS for spec in specs)
    assert "LANDSEA" not in static_geog_field_names()
    assert "SOILHGT" not in static_geog_field_names()

    geo_checks = {spec.name: _empty_check(spec.name) for spec in specs}
    met_em_checks = {spec.name: _empty_check(spec.name) for spec in specs}
    projection_checks: dict[str, dict[str, Any]] = {}
    category_bit_exact = {name: True for name in CATEGORY_FIELDS}
    non_s2_alias_checks: dict[str, dict[str, Any]] = {}

    cases = _case_dirs()
    for case in cases:
        for domain in DOMAINS:
            geo_path = case / "l3" / f"geo_em.{domain}.nc"
            met_path = _first_met_em(case, domain)
            data = load_static_geog(geo_path)
            assert set(data.arrays) == set(static_geog_field_names())
            assert data.domain == domain

            with Dataset(str(geo_path)) as geo, Dataset(str(met_path)) as met_em:
                for spec in specs:
                    candidate = data.arrays[spec.name]
                    geo_reference = np.asarray(geo.variables[spec.name][0], dtype=np.float32)
                    met_reference = np.asarray(met_em.variables[spec.name][0], dtype=np.float32)
                    _update_check(
                        geo_checks[spec.name],
                        candidate=candidate,
                        reference=geo_reference,
                        case=case.name,
                        domain=domain,
                        parity_tol=spec.parity_tol,
                        rel_tol=spec.rel_tol,
                    )
                    _update_check(
                        met_em_checks[spec.name],
                        candidate=candidate,
                        reference=met_reference,
                        case=case.name,
                        domain=domain,
                        parity_tol=spec.parity_tol,
                        rel_tol=spec.rel_tol,
                    )
                    if spec.name in category_bit_exact:
                        category_bit_exact[spec.name] = bool(
                            category_bit_exact[spec.name]
                            and np.array_equal(candidate, geo_reference)
                            and np.array_equal(candidate, met_reference)
                        )
                _merge_checks(projection_checks, _projection_report_item(data, geo, case.name, domain))

                landmask = np.asarray(geo.variables["LANDMASK"][0], dtype=np.float32)
                hgt_m = np.asarray(geo.variables["HGT_M"][0], dtype=np.float32)
                landsea = np.asarray(met_em.variables["LANDSEA"][0], dtype=np.float32)
                soilhgt = np.asarray(met_em.variables["SOILHGT"][0], dtype=np.float32)
                _update_non_s2_alias(
                    non_s2_alias_checks,
                    f"{domain}:LANDMASK_to_LANDSEA",
                    case=case.name,
                    schema_field="LANDSEA",
                    geo_em_candidate="LANDMASK",
                    candidate=landmask,
                    reference=landsea,
                )
                _update_non_s2_alias(
                    non_s2_alias_checks,
                    f"{domain}:HGT_M_to_SOILHGT",
                    case=case.name,
                    schema_field="SOILHGT",
                    geo_em_candidate="HGT_M",
                    candidate=hgt_m,
                    reference=soilhgt,
                )

    assert all(item["passes"] for item in geo_checks.values())
    assert all(item["passes"] for item in met_em_checks.values())
    assert all(item["passes"] for item in projection_checks.values())
    assert all(category_bit_exact.values())
    assert max(item["max_abs"] for item in non_s2_alias_checks.values()) > 1.0

    report = {
        "schema_version": load_static_geog(cases[0] / "l3" / "geo_em.d01.nc").schema_version,
        "cases": [case.name for case in cases],
        "domains": list(DOMAINS),
        "source_root": str(WPS_ROOT),
        "validation": {
            "geo_em_static_fields_pass": all(item["passes"] for item in geo_checks.values()),
            "met_em_static_block_pass": all(item["passes"] for item in met_em_checks.values()),
            "projection_derivation_pass": all(item["passes"] for item in projection_checks.values()),
            "category_bit_exact_pass": all(category_bit_exact.values()),
        },
        "field_checks_vs_geo_em": geo_checks,
        "field_checks_vs_met_em_static_block": met_em_checks,
        "projection_derivation_checks_vs_geo_em": projection_checks,
        "category_bit_exact": category_bit_exact,
        "non_s2_schema_alias_checks": non_s2_alias_checks,
    }
    PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROOF_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
