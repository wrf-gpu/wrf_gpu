"""F2 missing-scheme bundle oracle gate.

This proof checks the four requested schemes as WRF-source-recognized endpoints
and verifies the runtime gate is honest:

* schemes with local single-column evidence may be reference-only, but must not
  be operationally scan-wired;
* schemes without local single-column WRF oracle evidence must remain
  fail-closed at the namelist/catalog layer.

It does not run WRF or JAX kernels. It inspects committed proof objects and the
pristine WRF source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[2]
WRF_ROOT = Path(os.environ.get("WRF_PRISTINE_ROOT", "<DATA_ROOT>/src/wrf_pristine/WRF"))
REGISTRY = WRF_ROOT / "Registry" / "Registry.EM_COMMON"

DEFAULT_OUTPUT = REPO_ROOT / "proofs" / "v022" / "f2_missing_scheme_bundle_oracle_check.json"


TARGETS: tuple[dict[str, Any], ...] = (
    {
        "id": "new_tiedtke",
        "label": "New-Tiedtke cumulus",
        "key": "cu_physics",
        "code": 16,
        "wrf_sources": ("phys/module_cu_ntiedtke.F",),
        "registry_packages": ("ntiedtkescheme",),
        "oracle_kind": "single_column_savepoints",
        "oracle_globs": ("proofs/v013/savepoints/cumulus/ntiedtke_case_*.json",),
        "required_status": "reference_only",
    },
    {
        "id": "nssl_2mom",
        "label": "NSSL 2-moment microphysics",
        "key": "mp_physics",
        "code": 18,
        "wrf_sources": ("phys/module_mp_nssl_2mom.F",),
        "registry_packages": (
            "nssl_2mom",
            "nssl2mconc",
            "nssl_hail",
            "nssl_ccn_opt",
            "nssl_graupelvol",
            "nssl_hailvol",
        ),
        "oracle_kind": "single_column_savepoints",
        "oracle_globs": ("proofs/v022/f2_oracles/nssl_2mom/*.json",),
        "required_status": "recognized_fail_closed_if_oracle_absent",
    },
    {
        "id": "morrison_aero",
        "label": "Aerosol-coupled Morrison microphysics",
        "key": "mp_physics",
        "code": 40,
        "wrf_sources": (
            "phys/module_mp_morr_two_moment.F",
            "phys/module_mp_morr_two_moment_aero.F",
        ),
        "registry_packages": ("morr_tm_aero",),
        "oracle_kind": "single_column_savepoints",
        "oracle_globs": ("proofs/v022/f2_oracles/morrison_aero/*.json",),
        "required_status": "recognized_fail_closed_if_oracle_absent",
    },
    {
        "id": "ruc_lsm",
        "label": "RUC land-surface model",
        "key": "sf_surface_physics",
        "code": 3,
        "wrf_sources": ("phys/module_sf_ruclsm.F",),
        "registry_packages": ("ruclsmscheme",),
        "oracle_kind": "single_column_metrics",
        "oracle_metrics": "proofs/v018/ruc_lsm_parity_metrics.json",
        "oracle_raw_savepoint": "proofs/v017/savepoints/ruclsm/fp64/ruclsm_fp64.json",
        "required_status": "reference_only",
    },
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
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
    return ()


def _registry_lines(packages: Iterable[str]) -> dict[str, dict[str, Any]]:
    text = REGISTRY.read_text() if REGISTRY.exists() else ""
    lines = text.splitlines()
    out: dict[str, dict[str, Any]] = {}
    for package in packages:
        needle = f"package   {package}"
        match = next(
            ((idx, line) for idx, line in enumerate(lines, start=1) if line.startswith(needle)),
            None,
        )
        out[package] = {
            "present": match is not None,
            "line_number": match[0] if match else None,
            "line": match[1] if match else None,
        }
    return out


def _source_report(sources: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in sources:
        path = WRF_ROOT / rel
        out.append(
            {
                "path": str(path),
                "relative_path": rel,
                "exists": path.exists(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
    return out


def _savepoint_oracle(target: dict[str, Any]) -> dict[str, Any]:
    files: list[Path] = []
    for pattern in target["oracle_globs"]:
        files.extend(sorted(REPO_ROOT.glob(pattern)))
    files = sorted(dict.fromkeys(files))

    all_finite = True
    schemas: set[str] = set()
    raincv_values: list[float] = []
    numeric_count = 0
    for path in files:
        data = json.loads(path.read_text())
        if isinstance(data.get("schema"), str):
            schemas.add(data["schema"])
        numbers = list(_walk_numbers(data))
        numeric_count += len(numbers)
        all_finite = all_finite and all(math.isfinite(v) for v in numbers)
        if isinstance(data.get("outputs"), dict) and "RAINCV" in data["outputs"]:
            raincv_values.append(float(data["outputs"]["RAINCV"]))
        if isinstance(data.get("scalars"), dict) and "RAINCV" in data["scalars"]:
            raincv_values.append(float(data["scalars"]["RAINCV"]))

    return {
        "kind": target["oracle_kind"],
        "present": bool(files),
        "files": [str(path.relative_to(REPO_ROOT)) for path in files],
        "file_count": len(files),
        "all_finite": all_finite if files else False,
        "schemas": sorted(schemas),
        "numeric_count": numeric_count,
        "nontrivial": any(abs(v) > 0.0 for v in raincv_values) if files else False,
        "raincv_values": raincv_values,
    }


def _metrics_oracle(target: dict[str, Any]) -> dict[str, Any]:
    metrics_path = REPO_ROOT / target["oracle_metrics"]
    raw_path = REPO_ROOT / target["oracle_raw_savepoint"]
    present = metrics_path.exists()
    data = json.loads(metrics_path.read_text()) if present else {}
    fields = data.get("fields", {})
    return {
        "kind": target["oracle_kind"],
        "present": present and bool(data.get("all_green")),
        "metrics_path": str(metrics_path.relative_to(REPO_ROOT)),
        "metrics_exists": present,
        "metrics_sha256": _sha256(metrics_path),
        "schema": data.get("schema"),
        "all_green": data.get("all_green"),
        "field_count": len(fields) if isinstance(fields, dict) else 0,
        "n_columns": data.get("n_columns"),
        "n_steps": data.get("n_steps"),
        "worst_abs": data.get("worst_abs"),
        "worst_rel": data.get("worst_rel"),
        "raw_savepoint_path": str(raw_path.relative_to(REPO_ROOT)),
        "raw_savepoint_present": raw_path.exists(),
    }


def _oracle_report(target: dict[str, Any]) -> dict[str, Any]:
    if target["oracle_kind"] == "single_column_metrics":
        return _metrics_oracle(target)
    return _savepoint_oracle(target)


def _target_report(target: dict[str, Any]) -> dict[str, Any]:
    from gpuwrf.contracts.physics_registry import ACCEPTED_NAMELIST_OPTIONS
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme
    from gpuwrf.runtime.operational_mode import _SCAN_UNWIRED_REASON, _SCAN_WIRED_OPTIONS

    key = target["key"]
    code = int(target["code"])
    support = classify_scheme(key, code)
    accepted = code in ACCEPTED_NAMELIST_OPTIONS.get(key, ())
    scan_wired = code in _SCAN_WIRED_OPTIONS.get(key, ())
    scan_reason = _SCAN_UNWIRED_REASON.get(f"{key}={code}")
    oracle = _oracle_report(target)

    if oracle["present"]:
        expected_status = SupportStatus.REFERENCE_ONLY
        coverage = "reference_only_oracle_present"
        gate_ok = (
            support.status is expected_status
            and accepted
            and not scan_wired
            and bool(scan_reason)
        )
    else:
        expected_status = SupportStatus.RECOGNIZED_FAIL_CLOSED
        coverage = "fail_closed_oracle_absent"
        gate_ok = (
            support.status is expected_status
            and not accepted
            and not scan_wired
            and bool(scan_reason)
        )

    return {
        "id": target["id"],
        "label": target["label"],
        "key": key,
        "code": code,
        "wrf_sources": _source_report(target["wrf_sources"]),
        "registry_packages": _registry_lines(target["registry_packages"]),
        "catalog_status": support.status.value,
        "catalog_reason": support.reason,
        "catalog_alternative": support.alternative,
        "accepted_by_reference_validator": accepted,
        "scan_wired": scan_wired,
        "scan_unwired_reason": scan_reason,
        "oracle": oracle,
        "coverage": coverage,
        "gate_ok": gate_ok,
    }


def build_report() -> dict[str, Any]:
    schemes = {target["id"]: _target_report(target) for target in TARGETS}
    return {
        "proof": "v022-f2-missing-scheme-bundle-oracle-gate",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wrf_root": str(WRF_ROOT),
        "registry": str(REGISTRY),
        "gate_pass": all(entry["gate_ok"] for entry in schemes.values()),
        "full_bundle_landed": all(
            entry["catalog_status"] == "implemented" and entry["scan_wired"]
            for entry in schemes.values()
        ),
        "schemes": schemes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": report["gate_pass"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
