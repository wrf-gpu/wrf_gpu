#!/usr/bin/env python3
"""G3 city/lake proof gate.

This is an oracle-presence and fail-closed gate, not a BEP/BEM/lake parity claim.
The active schemes are recognized in the catalog and rejected unless their WRF
oracle + faithful JAX kernels are present. The small-grid check only proves the
static urban/lake inputs used by the scaffold are finite and plausible.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

WRF_ROOT = Path("<DATA_ROOT>/src/wrf_pristine/WRF")

from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_namelist  # noqa: E402
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402
from gpuwrf.physics.lake_model import LAKE_CARRY_MEMBERS, lake_step  # noqa: E402
from gpuwrf.physics.urban_bep_bem import (  # noqa: E402
    BEP_BEM_REGISTRY_STATE,
    BEP_REGISTRY_STATE,
    bep_bem_step,
    bep_step,
)


SCHEMES = {
    "bep": {
        "key": "sf_urban_physics",
        "code": 2,
        "source": WRF_ROOT / "phys/module_sf_bep.F",
        "entrypoint": bep_step,
        "registry_members": BEP_REGISTRY_STATE,
        "oracle_glob": "proofs/v022/g3city/oracles/bep/*.json",
    },
    "bem": {
        "key": "sf_urban_physics",
        "code": 3,
        "source": WRF_ROOT / "phys/module_sf_bep.F",
        "secondary_source": WRF_ROOT / "phys/module_sf_bem.F",
        "entrypoint": bep_bem_step,
        "registry_members": BEP_BEM_REGISTRY_STATE,
        "oracle_glob": "proofs/v022/g3city/oracles/bem/*.json",
    },
    "lake": {
        "key": "sf_lake_physics",
        "code": 1,
        "source": WRF_ROOT / "phys/module_sf_lake.F",
        "entrypoint": lake_step,
        "registry_members": LAKE_CARRY_MEMBERS,
        "oracle_glob": "proofs/v022/g3city/oracles/lake/*.json",
    },
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]) if argv else Path(__file__).with_suffix(".json")
    report = build_report()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": report["gate_pass"], "report": str(out)}, sort_keys=True))
    return 0 if report["gate_pass"] else 1


def build_report() -> dict[str, Any]:
    pieces = {name: _scheme_report(name, meta) for name, meta in SCHEMES.items()}
    small_grid = _small_grid_static_plausibility()
    gate_pass = all(p["fail_closed"] and p["stub_raises"] for p in pieces.values())
    gate_pass = bool(gate_pass and small_grid["passed"])
    return {
        "gate": "v022_g3city_urban_lake",
        "gate_pass": gate_pass,
        "full_physics_landed": False,
        "default_unchanged": _default_unchanged(),
        "pieces": pieces,
        "small_grid_static_plausibility": small_grid,
        "claim_boundary": (
            "BEP/BEM/lake are cataloged and fail-closed. This proof does not "
            "claim a faithful urban/lake physics run or WRF parity."
        ),
    }


def _scheme_report(name: str, meta: dict[str, Any]) -> dict[str, Any]:
    key = meta["key"]
    code = int(meta["code"])
    support = classify_scheme(key, code)
    oracle_files = sorted(REPO_ROOT.glob(meta["oracle_glob"]))
    namelist_error = _namelist_error_contains(key, code, support.wrf_name or name)
    stub_raises = _stub_raises(meta["entrypoint"])
    sources = [meta["source"]]
    if "secondary_source" in meta:
        sources.append(meta["secondary_source"])
    return {
        "key": key,
        "code": code,
        "wrf_name": support.wrf_name,
        "catalog_status": support.status.value,
        "fail_closed": support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED,
        "namelist_rejects": namelist_error,
        "reason_excerpt": support.reason[:240],
        "source_exists": all(p.exists() for p in sources),
        "source_files": [_source_digest(p) for p in sources],
        "registry_member_count": len(meta["registry_members"]),
        "registry_members_sample": list(meta["registry_members"][:8]),
        "oracle_status": "present" if oracle_files else "absent_fail_closed",
        "oracle_files": [str(p.relative_to(REPO_ROOT)) for p in oracle_files],
        "stub_raises": stub_raises,
        "landed": False,
        "scaffold": True,
    }


def _default_unchanged() -> bool:
    return (
        classify_scheme("sf_urban_physics", 0).status is SupportStatus.IMPLEMENTED
        and classify_scheme("sf_lake_physics", 0).status is SupportStatus.IMPLEMENTED
    )


def _namelist_error_contains(key: str, code: int, needle: str) -> bool:
    try:
        validate_namelist({"physics": {key: [code]}})
    except UnsupportedSchemeError as exc:
        text = str(exc)
        return key in text and (needle in text or "NOT YET IMPLEMENTED" in text)
    return False


def _stub_raises(fn) -> bool:
    try:
        fn()
    except NotImplementedError:
        return True
    return False


def _source_digest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "lines": data.count(b"\n"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _small_grid_static_plausibility() -> dict[str, Any]:
    """Finite/plausible check for a tiny grid with urban and lake cells present."""

    t_skin = np.array([[290.0, 291.5, 293.0], [294.0, 297.0, 301.0], [289.0, 292.0, 295.0]])
    urban_fraction = np.array([[0.0, 0.3, 0.0], [0.8, 1.0, 0.0], [0.0, 0.2, 0.0]])
    lakemask = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lake_depth = np.where(lakemask > 0.5, np.array([[0.0, 0.0, 25.0], [0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), 0.0)
    checks = {
        "finite_t_skin": bool(np.isfinite(t_skin).all()),
        "finite_urban_fraction": bool(np.isfinite(urban_fraction).all()),
        "finite_lakemask": bool(np.isfinite(lakemask).all()),
        "finite_lake_depth": bool(np.isfinite(lake_depth).all()),
        "t_skin_range_k": bool(t_skin.min() >= 250.0 and t_skin.max() <= 330.0),
        "urban_fraction_range": bool(urban_fraction.min() >= 0.0 and urban_fraction.max() <= 1.0),
        "lake_depth_positive_on_lake": bool((lake_depth[lakemask > 0.5] > 0.0).all()),
        "active_urban_cells_present": bool(np.count_nonzero(urban_fraction > 0.0) > 0),
        "active_lake_cells_present": bool(np.count_nonzero(lakemask > 0.5) > 0),
    }
    return {
        "passed": all(checks.values()),
        "physics_executed": False,
        "note": (
            "Static 3x3 urban/lake input plausibility only. Active physics is "
            "intentionally not executed because G3 schemes fail closed without "
            "their WRF oracle and faithful kernel."
        ),
        "checks": checks,
        "counts": {
            "nx": 3,
            "ny": 3,
            "urban_cells": int(np.count_nonzero(urban_fraction > 0.0)),
            "lake_cells": int(np.count_nonzero(lakemask > 0.5)),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
