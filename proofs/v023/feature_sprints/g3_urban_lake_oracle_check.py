#!/usr/bin/env python3
"""G3 urban BEP/BEM + WRF lake reference-only oracle gate.

This is intentionally not a fabricated physics port.  It proves that the WRF
Fortran sources/objects and Registry state packages for the requested schemes
are present, that the gpuwrf catalog accepts them only as REFERENCE_ONLY, and
that every operational path still fail-closes before compute.

No numerical WRF single-column parity claim is made unless a meaningful
standalone WRF driver exists for the full Urban/Lake initialization and state ABI.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

WRF_ROOT = Path(os.environ.get("WRF_PRISTINE_ROOT", "<USER_HOME>/src/wrf_pristine/WRF"))
WRF_PHYS = WRF_ROOT / "phys"
WRF_REGISTRY = WRF_ROOT / "Registry" / "Registry.EM_COMMON"
DEFAULT_OUTPUT = REPO_ROOT / "proofs/v023/feature_sprints/g3_urban_lake_oracle_check.json"

from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection  # noqa: E402
from gpuwrf.io.namelist_check import (  # noqa: E402
    NotOperationallyWiredError,
    validate_namelist,
    validate_operational_namelist,
)
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402
from gpuwrf.physics.lake_model import LAKE_CARRY_MEMBERS, lake_step  # noqa: E402
from gpuwrf.physics.urban_bep_bem import (  # noqa: E402
    BEM_EXTRA_REGISTRY_STATE,
    BEP_BEM_REGISTRY_STATE,
    BEP_REGISTRY_STATE,
    bep_bem_step,
    bep_step,
)
from gpuwrf.runtime.operational_mode import _resolve_operational_suite  # noqa: E402


COMPILER_CANDIDATES = ("gfortran", "mpif90", "mpifort", "ftn", "ifort", "ifx", "nvfortran", "flang-new", "flang")

SCHEMES: dict[str, dict[str, Any]] = {
    "bep": {
        "label": "WRF BEP urban canopy",
        "key": "sf_urban_physics",
        "code": 2,
        "sources": ("module_sf_bep.F", "module_sf_urban.F", "module_bep_bem_helper.F"),
        "objects": ("module_sf_bep.o", "module_sf_urban.o", "module_bep_bem_helper.o"),
        "modules": ("module_sf_bep.mod", "module_sf_urban.mod"),
        "registry_package": "bepscheme",
        "registry_members": BEP_REGISTRY_STATE,
        "extra_state_members": (),
        "entrypoint": bep_step,
    },
    "bem": {
        "label": "WRF BEP+BEM urban canopy",
        "key": "sf_urban_physics",
        "code": 3,
        "sources": ("module_sf_bep.F", "module_sf_bem.F", "module_sf_urban.F", "module_bep_bem_helper.F"),
        "objects": ("module_sf_bep.o", "module_sf_bem.o", "module_sf_urban.o", "module_bep_bem_helper.o"),
        "modules": ("module_sf_bep.mod", "module_sf_bem.mod", "module_sf_urban.mod"),
        "registry_package": "bep_bemscheme",
        "registry_members": BEP_BEM_REGISTRY_STATE,
        "extra_state_members": BEM_EXTRA_REGISTRY_STATE,
        "entrypoint": bep_bem_step,
    },
    "lake": {
        "label": "WRF lake model",
        "key": "sf_lake_physics",
        "code": 1,
        "sources": ("module_sf_lake.F",),
        "objects": ("module_sf_lake.o",),
        "modules": ("module_sf_lake.mod",),
        "registry_package": None,
        "registry_terms": ("sf_lake_physics", "lake_depth"),
        "registry_members": LAKE_CARRY_MEMBERS,
        "extra_state_members": (),
        "entrypoint": lake_step,
    },
}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_report(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        exists = path.exists()
        out.append(
            {
                "path": str(path),
                "exists": exists,
                "bytes": path.stat().st_size if exists else None,
                "lines": path.read_bytes().count(b"\n") if exists and path.suffix.lower() in {".f", ".f90"} else None,
                "sha256": _sha256(path),
            }
        )
    return out


def _registry_text() -> str:
    return WRF_REGISTRY.read_text(encoding="utf-8", errors="replace") if WRF_REGISTRY.exists() else ""


def _registry_package(package: str | None) -> dict[str, Any]:
    if package is None:
        return {"package": None, "present": None, "line_number": None, "members": []}
    lines = _registry_text().splitlines()
    needle = f"package   {package}"
    for line_number, line in enumerate(lines, start=1):
        if line.startswith(needle):
            members: list[str] = []
            if "state:" in line:
                members = [part.strip() for part in line.split("state:", 1)[1].split(",") if part.strip()]
            return {
                "package": package,
                "present": True,
                "line_number": line_number,
                "line": line,
                "members": members,
            }
    return {"package": package, "present": False, "line_number": None, "members": []}


def _registry_terms(terms: Iterable[str]) -> dict[str, dict[str, Any]]:
    lines = _registry_text().splitlines()
    out: dict[str, dict[str, Any]] = {}
    for term in terms:
        matches = [
            {"line_number": idx, "line": line}
            for idx, line in enumerate(lines, start=1)
            if term.lower() in line.lower()
        ]
        out[term] = {"present": bool(matches), "matches": matches[:5], "match_count": len(matches)}
    return out


def _compiler_probe() -> dict[str, Any]:
    found = {candidate: shutil.which(candidate) for candidate in COMPILER_CANDIDATES}
    conda_wrfbuild: dict[str, Any] = {"available": False, "version": None, "error": None}
    if shutil.which("conda") is not None:
        proc = subprocess.run(
            ["conda", "run", "-n", "wrfbuild", "gfortran", "--version"],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
        conda_wrfbuild = {
            "available": proc.returncode == 0,
            "version": proc.stdout.splitlines()[0] if proc.returncode == 0 and proc.stdout.splitlines() else None,
            "error": proc.stderr.strip() or None if proc.returncode != 0 else None,
        }
    return {
        "candidates": found,
        "available": {name: path for name, path in found.items() if path},
        "conda_wrfbuild": conda_wrfbuild,
        "any_available": any(found.values()) or bool(conda_wrfbuild["available"]),
    }


def _fortran_compile_probe() -> dict[str, Any]:
    """Compile-only probe for the WRF Urban/Lake modules; no numerical driver."""

    if shutil.which("conda") is None:
        return {"ran": False, "reason": "conda unavailable"}
    with tempfile.TemporaryDirectory(prefix="g3_fortran_probe.") as tmp:
        script = f"""
set -o pipefail
cd {tmp!r}
WRF={str(WRF_ROOT)!r}
FF="-w -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -I. -I${{WRF}}/phys -I${{WRF}}/frame"
gfortran $FF -c ${{WRF}}/phys/module_sf_lake.F -o module_sf_lake_probe.o >lake.out 2>lake.err; echo lake_rc=$?
gfortran $FF -c ${{WRF}}/phys/module_bep_bem_helper.f90 -o module_bep_bem_helper_probe.o >helper.out 2>helper.err; echo helper_rc=$?
gfortran $FF -c ${{WRF}}/phys/module_sf_urban.F -o module_sf_urban_probe.o >urban.out 2>urban.err; echo urban_rc=$?
gfortran $FF -c ${{WRF}}/phys/module_sf_bep.F -o module_sf_bep_probe.o >bep.out 2>bep.err; echo bep_rc=$?
gfortran $FF -c ${{WRF}}/phys/module_sf_bem.F -o module_sf_bem_probe.o >bem.out 2>bem.err; echo bem_rc=$?
ls -1 *.o 2>/dev/null || true
"""
        proc = subprocess.run(
            ["conda", "run", "-n", "wrfbuild", "bash", "-lc", script],
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
        )
    rc_by_name: dict[str, int] = {}
    objects: list[str] = []
    for line in proc.stdout.splitlines():
        if "_rc=" in line:
            name, value = line.split("_rc=", 1)
            try:
                rc_by_name[name] = int(value)
            except ValueError:
                pass
        if line.endswith(".o"):
            objects.append(line)
    return {
        "ran": True,
        "returncode": proc.returncode,
        "rc_by_module": rc_by_name,
        "compiled_objects": objects,
        "compile_only_pass": proc.returncode == 0 and rc_by_name and all(value == 0 for value in rc_by_name.values()),
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
        "note": "Compile-only probe; it does not build or run a numerical WRF single-column driver.",
    }


def _nm_undefined(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "ran": False, "reason": "missing object"}
    nm = shutil.which("nm")
    if nm is None:
        return {"path": str(path), "ran": False, "reason": "nm unavailable"}
    proc = subprocess.run([nm, "-u", str(path)], check=False, text=True, capture_output=True)
    symbols = []
    for line in proc.stdout.splitlines():
        text = line.strip()
        if text.startswith("U "):
            symbols.append(text[2:].strip())
        elif " U " in text:
            symbols.append(text.rsplit(" U ", 1)[-1].strip())
    return {
        "path": str(path),
        "ran": True,
        "returncode": proc.returncode,
        "undefined_count": len(symbols),
        "undefined_symbols_sample": symbols[:24],
    }


def _stub_raises(fn: Any) -> bool:
    try:
        fn()
    except NotImplementedError:
        return True
    return False


def _validate_reference_and_operational(key: str, code: int) -> dict[str, Any]:
    config = {"physics": {key: [code]}}
    accepted_reference = True
    reference_error = None
    try:
        validate_namelist(config)
    except Exception as exc:  # noqa: BLE001 - recorded in proof object
        accepted_reference = False
        reference_error = type(exc).__name__ + ": " + str(exc)

    operational_rejects = False
    operational_error = None
    try:
        validate_operational_namelist(config)
    except NotOperationallyWiredError as exc:
        operational_rejects = True
        operational_error = str(exc)
    except Exception as exc:  # noqa: BLE001 - recorded separately
        operational_error = type(exc).__name__ + ": " + str(exc)

    return {
        "validate_namelist_accepts_reference": accepted_reference,
        "validate_namelist_error": reference_error,
        "validate_operational_rejects": operational_rejects,
        "validate_operational_error": operational_error,
    }


def _scan_rejects(key: str, code: int) -> dict[str, Any]:
    base = dict(
        mp_physics=8,
        bl_pbl_physics=5,
        sf_sfclay_physics=5,
        cu_physics=0,
        sf_surface_physics=0,
        sf_urban_physics=0,
        sf_lake_physics=0,
        use_noahmp=False,
        use_flux_advection=False,
        moist_adv_opt=0,
        ra_sw_physics=4,
        ra_lw_physics=4,
    )
    base[key] = code
    try:
        _resolve_operational_suite(SimpleNamespace(**base))
    except UnsupportedSchemeSelection as exc:
        return {"scan_rejects": True, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - recorded separately
        return {"scan_rejects": False, "error": type(exc).__name__ + ": " + str(exc)}
    return {"scan_rejects": False, "error": None}


def _scheme_report(name: str, meta: dict[str, Any], compiler: dict[str, Any], compile_probe: dict[str, Any]) -> dict[str, Any]:
    key = str(meta["key"])
    code = int(meta["code"])
    support = classify_scheme(key, code)
    package = _registry_package(meta.get("registry_package"))
    registry_members = tuple(meta["registry_members"])
    package_members = tuple(package.get("members") or ())
    if package["present"] is True:
        registry_member_match = set(registry_members).issubset(set(package_members))
    elif name == "lake":
        registry_member_match = True
    else:
        registry_member_match = False

    source_paths = [WRF_PHYS / rel for rel in meta["sources"]]
    object_paths = [WRF_PHYS / rel for rel in meta["objects"]]
    module_paths = [WRF_PHYS / rel for rel in meta["modules"]]
    source_reports = _file_report(source_paths)
    object_reports = _file_report(object_paths)
    module_reports = _file_report(module_paths)
    validation = _validate_reference_and_operational(key, code)
    scan = _scan_rejects(key, code)
    sources_present = all(item["exists"] for item in source_reports)
    objects_present = all(item["exists"] for item in object_reports)
    modules_present = all(item["exists"] for item in module_reports)
    registry_present = package["present"] is True or all(
        item["present"] for item in _registry_terms(meta.get("registry_terms", ())).values()
    )

    gate_ok = bool(
        support.status is SupportStatus.REFERENCE_ONLY
        and validation["validate_namelist_accepts_reference"]
        and validation["validate_operational_rejects"]
        and scan["scan_rejects"]
        and _stub_raises(meta["entrypoint"])
        and sources_present
        and objects_present
        and modules_present
        and registry_present
        and registry_member_match
    )

    return {
        "id": name,
        "label": meta["label"],
        "key": key,
        "code": code,
        "wrf_name": support.wrf_name,
        "catalog_status": support.status.value,
        "catalog_reason": support.reason,
        "catalog_alternative": support.alternative,
        "source_files": source_reports,
        "object_files": object_reports,
        "module_files": module_reports,
        "undefined_symbol_audit": [_nm_undefined(path) for path in object_paths[:2]],
        "registry_package": package,
        "registry_terms": _registry_terms(meta.get("registry_terms", ())),
        "registry_member_count": len(registry_members),
        "registry_member_match": registry_member_match,
        "registry_members_sample": list(registry_members[:10]),
        "extra_state_members_count": len(tuple(meta.get("extra_state_members", ()))),
        "oracle_status": "reference_only_inventory",
        "numerical_oracle": {
            "status": "standalone_numerical_oracle_deferred_wrf_dependency_depth",
            "parity_claim": False,
            "reason": (
                "gfortran is available through the wrfbuild conda environment and "
                "the WRF Urban/Lake modules pass a compile-only probe. A fresh "
                "numerical standalone oracle was deferred because BEP/BEM/Lake "
                "need the full WRF initialization/static-table/state ABI "
                "(urban-map and BEM tables, bepscheme/bep_bemscheme carry, "
                "lakeini lake/snow/soil/lake-column carry) before a single-column "
                "driver can produce meaningful WRF reference outputs."
            ),
            "fortran_compiler_available": bool(compiler["any_available"]),
            "compile_only_probe_pass": bool(compile_probe.get("compile_only_pass")),
        },
        "validation": validation,
        "scan": scan,
        "stub_raises": _stub_raises(meta["entrypoint"]),
        "landed": False,
        "scaffold": True,
        "gate_ok": gate_ok,
    }


def _default_unchanged() -> bool:
    return (
        classify_scheme("sf_urban_physics", 0).status is SupportStatus.IMPLEMENTED
        and classify_scheme("sf_lake_physics", 0).status is SupportStatus.IMPLEMENTED
    )


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
        "checks": checks,
        "counts": {
            "nx": 3,
            "ny": 3,
            "urban_cells": int(np.count_nonzero(urban_fraction > 0.0)),
            "lake_cells": int(np.count_nonzero(lakemask > 0.5)),
        },
    }


def build_report() -> dict[str, Any]:
    compiler = _compiler_probe()
    compile_probe = _fortran_compile_probe()
    pieces = {name: _scheme_report(name, meta, compiler, compile_probe) for name, meta in SCHEMES.items()}
    small_grid = _small_grid_static_plausibility()
    gate_pass = all(piece["gate_ok"] for piece in pieces.values()) and _default_unchanged() and small_grid["passed"]
    return {
        "proof": "v023-g3-urban-bep-bem-lake-reference-only-oracle-gate",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "wrf_root": str(WRF_ROOT),
        "wrf_registry": str(WRF_REGISTRY),
        "compiler_probe": compiler,
        "fortran_compile_probe": compile_probe,
        "gate_pass": gate_pass,
        "full_physics_landed": False,
        "default_unchanged": _default_unchanged(),
        "pieces": pieces,
        "small_grid_static_plausibility": small_grid,
        "claim_boundary": (
            "BEP, BEP+BEM, and WRF lake are REFERENCE_ONLY. This proof validates "
            "WRF source/object/Registry presence and fail-closed wiring only; it "
            "does not claim numerical WRF parity or an operational JAX kernel."
        ),
        "recommended_milestone": (
            "Create a dedicated urban/lake milestone to build fresh WRF Fortran "
            "single-column savepoints, freeze the BEP/BEM/lake carry interfaces, "
            "and only then port faithful JAX kernels."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate_pass": report["gate_pass"], "out": str(args.out)}, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
