#!/usr/bin/env python3
"""Validate the v0.18 PBL4/QNSE pristine-WRF reference savepoints."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SP = ROOT / "proofs" / "v018" / "savepoints_fp64" / "qnse"
REPORT = ROOT / "proofs" / "v018" / "qnse_pbl4_reference_oracle.json"
CASES = (1, 2, 3, 4, 5, 6)


def _load(case: int) -> dict[str, Any]:
    return json.loads((SP / f"qnse_case_{case}.json").read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    cases = []
    all_finite = True
    for case in CASES:
        data = _load(case)
        finite = {}
        extrema = {}
        for name, values in data["columns"].items():
            arr = np.asarray(values, dtype=np.float64)
            ok = bool(np.all(np.isfinite(arr)))
            finite[name] = ok
            all_finite = all_finite and ok
            extrema[name] = {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        scalar_finite = {}
        for name, value in data["scalars"].items():
            if isinstance(value, (int, float)):
                ok = bool(math.isfinite(float(value)))
                scalar_finite[name] = ok
                all_finite = all_finite and ok
        cases.append({
            "case": case,
            "regime": data["scalars"]["REGIME"],
            "finite": finite,
            "scalar_finite": scalar_finite,
            "extrema": extrema,
            "pblh_m": float(data["scalars"]["PBL"]),
            "kpbl": int(data["scalars"]["KPBL"]),
        })

    report = {
        "schema": "gpuwrf.v018.qnse_pbl4_reference_oracle.v1",
        "scheme": "QNSE-EDMF PBL (bl_pbl_physics=4)",
        "endpoint_class": "reference_with_real_oracle",
        "operational_kernel": None,
        "oracle": {
            "type": "single-column Fortran driver linked against UNMODIFIED pristine WRF phys/module_bl_qnsepbl.F",
            "savepoints": str(SP.relative_to(ROOT)),
            "full_wrf_exe": False,
            "self_compare": False,
            "source_checksums_sha256": _read_lines(SP / "qnse_wrf_source_checksums.txt"),
            "build_manifest": _read_lines(SP / "qnse_build_manifest.txt"),
        },
        "git_head": _git_head(),
        "verdict": "PASS" if all_finite else "FAIL",
        "all_finite": bool(all_finite),
        "cases": cases,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all_finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
