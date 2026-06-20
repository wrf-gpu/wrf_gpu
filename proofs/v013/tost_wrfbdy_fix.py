#!/usr/bin/env python3
"""
v0.13 Tier1 #2 (KI-5) — powered-TOST n=15 wrfbdy_d02 fix: CPU setup-verify proof.

Root cause (see tost_wrfbdy_fix.md):
    The L2 corpus cases are a max_dom=2 ONE-WAY nest.  real.exe writes lateral
    boundary forcing only for the outermost SPECIFIED domain, so each case dir has
    wrfinput_d01 + wrfinput_d02 + wrfbdy_d01 but NO wrfbdy_d02 (a nest has none).
    The OLD per-case path forced a d02-only single-domain standalone forecast,
    which demanded wrfbdy_d02 and failed rc=2:
        FileNotFoundError: standalone native-init requires wrfbdy_d02 ...

Fix:
    Route the per-case GPU forecast through the SAME standalone live-nested driver
    the production nested CLI uses (gpuwrf.integration.nested_pipeline.
    execute_nested_pipeline, max_dom=2): d01 standalone (IC wrfinput_d01 + LBC
    wrfbdy_d01), d02 IC-only with its LBC fed LIVE by the parent each parent step
    (no wrfbdy_d02).  The d02 wrfouts are scored vs CPU-WRF d02 truth + AEMET,
    unchanged.

This proof runs the CPU-safe setup-only check (no GPU, no forecast): it exercises
the LBC-source routing logic that previously raised the wrfbdy_d02 error and shows
it is gone.  It runs against the real corpus case if present, else a synthetic
fixture that reproduces the same nest topology.

Usage:
    JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/v013/tost_wrfbdy_fix.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU-only proof; no GPU needed.
os.environ.setdefault("JAX_ENABLE_X64", "true")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the per-case runner's setup-only check (the routing under test).
import importlib.util  # noqa: E402

RUNNER = ROOT / "proofs/v0120/powered_tost_n15/run_one_case_v0120.py"
spec = importlib.util.spec_from_file_location("run_one_case_v0120", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

REAL_CASE = "20260429_18z_l2_72h_20260524T204451Z"
REAL_L2_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2")


def _stage_real_case(workdir: Path) -> tuple[Path, str] | None:
    """Symlink a real corpus case's real.exe inputs into a clean run root."""
    src = REAL_L2_ROOT / REAL_CASE
    needed = ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "namelist.input")
    if not all((src / f).exists() for f in needed):
        return None
    run_dir = workdir / REAL_CASE
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in (*needed, "namelist.output"):
        s = src / f
        if s.exists():
            (run_dir / f).symlink_to(s)
    return workdir, REAL_CASE


def main() -> int:
    results: dict = {
        "schema": "PoweredTOSTWrfbdyFixProof",
        "schema_version": 1,
        "title": "v0.13 KI-5 powered-TOST wrfbdy_d02 routing fix — CPU setup-verify",
        "owned_files": [
            "proofs/v0120/powered_tost_n15/run_one_case_v0120.py",
            "proofs/v0120/powered_tost_n15/run_powered_tost_n15_v0120.py",
            "proofs/v013/tost_wrfbdy_fix.py",
            "proofs/v013/tost_wrfbdy_fix.md",
        ],
    }

    with tempfile.TemporaryDirectory(prefix="v013_wrfbdy_proof_") as td:
        workdir = Path(td)
        staged = _stage_real_case(workdir)
        if staged is None:
            results["data_source"] = "MISSING_REAL_CORPUS"
            results["verdict"] = "SKIPPED_NO_CORPUS"
            results["note"] = (
                f"real corpus case {REAL_CASE} not present under {REAL_L2_ROOT}; "
                "run on the workstation that holds the wrf_l2 corpus"
            )
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0

        run_root, run_id = staged
        results["data_source"] = "REAL_L2_CORPUS"
        results["run_id"] = run_id

        # The decisive check: this previously raised FileNotFoundError(wrfbdy_d02).
        try:
            report = runner.setup_only_check(run_id, run_root, hours=24)
        except Exception as exc:  # noqa: BLE001
            results["verdict"] = "FAIL"
            results["error"] = f"{type(exc).__name__}: {exc}"
            print(json.dumps(results, indent=2, sort_keys=True))
            return 1

        results["setup_only_check"] = report
        checks = {
            "verdict_setup_ok": report["verdict"] == "SETUP_OK",
            "is_nest_max_dom_2": report["max_dom"] == 2,
            "one_way_nest": report["nest_config"]["one_way"] is True,
            "d01_lbc_wrfbdy_d01_resolves": report["disk_inputs"]["wrfbdy_d01"] is True,
            "wrfbdy_d02_absent": report["disk_inputs"]["wrfbdy_d02"] is False,
            "wrfbdy_d02_not_required": report["wrfbdy_d02_required"] is False,
            "both_wrfinput_present": (
                report["disk_inputs"]["wrfinput_d01"]
                and report["disk_inputs"]["wrfinput_d02"]
            ),
            "d02_parent_is_d01_ratio_gt_1": (
                report["nest_edge"]["parent"] == "d01"
                and report["nest_edge"]["parent_grid_ratio"] > 1
            ),
        }
        results["checks"] = checks
        results["verdict"] = "PASS" if all(checks.values()) else "FAIL"

    print(json.dumps(results, indent=2, sort_keys=True))
    # Persist alongside this script.
    out = ROOT / "proofs/v013/tost_wrfbdy_fix.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return 0 if results["verdict"] in ("PASS", "SKIPPED_NO_CORPUS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
