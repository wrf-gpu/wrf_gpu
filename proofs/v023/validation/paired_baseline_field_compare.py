#!/usr/bin/env python3
"""Fresh paired-baseline wrfout field comparison for v0.23 release gates.

This is deliberately a wrfout field comparator, not a stored-digest checker.
It reuses ``scripts/compare_wrfout_grid.py`` and adds the release-gate policy
needed by v0.23:

* compare a fresh same-environment baseline directory against a fresh candidate
  directory;
* predeclare exact or autotune-floor tolerances before the comparison starts;
* fail on missing/incompatible wrfout variables, unequal time metadata, or any
  tolerance failure;
* write one aggregate JSON verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[3]
COMPARE = ROOT / "scripts" / "compare_wrfout_grid.py"
WRFOUT_PREFIX = "wrfout_"
TIME_METADATA_FIELDS = {"Times", "XTIME"}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _wrfout_files(root: Path, domain: str) -> dict[str, Path]:
    files = {}
    for path in sorted(root.glob(f"{WRFOUT_PREFIX}{domain}_*")):
        if path.is_file():
            files[path.name] = path
    return files


def _is_numeric(var: Any) -> bool:
    return np.dtype(var.dtype).kind in {"b", "i", "u", "f", "c"}


def _same_shape_and_dims(left: Any, right: Any) -> bool:
    return tuple(left.dimensions) == tuple(right.dimensions) and tuple(left.shape) == tuple(right.shape)


def _load_user_tolerances(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"tolerance JSON must be an object: {path}")
    for key in ("fields", "variables", "tolerances"):
        value = payload.get(key)
        if isinstance(value, dict):
            return dict(value)
    return dict(payload)


def _load_manifest_fields(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("fields"), dict):
        raise ValueError(f"tolerance manifest missing object field 'fields': {path}")
    return dict(payload["fields"])


def _make_tolerance_manifest(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    domain: str,
    mode: str,
    autotune_floor: float,
    user_tolerances: dict[str, Any],
    requested_vars: list[str] | None,
    output: Path,
) -> dict[str, Any]:
    base_files = _wrfout_files(baseline_dir, domain)
    cand_files = _wrfout_files(candidate_dir, domain)
    common = sorted(set(base_files) & set(cand_files))
    if not common:
        raise FileNotFoundError(f"no paired wrfout files for {domain}: {baseline_dir} vs {candidate_dir}")

    selected = set(requested_vars or [])
    fields: dict[str, dict[str, float]] = {}
    first_base = base_files[common[0]]
    first_cand = cand_files[common[0]]
    with Dataset(first_base, "r") as bds, Dataset(first_cand, "r") as cds:
        for name in sorted(set(bds.variables) & set(cds.variables)):
            if selected and name not in selected:
                continue
            if name in TIME_METADATA_FIELDS:
                continue
            bvar = bds.variables[name]
            cvar = cds.variables[name]
            if not (_is_numeric(bvar) and _is_numeric(cvar) and _same_shape_and_dims(bvar, cvar)):
                continue
            if mode == "exact":
                fields[name] = {"max_abs": 0.0, "finite_pair_fraction_min": 1.0}
            elif mode == "autotune-floor":
                fields[name] = {"max_abs": float(autotune_floor), "finite_pair_fraction_min": 1.0}
            else:
                if name in user_tolerances:
                    fields[name] = user_tolerances[name]

    if mode == "tolerance":
        for name, spec in user_tolerances.items():
            if not selected or name in selected:
                fields[name] = spec

    if not fields:
        raise ValueError(f"no comparable numeric fields selected for {domain}")

    payload = {
        "schema": "v023-field-tolerance-manifest-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Predeclared before comparison; no tolerance is tuned from observed candidate output.",
        "mode": mode,
        "autotune_floor": float(autotune_floor),
        "fields": fields,
    }
    _write_json(output, payload)
    return payload


def _run_compare(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    domain: str,
    init: str | None,
    min_lead: int | None,
    max_lead: int | None,
    vars_: list[str] | None,
    tolerance_json: Path,
    out_json: Path,
    out_md: Path,
    progress: int,
    no_spatial_splits: bool,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(COMPARE),
        "--cpu-dir",
        str(baseline_dir),
        "--gpu-dir",
        str(candidate_dir),
        "--domain",
        domain,
        "--tolerance-json",
        str(tolerance_json),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    if init:
        cmd.extend(["--init", init])
    if min_lead is not None:
        cmd.extend(["--min-lead", str(min_lead)])
    if max_lead is not None:
        cmd.extend(["--max-lead", str(max_lead)])
    if vars_:
        cmd.append("--vars")
        cmd.extend(vars_)
    if progress:
        cmd.extend(["--progress", str(progress)])
    if no_spatial_splits:
        cmd.append("--no-spatial-splits")

    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, env=env, check=False)
    stdout = out_json.with_suffix(".stdout.txt")
    stderr = out_json.with_suffix(".stderr.txt")
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {
            "domain": domain,
            "status": "BLOCKED",
            "blocked": True,
            "returncode": proc.returncode,
            "command": cmd,
            "stdout": str(stdout),
            "stderr": str(stderr),
            "report_json": str(out_json),
            "report_md": str(out_md),
            "pass": False,
            "failures": [{"kind": "compare_script_failed", "returncode": proc.returncode}],
        }
    report = json.loads(out_json.read_text(encoding="utf-8"))
    return _evaluate_domain_report(domain, report, tolerance_json, out_json, out_md, cmd, stdout, stderr)


def _evaluate_domain_report(
    domain: str,
    report: dict[str, Any],
    tolerance_json: Path,
    out_json: Path,
    out_md: Path,
    cmd: list[str],
    stdout: Path,
    stderr: Path,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    manifest_fields = _load_manifest_fields(tolerance_json)
    summaries = report.get("summaries", {})
    inventory = report.get("inventory", {})
    field_summaries = report.get("field_summaries") or {}
    if summaries.get("verdict") != "PASS":
        failures.append({"kind": "tolerance_verdict", "verdict": summaries.get("verdict")})
    for key in ("cpu_only_count", "gpu_only_count"):
        count = int(inventory.get(key, 0) or 0)
        if count:
            failures.append({"kind": key, "count": count, "examples": inventory.get(key.replace("_count", ""), [])[:12]})
    incompatible = inventory.get("incompatible_common") or []
    if incompatible:
        failures.append({"kind": "incompatible_common", "count": len(incompatible), "examples": incompatible[:5]})
    non_numeric = inventory.get("non_numeric_common") or []
    unexpected_non_numeric = [item for item in non_numeric if item.get("field") not in TIME_METADATA_FIELDS]
    if unexpected_non_numeric:
        failures.append({"kind": "unexpected_non_numeric_common", "count": len(unexpected_non_numeric), "examples": unexpected_non_numeric[:5]})

    comparable_count = int(summaries.get("comparable_field_count", 0) or 0)
    if comparable_count < len(manifest_fields):
        failures.append(
            {
                "kind": "manifest_comparable_field_count_shortfall",
                "manifest_field_count": len(manifest_fields),
                "comparable_field_count": comparable_count,
            }
        )
    for name in sorted(manifest_fields):
        field = field_summaries.get(name)
        if field is None:
            failures.append({"kind": "manifest_field_missing_from_report", "field": name})
            continue
        overall = field.get("overall") or {}
        finite_pair = int(overall.get("finite_pair", 0) or 0)
        if finite_pair <= 0:
            failures.append(
                {
                    "kind": "manifest_field_no_finite_pairs",
                    "field": name,
                    "finite_pair": finite_pair,
                    "finite_cpu": overall.get("finite_cpu"),
                    "finite_gpu": overall.get("finite_gpu"),
                    "n": overall.get("n"),
                }
            )

    for name, field in field_summaries.items():
        if name in TIME_METADATA_FIELDS and field.get("classification") == "time_metadata" and not bool(field.get("all_equal")):
            failures.append({"kind": "time_metadata_mismatch", "field": name})
        if field.get("missing_leads"):
            failures.append({"kind": "missing_leads", "field": name, "examples": field.get("missing_leads", [])[:5]})
        if field.get("incompatible_leads"):
            failures.append({"kind": "incompatible_leads", "field": name, "examples": field.get("incompatible_leads", [])[:5]})

    return {
        "domain": domain,
        "status": "PASS" if not failures else "FAIL",
        "pass": not failures,
        "failures": failures,
        "summary_verdict": summaries.get("verdict"),
        "paired_files": report.get("pairing", {}).get("paired_file_count"),
        "compared_fields": summaries.get("comparable_field_count"),
        "manifest_field_count": len(manifest_fields),
        "tolerance_failure_count": summaries.get("tolerance_failure_count"),
        "top_fields_by_severity": summaries.get("top_fields_by_severity", [])[:10],
        "report_json": str(out_json),
        "report_md": str(out_md),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "command": cmd,
    }


def run_compare(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if not COMPARE.is_file():
        raise FileNotFoundError(f"missing reused comparator: {COMPARE}")
    user_tolerances = _load_user_tolerances(args.tolerance_json)
    work_dir = args.work_dir or args.out_json.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)

    domain_results = []
    for domain in args.domains:
        dom_dir = work_dir / domain
        dom_dir.mkdir(parents=True, exist_ok=True)
        tol_path = dom_dir / "tolerances.json"
        try:
            _make_tolerance_manifest(
                baseline_dir=args.baseline_dir,
                candidate_dir=args.candidate_dir,
                domain=domain,
                mode=args.mode,
                autotune_floor=args.autotune_floor,
                user_tolerances=user_tolerances,
                requested_vars=args.vars,
                output=tol_path,
            )
            domain_results.append(
                _run_compare(
                    baseline_dir=args.baseline_dir,
                    candidate_dir=args.candidate_dir,
                    domain=domain,
                    init=args.init,
                    min_lead=args.min_lead,
                    max_lead=args.max_lead,
                    vars_=args.vars,
                    tolerance_json=tol_path,
                    out_json=dom_dir / "compare.json",
                    out_md=dom_dir / "compare.md",
                    progress=args.progress,
                    no_spatial_splits=args.no_spatial_splits,
                )
            )
        except Exception as exc:
            domain_results.append(
                {
                    "domain": domain,
                    "status": "BLOCKED",
                    "blocked": True,
                    "pass": False,
                    "failures": [{"kind": "compare_wrapper_exception", "error": str(exc)}],
                    "report_json": str(dom_dir / "compare.json"),
                    "report_md": str(dom_dir / "compare.md"),
                }
            )

    failures = [
        {"domain": row["domain"], "failures": row["failures"]}
        for row in domain_results
        if not row.get("pass")
    ]
    blocked = [row for row in domain_results if row.get("blocked") or row.get("status") == "BLOCKED"]
    verdict = "BLOCKED" if blocked else "PASS" if not failures else "FAIL"
    payload = {
        "schema": "v023-paired-baseline-field-compare-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "fresh paired wrfout field comparison; no stored digest is read or trusted",
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "mode": args.mode,
        "autotune_floor": args.autotune_floor,
        "domains": args.domains,
        "verdict": verdict,
        "pass": verdict == "PASS",
        "domain_results": domain_results,
        "failures": failures,
        "blocked": [{"domain": row["domain"], "failures": row.get("failures", [])} for row in blocked],
    }
    _write_json(args.out_json, payload)
    return payload, 0 if verdict == "PASS" else 2 if verdict == "BLOCKED" else 1


def _write_tiny_wrfout(path: Path, *, perturb: float = 0.0, t2_all_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 3)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamp = np.array(list("2026-01-01_01:00:00"), dtype="S1")
        times[0, :] = stamp
        t2 = ds.createVariable("T2", "f8", ("Time", "south_north", "west_east"))
        u10 = ds.createVariable("U10", "f8", ("Time", "south_north", "west_east"))
        base = np.arange(6, dtype=np.float64).reshape(1, 2, 3)
        t2[:] = np.full_like(base, np.nan) if t2_all_nan else base + 273.15 + perturb
        u10[:] = base * 0.1


def run_self_test(out_json: Path) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="v023_field_compare_"))
    try:
        base = tmp / "base"
        same = tmp / "same"
        diff = tmp / "diff"
        all_nan = tmp / "all_nan"
        for root, perturb in ((base, 0.0), (same, 0.0), (diff, 0.5)):
            _write_tiny_wrfout(root / "wrfout_d01_2026-01-01_01:00:00", perturb=perturb)
        _write_tiny_wrfout(all_nan / "wrfout_d01_2026-01-01_01:00:00", t2_all_nan=True)
        positive_args = parse_args(
            [
                "--baseline-dir",
                str(base),
                "--candidate-dir",
                str(same),
                "--domains",
                "d01",
                "--mode",
                "exact",
                "--out-json",
                str(tmp / "positive.json"),
                "--no-spatial-splits",
            ]
        )
        negative_args = parse_args(
            [
                "--baseline-dir",
                str(base),
                "--candidate-dir",
                str(diff),
                "--domains",
                "d01",
                "--mode",
                "exact",
                "--out-json",
                str(tmp / "negative.json"),
                "--no-spatial-splits",
            ]
        )
        all_nan_args = parse_args(
            [
                "--baseline-dir",
                str(base),
                "--candidate-dir",
                str(all_nan),
                "--domains",
                "d01",
                "--mode",
                "exact",
                "--out-json",
                str(tmp / "all_nan.json"),
                "--no-spatial-splits",
            ]
        )
        positive, pos_rc = run_compare(positive_args)
        negative, neg_rc = run_compare(negative_args)
        all_nan_payload, all_nan_rc = run_compare(all_nan_args)
        passed = pos_rc == 0 and neg_rc != 0 and all_nan_rc != 0
        payload = {
            "schema": "v023-paired-baseline-field-compare-self-test-v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "PASS" if passed else "FAIL",
            "positive_control": {"expected": "PASS", "rc": pos_rc, "verdict": positive.get("verdict")},
            "negative_control": {"expected": "FAIL", "rc": neg_rc, "verdict": negative.get("verdict")},
            "all_nan_control": {
                "expected": "FAIL",
                "rc": all_nan_rc,
                "verdict": all_nan_payload.get("verdict"),
                "failure_kinds": [
                    failure.get("kind")
                    for row in all_nan_payload.get("domain_results", [])
                    for failure in row.get("failures", [])
                ],
            },
            "tmp_dir": str(tmp),
            "note": "CPU-only tiny NetCDF wrfout fixtures; no JAX, CUDA, or model execution.",
        }
        _write_json(out_json, payload)
        return 0 if passed else 1
    finally:
        if os.environ.get("V023_KEEP_SELFTEST_TMP", "0") != "1":
            shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, help="Fresh baseline wrfout directory.")
    parser.add_argument("--candidate-dir", type=Path, help="Fresh candidate wrfout directory.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--domains", nargs="+", default=["d01"])
    parser.add_argument("--mode", choices=["exact", "autotune-floor", "tolerance"], default="autotune-floor")
    parser.add_argument("--autotune-floor", type=float, default=1.0e-8)
    parser.add_argument("--tolerance-json", type=Path, default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--min-lead", type=int, default=None)
    parser.add_argument("--max-lead", type=int, default=None)
    parser.add_argument("--vars", nargs="+", default=None)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--no-spatial-splits", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only positive/negative controls on tiny local wrfout files.")
    args = parser.parse_args(argv)
    if not args.self_test and (args.baseline_dir is None or args.candidate_dir is None):
        parser.error("--baseline-dir and --candidate-dir are required unless --self-test is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args.out_json)
    _payload, rc = run_compare(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
