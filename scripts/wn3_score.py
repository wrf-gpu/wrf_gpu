"""Score the full WN3 CPU field set, enforcing 25 exact timestamps and D6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = ROOT / "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json"
# Alisios twin thresholds, FROZEN 2026-10-02T16:12Z (read-only; verdict by alisios_mgr with pinned R32-01 code).
TWIN_THRESHOLDS = Path("<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_WN3_TWIN_THRESHOLDS_v1.json")


def twin_thresholds_ref():
    """Path + sha256 of the frozen twin thresholds, recorded in every gate summary."""
    import hashlib
    try:
        return {"path": str(TWIN_THRESHOLDS), "sha256": hashlib.sha256(TWIN_THRESHOLDS.read_bytes()).hexdigest()}
    except OSError as exc:
        return {"path": str(TWIN_THRESHOLDS), "error": repr(exc)}


# alisios verdict 2026-10-03 13:55Z: their frozen code needs these WRF globals; D6 cannot see constant/zero GPU fields.
REQUIRED_GLOBAL_ATTRS = ("PARENT_ID", "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO", "BUCKET_MM", "BUCKET_J",
                         "DT", "MMINLU", "NUM_LAND_CAT")


# CPU-WRF variables the GPU may omit, each with a documented reason (non-WRF or optional output streams only).
# Empty: every variable of the CPU-WRF history file is required in the GPU file.
OPTIONAL_CPU_VARIABLES: dict[str, str] = {}
# CPU-WRF global attributes the GPU may omit, each with a documented reason. Empty: the full CPU header is required.
OPTIONAL_CPU_GLOBALS: dict[str, str] = {}


def output_integrity(cpu, gpu):
    """What the frozen D6 manifest cannot see, on paired CPU/GPU history files (same order): CPU-WRF variables missing
    from the GPU file (any frame), a GPU field that is constant (e.g. all 0) where CPU varies (first, middle and last
    frame), and WRF global attributes (every paired frame: the full CPU header and alisios' required minimum).
    Any of these, or an unequal/empty frame pairing, fails."""
    missing_vars: dict[str, list[str]] = {}
    optional_missing: set[str] = set()
    globals_by_frame: dict[str, dict] = {}
    for c_path, g_path in zip(cpu, gpu):
        with Dataset(c_path) as c, Dataset(g_path) as g:
            absent = set(c.variables) - set(g.variables)
            cpu_attrs, gpu_attrs = set(c.ncattrs()), set(g.ncattrs())
        optional_missing |= absent & set(OPTIONAL_CPU_VARIABLES)
        for name in sorted(absent - set(OPTIONAL_CPU_VARIABLES)):
            missing_vars.setdefault(name, []).append(Path(g_path).name)
        missing_cpu = sorted(cpu_attrs - gpu_attrs - set(OPTIONAL_CPU_GLOBALS))
        missing_req = [a for a in REQUIRED_GLOBAL_ATTRS if a not in gpu_attrs]
        if missing_cpu or missing_req:
            globals_by_frame[Path(g_path).name] = {"missing_vs_cpu": missing_cpu, "missing_required": missing_req}
    paired = len(cpu) == len(gpu) > 0
    picks = sorted({0, len(gpu) // 2, len(gpu) - 1}) if paired else []
    degenerate = {}
    for i in picks:
        with Dataset(cpu[i]) as c, Dataset(gpu[i]) as g:
            for name in sorted(set(c.variables) & set(g.variables)):
                if name == "Times" or not np.issubdtype(c[name].dtype, np.number) or c[name].shape != g[name].shape:
                    continue
                cv = np.ma.asarray(c[name][:], dtype=np.float64).filled(np.nan)
                gv = np.ma.asarray(g[name][:], dtype=np.float64).filled(np.nan)
                if not np.isfinite(cv).any() or not np.nanmax(cv) > np.nanmin(cv):
                    continue  # CPU constant (or empty): nothing to compare against
                if np.isfinite(gv).any() and np.nanmax(gv) > np.nanmin(gv):
                    continue
                degenerate.setdefault(name, []).append({"file": Path(gpu[i]).name,
                    "gpu_value": float(np.nanmin(gv)) if np.isfinite(gv).any() else None,
                    "cpu_range": [float(np.nanmin(cv)), float(np.nanmax(cv))]})
    counts = {}
    if paired:
        with Dataset(cpu[0]) as c, Dataset(gpu[0]) as g:
            counts = {"cpu_count": len(c.ncattrs()), "gpu_count": len(g.ncattrs())}
    union_cpu = sorted({a for v in globals_by_frame.values() for a in v["missing_vs_cpu"]})
    union_req = [a for a in REQUIRED_GLOBAL_ATTRS if any(a in v["missing_required"] for v in globals_by_frame.values())]
    return {"frames_checked": [Path(gpu[i]).name for i in picks], "frames_paired": len(gpu),
            "pairing_ok": paired, "cpu_frames": len(cpu),
            "missing_variables_vs_cpu": {k: (v if len(v) < len(gpu) else "all frames") for k, v in missing_vars.items()},
            "optional_missing_allowed": {k: OPTIONAL_CPU_VARIABLES[k] for k in sorted(optional_missing)},
            "degenerate_fields": degenerate,
            "global_attrs": {**counts, "frames_checked": len(gpu) if paired else 0, "missing_vs_cpu": union_cpu,
                             "missing_required": union_req, "by_frame": globals_by_frame,
                             "optional_missing_allowed": dict(OPTIONAL_CPU_GLOBALS)},
            "pass": paired and not missing_vars and not degenerate and not globals_by_frame}


def report_gate_failures(score, expected_frames=25):
    """Enforce complete, finite common fields and each frozen frame limit."""
    failures = []
    inventory = score["inventory"]
    for name in inventory["cpu_only"]:
        failures.append({"kind": "missing_cpu_field", "field": name})
    for item in inventory["incompatible_common"]:
        failures.append({"kind": "incompatible_common", "detail": item})
    non_numeric = {item["field"] for item in inventory["non_numeric_common"]}
    fields = score["field_summaries"]
    for name in sorted(set(inventory["common"]) - non_numeric):
        field = fields.get(name)
        if field is None:
            failures.append({"kind": "unscored_numeric_field", "field": name})
            continue
        # The comparator gives string Times a special equality report, while
        # numeric XTIME keeps ordinary statistics despite the same class label.
        if name == "Times":
            if field["compared_lead_count"] != expected_frames or not field["all_equal"]:
                failures.append({"kind": "time_metadata", "field": name,
                    "compared_frames": field["compared_lead_count"], "checks": field["checks"]})
            continue
        if field["compared_lead_count"] != expected_frames or field["missing_leads"] or field["incompatible_leads"]:
            failures.append({"kind": "incomplete_field", "field": name,
                "compared_frames": field["compared_lead_count"],
                "missing": field["missing_leads"], "incompatible": field["incompatible_leads"]})
        for frame in field["by_lead"]:
            if frame["finite_gpu"] != frame["n"]:
                failures.append({"kind": "nonfinite_gpu", "field": name,
                    "lead_h": frame["lead_h"], "count": frame["n"] - frame["finite_gpu"]})
            if frame["tolerance_result"]["pass"] is False:
                failures.append({"kind": "frame_tolerance", "field": name,
                    "lead_h": frame["lead_h"], "detail": frame["tolerance_result"]["failures"]})
    return failures


def early_hour_diagnostics(cpu, gpu):
    """G2b max/median absolute differences over actual elapsed hours0–3.

    No equivalence verdict here: the frozen Alisios twin thresholds (v1) are applied by the
    pinned R32-01 compare code; alisios_mgr sets the verdict.
    """
    rows = {}
    for cfile, gfile in zip(cpu[:4], gpu[:4], strict=True):
        with Dataset(cfile) as c, Dataset(gfile) as g:
            for name in sorted(set(c.variables) & set(g.variables)):
                if not np.issubdtype(c[name].dtype, np.number) or not np.issubdtype(g[name].dtype, np.number):
                    continue
                a = np.ma.asarray(c[name][:], dtype=np.float64).filled(np.nan)
                b = np.ma.asarray(g[name][:], dtype=np.float64).filled(np.nan)
                if a.shape != b.shape:
                    continue  # G2a incompatibility is already reported by the comparator.
                finite = np.isfinite(a) & np.isfinite(b)
                diff = np.abs(a[finite] - b[finite])
                rows.setdefault(name, []).append({"file": cfile.name,
                    "finite_pairs": int(diff.size), "total_values": int(a.size),
                    "median_abs": float(np.median(diff)) if diff.size else None,
                    "max_abs": float(diff.max()) if diff.size else None})
    return {"scope": "G2b diagnosis only, tau=model elapsed hours0–3 (WN3f006–f009)",
            "twin_thresholds": twin_thresholds_ref(),
            "equivalence_verdict": "by alisios_mgr (R32-01 v2 pinned code)", "fields": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=int, default=24, help="24 = release gate; shorter only for probes.")
    args = parser.parse_args()
    frames = args.hours + 1
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(TOLERANCE.read_text())
    mandatory = set(manifest["score_policy"]["mandatory_presence_fields_for_final_runs"])
    result = {"manifest": str(TOLERANCE), "pass_means": "frozen D6 manifest; output_integrity_pass is separate (degenerate fields + WRF globals)", "all_common_variables_scored": True, "all_cpu_variables_required": True, "domains": {}, "pass": True,
        "gate_scope": "D6 full24h release milestone only" if args.hours == 24 else f"{args.hours} h probe, NOT the release gate",
        "twin_thresholds": twin_thresholds_ref(),
        "training_equivalence": "separate twin gate: frozen thresholds v1 via pinned R32-01 code, verdict by alisios_mgr; "
                                "binding twins 0227/0502 (0614 validation only); 24 h provisional, 72 h decisive"}
    jobs = {}
    for domain in ("d01", "d02", "d03"):
        cpu = sorted(args.cpu_dir.glob(f"wrfout_{domain}_*"))[:frames]
        gpu = sorted(args.gpu_dir.glob(f"wrfout_{domain}_*"))
        if len(cpu) != frames or {p.name for p in cpu} != {p.name for p in gpu}:
            raise ValueError(f"{domain}: require all {frames} exact CPU filenames, got CPU={len(cpu)}, GPU={len(gpu)}")
        missing = {}
        output_nonfinite = []
        for file in gpu:
            with Dataset(file) as ds:
                absent = sorted(mandatory - set(ds.variables))
                # GPU-only fields have no paired comparator statistics.
                with Dataset(args.cpu_dir / file.name) as cpu_ds:
                    absent = sorted(set(absent) | (set(cpu_ds.variables) - set(ds.variables)))
                    gpu_only = set(ds.variables) - set(cpu_ds.variables)
                for name in sorted(gpu_only):
                    if np.issubdtype(ds[name].dtype, np.number):
                        values = np.ma.asarray(ds[name][:], dtype=np.float64).filled(np.nan)
                        count = int(values.size - np.count_nonzero(np.isfinite(values)))
                        if count:
                            output_nonfinite.append({"file": file.name, "field": name, "count": count})
            if absent:
                missing[file.name] = absent
        # Issue time is18z but actual run begins atf006=00z. Pass the FIRST
        # filename explicitly; inferring from the case path drops the last6h.
        init = cpu[0].name[11:].replace("_", "T") + "Z"
        # Frozen comparator, one netCDF handle per file (wn3_fast_compare: same JSON, ~5x faster).
        command = [sys.executable, str(ROOT / "scripts/wn3_fast_compare.py"),
            "--cpu-dir", str(args.cpu_dir), "--gpu-dir", str(args.gpu_dir), "--domain", domain,
            "--init", init, "--min-lead", "0", "--max-lead", str(args.hours),
            "--tolerance-json", str(TOLERANCE), "--progress", "25",
            "--out-json", str(args.out / f"{domain}.json"), "--out-md", str(args.out / f"{domain}.md")]
        log = (args.out / f"{domain}.log").open("w")
        jobs[domain] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log, cpu, gpu, missing, output_nonfinite)
    for domain, (proc, log, cpu, gpu, missing, output_nonfinite) in jobs.items():
        try:
            rc = proc.wait(timeout=3600)
        finally:
            log.close()
        if rc:
            raise subprocess.CalledProcessError(rc, proc.args)
        score = json.loads((args.out / f"{domain}.json").read_text())
        policy_failures = report_gate_failures(score, frames)
        (args.out / f"{domain}_early_hours.json").write_text(
            json.dumps(early_hour_diagnostics(cpu, gpu), indent=2) + "\n")
        ok = (not missing and not policy_failures and not output_nonfinite
              and score["pairing"]["paired_file_count"] == frames
              and score["summaries"]["verdict"] == "PASS")
        integrity = output_integrity(cpu, gpu)
        result["domains"][domain] = {"pass": ok, "missing_required_fields": missing,
            "policy_failures": policy_failures, "gpu_only_nonfinite": output_nonfinite,
            "pairing": score["pairing"], "summaries": score["summaries"], "output_integrity": integrity}
        result["pass"] &= ok
        result["output_integrity_pass"] = result.get("output_integrity_pass", True) and integrity["pass"]
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "domains": {d: v["pass"] for d, v in result["domains"].items()},
                      "output_integrity_pass": result["output_integrity_pass"],
                      "missing_vs_cpu": {d: sorted(v["output_integrity"]["missing_variables_vs_cpu"]) for d, v in result["domains"].items()},
                      "degenerate": {d: sorted(v["output_integrity"]["degenerate_fields"]) for d, v in result["domains"].items()},
                      "global_frames_failing": {d: sorted(v["output_integrity"]["global_attrs"]["by_frame"]) for d, v in result["domains"].items()}}))
    if not result["pass"] or not result["output_integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
