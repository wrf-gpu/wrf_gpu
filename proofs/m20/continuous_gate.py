#!/usr/bin/env python3
"""Continuous per-lead VALIDATION GATE (v0.2.0) — GPU vs corpus CPU-WRF.

WHAT THIS IS  (and is NOT)
--------------------------
A **standing regression + skill gate** that re-scores whatever GPU forecast
wrfout artifacts already exist on disk against the corpus CPU-WRF history, on a
PER-LEAD-HOUR basis, after EVERY physics/dycore merge. It is the cheap, reusable
"did this merge move T2/U10/V10 (and the surface-flux diagnostics) relative to
CPU-WRF?" tripwire that runs in seconds with no GPU.

It is explicitly **NOT** the formal TOST equivalence verdict. The descriptive
frozen-margin checks here (e.g. |bias_T2| <= 0.2149 K) are a *regression/skill
gate* — a 1-D sanity envelope, NOT a powered two-one-sided-tests equivalence
claim. The formal seasonal TOST (ADR-029, n>=15) stays in
proofs/m20/tost_ensemble_runner.py and requires the powered corpus + a stats
reviewer. This gate makes NO equivalence claim at small n.

WHY PER-LEAD GRIDPOINT-PAIRED (vs station-block TOST)
-----------------------------------------------------
The TOST scorer (paired_tost_scorer.py) interpolates to AEMET stations and bins
into 24h lead BLOCKS vs OBSERVATIONS. That answers "is GPU as skillful as CPU vs
reality, over a season?" — the publishable question. For a *regression* gate we
want the opposite: maximum sensitivity to "did the merge change the model?",
isolated from obs sampling noise. So here the reference is **CPU-WRF itself**, the
pairing is **every grid cell** (the GPU emit grid is bit-identical to the corpus
d02 grid — verified XLAT/XLONG max-abs-diff == 0.0), and the resolution is
**every individual lead hour**. Bias = mean(GPU - CPU); RMSE = sqrt(mean((GPU-CPU)^2)).

FIELDS
------
  core (always present in GPU artifacts):  T2, U10, V10
  diagnostics (scored WHERE PRESENT in both GPU + CPU frame; targeted
  regression catch for the surface-flux / radiation / PBL fixes landing in
  v0.2.0):  Q2, PSFC, PBLH, SWDOWN, GLW, HFX, LH, GRDFLX, W

  The current on-disk GPU artifacts (the v0.1.0 / S6b TOST emit) only carry
  T2/U10/V10/XLAT/XLONG, so the diagnostics report status
  "not_in_gpu_artifact" today. The moment a future emit writes them, this gate
  scores them automatically — no change here. (W is a 3-D bottom_top_stag field;
  when present it is reduced to the lowest model level k=0 as a representative
  surface-coupled vertical-velocity diagnostic.)

RE-RUN (the standing gate, after each merge)
--------------------------------------------
  PYTHONPATH=src taskset -c 0-3 python3 proofs/m20/continuous_gate.py \
      --gpu-root proofs/m20/tost_run/gpu_wrfout \
      --manifest proofs/v010_validation/v010_cases_manifest.json \
      --out proofs/m20/continuous_gate_$(date +%Y_%m_%d).json

It auto-discovers every <gpu-root>/<case>_<level>/ artifact dir, maps it to the
manifest CPU run_dir + init time, and scores whatever leads/fields are present in
BOTH sides. NON-GPU: reads NetCDF only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]

# Core fields the GPU emit always carries.
CORE_FIELDS = ("T2", "U10", "V10")
# Targeted-regression diagnostics: scored only where present in BOTH frames.
DIAG_FIELDS = ("Q2", "PSFC", "PBLH", "SWDOWN", "GLW", "HFX", "LH", "GRDFLX", "W")
ALL_FIELDS = CORE_FIELDS + DIAG_FIELDS

# Frozen DESCRIPTIVE regression/skill envelope (NOT a TOST equivalence margin).
# T2 reuses the ADR-029 predeclared 10%-of-CPU-RMSE margin so the gate tracks the
# SAME number the formal TOST will eventually test; U10/V10 likewise. The others
# are coarse descriptive sanity envelopes for catching gross regressions only.
REGRESSION_MARGINS = {
    "T2": 0.2148692978020805,    # K  (ADR-029)
    "U10": 0.23064713972582307,  # m/s (ADR-029)
    "V10": 0.2752320537920854,   # m/s (ADR-029)
    "Q2": 0.0005,                # kg/kg  (descriptive)
    "PSFC": 50.0,                # Pa     (descriptive)
    "PBLH": 100.0,               # m      (descriptive)
    "SWDOWN": 30.0,              # W/m^2  (descriptive)
    "GLW": 15.0,                 # W/m^2  (descriptive)
    "HFX": 25.0,                 # W/m^2  (descriptive)
    "LH": 25.0,                  # W/m^2  (descriptive)
    "GRDFLX": 20.0,              # W/m^2  (descriptive)
    "W": 0.05,                   # m/s    (descriptive)
}

WRFOUT_RE = re.compile(r"^wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")
DEFAULT_GPU_ROOT = ROOT / "proofs/m20/tost_run/gpu_wrfout"
DEFAULT_MANIFEST = ROOT / "proofs/v010_validation/v010_cases_manifest.json"

DEP_FILES = (
    "proofs/m20/continuous_gate.py",
    "proofs/m20/paired_tost_scorer.py",
    "proofs/m20/tost_ensemble_runner.py",
    "proofs/m20/case_manifest.json",
    "src/gpuwrf/runtime/operational_mode.py",
)


# ---------------------------------------------------------------------------
# Provenance helpers.
# ---------------------------------------------------------------------------
def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "UNKNOWN"


def dependency_shas() -> dict:
    out = {"branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
           "head_sha": _git(["rev-parse", "HEAD"]),
           "describe": _git(["describe", "--tags", "--always"])}
    files = {}
    for f in DEP_FILES:
        files[f] = _git(["log", "-1", "--format=%h", "--", f]) or "UNTRACKED/UNKNOWN"
    out["dependency_file_shas"] = files
    return out


# ---------------------------------------------------------------------------
# Frame discovery + field read.
# ---------------------------------------------------------------------------
def list_frames(run_dir: Path, domain: str) -> dict[datetime, Path]:
    """Map valid-time -> wrfout path for a domain in a run dir."""
    frames: dict[datetime, Path] = {}
    if not run_dir.is_dir():
        return frames
    for p in sorted(run_dir.glob(f"wrfout_{domain}_*")):
        m = WRFOUT_RE.match(p.name)
        if m and p.is_file():
            vt = datetime.strptime(m.group(2), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            frames[vt] = p
    return frames


def read_field(ds: Dataset, name: str) -> np.ndarray | None:
    """Read one field as a 2-D surface array (Time squeezed). 3-D W -> level k=0."""
    if name not in ds.variables:
        return None
    arr = np.asarray(ds.variables[name][:], dtype=np.float64)
    # Squeeze leading Time axis.
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    # 3-D staggered (e.g. W: bottom_top_stag, sn, we) -> lowest model level.
    if arr.ndim == 3:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# Per-lead, per-field gridpoint-paired bias / RMSE.
# ---------------------------------------------------------------------------
def score_field_pair(gpu: np.ndarray, cpu: np.ndarray) -> dict:
    """Complete-pair (finite both sides) bias + RMSE on a flattened grid."""
    if gpu.shape != cpu.shape:
        return {"status": "SHAPE_MISMATCH",
                "gpu_shape": list(gpu.shape), "cpu_shape": list(cpu.shape)}
    g = gpu.ravel()
    c = cpu.ravel()
    mask = np.isfinite(g) & np.isfinite(c)
    n = int(mask.sum())
    if n == 0:
        return {"status": "NO_COMPLETE_PAIRS", "n": 0}
    d = g[mask] - c[mask]
    return {
        "status": "OK",
        "n": n,
        "bias": float(np.mean(d)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "cpu_mean": float(np.mean(c[mask])),
        "gpu_mean": float(np.mean(g[mask])),
        "abs_bias_max_cell": float(np.max(np.abs(d))),
    }


def score_unit(unit_id: str, gpu_dir: Path, cpu_dir: Path,
               domain: str, init: datetime, max_lead: int) -> dict:
    """Score one (case, level): pair GPU vs CPU frame-by-frame, per lead, per field."""
    gpu_frames = list_frames(gpu_dir, domain)
    cpu_frames = list_frames(cpu_dir, domain)

    # Lead hours present in BOTH the GPU emit and the CPU corpus.
    shared = sorted(set(gpu_frames) & set(cpu_frames))
    leads_scored = []
    per_lead: dict[int, dict] = {}
    # field -> list of (lead, bias, rmse, n) for envelope aggregation
    field_series: dict[str, list[dict]] = {f: [] for f in ALL_FIELDS}

    for vt in shared:
        lead = int(round((vt - init).total_seconds() / 3600.0))
        if lead < 0 or lead > max_lead:
            continue
        gpath, cpath = gpu_frames[vt], cpu_frames[vt]
        with Dataset(gpath) as gd, Dataset(cpath) as cd:
            gvars = set(gd.variables.keys())
            lead_fields = {}
            for f in ALL_FIELDS:
                if f not in gvars:
                    # Field not emitted by GPU yet (the diagnostics, today).
                    lead_fields[f] = {"status": "not_in_gpu_artifact"}
                    continue
                if f not in cd.variables:
                    lead_fields[f] = {"status": "not_in_cpu_corpus"}
                    continue
                g = read_field(gd, f)
                c = read_field(cd, f)
                res = score_field_pair(g, c)
                lead_fields[f] = res
                if res.get("status") == "OK":
                    field_series[f].append(
                        {"lead": lead, "bias": res["bias"], "rmse": res["rmse"],
                         "n": res["n"]})
        per_lead[lead] = lead_fields
        leads_scored.append(lead)

    # Per-field envelope summary over the scored leads (descriptive gate).
    field_summary = {}
    for f in ALL_FIELDS:
        series = field_series[f]
        if not series:
            field_summary[f] = {"status": "not_scored",
                                "reason": "field absent from GPU artifact or no shared lead"}
            continue
        biases = np.array([s["bias"] for s in series])
        rmses = np.array([s["rmse"] for s in series])
        margin = REGRESSION_MARGINS.get(f)
        worst_abs_bias = float(np.max(np.abs(biases)))
        worst_rmse = float(np.max(rmses))
        passes = (worst_abs_bias <= margin) if margin is not None else None
        field_summary[f] = {
            "status": "scored",
            "n_leads": len(series),
            "n_pairs_per_lead_min": int(min(s["n"] for s in series)),
            "n_pairs_per_lead_max": int(max(s["n"] for s in series)),
            "mean_bias": float(np.mean(biases)),
            "worst_abs_bias": worst_abs_bias,
            "mean_rmse": float(np.mean(rmses)),
            "worst_rmse": worst_rmse,
            "regression_margin": margin,
            "within_margin": passes,
            "margin_basis": ("ADR-029 10%-of-CPU-RMSE" if f in CORE_FIELDS
                             else "descriptive sanity envelope"),
        }

    return {
        "unit_id": unit_id,
        "domain": domain,
        "init_utc": init.isoformat(),
        "max_lead_h": max_lead,
        "gpu_dir": str(gpu_dir),
        "cpu_run_dir": str(cpu_dir),
        "gpu_frames_found": len(gpu_frames),
        "cpu_frames_found": len(cpu_frames),
        "leads_scored": leads_scored,
        "n_leads_scored": len(leads_scored),
        "per_field_summary": field_summary,
        "per_lead": {str(k): per_lead[k] for k in sorted(per_lead)},
    }


# ---------------------------------------------------------------------------
# Manifest mapping: GPU artifact dir <unit_id> -> CPU run_dir + init.
# ---------------------------------------------------------------------------
def build_unit_index(manifest_path: Path) -> dict[str, dict]:
    """Map '<case_id>_<level>' -> {cpu_run_dir, domain, init, max_lead}."""
    man = json.loads(manifest_path.read_text())
    idx: dict[str, dict] = {}
    for c in man["cases"]:
        init = datetime.fromisoformat(c["init_utc"].replace("Z", "+00:00"))
        for level, spec in c["levels"].items():
            idx[f"{c['case_id']}_{level}"] = {
                "cpu_run_dir": Path(spec["run_dir"]),
                "domain": spec["domain"],
                "init": init,
                "max_lead": int(spec["max_lead_h"]),
            }
    return idx


def discover_units(gpu_root: Path) -> list[str]:
    if not gpu_root.is_dir():
        return []
    return sorted(d.name for d in gpu_root.iterdir() if d.is_dir())


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT,
                    help="dir of <case>_<level>/ GPU wrfout artifact subdirs")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="case manifest mapping unit_id -> CPU run_dir + init")
    ap.add_argument("--units", nargs="+", default=None,
                    help="subset of unit ids (default: all discovered under --gpu-root)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    unit_index = build_unit_index(a.manifest)
    discovered = discover_units(a.gpu_root)
    want = a.units if a.units else discovered

    units_out = []
    skipped = []
    for uid in want:
        gpu_dir = a.gpu_root / uid
        if not gpu_dir.is_dir():
            skipped.append({"unit_id": uid, "reason": "no GPU artifact dir"})
            continue
        meta = unit_index.get(uid)
        if meta is None:
            skipped.append({"unit_id": uid, "reason": "unit not in manifest"})
            continue
        if not meta["cpu_run_dir"].is_dir() or not list_frames(meta["cpu_run_dir"], meta["domain"]):
            skipped.append({"unit_id": uid,
                            "reason": "CPU corpus frames absent on disk",
                            "cpu_run_dir": str(meta["cpu_run_dir"])})
            continue
        units_out.append(score_unit(uid, gpu_dir, meta["cpu_run_dir"],
                                     meta["domain"], meta["init"], meta["max_lead"]))

    # Cross-unit per-field regression roll-up (descriptive).
    roll: dict[str, dict] = {}
    for f in ALL_FIELDS:
        scored = [u["per_field_summary"][f] for u in units_out
                  if u["per_field_summary"][f].get("status") == "scored"]
        if not scored:
            roll[f] = {"status": "not_scored"}
            continue
        worst_bias = max(s["worst_abs_bias"] for s in scored)
        worst_rmse = max(s["worst_rmse"] for s in scored)
        margin = REGRESSION_MARGINS.get(f)
        roll[f] = {
            "status": "scored",
            "n_units": len(scored),
            "worst_abs_bias_over_units": worst_bias,
            "worst_rmse_over_units": worst_rmse,
            "regression_margin": margin,
            "within_margin": (worst_bias <= margin) if margin is not None else None,
        }

    core_pass = all(roll[f].get("within_margin") is True for f in CORE_FIELDS
                    if roll[f].get("status") == "scored")
    core_scored = [f for f in CORE_FIELDS if roll[f].get("status") == "scored"]

    payload = {
        "schema": "ContinuousValidationGate",
        "schema_version": 1,
        "gate_kind": "regression/skill gate — NOT TOST equivalence",
        "claim_disclaimer": (
            "Per-lead gridpoint-paired GPU-vs-CPU-WRF bias/RMSE. The frozen "
            "margins are a DESCRIPTIVE regression/skill envelope, NOT a powered "
            "TOST equivalence test. No equivalence is claimed at this n. The "
            "formal seasonal TOST (ADR-029, n>=15) lives in "
            "proofs/m20/tost_ensemble_runner.py."),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dependency_shas(),
        "inputs": {
            "gpu_root": str(a.gpu_root),
            "manifest": str(a.manifest),
            "units_requested": want,
            "units_scored": [u["unit_id"] for u in units_out],
            "units_skipped": skipped,
        },
        "reference": "corpus CPU-WRF (truth); candidate = GPU forecast artifact",
        "pairing": "every grid cell, complete-pair (finite both sides), per lead hour",
        "core_fields": list(CORE_FIELDS),
        "diagnostic_fields": list(DIAG_FIELDS),
        "regression_margins": REGRESSION_MARGINS,
        "cross_unit_rollup": roll,
        "core_within_margin": bool(core_pass) if core_scored else None,
        "core_fields_scored": core_scored,
        "units": units_out,
        "rerun_command": (
            "PYTHONPATH=src taskset -c 0-3 python3 proofs/m20/continuous_gate.py "
            "--gpu-root proofs/m20/tost_run/gpu_wrfout "
            "--manifest proofs/v010_validation/v010_cases_manifest.json "
            "--out proofs/m20/continuous_gate_$(date +%Y_%m_%d).json"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    # Compact console summary.
    print(f"continuous gate -> {a.out}")
    print(f"  units scored: {[u['unit_id'] for u in units_out]}")
    if skipped:
        print(f"  units skipped: {[(s['unit_id'], s['reason']) for s in skipped]}")
    print("  CORE per-field (worst-over-units bias / rmse / margin / within):")
    for f in CORE_FIELDS:
        r = roll[f]
        if r.get("status") == "scored":
            print(f"    {f:7s} bias={r['worst_abs_bias_over_units']:+.4f} "
                  f"rmse={r['worst_rmse_over_units']:.4f} "
                  f"margin={r['regression_margin']:.4f} within={r['within_margin']}")
        else:
            print(f"    {f:7s} {r.get('status')}")
    print("  DIAGNOSTICS:")
    for f in DIAG_FIELDS:
        r = roll[f]
        print(f"    {f:7s} {r.get('status')}"
              + (f" bias={r['worst_abs_bias_over_units']:+.4f} rmse={r['worst_rmse_over_units']:.4f}"
                 if r.get("status") == "scored" else ""))
    print(f"  core_within_margin = {payload['core_within_margin']}")
    print("  [regression/skill gate — NOT TOST equivalence]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
