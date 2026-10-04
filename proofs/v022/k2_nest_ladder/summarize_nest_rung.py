#!/usr/bin/env python3
"""Summarize one K2 3-domain Canary nest-ladder rung.

This is proof infrastructure only. It reads the nested CLI payload plus wrfout
files and records per-domain Courant, finite/bounds, conservation proxies, and
operational tolerance against N1.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
K2_SINGLE = HERE.parent / "k2_dt_ladder"
sys.path.insert(0, str(K2_SINGLE))

from compare_to_r1 import compare as compare_fields  # noqa: E402
from courant_diag import courant_for_file  # noqa: E402


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _failure_from_log(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        if "NonFiniteStateError" in line or "gpuwrf: error" in line:
            return line.strip()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _dt_from_wrfout(path: Path, fallback: float | None) -> float:
    try:
        with Dataset(str(path)) as ds:
            return float(getattr(ds, "DT"))
    except AttributeError:
        pass
    except Exception:
        if fallback is None:
            raise
        return float(fallback)
    if fallback is None:
        raise AttributeError(f"{path} has no DT global attribute and no fallback dt was supplied")
    return float(fallback)


def _domain_files(out_dir: Path, dom: str) -> list[Path]:
    return sorted(out_dir.glob(f"wrfout_{dom}_*"))


def _finite_bounded(path: Path) -> dict[str, Any]:
    bounds = {
        "U": 250.0,
        "V": 250.0,
        "W": 150.0,
        "T": 1200.0,
        "QVAPOR": 0.2,
    }
    out: dict[str, Any] = {"file": str(path), "all_finite": True, "violations": [], "max_abs": {}}
    with Dataset(str(path)) as ds:
        for name in ("U", "V", "W", "T", "QVAPOR", "PH", "MU", "PSFC"):
            if name not in ds.variables:
                continue
            arr = np.asarray(ds.variables[name][:], dtype=np.float64)
            finite = bool(np.all(np.isfinite(arr)))
            if arr.size:
                out["max_abs"][name] = float(np.nanmax(np.abs(arr)))
            if not finite:
                out["all_finite"] = False
                out["violations"].append(f"{name}: non-finite")
                continue
            if name in bounds:
                amax = float(np.nanmax(np.abs(arr)))
                if amax > bounds[name]:
                    out["violations"].append(f"{name}: |max|={amax:.6g} > bound {bounds[name]:.6g}")
        if "QVAPOR" in ds.variables:
            qv = np.asarray(ds.variables["QVAPOR"][:], dtype=np.float64)
            if qv.size and float(np.nanmin(qv)) < -1.0e-9:
                out["violations"].append(f"QVAPOR: min={float(np.nanmin(qv)):.6g} < 0")
    out["bounded"] = len(out["violations"]) == 0
    return out


def _drift(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"frames": 0}
    first = values[0]
    rel = [((v - first) / first) if first != 0 else 0.0 for v in values]
    return {
        "frames": len(values),
        "first": first,
        "last": values[-1],
        "rel_drift_last": rel[-1],
        "rel_drift_max_abs": max(abs(v) for v in rel),
    }


def _conservation_proxies(paths: list[Path]) -> dict[str, Any]:
    dry_mass: list[float] = []
    theta_mass_proxy: list[float] = []
    water_mass_proxy: list[float] = []
    for path in paths:
        with Dataset(str(path)) as ds:
            if "MU" not in ds.variables or "MUB" not in ds.variables:
                continue
            mu = np.asarray(ds.variables["MU"][:], dtype=np.float64)
            mub = np.asarray(ds.variables["MUB"][:], dtype=np.float64)
            col = mu + mub
            for t in range(col.shape[0]):
                dry_mass.append(float(np.nansum(col[t])))
                if "T" in ds.variables:
                    theta = np.asarray(ds.variables["T"][t], dtype=np.float64) + 300.0
                    weights = col[t][None, :, :] / max(1, theta.shape[0])
                    theta_mass_proxy.append(float(np.nansum(theta * weights)))
                if "QVAPOR" in ds.variables:
                    qv = np.asarray(ds.variables["QVAPOR"][t], dtype=np.float64)
                    weights = col[t][None, :, :] / max(1, qv.shape[0])
                    water_mass_proxy.append(float(np.nansum(qv * weights)))
    return {
        "note": "Open-boundary run: drift is a runaway detector, not exact closed-domain conservation.",
        "dry_mass": _drift(dry_mass),
        "theta_mass_energy_proxy": _drift(theta_mass_proxy),
        "water_mass_proxy": _drift(water_mass_proxy),
    }


def _worst_courant(files: list[Path], fallback_dt: float | None) -> dict[str, Any]:
    per_file = []
    worst = {"Cx": 0.0, "Cy": 0.0, "Cz": 0.0, "C_total": 0.0}
    for path in files:
        dt_s = _dt_from_wrfout(path, fallback_dt)
        cur = courant_for_file(path, dt_s)
        per_file.append({
            "file": str(path),
            "dt_s": dt_s,
            "worst": cur["worst"],
            "dx_m": cur["dx_m"],
            "dy_m": cur["dy_m"],
        })
        for key in worst:
            worst[key] = max(worst[key], float(cur["worst"][key]))
    return {"worst": worst, "per_file": per_file}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root-dt", type=float, required=True)
    ap.add_argument("--n-sound", type=int, required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline-dir")
    ap.add_argument("--hours", type=int, required=True)
    ap.add_argument("--maxdom", type=int, default=3)
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "out"
    proof_payload = _load_json(run_dir / "proof" / "nested_pipeline_run.json")
    failure_summary = _failure_from_log(Path(args.log)) if args.log else None

    result: dict[str, Any] = {
        "schema": "v022_k2_nest_ladder_rung",
        "tag": args.tag,
        "root_dt_s_requested": args.root_dt,
        "n_sound": args.n_sound,
        "hours": args.hours,
        "maxdom": args.maxdom,
        "run_dir": str(run_dir),
        "cli_rc": args.rc,
        "failure_summary": failure_summary,
        "payload_present": proof_payload is not None,
        "domains": {},
    }

    if proof_payload is not None:
        result["payload_verdict"] = proof_payload.get("verdict")
        result["wall_clock_total_s"] = proof_payload.get("wall_clock_total_s")
        result["wall_clock_forecast_only_s"] = proof_payload.get("wall_clock_forecast_only_s")
        result["root_steps"] = proof_payload.get("root_steps")
        result["all_domains_finite"] = proof_payload.get("all_domains_finite")
        result["all_outputs_present"] = proof_payload.get("all_outputs_present")
        if args.hours > 0 and proof_payload.get("wall_clock_forecast_only_s") is not None:
            result["cold_inclusive_s_per_fc_h"] = float(proof_payload["wall_clock_forecast_only_s"]) / float(args.hours)
        result["per_domain_payload"] = proof_payload.get("per_domain")

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    all_finite_bounded = True
    all_within_band = True
    all_cfl_horiz_safe = True
    all_cfl_total_safe = True
    worst_c_total = 0.0
    worst_cz = 0.0

    for idx in range(1, args.maxdom + 1):
        dom = f"d{idx:02d}"
        files = _domain_files(out_dir, dom)
        entry: dict[str, Any] = {"wrfout_files": [str(p) for p in files]}
        fallback_dt = args.root_dt / (3.0 ** (idx - 1))
        if proof_payload and isinstance(proof_payload.get("per_domain"), dict):
            dmeta = proof_payload["per_domain"].get(dom) or {}
            fallback_dt = dmeta.get("dt_s") or fallback_dt
            entry["payload"] = dmeta
        if files:
            final = files[-1]
            entry["final_wrfout"] = str(final)
            entry["dt_s"] = _dt_from_wrfout(final, fallback_dt)
            entry["courant"] = _worst_courant(files, fallback_dt)
            entry["finite_bounded"] = _finite_bounded(final)
            entry["conservation"] = _conservation_proxies(files)
            worst = entry["courant"]["worst"]
            worst_c_total = max(worst_c_total, float(worst["C_total"]))
            worst_cz = max(worst_cz, float(worst["Cz"]))
            all_cfl_horiz_safe = all_cfl_horiz_safe and (float(worst["Cx"]) + float(worst["Cy"]) <= 1.0)
            all_cfl_total_safe = all_cfl_total_safe and (float(worst["C_total"]) <= 1.6)
            all_finite_bounded = all_finite_bounded and bool(entry["finite_bounded"]["bounded"])
            if baseline_dir is not None:
                bfiles = _domain_files(baseline_dir / "out", dom)
                if bfiles:
                    cmp = compare_fields(str(final), str(bfiles[-1]))
                    entry["compare_to_N1"] = cmp
                    all_within_band = all_within_band and bool(cmp["within_operational_band"])
        result["domains"][dom] = entry

    result["worst_C_total"] = worst_c_total
    result["worst_Cz"] = worst_cz
    result["cz_gt_1"] = bool(worst_cz > 1.0)
    result["gates"] = {
        "run_ok": bool(args.rc == 0 and proof_payload and proof_payload.get("verdict") == "PIPELINE_GREEN"),
        "finite_bounded": bool(all_finite_bounded),
        "cfl_horiz_target_le_1": bool(all_cfl_horiz_safe),
        "cfl_total_target_le_1p6": bool(all_cfl_total_safe),
        "within_operational_band_vs_N1": bool(all_within_band) if baseline_dir else None,
    }
    hard_gates = [
        result["gates"]["run_ok"],
        result["gates"]["finite_bounded"],
    ]
    if baseline_dir:
        hard_gates.append(result["gates"]["within_operational_band_vs_N1"])
    result["passes_all"] = all(bool(v) for v in hard_gates)
    result["verdict"] = "PASS" if result["passes_all"] else "FAIL"

    Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["gates"]["run_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
