"""v0.10.0 Wave-B1 Thompson NSED=16 fidelity proof helper.

This script does not change model code.  It packages three checks needed for the
NSED_MAX 64 -> 16 default flip:

* precip: WRF precipitating Thompson oracle parity at cap=16, plus cap=16 vs 64
  direct equality on the same oracle.
* run24h: the Wave-A L2 d02 workload for one cap, saving compact final arrays.
* compare24h: cap=16 vs cap=64 final-array identity, skill no-regression, and
  unchanged water/precip budget from the saved arrays.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "proofs" / "v0100"
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"
STATE_COMPARE_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "rainnc", "rainc")
DIAG_COMPARE_FIELDS = ("t2", "u10", "v10", "psfc")
SKILL_FIELDS = ("T2", "U10", "V10")
SKILL_NO_REGRESSION_TOL = {"T2": 0.01, "U10": 0.01, "V10": 0.01}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _sha256(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(a.shape),
        "finite": bool(np.isfinite(a).all()),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "sum": float(np.nansum(a)),
        "sha256": _sha256(np.asarray(arr)),
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).ravel()
    finite = np.isfinite(diff)
    return float(np.sqrt(np.mean(diff[finite] ** 2))) if finite.any() else float("nan")


def _bias(a: np.ndarray, b: np.ndarray) -> float:
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).ravel()
    finite = np.isfinite(diff)
    return float(np.mean(diff[finite])) if finite.any() else float("nan")


def _load_wrf_surface(path: Path, fields: tuple[str, ...] = SKILL_FIELDS) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    out: dict[str, np.ndarray] = {}
    with Dataset(path, "r") as ds:
        for name in fields:
            var = ds.variables[name]
            data = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
            out[name] = np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)
    return out


def _precip_run_for_cap(cap: int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))
    import gpuwrf.physics.thompson_column as tc
    from precip_parity_p1_5 import _sed_clip_audit
    from precip_oracle_validate import run_scheme, wrf_reference_precip

    tc.NSED_MAX = int(cap)
    tc.NSED_SUBSTEPS = int(cap)
    wrf = wrf_reference_precip()
    scheme = run_scheme(f"faithful_explicit_nsed{cap}")
    precip = scheme["precip_mass"]
    ratio = float(precip["total_surface_precip_mm"] / wrf["wrf_total_rainncv_mm"])
    per_col_rel = [
        abs(float(j) - float(w)) / float(w)
        for j, w in zip(precip["surface_precip_mm_per_col"], wrf["wrf_rainncv_mm_per_col"])
    ]
    qr = scheme["per_field"]["qr"]
    surface_species = ("rain", "snow", "graupel", "ice")
    species_sum = float(sum(scheme["precip_by_species_mm"].get(name, 0.0) for name in surface_species))
    total = float(precip["total_surface_precip_mm"])
    gates = {
        "surface_precip_ratio": {"value": ratio, "tol": [0.97, 1.03], "pass": bool(0.97 <= ratio <= 1.03)},
        "per_column_max_rel": {"value": float(max(per_col_rel)), "tol": 0.03, "pass": bool(max(per_col_rel) <= 0.03)},
        "qr_field_parity": {
            "mean_rel": float(qr["mean_rel"]),
            "max_rel": float(qr["max_rel"]),
            "tol_mean": 0.01,
            "tol_max": 0.02,
            "pass": bool(qr["mean_rel"] < 0.01 and qr["max_rel"] < 0.02),
        },
        "water_closure_rel": {
            "value": float(precip["water_closure_max_rel_residual"]),
            "tol": 1.0e-5,
            "pass": bool(precip["water_closure_max_rel_residual"] < 1.0e-5),
        },
        "accumulator_additivity_residual_mm": {
            "value": float(abs(species_sum - total)),
            "tol": 1.0e-9,
            "pass": bool(abs(species_sum - total) < 1.0e-9),
        },
        "sed_clip_fallback": {
            "per_species": _sed_clip_audit(),
        },
    }
    total_clipped = int(sum(s["clipped_at_NSED_MAX"] for s in gates["sed_clip_fallback"]["per_species"].values()))
    gates["sed_clip_fallback"]["total_clipped"] = total_clipped
    gates["sed_clip_fallback"]["pass"] = bool(total_clipped == 0)
    return {
        "NSED_MAX": int(cap),
        "wrf_reference": wrf,
        "scheme": scheme,
        "surface_precip_rate_mm_h": float(total / 18.0 * 3600.0),
        "gates": gates,
        "all_pass": bool(all(g["pass"] for g in gates.values())),
    }


def precip_mode(args: argparse.Namespace) -> int:
    os.environ["GPUWRF_THOMPSON_NSED"] = "16"
    r16 = _precip_run_for_cap(16)
    r64 = _precip_run_for_cap(64)
    p16 = np.asarray(r16["scheme"]["precip_mass"]["surface_precip_mm_per_col"], dtype=np.float64)
    p64 = np.asarray(r64["scheme"]["precip_mass"]["surface_precip_mm_per_col"], dtype=np.float64)
    precip_equal = bool(np.array_equal(p16, p64))
    delta = p16 - p64
    water16 = float(r16["scheme"]["precip_mass"]["water_closure_max_rel_residual"])
    water64 = float(r64["scheme"]["precip_mass"]["water_closure_max_rel_residual"])
    cmp_record = {
        "surface_precip_bit_identical": precip_equal,
        "surface_precip_max_abs_delta_mm": float(np.max(np.abs(delta))),
        "surface_precip_total_delta_mm": float(np.sum(p16) - np.sum(p64)),
        "surface_precip_rate_delta_mm_h": float((np.sum(p16) - np.sum(p64)) / 18.0 * 3600.0),
        "water_closure_rel_delta": float(water16 - water64),
        "predeclared_tolerances": {
            "nsed16_vs_64_surface_precip_max_abs_delta_mm": 0.0,
            "nsed16_vs_64_surface_precip_rate_delta_mm_h": 0.0,
        },
        "pass": bool(precip_equal),
    }
    payload = {
        "schema": "V0100WaveB1NSED16PrecipOracle",
        "schema_version": 1,
        "status": "PASS" if (r16["all_pass"] and cmp_record["pass"]) else "FAIL",
        "oracle": "WRF mp_gt_driver single-column precipitating Thompson oracle",
        "dt_s": 18.0,
        "default_candidate": r16,
        "cap64_control": r64,
        "nsed16_vs_64": cmp_record,
    }
    _write_json(Path(args.out), payload)
    print(json.dumps({"status": payload["status"], "precip": cmp_record}, indent=2))
    return 0 if payload["status"] == "PASS" else 2


def _prepare_jax_env(nsed: int) -> None:
    os.environ["GPUWRF_THOMPSON_NSED"] = str(int(nsed))
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("PYTHONPATH", "src")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except OSError:
            pass


def run24h_mode(args: argparse.Namespace) -> int:
    _prepare_jax_env(int(args.nsed))
    import jax
    import jax.numpy as jnp

    from gpuwrf.config import paths
    from gpuwrf.integration.daily_pipeline import DailyPipelineConfig, _build_real_case
    from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision, _initial_carry_for_run, _m9_snapshot

    if not [d for d in jax.devices() if d.platform == "gpu"]:
        raise RuntimeError("No JAX GPU backend visible")
    hours = float(args.hours)
    segment_steps = int(args.segment_steps)
    cadence = int(args.cadence)
    cfg = DailyPipelineConfig(
        hours=int(max(1, round(hours))),
        dt_s=10.0,
        acoustic_substeps=10,
        run_id=args.run_id,
        run_root=paths.wrf_l2_root(),
        domain="d02",
        radiation_cadence_steps=cadence,
    )
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=False,
        radiation_cadence_steps=cadence,
        time_utc=case.run_start,
    )
    state0 = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    carry = _initial_carry_for_run(state0, nl)
    steps = int(round(hours * 3600.0 / float(nl.dt_s)))
    start_step = 1
    t0 = time.perf_counter()
    segment_walls: list[float] = []
    while start_step <= steps:
        n = min(segment_steps, steps - start_step + 1)
        seg_t0 = time.perf_counter()
        carry = _advance_chunk(carry, nl, jnp.asarray(start_step, dtype=jnp.int32), n_steps=int(n), cadence=cadence)
        jax.block_until_ready(carry.state.theta)
        segment_walls.append(time.perf_counter() - seg_t0)
        start_step += n
    forecast_wall_s = time.perf_counter() - t0
    diag = _m9_snapshot(carry, nl, jnp.asarray(float(steps) * float(nl.dt_s), dtype=jnp.float64))
    jax.block_until_ready(diag.t2)

    state = carry.state
    arrays = {
        "t2": np.asarray(jax.device_get(diag.t2)),
        "u10": np.asarray(jax.device_get(diag.u10)),
        "v10": np.asarray(jax.device_get(diag.v10)),
        "psfc": np.asarray(jax.device_get(diag.psfc)),
        "qv": np.asarray(jax.device_get(state.qv)),
        "qc": np.asarray(jax.device_get(state.qc)),
        "qr": np.asarray(jax.device_get(state.qr)),
        "qi": np.asarray(jax.device_get(state.qi)),
        "qs": np.asarray(jax.device_get(state.qs)),
        "qg": np.asarray(jax.device_get(state.qg)),
        "rainnc": np.asarray(jax.device_get(state.rain_acc + state.snow_acc + state.graupel_acc + state.ice_acc)),
        "rainc": np.asarray(jax.device_get(state.rainc_acc)),
    }
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrays)

    all_stats = {name: _stats(arr) for name, arr in arrays.items()}
    all_finite = bool(all(item["finite"] for item in all_stats.values()))
    final_valid_time = case.run_start + timedelta(seconds=float(steps) * float(nl.dt_s))
    cpu_ref = Path(run_dir) / f"wrfout_d02_{final_valid_time.strftime('%Y-%m-%d_%H:%M:%S')}"
    cpu_skill: dict[str, Any] = {
        "available": bool(cpu_ref.is_file()),
        "reference_file": str(cpu_ref),
        "valid_time_utc": final_valid_time.isoformat(),
        "fields": {},
    }
    if cpu_ref.is_file():
        ref = _load_wrf_surface(cpu_ref)
        for wrf_name, arr_name in (("T2", "t2"), ("U10", "u10"), ("V10", "v10")):
            cpu_skill["fields"][wrf_name] = {
                "rmse": _rmse(arrays[arr_name], ref[wrf_name]),
                "bias": _bias(arrays[arr_name], ref[wrf_name]),
            }

    payload = {
        "schema": "V0100WaveB1NSED24HRun",
        "schema_version": 1,
        "status": "PASS" if all_finite else "FAIL",
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "NSED_MAX": int(args.nsed),
        "hours": hours,
        "steps": steps,
        "segment_steps": segment_steps,
        "radiation_cadence_steps": cadence,
        "config": {
            "force_fp64": bool(nl.force_fp64),
            "disable_guards": bool(nl.disable_guards),
            "run_physics": bool(nl.run_physics),
            "run_boundary": bool(nl.run_boundary),
            "dt_s": float(nl.dt_s),
            "acoustic_substeps": int(nl.acoustic_substeps),
            "GPUWRF_THOMPSON_NSED": os.environ.get("GPUWRF_THOMPSON_NSED"),
        },
        "forecast_wall_s": float(forecast_wall_s),
        "segment_wall_s": [float(x) for x in segment_walls],
        "arrays_npz": str(out_npz),
        "array_stats": all_stats,
        "all_saved_arrays_finite": all_finite,
        "cpu_skill": cpu_skill,
    }
    _write_json(Path(args.out), payload)
    print(json.dumps({"status": payload["status"], "NSED_MAX": args.nsed, "forecast_wall_s": forecast_wall_s, "cpu_skill_available": cpu_skill["available"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 2


def compare24h_mode(args: argparse.Namespace) -> int:
    j16 = json.loads(Path(args.nsed16_json).read_text())
    j64 = json.loads(Path(args.nsed64_json).read_text())
    a16 = np.load(args.nsed16_npz)
    a64 = np.load(args.nsed64_npz)
    field_diffs: dict[str, Any] = {}
    for name in sorted(set(a16.files) & set(a64.files)):
        left = np.asarray(a16[name])
        right = np.asarray(a64[name])
        diff = left.astype(np.float64) - right.astype(np.float64)
        field_diffs[name] = {
            "bit_identical": bool(np.array_equal(left, right)),
            "max_abs": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff.ravel() ** 2))),
            "sum_delta": float(np.sum(left.astype(np.float64)) - np.sum(right.astype(np.float64))),
        }
    all_identity = bool(all(v["bit_identical"] for v in field_diffs.values()))

    skill_fields: dict[str, Any] = {}
    cpu16 = j16.get("cpu_skill", {})
    cpu64 = j64.get("cpu_skill", {})
    cpu_available = bool(cpu16.get("available") and cpu64.get("available"))
    if cpu_available:
        for name in SKILL_FIELDS:
            r16 = float(cpu16["fields"][name]["rmse"])
            r64 = float(cpu64["fields"][name]["rmse"])
            delta = r16 - r64
            skill_fields[name] = {
                "nsed16_rmse": r16,
                "nsed64_rmse": r64,
                "delta": delta,
                "tol": SKILL_NO_REGRESSION_TOL[name],
                "pass": bool(delta <= SKILL_NO_REGRESSION_TOL[name]),
            }
    else:
        for name in SKILL_FIELDS:
            arr = name.lower()
            skill_fields[name] = {
                "nsed16_vs_64_bit_identical": bool(field_diffs[arr]["bit_identical"]),
                "max_abs_delta": float(field_diffs[arr]["max_abs"]),
                "pass": bool(field_diffs[arr]["bit_identical"]),
            }

    skill_payload = {
        "schema": "V0100WaveB1NSED16Skill24H",
        "schema_version": 1,
        "status": "PASS" if all(v["pass"] for v in skill_fields.values()) else "FAIL",
        "run_id": j16.get("run_id"),
        "hours": j16.get("hours"),
        "NSED16_run": str(args.nsed16_json),
        "NSED64_run": str(args.nsed64_json),
        "cpu_reference_available": cpu_available,
        "cpu_reference_note": (
            "CPU-WRF truth was available for the final valid time and RMSE deltas are reported."
            if cpu_available else
            "No CPU-WRF wrfout exists locally for the Wave-A run's +24h valid time; no-regression is proven by cap16-vs-cap64 bit identity of T2/U10/V10."
        ),
        "predeclared_rmse_delta_tolerances": SKILL_NO_REGRESSION_TOL,
        "fields": skill_fields,
        "all_saved_arrays_bit_identical": all_identity,
    }

    conservation_fields = {name: field_diffs[name] for name in STATE_COMPARE_FIELDS if name in field_diffs}
    conservation_payload = {
        "schema": "V0100WaveB1NSED16Conservation",
        "schema_version": 1,
        "status": "PASS" if all(v["bit_identical"] for v in conservation_fields.values()) else "FAIL",
        "run_id": j16.get("run_id"),
        "hours": j16.get("hours"),
        "NSED16_run": str(args.nsed16_json),
        "NSED64_run": str(args.nsed64_json),
        "interpretation": "Water budget unchanged if vapor/condensate fields and precip accumulators are bit-identical between cap=16 and cap=64.",
        "fields": conservation_fields,
        "total_precip_delta_mm_sum": float(conservation_fields["rainnc"]["sum_delta"]),
        "total_cumulus_precip_delta_mm_sum": float(conservation_fields["rainc"]["sum_delta"]),
        "pass": bool(all(v["bit_identical"] for v in conservation_fields.values())),
    }

    _write_json(Path(args.skill_out), skill_payload)
    _write_json(Path(args.conservation_out), conservation_payload)
    print(json.dumps({"skill": skill_payload["status"], "conservation": conservation_payload["status"], "all_identity": all_identity}, indent=2))
    return 0 if skill_payload["status"] == "PASS" and conservation_payload["status"] == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("precip")
    p.add_argument("--out", type=Path, default=PROOF / "wave_b1_nsed16_precip_oracle.json")
    r = sub.add_parser("run24h")
    r.add_argument("--nsed", type=int, required=True)
    r.add_argument("--hours", type=float, default=24.0)
    r.add_argument("--segment-steps", type=int, default=180)
    r.add_argument("--cadence", type=int, default=180)
    r.add_argument("--run-id", default=L2_RUN_ID)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--out-npz", type=Path, required=True)
    c = sub.add_parser("compare24h")
    c.add_argument("--nsed16-json", type=Path, required=True)
    c.add_argument("--nsed16-npz", type=Path, required=True)
    c.add_argument("--nsed64-json", type=Path, required=True)
    c.add_argument("--nsed64-npz", type=Path, required=True)
    c.add_argument("--skill-out", type=Path, default=PROOF / "wave_b1_nsed16_skill_24h.json")
    c.add_argument("--conservation-out", type=Path, default=PROOF / "wave_b1_nsed16_conservation.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "precip":
        return precip_mode(args)
    if args.mode == "run24h":
        return run24h_mode(args)
    if args.mode == "compare24h":
        return compare24h_mode(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
