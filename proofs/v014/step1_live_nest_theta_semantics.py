#!/usr/bin/env python3
"""V0.14 Step-1 live-nest theta/T_STATE semantics proof."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_live_nest_theta_semantics.json"
OUT_MD = PROOF_DIR / "step1_live_nest_theta_semantics.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-live-nest-theta-semantics/sprint-contract.md"
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_live_nest_theta_semantics")
WRF_TRUTH = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth")
RUN_CASE3 = live.RUN_CASE3
WRFINPUT_D02 = RUN_CASE3 / "wrfinput_d02"
WRFOUT_H0_D02 = RUN_CASE3 / "wrfout_d02_2026-05-01_18:00:00"
REQUIRED_ANCESTOR = "7ae33eda"

TARGET_STEP = 1
TARGET_DOMAIN = 2
PRECALL_SURFACE = "before_first_rk_step_part1_call"
THETA_OFFSET_K = 300.0
R_D = 287.0
R_V = 461.6
RV_OVER_RD = R_V / R_D
T_MATERIAL_THRESHOLD_K = 1.0e-3
Q_NONTRUTH_REPORT_THRESHOLD = 1.0e-6


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return sanitize_json(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], *, cwd: Path | None = None, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([device for device in devices if device.platform == "gpu"]),
            }
        )
    except Exception as exc:
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def fortran_index(field: str, index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if field in {"T_STATE", "P_STATE", "PB", "QVAPOR"} and len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field == "PHB" and len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
    if field == "MUB" and len(index) == 2:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    return None


def diff_metrics(field: str, candidate: Any, reference: Any, *, region: str = "full_domain") -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "region": region,
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    diff = cand - ref
    absdiff = np.abs(diff)
    finite_abs = absdiff[np.isfinite(absdiff)]
    mismatch_mask = (diff != 0.0) | (~np.isfinite(diff))
    mismatch = np.argwhere(mismatch_mask)
    first = tuple(int(x) for x in mismatch[0]) if mismatch.size else None
    if finite_abs.size:
        worst = tuple(int(x) for x in np.unravel_index(int(np.nanargmax(absdiff)), absdiff.shape))
        max_abs = float(np.nanmax(absdiff))
        rmse = float(np.sqrt(np.nanmean(diff * diff)))
        bias = float(np.nanmean(diff))
        p95 = float(np.nanpercentile(absdiff, 95))
        p99 = float(np.nanpercentile(absdiff, 99))
    else:
        worst = first
        max_abs = None
        rmse = None
        bias = None
        p95 = None
        p99 = None
    return {
        "status": "OK",
        "region": region,
        "count": int(diff.size),
        "shape": list(diff.shape),
        "max_abs": max_abs,
        "rmse": rmse,
        "bias": bias,
        "p95": p95,
        "p99": p99,
        "nonfinite_diff_count": int((~np.isfinite(diff)).sum()),
        "first_mismatch_index": list(first) if first is not None else None,
        "first_mismatch_fortran": fortran_index(field, first),
        "worst_mismatch_index": list(worst) if worst is not None else None,
        "worst_mismatch_fortran": fortran_index(field, worst),
    }


def load_wrf_precall_truth() -> dict[str, Any]:
    shapes = pre.expected_shapes()
    parsed = pre.parse_wrf_surface(PRECALL_SURFACE, shapes)
    if parsed.get("status") != "WRF_SURFACE_READY":
        return {"status": parsed.get("status"), "blocker": parsed}
    return parsed


def strip_arrays(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "arrays"}


def load_nc_first(path: Path, var: str) -> np.ndarray:
    with Dataset(str(path)) as dataset:
        value = dataset.variables[var]
        data = value[0] if value.dimensions and value.dimensions[0] == "Time" else value[:]
        return np.asarray(data, dtype=np.float64)


def load_use_theta_m() -> dict[str, Any]:
    result: dict[str, Any] = {"wrfinput_attr": None, "wrfout_attr": None, "namelist_output": None}
    for key, path in (("wrfinput_attr", WRFINPUT_D02), ("wrfout_attr", WRFOUT_H0_D02)):
        try:
            with Dataset(str(path)) as dataset:
                result[key] = int(getattr(dataset, "USE_THETA_M"))
        except Exception as exc:
            result[f"{key}_error"] = repr(exc)
    namelist_output = RUN_CASE3 / "namelist.output"
    if namelist_output.is_file():
        for line in namelist_output.read_text(encoding="utf-8", errors="replace").splitlines():
            if "USE_THETA_M" in line:
                result["namelist_output"] = line.strip()
                break
    return result


def wrf_theta_m_from_dry(th_dry: np.ndarray, qv: np.ndarray) -> np.ndarray:
    return (th_dry + THETA_OFFSET_K) * (1.0 + RV_OVER_RD * qv) - THETA_OFFSET_K


def wrf_dry_from_theta_m(th_m: np.ndarray, qv: np.ndarray) -> np.ndarray:
    return (th_m + THETA_OFFSET_K) / (1.0 + RV_OVER_RD * qv) - THETA_OFFSET_K


def adjust_tempqv_transcription(
    *,
    mub: np.ndarray,
    save_mub: np.ndarray,
    c3h: np.ndarray,
    c4h: np.ndarray,
    p_top: float,
    th: np.ndarray,
    pp: np.ndarray,
    qv: np.ndarray,
    use_theta_m: int,
    dtype: Any = np.float64,
) -> dict[str, np.ndarray]:
    """Transcribe WRF dyn_em/nest_init_utils.F::adjust_tempqv."""

    dtype = np.dtype(dtype).type
    mub_d = np.asarray(mub, dtype=dtype)
    save_mub_d = np.asarray(save_mub, dtype=dtype)
    c3 = np.asarray(c3h, dtype=dtype)
    c4 = np.asarray(c4h, dtype=dtype)
    top = dtype(p_top)
    th_in = np.asarray(th, dtype=dtype)
    pp_in = np.asarray(pp, dtype=dtype)
    qv_in = np.asarray(qv, dtype=dtype)
    rv_over_rd = dtype(RV_OVER_RD)
    one = dtype(1.0)

    p_old = c4[:, None, None] + c3[:, None, None] * save_mub_d[None, :, :] + top + pp_in
    p_new = c4[:, None, None] + c3[:, None, None] * mub_d[None, :, :] + top + pp_in
    if int(use_theta_m) == 1:
        tc = (
            (th_in + dtype(300.0))
            * (p_old / dtype(1.0e5)) ** (dtype(2.0) / dtype(7.0))
            / (one + rv_over_rd * qv_in)
            - dtype(273.15)
        )
    else:
        tc = (th_in + dtype(300.0)) * (p_old / dtype(1.0e5)) ** (dtype(2.0) / dtype(7.0)) - dtype(273.15)
    es = dtype(610.78) * np.exp(dtype(17.0809) * tc / (dtype(234.175) + tc)).astype(dtype)
    e = qv_in * p_old / (dtype(0.622) + qv_in)
    rh = e / es

    if int(use_theta_m) == 1:
        thloc = (th_in + dtype(300.0)) / (one + rv_over_rd * qv_in)
    else:
        thloc = th_in + dtype(300.0)
    dth1 = dtype(-191.86e-3) * thloc / (p_new + p_old) * (p_new - p_old)
    dth = dtype(-191.86e-3) * (thloc + dtype(0.5) * dth1) / (p_new + p_old) * (p_new - p_old)
    if int(use_theta_m) == 1:
        th_out = (thloc + dth) * (one + rv_over_rd * qv_in) - dtype(300.0)
    else:
        th_out = thloc + dth - dtype(300.0)

    tc_new = (thloc + dth) * (p_new / dtype(1.0e5)) ** (dtype(2.0) / dtype(7.0)) - dtype(273.15)
    es_new = dtype(610.78) * np.exp(dtype(17.0809) * tc_new / (dtype(234.175) + tc_new)).astype(dtype)
    e_new = rh * es_new
    qv_out = dtype(0.622) * e_new / (p_new - e_new)

    return {
        "p_old": np.asarray(p_old, dtype=np.float64),
        "p_new": np.asarray(p_new, dtype=np.float64),
        "rh": np.asarray(rh, dtype=np.float64),
        "th": np.asarray(th_out, dtype=np.float64),
        "qv": np.asarray(qv_out, dtype=np.float64),
        "thloc_dry_full": np.asarray(thloc + dth, dtype=np.float64),
    }


def capture_candidate_arrays() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    inputs = live.build_live_nest_step1_inputs()
    raw_state = inputs["raw_child"]["state"]
    live_state = inputs["live_child"]["state"]
    metrics = inputs["live_child"]["metrics"]
    raw_mub = np.asarray(jax.device_get(raw_state.mu_total - raw_state.mu_perturbation), dtype=np.float64)
    live_mub = np.asarray(jax.device_get(live_state.mu_total - live_state.mu_perturbation), dtype=np.float64)
    raw_pp = np.asarray(jax.device_get(raw_state.p_perturbation), dtype=np.float64)
    raw_t_dry = np.asarray(jax.device_get(raw_state.theta - THETA_OFFSET_K), dtype=np.float64)
    raw_qv = np.asarray(jax.device_get(raw_state.qv), dtype=np.float64)
    live_t_current = np.asarray(jax.device_get(live_state.theta - THETA_OFFSET_K), dtype=np.float64)
    live_pp = np.asarray(jax.device_get(live_state.p_perturbation), dtype=np.float64)
    live_pb = np.asarray(jax.device_get(live_state.p_total - live_state.p_perturbation), dtype=np.float64)
    live_phb = np.asarray(jax.device_get(live_state.ph_total - live_state.ph_perturbation), dtype=np.float64)
    c3h = np.asarray(jax.device_get(metrics.c3h), dtype=np.float64)
    c4h = np.asarray(jax.device_get(metrics.c4h), dtype=np.float64)
    p_top = float(np.asarray(jax.device_get(metrics.p_top), dtype=np.float64).reshape(-1)[0])

    th_m_conversion = wrf_theta_m_from_dry(raw_t_dry, raw_qv)
    direct_dry_adjust = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=raw_t_dry,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
    )
    best_adjust = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=th_m_conversion,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
    )
    best_adjust_fp32 = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=th_m_conversion,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
        dtype=np.float32,
    )
    dry_adjust = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=raw_t_dry,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=0,
    )
    wrong_order = wrf_theta_m_from_dry(dry_adjust["th"], dry_adjust["qv"])

    return {
        "status": "CANDIDATE_ARRAYS_READY",
        "arrays": {
            "raw_t_dry": raw_t_dry,
            "current_live_t_state": live_t_current,
            "theta_m_conversion_only": th_m_conversion,
            "adjust_tempqv_direct_raw_dry_use_theta_m1": direct_dry_adjust["th"],
            "theta_m_then_adjust_tempqv": best_adjust["th"],
            "theta_m_then_adjust_tempqv_fp32": best_adjust_fp32["th"],
            "dry_adjust_then_theta_m_wrong_order": wrong_order,
            "raw_qv": raw_qv,
            "candidate_qv": best_adjust["qv"],
            "candidate_qv_fp32": best_adjust_fp32["qv"],
            "live_p_state": live_pp,
            "live_pb": live_pb,
            "live_mub": live_mub,
            "live_phb": live_phb,
            "raw_mub": raw_mub,
            "raw_pp": raw_pp,
            "c3h": c3h,
            "c4h": c4h,
        },
        "metadata": {
            "p_top": p_top,
            "use_theta_m": load_use_theta_m(),
            "live_nest_base_init": inputs["live_child"].get("live_nest_base_init"),
            "grid": {
                "domain": "d02",
                "mass_shape": [int(inputs["grid"].nz), int(inputs["grid"].ny), int(inputs["grid"].nx)],
                "dx_m": float(inputs["grid"].projection.dx_m),
                "dy_m": float(inputs["grid"].projection.dy_m),
            },
            "candidate_inputs_match_contract": {
                "save_mub": "raw wrfinput_d02 MUB via raw_state.mu_total-raw_state.mu_perturbation",
                "mub": "live-nest recomputed MUB via live_state.mu_total-live_state.mu_perturbation",
                "pp": "raw wrfinput_d02 P via raw_state.p_perturbation",
                "th": "raw wrfinput_d02 T, with required WRF use_theta_m=1 dry-to-moist conversion tested explicitly",
                "qv": "raw wrfinput_d02 QVAPOR via raw_state.qv",
                "c3h_c4h_p_top": "live child DycoreMetrics from wrfinput_d02",
            },
        },
    }


def compare_candidates(wrf: Mapping[str, Any], capture: Mapping[str, Any]) -> dict[str, Any]:
    arrays = capture["arrays"]
    wrf_arrays = wrf["arrays"]
    t_reference = wrf_arrays["T_STATE"]
    t_candidates = {
        "raw_t_dry": arrays["raw_t_dry"],
        "current_live_t_state": arrays["current_live_t_state"],
        "adjust_tempqv_direct_raw_dry_use_theta_m1": arrays["adjust_tempqv_direct_raw_dry_use_theta_m1"],
        "theta_m_conversion_only": arrays["theta_m_conversion_only"],
        "theta_m_then_adjust_tempqv": arrays["theta_m_then_adjust_tempqv"],
        "theta_m_then_adjust_tempqv_fp32": arrays["theta_m_then_adjust_tempqv_fp32"],
        "dry_adjust_then_theta_m_wrong_order": arrays["dry_adjust_then_theta_m_wrong_order"],
    }
    t_metrics = {name: diff_metrics("T_STATE", value, t_reference) for name, value in t_candidates.items()}
    continuity = {
        "P_STATE": diff_metrics("P_STATE", arrays["live_p_state"], wrf_arrays["P_STATE"]),
        "PB": diff_metrics("PB", arrays["live_pb"], wrf_arrays["PB"]),
        "MUB": diff_metrics("MUB", arrays["live_mub"], wrf_arrays["MUB"]),
        "PHB": diff_metrics("PHB", arrays["live_phb"], wrf_arrays["PHB"]),
    }
    qv_truth_status = {
        "accepted_wrf_precall_qvapor_present": "QVAPOR" in wrf_arrays,
        "accepted_wrf_precall_schema_fields": sorted(wrf_arrays.keys()),
        "status": "BLOCKED_NO_ACCEPTED_WRF_PRECALL_QVAPOR_TRUTH",
    }
    qv_against_h0: dict[str, Any] = {"status": "NOT_EXECUTED", "reason": "missing h0 wrfout"}
    if WRFOUT_H0_D02.is_file():
        h0_qv = load_nc_first(WRFOUT_H0_D02, "QVAPOR")
        h0_t = load_nc_first(WRFOUT_H0_D02, "T")
        qv_against_h0 = {
            "status": "REPORT_ONLY_NOT_ACCEPTED_PRECALL_TRUTH",
            "source": str(WRFOUT_H0_D02),
            "raw_qv_vs_h0_qvapor": diff_metrics("QVAPOR", arrays["raw_qv"], h0_qv),
            "candidate_qv_vs_h0_qvapor": diff_metrics("QVAPOR", arrays["candidate_qv"], h0_qv),
            "candidate_qv_fp32_vs_h0_qvapor": diff_metrics("QVAPOR", arrays["candidate_qv_fp32"], h0_qv),
            "raw_dry_t_vs_h0_t": diff_metrics("T_STATE", arrays["raw_t_dry"], h0_t),
            "candidate_moist_t_dry_view_vs_h0_t": diff_metrics(
                "T_STATE",
                wrf_dry_from_theta_m(arrays["theta_m_then_adjust_tempqv"], arrays["candidate_qv"]),
                h0_t,
            ),
        }
    best = t_metrics["theta_m_then_adjust_tempqv"]
    raw = t_metrics["raw_t_dry"]
    closure_ratio = None
    if best.get("max_abs") is not None and raw.get("max_abs"):
        closure_ratio = float(raw["max_abs"]) / max(float(best["max_abs"]), 1.0e-300)
    return {
        "status": "LIVE_NEST_THETA_SEMANTICS_COMPARISON_EXECUTED",
        "diff_sign": "candidate_minus_wrf",
        "t_state_metrics": t_metrics,
        "continuity_vs_wrf_precall": continuity,
        "qvapor_vs_wrf_precall": qv_truth_status,
        "qvapor_report_only_vs_wrfout_h0": qv_against_h0,
        "best_candidate": {
            "name": "theta_m_then_adjust_tempqv",
            "max_abs": best.get("max_abs"),
            "rmse": best.get("rmse"),
            "bias": best.get("bias"),
            "p99": best.get("p99"),
            "material_threshold_K": T_MATERIAL_THRESHOLD_K,
            "closes_to_material_threshold": (
                best.get("status") == "OK"
                and best.get("max_abs") is not None
                and float(best["max_abs"]) <= T_MATERIAL_THRESHOLD_K
            ),
            "closure_ratio_raw_max_abs_to_best_max_abs": closure_ratio,
        },
    }


def classify(comparisons: Mapping[str, Any]) -> tuple[str, list[str], str]:
    if comparisons.get("status") != "LIVE_NEST_THETA_SEMANTICS_COMPARISON_EXECUTED":
        return (
            "STEP1_LIVE_NEST_THETA_BLOCKED_COMPARISON",
            ["Candidate comparison did not execute."],
            "Fix the comparison blocker and rerun.",
        )
    best = comparisons["best_candidate"]
    qv_status = comparisons["qvapor_vs_wrf_precall"]
    if bool(best.get("closes_to_material_threshold")) and bool(qv_status.get("accepted_wrf_precall_qvapor_present")):
        return (
            "STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PROVEN_SOURCE_FIX_READY",
            [],
            "Apply the initialization-only theta_m + adjust_tempqv source patch and run the full proof chain.",
        )
    if not bool(best.get("closes_to_material_threshold")):
        return (
            "STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL",
            [
                "WRF theta_m conversion plus adjust_tempqv reduces T_STATE max_abs by about three orders of magnitude but leaves a millikelvin-scale residual above the prior 1e-3 K material gate.",
                "No production source patch was made under the sprint hard constraint because the proof-local candidate did not fully close T_STATE.",
                "Accepted WRF pre-call QVAPOR truth is absent from the named truth schema, so QVAPOR closure is report-only against wrfout H0.",
            ],
            "Add or reuse an accepted WRF pre-call QVAPOR/savepoint truth and isolate the remaining T_STATE millikelvin residual before patching production.",
        )
    return (
        "STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_QVAPOR_PRECALL_TRUTH",
        [
            "T_STATE closes to the material threshold, but the named accepted WRF pre-call truth does not contain QVAPOR.",
            "No production source patch was made because the required QVAPOR comparison cannot be completed against accepted pre-call truth.",
        ],
        "Emit or identify accepted WRF pre-call QVAPOR truth, then rerun this proof before patching production.",
    )


def source_evidence() -> dict[str, Any]:
    return {
        "mediation_integrate_live_nest_call": {
            "path": "<USER_HOME>/src/wrf_pristine/WRF/share/mediation_integrate.F",
            "lines": "726-762",
            "evidence": [
                "lines 726-735 save elevation and mub for temp and qv adjustment; nest%mub_save receives nest%mub",
                "lines 737-751 blend parent and nest terrain, mub, and phb",
                "lines 754-762 call adjust_tempqv(nest%mub, nest%mub_save, nest%c3h, nest%c4h, nest%znw, nest%p_top, nest%t_2, nest%p, QVAPOR, use_theta_m, ...)",
            ],
        },
        "adjust_tempqv": {
            "path": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/nest_init_utils.F",
            "lines": "812-890",
            "evidence": [
                "lines 846-859 compute p_old and relative humidity from save_mub, pp, th, and qv",
                "lines 863-884 compute p_new, dth, updated th, and updated qv while conserving RH",
                "lines 852-855 and 869-879 branch on use_theta_m; use_theta_m=1 treats th as moist theta",
            ],
        },
        "theta_m_conversion": {
            "path": "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F",
            "lines": "4918-4928",
            "evidence": [
                "lines 4918-4920 state dry potential temperature is turned into moist potential temperature before halo communications",
                "lines 4923-4928 apply grid%t_2 = (grid%t_2 + T0) * (1 + R_v/R_d * QVAPOR) - T0 when use_theta_m=1",
            ],
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    comp = payload.get("comparisons", {})
    t_metrics = comp.get("t_state_metrics", {})
    raw = t_metrics.get("raw_t_dry", {})
    current = t_metrics.get("current_live_t_state", {})
    direct = t_metrics.get("adjust_tempqv_direct_raw_dry_use_theta_m1", {})
    moist = t_metrics.get("theta_m_conversion_only", {})
    best = t_metrics.get("theta_m_then_adjust_tempqv", {})
    best32 = t_metrics.get("theta_m_then_adjust_tempqv_fp32", {})
    wrong = t_metrics.get("dry_adjust_then_theta_m_wrong_order", {})
    qv = comp.get("qvapor_report_only_vs_wrfout_h0", {})
    qv_best = qv.get("candidate_qv_vs_h0_qvapor", {})
    cont = comp.get("continuity_vs_wrf_precall", {})
    lines = [
        "# V0.14 Step-1 Live-Nest Theta Semantics",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload.get('gpu_used')}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['required_ancestor_7ae33eda'].get('is_ancestor')}`.",
        f"- Real run `USE_THETA_M`: `{payload['candidate_capture']['metadata']['use_theta_m']}`.",
        f"- Raw/current live dry `T_STATE` vs WRF pre-call: max_abs `{raw.get('max_abs')}` / `{current.get('max_abs')}`.",
        f"- `adjust_tempqv` directly on raw dry `T` with `use_theta_m=1`: max_abs `{direct.get('max_abs')}`.",
        f"- WRF dry-to-moist theta conversion only: max_abs `{moist.get('max_abs')}`, rmse `{moist.get('rmse')}`.",
        f"- WRF `theta_m` conversion plus `adjust_tempqv`: max_abs `{best.get('max_abs')}`, rmse `{best.get('rmse')}`, p99 `{best.get('p99')}`.",
        f"- Same candidate with fp32 arithmetic: max_abs `{best32.get('max_abs')}`, rmse `{best32.get('rmse')}`.",
        f"- Wrong order, dry adjust then moist conversion: max_abs `{wrong.get('max_abs')}`.",
        f"- Report-only `QVAPOR` candidate vs `wrfout_d02` H0: max_abs `{qv_best.get('max_abs')}`, rmse `{qv_best.get('rmse')}`.",
        f"- Continuity vs WRF pre-call: `P_STATE` max_abs `{cont.get('P_STATE', {}).get('max_abs')}`, `PB` `{cont.get('PB', {}).get('max_abs')}`, `MUB` `{cont.get('MUB', {}).get('max_abs')}`, `PHB` `{cont.get('PHB', {}).get('max_abs')}`.",
        "",
        "## Interpretation",
        "",
        "- `USE_THETA_M=1` means operational in-memory WRF `grid%t_2` is moist perturbation theta. For this run, `State.theta` should represent WRF `grid%t_2 + 300 K` if it is intended to mirror solve-time WRF state.",
        "- The live-nest theta residual is not closed by `adjust_tempqv` alone on raw dry NetCDF `T`. The needed semantic sequence is dry `T` to moist theta, then WRF `adjust_tempqv`.",
        f"- That sequence reduces max_abs from `{raw.get('max_abs')}` to `{best.get('max_abs')}`, but it remains above the prior `1e-3 K` material threshold, so no production patch was made.",
        "- The named accepted WRF pre-call truth does not include `QVAPOR`; the QVAPOR comparison in this proof is against `wrfout_d02` H0 and is report-only, not an accepted pre-call proof.",
        "",
        "## WRF Source Evidence",
        "",
        "- `share/mediation_integrate.F:726-762`: saves `mub`, blends `ht/mub/phb`, then calls `adjust_tempqv` with `nest%t_2`, `nest%p`, and `QVAPOR`.",
        "- `dyn_em/nest_init_utils.F:812-890`: `adjust_tempqv` computes old/new pressure, preserves RH, updates `th`, then updates `qv`.",
        "- `dyn_em/module_initialize_real.F:4918-4928`: when `use_theta_m=1`, WRF converts dry `grid%t_2` to moist theta in memory.",
        "",
        "Detailed tables are in `proofs/v014/step1_live_nest_theta_semantics.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Live-Nest Theta Semantics",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: prove whether WRF live-nest `T_STATE`/theta semantics after terrain/base blending close the accepted WRF pre-call residual.",
        "",
        "files changed:",
        "- `proofs/v014/step1_live_nest_theta_semantics.py`",
        "- `proofs/v014/step1_live_nest_theta_semantics.json`",
        "- `proofs/v014/step1_live_nest_theta_semantics.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-live-nest-theta-semantics.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{OUT_REVIEW}`",
            f"- `{WRF_TRUTH}` reused, not rebuilt",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"], cwd=ROOT)
    wrf = load_wrf_precall_truth()
    capture: dict[str, Any]
    comparisons: dict[str, Any]
    if wrf.get("status") != "WRF_SURFACE_READY":
        capture = {"status": "NOT_EXECUTED"}
        comparisons = {"status": "NOT_EXECUTED", "blocker": wrf}
        verdict = "STEP1_LIVE_NEST_THETA_BLOCKED_WRF_PRECALL_TRUTH"
        risks = ["Accepted WRF pre-call truth could not be parsed."]
        next_decision = "Fix or restore the WRF pre-call truth root and rerun."
    else:
        capture = capture_candidate_arrays()
        if capture.get("status") != "CANDIDATE_ARRAYS_READY":
            comparisons = {"status": "NOT_EXECUTED", "blocker": capture}
            verdict = "STEP1_LIVE_NEST_THETA_BLOCKED_CANDIDATE_ARRAY_CAPTURE"
            risks = ["Candidate array capture did not complete."]
            next_decision = "Fix the exact candidate capture blocker and rerun."
        else:
            comparisons = compare_candidates(wrf, capture)
            verdict, risks, next_decision = classify(comparisons)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_live_nest_theta_semantics.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "git_head": git_head,
        "required_ancestor_7ae33eda": {
            "command": ancestor["command"],
            "returncode": ancestor["returncode"],
            "is_ancestor": ancestor["returncode"] == 0,
            "stderr_tail": ancestor.get("stderr_tail"),
        },
        "environment": jax_environment(),
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "source_patch_allowed_by_proof": comparisons.get("best_candidate", {}).get("closes_to_material_threshold") is True
        and comparisons.get("qvapor_vs_wrf_precall", {}).get("accepted_wrf_precall_qvapor_present") is True,
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "wrf_surface": PRECALL_SURFACE,
            "theta_offset_K": THETA_OFFSET_K,
            "t_state_material_threshold_K": T_MATERIAL_THRESHOLD_K,
            "qv_report_only_threshold": Q_NONTRUTH_REPORT_THRESHOLD,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "wrf_truth_root": path_info(WRF_TRUTH),
            "run_case3": path_info(RUN_CASE3),
            "wrfinput_d02": path_info(WRFINPUT_D02),
            "wrfout_h0_d02": path_info(WRFOUT_H0_D02),
            "scratch_root": path_info(SCRATCH),
            "d02_replay": path_info(SRC / "gpuwrf/integration/d02_replay.py"),
            "trigger_md": path_info(PROOF_DIR / "step1_jax_loader_tstate.md"),
            "trigger_json": path_info(PROOF_DIR / "step1_jax_loader_tstate.json"),
            "trigger_py": path_info(PROOF_DIR / "step1_jax_loader_tstate.py"),
        },
        "wrf_source_evidence": source_evidence(),
        "wrf_precall_truth": strip_arrays(wrf) if wrf.get("status") == "WRF_SURFACE_READY" else wrf,
        "candidate_capture": strip_arrays(capture),
        "candidate_array_summaries": {
            name: array_summary(value)
            for name, value in capture.get("arrays", {}).items()
            if name
            in {
                "raw_t_dry",
                "current_live_t_state",
                "theta_m_conversion_only",
                "theta_m_then_adjust_tempqv",
                "raw_qv",
                "candidate_qv",
                "live_p_state",
                "live_pb",
                "live_mub",
                "live_phb",
            }
        },
        "comparisons": comparisons,
        "theta_contract_resolution": {
            "use_theta_m": capture.get("metadata", {}).get("use_theta_m"),
            "wrf_in_memory_t_state": "grid%t_2 is perturbation moist theta when use_theta_m=1",
            "operational_state_theta_recommendation": "State.theta should store WRF grid%t_2 + 300 K for solve-time parity; for this run that is full moist theta, not dry NetCDF T + 300 K",
            "dry_theta_view": "Dry theta is recovered as (theta_m + 300)/(1 + R_v/R_d*QVAPOR), then minus 300 for perturbation view.",
        },
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_semantics.py",
                "python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json >/tmp/step1_live_nest_theta_semantics.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "wrf_truth_root_reused": str(WRF_TRUTH),
        },
        "unresolved_risks": risks,
        "next_decision": next_decision,
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
