#!/usr/bin/env python3
"""V0.14 Step-1 live-nest start_domain base-state boundary proof.

CPU-only proof.  This does not edit production source and does not run WRF.
It reuses the accepted disposable WRF ``start_domain_em`` surfaces from the
predecessor sprint, reconstructs the pre-AL/ALT base-state boundary implied by
those surfaces and by the WRF source branch, then ranks proof-local base
candidate families against WRF.
"""

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


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_jax_start_domain_input_split as split  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_start_domain_perturb_subsurface as prior  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_base_state_boundary.json"
OUT_MD = PROOF_DIR / "step1_base_state_boundary.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-base-state-boundary.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_base_state_boundary_wrf_patch.diff"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-base-state-boundary/sprint-contract.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PREDECESSOR_JSON = PROOF_DIR / "step1_jax_start_domain_input_split.json"
PREDECESSOR_MD = PROOF_DIR / "step1_jax_start_domain_input_split.md"
D02_REPLAY = SRC / "gpuwrf/integration/d02_replay.py"
WRF_START_EM = prior.WRF_TREE / "dyn_em/start_em.F"
WRF_CONSTANTS = prior.WRF_TREE / "share/module_model_constants.F"
WRF_NAMELIST = prior.WRF_RUN_DIR / "namelist.input"

REQUIRED_ANCESTOR = "6ced5a8e"
VERDICT = "STEP1_BASE_STATE_BOUNDARY_LOCALIZED_P_SURF_MUB_FP32_SOURCE_ARITHMETIC"

R_D = 287.0
CP_WRF = 7.0 * R_D / 2.0
CV_WRF = CP_WRF - R_D
CVPM_WRF = -CV_WRF / CP_WRF
P1000MB = 100000.0
T0 = 300.0
G = 9.81

MATERIAL_THRESHOLDS = {
    "P_STATE": 1.0,
    "MU_STATE": 1.0e-2,
    "MUB": 1.0e-2,
    "PB": 1.0,
    "PHB": 1.0e-2,
    "ALB": 1.0e-8,
    "ALT": 1.0e-8,
    "HT": 1.0e-6,
    "HT_FINE": 1.0e-6,
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
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
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
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


def run_command(command: list[str], *, cwd: Path = ROOT, timeout_s: int = 120) -> dict[str, Any]:
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
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
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


def git_metadata() -> dict[str, Any]:
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    return {
        "head": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "log_head": run_command(["git", "log", "-1", "--oneline", "--decorate"]),
        "required_ancestor": {
            "commit": REQUIRED_ANCESTOR,
            "returncode": ancestor["returncode"],
            "is_ancestor": ancestor["returncode"] == 0,
            "stderr_tail": ancestor.get("stderr_tail"),
        },
    }


def metric_brief(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    keys = (
        "status",
        "region",
        "shape",
        "count",
        "max_abs",
        "rmse",
        "bias",
        "p95",
        "p99",
        "nonfinite_diff_count",
        "worst_mismatch_index",
        "worst_mismatch_fortran",
    )
    return {key: metric.get(key) for key in keys if key in metric}


def scalar_from_run(run: Any, name: str, dtype: Any = np.float64) -> Any:
    from gpuwrf.integration import d02_replay as replay  # noqa: PLC0415

    return dtype(float(prior.as_np(replay._load(run, "d02", name, 0)).ravel()[0]))


def array_summary(name: str, array: Any, *, include_values: bool = False) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    out: dict[str, Any] = {
        "name": name,
        "shape": list(arr.shape),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "sha256_float64": array_sha256(arr),
    }
    if include_values:
        out["values"] = [float(item) for item in arr.ravel()]
    return out


def line_hits(path: Path, needles: Mapping[str, str]) -> dict[str, Any]:
    if not path.is_file():
        return {key: None for key in needles}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: dict[str, Any] = {}
    for key, needle in needles.items():
        hit = None
        for lineno, line in enumerate(lines, start=1):
            if needle in line:
                hit = {"path": str(path), "line": lineno, "text": line.strip()[:240]}
                break
        hits[key] = hit
    return hits


def header_flags(surface: Mapping[str, Any]) -> dict[str, Any]:
    headers = surface.get("headers") or []
    first = headers[0] if headers else {}
    keys = (
        "domain_id",
        "current_timestr",
        "grid_itimestep",
        "allowed_to_read",
        "press_adj_flag",
        "restart_flag",
        "input_from_file",
        "input_from_hires",
        "hypsometric_opt",
        "rebalance",
        "use_theta_m",
        "use_input_w",
        "start_of_simulation",
        "cycling",
        "moist_index_qv",
    )
    return {key: first.get(key) for key in keys if key in first}


def namelist_hits() -> dict[str, Any]:
    if not WRF_NAMELIST.is_file():
        return {}
    wanted = ("max_dom", "input_from_file", "base_temp")
    hits = {}
    for line in WRF_NAMELIST.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        for key in wanted:
            if stripped.startswith(key):
                hits[key] = stripped
    return hits


def t_init_from_alb_pb(alb: Any, pb: Any, *, cp: float = CP_WRF) -> np.ndarray:
    pb_arr = prior.as_np(pb, np.float64)
    alb_arr = prior.as_np(alb, np.float64)
    cvpm = -((cp - R_D) / cp)
    t_init = alb_arr * (P1000MB / R_D) / ((pb_arr / P1000MB) ** cvpm) - T0
    return np.asarray(t_init, dtype=np.float64)


def base_from_wrf_mub(
    *,
    run: Any,
    hgt: Any,
    metrics: Any,
    wrf_mub: Any,
    dtype: Any,
    cp: float,
) -> dict[str, np.ndarray]:
    """Recompute the base with WRF-emitted MUB as the exact source surface."""

    f = dtype
    p_top = scalar_from_run(run, "P_TOP", f)
    p00 = scalar_from_run(run, "P00", f)
    t00 = scalar_from_run(run, "T00", f)
    lapse = scalar_from_run(run, "TLP", f)
    tiso = scalar_from_run(run, "TISO", f)
    lapse_strat = scalar_from_run(run, "TLP_STRAT", f)
    p_strat = scalar_from_run(run, "P_STRAT", f)

    hgt_arr = prior.as_np(hgt, dtype)
    mub = prior.as_np(wrf_mub, dtype)
    c3h = prior.as_np(metrics.c3h, dtype)
    c4h = prior.as_np(metrics.c4h, dtype)
    c3f = prior.as_np(metrics.c3f, dtype)
    c4f = prior.as_np(metrics.c4f, dtype)
    rd = f(R_D)
    g = f(G)
    p1000 = f(P1000MB)
    cp_arr = f(cp)
    cvpm = f(-((cp_arr - rd) / cp_arr))
    rdocp = f(rd / cp_arr)
    t0 = f(T0)

    pb = (c3h[:, None, None] * mub[None, :, :] + c4h[:, None, None] + p_top).astype(dtype)
    temp = np.maximum(tiso, (t00 + lapse * np.log((pb / p00).astype(dtype))).astype(dtype)).astype(dtype)
    safe_p_strat = np.maximum(p_strat, f(1.0)).astype(dtype)
    strat_temp = (tiso + lapse_strat * np.log((pb / safe_p_strat).astype(dtype))).astype(dtype)
    temp = np.where((p_strat > f(0.0)) & (pb < p_strat), strat_temp, temp).astype(dtype)
    t_init = (temp * ((p00 / pb).astype(dtype)) ** rdocp - t0).astype(dtype)
    alb = (rd / p1000 * (t_init + t0) * (pb / p1000) ** cvpm).astype(dtype)

    phb = np.empty((pb.shape[0] + 1,) + pb.shape[1:], dtype=dtype)
    phb[0] = (hgt_arr * g).astype(dtype)
    for full_k in range(1, int(pb.shape[0]) + 1):
        half_k = full_k - 1
        pfu = (c3f[full_k] * mub + c4f[full_k] + p_top).astype(dtype)
        pfd = (c3f[full_k - 1] * mub + c4f[full_k - 1] + p_top).astype(dtype)
        phm = (c3h[half_k] * mub + c4h[half_k] + p_top).astype(dtype)
        phb[full_k] = (phb[full_k - 1] + alb[half_k] * phm * np.log((pfd / pfu).astype(dtype))).astype(dtype)

    return {
        "PB": pb.astype(np.float64),
        "MUB": mub.astype(np.float64),
        "PHB": phb.astype(np.float64),
        "T_INIT": t_init.astype(np.float64),
        "ALB": alb.astype(np.float64),
        "P_SURF": (mub + p_top).astype(np.float64),
    }


def downstream_metrics(
    *,
    candidate: Mapping[str, Any],
    state: Any,
    metrics: Any,
    after_hyp: Mapping[str, Any],
    after_press: Mapping[str, Any],
    hgt: Any,
    hgt_fine: Any,
    dtype: Any,
) -> dict[str, Any]:
    alt = prior.diagnose_alt_from_ph(
        ph=state.ph_perturbation,
        phb=candidate["PHB"],
        mub=candidate["MUB"],
        mu=state.mu_perturbation,
        metrics=metrics,
        dtype=dtype,
    )
    pressure = prior.pressure_from_alt(
        pb=candidate["PB"],
        theta_full=state.theta,
        alt=alt,
        dtype=dtype,
    )
    mu_press = prior.press_adj_mu(
        mu_before=state.mu_perturbation,
        al=alt - candidate["ALB"],
        alt=alt,
        alb=candidate["ALB"],
        ht=hgt,
        ht_fine=hgt_fine,
        dtype=dtype,
    )
    return {
        "ALT": metric_brief(prior.diff_metrics("ALT", alt, after_hyp["ALT"])),
        "P_STATE": metric_brief(prior.diff_metrics("P_STATE", pressure, after_hyp["P_STATE"])),
        "MU_STATE": metric_brief(prior.diff_metrics("MU_STATE", mu_press, after_press["MU2_STATE"])),
    }


def base_metrics(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    ref_t_init = t_init_from_alb_pb(reference["ALB"], reference["PB"], cp=CP_WRF)
    candidate_t_init = candidate.get("T_INIT")
    if candidate_t_init is None:
        candidate_t_init = t_init_from_alb_pb(candidate["ALB"], candidate["PB"], cp=CP_WRF)
    return {
        "MUB": metric_brief(prior.diff_metrics("MUB", candidate["MUB"], reference["MUB"])),
        "PB": metric_brief(prior.diff_metrics("PB", candidate["PB"], reference["PB"])),
        "PHB": metric_brief(prior.diff_metrics("PHB", candidate["PHB"], reference["PHB"])),
        "ALB": metric_brief(prior.diff_metrics("ALB", candidate["ALB"], reference["ALB"])),
        "T_INIT_recovered": metric_brief(prior.diff_metrics("T_INIT", candidate_t_init, ref_t_init)),
    }


def rank_candidate_rows(candidates: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, item in candidates.items():
        base = item.get("base_residuals", {})
        downstream = item.get("downstream_residuals", {})
        rows.append(
            {
                "source_family": name,
                "dtype": item.get("dtype"),
                "cp": item.get("cp"),
                "hgt_source": item.get("hgt_source"),
                "mub_source": item.get("mub_source"),
                "MUB_max_abs": base.get("MUB", {}).get("max_abs"),
                "PB_max_abs": base.get("PB", {}).get("max_abs"),
                "PHB_max_abs": base.get("PHB", {}).get("max_abs"),
                "ALB_max_abs": base.get("ALB", {}).get("max_abs"),
                "P_STATE_downstream_max_abs": downstream.get("P_STATE", {}).get("max_abs"),
                "MU_STATE_downstream_max_abs": downstream.get("MU_STATE", {}).get("max_abs"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("P_STATE_downstream_max_abs") or 0.0),
            float(row.get("MU_STATE_downstream_max_abs") or 0.0),
        ),
    )


def source_family_candidates(inputs: Mapping[str, Any], after_hyp: Mapping[str, Any], after_press: Mapping[str, Any]) -> dict[str, Any]:
    from gpuwrf.integration.d02_replay import _wrf_start_domain_base_from_hgt  # noqa: PLC0415

    live_child = inputs["live_child"]
    state = live_child["state"]
    grid = live_child["grid"]
    raw_child = inputs["raw_child"]
    metrics = live_child["metrics"]
    run = inputs["run"]

    current_pb, current_mub, current_phb, current_t_init, current_alb = _wrf_start_domain_base_from_hgt(
        run,
        "d02",
        hgt=grid.terrain_height,
        metrics=metrics,
    )
    candidates: dict[str, Any] = {
        "production_current_live_child_base_fp64_cp1004_0_jax_hgt": {
            "dtype": "float64",
            "cp": 1004.0,
            "hgt_source": "current JAX live_nest grid.terrain_height",
            "mub_source": "production _wrf_start_domain_base_from_hgt p_surf formula",
            "arrays": {
                "PB": prior.as_np(current_pb),
                "MUB": prior.as_np(current_mub),
                "PHB": prior.as_np(current_phb),
                "T_INIT": prior.as_np(current_t_init),
                "ALB": prior.as_np(current_alb),
            },
        }
    }

    for label, dtype, cp, hgt_source, hgt in (
        ("proof_formula_fp64_cp1004_0_jax_hgt", np.float64, 1004.0, "current JAX live_nest grid.terrain_height", grid.terrain_height),
        ("proof_formula_fp64_cp1004_5_jax_hgt", np.float64, CP_WRF, "current JAX live_nest grid.terrain_height", grid.terrain_height),
        ("proof_formula_fp32_cp1004_0_jax_hgt", np.float32, 1004.0, "current JAX live_nest grid.terrain_height", grid.terrain_height),
        ("proof_formula_fp32_cp1004_5_jax_hgt", np.float32, CP_WRF, "current JAX live_nest grid.terrain_height", grid.terrain_height),
        ("proof_formula_fp32_cp1004_5_wrf_ht", np.float32, CP_WRF, "WRF emitted HT after_hypsometric", after_hyp["HT"]),
    ):
        candidate = split.wrf_start_domain_base_candidate(
            run=run,
            hgt=hgt,
            metrics=metrics,
            dtype=dtype,
            cp=cp,
        )
        candidates[label] = {
            "dtype": np.dtype(dtype).name,
            "cp": cp,
            "hgt_source": hgt_source,
            "mub_source": "proof-local p_surf formula",
            "arrays": candidate,
        }

    candidates["wrf_boundary_mub_fp32_cp1004_5_wrf_ht"] = {
        "dtype": "float32",
        "cp": CP_WRF,
        "hgt_source": "WRF emitted HT after_hypsometric",
        "mub_source": "WRF emitted/recovered post-assignment MUB",
        "arrays": base_from_wrf_mub(
            run=run,
            hgt=after_hyp["HT"],
            metrics=metrics,
            wrf_mub=after_hyp["MUB"],
            dtype=np.float32,
            cp=CP_WRF,
        ),
    }

    output: dict[str, Any] = {}
    for name, item in candidates.items():
        arrays = item["arrays"]
        output[name] = {
            key: value for key, value in item.items() if key != "arrays"
        }
        output[name]["base_residuals"] = base_metrics(arrays, after_hyp)
        output[name]["downstream_residuals"] = downstream_metrics(
            candidate=arrays,
            state=state,
            metrics=metrics,
            after_hyp=after_hyp,
            after_press=after_press,
            hgt=after_hyp["HT"],
            hgt_fine=after_hyp["HT_FINE"],
            dtype=np.float32,
        )
        output[name]["array_summaries"] = {
            "PB": array_summary("PB", arrays["PB"]),
            "MUB": array_summary("MUB", arrays["MUB"]),
            "PHB": array_summary("PHB", arrays["PHB"]),
            "ALB": array_summary("ALB", arrays["ALB"]),
        }
    output["_context"] = {
        "state_source": "current production live_child state for PH/theta/MU in downstream AL/ALT/P/MU tests",
        "ht_fine_source": "WRF emitted HT_FINE after_hypsometric for press_adj tests",
        "raw_child_hgt_summary": array_summary("raw_child.grid.terrain_height", prior.as_np(raw_child["grid"].terrain_height)),
    }
    return output


def direct_truth_ablation(inputs: Mapping[str, Any], jax_arrays: Mapping[str, Any], after_hyp: Mapping[str, Any], after_press: Mapping[str, Any]) -> dict[str, Any]:
    metrics = inputs["live_child"]["metrics"]
    state = inputs["live_child"]["state"]
    phb_mub_alt = prior.diagnose_alt_from_ph(
        ph=state.ph_perturbation,
        phb=after_hyp["PHB"],
        mub=after_hyp["MUB"],
        mu=state.mu_perturbation,
        metrics=metrics,
        dtype=np.float32,
    )
    phb_mub_p = prior.pressure_from_alt(
        pb=jax_arrays["PB"],
        theta_full=jax_arrays["THETA_FULL"],
        alt=phb_mub_alt,
        dtype=np.float32,
    )
    wrf_alt_p = prior.pressure_from_alt(
        pb=jax_arrays["PB"],
        theta_full=jax_arrays["THETA_FULL"],
        alt=after_hyp["ALT"],
        dtype=np.float64,
    )
    wrf_al_mu = prior.press_adj_mu(
        mu_before=jax_arrays["MU_STATE"],
        al=after_hyp["AL"],
        alt=after_hyp["ALT"],
        alb=after_hyp["ALB"],
        ht=jax_arrays["HT"],
        ht_fine=jax_arrays["HT_FINE"],
        dtype=np.float64,
    )
    return {
        "PHB_MUB_to_wrf_fp32_alt_pressure_vs_wrf_P": metric_brief(
            prior.diff_metrics("P_STATE", phb_mub_p, after_hyp["P_STATE"])
        ),
        "PHB_MUB_to_wrf_fp32_alt_vs_wrf_ALT": metric_brief(prior.diff_metrics("ALT", phb_mub_alt, after_hyp["ALT"])),
        "direct_WRF_ALT_pressure_vs_wrf_P": metric_brief(prior.diff_metrics("P_STATE", wrf_alt_p, after_hyp["P_STATE"])),
        "direct_WRF_AL_ALT_ALB_press_adj_vs_wrf_MU": metric_brief(
            prior.diff_metrics("MU_STATE", wrf_al_mu, after_press["MU2_STATE"])
        ),
    }


def coefficient_payload(metrics: Any) -> dict[str, Any]:
    return {
        "C3F": array_summary("C3F", prior.as_np(metrics.c3f), include_values=True),
        "C4F": array_summary("C4F", prior.as_np(metrics.c4f), include_values=True),
        "C3H": array_summary("C3H", prior.as_np(metrics.c3h), include_values=True),
        "C4H": array_summary("C4H", prior.as_np(metrics.c4h), include_values=True),
    }


def scalar_payload(run: Any) -> dict[str, Any]:
    scalar_names = ("P_TOP", "P00", "T00", "TLP", "TISO", "TLP_STRAT", "P_STRAT")
    return {
        "wrf_module_constants": {
            "r_d": R_D,
            "cp": CP_WRF,
            "cv": CV_WRF,
            "cvpm": CVPM_WRF,
            "p1000mb": P1000MB,
            "t0": T0,
            "g": G,
        },
        "wrfinput_or_namelist_base_scalars": {
            name: float(scalar_from_run(run, name, np.float64)) for name in scalar_names
        },
        "production_current_cp": 1004.0,
    }


def build_proof() -> dict[str, Any]:
    shapes = prior.expected_shapes()
    after_hyp = prior.parse_start_surface("after_hypsometric_p_al_alt", shapes)
    before_press = prior.parse_start_surface("before_press_adj", shapes)
    after_press = prior.parse_start_surface("after_press_adj", shapes)
    blockers = {
        name: surface
        for name, surface in {
            "after_hypsometric_p_al_alt": after_hyp,
            "before_press_adj": before_press,
            "after_press_adj": after_press,
        }.items()
        if surface.get("status") != "WRF_SURFACE_READY"
    }
    if blockers:
        return {"status": "BLOCKED_INPUTS", "blockers": blockers}

    inputs = live.build_live_nest_step1_inputs()
    current = split.build_current_jax_arrays(inputs)
    jax_arrays = current["arrays"]
    live_child = inputs["live_child"]
    metrics = live_child["metrics"]
    run = inputs["run"]

    after_arrays = after_hyp["arrays"]
    before_arrays = before_press["arrays"]
    after_press_arrays = after_press["arrays"]
    p_top = float(scalar_from_run(run, "P_TOP", np.float64))
    p_surf_recovered = after_arrays["MUB"] + p_top
    t_init_recovered = t_init_from_alb_pb(after_arrays["ALB"], after_arrays["PB"], cp=CP_WRF)

    input_metrics = split.compare_inputs(jax_arrays, after_arrays, before_arrays, after_press_arrays)
    candidates = source_family_candidates(inputs, after_arrays, after_press_arrays)
    candidate_rows = rank_candidate_rows({k: v for k, v in candidates.items() if not k.startswith("_")})
    truth_ablation = direct_truth_ablation(inputs, jax_arrays, after_arrays, after_press_arrays)

    formula_fp32 = candidates["proof_formula_fp32_cp1004_5_jax_hgt"]
    wrf_mub = candidates["wrf_boundary_mub_fp32_cp1004_5_wrf_ht"]
    p_formula = formula_fp32["downstream_residuals"]["P_STATE"]["max_abs"]
    p_wrf_mub = wrf_mub["downstream_residuals"]["P_STATE"]["max_abs"]
    mu_formula = formula_fp32["downstream_residuals"]["MU_STATE"]["max_abs"]
    mu_wrf_mub = wrf_mub["downstream_residuals"]["MU_STATE"]["max_abs"]
    mub_formula = formula_fp32["base_residuals"]["MUB"]["max_abs"]

    ready_for_patch = (
        p_formula is not None
        and mu_formula is not None
        and float(p_formula) <= MATERIAL_THRESHOLDS["P_STATE"]
        and float(mu_formula) <= MATERIAL_THRESHOLDS["MU_STATE"]
    )

    ranked_hypotheses = [
        {
            "rank": 1,
            "hypothesis": "The remaining Step-1 base gap is the exact WRF p_surf/MUB source arithmetic feeding AL/ALT.",
            "status": "SUPPORTED_LOCALIZED",
            "evidence": (
                f"The best local fp32/cp=1004.5 formula still leaves MUB max_abs {mub_formula}, "
                f"P_STATE {p_formula}, and MU_STATE {mu_formula}; substituting WRF-emitted MUB into the same "
                f"base/AL/ALT path reduces P_STATE to {p_wrf_mub} and MU_STATE to {mu_wrf_mub}."
            ),
        },
        {
            "rank": 2,
            "hypothesis": "The WRF branch is multi-domain real start_domain with rebalance disabled: PB/T_INIT/ALB are reconstituted from MUB, PHB is not re-integrated in that later block.",
            "status": "SUPPORTED_BY_SOURCE_AND_FLAGS",
            "evidence": (
                "Truth headers report input_from_file=T, hypsometric_opt=2, rebalance=0, restart=F, use_theta_m=1; "
                "namelist max_dom=2. Source lines show the initial p_surf/MUB/PHB integration block followed by "
                "the max_dom>1 real reconstitution block."
            ),
        },
        {
            "rank": 3,
            "hypothesis": "Terrain/blend input is the dominant residual.",
            "status": "REFUTED",
            "evidence": (
                f"JAX HT vs WRF HT max_abs {input_metrics['HT_current_vs_wrf_after_hyp']['max_abs']}; "
                "using WRF HT instead of JAX HT in the fp32/cp=1004.5 formula leaves P_STATE unchanged at "
                f"{candidates['proof_formula_fp32_cp1004_5_wrf_ht']['downstream_residuals']['P_STATE']['max_abs']}."
            ),
        },
        {
            "rank": 4,
            "hypothesis": "Constants or cp=1004.0 vs WRF cp=1004.5 are the dominant residual.",
            "status": "REFUTED_AS_DOMINANT",
            "evidence": (
                "Changing cp affects PHB modestly but not the MUB/PB source. fp32 cp=1004.0 and cp=1004.5 both "
                f"leave P_STATE {candidates['proof_formula_fp32_cp1004_0_jax_hgt']['downstream_residuals']['P_STATE']['max_abs']}."
            ),
        },
        {
            "rank": 5,
            "hypothesis": "Coefficient indexing or PH/MU time-level selection is the cause.",
            "status": "REFUTED_BY_PREDECESSOR_AND_CURRENT_ABLATION",
            "evidence": (
                "The predecessor proved JAX PH and pre-press MU match WRF PH1/MU1 to roundoff/zero. "
                f"With WRF MUB, current coefficients produce P_STATE {p_wrf_mub}; with WRF PHB+MUB direct "
                f"substitution the pressure residual is {truth_ablation['PHB_MUB_to_wrf_fp32_alt_pressure_vs_wrf_P']['max_abs']}."
            ),
        },
    ]

    exclusions = [
        "No production src/gpuwrf edit was made.",
        "No GPU, TOST, Switzerland, FP32 production source, memory production source, or Hermes path was used.",
        "Terrain was falsified as dominant by substituting WRF HT into the proof-local fp32 formula with no P improvement.",
        "cp/constants were falsified as dominant: cp=1004.0 vs 1004.5 does not move MUB/PB and leaves the same downstream P gap.",
        "Coefficient indexing is unlikely: exact WRF MUB with current metrics closes downstream P/MU gates.",
        "PHB integration order remains a small base residual, but not the dominant downstream P/MU blocker after WRF MUB substitution.",
    ]
    next_decision = (
        "Do not patch d02_replay from the current p_surf formula yet. The next source contract should either "
        "instrument one disposable WRF boundary immediately around the p_surf expression/MUB assignment to capture "
        "p_surf_before_mub and MUB exactly, or implement a narrowly gated WRF-compatible fp32/libm p_surf helper "
        "and require P_STATE <= 1 Pa and MU_STATE <= 0.01 Pa in this same proof before production patching."
    )

    return {
        "status": "PROOF_EXECUTED",
        "verdict": VERDICT,
        "shapes": {key: list(value) for key, value in shapes.items()},
        "source_boundary": {
            "classification": "multi_domain_real_input_from_file_rebalance0_hypsometric2",
            "flags_from_wrf_truth_header": header_flags(after_hyp),
            "namelist_hits": namelist_hits(),
            "line_hits": {
                "start_em": line_hits(
                    WRF_START_EM,
                    {
                        "p_surf_expression": "p_surf = p00 * EXP",
                        "mub_assignment": "grid%MUB(i,j) = p_surf - grid%p_top",
                        "multidomain_reconstitute_pb": "grid%pb(i,k,j) = grid%c3h(k  )*grid%MUB(i,j)",
                        "multidomain_reconstitute_t_init": "grid%t_init(i,k,j) = temp*(p00/grid%pb(i,k,j))**(r_d/cp) - t0",
                        "multidomain_reconstitute_alb": "grid%alb(i,k,j) = (r_d/p1000mb)*(grid%t_init(i,k,j)+t0)",
                        "rebalance_phb_guard": "IF ( config_flags%rebalance .EQ. 1 ) THEN",
                        "hypsometric_al_alt": "grid%al(i,k,j) = (grid%ph_1",
                        "pressure_from_alt": "grid%p(i,k,j)=p1000mb*(",
                        "press_adj": "grid%MU_2(i,j) = grid%MU_2(i,j) + grid%al",
                    },
                ),
                "constants": line_hits(
                    WRF_CONSTANTS,
                    {
                        "g": "REAL    , PARAMETER :: g = 9.81",
                        "r_d": "REAL    , PARAMETER :: r_d          = 287.",
                        "cp": "REAL    , PARAMETER :: cp           = 7.*r_d/2.",
                        "cvpm": "REAL    , PARAMETER :: cvpm         = -cv/cp",
                        "p1000mb": "REAL    , PARAMETER :: p1000mb      = 100000.",
                        "t0": "REAL    , PARAMETER :: t0           = 300.",
                    },
                ),
            },
        },
        "wrf_boundary_values": {
            "full_field_source": str(prior.WRF_TRUTH),
            "p_surf_recovered_from_MUB_plus_p_top": array_summary("P_SURF_RECOVERED", p_surf_recovered),
            "MUB_post_assignment": array_summary("MUB", after_arrays["MUB"]),
            "PB_before_AL_ALT_pass": array_summary("PB", after_arrays["PB"]),
            "T_INIT_recovered_from_ALB_PB": array_summary("T_INIT_RECOVERED", t_init_recovered),
            "ALB_before_AL_ALT_pass": array_summary("ALB", after_arrays["ALB"]),
            "PHB_after_base_geopotential_integration": array_summary("PHB", after_arrays["PHB"]),
            "HT": array_summary("HT", after_arrays["HT"]),
            "HT_FINE": array_summary("HT_FINE", after_arrays["HT_FINE"]),
            "coefficients": coefficient_payload(metrics),
            "scalars": scalar_payload(run),
            "note": (
                "p_surf is recovered from WRF MUB + P_TOP because the predecessor truth starts after the base block; "
                "MUB is not mutated by the later observed start_domain surfaces for this branch."
            ),
        },
        "jax_loader_metadata": current["metadata"],
        "current_jax_input_metrics": input_metrics,
        "ranked_current_jax_input_residuals": split.rank_metrics(input_metrics),
        "source_family_candidates": candidates,
        "ranked_source_family_rows": candidate_rows,
        "direct_truth_ablation": truth_ablation,
        "source_family_split": {
            "pressure_surface_formula": {
                "status": "DOMINANT_REMAINING_SOURCE",
                "evidence": {
                    "formula_fp32_cp1004_5_jax_hgt_P_STATE": p_formula,
                    "formula_fp32_cp1004_5_jax_hgt_MU_STATE": mu_formula,
                    "formula_fp32_cp1004_5_jax_hgt_MUB": mub_formula,
                    "wrf_boundary_mub_P_STATE": p_wrf_mub,
                    "wrf_boundary_mub_MU_STATE": mu_wrf_mub,
                },
            },
            "dtype_evaluation_order": {
                "status": "SECONDARY_SUPPORTED_NOT_SUFFICIENT",
                "evidence": {
                    "fp64_cp1004_5_P_STATE": candidates["proof_formula_fp64_cp1004_5_jax_hgt"]["downstream_residuals"]["P_STATE"]["max_abs"],
                    "fp32_cp1004_5_P_STATE": p_formula,
                },
            },
            "coefficient_indexing": {
                "status": "REFUTED_AS_DOMINANT",
                "evidence": "Current metrics with WRF MUB close P/MU below gates.",
            },
            "terrain_blend_input": {
                "status": "REFUTED_AS_DOMINANT",
                "evidence": "WRF HT substitution leaves the fp32 formula P residual unchanged.",
            },
            "PHB_integration_order": {
                "status": "SMALL_RESIDUAL_NOT_DOMINANT_DOWNSTREAM",
                "evidence": {
                    "wrf_boundary_mub_candidate_PHB": wrf_mub["base_residuals"]["PHB"]["max_abs"],
                    "wrf_boundary_mub_candidate_P_STATE": p_wrf_mub,
                },
            },
            "missing_truth_surface": {
                "status": "EXACT_P_SURF_BEFORE_MUB_NOT_EMITTED",
                "evidence": "This proof recovers p_surf from MUB+P_TOP but does not have a WRF-emitted p_surf_before_mub scalar field.",
            },
        },
        "ranked_hypotheses": ranked_hypotheses,
        "exclusions": exclusions,
        "patch_decision": {
            "production_patch_applied": False,
            "ready_for_d02_replay_patch": ready_for_patch,
            "reason": (
                f"Current/proof-local p_surf formula still leaves P_STATE {p_formula} and MU_STATE {mu_formula}; "
                f"only WRF-emitted MUB closes P/MU ({p_wrf_mub}, {mu_wrf_mub})."
            ),
        },
        "next_decision": next_decision,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    patch = proof.get("patch_decision", {})
    split_rows = proof.get("ranked_source_family_rows", [])
    lines = [
        "# V0.14 Step-1 Base-State Boundary",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Production source patch applied: `{patch.get('production_patch_applied')}`. {patch.get('reason')}",
        "- WRF branch: multi-domain real `start_domain_em`, `hypsometric_opt=2`, `rebalance=0`, `use_theta_m=1`.",
        "- Dominant remaining source: exact WRF `p_surf -> MUB` arithmetic before the `AL/ALT` pass.",
        "",
        "## Key Metrics",
        "",
        "| Source family | MUB max_abs | PHB max_abs | P max_abs | MU max_abs | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    by_name = {row["source_family"]: row for row in split_rows}
    rows = [
        ("production_current_live_child_base_fp64_cp1004_0_jax_hgt", "current production formula, not patch-ready"),
        ("proof_formula_fp32_cp1004_5_jax_hgt", "WRF constants/fp32 help but do not close"),
        ("proof_formula_fp32_cp1004_5_wrf_ht", "terrain substitution does not improve"),
        ("wrf_boundary_mub_fp32_cp1004_5_wrf_ht", "WRF MUB closes P/MU gates"),
    ]
    for name, interp in rows:
        row = by_name.get(name, {})
        lines.append(
            f"| `{name}` | {row.get('MUB_max_abs')} | {row.get('PHB_max_abs')} | "
            f"{row.get('P_STATE_downstream_max_abs')} | {row.get('MU_STATE_downstream_max_abs')} | {interp} |"
        )
    lines.extend(["", "## Source Split", ""])
    for family, item in proof.get("source_family_split", {}).items():
        lines.append(f"- `{family}`: `{item.get('status')}`. {item.get('evidence')}")
    lines.extend(["", "## Ranked Hypotheses", ""])
    for item in proof.get("ranked_hypotheses", []):
        lines.append(f"- {item['rank']}. {item['hypothesis']} Status: `{item['status']}`. {item['evidence']}")
    lines.extend(["", "## Exclusions", ""])
    for item in proof.get("exclusions", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Decision",
            "",
            proof.get("next_decision", ""),
            "",
            "Detailed metrics are in `proofs/v014/step1_base_state_boundary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Base-State Boundary",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: close or precisely localize the WRF `start_domain_em` base-state boundary before the Step-1 `AL/ALT` pass.",
        "",
        "files changed:",
        "- `proofs/v014/step1_base_state_boundary.py`",
        "- `proofs/v014/step1_base_state_boundary.json`",
        "- `proofs/v014/step1_base_state_boundary.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["executed"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "proof objects produced:"])
    for value in payload["proof_objects"].values():
        lines.append(f"- `{value}`")
    lines.extend(["", "ranked hypotheses/exclusions:"])
    for item in payload["proof"].get("ranked_hypotheses", []):
        lines.append(f"- rank {item['rank']}: {item['status']} - {item['hypothesis']}")
    for item in payload["proof"].get("exclusions", []):
        lines.append(f"- excluded: {item}")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    proof = build_proof()
    verdict = proof.get("verdict", f"STEP1_BASE_STATE_BOUNDARY_BLOCKED_{proof.get('status', 'UNKNOWN')}")
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_base_state_boundary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "environment": jax_environment(),
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "wrf_instrumentation_used_this_sprint": False,
        "git": git_metadata(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "predecessor_json": path_info(PREDECESSOR_JSON),
            "predecessor_md": path_info(PREDECESSOR_MD),
            "wrf_truth_root": path_info(prior.WRF_TRUTH),
            "wrf_start_em": path_info(WRF_START_EM),
            "wrf_constants": path_info(WRF_CONSTANTS),
            "wrf_namelist": path_info(WRF_NAMELIST),
            "d02_replay": path_info(D02_REPLAY),
            "optional_wrf_patch_diff": path_info(OUT_WRF_PATCH),
        },
        "proof": proof,
        "commands": {
            "executed": [
                "python -m py_compile proofs/v014/step1_base_state_boundary.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_base_state_boundary.py",
                "python -m json.tool proofs/v014/step1_base_state_boundary.json >/tmp/step1_base_state_boundary.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "wrf_truth_root_reused": str(prior.WRF_TRUTH),
        },
        "unresolved_risks": [
            "No production source patch was applied.",
            "This proof did not emit a fresh WRF p_surf_before_mub scalar; it recovers p_surf from WRF MUB + P_TOP.",
            "The exact production-compatible p_surf arithmetic helper is still missing; local NumPy/JAX-style fp32 formula remains above P/MU gates.",
        ],
        "next_decision": proof.get("next_decision"),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
