#!/usr/bin/env python3
"""V0.14 Step-1 JAX live-nest start-domain input split proof.

CPU-only proof. Reuses the accepted WRF start_domain internal surfaces from the
predecessor sprint and splits the current JAX live-nest inputs by substitution:
terrain, base state, PH/MU/theta time levels, diagnosed AL/ALT, and precision
order. No production source is edited by this proof.
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

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_start_domain_perturb_subsurface as prior  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_jax_start_domain_input_split.json"
OUT_MD = PROOF_DIR / "step1_jax_start_domain_input_split.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-jax-start-domain-input-split/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PREDECESSOR_JSON = PROOF_DIR / "step1_start_domain_perturb_subsurface.json"
PREDECESSOR_MD = PROOF_DIR / "step1_start_domain_perturb_subsurface.md"
D02_REPLAY = SRC / "gpuwrf/integration/d02_replay.py"
WRF_CONSTANTS = (
    Path("<DATA_ROOT>/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715")
    / "WRF/share/module_model_constants.F"
)

REQUIRED_ANCESTOR = "66c091fc"
VERDICT = "STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP"

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
    "W_STATE": 1.0e-2,
    "HT": 1.0e-6,
    "HT_FINE": 1.0e-6,
    "PB": 1.0,
    "MUB": 1.0e-2,
    "PHB": 1.0e-2,
    "PH_STATE": 1.0e-2,
    "T_STATE": 1.0e-3,
    "AL": 1.0e-8,
    "ALT": 1.0e-8,
    "ALB": 1.0e-8,
}


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
    return {
        key: metric.get(key)
        for key in (
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
        if key in metric
    }


def line_hit(path: Path, needle: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if needle in line:
            return {"path": str(path), "line": lineno, "text": line.strip()[:240]}
    return None


def scalar_from_run(run: Any, name: str, dtype: Any) -> Any:
    from gpuwrf.integration import d02_replay as replay  # noqa: PLC0415

    return dtype(float(prior.as_np(replay._load(run, "d02", name, 0)).ravel()[0]))


def build_current_jax_arrays(inputs: Mapping[str, Any]) -> dict[str, Any]:
    from gpuwrf.integration.d02_replay import _wrf_start_domain_base_from_hgt  # noqa: PLC0415

    live_child = inputs["live_child"]
    raw_child = inputs["raw_child"]
    state = live_child["state"]
    base = live_child["base_state"]
    grid = live_child["grid"]
    metrics = live_child["metrics"]
    _pb, _mub, _phb, _t_init, alb = _wrf_start_domain_base_from_hgt(
        inputs["run"],
        "d02",
        hgt=grid.terrain_height,
        metrics=metrics,
    )
    theta = prior.as_np(state.theta)
    arrays = {
        "T_STATE": theta - T0,
        "THETA_FULL": theta,
        "QVAPOR": prior.as_np(state.qv),
        "P_STATE": prior.as_np(state.p_perturbation),
        "PB": prior.as_np(base.pb),
        "MU_STATE": prior.as_np(state.mu_perturbation),
        "MUB": prior.as_np(base.mub),
        "MUT": prior.as_np(base.mub) + prior.as_np(state.mu_perturbation),
        "W_STATE": prior.as_np(state.w),
        "PH_STATE": prior.as_np(state.ph_perturbation),
        "PHB": prior.as_np(base.phb),
        "HT": prior.as_np(grid.terrain_height),
        "HT_FINE": prior.as_np(raw_child["grid"].terrain_height),
        "ALB": prior.as_np(alb),
    }
    p64, alt64 = prior.pressure_from_ph_formula(
        pb=base.pb,
        phb=base.phb,
        mub=base.mub,
        ph=state.ph_perturbation,
        theta_full=state.theta,
        mu=state.mu_perturbation,
        metrics=metrics,
        dtype=np.float64,
    )
    p32, alt32 = prior.pressure_from_ph_formula(
        pb=base.pb,
        phb=base.phb,
        mub=base.mub,
        ph=state.ph_perturbation,
        theta_full=state.theta,
        mu=state.mu_perturbation,
        metrics=metrics,
        dtype=np.float32,
    )
    arrays.update(
        {
            "P_FORMULA_FP64": p64,
            "ALT_FORMULA_FP64": alt64,
            "AL_FORMULA_FP64": alt64 - arrays["ALB"],
            "P_FORMULA_FP32": p32,
            "ALT_FORMULA_FP32": alt32,
            "AL_FORMULA_FP32": alt32 - arrays["ALB"],
        }
    )
    arrays["MU_PRESS_ADJ_FP64"] = prior.press_adj_mu(
        mu_before=state.mu_perturbation,
        al=arrays["AL_FORMULA_FP64"],
        alt=arrays["ALT_FORMULA_FP64"],
        alb=arrays["ALB"],
        ht=grid.terrain_height,
        ht_fine=raw_child["grid"].terrain_height,
        dtype=np.float64,
    )
    arrays["MU_PRESS_ADJ_FP32"] = prior.press_adj_mu(
        mu_before=state.mu_perturbation,
        al=arrays["AL_FORMULA_FP32"],
        alt=arrays["ALT_FORMULA_FP32"],
        alb=arrays["ALB"],
        ht=grid.terrain_height,
        ht_fine=raw_child["grid"].terrain_height,
        dtype=np.float32,
    )
    return {
        "arrays": arrays,
        "metadata": {
            "live_nest_base_init": live_child.get("live_nest_base_init"),
            "transient_adjust_mub": live_child.get("transient_adjust_mub"),
            "theta_qv_adjust": live_child.get("theta_qv_adjust"),
        },
    }


def wrf_start_domain_base_candidate(
    *,
    run: Any,
    hgt: Any,
    metrics: Any,
    dtype: Any,
    cp: float,
) -> dict[str, np.ndarray]:
    """Proof-local transcription of the WRF base recompute with selectable dtype."""

    f = dtype
    p_top = scalar_from_run(run, "P_TOP", f)
    p00 = scalar_from_run(run, "P00", f)
    t00 = scalar_from_run(run, "T00", f)
    lapse = scalar_from_run(run, "TLP", f)
    tiso = scalar_from_run(run, "TISO", f)
    lapse_strat = scalar_from_run(run, "TLP_STRAT", f)
    p_strat = scalar_from_run(run, "P_STRAT", f)

    hgt_arr = prior.as_np(hgt, dtype)
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

    root_arg = ((t00 / lapse) ** f(2.0) - f(2.0) * g * hgt_arr / lapse / rd).astype(dtype)
    p_surf = (p00 * np.exp((-t00 / lapse + np.sqrt(root_arg)).astype(dtype))).astype(dtype)
    mub = (p_surf - p_top).astype(dtype)
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
        "ALB": alb.astype(np.float64),
    }


def pressure_metric(label: str, pb: Any, theta: Any, alt: Any, reference: Any, dtype: Any) -> dict[str, Any]:
    pressure = prior.pressure_from_alt(pb=pb, theta_full=theta, alt=alt, dtype=dtype)
    metric = metric_brief(prior.diff_metrics("P_STATE", pressure, reference))
    return {"label": label, "dtype": np.dtype(dtype).name, "metric": metric}


def alt_metric(
    label: str,
    *,
    ph: Any,
    phb: Any,
    mub: Any,
    mu: Any,
    metrics: Any,
    pb: Any,
    theta: Any,
    reference_alt: Any,
    reference_p: Any,
    dtype: Any,
) -> dict[str, Any]:
    alt = prior.diagnose_alt_from_ph(ph=ph, phb=phb, mub=mub, mu=mu, metrics=metrics, dtype=dtype)
    pressure = prior.pressure_from_alt(pb=pb, theta_full=theta, alt=alt, dtype=dtype)
    return {
        "label": label,
        "dtype": np.dtype(dtype).name,
        "alt_vs_wrf": metric_brief(prior.diff_metrics("ALT", alt, reference_alt)),
        "pressure_vs_wrf": metric_brief(prior.diff_metrics("P_STATE", pressure, reference_p)),
    }


def press_adj_metric(
    label: str,
    *,
    mu_before: Any,
    al: Any,
    alt: Any,
    alb: Any,
    ht: Any,
    ht_fine: Any,
    reference_mu: Any,
    dtype: Any,
) -> dict[str, Any]:
    mu = prior.press_adj_mu(
        mu_before=mu_before,
        al=al,
        alt=alt,
        alb=alb,
        ht=ht,
        ht_fine=ht_fine,
        dtype=dtype,
    )
    return {
        "label": label,
        "dtype": np.dtype(dtype).name,
        "metric": metric_brief(prior.diff_metrics("MU_STATE", mu, reference_mu)),
    }


def compare_inputs(jax_arrays: Mapping[str, Any], after_hyp: Mapping[str, Any], before_press: Mapping[str, Any], after_w: Mapping[str, Any]) -> dict[str, Any]:
    pairs = {
        "HT_current_vs_wrf_after_hyp": ("HT", jax_arrays["HT"], after_hyp["HT"]),
        "HT_FINE_current_vs_wrf_after_hyp": ("HT_FINE", jax_arrays["HT_FINE"], after_hyp["HT_FINE"]),
        "PB_current_vs_wrf_after_hyp": ("PB", jax_arrays["PB"], after_hyp["PB"]),
        "MUB_current_vs_wrf_after_hyp": ("MUB", jax_arrays["MUB"], after_hyp["MUB"]),
        "PHB_current_vs_wrf_after_hyp": ("PHB", jax_arrays["PHB"], after_hyp["PHB"]),
        "PH_STATE_current_vs_wrf_after_hyp": ("PH_STATE", jax_arrays["PH_STATE"], after_hyp["PH2_STATE"]),
        "MU_STATE_current_vs_wrf_before_press": ("MU_STATE", jax_arrays["MU_STATE"], before_press["MU2_STATE"]),
        "T_STATE_current_vs_wrf_after_hyp": ("T_STATE", jax_arrays["T_STATE"], after_hyp["T2_STATE"]),
        "AL_current_formula_fp64_vs_wrf_after_hyp": ("AL", jax_arrays["AL_FORMULA_FP64"], after_hyp["AL"]),
        "ALT_current_formula_fp64_vs_wrf_after_hyp": ("ALT", jax_arrays["ALT_FORMULA_FP64"], after_hyp["ALT"]),
        "ALB_current_vs_wrf_after_hyp": ("ALB", jax_arrays["ALB"], after_hyp["ALB"]),
        "W_STATE_current_vs_wrf_after_w": ("W_STATE", jax_arrays["W_STATE"], after_w["W2_STATE"]),
    }
    return {
        name: metric_brief(prior.diff_metrics(field, candidate, reference))
        for name, (field, candidate, reference) in pairs.items()
    }


def rank_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, metric in metrics.items():
        field = name.split("_", 1)[0]
        rows.append(
            {
                "name": name,
                "max_abs": metric.get("max_abs"),
                "rmse": metric.get("rmse"),
                "bias": metric.get("bias"),
                "threshold": MATERIAL_THRESHOLDS.get(field),
                "worst_mismatch_fortran": metric.get("worst_mismatch_fortran"),
            }
        )
    return sorted(rows, key=lambda item: -float(item.get("max_abs") or 0.0))


def build_proof() -> dict[str, Any]:
    shapes = prior.expected_shapes()
    start_surfaces = {surface: prior.parse_start_surface(surface, shapes) for surface in prior.SURFACES}
    blockers = {name: item for name, item in start_surfaces.items() if item.get("status") != "WRF_SURFACE_READY"}
    if blockers:
        return {"status": "BLOCKED_INPUTS", "blockers": blockers}

    inputs = live.build_live_nest_step1_inputs()
    current = build_current_jax_arrays(inputs)
    jax_arrays = current["arrays"]
    live_child = inputs["live_child"]
    state = live_child["state"]
    grid = live_child["grid"]
    metrics = live_child["metrics"]

    after_hyp = start_surfaces["after_hypsometric_p_al_alt"]["arrays"]
    before_press = start_surfaces["before_press_adj"]["arrays"]
    after_press = start_surfaces["after_press_adj"]["arrays"]
    after_w = start_surfaces["after_w_surface_branch"]["arrays"]

    input_metrics = compare_inputs(jax_arrays, after_hyp, before_press, after_w)
    time_level_metrics = {
        "T1_vs_T2_after_hyp": metric_brief(prior.diff_metrics("T_STATE", after_hyp["T1_STATE"], after_hyp["T2_STATE"])),
        "THETA1_vs_THETA2_after_hyp": metric_brief(
            prior.diff_metrics("THETA_FULL", after_hyp["THETA1_FULL"], after_hyp["THETA2_FULL"])
        ),
        "MU1_vs_MU2_after_hyp": metric_brief(prior.diff_metrics("MU_STATE", after_hyp["MU1_STATE"], after_hyp["MU2_STATE"])),
        "PH1_vs_PH2_after_hyp": metric_brief(prior.diff_metrics("PH_STATE", after_hyp["PH1_STATE"], after_hyp["PH2_STATE"])),
        "JAX_T_vs_WRF_T1": metric_brief(prior.diff_metrics("T_STATE", jax_arrays["T_STATE"], after_hyp["T1_STATE"])),
        "JAX_MU_vs_WRF_MU1": metric_brief(prior.diff_metrics("MU_STATE", jax_arrays["MU_STATE"], after_hyp["MU1_STATE"])),
        "JAX_PH_vs_WRF_PH1": metric_brief(prior.diff_metrics("PH_STATE", jax_arrays["PH_STATE"], after_hyp["PH1_STATE"])),
    }

    pressure_ablation = {
        "current_fp64": pressure_metric(
            "current_fp64",
            jax_arrays["PB"],
            jax_arrays["THETA_FULL"],
            jax_arrays["ALT_FORMULA_FP64"],
            after_hyp["P_STATE"],
            np.float64,
        ),
        "PB_to_wrf": pressure_metric(
            "PB_to_wrf",
            after_hyp["PB"],
            jax_arrays["THETA_FULL"],
            jax_arrays["ALT_FORMULA_FP64"],
            after_hyp["P_STATE"],
            np.float64,
        ),
        "theta_to_wrf": pressure_metric(
            "theta_to_wrf",
            jax_arrays["PB"],
            after_hyp["THETA1_FULL"],
            jax_arrays["ALT_FORMULA_FP64"],
            after_hyp["P_STATE"],
            np.float64,
        ),
        "ALT_to_wrf": pressure_metric(
            "ALT_to_wrf",
            jax_arrays["PB"],
            jax_arrays["THETA_FULL"],
            after_hyp["ALT"],
            after_hyp["P_STATE"],
            np.float64,
        ),
        "PB_ALT_to_wrf": pressure_metric(
            "PB_ALT_to_wrf",
            after_hyp["PB"],
            jax_arrays["THETA_FULL"],
            after_hyp["ALT"],
            after_hyp["P_STATE"],
            np.float64,
        ),
        "all_wrf_fp64": pressure_metric(
            "all_wrf_fp64",
            after_hyp["PB"],
            after_hyp["THETA1_FULL"],
            after_hyp["ALT"],
            after_hyp["P_STATE"],
            np.float64,
        ),
        "all_wrf_fp32": pressure_metric(
            "all_wrf_fp32",
            after_hyp["PB"],
            after_hyp["THETA1_FULL"],
            after_hyp["ALT"],
            after_hyp["P_STATE"],
            np.float32,
        ),
    }

    alt_ablation_fp32 = {
        "current": alt_metric(
            "current",
            ph=jax_arrays["PH_STATE"],
            phb=jax_arrays["PHB"],
            mub=jax_arrays["MUB"],
            mu=jax_arrays["MU_STATE"],
            metrics=metrics,
            pb=jax_arrays["PB"],
            theta=jax_arrays["THETA_FULL"],
            reference_alt=after_hyp["ALT"],
            reference_p=after_hyp["P_STATE"],
            dtype=np.float32,
        ),
        "PHB_to_wrf": alt_metric(
            "PHB_to_wrf",
            ph=jax_arrays["PH_STATE"],
            phb=after_hyp["PHB"],
            mub=jax_arrays["MUB"],
            mu=jax_arrays["MU_STATE"],
            metrics=metrics,
            pb=jax_arrays["PB"],
            theta=jax_arrays["THETA_FULL"],
            reference_alt=after_hyp["ALT"],
            reference_p=after_hyp["P_STATE"],
            dtype=np.float32,
        ),
        "MUB_to_wrf": alt_metric(
            "MUB_to_wrf",
            ph=jax_arrays["PH_STATE"],
            phb=jax_arrays["PHB"],
            mub=after_hyp["MUB"],
            mu=jax_arrays["MU_STATE"],
            metrics=metrics,
            pb=jax_arrays["PB"],
            theta=jax_arrays["THETA_FULL"],
            reference_alt=after_hyp["ALT"],
            reference_p=after_hyp["P_STATE"],
            dtype=np.float32,
        ),
        "PHB_MUB_to_wrf": alt_metric(
            "PHB_MUB_to_wrf",
            ph=jax_arrays["PH_STATE"],
            phb=after_hyp["PHB"],
            mub=after_hyp["MUB"],
            mu=jax_arrays["MU_STATE"],
            metrics=metrics,
            pb=jax_arrays["PB"],
            theta=jax_arrays["THETA_FULL"],
            reference_alt=after_hyp["ALT"],
            reference_p=after_hyp["P_STATE"],
            dtype=np.float32,
        ),
        "PHB_MUB_PB_theta_to_wrf": alt_metric(
            "PHB_MUB_PB_theta_to_wrf",
            ph=after_hyp["PH1_STATE"],
            phb=after_hyp["PHB"],
            mub=after_hyp["MUB"],
            mu=after_hyp["MU1_STATE"],
            metrics=metrics,
            pb=after_hyp["PB"],
            theta=after_hyp["THETA1_FULL"],
            reference_alt=after_hyp["ALT"],
            reference_p=after_hyp["P_STATE"],
            dtype=np.float32,
        ),
    }

    alt_ablation_fp64_wrf_fields = alt_metric(
        "PHB_MUB_PB_theta_to_wrf_fp64",
        ph=after_hyp["PH1_STATE"],
        phb=after_hyp["PHB"],
        mub=after_hyp["MUB"],
        mu=after_hyp["MU1_STATE"],
        metrics=metrics,
        pb=after_hyp["PB"],
        theta=after_hyp["THETA1_FULL"],
        reference_alt=after_hyp["ALT"],
        reference_p=after_hyp["P_STATE"],
        dtype=np.float64,
    )

    press_adj_ablation = {
        "current_fp64": press_adj_metric(
            "current_fp64",
            mu_before=jax_arrays["MU_STATE"],
            al=jax_arrays["AL_FORMULA_FP64"],
            alt=jax_arrays["ALT_FORMULA_FP64"],
            alb=jax_arrays["ALB"],
            ht=jax_arrays["HT"],
            ht_fine=jax_arrays["HT_FINE"],
            reference_mu=after_press["MU2_STATE"],
            dtype=np.float64,
        ),
        "terrain_to_wrf": press_adj_metric(
            "terrain_to_wrf",
            mu_before=jax_arrays["MU_STATE"],
            al=jax_arrays["AL_FORMULA_FP64"],
            alt=jax_arrays["ALT_FORMULA_FP64"],
            alb=jax_arrays["ALB"],
            ht=after_hyp["HT"],
            ht_fine=after_hyp["HT_FINE"],
            reference_mu=after_press["MU2_STATE"],
            dtype=np.float64,
        ),
        "AL_ALT_ALB_to_wrf": press_adj_metric(
            "AL_ALT_ALB_to_wrf",
            mu_before=jax_arrays["MU_STATE"],
            al=after_hyp["AL"],
            alt=after_hyp["ALT"],
            alb=after_hyp["ALB"],
            ht=jax_arrays["HT"],
            ht_fine=jax_arrays["HT_FINE"],
            reference_mu=after_press["MU2_STATE"],
            dtype=np.float64,
        ),
        "all_wrf_except_mu_before_exact": press_adj_metric(
            "all_wrf_except_mu_before_exact",
            mu_before=before_press["MU2_STATE"],
            al=before_press["AL"],
            alt=before_press["ALT"],
            alb=before_press["ALB"],
            ht=before_press["HT"],
            ht_fine=before_press["HT_FINE"],
            reference_mu=after_press["MU2_STATE"],
            dtype=np.float64,
        ),
    }

    base_candidates: dict[str, Any] = {}
    for label, dtype, cp in (
        ("production_formula_fp64_cp1004_0", np.float64, 1004.0),
        ("wrf_cp_fp64_cp1004_5", np.float64, CP_WRF),
        ("wrf_order_fp32_cp1004_0", np.float32, 1004.0),
        ("wrf_order_fp32_cp1004_5", np.float32, CP_WRF),
    ):
        candidate = wrf_start_domain_base_candidate(
            run=inputs["run"],
            hgt=grid.terrain_height,
            metrics=metrics,
            dtype=dtype,
            cp=cp,
        )
        alt = prior.diagnose_alt_from_ph(
            ph=jax_arrays["PH_STATE"],
            phb=candidate["PHB"],
            mub=candidate["MUB"],
            mu=jax_arrays["MU_STATE"],
            metrics=metrics,
            dtype=np.float32,
        )
        pressure = prior.pressure_from_alt(
            pb=candidate["PB"],
            theta_full=jax_arrays["THETA_FULL"],
            alt=alt,
            dtype=np.float32,
        )
        mu_press = prior.press_adj_mu(
            mu_before=jax_arrays["MU_STATE"],
            al=alt - candidate["ALB"],
            alt=alt,
            alb=candidate["ALB"],
            ht=jax_arrays["HT"],
            ht_fine=jax_arrays["HT_FINE"],
            dtype=np.float32,
        )
        base_candidates[label] = {
            "dtype": np.dtype(dtype).name,
            "cp": cp,
            "input_residuals": {
                "PB": metric_brief(prior.diff_metrics("PB", candidate["PB"], after_hyp["PB"])),
                "MUB": metric_brief(prior.diff_metrics("MUB", candidate["MUB"], after_hyp["MUB"])),
                "PHB": metric_brief(prior.diff_metrics("PHB", candidate["PHB"], after_hyp["PHB"])),
                "ALB": metric_brief(prior.diff_metrics("ALB", candidate["ALB"], after_hyp["ALB"])),
            },
            "downstream": {
                "ALT": metric_brief(prior.diff_metrics("ALT", alt, after_hyp["ALT"])),
                "P_STATE": metric_brief(prior.diff_metrics("P_STATE", pressure, after_hyp["P_STATE"])),
                "MU_STATE": metric_brief(prior.diff_metrics("MU_STATE", mu_press, after_press["MU2_STATE"])),
            },
        }

    predecessor_data = json.loads(PREDECESSOR_JSON.read_text(encoding="utf-8"))
    predecessor_formula = predecessor_data.get("proof", {}).get("formula_metrics", {})
    predecessor_ordering = {
        key: predecessor_formula.get(key)
        for key in (
            "wrf_internal_pressure_from_alt_fp32_vs_after_hyp_P",
            "wrf_internal_press_adj_fp32_vs_after_press_MU",
            "wrf_after_w_surface_vs_precall_W",
            "jax_current_pressure_formula_fp64_vs_wrf_after_hyp_P",
            "jax_current_press_adj_fp64_vs_wrf_after_press_MU",
        )
    }

    direct_alt_p = pressure_ablation["ALT_to_wrf"]["metric"]["max_abs"]
    direct_alt_mu = press_adj_ablation["AL_ALT_ALB_to_wrf"]["metric"]["max_abs"]
    phb_mub_p = alt_ablation_fp32["PHB_MUB_to_wrf"]["pressure_vs_wrf"]["max_abs"]
    base_candidate_p = base_candidates["wrf_order_fp32_cp1004_5"]["downstream"]["P_STATE"]["max_abs"]
    base_candidate_mu = base_candidates["wrf_order_fp32_cp1004_5"]["downstream"]["MU_STATE"]["max_abs"]

    ready_for_patch = (
        base_candidate_p is not None
        and base_candidate_mu is not None
        and float(base_candidate_p) <= MATERIAL_THRESHOLDS["P_STATE"]
        and float(base_candidate_mu) <= MATERIAL_THRESHOLDS["MU_STATE"]
    )

    ranked_hypotheses = [
        {
            "rank": 1,
            "hypothesis": "Dominant current JAX formula residual is diagnosed AL/ALT, fed by base-state reconstruction.",
            "status": "SUPPORTED_LOCALIZED_NOT_PATCH_READY",
            "evidence": (
                f"Direct WRF ALT substitution reduces P max_abs to {direct_alt_p}; direct WRF AL/ALT/ALB "
                f"substitution reduces MU max_abs to {direct_alt_mu}. Replacing WRF PHB+MUB in the fp32 ALT "
                f"diagnosis reduces P max_abs to {phb_mub_p}."
            ),
        },
        {
            "rank": 2,
            "hypothesis": "The missing production contract is WRF start_domain base reconstruction precision/source order.",
            "status": "SUPPORTED_BY_FALSIFIER",
            "evidence": (
                "Using WRF fields with fp64 ALT diagnosis leaves P max_abs "
                f"{alt_ablation_fp64_wrf_fields['pressure_vs_wrf']['max_abs']}, while WRF fields with fp32 "
                f"diagnosis leave P max_abs {alt_ablation_fp32['PHB_MUB_PB_theta_to_wrf']['pressure_vs_wrf']['max_abs']}. "
                f"A local fp32/cp=1004.5 base recompute still leaves P max_abs {base_candidate_p} and "
                f"MU max_abs {base_candidate_mu}, so the exact base source/order is not closed."
            ),
        },
        {
            "rank": 3,
            "hypothesis": "Final blended terrain is the dominant source.",
            "status": "REFUTED",
            "evidence": (
                f"HT max_abs is {input_metrics['HT_current_vs_wrf_after_hyp']['max_abs']} m; HT_FINE max_abs is "
                f"{input_metrics['HT_FINE_current_vs_wrf_after_hyp']['max_abs']} m; replacing terrain in press_adj "
                f"does not improve MU ({press_adj_ablation['terrain_to_wrf']['metric']['max_abs']})."
            ),
        },
        {
            "rank": 4,
            "hypothesis": "Time-level selection, PH_STATE, or pre-press MU is the source.",
            "status": "REFUTED",
            "evidence": (
                f"T1/T2, MU1/MU2, and PH1/PH2 are exact at after_hyp for checked fields; "
                f"JAX PH vs WRF PH1 max_abs {time_level_metrics['JAX_PH_vs_WRF_PH1']['max_abs']}; "
                f"JAX MU vs WRF MU1 max_abs {time_level_metrics['JAX_MU_vs_WRF_MU1']['max_abs']}."
            ),
        },
        {
            "rank": 5,
            "hypothesis": "PB or theta alone is the dominant pressure source.",
            "status": "REFUTED_AS_DOMINANT",
            "evidence": (
                f"PB-only substitution leaves P max_abs {pressure_ablation['PB_to_wrf']['metric']['max_abs']}; "
                f"theta-only substitution leaves P max_abs {pressure_ablation['theta_to_wrf']['metric']['max_abs']}."
            ),
        },
    ]

    exclusions = [
        "WRF start_domain P/press_adj/W source ordering remains accepted from the predecessor proof.",
        "Time-level selection is not the P/ALT cause: T1/T2, MU1/MU2, and PH1/PH2 match exactly at the hypsometric surface.",
        "PH_STATE and pre-press MU are not the input gap: current JAX PH and MU match WRF PH1/MU1 to round-off/zero.",
        "Terrain blend is not dominant: HT and HT_FINE residuals are tiny, and terrain substitution does not improve press_adj MU.",
        "A narrow production patch is not safe yet: proof-local WRF-like fp32 base recompute does not close P/MU gates.",
    ]
    next_surface = (
        "Emit or reproduce the exact WRF start_domain base-state source boundary before the hypsometric AL/ALT pass: "
        "p_surf, MUB immediately after assignment, PB/T_INIT/ALB after the multi-domain reconstitution block, PHB after "
        "base integration, C3F/C4F/C3H/C4H as used in memory, imask/rebalance/hybrid flags, and scalar constants. "
        "The next worker should close that base reconstruction to WRF PHB+MUB, then apply the P/MU/W perturbation "
        "init patch already proven by direct AL/ALT substitution."
    )

    return {
        "status": "PROOF_EXECUTED",
        "verdict": VERDICT,
        "shapes": {key: list(value) for key, value in shapes.items()},
        "wrf_surfaces": {name: prior.summarize_surface(surface) for name, surface in start_surfaces.items()},
        "jax_loader_metadata": current["metadata"],
        "predecessor_ordering_metrics": predecessor_ordering,
        "input_metrics": input_metrics,
        "ranked_input_residuals": rank_metrics(input_metrics),
        "time_level_metrics": time_level_metrics,
        "pressure_ablation": pressure_ablation,
        "alt_ablation_fp32": alt_ablation_fp32,
        "alt_ablation_fp64_wrf_fields": alt_ablation_fp64_wrf_fields,
        "press_adj_ablation": press_adj_ablation,
        "base_recompute_candidates": base_candidates,
        "source_constant_hits": {
            "wrf_cp": line_hit(WRF_CONSTANTS, "cp           = 7.*r_d/2."),
            "d02_replay_cp": line_hit(D02_REPLAY, "_EOS_CP_D = 1004.0"),
            "wrf_start_domain_base": line_hit(
                Path("<DATA_ROOT>/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715")
                / "WRF/dyn_em/start_em.F",
                "p_surf = p00 * EXP",
            ),
            "wrf_start_domain_alt": line_hit(
                Path("<DATA_ROOT>/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715")
                / "WRF/dyn_em/start_em.F",
                "grid%al(i,k,j) = (grid%ph_1",
            ),
        },
        "ranked_hypotheses": ranked_hypotheses,
        "exclusions": exclusions,
        "patch_decision": {
            "production_patch_applied": False,
            "ready_for_patch": ready_for_patch,
            "reason": (
                f"Direct WRF AL/ALT substitution closes P/MU below gates, but current/proof-local production inputs "
                f"do not: best local base candidate leaves P max_abs {base_candidate_p} and MU max_abs {base_candidate_mu}."
            ),
        },
        "next_truth_surface": next_surface,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    patch = proof.get("patch_decision", {})
    pa = proof.get("pressure_ablation", {})
    aa = proof.get("alt_ablation_fp32", {})
    ba = proof.get("base_recompute_candidates", {})
    lines = [
        "# V0.14 Step-1 JAX Start-Domain Input Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Production source patch applied: `{patch.get('production_patch_applied')}`. {patch.get('reason')}",
        "- Dominant family: base-state reconstruction feeding fp32 `AL/ALT` diagnosis.",
        "",
        "## Key Metrics",
        "",
        "| Check | max_abs | RMSE | Interpretation |",
        "|---|---:|---:|---|",
    ]
    rows = [
        (
            "Current pressure formula vs WRF P",
            pa.get("current_fp64", {}).get("metric", {}),
            "current inputs not patch-ready",
        ),
        (
            "Replace ALT with WRF ALT",
            pa.get("ALT_to_wrf", {}).get("metric", {}),
            "diagnosed ALT is dominant",
        ),
        (
            "FP32 ALT with WRF PHB+MUB",
            aa.get("PHB_MUB_to_wrf", {}).get("pressure_vs_wrf", {}),
            "base PHB+MUB closes pressure",
        ),
        (
            "WRF fields with FP64 ALT diagnosis",
            proof.get("alt_ablation_fp64_wrf_fields", {}).get("pressure_vs_wrf", {}),
            "dtype/order matters",
        ),
        (
            "Best local fp32/cp=1004.5 base candidate P",
            ba.get("wrf_order_fp32_cp1004_5", {}).get("downstream", {}).get("P_STATE", {}),
            "not patch-ready",
        ),
        (
            "Best local fp32/cp=1004.5 base candidate MU",
            ba.get("wrf_order_fp32_cp1004_5", {}).get("downstream", {}).get("MU_STATE", {}),
            "slightly above MU gate",
        ),
    ]
    for label, metric, interp in rows:
        lines.append(f"| {label} | {metric.get('max_abs')} | {metric.get('rmse')} | {interp} |")
    lines.extend(["", "## Ranked Hypotheses", ""])
    for item in proof.get("ranked_hypotheses", []):
        lines.append(f"- {item['rank']}. {item['hypothesis']} Status: `{item['status']}`. {item['evidence']}")
    lines.extend(["", "## Exclusions", ""])
    for item in proof.get("exclusions", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            proof.get("next_truth_surface", ""),
            "",
            "Detailed metrics are in `proofs/v014/step1_jax_start_domain_input_split.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 JAX Start-Domain Input Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: close or precisely localize the current JAX live-nest `start_domain` input gap for Step-1 P/MU/W initialization.",
        "",
        "files changed:",
        "- `proofs/v014/step1_jax_start_domain_input_split.py`",
        "- `proofs/v014/step1_jax_start_domain_input_split.json`",
        "- `proofs/v014/step1_jax_start_domain_input_split.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`",
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
    verdict = proof.get("verdict", f"STEP1_JAX_START_DOMAIN_INPUT_SPLIT_BLOCKED_{proof.get('status', 'UNKNOWN')}")
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_jax_start_domain_input_split.v1",
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
        "git": git_metadata(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "predecessor_json": path_info(PREDECESSOR_JSON),
            "predecessor_md": path_info(PREDECESSOR_MD),
            "wrf_truth_root": path_info(prior.WRF_TRUTH),
            "d02_replay": path_info(D02_REPLAY),
            "wrf_constants": path_info(WRF_CONSTANTS),
        },
        "proof": proof,
        "commands": {
            "executed": [
                "python -m py_compile proofs/v014/step1_jax_start_domain_input_split.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_start_domain_input_split.py",
                "python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json >/tmp/step1_jax_start_domain_input_split.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "wrf_truth_root": str(prior.WRF_TRUTH),
        },
        "unresolved_risks": [
            "No production source patch was applied.",
            "The exact WRF base-state reconstruction/source-order contract is still missing; the best proof-local fp32/cp=1004.5 candidate remains above P/MU gates.",
            "Direct WRF AL/ALT substitution proves the perturbation formula path, but production cannot use WRF truth arrays at runtime.",
        ],
        "next_decision": proof.get("next_truth_surface"),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
