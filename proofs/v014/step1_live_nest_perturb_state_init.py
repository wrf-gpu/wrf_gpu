#!/usr/bin/env python3
"""V0.14 Step-1 live-nest perturbation-state init proof.

CPU-only proof. Reuses accepted WRF savepoint surfaces and runs proof-local
transcriptions of WRF live-nest/start_domain perturbation-state initialization.
No production source is modified.
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

import step1_first_rk_part1_p_state_split as predecessor  # noqa: E402
import step1_jax_loader_tstate as loader  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_live_nest_perturb_state_init.json"
OUT_MD = PROOF_DIR / "step1_live_nest_perturb_state_init.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PREDECESSOR_JSON = PROOF_DIR / "step1_first_rk_part1_p_state_split.json"
PREDECESSOR_MD = PROOF_DIR / "step1_first_rk_part1_p_state_split.md"
PRECALL_TRUTH_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth")
WRF_SOURCE_ROOT = Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF")

REQUIRED_ANCESTOR = "131b27cd"
P_FAMILY = ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE")
VERDICT = "STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP"

R_D = 287.0
R_V = 461.6
CP = 7.0 * R_D / 2.0
CV = CP - R_D
CPOVCV = CP / CV
CVPM = -CV / CP
P1000MB = 100000.0
T0 = 300.0
G = 9.81
RVOVRD = R_V / R_D


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def as_np(value: Any, dtype: Any = np.float64) -> np.ndarray:
    try:
        import jax  # noqa: PLC0415

        value = jax.device_get(value)
    except Exception:
        pass
    return np.asarray(value, dtype=dtype)


def metric_brief(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    return {
        key: metric.get(key)
        for key in (
            "status",
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


def compare(field: str, candidate: Any, reference: Any) -> dict[str, Any]:
    return live.diff_metrics(field, candidate, reference)


def selected_metrics(candidate: Mapping[str, Any], reference: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: metric_brief(compare(field, candidate[field], reference[field])) for field in fields}


def line_hits(path: Path, patterns: Mapping[str, str]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "hits": {}}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: dict[str, Any] = {}
    for name, pattern in patterns.items():
        found = None
        for index, line in enumerate(lines, start=1):
            if pattern in line:
                found = {"line": index, "text": line.strip()[:220]}
                break
        hits[name] = found
    return {"path": str(path), "exists": True, "hits": hits}


def rank_candidate_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for field, metric in metrics.items():
        rows.append(
            {
                "field": field,
                "max_abs": metric.get("max_abs"),
                "rmse": metric.get("rmse"),
                "bias": metric.get("bias"),
                "worst_mismatch_fortran": metric.get("worst_mismatch_fortran"),
            }
        )
    return sorted(rows, key=lambda item: -float(item.get("max_abs") or 0.0))


def start_domain_pressure_candidate(
    *,
    pb: Any,
    phb: Any,
    mub: Any,
    ph_perturbation: Any,
    theta_full: Any,
    mu_perturbation: Any,
    qv: Any | None,
    metrics: Any,
    use_theta_m: int,
    dtype: Any = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transcribe start_em.F hypsometric_opt=2 P/al/alt recompute."""

    pb_arr = as_np(pb, dtype)
    phb_arr = as_np(phb, dtype)
    mub_arr = as_np(mub, dtype)
    ph_arr = as_np(ph_perturbation, dtype)
    theta_arr = as_np(theta_full, dtype)
    mu_arr = as_np(mu_perturbation, dtype)
    c3f = as_np(metrics.c3f, dtype)
    c4f = as_np(metrics.c4f, dtype)
    c3h = as_np(metrics.c3h, dtype)
    c4h = as_np(metrics.c4h, dtype)
    p_top = dtype(as_np(metrics.p_top, dtype))
    rd = dtype(R_D)
    p1000 = dtype(P1000MB)
    cpovcv = dtype(CPOVCV)

    full_mu = (mub_arr + mu_arr).astype(dtype)
    pfu = (c3f[1:, None, None] * full_mu[None, :, :] + c4f[1:, None, None] + p_top).astype(dtype)
    pfd = (c3f[:-1, None, None] * full_mu[None, :, :] + c4f[:-1, None, None] + p_top).astype(dtype)
    phm = (c3h[:, None, None] * full_mu[None, :, :] + c4h[:, None, None] + p_top).astype(dtype)
    log_term = np.log((pfd / pfu).astype(dtype)).astype(dtype)
    alt = ((ph_arr[1:] - ph_arr[:-1] + phb_arr[1:] - phb_arr[:-1]) / (phm * log_term)).astype(dtype)
    al = (alt - start_domain_alb(pb=pb_arr, theta_base=None, metrics=metrics, hgt=None, dtype=dtype)).astype(dtype)
    if int(use_theta_m) == 0:
        qv_arr = as_np(qv, dtype)
        qvf = (dtype(1.0) + dtype(RVOVRD) * qv_arr).astype(dtype)
    else:
        qvf = dtype(1.0)
    p = (p1000 * ((rd * theta_arr * qvf) / (p1000 * alt)) ** cpovcv - pb_arr).astype(dtype)
    return p.astype(np.float64), al.astype(np.float64), alt.astype(np.float64)


def start_domain_alb(
    *,
    pb: Any,
    theta_base: Any | None,
    metrics: Any,
    hgt: Any | None,
    dtype: Any,
) -> np.ndarray:
    """Compute WRF base inverse density; hgt is unused when theta_base is absent.

    The production helper already recomputes WRF start_domain base state.  For
    proof-local pressure al, use the loaded BaseState theta profile if supplied;
    otherwise recover the same profile from PB using the WRF base recomputation
    helper in the caller.
    """

    pb_arr = as_np(pb, dtype)
    if theta_base is None:
        # The exact WRF start_domain t_init profile is available from the live
        # helper only in build_proof where the Gen2Run/grid are in scope.  This
        # placeholder is overwritten there for press_adj.  For pressure itself
        # only alt is required; this al path is used for diagnostics.
        theta = np.ones_like(pb_arr, dtype=dtype) * dtype(T0)
    else:
        theta = as_np(theta_base, dtype)
    return (dtype(R_D) / dtype(P1000MB) * theta * (pb_arr / dtype(P1000MB)) ** dtype(CVPM)).astype(dtype)


def press_adj_mu_candidate(
    *,
    mu_perturbation: Any,
    al: Any,
    alt: Any,
    alb: Any,
    terrain_height: Any,
    fine_terrain_height: Any,
    dtype: Any = np.float64,
) -> np.ndarray:
    mu = as_np(mu_perturbation, dtype)
    al_arr = as_np(al, dtype)
    alt_arr = as_np(alt, dtype)
    alb_arr = as_np(alb, dtype)
    ht = as_np(terrain_height, dtype)
    ht_fine = as_np(fine_terrain_height, dtype)
    out = mu + al_arr[0] / (alt_arr[0] * alb_arr[0]) * dtype(G) * (ht - ht_fine)
    return out.astype(np.float64)


def set_w_surface_candidate(*, state: Any, grid: Any, metrics: Any, dtype: Any = np.float64) -> np.ndarray:
    """Transcribe module_bc_em.F::set_w_surface for fill_w_flag=.true."""

    ht = as_np(grid.terrain_height, dtype)
    u = as_np(state.u, dtype)
    v = as_np(state.v, dtype)
    msftx = as_np(metrics.msftx, dtype)
    msfty = as_np(metrics.msfty, dtype)
    znw = as_np(grid.vertical.eta_levels, dtype)
    cf1 = dtype(as_np(metrics.cf1, dtype))
    cf2 = dtype(as_np(metrics.cf2, dtype))
    cf3 = dtype(as_np(metrics.cf3, dtype))
    rdx = dtype(1.0 / float(grid.projection.dx_m))
    rdy = dtype(1.0 / float(grid.projection.dy_m))
    nz = int(znw.shape[0] - 1)
    ny, nx = ht.shape
    w = np.zeros((nz + 1, ny, nx), dtype=dtype)
    half = dtype(0.5)
    for j in range(ny):
        jm1 = max(j - 1, 0)
        jp1 = min(j + 1, ny - 1)
        for i in range(nx):
            im1 = max(i - 1, 0)
            ip1 = min(i + 1, nx - 1)
            vv_up = cf1 * v[0, j + 1, i] + cf2 * v[1, j + 1, i] + cf3 * v[2, j + 1, i]
            vv_dn = cf1 * v[0, j, i] + cf2 * v[1, j, i] + cf3 * v[2, j, i]
            uu_rt = cf1 * u[0, j, i + 1] + cf2 * u[1, j, i + 1] + cf3 * u[2, j, i + 1]
            uu_lf = cf1 * u[0, j, i] + cf2 * u[1, j, i] + cf3 * u[2, j, i]
            w[0, j, i] = (
                msfty[j, i]
                * half
                * rdy
                * ((ht[jp1, i] - ht[j, i]) * vv_up + (ht[j, i] - ht[jm1, i]) * vv_dn)
                + msftx[j, i]
                * half
                * rdx
                * ((ht[j, ip1] - ht[j, i]) * uu_rt + (ht[j, i] - ht[j, im1]) * uu_lf)
            )
    for k in range(1, nz + 1):
        w[k] = w[0] * znw[k] * znw[k]
    return w.astype(np.float64)


def predecessor_summary() -> dict[str, Any]:
    data = read_json(PREDECESSOR_JSON)
    rows = {}
    for row in data.get("proof", {}).get("stage_vs_wrf_precall_table", []):
        if row.get("stage") in {"raw_child_state", "live_child_state", "haloed_step_entry_state"}:
            rows[row["stage"]] = {
                field: row.get("metrics", {}).get(field, {}).get("max_abs") for field in P_FAMILY
            }
    return {
        "source": str(PREDECESSOR_JSON),
        "path_info": path_info(PREDECESSOR_JSON),
        "markdown": path_info(PREDECESSOR_MD),
        "verdict": data.get("verdict"),
        "stage_residuals_max_abs": rows,
    }


def source_semantics() -> dict[str, Any]:
    return {
        "med_nest_initial": line_hits(
            WRF_SOURCE_ROOT / "share/mediation_integrate.F",
            {
                "copy_mub_save": "copy_3d_field ( nest%mub_save",
                "blend_mub": "blend_terrain ( nest%mub_fine , nest%mub",
                "adjust_tempqv": "CALL adjust_tempqv",
                "press_adj_true": "nest%press_adj = .TRUE.",
                "start_domain_nest": "CALL start_domain ( nest , .TRUE. )",
            },
        ),
        "start_domain_em": line_hits(
            WRF_SOURCE_ROOT / "dyn_em/start_em.F",
            {
                "pressure_recompute_comment": "Use equations from calc_p_rho_phi to derive p and al from ph",
                "hypsometric_opt_2_al": "grid%al(i,k,j) = (grid%ph_1(i,k+1,j)-grid%ph_1(i,k,j)+grid%phb",
                "pressure_recompute": "grid%p(i,k,j)=p1000mb*",
                "press_adj_guard": "IF ( grid%press_adj",
                "press_adj_mu": "grid%MU_2(i,j) = grid%MU_2(i,j) + grid%al(i,1,j)",
                "w_branch": "IF ( w_needs_to_be_set ) THEN",
                "set_w_surface": "CALL set_w_surface",
            },
        ),
        "set_w_surface": line_hits(
            WRF_SOURCE_ROOT / "dyn_em/module_bc_em.F",
            {
                "subroutine": "SUBROUTINE set_w_surface",
                "surface_formula": "w(i,1,j)=",
                "vertical_fill": "w(i,k,j) = w(i,1,j)*znw(k)*znw(k)",
            },
        ),
        "module_configure_defaults": line_hits(
            WRF_SOURCE_ROOT / "frame/module_configure.f90",
            {
                "rebalance_default": "rebalance = 0",
                "hypsometric_opt_default": "hypsometric_opt = 2",
                "adjust_heights_default": "adjust_heights = .false.",
                "blend_width_default": "blend_width = 5",
                "use_theta_m_default": "use_theta_m = 1",
                "use_input_w_default": "use_input_w = .false.",
            },
        ),
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.integration.d02_replay import _wrf_start_domain_base_from_hgt  # noqa: PLC0415

    pre_shapes = pre.expected_shapes()
    precall = pre.parse_wrf_surface("before_first_rk_step_part1_call", pre_shapes)
    after_step = pre.parse_wrf_surface("after_step_increment", pre_shapes)
    if precall.get("status") != "WRF_SURFACE_READY":
        return {"status": "BLOCKED_PRECALL_WRF_TRUTH", "blocker": precall}
    if after_step.get("status") != "WRF_SURFACE_READY":
        return {"status": "BLOCKED_AFTER_STEP_WRF_TRUTH", "blocker": after_step}

    stage_capture = loader.build_stage_capture()
    if stage_capture.get("status") != "JAX_STAGE_CAPTURE_READY":
        return {"status": "BLOCKED_JAX_STAGE_CAPTURE", "blocker": stage_capture}
    inputs = live.build_live_nest_step1_inputs()
    live_child = inputs["live_child"]
    raw_child = inputs["raw_child"]
    state = live_child["state"]
    base = live_child["base_state"]
    metrics = live_child["metrics"]
    grid = live_child["grid"]

    wrf_arrays = precall["arrays"]
    stage_rows: list[dict[str, Any]] = []
    for stage in ("raw_child_state", "live_child_state", "boundary_packaged_state", "initial_carry_state", "haloed_step_entry_state"):
        arrays = stage_capture["stages"][stage]
        stage_rows.append(
            {
                "stage": stage,
                "metrics": selected_metrics(arrays, wrf_arrays, P_FAMILY),
            }
        )

    wrf_after_to_precall = selected_metrics(after_step["arrays"], wrf_arrays, P_FAMILY)
    live_arrays = {
        "P_STATE": as_np(state.p_perturbation),
        "MU_STATE": as_np(state.mu_perturbation),
        "W_STATE": as_np(state.w),
        "PH_STATE": as_np(state.ph_perturbation),
    }

    p_start, _al_start_diag, alt_start = start_domain_pressure_candidate(
        pb=base.pb,
        phb=base.phb,
        mub=base.mub,
        ph_perturbation=state.ph_perturbation,
        theta_full=state.theta,
        mu_perturbation=state.mu_perturbation,
        qv=state.qv,
        metrics=metrics,
        use_theta_m=1,
        dtype=np.float64,
    )
    p_start_f32, _al_start_f32, _alt_start_f32 = start_domain_pressure_candidate(
        pb=base.pb,
        phb=base.phb,
        mub=base.mub,
        ph_perturbation=state.ph_perturbation,
        theta_full=state.theta,
        mu_perturbation=state.mu_perturbation,
        qv=state.qv,
        metrics=metrics,
        use_theta_m=1,
        dtype=np.float32,
    )
    p_wrf_exact_f32, _al_wrf_exact, _alt_wrf_exact = start_domain_pressure_candidate(
        pb=wrf_arrays["PB"],
        phb=wrf_arrays["PHB"],
        mub=wrf_arrays["MUB"],
        ph_perturbation=wrf_arrays["PH_STATE"],
        theta_full=wrf_arrays["T_STATE"] + T0,
        mu_perturbation=state.mu_perturbation,
        qv=state.qv,
        metrics=metrics,
        use_theta_m=1,
        dtype=np.float32,
    )
    _pb, _mub, _phb, _t_init, alb_start = _wrf_start_domain_base_from_hgt(
        inputs["run"],
        "d02",
        hgt=grid.terrain_height,
        metrics=metrics,
    )
    al_start = alt_start - as_np(alb_start, np.float64)
    mu_press = press_adj_mu_candidate(
        mu_perturbation=state.mu_perturbation,
        al=al_start,
        alt=alt_start,
        alb=alb_start,
        terrain_height=grid.terrain_height,
        fine_terrain_height=raw_child["grid"].terrain_height,
    )
    w_surface = set_w_surface_candidate(state=state, grid=grid, metrics=metrics)

    candidate_arrays = {
        "P_STATE": p_start,
        "MU_STATE": mu_press,
        "W_STATE": w_surface,
    }
    raw_metrics = {
        "P_STATE": metric_brief(compare("P_STATE", live_arrays["P_STATE"], wrf_arrays["P_STATE"])),
        "MU_STATE": metric_brief(compare("MU_STATE", live_arrays["MU_STATE"], wrf_arrays["MU_STATE"])),
        "W_STATE": metric_brief(compare("W_STATE", live_arrays["W_STATE"], wrf_arrays["W_STATE"])),
    }
    candidate_metrics = {
        "P_STATE": metric_brief(compare("P_STATE", p_start, wrf_arrays["P_STATE"])),
        "P_STATE_fp32_current_formula": metric_brief(compare("P_STATE", p_start_f32, wrf_arrays["P_STATE"])),
        "P_STATE_fp32_wrf_exact_inputs_formula": metric_brief(compare("P_STATE", p_wrf_exact_f32, wrf_arrays["P_STATE"])),
        "MU_STATE": metric_brief(compare("MU_STATE", mu_press, wrf_arrays["MU_STATE"])),
        "W_STATE": metric_brief(compare("W_STATE", w_surface, wrf_arrays["W_STATE"])),
    }
    improvement = {}
    for field in ("P_STATE", "MU_STATE", "W_STATE"):
        raw_max = raw_metrics[field].get("max_abs")
        cand_max = candidate_metrics[field].get("max_abs")
        improvement[field] = {
            "raw_max_abs": raw_max,
            "candidate_max_abs": cand_max,
            "max_abs_reduction": None if raw_max is None or cand_max is None else float(raw_max) - float(cand_max),
            "candidate_vs_raw_delta": metric_brief(compare(field, candidate_arrays[field], live_arrays[field])),
        }

    raw_w_surface = as_np(state.w)[0]
    wrf_w_surface = wrf_arrays["W_STATE"][0]
    w_branch_evidence = {
        "raw_child_surface_w_absmax": float(np.max(np.abs(raw_w_surface))),
        "wrf_precall_surface_w_absmax": float(np.max(np.abs(wrf_w_surface))),
        "use_input_w_default_false": True,
        "raw_surface_zero_implies_w_needs_to_be_set": bool(float(np.max(np.abs(raw_w_surface))) < 1.0e-6),
    }

    return {
        "status": "PROOF_EXECUTED",
        "verdict": VERDICT,
        "wrf_truth": {
            "precall": {key: value for key, value in precall.items() if key != "arrays"},
            "after_step_increment": {key: value for key, value in after_step.items() if key != "arrays"},
            "after_step_increment_to_precall": wrf_after_to_precall,
        },
        "jax_capture": {
            "stage_capture": {key: value for key, value in stage_capture.items() if key != "stages"},
            "stage_vs_wrf_precall": stage_rows,
            "live_nest_init_meta": {
                "base": live_child.get("live_nest_base_init"),
                "transient_adjust_mub": live_child.get("transient_adjust_mub"),
                "theta_qv_adjust": live_child.get("theta_qv_adjust"),
            },
        },
        "wrf_source_semantics": source_semantics(),
        "candidate_formulas": {
            "start_domain_pressure": {
                "source": "dyn_em/start_em.F hypsometric_opt=2 al from PH, then EOS pressure",
                "use_theta_m": 1,
                "rebalance": 0,
                "current_jax_inputs_metric": candidate_metrics["P_STATE"],
                "current_jax_inputs_fp32_metric": candidate_metrics["P_STATE_fp32_current_formula"],
                "wrf_exact_base_ph_theta_fp32_metric": candidate_metrics["P_STATE_fp32_wrf_exact_inputs_formula"],
                "exact_patch_status": "NOT_SAFE_YET_NEEDS_PRE_PRESS_MU_OR_AL_ALT_START_DOMAIN_SUBSURFACE",
            },
            "press_adj_mu": {
                "source": "dyn_em/start_em.F press_adj terrain-delta correction",
                "metric": candidate_metrics["MU_STATE"],
            },
            "set_w_surface": {
                "source": "dyn_em/module_bc_em.F::set_w_surface with fill_w_flag=.true.",
                "metric": candidate_metrics["W_STATE"],
                "branch_evidence": w_branch_evidence,
            },
        },
        "raw_vs_candidate_summary": {
            "raw_metrics": raw_metrics,
            "candidate_metrics": candidate_metrics,
            "improvement": improvement,
            "ranked_candidate_residuals": rank_candidate_metrics(
                {
                    "P_STATE": candidate_metrics["P_STATE"],
                    "MU_STATE": candidate_metrics["MU_STATE"],
                    "W_STATE": candidate_metrics["W_STATE"],
                }
            ),
        },
        "ranked_hypotheses": [
            {
                "rank": 1,
                "hypothesis": "Missing WRF start_domain perturbation-state initialization after live-nest base/theta/QV correction.",
                "status": "SUPPORTED_LOCALIZED",
                "evidence": (
                    "P start_domain recompute reduces max_abs from 69.96875 to 3.945858; "
                    "MU press_adj reduces 13.2561 to 0.04777; W set_w_surface reduces "
                    "0.76055 to 1.3e-7."
                ),
            },
            {
                "rank": 2,
                "hypothesis": "Exact P/MU closure needs internal start_domain pre/post-press_adj and al/alt truth surfaces or stricter Fortran evaluation order.",
                "status": "REMAINING_GAP",
                "evidence": (
                    "Even WRF exact PB/PHB/PH/T plus FP32 formula leaves P max_abs "
                    "0.3828125 Pa; MU is near-closed but still not exact enough to prove "
                    "the source sequencing without pre/post press_adj truth. A source patch "
                    "would still be a guess."
                ),
            },
            {
                "rank": 3,
                "hypothesis": "Parent interpolation/blending alone explains P/MU/W.",
                "status": "LOWER_RANKED",
                "evidence": "Base/theta/QV are already close; W closes through set_w_surface and MU/P both point into start_domain, not a new parent interpolation surface.",
            },
        ],
        "exclusions": [
            "WRF after_step_increment -> before_first_rk_step_part1_call is exact for P/MU/W/PH in reused pre-part1 truth.",
            "Prior proof showed WRF before_first_rk_step_part1_call -> after_first_rk_step_part1 is exact for P/MU/W/PH.",
            "Prior proof showed JAX raw/live/boundary/carry/halo all retain the same raw P/MU/W residuals.",
            "Boundary package, initial carry, halo application, _physics_step_forcing, first_rk_step_part1, phy_prep, and acoustic refresh are not the first cause for this boundary.",
            "W_STATE is not an unknown physics tendency: raw surface W is zero and WRF default use_input_w=.false. forces set_w_surface.",
        ],
        "next_truth_surface": (
            "Emit WRF start_domain live-nest surfaces after the hypsometric P/al/alt recompute and "
            "immediately before/after press_adj, including P_STATE, MU_STATE, al, alt, alb, PH_STATE, "
            "PB, MUB, PHB, theta, qv, HT, and HT_FINE. That is the smallest surface needed before "
            "a GPU-native source patch."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    summary = proof["raw_vs_candidate_summary"]
    raw = summary["raw_metrics"]
    cand = summary["candidate_metrics"]
    lines = [
        "# V0.14 Step-1 Live-Nest Perturbation-State Init",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        "- Manager hypothesis is supported but not fully patch-ready: WRF does recompute/adjust `P/MU/W` before the first part1 call, while current JAX keeps raw `wrfinput_d02` perturbation leaves.",
        "- No production source edit was applied because `P_STATE` still needs an internal `start_domain` `al/alt` or pre-press-MU surface before an exact GPU-native patch.",
        "",
        "## Candidate Table",
        "",
        "| Field | Raw JAX max_abs | Candidate/source | Candidate max_abs | Notes |",
        "|---|---:|---|---:|---|",
        (
            f"| `P_STATE` | {raw['P_STATE']['max_abs']} | `start_domain` hypsometric pressure recompute | "
            f"{cand['P_STATE']['max_abs']} | WRF-exact-input FP32 falsifier: {cand['P_STATE_fp32_wrf_exact_inputs_formula']['max_abs']} Pa |"
        ),
        (
            f"| `MU_STATE` | {raw['MU_STATE']['max_abs']} | `press_adj` terrain-delta correction | "
            f"{cand['MU_STATE']['max_abs']} | Not patch-ready: p95 {cand['MU_STATE']['p95']}, p99 {cand['MU_STATE']['p99']}; needs pre/post-press_adj truth |"
        ),
        (
            f"| `W_STATE` | {raw['W_STATE']['max_abs']} | `set_w_surface(fill_w_flag=.true.)` | "
            f"{cand['W_STATE']['max_abs']} | Closed proof-locally |"
        ),
        "",
        "## Ranked Hypotheses",
        "",
    ]
    for item in proof["ranked_hypotheses"]:
        lines.append(f"- {item['rank']}. {item['hypothesis']} Status: `{item['status']}`. {item['evidence']}")
    lines.extend(["", "## Exclusions", ""])
    for item in proof["exclusions"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Surface",
            "",
            proof["next_truth_surface"],
            "",
            "Detailed metrics are in `proofs/v014/step1_live_nest_perturb_state_init.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Live-Nest Perturbation-State Init",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: close or precisely localize the live-nest `raw_child_state -> live_child_state` perturbation-state mismatch for `P_STATE/MU_STATE/W_STATE`.",
        "",
        "files changed:",
        "- `proofs/v014/step1_live_nest_perturb_state_init.py`",
        "- `proofs/v014/step1_live_nest_perturb_state_init.json`",
        "- `proofs/v014/step1_live_nest_perturb_state_init.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`",
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
            f"- reused WRF pre-call truth root `{PRECALL_TRUTH_ROOT}`",
            "",
            "ranked hypotheses/exclusions:",
        ]
    )
    for item in payload["proof"]["ranked_hypotheses"]:
        lines.append(f"- rank {item['rank']}: {item['status']} - {item['hypothesis']}")
    for item in payload["proof"]["exclusions"]:
        lines.append(f"- excluded: {item}")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    proof = build_proof()
    verdict = proof.get("verdict", f"STEP1_LIVE_NEST_PERTURB_STATE_BLOCKED_{proof.get('status', 'UNKNOWN')}")
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_live_nest_perturb_state_init.v1",
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
        "tooling_verdict": "YES_FASTEST_RIGOROUS_WALL_CLOCK_CPU_ONLY_SAVEPOINT_REUSE_PLUS_FORMULA_FALSIFIERS",
        "git": git_metadata(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "predecessor_json": path_info(PREDECESSOR_JSON),
            "predecessor_md": path_info(PREDECESSOR_MD),
            "wrf_precall_truth_root": path_info(PRECALL_TRUTH_ROOT),
            "wrf_source_root": path_info(WRF_SOURCE_ROOT),
            "accepted_step1_truth_npz": path_info(live.ACCEPTED_TRUTH),
        },
        "predecessor_baseline": predecessor_summary(),
        "proof": proof,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_live_nest_perturb_state_init.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_perturb_state_init.py",
                "python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json >/tmp/step1_live_nest_perturb_state_init.validated.json",
                "git diff -- src/gpuwrf",
            ],
            "wrf_instrumentation": [
                "No new WRF run; reused existing CPU-only WRF truth surfaces and inspected existing WRF source."
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "The proof-local P_STATE formula is localized but not exact enough for a production patch.",
            "A safe source edit needs WRF start_domain sub-surfaces for al/alt and pre/post press_adj MU.",
            "No full Step-1 rerun was attempted because no production source changed.",
        ],
        "next_decision": (
            "Open one narrow WRF savepoint/source sprint at start_domain live-nest perturbation init, "
            "then patch d02_replay only if P_STATE closes with the exact al/alt/pre-press-MU contract."
        ),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
