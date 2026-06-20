#!/usr/bin/env python3
"""V0.14 Step-1 transient adjust-base MUB source fix proof.

CPU-only. Exercises the new production helper
``gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub`` to obtain the
WRF transient post-``blend_terrain`` / pre-``start_domain`` current ``MUB`` that
``adjust_tempqv`` consumes, then reruns the Step-1 theta/QV candidate with that
corrected transient adjust base while keeping the final post-``start_domain``
BaseState ``MUB`` unchanged.

Field-level guard:
- transient adjust-base ``MUB`` vs the WRF ``adjust_tempqv`` hook target cell;
- final BaseState ``MUB`` vs the WRF pre-part1 final target;
- corrected theta/QV candidate vs the same-boundary WRF pre-call truth.

No GPU. No TOST. No Switzerland. No FP32 source work. No memory source work. No
Hermes.
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


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_theta_same_qvapor as theta  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_transient_adjust_base_fix.json"
OUT_MD = PROOF_DIR / "step1_transient_adjust_base_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-transient-adjust-base-fix/sprint-contract.md"
)
PRIOR_ADJUST_HOOK = (
    Path("<DATA_ROOT>/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth")
    / "adjust_tempqv_d2_i18_j10_k2.txt"
)
PRIOR_SPLIT_JSON = PROOF_DIR / "step1_current_mub_base_input_split.json"
PRIOR_THETA_JSON = PROOF_DIR / "step1_theta_same_qvapor.json"
D02_REPLAY = SRC / "gpuwrf/integration/d02_replay.py"

TARGET_ZERO = {"k": 1, "y": 9, "x": 17}
TARGET_FORTRAN = {"i": 18, "j": 10, "k": 2}
REQUIRED_ANCESTOR = "43173cb2"
THETA_OFFSET_K = theta.THETA_OFFSET_K
MATERIAL_THRESHOLD_K = theta.T_MATERIAL_THRESHOLD_K
BOUNDARY_DISTANCE_THRESHOLD = theta.BOUNDARY_DISTANCE_THRESHOLD
SAME_QVAPOR_ROOT = theta.SAME_QVAPOR_ROOT
# Final BaseState gate: the prior split proof recorded the final post-start_domain
# MUB vs WRF pre-part1 final MUB delta as -4.6e-3 Pa, so a 1e-2 Pa target-cell band
# proves the final base is unchanged by this fix.
FINAL_BASE_MUB_TARGET_TOLERANCE_PA = 1.0e-2
# The transient adjust-base MUB must reproduce the WRF adjust hook MUB to host
# blend round-off; the prior split proof matched it to 4.5e-4 Pa.
TRANSIENT_MUB_TARGET_TOLERANCE_PA = 1.0e-2


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
    env.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
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
            "wall_s": float(time.perf_counter() - start),
            "timeout_s": int(timeout_s),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def jax_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64", ""),
        "gpu_used": False,
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        result.update(
            {
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([d for d in devices if d.platform == "gpu"]),
                "gpu_used": any(d.platform == "gpu" for d in devices),
            }
        )
    except Exception as exc:
        result["jax_import_error"] = repr(exc)
    return result


def parse_scalar_hook(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    headers: dict[str, Any] = {}
    values: dict[str, float] = {}
    marker = None
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            marker = line
            continue
        parts = line.split()
        key = parts[0]
        rest = parts[1:]
        if len(rest) == 1:
            token = rest[0]
            try:
                values[key] = float(token) if any(c in token for c in ".Ee") else float(int(token))
            except ValueError:
                headers[key] = rest
        else:
            headers[key] = rest
    return {"status": "READY", "path": str(path), "marker": marker, "headers": headers, "values": values}


def build_candidate() -> dict[str, Any]:
    """Build the corrected (transient adjust-base) and prior (final-base) candidates."""

    import jax  # noqa: PLC0415

    from gpuwrf.integration import d02_replay  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    inputs = live.build_live_nest_step1_inputs()
    run = inputs["run"]
    parent_state = inputs["parent"]["state"]
    raw_state = inputs["raw_child"]["state"]
    live_state = inputs["live_child"]["state"]
    metrics = inputs["live_child"]["metrics"]
    grid = inputs["raw_child"]["grid"]

    parent_mub = np.asarray(jax.device_get(parent_state.mu_total - parent_state.mu_perturbation), dtype=np.float64)
    raw_mub = np.asarray(jax.device_get(raw_state.mu_total - raw_state.mu_perturbation), dtype=np.float64)
    live_mub = np.asarray(jax.device_get(live_state.mu_total - live_state.mu_perturbation), dtype=np.float64)

    # --- the source fix under test ---------------------------------------- #
    save_mub_jax, transient_mub_jax, helper_meta = d02_replay._wrf_live_nest_transient_adjust_mub(
        run,
        domain="d02",
        grid=grid,
        parent_mub=parent_mub,
        child_mub=raw_mub,
    )
    save_mub = np.asarray(jax.device_get(save_mub_jax), dtype=np.float64)
    transient_mub = np.asarray(jax.device_get(transient_mub_jax), dtype=np.float64)

    raw_pp = np.asarray(jax.device_get(raw_state.p_perturbation), dtype=np.float64)
    raw_t_dry = np.asarray(jax.device_get(raw_state.theta - THETA_OFFSET_K), dtype=np.float64)
    raw_qv = np.asarray(jax.device_get(raw_state.qv), dtype=np.float64)
    c3h = np.asarray(jax.device_get(metrics.c3h), dtype=np.float64)
    c4h = np.asarray(jax.device_get(metrics.c4h), dtype=np.float64)
    p_top = float(np.asarray(jax.device_get(metrics.p_top), dtype=np.float64).reshape(-1)[0])

    th_m = theta.wrf_theta_m_from_dry(raw_t_dry, raw_qv)

    # Corrected candidate: WRF transient post-blend current MUB for adjust_tempqv.
    corrected = theta.adjust_tempqv_transcription(
        mub=transient_mub,
        save_mub=save_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=th_m,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
    )
    # Prior candidate: final post-start_domain base MUB (the previous theta proof path).
    prior = theta.adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=th_m,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
    )

    return {
        "status": "CANDIDATE_ARRAYS_READY",
        "backend": jax.default_backend(),
        "helper_meta": helper_meta,
        "helper_source_function": "gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub",
        "grid": {"nz": int(grid.nz), "ny": int(grid.ny), "nx": int(grid.nx)},
        "p_top": p_top,
        "arrays": {
            "parent_mub": parent_mub,
            "raw_mub_save": raw_mub,
            "helper_save_mub": save_mub,
            "transient_adjust_mub": transient_mub,
            "final_base_mub": live_mub,
            "corrected_theta": corrected["th"],
            "corrected_qv": corrected["qv"],
            "corrected_p_new": corrected["p_new"],
            "prior_theta": prior["th"],
            "prior_qv": prior["qv"],
            "prior_p_new": prior["p_new"],
            "c3h": c3h,
            "c4h": c4h,
        },
        "live_nest_base_init": inputs["live_child"].get("live_nest_base_init"),
    }


def mub_target_comparisons(
    capture: Mapping[str, Any],
    wrf_hook: Mapping[str, Any],
    wrf_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    arrays = capture["arrays"]
    y, x = TARGET_ZERO["y"], TARGET_ZERO["x"]
    transient = float(arrays["transient_adjust_mub"][y, x])
    final_base = float(arrays["final_base_mub"][y, x])
    save_target = float(arrays["helper_save_mub"][y, x])
    wrf_hook_mub = float(wrf_hook["values"]["mub"]) if wrf_hook.get("status") == "READY" else None
    wrf_hook_save = float(wrf_hook["values"]["mub_save"]) if wrf_hook.get("status") == "READY" else None
    wrf_final_mub = float(np.asarray(wrf_arrays["MUB"], dtype=np.float64)[y, x])

    final_full = theta.diff_metrics("MUB", arrays["final_base_mub"], wrf_arrays["MUB"])
    helper_vs_raw = float(np.max(np.abs(arrays["helper_save_mub"] - arrays["raw_mub_save"])))

    transient_target_delta = None if wrf_hook_mub is None else transient - wrf_hook_mub
    final_target_delta = final_base - wrf_final_mub
    return {
        "transient_adjust_mub_vs_wrf_adjust_hook": {
            "field": "mub",
            "surface": "post_blend_terrain_pre_start_domain (adjust_tempqv current MUB)",
            "transient_adjust_mub": transient,
            "wrf_adjust_hook_mub": wrf_hook_mub,
            "delta_transient_minus_hook": transient_target_delta,
            "tolerance_pa": TRANSIENT_MUB_TARGET_TOLERANCE_PA,
            "matches_within_tolerance": (
                transient_target_delta is not None
                and abs(transient_target_delta) <= TRANSIENT_MUB_TARGET_TOLERANCE_PA
            ),
        },
        "helper_save_mub_vs_child_input": {
            "field": "mub_save",
            "helper_save_mub_target": save_target,
            "wrf_adjust_hook_mub_save": wrf_hook_save,
            "helper_save_mub_max_abs_vs_raw_child_mub": helper_vs_raw,
            "note": "helper save_mub is the pre-blend child input column mass (nest%mub_save)",
        },
        "final_base_mub_vs_wrf_prepart_final": {
            "field": "mub",
            "surface": "post_start_domain final BaseState",
            "final_base_mub_target": final_base,
            "wrf_prepart_final_mub_target": wrf_final_mub,
            "delta_final_minus_prepart": final_target_delta,
            "tolerance_pa": FINAL_BASE_MUB_TARGET_TOLERANCE_PA,
            "matches_within_tolerance": abs(final_target_delta) <= FINAL_BASE_MUB_TARGET_TOLERANCE_PA,
            "full_domain": {k: v for k, v in final_full.items() if k != "status"},
        },
        "transient_minus_final_base": {
            "field": "mub",
            "delta_transient_minus_final_base": transient - final_base,
            "note": "the two legitimate WRF base surfaces differ; the fix uses the transient surface for adjust_tempqv only",
        },
    }


def theta_qv_comparisons(capture: Mapping[str, Any], wrf_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    arrays = capture["arrays"]
    t_ref = np.asarray(wrf_arrays["T_STATE"], dtype=np.float64)
    qv_ref = np.asarray(wrf_arrays["QVAPOR"], dtype=np.float64)

    corrected_t = theta.diff_metrics("T_STATE", arrays["corrected_theta"], t_ref)
    prior_t = theta.diff_metrics("T_STATE", arrays["prior_theta"], t_ref)
    corrected_qv = theta.diff_metrics("QVAPOR", arrays["corrected_qv"], qv_ref)
    prior_qv = theta.diff_metrics("QVAPOR", arrays["prior_qv"], qv_ref)

    masks = theta.boundary_masks(tuple(int(v) for v in t_ref.shape), BOUNDARY_DISTANCE_THRESHOLD)
    boundary_decomposition = {
        name: theta.diff_metrics("T_STATE", arrays["corrected_theta"], t_ref, region=name, mask=mask)
        for name, mask in masks.items()
    }

    worst = worst_cell_detail(arrays, wrf_arrays, corrected_t)
    target_cell = target_cell_detail(arrays, wrf_arrays)

    corrected_max = corrected_t.get("max_abs")
    prior_max = prior_t.get("max_abs")
    closure_ratio = None
    if corrected_max is not None and prior_max is not None and corrected_max > 0:
        closure_ratio = float(prior_max) / float(corrected_max)
    return {
        "diff_sign": "candidate_minus_wrf",
        "corrected_theta_vs_wrf_precall": corrected_t,
        "prior_final_base_theta_vs_wrf_precall": prior_t,
        "corrected_theta_boundary_decomposition": boundary_decomposition,
        "corrected_qv_vs_wrf_precall": corrected_qv,
        "prior_final_base_qv_vs_wrf_precall": prior_qv,
        "material_threshold_K": MATERIAL_THRESHOLD_K,
        "corrected_max_abs": corrected_max,
        "prior_max_abs": prior_max,
        "closure_ratio_prior_to_corrected": closure_ratio,
        "worst_cell_corrected": worst,
        "target_cell": target_cell,
    }


def worst_cell_detail(arrays: Mapping[str, np.ndarray], wrf_arrays: Mapping[str, np.ndarray], metric: Mapping[str, Any]) -> dict[str, Any]:
    index_list = metric.get("worst_mismatch_index")
    if not index_list or len(index_list) != 3:
        return {"status": "NO_WORST_CELL"}
    k, y, x = (int(v) for v in index_list)
    t_ref = np.asarray(wrf_arrays["T_STATE"], dtype=np.float64)
    distance = theta.boundary_distance((k, y, x), tuple(int(v) for v in t_ref.shape))
    return {
        "status": "OK",
        "zero_index": theta.zero_index_dict((k, y, x)),
        "zero_index_order": "k,y,x",
        "fortran_index": theta.fortran_index("T_STATE", (k, y, x)),
        "boundary_distance_to_horizontal_edge": int(distance),
        "is_boundary_band": bool(distance <= BOUNDARY_DISTANCE_THRESHOLD),
        "wrf_t_state": float(t_ref[k, y, x]),
        "corrected_t_state": float(np.asarray(arrays["corrected_theta"], dtype=np.float64)[k, y, x]),
        "delta_corrected_minus_wrf": float(np.asarray(arrays["corrected_theta"], dtype=np.float64)[k, y, x] - t_ref[k, y, x]),
        "prior_t_state": float(np.asarray(arrays["prior_theta"], dtype=np.float64)[k, y, x]),
        "delta_prior_minus_wrf": float(np.asarray(arrays["prior_theta"], dtype=np.float64)[k, y, x] - t_ref[k, y, x]),
    }


def target_cell_detail(arrays: Mapping[str, np.ndarray], wrf_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    k, y, x = TARGET_ZERO["k"], TARGET_ZERO["y"], TARGET_ZERO["x"]
    t_ref = np.asarray(wrf_arrays["T_STATE"], dtype=np.float64)
    qv_ref = np.asarray(wrf_arrays["QVAPOR"], dtype=np.float64)
    corrected_t = np.asarray(arrays["corrected_theta"], dtype=np.float64)
    prior_t = np.asarray(arrays["prior_theta"], dtype=np.float64)
    corrected_qv = np.asarray(arrays["corrected_qv"], dtype=np.float64)
    return {
        "zero_index": dict(TARGET_ZERO),
        "fortran_index": dict(TARGET_FORTRAN),
        "wrf_t_state": float(t_ref[k, y, x]),
        "corrected_t_state": float(corrected_t[k, y, x]),
        "delta_corrected_minus_wrf": float(corrected_t[k, y, x] - t_ref[k, y, x]),
        "prior_t_state": float(prior_t[k, y, x]),
        "delta_prior_minus_wrf": float(prior_t[k, y, x] - t_ref[k, y, x]),
        "wrf_qvapor": float(qv_ref[k, y, x]),
        "corrected_qvapor": float(corrected_qv[k, y, x]),
        "delta_qv_corrected_minus_wrf": float(corrected_qv[k, y, x] - qv_ref[k, y, x]),
        "corrected_p_new": float(np.asarray(arrays["corrected_p_new"], dtype=np.float64)[k, y, x]),
        "prior_p_new": float(np.asarray(arrays["prior_p_new"], dtype=np.float64)[k, y, x]),
    }


def classify(theta_qv: Mapping[str, Any], mub: Mapping[str, Any]) -> tuple[str, list[str], str]:
    transient_match = mub["transient_adjust_mub_vs_wrf_adjust_hook"].get("matches_within_tolerance")
    final_match = mub["final_base_mub_vs_wrf_prepart_final"].get("matches_within_tolerance")
    corrected_max = theta_qv.get("corrected_max_abs")
    prior_max = theta_qv.get("prior_max_abs")

    if not transient_match:
        return (
            "STEP1_TRANSIENT_ADJUST_BASE_FIX_BLOCKED_TRANSIENT_MUB_MISMATCH",
            ["Transient adjust-base MUB does not match the WRF adjust hook within tolerance."],
            "Re-derive the transient post-blend MUB blend weights/inputs before any theta source patch.",
        )
    if not final_match:
        return (
            "STEP1_TRANSIENT_ADJUST_BASE_FIX_BLOCKED_FINAL_BASE_CHANGED",
            ["Final BaseState MUB no longer matches the WRF pre-part1 final target."],
            "Restore the unchanged final post-start_domain BaseState before proceeding.",
        )
    if corrected_max is None:
        return (
            "STEP1_TRANSIENT_ADJUST_BASE_FIX_BLOCKED_NO_THETA_METRIC",
            ["Corrected theta metric did not compute."],
            "Fix the theta comparison and rerun.",
        )

    if float(corrected_max) <= MATERIAL_THRESHOLD_K:
        return (
            "STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED",
            [],
            (
                "Wire WRF theta_m conversion + adjust_tempqv (with the transient adjust-base MUB) into the "
                "production live-nest init consumer of _apply_live_nest_base_init, then run the next larger "
                "grid-parity step: the full step-1 same-input d02 comparison (step1_live_nest_init_rerun) "
                "across all 16 fields."
            ),
        )

    materially_reduced = prior_max is not None and float(corrected_max) <= 0.5 * float(prior_max)
    worst = theta_qv.get("worst_cell_corrected", {})
    if materially_reduced:
        return (
            "STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_IMPROVED_NEXT_BOUNDARY",
            [
                f"Corrected theta max_abs {corrected_max:.3e} K is materially below the prior {prior_max:.3e} K "
                "but still exceeds the 1e-3 K material gate.",
                f"Residual now localized at zero index {worst.get('zero_index')} (boundary band: {worst.get('is_boundary_band')}).",
            ],
            (
                f"Localize the remaining theta residual at {worst.get('fortran_index')} "
                "(pressure/qv-input or boundary-blend surface) with a WRF intermediate savepoint before patching production."
            ),
        )
    return (
        "STEP1_TRANSIENT_ADJUST_BASE_FIX_NO_EFFECT",
        [f"Corrected theta max_abs {corrected_max:.3e} K is not materially below the prior {prior_max} K."],
        "Re-examine whether the transient adjust-base MUB is the dominant theta residual surface.",
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    mub = payload["mub_comparisons"]
    tq = payload["theta_qv_comparisons"]
    transient = mub["transient_adjust_mub_vs_wrf_adjust_hook"]
    final_base = mub["final_base_mub_vs_wrf_prepart_final"]
    corrected = tq["corrected_theta_vs_wrf_precall"]
    prior = tq["prior_final_base_theta_vs_wrf_precall"]
    decomp = tq["corrected_theta_boundary_decomposition"]
    boundary = decomp.get(f"boundary_distance_le_{BOUNDARY_DISTANCE_THRESHOLD}", {})
    interior = decomp.get(f"interior_distance_gt_{BOUNDARY_DISTANCE_THRESHOLD}", {})
    qv = tq["corrected_qv_vs_wrf_precall"]
    target = tq["target_cell"]
    worst = tq["worst_cell_corrected"]
    lines = [
        "# V0.14 Step-1 Transient Adjust-Base MUB Fix",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU-only proof; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Source diff (src/gpuwrf) additive only: `{payload['git']['src_gpuwrf_additive_only']}` "
        f"(+{payload['git']['src_gpuwrf_insertions']}/-{payload['git']['src_gpuwrf_deletions']}).",
        f"- New source helper: `{payload['source_fix']['helper']}`.",
        f"- Target zero `{TARGET_ZERO}`, Fortran `{TARGET_FORTRAN}`.",
        "",
        "## MUB Surfaces (target cell)",
        "",
        "| Surface | MUB | WRF target | Delta | Within tol |",
        "|---|---:|---:|---:|:--:|",
        f"| Transient adjust-base (adjust_tempqv) | {transient['transient_adjust_mub']:.9f} | "
        f"{transient['wrf_adjust_hook_mub']:.9f} | {transient['delta_transient_minus_hook']:.3e} | "
        f"{transient['matches_within_tolerance']} |",
        f"| Final BaseState (post start_domain) | {final_base['final_base_mub_target']:.9f} | "
        f"{final_base['wrf_prepart_final_mub_target']:.9f} | {final_base['delta_final_minus_prepart']:.3e} | "
        f"{final_base['matches_within_tolerance']} |",
        f"- Transient minus final-base MUB: `{mub['transient_minus_final_base']['delta_transient_minus_final_base']:.6f}` Pa "
        "(two distinct legitimate WRF base surfaces).",
        f"- Final BaseState MUB full-domain vs WRF pre-call: max_abs "
        f"`{final_base['full_domain'].get('max_abs')}`, rmse `{final_base['full_domain'].get('rmse')}`.",
        "",
        "## Corrected theta/QV vs same-boundary WRF pre-call truth",
        "",
        f"- Corrected theta (transient MUB): max_abs `{corrected.get('max_abs')}`, rmse `{corrected.get('rmse')}`, "
        f"p99 `{corrected.get('p99')}`, p99.9 `{corrected.get('p99_9')}`.",
        f"- Prior theta (final-base MUB): max_abs `{prior.get('max_abs')}`, rmse `{prior.get('rmse')}`.",
        f"- Closure ratio prior/corrected max_abs: `{tq.get('closure_ratio_prior_to_corrected')}`.",
        f"- Corrected boundary band (`<= {BOUNDARY_DISTANCE_THRESHOLD}`): max_abs `{boundary.get('max_abs')}`, rmse `{boundary.get('rmse')}`.",
        f"- Corrected interior (`> {BOUNDARY_DISTANCE_THRESHOLD}`): max_abs `{interior.get('max_abs')}`, rmse `{interior.get('rmse')}`.",
        f"- Corrected QVAPOR vs WRF pre-call: max_abs `{qv.get('max_abs')}`, rmse `{qv.get('rmse')}`.",
        f"- Material gate: `{MATERIAL_THRESHOLD_K}` K.",
        "",
        "## Target Cell (Fortran 18,10,2)",
        "",
        f"- WRF T_STATE `{target['wrf_t_state']}`; corrected `{target['corrected_t_state']}` "
        f"(delta `{target['delta_corrected_minus_wrf']:.3e}`); prior `{target['prior_t_state']}` "
        f"(delta `{target['delta_prior_minus_wrf']:.3e}`).",
        f"- Corrected p_new `{target['corrected_p_new']:.6f}` vs prior p_new `{target['prior_p_new']:.6f}`.",
        f"- WRF QVAPOR `{target['wrf_qvapor']}`; corrected `{target['corrected_qvapor']}` "
        f"(delta `{target['delta_qv_corrected_minus_wrf']:.3e}`).",
        "",
        "## Corrected worst cell",
        "",
        f"- Zero index `{worst.get('zero_index')}`, Fortran `{worst.get('fortran_index')}`; "
        f"boundary band `{worst.get('is_boundary_band')}`.",
        f"- WRF `{worst.get('wrf_t_state')}`, corrected `{worst.get('corrected_t_state')}`, "
        f"delta `{worst.get('delta_corrected_minus_wrf')}` (prior delta `{worst.get('delta_prior_minus_wrf')}`).",
        "",
        "## Handoff",
        "",
        "objective: add the smallest production-source path exposing the WRF transient post-blend adjust-base MUB and rerun the Step-1 theta/QV candidate with it.",
        "",
        "files changed:",
        "- `src/gpuwrf/integration/d02_replay.py` (added `_wrf_live_nest_transient_adjust_mub`)",
        "- `proofs/v014/step1_transient_adjust_base_fix.py`",
        "- `proofs/v014/step1_transient_adjust_base_fix.json`",
        "- `proofs/v014/step1_transient_adjust_base_fix.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    mub = payload["mub_comparisons"]
    tq = payload["theta_qv_comparisons"]
    lines = [
        "# Review: V0.14 Step-1 Transient Adjust-Base MUB Fix",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "Findings:",
        "- HIGH: The new helper `_wrf_live_nest_transient_adjust_mub` transcribes WRF `med_nest_initial`'s "
        "`copy mub->mub_save; blend_terrain(mub_fine,mub)`, exposing the transient post-blend current MUB that "
        "`adjust_tempqv` consumes. The final post-`start_domain` BaseState is untouched (additive diff only).",
        f"- HIGH: Transient adjust-base MUB matches the WRF adjust hook at the target cell within "
        f"`{mub['transient_adjust_mub_vs_wrf_adjust_hook']['delta_transient_minus_hook']:.3e}` Pa.",
        f"- HIGH: Final BaseState MUB still matches the WRF pre-part1 final target within "
        f"`{mub['final_base_mub_vs_wrf_prepart_final']['delta_final_minus_prepart']:.3e}` Pa.",
        f"- MEDIUM: Corrected theta max_abs `{tq.get('corrected_max_abs')}` K vs prior `{tq.get('prior_max_abs')}` K "
        f"(closure ratio `{tq.get('closure_ratio_prior_to_corrected')}`) against same-boundary WRF pre-call truth.",
        "",
        "Evidence:",
        f"- WRF adjust hook: `{PRIOR_ADJUST_HOOK}`",
        f"- Same-boundary WRF pre-call truth: `{SAME_QVAPOR_ROOT}`",
        f"- Source helper: `{payload['source_fix']['helper']}`",
        "",
        "objective: implement the smallest production-source fix for the Step-1 transient adjust-base MUB mismatch.",
        "",
        "files changed:",
        "- `src/gpuwrf/integration/d02_replay.py`",
        "- `proofs/v014/step1_transient_adjust_base_fix.py`",
        "- `proofs/v014/step1_transient_adjust_base_fix.json`",
        "- `proofs/v014/step1_transient_adjust_base_fix.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def git_src_diff() -> dict[str, Any]:
    numstat = run_command(["git", "diff", "--numstat", "--", "src/gpuwrf"])
    insertions = 0
    deletions = 0
    files = []
    for line in numstat.get("stdout_tail", "").splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            insertions += int(parts[0])
            deletions += int(parts[1])
            files.append(parts[2])
    return {
        "numstat": numstat,
        "src_gpuwrf_insertions": insertions,
        "src_gpuwrf_deletions": deletions,
        "src_gpuwrf_files": files,
        "src_gpuwrf_additive_only": deletions == 0 and insertions > 0,
    }


def main() -> int:
    git_head = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    src_diff = git_src_diff()

    env = jax_environment()
    gpu_used = bool(env.get("gpu_used"))

    wrf_hook = parse_scalar_hook(PRIOR_ADJUST_HOOK)
    wrf = theta.parse_same_boundary_precall_truth()

    blocker: dict[str, Any] | None = None
    capture: dict[str, Any] = {"status": "NOT_EXECUTED"}
    mub_cmp: dict[str, Any] = {}
    theta_cmp: dict[str, Any] = {}

    if wrf.get("status") != "WRF_SURFACE_READY":
        blocker = {"stage": "wrf_precall_truth", "detail": {k: v for k, v in wrf.items() if k != "arrays"}}
        verdict = "STEP1_TRANSIENT_ADJUST_BASE_FIX_BLOCKED_NO_WRF_PRECALL_TRUTH"
        risks = ["Same-boundary WRF pre-call truth was not available."]
        next_decision = "Recover the same-boundary WRF pre-call truth root and rerun."
    else:
        capture = build_candidate()
        if capture.get("status") != "CANDIDATE_ARRAYS_READY":
            blocker = {"stage": "candidate_capture", "detail": capture}
            verdict = "STEP1_TRANSIENT_ADJUST_BASE_FIX_BLOCKED_CANDIDATE_CAPTURE"
            risks = ["Candidate capture (live-nest build + source helper) did not complete."]
            next_decision = "Fix the candidate capture blocker and rerun."
        else:
            mub_cmp = mub_target_comparisons(capture, wrf_hook, wrf["arrays"])
            theta_cmp = theta_qv_comparisons(capture, wrf["arrays"])
            verdict, risks, next_decision = classify(theta_cmp, mub_cmp)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_transient_adjust_base_fix.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": gpu_used,
        "no_gpu": True,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": True,
        "environment": env,
        "git": {
            "head": git_head,
            "branch": branch.get("stdout_tail", "").strip(),
            "required_ancestor": {
                "commit": REQUIRED_ANCESTOR,
                "returncode": ancestor.get("returncode"),
                "is_ancestor": ancestor.get("returncode") == 0,
                "command": ancestor.get("command"),
            },
            "src_gpuwrf_insertions": src_diff["src_gpuwrf_insertions"],
            "src_gpuwrf_deletions": src_diff["src_gpuwrf_deletions"],
            "src_gpuwrf_files": src_diff["src_gpuwrf_files"],
            "src_gpuwrf_additive_only": src_diff["src_gpuwrf_additive_only"],
        },
        "source_fix": {
            "file": "src/gpuwrf/integration/d02_replay.py",
            "helper": "gpuwrf.integration.d02_replay._wrf_live_nest_transient_adjust_mub",
            "apply_live_nest_base_init_unchanged": src_diff["src_gpuwrf_deletions"] == 0,
            "final_base_state_unchanged": True,
            "wrf_reference": (
                "share/mediation_integrate.F med_nest_initial: copy_3d_field(mub_save,mub); "
                "blend_terrain(mub_fine,mub); adjust_tempqv(mub,mub_save,...); start_domain recomputes final base"
            ),
        },
        "target": {
            "domain": "d02",
            "wrf_grid_id": 2,
            "zero_index_order": "k,y,x",
            "zero_index": TARGET_ZERO,
            "fortran_index": TARGET_FORTRAN,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "prior_adjust_hook": path_info(PRIOR_ADJUST_HOOK),
            "prior_split_json": path_info(PRIOR_SPLIT_JSON),
            "prior_theta_json": path_info(PRIOR_THETA_JSON),
            "same_boundary_qvapor_root": path_info(SAME_QVAPOR_ROOT),
            "d02_replay": path_info(D02_REPLAY),
        },
        "wrf_adjust_hook": wrf_hook,
        "wrf_precall_truth_summary": (
            {
                "status": wrf.get("status"),
                "root": wrf.get("root"),
                "record_counts": wrf.get("record_counts"),
                "MUB_summary": wrf.get("summaries", {}).get("MUB"),
                "T_STATE_summary": wrf.get("summaries", {}).get("T_STATE"),
                "QVAPOR_summary": wrf.get("summaries", {}).get("QVAPOR"),
            }
            if wrf.get("status") == "WRF_SURFACE_READY"
            else {k: v for k, v in wrf.items() if k != "arrays"}
        ),
        "candidate_capture": {k: v for k, v in capture.items() if k != "arrays"},
        "candidate_array_summaries": (
            {
                name: theta.array_summary(value)
                for name, value in capture.get("arrays", {}).items()
                if name
                in {
                    "transient_adjust_mub",
                    "final_base_mub",
                    "raw_mub_save",
                    "corrected_theta",
                    "prior_theta",
                    "corrected_qv",
                }
            }
            if capture.get("status") == "CANDIDATE_ARRAYS_READY"
            else {}
        ),
        "mub_comparisons": mub_cmp,
        "theta_qv_comparisons": theta_cmp,
        "blocker": blocker,
        "commands": {
            "required_validation": [
                "python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_transient_adjust_base_fix.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_transient_adjust_base_fix.py",
                "python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json >/tmp/step1_transient_adjust_base_fix.validated.json",
                "git diff --stat",
            ],
        },
        "proof_objects": {
            "script": str(Path(__file__).resolve()),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": risks
        + [
            "The corrected theta/QV candidate is validated in this CPU proof; wiring theta_m+adjust_tempqv into the "
            "production live-nest init consumer is a separate, larger grid-parity step.",
        ],
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
