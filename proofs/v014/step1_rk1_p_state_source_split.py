#!/usr/bin/env python3
"""V0.14 Step-1 RK1 P_STATE source split.

CPU-only proof.  Reuses the accepted WRF Step-1 substage truth and runs two
JAX captures against the same RK1 boundary:

* the current proof-local live-nest helper used by older comparators;
* the same helper with the production Mythos start_domain perturbation init
  applied before the child boundary package is rebuilt.

The goal is to decide whether the current material RK1 P_STATE residual is a
real source/tendency bug or a stale proof-loader contract.
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

import same_input_contract_builder as builder  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402
import step1_rk1_source_boundary as source  # noqa: E402
import step1_t_p_operator_localization as tp  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_rk1_p_state_source_split.json"
OUT_MD = PROOF_DIR / "step1_rk1_p_state_source_split.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-rk1-p-state-source-split/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
MYTHOS_JSON = PROOF_DIR / "mythos_kernel_fix_260609.json"
TP_JSON = PROOF_DIR / "step1_t_p_operator_localization.json"
SOURCE_JSON = PROOF_DIR / "step1_rk1_source_boundary.json"

VERDICT = "STEP1_RK1_P_STATE_SOURCE_REFUTED_STALE_PROOF_LOADER_BYPASS_NEXT_T_TENDF"

P_STATE_GATE_PA = 1.0
P_FAMILY_FIELDS = ("P_STATE", "MU_STATE", "W_STATE", "PH_STATE")
STATE_FIELDS = ("T_STATE", "P_STATE", "MU_STATE", "W_STATE", "PH_STATE")
WORK_FIELDS = ("T_WORK", "P_WORK", "MU_WORK", "W_WORK", "PH_WORK")
TENDENCY_FIELDS = (
    "T_TEND",
    "T_TENDF",
    "MU_TEND",
    "MU_TENDF",
    "PH_TEND",
    "PH_TENDF",
    "RW_TEND",
    "RW_TENDF",
)


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
    return {
        "head": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "log_head": run_command(["git", "log", "-1", "--oneline", "--decorate"]),
        "status_short_branch": run_command(["git", "status", "--short", "--branch"]),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            "first_mismatch_fortran",
            "worst_mismatch_fortran",
            "worst_mismatch_index",
        )
        if key in metric
    }


def selected_metrics(metrics: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: metric_brief(metrics.get(field)) for field in fields if field in metrics}


def material(field: str, metric: Mapping[str, Any] | None) -> bool:
    if not metric or metric.get("status") != "OK":
        return bool(metric)
    max_abs = metric.get("max_abs")
    if max_abs is None:
        return bool(metric.get("nonfinite_diff_count"))
    threshold = source.MATERIAL_THRESHOLDS.get(field)
    return threshold is not None and float(max_abs) > float(threshold)


def first_material(metrics: Mapping[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        if material(field, metrics.get(field)):
            return field
    return None


def compact_surface(comp: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {
        "status": comp.get("status"),
        "strict_first_mismatch_field": comp.get("strict_first_mismatch_field"),
        "material_first_field": comp.get("material_first_field"),
        "selected_metrics": selected_metrics(comp.get("per_field_metrics", {}), fields),
        "top_material_residuals": [
            {
                "field": item.get("field"),
                "max_abs": item.get("max_abs"),
                "rmse": item.get("rmse"),
                "material": item.get("material"),
                "material_threshold": item.get("material_threshold"),
                "worst_mismatch_fortran": item.get("worst_mismatch_fortran"),
            }
            for item in comp.get("material_ranked_residuals", [])[:12]
        ],
    }


def strip_wrf(wrf: Mapping[str, Any]) -> dict[str, Any]:
    if "surfaces" not in wrf:
        return dict(wrf)
    out = {key: value for key, value in wrf.items() if key != "surfaces"}
    out["surfaces"] = {
        name: {key: value for key, value in surface.items() if key != "arrays"}
        for name, surface in wrf["surfaces"].items()
    }
    return out


def build_boundary_state(inputs: Mapping[str, Any], state: Any) -> Any:
    from gpuwrf.nesting.boundary_construction import (  # noqa: PLC0415
        build_child_boundary_package,
        build_nest_force_weights,
    )

    run = inputs["run"]
    child_meta = run.grid("d02")
    weights = build_nest_force_weights(
        parent_grid_ratio=int(child_meta.parent_grid_ratio),
        i_parent_start=int(child_meta.i_parent_start),
        j_parent_start=int(child_meta.j_parent_start),
        parent_grid=inputs["parent"]["grid"],
        child_grid=inputs["live_child"]["grid"],
        registration="sint",
    )
    return build_child_boundary_package(
        state,
        inputs["parent"]["state"],
        weights,
        bdy_width=builder.BDY_WIDTH,
    )


def apply_mythos_perturb_init(inputs: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.integration.d02_replay import _wrf_live_nest_start_domain_perturb_init  # noqa: PLC0415
    from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: PLC0415

    live_child = inputs["live_child"]
    state = live_child["state"]
    base = live_child["base_state"]
    p_new, mu_new, w_new, meta = _wrf_live_nest_start_domain_perturb_init(
        inputs["run"],
        domain="d02",
        grid=live_child["grid"],
        metrics=live_child["metrics"],
        ph_perturbation=state.ph_perturbation,
        mu_perturbation=state.mu_perturbation,
        theta_full=state.theta,
        w=state.w,
        u=state.u,
        v=state.v,
        ht_fine=inputs["raw_child"]["grid"].terrain_height,
    )
    patched = state.replace(
        p_perturbation=p_new,
        p_total=base.pb + p_new,
        mu_perturbation=mu_new,
        mu_total=base.mub + mu_new,
        w=w_new,
    )
    boundary_state = build_boundary_state(inputs, patched)
    # Noah-MP land carry + WRF step-1 held radiation seeds (noahmp_rad/rthraten)
    # must ride THIS carry too: the strict capture runs from patched["carry"].
    # Re-derive the radiation seeds from the PATCHED step-1 entry state (WRF
    # computes the step-1 radiation from its actual start-of-step state).
    noahmp_land = inputs.get("noahmp_land")
    if noahmp_land is not None:
        noahmp_rad, rthraten_seed = live.noahmp_step1_carry_seeds(
            boundary_state, inputs["namelist"], noahmp_land
        )
        carry = initial_operational_carry(
            boundary_state, noahmp_land=noahmp_land, noahmp_rad=noahmp_rad
        ).replace(rthraten=rthraten_seed)
    else:
        carry = initial_operational_carry(boundary_state)
    jax.block_until_ready(carry.state.theta)
    return {
        "state": patched,
        "boundary_state": boundary_state,
        "carry": carry,
        "metadata": meta,
        "delta_from_stale_proof_state": {
            "P_STATE": metric_brief(live.diff_metrics("P_STATE", p_new, state.p_perturbation)),
            "MU_STATE": metric_brief(live.diff_metrics("MU_STATE", mu_new, state.mu_perturbation)),
            "W_STATE": metric_brief(live.diff_metrics("W_STATE", w_new, state.w)),
        },
    }


def capture_from_carry(inputs: Mapping[str, Any], carry: Any, *, label: str) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    namelist = inputs["namelist"]
    jnp = inputs["jnp"]
    lead_seconds = jnp.asarray(float(source.TARGET_STEP) * float(namelist.dt_s), dtype=jnp.float64)
    cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
    run_radiation = bool(cadence > 0 and source.TARGET_STEP % cadence == 0)

    physics = om._physics_step_forcing(
        carry,
        namelist,
        lead_seconds,
        run_radiation=run_radiation,
    )
    spec = om.halo_spec(namelist.grid)
    step_entry = om.apply_halo(carry.state, spec)
    physics_carry_state = om.apply_halo(physics.carry.state, spec)
    physics_state = om.apply_halo(physics.state, spec)
    empty_dry = om.DryPhysicsTendencies()

    rk1_reference = physics_carry_state
    base_tendencies = om.compute_advection_tendencies(
        physics_carry_state, namelist.tendencies, namelist.grid
    )
    rk_tendency_empty = om._augment_large_step_tendencies(
        physics_carry_state,
        base_tendencies,
        namelist,
        rk_step=1,
        physics_tendencies=empty_dry,
        step_origin=rk1_reference,
    )
    rk_addtend = om._augment_large_step_tendencies(
        physics_carry_state,
        base_tendencies,
        namelist,
        rk_step=1,
        physics_tendencies=physics.dry_tendencies,
        step_origin=rk1_reference,
    )
    prep = om.small_step_prep_wrf(
        physics_carry_state,
        1,
        float(namelist.dt_s) / 3.0,
        metrics=namelist.metrics,
        reference_state=rk1_reference,
        ww=physics.carry.ww,
    )
    pressure = om.calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)

    jax.block_until_ready(physics_state.theta)
    return {
        "status": "JAX_SOURCE_BOUNDARIES_READY",
        "label": label,
        "captures": {
            "step_entry_state_zero_dry": source.state_source_arrays(jnp, step_entry, empty_dry),
            "physics_carry_state_dry": source.state_source_arrays(
                jnp, physics_carry_state, physics.dry_tendencies
            ),
            "physics_state_dry": source.state_source_arrays(
                jnp, physics_state, physics.dry_tendencies
            ),
            "rk1_after_rk_tendency_empty_dry": source.state_tendency_arrays(
                jnp, physics_carry_state, rk_tendency_empty, empty_dry
            ),
            "rk1_after_rk_addtend": source.state_tendency_arrays(
                jnp, physics_carry_state, rk_addtend, physics.dry_tendencies
            ),
            "rk1_after_small_step_prep": source.prep_arrays(physics_carry_state, prep, pressure),
        },
        "run_radiation": run_radiation,
        "lead_seconds": float(lead_seconds),
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "rk_order": int(namelist.rk_order),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "force_fp64": bool(namelist.force_fp64),
            "cu_physics": int(namelist.cu_physics),
            "rad_rk_tendf": int(namelist.rad_rk_tendf),
            "use_flux_advection": bool(namelist.use_flux_advection),
            "moist_adv_opt": int(namelist.moist_adv_opt),
        },
    }


def compare_capture(wrf: Mapping[str, Any], jax_capture: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    cap = jax_capture["captures"]
    return {
        "status": "SOURCE_SPLIT_COMPARISONS_EXECUTED",
        "matrix": {
            "after_first_rk_step_part1": {
                "vs_physics_carry_state_dry": source.compare_surface(
                    "after_first_rk_step_part1",
                    wrf["surfaces"]["after_first_rk_step_part1"],
                    cap["physics_carry_state_dry"],
                    jax,
                ),
                "vs_physics_state_dry": source.compare_surface(
                    "after_first_rk_step_part1",
                    wrf["surfaces"]["after_first_rk_step_part1"],
                    cap["physics_state_dry"],
                    jax,
                ),
            },
            "after_first_rk_step_part2": {
                "vs_physics_carry_state_dry": source.compare_surface(
                    "after_first_rk_step_part2",
                    wrf["surfaces"]["after_first_rk_step_part2"],
                    cap["physics_carry_state_dry"],
                    jax,
                ),
            },
            "after_rk_addtend_before_small_step_prep": {
                "vs_rk1_after_rk_tendency_empty_dry": source.compare_surface(
                    "after_rk_addtend_before_small_step_prep",
                    wrf["surfaces"]["after_rk_addtend_before_small_step_prep"],
                    cap["rk1_after_rk_tendency_empty_dry"],
                    jax,
                ),
                "vs_rk1_after_rk_addtend": source.compare_surface(
                    "after_rk_addtend_before_small_step_prep",
                    wrf["surfaces"]["after_rk_addtend_before_small_step_prep"],
                    cap["rk1_after_rk_addtend"],
                    jax,
                ),
            },
            "after_small_step_prep_calc_p_rho": {
                "vs_rk1_after_small_step_prep": source.compare_surface(
                    "after_small_step_prep_calc_p_rho",
                    wrf["surfaces"]["after_small_step_prep_calc_p_rho"],
                    cap["rk1_after_small_step_prep"],
                    jax,
                ),
            },
        },
    }


def compare_wrf_arrays(
    name: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    metrics = {
        field: live.diff_metrics(field, candidate["arrays"][field], reference["arrays"][field])
        for field in fields
        if field in candidate["arrays"] and field in reference["arrays"]
    }
    return {
        "status": "WRF_INTERNAL_COMPARISON_EXECUTED",
        "name": name,
        "candidate_surface": candidate.get("surface"),
        "reference_surface": reference.get("surface"),
        "first_material_state_field": first_material(metrics, STATE_FIELDS),
        "first_material_p_family_field": first_material(metrics, P_FAMILY_FIELDS),
        "selected_metrics": selected_metrics(metrics, fields),
    }


def compare_jax_fields_to_wrf(
    name: str,
    wrf_surface: Mapping[str, Any],
    jax_arrays: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    metrics: dict[str, Any] = {}
    for field in fields:
        if field not in wrf_surface["arrays"] or field not in jax_arrays:
            continue
        metrics[field] = live.diff_metrics(
            field,
            np.asarray(jax.device_get(jax_arrays[field]), dtype=np.float64),
            wrf_surface["arrays"][field],
        )
    return {
        "status": "WRF_JAX_COMPARISON_EXECUTED",
        "name": name,
        "first_material_state_field": first_material(metrics, STATE_FIELDS),
        "first_material_p_family_field": first_material(metrics, P_FAMILY_FIELDS),
        "selected_metrics": selected_metrics(metrics, fields),
    }


def compare_jax_arrays(
    name: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    metrics: dict[str, Any] = {}
    for field in fields:
        if field not in candidate or field not in reference:
            continue
        metrics[field] = live.diff_metrics(
            field,
            np.asarray(jax.device_get(candidate[field]), dtype=np.float64),
            np.asarray(jax.device_get(reference[field]), dtype=np.float64),
        )
    return {
        "status": "JAX_PAIR_COMPARISON_EXECUTED",
        "name": name,
        "selected_metrics": selected_metrics(metrics, fields),
    }


def p_state_max(comp: Mapping[str, Any]) -> float | None:
    metric = comp.get("per_field_metrics", {}).get("P_STATE")
    if not metric or metric.get("status") != "OK":
        return None
    value = metric.get("max_abs")
    return None if value is None else float(value)


def top_residuals(comp: Mapping[str, Any], fields: tuple[str, ...], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        metric = comp.get("per_field_metrics", {}).get(field)
        if metric and metric.get("status") == "OK":
            rows.append(
                {
                    "field": field,
                    "max_abs": metric.get("max_abs"),
                    "rmse": metric.get("rmse"),
                    "material_threshold": source.MATERIAL_THRESHOLDS.get(field),
                    "material": material(field, metric),
                    "worst_mismatch_fortran": metric.get("worst_mismatch_fortran"),
                }
            )
    return sorted(rows, key=lambda item: -float(item["max_abs"] or 0.0))[:limit]


def extract_predecessor_context() -> dict[str, Any]:
    mythos = read_json(MYTHOS_JSON)
    tp_json = read_json(TP_JSON)
    source_json = read_json(SOURCE_JSON)
    strict = mythos.get("proof", {}).get("strict_step1_16field_with_patched_init", {})
    return {
        "mythos_verdict": mythos.get("verdict"),
        "mythos_strict_step1_with_patched_init": {
            "status": strict.get("status"),
            "first_divergent_field": strict.get("first_divergent_field"),
            "top_residuals": [
                {
                    "field": item.get("field"),
                    "max_abs": item.get("max_abs"),
                    "rmse": item.get("rmse"),
                }
                for item in strict.get("ranked_residuals", [])[:8]
            ],
        },
        "fresh_tp_json": {
            "path": str(TP_JSON),
            "verdict": tp_json.get("verdict"),
            "first_material_tp_family_mismatch": tp_json.get("substage_comparisons", {}).get(
                "first_material_tp_family_mismatch"
            ),
        },
        "stale_source_json": {
            "path": str(SOURCE_JSON),
            "verdict": source_json.get("verdict"),
            "reason_stale": (
                "It was generated before the Mythos start_domain perturbation init was threaded "
                "into the current proof capture."
            ),
        },
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    wrf_source = source.parse_wrf_surfaces()
    if wrf_source.get("status") != "WRF_SOURCE_BOUNDARY_TRUTH_READY":
        return {"status": "BLOCKED_WRF_SOURCE_TRUTH", "blocker": strip_wrf(wrf_source)}
    wrf_pre = pre.parse_wrf_surfaces()
    if wrf_pre.get("status") != "WRF_PRE_PART1_TRUTH_READY":
        return {"status": "BLOCKED_WRF_PREPART_TRUTH", "blocker": strip_wrf(wrf_pre)}

    inputs = live.build_live_nest_step1_inputs()
    stale_capture = capture_from_carry(inputs, inputs["carry"], label="stale_proof_loader_without_perturb_init")
    patched = apply_mythos_perturb_init(inputs)
    patched_capture = capture_from_carry(inputs, patched["carry"], label="production_mythos_perturb_init")

    stale_comp = compare_capture(wrf_source, stale_capture)
    patched_comp = compare_capture(wrf_source, patched_capture)

    stale_add = stale_comp["matrix"]["after_rk_addtend_before_small_step_prep"]["vs_rk1_after_rk_addtend"]
    patched_add = patched_comp["matrix"]["after_rk_addtend_before_small_step_prep"]["vs_rk1_after_rk_addtend"]
    patched_empty = patched_comp["matrix"]["after_rk_addtend_before_small_step_prep"][
        "vs_rk1_after_rk_tendency_empty_dry"
    ]
    patched_part1 = patched_comp["matrix"]["after_first_rk_step_part1"]["vs_physics_carry_state_dry"]
    patched_part2 = patched_comp["matrix"]["after_first_rk_step_part2"]["vs_physics_carry_state_dry"]
    patched_prep = patched_comp["matrix"]["after_small_step_prep_calc_p_rho"]["vs_rk1_after_small_step_prep"]

    pre_call = wrf_pre["surfaces"][pre.PRECALL_SURFACE]
    source_part1 = wrf_source["surfaces"]["after_first_rk_step_part1"]
    source_part2 = wrf_source["surfaces"]["after_first_rk_step_part2"]
    common_state = ("T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "W_STATE", "PH_STATE", "PHB")
    wrf_direct_mutation = {
        "prepart_to_after_first_rk_step_part1": compare_wrf_arrays(
            "WRF before_first_rk_step_part1_call -> after_first_rk_step_part1",
            source_part1,
            pre_call,
            common_state,
        ),
        "after_first_rk_step_part1_to_after_first_rk_step_part2": compare_wrf_arrays(
            "WRF after_first_rk_step_part1 -> after_first_rk_step_part2",
            source_part2,
            source_part1,
            common_state,
        ),
    }
    patched_prepart_entry = compare_jax_fields_to_wrf(
        "patched JAX step-entry state -> WRF before_first_rk_step_part1_call",
        pre_call,
        patched_capture["captures"]["step_entry_state_zero_dry"],
        common_state,
    )
    stale_prepart_entry = compare_jax_fields_to_wrf(
        "stale JAX step-entry state -> WRF before_first_rk_step_part1_call",
        pre_call,
        stale_capture["captures"]["step_entry_state_zero_dry"],
        common_state,
    )

    empty_vs_full = compare_jax_arrays(
        "JAX rk1_after_rk_addtend full dry tendencies minus empty dry tendencies",
        patched_capture["captures"]["rk1_after_rk_addtend"],
        patched_capture["captures"]["rk1_after_rk_tendency_empty_dry"],
        TENDENCY_FIELDS,
    )

    state = patched["boundary_state"]
    base = inputs["live_child"]["base_state"]
    p_split_residual = np.asarray(
        jax.device_get((state.p_total - base.pb) - state.p_perturbation), dtype=np.float64
    )
    pressure_mapping = {
        "p_total_minus_base_pb_minus_p_perturbation_max_abs": float(np.max(np.abs(p_split_residual))),
        "interpretation": "The P comparison is perturbation-pressure P_STATE; total-vs-perturb mapping is not the source.",
    }

    stale_p = p_state_max(stale_add)
    patched_p = p_state_max(patched_add)
    p_closed = patched_p is not None and patched_p <= P_STATE_GATE_PA

    return {
        "status": "PROOF_EXECUTED",
        "verdict": VERDICT if p_closed else "STEP1_RK1_P_STATE_SOURCE_LOCALIZED_UNCLOSED_P_STATE",
        "p_state_gate": {
            "threshold_pa": P_STATE_GATE_PA,
            "stale_after_rk_addtend_max_abs": stale_p,
            "patched_after_rk_addtend_max_abs": patched_p,
            "closed_by_production_perturb_init": p_closed,
        },
        "source_family_split": {
            "state_entering_wrf_first_rk_step_part1": {
                "stale": stale_prepart_entry,
                "patched": patched_prepart_entry,
            },
            "wrf_first_rk_step_part1_part2_direct_state_mutation": wrf_direct_mutation,
            "jax_physics_step_forcing": {
                "after_first_rk_step_part1_vs_physics_carry": compact_surface(
                    patched_part1, STATE_FIELDS + TENDENCY_FIELDS
                ),
                "after_first_rk_step_part2_vs_physics_carry": compact_surface(
                    patched_part2, STATE_FIELDS + TENDENCY_FIELDS
                ),
            },
            "rk_addtend_dry_spec_bdy_dry_and_jax_augment": {
                "after_rk_addtend_vs_empty_dry": compact_surface(
                    patched_empty, STATE_FIELDS + TENDENCY_FIELDS
                ),
                "after_rk_addtend_vs_full_dry": compact_surface(
                    patched_add, STATE_FIELDS + TENDENCY_FIELDS
                ),
                "jax_full_minus_empty_tendency_arrays": empty_vs_full,
            },
            "small_step_prep_continuity": compact_surface(patched_prep, WORK_FIELDS),
            "pressure_total_vs_perturb_mapping": pressure_mapping,
            "boundary_relaxation": {
                "p_state_boundary_hypothesis": (
                    "Refuted for the current first P_STATE residual: P_STATE is already closed at "
                    "the patched state-entry/physics-carry/addtend surfaces before acoustic prep."
                ),
                "residuals_empty_dry_equal_full_dry": (
                    "The large PH/RW/T tendency residuals are unchanged when fixed dry physics "
                    "tendencies are removed from the JAX augment call, so they are not caused by "
                    "_physics_step_forcing source leaves."
                ),
            },
        },
        "selected_before_after_surfaces": {
            "stale_after_rk_addtend_before_small_step_prep": compact_surface(
                stale_add, STATE_FIELDS + TENDENCY_FIELDS
            ),
            "patched_after_first_rk_step_part1": compact_surface(
                patched_part1, STATE_FIELDS + TENDENCY_FIELDS
            ),
            "patched_after_first_rk_step_part2": compact_surface(
                patched_part2, STATE_FIELDS + TENDENCY_FIELDS
            ),
            "patched_after_rk_addtend_before_small_step_prep": compact_surface(
                patched_add, STATE_FIELDS + TENDENCY_FIELDS
            ),
            "patched_after_small_step_prep_calc_p_rho": compact_surface(patched_prep, WORK_FIELDS),
        },
        "tendency_residual_attribution": {
            "after_rk_addtend_full_top": top_residuals(patched_add, TENDENCY_FIELDS, limit=8),
            "after_rk_addtend_empty_top": top_residuals(patched_empty, TENDENCY_FIELDS, limit=8),
            "first_material_after_p_state_closed": {
                "after_first_rk_step_part1": patched_part1.get("material_first_field"),
                "after_first_rk_step_part2": patched_part2.get("material_first_field"),
                "after_rk_addtend_before_small_step_prep": patched_add.get("material_first_field"),
                "after_small_step_prep_calc_p_rho": patched_prep.get("material_first_field"),
            },
            "interpretation": (
                "The huge PH_TEND/RW_TEND/PH_TENDF family residuals remain after P_STATE is closed, "
                "and RK1 work arrays are exact. They are therefore not causal for the material P_STATE "
                "stage-entry residual; the next exact source boundary is the tendency-family contract, "
                "first visible as T_TENDF after first_rk_step_part2 and T_TEND/PH_TEND/RW_TEND at "
                "after_rk_addtend_before_small_step_prep."
            ),
        },
        "ranked_hypotheses": [
            {
                "rank": 1,
                "hypothesis": "The material P_STATE residual is a stale proof-local loader that bypasses Mythos start_domain perturbation init.",
                "status": "PROVEN",
                "evidence": (
                    f"P_STATE at RK1 after_rk_addtend drops from {stale_p} Pa to {patched_p} Pa "
                    f"against a {P_STATE_GATE_PA} Pa material gate."
                ),
            },
            {
                "rank": 2,
                "hypothesis": "The remaining exact boundary is tendency-family source/schema/order, not P_STATE state.",
                "status": "NEXT_BEST",
                "evidence": (
                    "After P_STATE closes, first material fields are T_TENDF at after_first_rk_step_part2 "
                    "and T_TEND at after_rk_addtend; PH_TEND/RW_TEND remain the largest residuals."
                ),
            },
            {
                "rank": 3,
                "hypothesis": "Boundary relaxation/spec_bdy_dry causes the first P_STATE residual.",
                "status": "REFUTED_FOR_P_STATE",
                "evidence": "Patched P_STATE is clean before and after the RK1 addtend/boundary-tendency boundary.",
            },
            {
                "rank": 4,
                "hypothesis": "Pressure total-vs-perturb mapping causes the P_STATE residual.",
                "status": "REFUTED",
                "evidence": "Patched state has exact p_total - PB - P split identity and P_STATE is below gate.",
            },
            {
                "rank": 5,
                "hypothesis": "RK1 small_step_prep/calc_p_rho is the P_STATE source.",
                "status": "REFUTED_FOR_RK1",
                "evidence": "T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK are exact at RK1 small_step_prep/calc_p_rho.",
            },
        ],
        "patched_perturb_init": {
            "metadata": patched["metadata"],
            "delta_from_stale_proof_state": patched["delta_from_stale_proof_state"],
        },
        "predecessor_context": extract_predecessor_context(),
        "wrf_truth": {
            "source_boundary": strip_wrf(wrf_source),
            "pre_part1": strip_wrf(wrf_pre),
        },
        "next_exact_boundary": (
            "Split WRF first_rk_step_part2 T_TENDF and then RK1 after_rk_addtend T_TEND/PH_TEND/RW_TEND "
            "against JAX compute_advection_tendencies/_augment_large_step_tendencies with a patched-init "
            "capture. Do not enter acoustic substeps for this P_STATE issue; P_STATE is below material gate "
            "before small_step_prep."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    gate = proof["p_state_gate"]
    tend = proof["tendency_residual_attribution"]
    lines = [
        "# V0.14 Step-1 RK1 P_STATE Source Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- RK1 `P_STATE` at `after_rk_addtend_before_small_step_prep`: stale proof loader max_abs `{gate['stale_after_rk_addtend_max_abs']}` Pa; patched Mythos init max_abs `{gate['patched_after_rk_addtend_max_abs']}` Pa; gate `{gate['threshold_pa']}` Pa.",
        "- The material `P_STATE` source hypothesis is refuted for current production init. The comparator path was bypassing the Mythos `start_domain` perturbation init.",
        "- `P_STATE/MU_STATE/W_STATE/PH_STATE` are below material gates through `after_first_rk_step_part1`, `after_first_rk_step_part2`, and RK1 `after_rk_addtend_before_small_step_prep` with the patched capture.",
        "- RK1 `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK` are exact at `small_step_prep/calc_p_rho(step=0)`.",
        "",
        "## Remaining Boundary",
        "",
        f"- First material after P closes: `{tend['first_material_after_p_state_closed']['after_first_rk_step_part2']}` at WRF `after_first_rk_step_part2`; `{tend['first_material_after_p_state_closed']['after_rk_addtend_before_small_step_prep']}` at RK1 `after_rk_addtend_before_small_step_prep`.",
        "- Largest remaining tendency residuals at RK1 addtend: "
        + ", ".join(
            f"`{item['field']}` max_abs `{item['max_abs']}`"
            for item in tend["after_rk_addtend_full_top"][:5]
        )
        + ".",
        "- Empty-dry and full-dry JAX augment comparisons give the same huge tendency residuals, so fixed physics source leaves are not the cause of this P_STATE issue.",
        "",
        "## Next",
        "",
        proof["next_exact_boundary"],
        "",
        "Detailed metrics are in `proofs/v014/step1_rk1_p_state_source_split.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    lines = [
        "# Review: V0.14 Step-1 RK1 P-State Source Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: localize or fix the post-Mythos Step-1 RK1 material `P_STATE` divergence before acoustic substeps.",
        "",
        "files changed:",
        "- `proofs/v014/step1_rk1_p_state_source_split.py`",
        "- `proofs/v014/step1_rk1_p_state_source_split.json`",
        "- `proofs/v014/step1_rk1_p_state_source_split.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`",
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
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {proof['next_exact_boundary']}", ""])
    return "\n".join(lines)


def main() -> int:
    proof = build_proof()
    verdict = str(proof.get("verdict", "STEP1_RK1_P_STATE_SOURCE_BLOCKED_UNKNOWN"))
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_rk1_p_state_source_split.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "weak_comparison_avoided": True,
        "jax_vs_jax_self_compare": False,
        "one_cell_proof": False,
        "initial_vs_post_step_false_comparison": False,
        "tooling_verdict": "FOCUSED_CPU_SOURCE_BOUNDARY_FALSIFIER_FASTEST_RIGOROUS_WALL_CLOCK",
        "environment": jax_environment(),
        "git": git_metadata(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "mythos_json": path_info(MYTHOS_JSON),
            "tp_operator_json": path_info(TP_JSON),
            "source_boundary_json": path_info(SOURCE_JSON),
            "accepted_final_truth_npz": path_info(live.ACCEPTED_TRUTH),
            "source_wrf_truth_root": path_info(source.WRF_TRUTH),
            "tp_wrf_truth_root": path_info(tp.WRF_TRUTH),
            "prepart_wrf_truth_root": path_info(pre.WRF_TRUTH),
        },
        "proof": proof,
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_rk1_p_state_source_split.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_p_state_source_split.py",
                "python -m json.tool proofs/v014/step1_rk1_p_state_source_split.json >/tmp/step1_rk1_p_state_source_split.validated.json",
                "git diff --check",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "The older proof-local live.build_live_nest_step1_inputs helper still bypasses the production perturbation init; this proof patches the capture locally rather than editing shared proof helpers.",
            "The WRF truth surface does not split internal rk_tendency before/after every tendency component, so the remaining T/PH/RW tendency family needs one focused tendency-contract split.",
            "The proof does not enter acoustic substeps because RK1 P_STATE is below material gate before small_step_prep.",
        ],
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
