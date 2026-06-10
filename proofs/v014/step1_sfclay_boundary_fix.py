#!/usr/bin/env python3
"""V0.14 Step-1 MYNN surface-layer boundary proof.

CPU-only proof for the Step-1 surface-layer boundary feeding MYNN.  It tests the
WRF ``itimestep<=1`` MYNN surface-layer branch against the current Step-1 WRF
MYNN-driver hook, then reruns the strict source-fidelity metric through the
updated production path.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
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

import mynn_driver_source_output_fix as mfix  # noqa: E402
import step1_dry_source_leaf_fix as dryfix  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part2_source_leaves_split as split  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_sfclay_boundary_fix.json"
OUT_MD = PROOF_DIR / "step1_sfclay_boundary_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-sfclay-boundary.md"

SCRATCH = Path("/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609")
WRF_SF_MYNN = SCRATCH / "WRF/phys/module_sf_mynn.F"
WRF_SURFACE_DRIVER = SCRATCH / "WRF/phys/module_surface_driver.F"

DT_S = 6.0
DX_M = 3000.0
STRICT_PASS_MAX_ABS = 1.0e-3
STRICT_PASS_RMSE = 1.0e-5


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


def path_info(path: Path) -> dict[str, Any]:
    return mfix.path_info(path)


def run_command(command: list[str], *, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "PYTHONPATH": "src",
        }
    )
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-3000:],
            "stderr_tail": proc.stderr[-3000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timeout_s": int(timeout_s),
            "stdout_tail": (exc.stdout or "")[-3000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-3000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def compare_flux_state(state: Any, wrf: Mapping[str, np.ndarray]) -> dict[str, Any]:
    from gpuwrf.coupling.physics_couplers import _surface_fluxes_from_state  # noqa: PLC0415

    flux = _surface_fluxes_from_state(state)
    return {
        "ustar_vs_wrf_ust": mfix.diffstat(np.asarray(flux.ustar), wrf["ust"]),
        "theta_flux_vs_wrf_flt": mfix.diffstat(np.asarray(flux.theta_flux), wrf["flt"]),
        "qv_flux_vs_wrf_flqv": mfix.diffstat(np.asarray(flux.qv_flux), wrf["flqv"]),
        "fltv_vs_wrf_fltv": mfix.diffstat(np.asarray(flux.fltv), wrf["fltv"]),
    }


def improvement(new: Mapping[str, Any], old: Mapping[str, Any], key: str = "rmse") -> dict[str, Any]:
    old_v = old.get(key)
    new_v = new.get(key)
    if old_v is None or new_v is None:
        return {"old": old_v, "new": new_v, "ratio_new_over_old": None, "reduced": False}
    ratio = float(new_v) / max(float(old_v), 1.0e-300)
    return {"old": float(old_v), "new": float(new_v), "ratio_new_over_old": ratio, "reduced": ratio < 1.0}


def masked_diffstat(candidate: Any, reference: Any, mask: Any) -> dict[str, Any]:
    mask_arr = np.asarray(mask, dtype=bool)
    return mfix.diffstat(np.asarray(candidate, dtype=np.float64)[mask_arr], np.asarray(reference, dtype=np.float64)[mask_arr])


def run_mynn_response(state: Any, wrf_rth: np.ndarray, wrf_rqv: np.ndarray, grid: Any) -> dict[str, Any]:
    import jax.numpy as jnp  # noqa: PLC0415
    from gpuwrf.coupling.physics_couplers import (  # noqa: PLC0415
        _flatten_columns_to_batch,
        _from_columns,
        _mynn_column_from_state,
        _surface_fluxes_from_state,
        _unflatten_batch_to_columns,
    )
    from gpuwrf.physics.mynn_pbl import step_mynn_pbl_column  # noqa: PLC0415

    column = _mynn_column_from_state(state, grid)
    surface = _surface_fluxes_from_state(state)
    ny, nx = column.theta.shape[0], column.theta.shape[1]
    col_b = _flatten_columns_to_batch(column, ny, nx)
    surface_b = _flatten_columns_to_batch(surface, ny, nx)
    out_b = step_mynn_pbl_column(col_b, DT_S, debug=False, surface=surface_b, edmf=True, dx=DX_M)
    out = _unflatten_batch_to_columns(out_b, ny, nx)
    th_before = np.asarray(_from_columns(column.theta), dtype=np.float64)
    qv_before = np.asarray(_from_columns(column.qv), dtype=np.float64)
    th_after = np.asarray(_from_columns(out.theta), dtype=np.float64)
    qv_after = np.asarray(_from_columns(out.qv), dtype=np.float64)
    return {
        "rthblten_vs_wrf_raw": mfix.tendency_stat((th_after - th_before) / DT_S, wrf_rth),
        "rqvblten_vs_wrf_raw": mfix.tendency_stat((qv_after - qv_before) / DT_S, wrf_rqv),
        "jax_output_summary": {
            "theta_tendency_max_abs": float(np.nanmax(np.abs((th_after - th_before) / DT_S))),
            "qv_tendency_max_abs": float(np.nanmax(np.abs((qv_after - qv_before) / DT_S))),
            "surface_ustar_max": float(np.nanmax(np.asarray(surface.ustar))),
            "surface_theta_flux_max_abs": float(np.nanmax(np.abs(np.asarray(surface.theta_flux)))),
            "surface_qv_flux_max_abs": float(np.nanmax(np.abs(np.asarray(surface.qv_flux)))),
        },
        "shape": {"ny": int(ny), "nx": int(nx)},
    }


def strict_step1_metric(inputs: Mapping[str, Any], carry: Any) -> dict[str, Any] | None:
    shapes = split.expected_shapes()
    part2 = split.parse_part2_surfaces(shapes)
    if part2.get("status") != "WRF_PART2_TRUTH_READY":
        return {"status": "BLOCKED_PART2_TRUTH", "blocker": split.strip_arrays(part2)}
    source_surfaces = split.parse_existing_source_surfaces(shapes)
    source_save = split.parse_source_save()
    if source_surfaces.get("status") != "WRF_SOURCE_SURFACES_READY":
        return {"status": "BLOCKED_SOURCE_SURFACES", "blocker": split.strip_arrays(source_surfaces)}
    if source_save.get("status") != "SOURCE_SAVE_READY":
        return {"status": "BLOCKED_SOURCE_SAVE", "blocker": source_save}
    capture = dryfix.build_source_capture(
        inputs,
        carry,
        label="step1_sfclay_boundary_fix",
        force_radiation=False,
    )
    if capture.get("status") != "JAX_TENDENCY_BOUNDARIES_READY":
        return {"status": "BLOCKED_CAPTURE", "capture": capture}
    formulas = split.compare_stage_formulas(
        part2,
        source_surfaces,
        source_save,
        {"capture": capture, "patched": {"carry": carry}},
    )
    strict = split.compact_metric(
        formulas["comparisons"]["after_conv_t_tendf_vs_current_jax_dry_t_tendf"]["nested_interior"]
    )
    return {"status": "STRICT_METRIC_READY", "metric": strict, "capture": dryfix.compact_capture(capture)}


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from gpuwrf.coupling.physics_couplers import (  # noqa: PLC0415
        _surface_column_view,
        surface_adapter,
    )
    from gpuwrf.coupling.physics_dispatch import DEFAULT_MP_PHYSICS  # noqa: PLC0415
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS, SFCLAY_SCAN_ADAPTERS  # noqa: PLC0415
    from gpuwrf.physics.surface_constants import XLV  # noqa: PLC0415
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    hooks = mfix.parse_hook_set(mfix.HOOK_ROOT)
    if hooks is None:
        return {"status": "BLOCKED_HOOK_TRUTH_MISSING", "hook_root": str(mfix.HOOK_ROOT)}
    pre_c, pre_s = hooks["pre_c"], hooks["pre_s"]
    post_c = hooks["post_c"]
    wrf = mfix.wrf_kinematic_fluxes(pre_c, pre_s)

    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    namelist = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    state0 = patched["carry"].state
    state = state0
    if int(namelist.mp_physics) == DEFAULT_MP_PHYSICS:
        state = om.thompson_adapter(state, DT_S)
    elif int(namelist.mp_physics) in MP_SCAN_ADAPTERS:
        state = MP_SCAN_ADAPTERS[int(namelist.mp_physics)](state, DT_S, namelist.grid)

    if int(namelist.sf_sfclay_physics) in SFCLAY_SCAN_ADAPTERS:
        return {
            "status": "BLOCKED_NOT_MYNN_SURFACE_LAYER",
            "sf_sfclay_physics": int(namelist.sf_sfclay_physics),
        }

    warm_state = surface_adapter(state, DT_S, first_timestep=False)
    first_state = surface_adapter(state, DT_S, first_timestep=True)
    warm_diag = surface_layer_with_diagnostics(_surface_column_view(state), first_timestep=False)
    first_diag = surface_layer_with_diagnostics(_surface_column_view(state), first_timestep=True)

    u0 = np.asarray(state.u, dtype=np.float64)
    v0 = np.asarray(state.v, dtype=np.float64)
    jax_wspd0 = np.hypot(
        0.5 * (u0[0, :, :-1] + u0[0, :, 1:]),
        0.5 * (v0[0, :-1, :] + v0[0, 1:, :]),
    )
    wrf_first_ust_guess = np.maximum(0.04 * np.hypot(pre_c["COL_U"][0], pre_c["COL_V"][0]), 0.001)
    jax_first_ust_guess = np.maximum(0.04 * jax_wspd0, 0.001)
    qv0 = np.asarray(state.qv[0], dtype=np.float64)
    qsfc_first_formula = qv0 / (1.0 + qv0)
    is_land = np.asarray(state0.xland, dtype=np.float64) < 1.5

    surface_metrics = {
        "warm": compare_flux_state(warm_state, wrf),
        "first_timestep": compare_flux_state(first_state, wrf),
        "wrf_first_ust_guess_vs_jax_first_formula": mfix.diffstat(jax_first_ust_guess, wrf_first_ust_guess),
        "first_qsfc_all_vs_qv_over_1_plus_qv": mfix.diffstat(np.asarray(first_diag.qsfc), qsfc_first_formula),
        "first_qsfc_land_vs_qv_over_1_plus_qv": masked_diffstat(
            np.asarray(first_diag.qsfc),
            qsfc_first_formula,
            is_land,
        ),
        "warm_qsfc_vs_first_qsfc": mfix.diffstat(np.asarray(warm_diag.qsfc), np.asarray(first_diag.qsfc)),
        "diag_hfx_vs_wrf": {
            "warm": mfix.diffstat(np.asarray(warm_diag.hfx), wrf["hfx"]),
            "first_timestep": mfix.diffstat(np.asarray(first_diag.hfx), wrf["hfx"]),
        },
        "diag_qfx_vs_wrf": {
            "warm": mfix.diffstat(np.asarray(warm_diag.lh) / XLV, wrf["qfx"]),
            "first_timestep": mfix.diffstat(np.asarray(first_diag.lh) / XLV, wrf["qfx"]),
        },
        "tskin_vs_wrf_ts": mfix.diffstat(np.asarray(state0.t_skin), wrf["ts"]),
        "znt_input_vs_wrf_znt": mfix.diffstat(np.asarray(state0.roughness_m), pre_s[9]),
        "surface_znt_after_mynn_vs_wrf_znt": {
            "warm": mfix.diffstat(np.asarray(warm_diag.znt), pre_s[9]),
            "first_timestep": mfix.diffstat(np.asarray(first_diag.znt), pre_s[9]),
        },
    }
    surface_improvements = {
        "ustar_rmse": improvement(
            surface_metrics["first_timestep"]["ustar_vs_wrf_ust"],
            surface_metrics["warm"]["ustar_vs_wrf_ust"],
        ),
        "theta_flux_rmse": improvement(
            surface_metrics["first_timestep"]["theta_flux_vs_wrf_flt"],
            surface_metrics["warm"]["theta_flux_vs_wrf_flt"],
        ),
        "qv_flux_rmse": improvement(
            surface_metrics["first_timestep"]["qv_flux_vs_wrf_flqv"],
            surface_metrics["warm"]["qv_flux_vs_wrf_flqv"],
        ),
    }

    warm_mynn = run_mynn_response(warm_state, post_c["COL_RTHBLTEN"], post_c["COL_RQVBLTEN"], namelist.grid)
    first_mynn = run_mynn_response(first_state, post_c["COL_RTHBLTEN"], post_c["COL_RQVBLTEN"], namelist.grid)
    mynn_improvements = {
        "rthblten_rmse": improvement(
            first_mynn["rthblten_vs_wrf_raw"],
            warm_mynn["rthblten_vs_wrf_raw"],
        ),
        "rqvblten_rmse": improvement(
            first_mynn["rqvblten_vs_wrf_raw"],
            warm_mynn["rqvblten_vs_wrf_raw"],
        ),
    }

    strict = strict_step1_metric(inputs, patched["carry"])
    strict_metric = strict.get("metric") if isinstance(strict, Mapping) else None
    strict_closed = bool(
        strict_metric
        and strict_metric.get("max_abs") is not None
        and float(strict_metric["max_abs"]) <= STRICT_PASS_MAX_ABS
        and float(strict_metric["rmse"]) <= STRICT_PASS_RMSE
    )

    first_call_qsfc_exact = surface_metrics["first_qsfc_land_vs_qv_over_1_plus_qv"]["max_abs"] <= 1.0e-12
    flux_any_improved = any(item["reduced"] for item in surface_improvements.values())
    mynn_any_improved = any(item["reduced"] for item in mynn_improvements.values())

    if strict_closed:
        verdict = "STEP1_SFCLAY_BOUNDARY_CLOSED"
    elif first_call_qsfc_exact and (flux_any_improved or mynn_any_improved):
        verdict = "STEP1_SFCLAY_FIRST_CALL_FIXED_NEXT_BLOCKER_TSK_ZNT_SURFACE_INPUTS"
    elif first_call_qsfc_exact:
        verdict = "STEP1_SFCLAY_FIRST_CALL_IMPLEMENTED_BUT_NOT_DOMINANT"
    else:
        verdict = "STEP1_SFCLAY_FIRST_CALL_GATE_FAILED"

    ranked_findings = [
        {
            "rank": 1,
            "status": "BLOCKING" if not strict_closed else "CLOSED",
            "hypothesis": "WRF first-call MYNN surface semantics were missing from the JAX Step-1 surface boundary.",
            "evidence": {
                "qsfc_exact_gate": first_call_qsfc_exact,
                "surface_improvements": surface_improvements,
                "mynn_improvements": mynn_improvements,
                "strict_metric": strict_metric,
            },
        },
        {
            "rank": 2,
            "status": "SURVIVES" if not strict_closed else "SECONDARY",
            "hypothesis": "Skin-temperature and roughness sourcing remain the narrower WRF-anchored blocker.",
            "evidence": {
                "tskin_vs_wrf_ts": surface_metrics["tskin_vs_wrf_ts"],
                "znt_input_vs_wrf_znt": surface_metrics["znt_input_vs_wrf_znt"],
                "surface_znt_after_mynn_vs_wrf_znt": surface_metrics["surface_znt_after_mynn_vs_wrf_znt"],
            },
        },
        {
            "rank": 3,
            "status": "RULED_OUT",
            "hypothesis": "MYNN kernel algebra or MYNN cold-start QKE initialization is the active Step-1 blocker.",
            "evidence": (
                "Accepted by proofs/v014/mynn_driver_source_output_fix.md; this proof changes only "
                "the surface boundary and reuses the same WRF MYNN-driver hook."
            ),
        },
    ]

    next_boundary = (
        "Strict Step-1 is closed; draft sprint closeout can proceed."
        if strict_closed
        else (
            "Narrow next blocker: WRF-anchored TSK/ZNT surface input sourcing before sfclay_mynn. "
            "Fastest next command is a tiny surface-driver hook around module_surface_driver/module_sf_mynn "
            "for incoming TSK/ZNT/UST/QSFC/MOL and outgoing UST/HFX/QFX/ZNT on the current d02 Step-1 case, "
            "then compare those exact arrays against JAX _surface_column_view inputs and diagnostics."
        )
    )

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.step1_sfclay_boundary_fix.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "cpu_only": True,
        },
        "target": {
            "domain": 2,
            "step": 1,
            "dt_s": DT_S,
            "dx_m": DX_M,
            "strict_pass": {"max_abs": STRICT_PASS_MAX_ABS, "rmse": STRICT_PASS_RMSE},
        },
        "wrf_truth": {
            "mynn_driver_hook_root": str(mfix.HOOK_ROOT),
            "hook_files": hooks["paths"],
            "module_sf_mynn_source": path_info(WRF_SF_MYNN),
            "module_surface_driver_source": path_info(WRF_SURFACE_DRIVER),
            "new_wrf_hook_used": False,
        },
        "production_change": {
            "summary": (
                "Threaded first_timestep through MYNN surface_layer/surface_adapter. "
                "On first Step-1 calls, UST=max(0.04*sqrt(u^2+v^2),0.001), MOL=0, "
                "QSFC=qv/(1+qv), and zolrib starts from Li_etal_2010."
            ),
            "files": [
                "src/gpuwrf/physics/surface_layer.py",
                "src/gpuwrf/coupling/physics_couplers.py",
                "src/gpuwrf/integration/d02_replay.py",
                "src/gpuwrf/runtime/operational_mode.py",
                "proofs/v014/step1_tendency_contract_split.py",
                "proofs/v014/mynn_driver_source_output_fix.py",
                "proofs/v014/step1_source_fidelity_closure.py",
                "tests/test_m6_surface_layer_kernel.py",
            ],
        },
        "surface_boundary_metrics": surface_metrics,
        "surface_improvements": surface_improvements,
        "mynn_response": {"warm": warm_mynn, "first_timestep": first_mynn},
        "mynn_improvements": mynn_improvements,
        "strict_step1": strict,
        "gates": {
            "first_call_qsfc_exact": first_call_qsfc_exact,
            "surface_any_rmse_reduced": flux_any_improved,
            "mynn_any_rmse_reduced": mynn_any_improved,
            "strict_step1_closed": strict_closed,
        },
        "ranked_findings": ranked_findings,
        "single_remaining_blocker": ranked_findings[1] if not strict_closed else None,
        "next_boundary": next_boundary,
        "commands": {
            "focused_test": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_m6_surface_layer_kernel.py",
            "proof": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_sfclay_boundary_fix.py",
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": run_command(["git", "status", "--short"]),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return (
            "# V0.14 Step-1 SFCLAY Boundary Fix\n\n"
            f"Blocked: `{payload.get('status')}`. See `proofs/v014/step1_sfclay_boundary_fix.json`.\n"
        )

    s = payload["strict_step1"].get("metric") if payload.get("strict_step1") else {}
    surf = payload["surface_boundary_metrics"]
    imp = payload["surface_improvements"]
    myimp = payload["mynn_improvements"]
    lines = [
        "# V0.14 Step-1 SFCLAY Boundary Fix",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Production Fix",
        "",
        "- Added WRF MYNN surface first-call semantics to `surface_layer(..., first_timestep=True)`:",
        "  UST first guess, MOL=0, QSFC=qv/(1+qv), and Li_etal_2010 z/L seed.",
        "- Threaded Step-1 flags through d02 replay and operational `_physics_step_forcing`; updated the Step-1 proof helpers.",
        "",
        "## WRF-Anchored Evidence",
        "",
        f"- First-call land QSFC gate: max_abs `{surf['first_qsfc_land_vs_qv_over_1_plus_qv']['max_abs']}`.",
        f"- All-domain QSFC diagnostic (water recomputes inside WRF): max_abs `{surf['first_qsfc_all_vs_qv_over_1_plus_qv']['max_abs']}`.",
        f"- UST rmse warm -> first: `{imp['ustar_rmse']['old']}` -> `{imp['ustar_rmse']['new']}`.",
        f"- theta-flux rmse warm -> first: `{imp['theta_flux_rmse']['old']}` -> `{imp['theta_flux_rmse']['new']}`.",
        f"- qv-flux rmse warm -> first: `{imp['qv_flux_rmse']['old']}` -> `{imp['qv_flux_rmse']['new']}`.",
        f"- MYNN RTHBLTEN rmse warm -> first: `{myimp['rthblten_rmse']['old']}` -> `{myimp['rthblten_rmse']['new']}`.",
        "",
        "## Strict Step-1",
        "",
        f"- after-conv `T_TENDF` max_abs `{s.get('max_abs')}`, rmse `{s.get('rmse')}`.",
        "",
        "## Remaining Blocker",
        "",
        payload["next_boundary"],
        "",
        "Proof objects: `proofs/v014/step1_sfclay_boundary_fix.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return (
            "# Review: V0.14 Step-1 SFCLAY Boundary\n\n"
            f"Blocked: `{payload.get('status')}`.\n"
        )
    s = payload["strict_step1"].get("metric") if payload.get("strict_step1") else {}
    blocker = payload.get("single_remaining_blocker")
    return "\n".join(
        [
            "# Review: V0.14 Step-1 SFCLAY Boundary",
            "",
            f"Verdict: `{payload['verdict']}`.",
            "",
            "Implemented WRF `itimestep<=1` MYNN surface semantics in production and direct Step-1 proof helpers.",
            f"Strict Step-1 after-conv metric: max_abs `{s.get('max_abs')}`, rmse `{s.get('rmse')}`.",
            "",
            "Primary residual status:",
            json.dumps(sanitize_json(blocker), indent=2, sort_keys=True),
            "",
            f"Next boundary: {payload['next_boundary']}",
            "",
        ]
    )


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_REVIEW}")
    print(payload.get("verdict", payload.get("status")))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
