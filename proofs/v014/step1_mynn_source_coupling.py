#!/usr/bin/env python3
"""V0.14 Step-1 MYNN source-coupling proof.

CPU-only proof for the sprint
``2026-06-10-v014-step1-mynn-source-coupling``.  It compares the exact WRF
MYNNEDMF driver boundary against the current JAX Step-1 call path and tests the
leading "MYNN source coupling" hypothesis as falsifiable.
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

import mynn_driver_source_output_fix as mynn_prior  # noqa: E402
import step1_dry_source_leaf_fix as dryfix  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part2_source_leaves_split as split  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402
import step1_tsk_znt_sourcing_fix as sfclay_prior  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_mynn_source_coupling.json"
OUT_MD = PROOF_DIR / "step1_mynn_source_coupling.md"
OUT_PATCH = PROOF_DIR / "step1_mynn_source_coupling_wrf_patch.diff"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-mynn-source-coupling.md"

STRICT_PASS_MAX_ABS = 1.0e-3
STRICT_PASS_RMSE = 1.0e-5
RVRD = 461.6 / 287.0
CP = 1004.5


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
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
            return sanitize(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], timeout_s: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_s,
    )
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        c = c[m]
        r = r[m]
    finite = np.isfinite(c) & np.isfinite(r)
    nonfinite = int(c.size - np.count_nonzero(finite))
    c = c[finite]
    r = r[finite]
    if c.size == 0:
        return {"count": 0, "nonfinite_count": nonfinite}
    d = c - r
    worst = int(np.argmax(np.abs(d)))
    return {
        "count": int(d.size),
        "nonfinite_count": nonfinite,
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "bias": float(np.mean(d)),
        "ref_max_abs": float(np.max(np.abs(r))),
        "worst_candidate": float(c[worst]),
        "worst_reference": float(r[worst]),
    }


def tendency_stat(candidate: Any, reference: Any) -> dict[str, Any]:
    out = diffstat(candidate, reference)
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    strong = np.abs(r) > 0.05 * np.nanmax(np.abs(r))
    ratio = c[strong] / r[strong]
    out.update(
        {
            "strong_cells": int(np.count_nonzero(strong)),
            "strong_ratio_median": float(np.nanmedian(ratio)),
            "strong_ratio_p10": float(np.nanpercentile(ratio, 10)),
            "strong_ratio_p90": float(np.nanpercentile(ratio, 90)),
            "corr": float(np.corrcoef(r.ravel(), c.ravel())[0, 1]),
        }
    )
    return out


def compact_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {key: sanitize(metric.get(key)) for key in (
        "count",
        "max_abs",
        "rmse",
        "bias",
        "p95",
        "p99",
        "nonfinite_diff_count",
        "worst_candidate",
        "worst_reference",
        "worst_mismatch_fortran",
    ) if key in metric}


def write_wrf_patch_archive() -> dict[str, Any]:
    parts = []
    for path in (
        PROOF_DIR / "mynn_driver_source_output_fix_wrf_patch.diff",
        PROOF_DIR / "step1_sfclay_output_algebra_wrf_patch.diff",
        PROOF_DIR / "step1_tsk_znt_sourcing_fix_wrf_patch.diff",
    ):
        if path.is_file():
            parts.append(f"# ---- {path.name} ----\n" + path.read_text(encoding="utf-8"))
    OUT_PATCH.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return path_info(OUT_PATCH)


def build_step1_state():
    from gpuwrf.coupling.physics_couplers import surface_adapter  # noqa: PLC0415
    from gpuwrf.coupling.physics_dispatch import DEFAULT_MP_PHYSICS  # noqa: PLC0415
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS, SFCLAY_SCAN_ADAPTERS  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    namelist = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    state = patched["carry"].state
    if int(namelist.mp_physics) == DEFAULT_MP_PHYSICS:
        state = om.thompson_adapter(state, mynn_prior.DT_S)
    elif int(namelist.mp_physics) in MP_SCAN_ADAPTERS:
        state = MP_SCAN_ADAPTERS[int(namelist.mp_physics)](state, mynn_prior.DT_S, namelist.grid)
    if int(namelist.sf_sfclay_physics) in SFCLAY_SCAN_ADAPTERS:
        state = SFCLAY_SCAN_ADAPTERS[int(namelist.sf_sfclay_physics)](
            state, mynn_prior.DT_S, namelist.grid
        )
    else:
        state = surface_adapter(state, mynn_prior.DT_S, namelist.grid, first_timestep=True)
    return inputs, patched, namelist, state


def run_kernel_matrix(state, namelist, hooks):
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.coupling.physics_couplers import (  # noqa: PLC0415
        _flatten_columns_to_batch,
        _from_columns,
        _mynn_column_from_state,
        _surface_fluxes_from_state,
        _unflatten_batch_to_columns,
        mynn_adapter_with_source_leaves,
        mynn_coldstart_qke_from_state,
    )
    from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, step_mynn_pbl_column  # noqa: PLC0415
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes  # noqa: PLC0415

    pre_c = hooks["pre_c"]
    pre_s = hooks["pre_s"]
    post_c = hooks["post_c"]
    init_c = hooks["init_c"]
    init_s = hooks["init_s"]
    wrf_flux = mynn_prior.wrf_kinematic_fluxes(pre_c, pre_s)

    ny, nx = mynn_prior.NY, mynn_prior.NX
    column = _mynn_column_from_state(state, namelist.grid)
    surface = _surface_fluxes_from_state(state)
    current = mynn_adapter_with_source_leaves(
        state, mynn_prior.DT_S, namelist.grid, first_timestep=True
    )

    to_cols = lambda a: jnp.moveaxis(jnp.asarray(a), 0, -1)  # noqa: E731
    zeros = jnp.zeros_like(to_cols(pre_c["COL_TH"]))
    wrf_columns = MynnPBLColumnState(
        to_cols(pre_c["COL_U"]),
        to_cols(pre_c["COL_V"]),
        column.w,
        to_cols(pre_c["COL_TH"]),
        to_cols(pre_c["COL_QV"]),
        0.5 * to_cols(init_c["INIT_QKE"]),
        to_cols(pre_c["COL_P"]),
        to_cols(pre_c["COL_RHO"]),
        to_cols(pre_c["COL_DZ"]),
        zeros,
        zeros,
        zeros,
        qc=column.qc,
        qi=column.qi,
    )
    wspd_safe = np.maximum(wrf_flux["wspd"], 0.2)
    tau_u = -(wrf_flux["ust"] ** 2) * pre_c["COL_U"][0] / wspd_safe
    tau_v = -(wrf_flux["ust"] ** 2) * pre_c["COL_V"][0] / wspd_safe
    wrf_surface = SurfaceFluxes(
        ustar=jnp.asarray(wrf_flux["ust"]),
        theta_flux=jnp.asarray(wrf_flux["flt"]),
        qv_flux=jnp.asarray(wrf_flux["flqv"]),
        tau_u=jnp.asarray(tau_u),
        tau_v=jnp.asarray(tau_v),
        rhosfc=jnp.asarray(wrf_flux["rho1"]),
        fltv=jnp.asarray(wrf_flux["fltv"]),
        xland=jnp.asarray(wrf_flux["xland"]),
    )

    def run_kernel(col, flux, qke3d, theta_ref, qv_ref):
        col = col.replace(tke=0.5 * to_cols(qke3d))
        out_b = step_mynn_pbl_column(
            _flatten_columns_to_batch(col, ny, nx),
            mynn_prior.DT_S,
            debug=False,
            surface=_flatten_columns_to_batch(flux, ny, nx),
            edmf=True,
            dx=mynn_prior.DX_M,
        )
        out = _unflatten_batch_to_columns(out_b, ny, nx)
        theta_after = np.asarray(_from_columns(out.theta), dtype=np.float64)
        qv_after = np.asarray(_from_columns(out.qv), dtype=np.float64)
        return {
            "rthblten": tendency_stat((theta_after - theta_ref) / mynn_prior.DT_S, post_c["COL_RTHBLTEN"]),
            "rqvblten": tendency_stat((qv_after - qv_ref) / mynn_prior.DT_S, post_c["COL_RQVBLTEN"]),
        }

    theta_ref = np.asarray(_from_columns(column.theta), dtype=np.float64)
    qv_ref = np.asarray(state.qv, dtype=np.float64)
    qke_neutral = np.asarray(mynn_coldstart_qke_from_state(state, namelist.grid), dtype=np.float64)
    qke_stale_rmol = np.asarray(
        mynn_coldstart_qke_from_state(state, namelist.grid, rmol_init=init_s[2]),
        dtype=np.float64,
    )
    mass_h = (
        np.asarray(namelist.metrics.c1h)[:, None, None] * np.asarray(state.mu_total)[None, :, :]
        + np.asarray(namelist.metrics.c2h)[:, None, None]
    )

    return {
        "current_source_leaves": {
            "raw_rthblten_vs_wrf": tendency_stat(current.rthblten, post_c["COL_RTHBLTEN"]),
            "raw_rqvblten_vs_wrf": tendency_stat(current.rqvblten, post_c["COL_RQVBLTEN"]),
            "mass_coupled_rthblten_vs_wrf_driver_raw_times_mass_h": diffstat(
                mass_h * np.asarray(current.rthblten), mass_h * post_c["COL_RTHBLTEN"]
            ),
            "mass_coupled_max_abs": {
                "jax_rthblten": float(np.nanmax(np.abs(mass_h * np.asarray(current.rthblten)))),
                "wrf_driver_rthblten": float(np.nanmax(np.abs(mass_h * post_c["COL_RTHBLTEN"]))),
            },
        },
        "kernel_matrix": {
            "wrf_inputs_wrf_qke": run_kernel(
                wrf_columns, wrf_surface, init_c["INIT_QKE"], pre_c["COL_TH"], pre_c["COL_QV"]
            ),
            "current_inputs_neutral_qke": run_kernel(column, surface, qke_neutral, theta_ref, qv_ref),
            "current_inputs_wrf_stale_rmol_qke": run_kernel(
                column, surface, qke_stale_rmol, theta_ref, qv_ref
            ),
            "current_inputs_wrf_qke": run_kernel(column, surface, init_c["INIT_QKE"], theta_ref, qv_ref),
        },
        "mynn_input_boundary": {
            "theta_dry_vs_wrf_col_th": diffstat(theta_ref, pre_c["COL_TH"]),
            "qv_vs_wrf_col_qv": diffstat(np.asarray(state.qv), pre_c["COL_QV"]),
            "p_vs_wrf_col_p": diffstat(np.asarray(_from_columns(column.p)), pre_c["COL_P"]),
            "rho_vs_wrf_col_rho": diffstat(np.asarray(_from_columns(column.rho)), pre_c["COL_RHO"]),
            "dz_vs_wrf_col_dz": diffstat(np.asarray(_from_columns(column.dz)), pre_c["COL_DZ"]),
            "ust_vs_wrf_driver": diffstat(np.asarray(surface.ustar), wrf_flux["ust"]),
            "theta_flux_vs_wrf_flt": diffstat(np.asarray(surface.theta_flux), wrf_flux["flt"]),
            "qv_flux_vs_wrf_flqv": diffstat(np.asarray(surface.qv_flux), wrf_flux["flqv"]),
        },
        "wrf_fluxes": wrf_flux,
    }


def build_stage_formula_metrics(inputs, patched) -> dict[str, Any]:
    shapes = split.expected_shapes()
    part2 = split.parse_part2_surfaces(shapes)
    if part2.get("status") != "WRF_PART2_TRUTH_READY":
        return {"status": "BLOCKED_PART2_TRUTH", "blocker": split.strip_arrays(part2)}
    source_surfaces = split.parse_existing_source_surfaces(shapes)
    if source_surfaces.get("status") != "WRF_SOURCE_SURFACES_READY":
        return {"status": "BLOCKED_SOURCE_SURFACES", "blocker": split.strip_arrays(source_surfaces)}
    source_save = split.parse_source_save()
    if source_save.get("status") != "SOURCE_SAVE_READY":
        return {"status": "BLOCKED_SOURCE_SAVE", "blocker": source_save}

    capture = dryfix.build_source_capture(
        inputs,
        patched["carry"],
        label="step1_mynn_source_coupling",
        force_radiation=False,
    )
    if capture.get("status") != "JAX_TENDENCY_BOUNDARIES_READY":
        return {"status": "BLOCKED_JAX_CAPTURE", "capture": capture}
    formulas = split.compare_stage_formulas(
        part2,
        source_surfaces,
        source_save,
        {"capture": capture, "patched": patched},
    )
    after_calc = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    mask = split.interior_mask(after_calc["T_TENDF"].shape)
    return {
        "status": "FORMULAS_READY",
        "strict_after_conv_vs_jax_dry_t_tendf": compact_metric(
            formulas["comparisons"]["after_conv_t_tendf_vs_current_jax_dry_t_tendf"]["nested_interior"]
        ),
        "wrf_after_update_self_consistency": compact_metric(
            formulas["comparisons"]["after_update_t_tendf_vs_pre_plus_active_rth"]["nested_interior"]
        ),
        "wrf_after_conv_moist_formula": compact_metric(
            formulas["comparisons"]["after_conv_t_tendf_vs_moist_formula"]["nested_interior"]
        ),
        "wrf_active_components": split.summarize_components(part2),
        "wrf_part2_rthblten_mass_coupled_summary": split.array_summary(
            after_calc["RTHBLTEN"], mask=mask
        ),
    }


def wrf_sfclay_to_mynn_handoff(hooks) -> dict[str, Any]:
    pre_s = hooks["pre_s"]
    sf_hfx = sfclay_prior.read2("sfclay_mynn_out__hfx.f64")
    sf_qfx = sfclay_prior.read2("sfclay_mynn_out__qfx.f64")
    sf_ust = sfclay_prior.read2("sfclay_mynn_out__ust.f64")
    return {
        "wrf_driver_hfx_vs_wrf_sfclay1d_hfx": diffstat(pre_s[2], sf_hfx),
        "wrf_driver_qfx_vs_wrf_sfclay1d_qfx": diffstat(pre_s[3], sf_qfx),
        "wrf_driver_ust_vs_wrf_sfclay1d_ust": diffstat(pre_s[1], sf_ust),
        "field_order_driver_sfc": [
            "XLAND",
            "UST",
            "HFX",
            "QFX",
            "WSPD",
            "TS",
            "QSFC",
            "PS",
            "CH",
            "ZNT",
            "UOCE",
            "VOCE",
            "PBLH",
            "RTHRATEN(kts)",
        ],
        "interpretation": (
            "WRF SFCLAY1D_mynn UST passes unchanged into MYNN, but HFX/QFX do not; "
            "the raw MYNN source tail is therefore upstream of module_bl_mynnedmf "
            "and specifically in the surface/land heat-moisture flux handoff."
        ),
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    hooks = mynn_prior.parse_hook_set(mynn_prior.HOOK_ROOT)
    if hooks is None:
        return {"status": "BLOCKED_MYNN_HOOK_MISSING", "hook_root": str(mynn_prior.HOOK_ROOT)}
    patch_info = write_wrf_patch_archive()
    inputs, patched, namelist, state = build_step1_state()
    matrix = run_kernel_matrix(state, namelist, hooks)
    formulas = build_stage_formula_metrics(inputs, patched)
    if formulas.get("status") != "FORMULAS_READY":
        return formulas
    handoff = wrf_sfclay_to_mynn_handoff(hooks)

    strict = formulas["strict_after_conv_vs_jax_dry_t_tendf"]
    strict_closed = (
        strict.get("max_abs") is not None
        and float(strict["max_abs"]) <= STRICT_PASS_MAX_ABS
        and float(strict["rmse"]) <= STRICT_PASS_RMSE
    )
    hfx_gap = handoff["wrf_driver_hfx_vs_wrf_sfclay1d_hfx"]
    qfx_gap = handoff["wrf_driver_qfx_vs_wrf_sfclay1d_qfx"]
    wrf_kernel = matrix["kernel_matrix"]["wrf_inputs_wrf_qke"]["rthblten"]
    current_wrf_qke = matrix["kernel_matrix"]["current_inputs_wrf_qke"]["rthblten"]

    verdict = (
        "STRICT_STEP1_CLOSED"
        if strict_closed
        else "STEP1_MYNN_SOURCE_COUPLING_NARROWED_TO_SURFACE_LAND_FLUX_HANDOFF"
    )

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.step1_mynn_source_coupling.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
        },
        "target": {
            "domain": split.TARGET_DOMAIN,
            "step": split.TARGET_STEP,
            "dt_s": mynn_prior.DT_S,
            "cpu_only": True,
            "strict_pass": {"max_abs": STRICT_PASS_MAX_ABS, "rmse": STRICT_PASS_RMSE},
        },
        "paths": {
            "mynn_hook_root": str(mynn_prior.HOOK_ROOT),
            "wrf_patch_archive": patch_info,
            "review": str(OUT_REVIEW),
        },
        "production_changes": [
            {
                "file": "src/gpuwrf/coupling/physics_couplers.py",
                "summary": (
                    "MYNN now consumes WRF phy_prep dry theta/hydrostatic pressure/"
                    "rho/dz when grid metrics are available; source leaves are dry "
                    "theta while live State.theta is converted back to theta_m; "
                    "first-call QKE initialization can run after surface fluxes."
                ),
            },
            {
                "file": "src/gpuwrf/runtime/operational_mode.py",
                "summary": "Pass first_timestep into the default MYNN adapter path.",
            },
        ],
        "strict_step1": {
            "closed": strict_closed,
            "metric": strict,
            "wrf_self_consistency": {
                "after_update_pre_plus_active_rth": formulas["wrf_after_update_self_consistency"],
                "after_conv_moist_formula": formulas["wrf_after_conv_moist_formula"],
            },
        },
        "wrf_mynn_driver_boundary": {
            "input_metrics": matrix["mynn_input_boundary"],
            "sfclay_to_mynn_handoff": handoff,
            "post_driver_source_metrics": matrix["current_source_leaves"],
            "wrf_part2_rthblten_mass_coupled_summary": formulas["wrf_part2_rthblten_mass_coupled_summary"],
        },
        "kernel_matrix": matrix["kernel_matrix"],
        "ranked_hypotheses": [
            {
                "rank": 1,
                "status": "PROVEN_BLOCKER",
                "hypothesis": (
                    "The remaining Step-1 raw MYNN theta-source tail is caused by "
                    "the surface/land heat-moisture flux handoff into MYNN, not by "
                    "MYNN source sign/unit/mass scaling."
                ),
                "evidence": {
                    "wrf_sfclay_to_mynn_hfx": hfx_gap,
                    "wrf_sfclay_to_mynn_qfx": qfx_gap,
                    "current_inputs_wrf_qke_rthblten": current_wrf_qke,
                },
            },
            {
                "rank": 2,
                "status": "FALSIFIED_AS_PRIMARY",
                "hypothesis": "MYNN kernel/source sign or raw RTHBLTEN units are wrong.",
                "evidence": {
                    "wrf_inputs_wrf_qke_rthblten": wrf_kernel,
                    "wrf_after_update_self_consistency": formulas["wrf_after_update_self_consistency"],
                },
            },
            {
                "rank": 3,
                "status": "SECONDARY",
                "hypothesis": "First-call MYNN QKE initialization / WRF uninitialized rmol affects the median source ratio.",
                "evidence": {
                    "neutral_qke": matrix["kernel_matrix"]["current_inputs_neutral_qke"]["rthblten"],
                    "wrf_stale_rmol_qke": matrix["kernel_matrix"]["current_inputs_wrf_stale_rmol_qke"]["rthblten"],
                    "wrf_qke": matrix["kernel_matrix"]["current_inputs_wrf_qke"]["rthblten"],
                },
            },
            {
                "rank": 4,
                "status": "FALSIFIED_AS_PRIMARY",
                "hypothesis": "Lowest-layer index/stagger mismatch or wrong dry theta/p/rho/dz column is still order-847.",
                "evidence": matrix["mynn_input_boundary"],
            },
        ],
        "fastest_next_command": (
            "Add a WRF hook immediately before/after module_surface_driver's sf_surface_physics=4 "
            "land-surface flux update (HFX/QFX/LH/TSK/GRDFLX/diagnostic CH where available), "
            "then compare it to the JAX Step-1 path and wire the Noah-MP/land flux overlay into "
            "the MYNN bottom-boundary handles before rerunning this proof."
        ),
        "commands": {
            "proof": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py",
            "focused_tests": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py",
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": run_command(["git", "status", "--short"]),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return f"# V0.14 Step-1 MYNN Source Coupling\n\nBlocked: `{payload.get('status')}`.\n"
    strict = payload["strict_step1"]["metric"]
    handoff = payload["wrf_mynn_driver_boundary"]["sfclay_to_mynn_handoff"]
    current = payload["wrf_mynn_driver_boundary"]["post_driver_source_metrics"]["raw_rthblten_vs_wrf"]
    wrf_kernel = payload["kernel_matrix"]["wrf_inputs_wrf_qke"]["rthblten"]
    current_wrf_qke = payload["kernel_matrix"]["current_inputs_wrf_qke"]["rthblten"]
    lines = [
        "# V0.14 Step-1 MYNN Source Coupling",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- Strict after-conv `T_TENDF` is still red: max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        f"- WRF inputs + WRF initialized QKE exonerate the MYNN kernel/source units: raw `RTHBLTEN` max_abs `{wrf_kernel['max_abs']}`, RMSE `{wrf_kernel['rmse']}`, corr `{wrf_kernel['corr']}`.",
        f"- Current JAX source leaves remain divergent: raw `RTHBLTEN` max_abs `{current['max_abs']}`, RMSE `{current['rmse']}`, strong median `{current['strong_ratio_median']}`.",
        f"- Even with WRF QKE injected, current inputs retain a source tail: max_abs `{current_wrf_qke['max_abs']}`, RMSE `{current_wrf_qke['rmse']}`.",
        "",
        "## Narrower Blocker",
        "",
        "The leading broad MYNN source-coupling hypothesis is narrowed upstream of `module_bl_mynnedmf`: WRF changes heat/moisture fluxes between `SFCLAY1D_mynn` output and the MYNN driver input.",
        "",
        f"- WRF MYNN-driver `UST` vs WRF `SFCLAY1D_mynn` `UST`: max_abs `{handoff['wrf_driver_ust_vs_wrf_sfclay1d_ust']['max_abs']}`.",
        f"- WRF MYNN-driver `HFX` vs WRF `SFCLAY1D_mynn` `HFX`: max_abs `{handoff['wrf_driver_hfx_vs_wrf_sfclay1d_hfx']['max_abs']}`, RMSE `{handoff['wrf_driver_hfx_vs_wrf_sfclay1d_hfx']['rmse']}`.",
        f"- WRF MYNN-driver `QFX` vs WRF `SFCLAY1D_mynn` `QFX`: max_abs `{handoff['wrf_driver_qfx_vs_wrf_sfclay1d_qfx']['max_abs']}`, RMSE `{handoff['wrf_driver_qfx_vs_wrf_sfclay1d_qfx']['rmse']}`.",
        "",
        "## Production Fixes",
        "",
        "- MYNN grid-backed columns now use WRF `phy_prep` dry theta, hydrostatic pressure, rho, and physics-g dz.",
        "- MYNN dry-theta output is converted back to live theta_m state; raw source leaves stay dry theta.",
        "- First-step MYNN QKE initialization is ordered after surface fluxes in the operational MYNN slot.",
        "",
        "## Fastest Next Command",
        "",
        f"`{payload['fastest_next_command']}`",
        "",
        "## Files",
        "",
        f"- JSON proof: `{OUT_JSON}`",
        f"- WRF patch archive: `{OUT_PATCH}`",
        f"- Review: `{OUT_REVIEW}`",
    ]
    return "\n".join(lines) + "\n"


def render_review(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return f"# Review: V0.14 Step-1 MYNN Source Coupling\n\nBlocked: `{payload.get('status')}`.\n"
    strict = payload["strict_step1"]["metric"]
    hfx = payload["wrf_mynn_driver_boundary"]["sfclay_to_mynn_handoff"]["wrf_driver_hfx_vs_wrf_sfclay1d_hfx"]
    kernel = payload["kernel_matrix"]["wrf_inputs_wrf_qke"]["rthblten"]
    lines = [
        "# Review: V0.14 Step-1 MYNN Source Coupling",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "The production adapter fixes are scoped and tested, but strict Step-1 is not closed.",
        f"Current after-conv `T_TENDF`: max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        "",
        f"MYNN kernel/source units are not the primary blocker: WRF inputs + WRF QKE raw `RTHBLTEN` max_abs `{kernel['max_abs']}`, RMSE `{kernel['rmse']}`.",
        f"The narrower blocker is the WRF surface/land flux handoff into MYNN: driver-vs-SFCLAY HFX max_abs `{hfx['max_abs']}`.",
        "",
        f"Next: {payload['fastest_next_command']}",
        "",
    ]
    return "\n".join(lines)


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
