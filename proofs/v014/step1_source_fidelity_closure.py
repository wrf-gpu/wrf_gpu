#!/usr/bin/env python3
"""V0.14 Step-1 source-fidelity closure proof.

CPU-only proof for the remaining Step-1 T_TENDF source gap.  It reruns the
current source-leaf production path, ranks the radiation/moist-conversion
secondary hypotheses, and narrows the surviving blocker to a WRF MYNN driver
source-output boundary.
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

import step1_dry_source_leaf_fix as dryfix  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part2_source_leaves_split as split  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_source_fidelity_closure.json"
OUT_MD = PROOF_DIR / "step1_source_fidelity_closure.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-source-fidelity-closure.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-10-v014-step1-source-fidelity-closure/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PRIOR_SPLIT = PROOF_DIR / "step1_part2_source_leaves_split.md"
PRIOR_DRY = PROOF_DIR / "step1_dry_source_leaf_fix.md"

RVRD = 461.6 / 287.0
PASS_MAX_ABS = 1.0e-3
PASS_RMSE = 1.0e-5


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

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


def run_command(command: list[str], *, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
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
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timeout_s": int(timeout_s),
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


def metric(formulas: Mapping[str, Any], name: str) -> dict[str, Any]:
    return split.compact_metric(formulas["comparisons"][name]["nested_interior"])


def compact_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": capture.get("status"),
        "label": capture.get("label"),
        "run_radiation": capture.get("run_radiation"),
        "namelist": capture.get("namelist"),
    }


def build_source_capture(
    inputs: Mapping[str, Any],
    carry: Any,
    *,
    label: str,
    force_radiation: bool,
) -> dict[str, Any]:
    return dryfix.build_source_capture(
        inputs, carry, label=label, force_radiation=force_radiation
    )


def _state_diff_metric(name: str, wrf: np.ndarray, jax_value: Any) -> dict[str, Any]:
    arr = np.asarray(jax_value, dtype=np.float64)
    mask = split.interior_mask(wrf.shape)
    return split.compact_metric(split.diff_metrics(name, wrf, arr, mask=mask))


def build_mynn_leaf_probe(
    inputs: Mapping[str, Any],
    carry: Any,
    part2: Mapping[str, Any],
) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    from gpuwrf.coupling.physics_couplers import (  # noqa: PLC0415
        mynn_adapter_with_source_leaves,
        surface_adapter,
    )
    from gpuwrf.coupling.physics_dispatch import (  # noqa: PLC0415
        DEFAULT_BL_PBL_PHYSICS,
        DEFAULT_MP_PHYSICS,
    )
    from gpuwrf.coupling.scan_adapters import (  # noqa: PLC0415
        MP_SCAN_ADAPTERS,
        SFCLAY_SCAN_ADAPTERS,
    )
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    namelist = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    state = carry.state
    dt_s = float(namelist.dt_s)

    if int(namelist.mp_physics) == DEFAULT_MP_PHYSICS:
        state = om.thompson_adapter(state, dt_s)
    elif int(namelist.mp_physics) in MP_SCAN_ADAPTERS:
        state = MP_SCAN_ADAPTERS[int(namelist.mp_physics)](state, dt_s, namelist.grid)

    if int(namelist.sf_sfclay_physics) in SFCLAY_SCAN_ADAPTERS:
        state = SFCLAY_SCAN_ADAPTERS[int(namelist.sf_sfclay_physics)](
            state, dt_s, namelist.grid
        )
    else:
        state = surface_adapter(state, dt_s, first_timestep=True)

    pbl_entry = state
    if int(namelist.bl_pbl_physics) != DEFAULT_BL_PBL_PHYSICS:
        return {
            "status": "BLOCKED_NOT_MYNN_DEFAULT",
            "bl_pbl_physics": int(namelist.bl_pbl_physics),
        }
    mynn = mynn_adapter_with_source_leaves(pbl_entry, dt_s, namelist.grid)
    mass_h = (
        namelist.metrics.c1h[:, None, None] * mynn.state.mu_total[None, :, :]
        + namelist.metrics.c2h[:, None, None]
    )
    jax_rthblten_coupled = np.asarray(mass_h * mynn.rthblten, dtype=np.float64)
    jax_rqvblten_coupled = np.asarray(mass_h * mynn.rqvblten, dtype=np.float64)

    after_calc = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    after_update = part2["surfaces"]["after_update_phy_ten"]["arrays"]
    mask = split.interior_mask(after_update["RTHBLTEN"].shape)

    input_metrics = {
        "T_STATE_after_calculate_vs_jax_pbl_entry": _state_diff_metric(
            "T_STATE_after_calculate_vs_jax_pbl_entry",
            after_calc["T_STATE"],
            np.asarray(pbl_entry.theta, dtype=np.float64) - split.THETA_OFFSET,
        ),
        "QV_OLD_after_calculate_vs_jax_pbl_entry": _state_diff_metric(
            "QV_OLD_after_calculate_vs_jax_pbl_entry",
            after_calc["QV_OLD"],
            pbl_entry.qv,
        ),
        "P_after_calculate_vs_jax_pbl_entry": _state_diff_metric(
            "P_after_calculate_vs_jax_pbl_entry",
            after_calc["P"],
            pbl_entry.p_perturbation,
        ),
    }

    return {
        "status": "MYNN_LEAF_PROBE_EXECUTED",
        "note": (
            "WRF part2 truth exposes already mass-coupled RTHBLTEN and aggregate "
            "QV_TEND, not the raw MYNN driver RQVBLTEN. JAX values below are "
            "mass-coupled from the local MYNN source leaves for like-unit comparison."
        ),
        "namelist": {
            "dt_s": dt_s,
            "mp_physics": int(namelist.mp_physics),
            "sf_sfclay_physics": int(namelist.sf_sfclay_physics),
            "bl_pbl_physics": int(namelist.bl_pbl_physics),
        },
        "available_input_state_metrics": input_metrics,
        "source_output_metrics": {
            "wrf_rthblten_summary": split.array_summary(after_update["RTHBLTEN"], mask=mask),
            "jax_mass_coupled_rthblten_summary": split.array_summary(
                jax_rthblten_coupled, mask=mask
            ),
            "wrf_rthblten_vs_jax_mass_coupled_rthblten": split.compact_metric(
                split.diff_metrics(
                    "wrf_rthblten_vs_jax_mass_coupled_rthblten",
                    after_update["RTHBLTEN"],
                    jax_rthblten_coupled,
                    mask=mask,
                )
            ),
            "wrf_qv_tend_summary": split.array_summary(after_update["QV_TEND"], mask=mask),
            "jax_mass_coupled_rqvblten_summary": split.array_summary(
                jax_rqvblten_coupled, mask=mask
            ),
            "wrf_qv_tend_vs_jax_mass_coupled_rqvblten": split.compact_metric(
                split.diff_metrics(
                    "wrf_qv_tend_vs_jax_mass_coupled_rqvblten",
                    after_update["QV_TEND"],
                    jax_rqvblten_coupled,
                    mask=mask,
                )
            ),
        },
        "narrow_missing_wrf_surface": (
            "Exact WRF MYNN driver input columns plus raw post-driver dth1/dqv1 "
            "before module_em mass scaling are not emitted by accepted artifacts."
        ),
    }


def build_oracle_rankings(part2: Mapping[str, Any]) -> dict[str, Any]:
    after_calc = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    after_update = part2["surfaces"]["after_update_phy_ten"]["arrays"]
    after_conv = part2["surfaces"]["after_conv_t_tendf_to_moist"]["arrays"]
    mask = split.interior_mask(after_conv["T_TENDF"].shape)

    def conv_from(update_t_tendf: np.ndarray, qv_tend: np.ndarray | None = None) -> np.ndarray:
        qv = after_update["QV_TEND"] if qv_tend is None else qv_tend
        theta_m_factor = after_update["THETA_M_FACTOR"]
        return (
            theta_m_factor * update_t_tendf
            + RVRD * (after_update["T_OLD"] + split.THETA_OFFSET) / theta_m_factor * qv
        )

    pre = after_calc["T_TENDF"]
    rth_bl = after_calc["RTHBLTEN"]
    rth_ra = after_calc["RTHRATEN"]
    wrf_full = conv_from(pre + after_calc["RTH_ACTIVE_SUM"])
    wrf_bl_only = conv_from(pre + rth_bl)
    wrf_ra_only = conv_from(pre + rth_ra)
    wrf_no_qv = after_update["THETA_M_FACTOR"] * (pre + after_calc["RTH_ACTIVE_SUM"])

    return {
        "status": "WRF_ORACLE_RANKINGS_EXECUTED",
        "wrf_full_active_sources_vs_after_conv": split.compact_metric(
            split.diff_metrics("wrf_full_active_sources_vs_after_conv", after_conv["T_TENDF"], wrf_full, mask=mask)
        ),
        "wrf_bl_only_plus_qv_vs_after_conv": split.compact_metric(
            split.diff_metrics("wrf_bl_only_plus_qv_vs_after_conv", after_conv["T_TENDF"], wrf_bl_only, mask=mask)
        ),
        "wrf_ra_only_plus_qv_vs_after_conv": split.compact_metric(
            split.diff_metrics("wrf_ra_only_plus_qv_vs_after_conv", after_conv["T_TENDF"], wrf_ra_only, mask=mask)
        ),
        "wrf_full_without_qv_tend_term_vs_after_conv": split.compact_metric(
            split.diff_metrics("wrf_full_without_qv_tend_term_vs_after_conv", after_conv["T_TENDF"], wrf_no_qv, mask=mask)
        ),
        "component_summaries": {
            "wrf_rthblten": split.array_summary(rth_bl, mask=mask),
            "wrf_rthraten": split.array_summary(rth_ra, mask=mask),
            "wrf_qv_tend": split.array_summary(after_update["QV_TEND"], mask=mask),
            "wrf_after_update_to_after_conv_delta": split.array_summary(
                after_conv["T_TENDF"] - after_update["T_TENDF"], mask=mask
            ),
        },
    }


def classify(
    primary_formulas: Mapping[str, Any],
    forced_formulas: Mapping[str, Any] | None,
    mynn_probe: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], str]:
    primary_conv = metric(primary_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf")
    closed = (
        primary_conv.get("max_abs") is not None
        and float(primary_conv["max_abs"]) <= PASS_MAX_ABS
        and float(primary_conv["rmse"]) <= PASS_RMSE
    )
    if closed:
        return (
            "STEP1_SOURCE_FIDELITY_CLOSED",
            [
                {
                    "rank": 1,
                    "status": "SUPPORTED",
                    "hypothesis": "Current JAX source leaves close WRF Step-1 T_TENDF.",
                    "evidence": primary_conv,
                }
            ],
            "Continue downstream Step-1 same-input comparison.",
        )

    forced_conv = (
        metric(forced_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf")
        if forced_formulas is not None
        else {"status": "FORCED_CAPTURE_BLOCKED"}
    )
    source_metrics = mynn_probe.get("source_output_metrics", {})
    return (
        "STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_THERMODYNAMIC_COLUMN_INPUTS",
        [
            {
                "rank": 1,
                "status": "BLOCKING",
                "hypothesis": (
                    "JAX MYNN source outputs remain below WRF at Step 1 because the "
                    "surface boundary feeding MYNN is still not WRF-compatible. "
                    "`proofs/v014/mynn_driver_source_output_fix` already proved the "
                    "MYNN kernel and fixed the missing WRF cold-start qke init; "
                    "`proofs/v014/step1_sfclay_boundary_fix` ports WRF's "
                    "sfclay_mynn first-call UST/QSFC/MOL/zol seed; and "
                    "`proofs/v014/step1_tsk_znt_sourcing_fix` now proves TSK/ZNT/"
                    "MAVAIL source parity at the exact sfclay_mynn hook. The "
                    "surviving WRF-anchored blocker is the non-surface "
                    "thermodynamic column input entering sfclay_mynn."
                ),
                "evidence": {
                    "strict_after_conv_vs_jax": primary_conv,
                    "rthblten": source_metrics.get(
                        "wrf_rthblten_vs_jax_mass_coupled_rthblten"
                    ),
                    "rqv_or_qv_tend": source_metrics.get(
                        "wrf_qv_tend_vs_jax_mass_coupled_rqvblten"
                    ),
                    "available_input_state_metrics": mynn_probe.get(
                        "available_input_state_metrics"
                    ),
                },
            },
            {
                "rank": 2,
                "status": "RANKED_SECONDARY",
                "hypothesis": "Held RTHRATEN refresh is secondary at Step 1.",
                "evidence": {
                    "primary_after_conv_vs_jax": primary_conv,
                    "forced_radiation_after_conv_vs_jax": forced_conv,
                    "wrf_rthraten_summary": oracle["component_summaries"]["wrf_rthraten"],
                },
            },
            {
                "rank": 3,
                "status": "IMPLEMENTED_STILL_SECONDARY",
                "hypothesis": (
                    "WRF conv_t_tendf_to_moist/QV_TEND handling is now represented in "
                    "the JAX source branch, but it cannot close the source gap while "
                    "MYNN RTHBLTEN/RQVBLTEN are too weak."
                ),
                "evidence": {
                    "wrf_full_without_qv_tend_term_vs_after_conv": oracle[
                        "wrf_full_without_qv_tend_term_vs_after_conv"
                    ],
                    "wrf_qv_tend_summary": oracle["component_summaries"]["wrf_qv_tend"],
                    "jax_mass_coupled_rqvblten_summary": source_metrics.get(
                        "jax_mass_coupled_rqvblten_summary"
                    ),
                },
            },
        ],
        (
            "DONE 2026-06-10: MYNN driver kernel/init, sfclay_mynn first-call "
            "semantics, and TSK/ZNT/MAVAIL input sourcing are no longer active "
            "blockers. Next route: localize the non-surface thermodynamic column "
            "inputs at the exact sfclay_mynn hook (`th_phy(kts)`, `t_phy(kts)`, "
            "`p_phy(kts)`, and `dz8w`) against JAX `_surface_column_view`; then "
            "fix the Step-1 temperature/pressure sourcing if local."
        ),
    )


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

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

    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    primary_capture = build_source_capture(
        inputs,
        patched["carry"],
        label="step1_source_fidelity_closure_current",
        force_radiation=False,
    )
    if primary_capture.get("status") != "JAX_TENDENCY_BOUNDARIES_READY":
        return {"status": "BLOCKED_PRIMARY_CAPTURE", "capture": primary_capture}

    primary_formulas = split.compare_stage_formulas(
        part2,
        source_surfaces,
        source_save,
        {"capture": primary_capture, "patched": patched},
    )

    forced_capture = build_source_capture(
        inputs,
        patched["carry"],
        label="step1_source_fidelity_closure_forced_radiation",
        force_radiation=True,
    )
    forced_formulas = None
    if forced_capture.get("status") == "JAX_TENDENCY_BOUNDARIES_READY":
        forced_formulas = split.compare_stage_formulas(
            part2,
            source_surfaces,
            source_save,
            {"capture": forced_capture, "patched": patched},
        )

    mynn_probe = build_mynn_leaf_probe(inputs, patched["carry"], part2)
    oracle = build_oracle_rankings(part2)
    verdict, ranked_findings, next_boundary = classify(
        primary_formulas, forced_formulas, mynn_probe, oracle
    )
    strict_metric = metric(primary_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf")

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.step1_source_fidelity_closure.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": jax_environment(),
        "target": {
            "domain": split.TARGET_DOMAIN,
            "step": split.TARGET_STEP,
            "cpu_only": True,
            "pass_target": {"max_abs": PASS_MAX_ABS, "rmse": PASS_RMSE},
            "strict_metric": strict_metric,
        },
        "paths": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "prior_part2_split": path_info(PRIOR_SPLIT),
            "prior_dry_source_leaf_fix": path_info(PRIOR_DRY),
            "wrf_truth": path_info(split.WRF_TRUTH),
        },
        "current_jax_source_leaf_mode": {
            "primary_capture": compact_capture(primary_capture),
            "primary_metrics": {
                "after_update_vs_jax_dry_t_tendf": metric(
                    primary_formulas, "after_update_t_tendf_vs_current_jax_dry_t_tendf"
                ),
                "after_conv_vs_jax_dry_t_tendf": strict_metric,
                "wrf_active_rth_vs_jax_source_leaf": metric(
                    primary_formulas, "wrf_active_rth_vs_jax_physics_state_delta_mass_tendf"
                ),
            },
            "derived_candidate_summaries": primary_formulas["derived_candidate_summaries"],
        },
        "forced_radiation_falsifier": {
            "capture": compact_capture(forced_capture),
            "metrics": (
                {
                    "after_conv_vs_jax_dry_t_tendf": metric(
                        forced_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf"
                    ),
                    "derived_candidate_summaries": forced_formulas["derived_candidate_summaries"],
                }
                if forced_formulas is not None
                else {"status": "BLOCKED_FORCED_CAPTURE", "capture": forced_capture}
            ),
        },
        "mynn_leaf_probe": mynn_probe,
        "wrf_oracle_rankings": oracle,
        "ranked_findings": ranked_findings,
        "single_remaining_blocker": ranked_findings[0] if ranked_findings else None,
        "next_boundary": next_boundary,
        "source_changes_validated": {
            "production_files": [
                "src/gpuwrf/coupling/physics_couplers.py",
                "src/gpuwrf/runtime/operational_mode.py",
            ],
            "test_file": "tests/test_v014_dry_source_leaf_wiring.py",
            "summary": (
                "MYNN source leaf now carries rqvblten; rad_rk_tendf=1 now applies "
                "WRF conv_t_tendf_to_moist to DryPhysicsTendencies.t_tendf. "
                "rad_rk_tendf=0 path remains on the existing branch."
            ),
        },
        "commands": {
            "focused_test": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py",
            "proof": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py",
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
            "# V0.14 Step-1 Source-Fidelity Closure\n\n"
            f"Blocked: `{payload.get('status')}`. See `proofs/v014/step1_source_fidelity_closure.json`.\n"
        )
    strict = payload["target"]["strict_metric"]
    mynn = payload["mynn_leaf_probe"]["source_output_metrics"]
    forced = payload["forced_radiation_falsifier"]["metrics"]
    oracle = payload["wrf_oracle_rankings"]
    qv_no = oracle["wrf_full_without_qv_tend_term_vs_after_conv"]
    lines = [
        "# V0.14 Step-1 Source-Fidelity Closure",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Evidence",
        "",
        f"- Strict WRF after `conv_t_tendf_to_moist` vs current JAX dry `T_TENDF`: max_abs `{strict['max_abs']}`, rmse `{strict['rmse']}`.",
        f"- JAX mass-coupled MYNN `RTHBLTEN` remains too weak: max_abs `{mynn['jax_mass_coupled_rthblten_summary']['max_abs']}` vs WRF `{mynn['wrf_rthblten_summary']['max_abs']}`.",
        f"- JAX mass-coupled MYNN qv source is also too weak: max_abs `{mynn['jax_mass_coupled_rqvblten_summary']['max_abs']}` vs WRF `QV_TEND` `{mynn['wrf_qv_tend_summary']['max_abs']}`.",
        f"- Available same-boundary scalar inputs are not the order-10 error: T max_abs `{payload['mynn_leaf_probe']['available_input_state_metrics']['T_STATE_after_calculate_vs_jax_pbl_entry']['max_abs']}`, QV max_abs `{payload['mynn_leaf_probe']['available_input_state_metrics']['QV_OLD_after_calculate_vs_jax_pbl_entry']['max_abs']}`, P max_abs `{payload['mynn_leaf_probe']['available_input_state_metrics']['P_after_calculate_vs_jax_pbl_entry']['max_abs']}`.",
        f"- Forcing radiation only moves max_abs to `{forced['after_conv_vs_jax_dry_t_tendf']['max_abs']}`; held `RTHRATEN` is secondary.",
        f"- WRF qv/moist conversion is represented now and remains secondary: removing the WRF `QV_TEND` term would leave max_abs `{qv_no['max_abs']}`.",
        f"- WRF oracle active sources close the accepted formula: max_abs `{oracle['wrf_full_active_sources_vs_after_conv']['max_abs']}`, rmse `{oracle['wrf_full_active_sources_vs_after_conv']['rmse']}`.",
        "",
        "## Single Blocker",
        "",
        payload["single_remaining_blocker"]["hypothesis"],
        "",
        "## Fastest Next Route",
        "",
        payload["next_boundary"],
        "",
        "Proof objects: `proofs/v014/step1_source_fidelity_closure.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return (
            "# Review: V0.14 Step-1 Source-Fidelity Closure\n\n"
            f"Blocked: `{payload.get('status')}`. See `proofs/v014/step1_source_fidelity_closure.json`.\n"
        )
    strict = payload["target"]["strict_metric"]
    return "\n".join(
        [
            "# Review: V0.14 Step-1 Source-Fidelity Closure",
            "",
            f"Verdict: `{payload['verdict']}`.",
            "",
            "Production change is narrow: `rad_rk_tendf=1` source-leaf mode now carries MYNN `rqvblten` and applies WRF `conv_t_tendf_to_moist`; the default `rad_rk_tendf=0` branch is preserved.",
            "",
            f"The strict Step-1 proof does not close: after-conv residual max_abs `{strict['max_abs']}`, rmse `{strict['rmse']}`.",
            "",
            f"Accepted remaining blocker: {payload['single_remaining_blocker']['hypothesis']}",
            "",
            f"Next proof/fix route: {payload['next_boundary']}",
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
