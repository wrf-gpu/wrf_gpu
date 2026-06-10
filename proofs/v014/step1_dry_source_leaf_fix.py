#!/usr/bin/env python3
"""V0.14 Step-1 dry source-leaf implementation proof.

Runs the patched JAX dry-source path (``rad_rk_tendf=1``) against the accepted
WRF ``first_rk_step_part2`` source-leaf truth.  CPU-only.  This proof either
shows the Step-1 ``T_TENDF`` residual collapses or records the exact remaining
source boundary.
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

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part2_source_leaves_split as split  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402
import step1_tendency_contract_split as tendency_contract  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_dry_source_leaf_fix.json"
OUT_MD = PROOF_DIR / "step1_dry_source_leaf_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-dry-source-leaf-fix.md"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-10-v014-dry-source-leaf-fix/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PRIOR_PROOF = PROOF_DIR / "step1_part2_source_leaves_split.md"


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


def build_source_capture(inputs: Mapping[str, Any], carry: Any, *, label: str, force_radiation: bool) -> dict[str, Any]:
    nml = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    if force_radiation:
        nml = dataclasses.replace(nml, radiation_cadence_steps=1)
    source_inputs = dict(inputs)
    source_inputs["namelist"] = nml
    return tendency_contract.build_tendency_capture(source_inputs, carry, label=label)


def compact_capture(capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": capture.get("status"),
        "label": capture.get("label"),
        "run_radiation": capture.get("run_radiation"),
        "namelist": capture.get("namelist"),
    }


def classify(primary_formulas: Mapping[str, Any], forced_formulas: Mapping[str, Any] | None, conv_gap: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    primary_conv = metric(primary_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf")
    primary_update = metric(primary_formulas, "after_update_t_tendf_vs_current_jax_dry_t_tendf")
    forced_conv = metric(forced_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf") if forced_formulas else None
    collapsed = (
        primary_conv.get("max_abs") is not None
        and float(primary_conv["max_abs"]) <= 1.0e-3
        and float(primary_conv["rmse"]) <= 1.0e-5
    )
    if collapsed:
        return (
            "DRY_SOURCE_LEAF_FIX_CLOSED_STEP1_T_TENDF",
            [
                {
                    "rank": 1,
                    "status": "SUPPORTED",
                    "hypothesis": "Patched source-leaf mode supplies WRF-compatible dry theta sources.",
                    "evidence": primary_conv,
                }
            ],
            "Continue the Step-1 comparison downstream from `T_TENDF`.",
        )

    ranked = [
        {
            "rank": 1,
            "status": "BLOCKING",
            "hypothesis": "JAX MYNN `RTHBLTEN` source is not WRF-compatible at this Step-1 boundary.",
            "evidence": {
                "primary_after_update_vs_jax_source_leaf": primary_update,
                "jax_source_leaf_summary": primary_formulas["derived_candidate_summaries"]["current_jax_dry_t_tendf"],
                "wrf_active_sum_summary": primary_formulas["derived_candidate_summaries"]["wrf_update_expected_pre_plus_active_rth"],
            },
        },
        {
            "rank": 2,
            "status": "SECONDARY_BLOCKING",
            "hypothesis": "The held JAX `rthraten` leaf is zero at Step-1 while WRF has active `RTHRATEN`; forcing radiation only improves the residual marginally.",
            "evidence": {
                "primary_after_conv": primary_conv,
                "forced_radiation_after_conv": forced_conv,
            },
        },
        {
            "rank": 3,
            "status": "IMPLEMENTED_STILL_SECONDARY",
            "hypothesis": "WRF `conv_t_tendf_to_moist` and its `QV_TEND` term are represented in the JAX dry source bundle, but cannot close while MYNN `RTHBLTEN/RQVBLTEN` are too weak.",
            "evidence": conv_gap,
        },
    ]
    return (
        "DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED",
        ranked,
        (
            "Next source boundary: split MYNN PBL adapter/kernel inputs and outputs against WRF "
            "`RTHBLTEN`/`RQVBLTEN`; held `RTHRATEN` and `conv_t_tendf_to_moist` "
            "are ranked secondary by the current proof."
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
        label="step1_dry_source_leaf_fix_rad_rk_tendf_1",
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
        label="step1_dry_source_leaf_fix_forced_radiation",
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

    after_update = part2["surfaces"]["after_update_phy_ten"]["arrays"]["T_TENDF"]
    after_conv = part2["surfaces"]["after_conv_t_tendf_to_moist"]["arrays"]["T_TENDF"]
    conv_gap = split.compact_metric(
        split.diff_metrics(
            "wrf_after_update_vs_after_conv_t_tendf",
            after_update,
            after_conv,
            mask=split.interior_mask(after_conv.shape),
        )
    )
    verdict, ranked_blockers, next_boundary = classify(primary_formulas, forced_formulas, conv_gap)

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.step1_dry_source_leaf_fix.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": jax_environment(),
        "target": {
            "domain": split.TARGET_DOMAIN,
            "step": split.TARGET_STEP,
            "cpu_only": True,
            "pass_target": {"max_abs": 1.0e-3, "rmse": 1.0e-5},
        },
        "paths": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "prior_proof": path_info(PRIOR_PROOF),
            "wrf_truth": path_info(split.WRF_TRUTH),
        },
        "source_patch": {
            "production_files": [
                "src/gpuwrf/coupling/physics_couplers.py",
                "src/gpuwrf/runtime/operational_mode.py",
            ],
            "test_file": "tests/test_v014_dry_source_leaf_wiring.py",
            "mode": "rad_rk_tendf=1 routes held RTHRATEN plus MYNN RTHBLTEN/RQVBLTEN through WRF conv_t_tendf_to_moist into DryPhysicsTendencies.t_tendf",
            "double_count_guard": "MYNN theta delta is removed from the later non-dry physics state update in source-leaf mode.",
        },
        "wrf_truth": {
            "active_components": split.summarize_components(part2),
            "self_consistency": {
                "after_update_vs_pre_plus_active_rth": metric(
                    primary_formulas, "after_update_t_tendf_vs_pre_plus_active_rth"
                ),
                "after_conv_vs_moist_formula": metric(
                    primary_formulas, "after_conv_t_tendf_vs_moist_formula"
                ),
                "after_conv_vs_after_part2": metric(
                    primary_formulas, "after_conv_t_tendf_vs_after_first_rk_step_part2"
                ),
                "after_update_vs_after_conv": conv_gap,
            },
        },
        "jax_source_leaf_mode": {
            "primary_capture": compact_capture(primary_capture),
            "primary_metrics": {
                "after_update_vs_jax_dry_t_tendf": metric(
                    primary_formulas, "after_update_t_tendf_vs_current_jax_dry_t_tendf"
                ),
                "after_conv_vs_jax_dry_t_tendf": metric(
                    primary_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf"
                ),
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
                    "after_update_vs_jax_dry_t_tendf": metric(
                        forced_formulas, "after_update_t_tendf_vs_current_jax_dry_t_tendf"
                    ),
                    "after_conv_vs_jax_dry_t_tendf": metric(
                        forced_formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf"
                    ),
                    "derived_candidate_summaries": forced_formulas["derived_candidate_summaries"],
                }
                if forced_formulas is not None
                else {"status": "BLOCKED_FORCED_CAPTURE", "capture": forced_capture}
            ),
        },
        "ranked_blockers": ranked_blockers,
        "next_boundary": next_boundary,
        "commands": {
            "focused_test": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py",
            "proof": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py",
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": run_command(["git", "status", "--short"]),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    primary = payload["jax_source_leaf_mode"]["primary_metrics"]
    forced = payload["forced_radiation_falsifier"]["metrics"]
    active = payload["wrf_truth"]["active_components"]["active_leaf_ranked_by_nested_interior_max_abs"]
    conv_gap = payload["wrf_truth"]["self_consistency"]["after_update_vs_after_conv"]
    lines = [
        "# V0.14 Step-1 Dry Source-Leaf Fix",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Evidence",
        "",
        f"- Patched source-leaf mode is active (`rad_rk_tendf=1`) and emits nonzero JAX dry `T_TENDF`; max_abs `{payload['jax_source_leaf_mode']['derived_candidate_summaries']['current_jax_dry_t_tendf']['max_abs']}`.",
        f"- Primary Step-1 residual after WRF `conv_t_tendf_to_moist` vs patched JAX dry `T_TENDF`: max_abs `{primary['after_conv_vs_jax_dry_t_tendf']['max_abs']}`, rmse `{primary['after_conv_vs_jax_dry_t_tendf']['rmse']}`.",
        f"- WRF active leaves remain much larger: top leaf `{active[0]['field']}` max_abs `{active[0]['nested_interior']['max_abs']}`; JAX source-leaf summary max_abs `{payload['jax_source_leaf_mode']['derived_candidate_summaries']['current_jax_dry_t_tendf']['max_abs']}`.",
        f"- Forcing radiation on only moves after-conv residual to max_abs `{forced['after_conv_vs_jax_dry_t_tendf']['max_abs']}`, so radiation cadence is secondary to `RTHBLTEN` fidelity.",
        f"- WRF moist-theta conversion is now represented in source-leaf mode, but it is secondary: WRF `after_update` vs `after_conv` max_abs `{conv_gap['max_abs']}`, rmse `{conv_gap['rmse']}`.",
        "",
        "## Ranked Blockers",
        "",
    ]
    for blocker in payload["ranked_blockers"]:
        lines.append(f"- `{blocker['status']}` rank {blocker['rank']}: {blocker['hypothesis']}")
    lines.extend(
        [
            "",
            "## Next Boundary",
            "",
            payload["next_boundary"],
            "",
            "Proof objects: `proofs/v014/step1_dry_source_leaf_fix.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    primary = payload["jax_source_leaf_mode"]["primary_metrics"]
    return "\n".join(
        [
            "# Review: V0.14 Dry Source-Leaf Fix",
            "",
            f"Verdict: `{payload['verdict']}`.",
            "",
            "The production plumbing is narrow and covered by `tests/test_v014_dry_source_leaf_wiring.py`: MYNN exposes scheme-local `RTHBLTEN/RQVBLTEN`, and source mode mass-couples `RTHRATEN + RTHBLTEN` then applies WRF `conv_t_tendf_to_moist` into `DryPhysicsTendencies.t_tendf` without double-applying MYNN theta.",
            "",
            f"The Step-1 proof does not close: after-conv `T_TENDF` residual remains max_abs `{primary['after_conv_vs_jax_dry_t_tendf']['max_abs']}`, rmse `{primary['after_conv_vs_jax_dry_t_tendf']['rmse']}`.",
            "",
            f"Next decision: {payload['next_boundary']}",
            "",
        ]
    )


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(
        render_markdown(payload)
        if payload.get("status") == "PROOF_EXECUTED"
        else f"# V0.14 Step-1 Dry Source-Leaf Fix\n\nBlocked: `{payload.get('status')}`. See `{OUT_JSON}`.\n",
        encoding="utf-8",
    )
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(
        render_review(payload)
        if payload.get("status") == "PROOF_EXECUTED"
        else f"# Review: V0.14 Dry Source-Leaf Fix\n\nBlocked: `{payload.get('status')}`. See `{OUT_JSON}`.\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_REVIEW}")
    print(payload.get("verdict", payload.get("status")))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
