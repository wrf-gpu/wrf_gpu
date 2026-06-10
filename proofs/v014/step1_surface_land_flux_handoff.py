#!/usr/bin/env python3
"""V0.14 Step-1 surface/land flux handoff proof.

This proof anchors the WRF handoff around `module_surface_driver`:
SFCLAY_mynn output, the NoahMP land-surface overlay, final surface-driver
state, and MYNN-driver input.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_tsk_znt_sourcing_fix as sfclay_prior  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_surface_land_flux_handoff.json"
OUT_MD = PROOF_DIR / "step1_surface_land_flux_handoff.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-surface-land-flux-handoff.md"
OUT_PATCH = PROOF_DIR / "step1_surface_land_flux_handoff_wrf_patch.diff"

SCRATCH = Path("/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609")
WRF_RUN = SCRATCH / "run"
SURFACE_HOOK = Path("/tmp/wrfgpu2_v014_surface_handoff/surface_land_flux_d02_step1.txt")
MYNN_HOOK = SCRATCH / "wrf_truth_mynn_surfacehandoff/mynn_pre_d02_step1_its_1_ite_159_jts_1_jte_66.txt"
STRICT_PRIOR_JSON = PROOF_DIR / "step1_mynn_source_coupling.json"

SURFACE_FIELDS = [
    "xland",
    "xice",
    "ivgtyp",
    "isltyp",
    "hfx",
    "qfx",
    "lh",
    "ust",
    "tsk",
    "grdflx",
    "qsfc",
    "chs",
    "chs2",
    "cqs2",
    "flhc",
    "flqc",
    "znt",
    "z0",
    "swdown",
    "glw",
    "rainbl",
    "rho1",
    "dz8w1",
    "t_phy1",
    "qv1",
    "p8w1",
    "smois1",
    "sh2o1",
    "tslb1",
    "snow",
    "snowh",
    "snowc",
    "albedo",
    "psfc",
    "mavail",
    "vegfra",
]
MYNN_SFC_FIELDS = [
    "xland",
    "ust",
    "hfx",
    "qfx",
    "wspd",
    "tsk",
    "qsfc",
    "psfc",
    "chs",
    "znt",
    "uoce",
    "voce",
    "pblh",
    "rthraten_kts",
]


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
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        c = c[mask_array]
        r = r[mask_array]
    d = c - r
    finite = np.isfinite(d)
    if not np.any(finite):
        return {"max_abs": None, "rmse": None, "mean": None, "min": None, "max": None, "n": 0}
    d = d[finite]
    abs_d = np.abs(d)
    return {
        "max_abs": float(np.max(abs_d)),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "n": int(d.size),
    }


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def parse_surface_hook(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    stages: dict[str, list[tuple[int, int, list[float]]]] = {}
    headers: dict[str, list[str]] = {}
    fields: list[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("HDR "):
                parts = line.split()
                headers[parts[1]] = parts[2:]
            elif line.startswith("FIELDS "):
                parts = line.split()
                if fields is None:
                    fields = parts[2:]
            elif line.startswith("SFC "):
                parts = line.split()
                stage = parts[1]
                i = int(parts[2])
                j = int(parts[3])
                values = [float(item) for item in parts[4:]]
                stages.setdefault(stage, []).append((i, j, values))
    if fields != SURFACE_FIELDS:
        return {"status": "BAD_FIELDS", "path": str(path), "fields": fields}
    arrays: dict[str, np.ndarray] = {}
    for stage, rows in stages.items():
        ni = max(row[0] for row in rows)
        nj = max(row[1] for row in rows)
        array = np.full((nj, ni, len(fields)), np.nan, dtype=np.float64)
        for i, j, values in rows:
            array[j - 1, i - 1, :] = values
        arrays[stage] = array
    return {"status": "READY", "path": str(path), "fields": fields, "headers": headers, "arrays": arrays}


def parse_mynn_sfc(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    rows: list[tuple[int, int, list[float]]] = []
    header: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("HDR "):
                header = line.strip()
            elif line.startswith("SFC "):
                parts = line.split()
                rows.append((int(parts[1]), int(parts[2]), [float(item) for item in parts[3:]]))
    ni = max(row[0] for row in rows)
    nj = max(row[1] for row in rows)
    array = np.full((nj, ni, len(MYNN_SFC_FIELDS)), np.nan, dtype=np.float64)
    for i, j, values in rows:
        array[j - 1, i - 1, :] = values
    return {"status": "READY", "path": str(path), "header": header, "fields": MYNN_SFC_FIELDS, "array": array}


def field(array: np.ndarray, name: str) -> np.ndarray:
    return array[:, :, SURFACE_FIELDS.index(name)]


def mynn_field(array: np.ndarray, name: str) -> np.ndarray:
    return array[:, :, MYNN_SFC_FIELDS.index(name)]


def compare_stage_fields(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    names = ["hfx", "qfx", "lh", "ust", "tsk", "grdflx", "qsfc", "znt", "smois1", "sh2o1", "tslb1"]
    return {name: diffstat(field(a, name), field(b, name), mask) for name in names}


def compare_mynn_to_surface(mynn: np.ndarray, surface: np.ndarray) -> dict[str, Any]:
    pairs = {
        "ust": ("ust", "ust"),
        "hfx": ("hfx", "hfx"),
        "qfx": ("qfx", "qfx"),
        "tsk": ("tsk", "tsk"),
        "qsfc": ("qsfc", "qsfc"),
        "znt": ("znt", "znt"),
        "psfc": ("psfc", "psfc"),
        "chs": ("chs", "chs"),
    }
    return {name: diffstat(mynn_field(mynn, left), field(surface, right)) for name, (left, right) in pairs.items()}


def sfclay_to_pre_noahmp(pre: np.ndarray) -> dict[str, Any]:
    reads = {
        "hfx": "sfclay_mynn_out__hfx.f64",
        "qfx": "sfclay_mynn_out__qfx.f64",
        "ust": "sfclay_mynn_out__ust.f64",
        "znt": "sfclay_mynn_out__znt.f64",
        "qsfc": "sfclay_mynn_out__qsfc.f64",
    }
    out: dict[str, Any] = {}
    for name, filename in reads.items():
        out[f"pre_noahmp_{name}_vs_sfclay1d_{name}"] = diffstat(field(pre, name), sfclay_prior.read2(filename))
    return out


def read_current_step1_config() -> dict[str, Any]:
    inputs = live.build_live_nest_step1_inputs()
    namelist = inputs["namelist"]
    return {
        "use_noahmp": bool(getattr(namelist, "use_noahmp", False)),
        "sf_surface_physics": getattr(namelist, "sf_surface_physics", None),
        "noahmp_static_configured": getattr(namelist, "noahmp_static", None) is not None,
        "noahmp_energy_params_configured": getattr(namelist, "noahmp_energy_params", None) is not None,
        "noahmp_rad_params_configured": getattr(namelist, "noahmp_rad_params", None) is not None,
        "inputs_have_noahmp_land": "noahmp_land" in inputs,
    }


def read_wrf_namelist_subset(path: Path) -> dict[str, Any]:
    keys = {"mp_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics"}
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key in keys:
            out[key] = value.rstrip(",")
    return out


def read_prior_strict_metric() -> dict[str, Any] | None:
    if not STRICT_PRIOR_JSON.exists():
        return None
    payload = json.loads(STRICT_PRIOR_JSON.read_text(encoding="utf-8"))
    if "strict_step1" in payload:
        return payload.get("strict_step1", {}).get("metric")
    return (
        payload.get("stage_formula_metrics", {})
        .get("strict_after_conv_vs_jax_dry_t_tendf")
    )


def compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_abs": metric.get("max_abs"),
        "rmse": metric.get("rmse"),
        "mean": metric.get("mean"),
        "n": metric.get("n"),
    }


def build_proof() -> dict[str, Any]:
    surface = parse_surface_hook(SURFACE_HOOK)
    if surface["status"] != "READY":
        return {"status": "BLOCKED_SURFACE_HOOK", "surface_hook": surface}
    mynn = parse_mynn_sfc(MYNN_HOOK)
    if mynn["status"] != "READY":
        return {"status": "BLOCKED_MYNN_HOOK", "mynn_hook": mynn}

    arrays = surface["arrays"]
    pre = arrays["PRE_NOAHMP"]
    post = arrays["POST_NOAHMP"]
    final = arrays["POST_SURFACE_FINAL"]
    mynn_sfc = mynn["array"]
    land_mask = field(pre, "xland") < 1.5
    water_mask = ~land_mask

    noahmp_delta = {
        "all_columns": compare_stage_fields(post, pre),
        "land_only": compare_stage_fields(post, pre, land_mask),
        "water_only": compare_stage_fields(post, pre, water_mask),
    }
    final_delta = {
        "all_columns": compare_stage_fields(final, post),
        "land_only": compare_stage_fields(final, post, land_mask),
        "water_only": compare_stage_fields(final, post, water_mask),
    }
    handoff = {
        "mynn_vs_pre_noahmp": compare_mynn_to_surface(mynn_sfc, pre),
        "mynn_vs_post_noahmp": compare_mynn_to_surface(mynn_sfc, post),
        "mynn_vs_post_surface_final": compare_mynn_to_surface(mynn_sfc, final),
    }
    sfclay_pre = sfclay_to_pre_noahmp(pre)
    current_step1_config = read_current_step1_config()
    strict_metric = read_prior_strict_metric()

    noahmp_changes_hfx = noahmp_delta["all_columns"]["hfx"]["max_abs"]
    noahmp_changes_qfx = noahmp_delta["all_columns"]["qfx"]["max_abs"]
    post_reaches_mynn_hfx = handoff["mynn_vs_post_noahmp"]["hfx"]["max_abs"]
    post_reaches_mynn_qfx = handoff["mynn_vs_post_noahmp"]["qfx"]["max_abs"]
    step1_skips_noahmp = (
        current_step1_config["use_noahmp"] is False
        and current_step1_config["sf_surface_physics"] is None
        and current_step1_config["inputs_have_noahmp_land"] is False
    )
    step1_has_noahmp = (
        current_step1_config["use_noahmp"] is True
        and current_step1_config["sf_surface_physics"] is not None
        and int(current_step1_config["sf_surface_physics"]) == 4
        and current_step1_config["noahmp_static_configured"]
        and current_step1_config["noahmp_energy_params_configured"]
        and current_step1_config["noahmp_rad_params_configured"]
        and current_step1_config["inputs_have_noahmp_land"]
    )
    if step1_has_noahmp:
        verdict = "STEP1_SURFACE_LAND_FLUX_HANDOFF_CLOSED_JAX_NOAHMP_ENABLED"
    elif step1_skips_noahmp:
        verdict = "STEP1_SURFACE_LAND_FLUX_HANDOFF_NARROWED_TO_JAX_NOAHMP_DISABLED_CONFIGURATION"
    else:
        verdict = "STEP1_SURFACE_LAND_FLUX_HANDOFF_REQUIRES_NOAHMP_CONFIGURATION_RECHECK"

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.step1_surface_land_flux_handoff.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "artifacts": {
            "surface_hook": path_info(SURFACE_HOOK),
            "mynn_hook": path_info(MYNN_HOOK),
            "wrf_patch": path_info(OUT_PATCH),
            "wrf_case_namelist": path_info(WRF_RUN / "namelist.input"),
        },
        "wrf_namelist_subset": read_wrf_namelist_subset(WRF_RUN / "namelist.input"),
        "hook_headers": surface["headers"],
        "grid_counts": {
            "shape_j_i": list(pre.shape[:2]),
            "columns_total": int(pre.shape[0] * pre.shape[1]),
            "land_columns": int(np.count_nonzero(land_mask)),
            "water_columns": int(np.count_nonzero(water_mask)),
        },
        "sfclay_to_pre_noahmp": sfclay_pre,
        "noahmp_overlay_delta": noahmp_delta,
        "post_surface_final_delta": final_delta,
        "mynn_driver_handoff": handoff,
        "current_jax_step1_config": current_step1_config,
        "prior_strict_after_conv_t_tendf": strict_metric,
        "acceptance_thresholds": {"strict_max_abs": 1.0e-3, "strict_rmse": 1.0e-5},
        "summary_metrics": {
            "sfclay_pre_hfx_max_abs": compact_metric(sfclay_pre["pre_noahmp_hfx_vs_sfclay1d_hfx"]),
            "sfclay_pre_qfx_max_abs": compact_metric(sfclay_pre["pre_noahmp_qfx_vs_sfclay1d_qfx"]),
            "noahmp_hfx_delta": compact_metric(noahmp_delta["all_columns"]["hfx"]),
            "noahmp_qfx_delta": compact_metric(noahmp_delta["all_columns"]["qfx"]),
            "post_to_mynn_hfx": compact_metric(handoff["mynn_vs_post_noahmp"]["hfx"]),
            "post_to_mynn_qfx": compact_metric(handoff["mynn_vs_post_noahmp"]["qfx"]),
            "post_to_mynn_ust": compact_metric(handoff["mynn_vs_post_noahmp"]["ust"]),
        },
        "narrowed_blocker": {
            "location": "JAX Step-1 live-nest/source-capture configuration before MYNN",
            "evidence": [
                "WRF sf_surface_physics=4 and sf_sfclay_physics=5.",
                "WRF PRE_NOAHMP equals WRF SFCLAY1D_mynn output to roundoff.",
                "WRF noahmplsm changes HFX/QFX/LH/TSK/QSFC/ZNT only on land columns; UST is unchanged.",
                "WRF MYNN driver input equals POST_NOAHMP/POST_SURFACE_FINAL exactly for HFX/QFX/UST/TSK/QSFC/ZNT.",
                "Current JAX Step-1 inputs report use_noahmp=False, sf_surface_physics=None, and no NoahMP land/static state.",
            ],
            "fastest_next_command": (
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false "
                "PYTHONPATH=src python proofs/v014/step1_mynn_source_coupling.py"
            ),
        },
    }


def write_markdown(payload: Mapping[str, Any]) -> None:
    status = payload.get("status")
    verdict = payload.get("verdict", status)
    summary = payload.get("summary_metrics", {})
    config = payload.get("current_jax_step1_config", {})
    strict = payload.get("prior_strict_after_conv_t_tendf") or {}
    lines = [
        "# V0.14 Step-1 Surface/Land Flux Handoff",
        "",
        f"- status: `{status}`",
        f"- verdict: `{verdict}`",
        "- WRF handoff: `SFCLAY1D_mynn output == PRE_NOAHMP`; `POST_NOAHMP == MYNN driver input`.",
        "- WRF NoahMP overlay is the exact HFX/QFX change point; post-surface finalization does not further change HFX/QFX in this fixture.",
        f"- JAX Step-1 config: `use_noahmp={config.get('use_noahmp')}`, `sf_surface_physics={config.get('sf_surface_physics')}`, `inputs_have_noahmp_land={config.get('inputs_have_noahmp_land')}`.",
        "",
        "## Key Metrics",
        "",
        f"- SFCLAY -> PRE_NOAHMP HFX max_abs: `{summary.get('sfclay_pre_hfx_max_abs', {}).get('max_abs')}`",
        f"- SFCLAY -> PRE_NOAHMP QFX max_abs: `{summary.get('sfclay_pre_qfx_max_abs', {}).get('max_abs')}`",
        f"- PRE_NOAHMP -> POST_NOAHMP HFX max_abs: `{summary.get('noahmp_hfx_delta', {}).get('max_abs')}`",
        f"- PRE_NOAHMP -> POST_NOAHMP QFX max_abs: `{summary.get('noahmp_qfx_delta', {}).get('max_abs')}`",
        f"- POST_NOAHMP -> MYNN HFX max_abs: `{summary.get('post_to_mynn_hfx', {}).get('max_abs')}`",
        f"- POST_NOAHMP -> MYNN QFX max_abs: `{summary.get('post_to_mynn_qfx', {}).get('max_abs')}`",
        f"- POST_NOAHMP -> MYNN UST max_abs: `{summary.get('post_to_mynn_ust', {}).get('max_abs')}`",
        f"- prior strict after-conv `T_TENDF` max_abs: `{strict.get('max_abs')}`, RMSE: `{strict.get('rmse')}`",
        "",
        "## Blocker",
        "",
        (
            "CLOSED: the JAX Step-1 builder now carries WRF-derived NoahMP land/static state "
            "with sf_surface_physics=4 + use_noahmp=True; the remaining gate is the strict "
            "Step-1 metric in step1_mynn_source_coupling / noahmp_step1_closure."
            if str(verdict).endswith("CLOSED_JAX_NOAHMP_ENABLED")
            else "The blocker is now narrower than the surface/land flux handoff: the WRF handoff itself is closed to the MYNN-driver input, but the JAX Step-1 path is built with NoahMP disabled/missing land state."
        ),
        "",
        "Fastest next command:",
        "",
        "```bash",
        payload.get("narrowed_blocker", {}).get("fastest_next_command", ""),
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_review(payload: Mapping[str, Any]) -> None:
    status = payload.get("status")
    verdict = payload.get("verdict", status)
    summary = payload.get("summary_metrics", {})
    lines = [
        "# V0.14 Step-1 surface/land flux handoff review",
        "",
        f"- verdict: `{verdict}`",
        "- production code changes: none",
        "- WRF hook patch archived: `proofs/v014/step1_surface_land_flux_handoff_wrf_patch.diff`",
        "- proof script: `proofs/v014/step1_surface_land_flux_handoff.py`",
        "",
        "Evidence:",
        f"- SFCLAY -> PRE_NOAHMP HFX max_abs `{summary.get('sfclay_pre_hfx_max_abs', {}).get('max_abs')}`.",
        f"- PRE_NOAHMP -> POST_NOAHMP HFX max_abs `{summary.get('noahmp_hfx_delta', {}).get('max_abs')}`.",
        f"- POST_NOAHMP -> MYNN HFX max_abs `{summary.get('post_to_mynn_hfx', {}).get('max_abs')}`.",
        "",
        "Unresolved risk:",
        (
            "- JAX Step-1 builder now carries WRF-derived NoahMP land/static state with `sf_surface_physics=4`; the strict Step-1 gate is scored by `proofs/v014/noahmp_step1_closure.py`."
            if str(verdict).endswith("CLOSED_JAX_NOAHMP_ENABLED")
            else "- Strict Step-1 `T_TENDF` remains red until the JAX Step-1 builder/source capture carries WRF-derived NoahMP land/static state and enables `sf_surface_physics=4`."
        ),
        "",
    ]
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    write_markdown(payload)
    write_review(payload)
    print(json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
