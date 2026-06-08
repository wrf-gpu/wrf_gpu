#!/usr/bin/env python3
"""Build the V0.14 same-state CPU-WRF savepoint request manifest.

This proof packages the accepted dynamic attribution manifest into a compact
request for a future WRF instrumentation worker. It does not instrument WRF,
import JAX, run model code, or make an equivalence/root-cause claim.

Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 24-31 \
    python proofs/v014/same_state_savepoint_request.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DYNAMIC_JSON = ROOT / "proofs/v014/dynamic_field_attribution.json"
DYNAMIC_MD = ROOT / "proofs/v014/dynamic_field_attribution.md"
LOCALIZATION_PLAN_MD = ROOT / "proofs/v014/same_state_tendency_localization_plan.md"
HANDOFF_MD = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
FEASIBILITY_JSON = ROOT / "proofs/v014/same_state_wrf_savepoint_feasibility.json"
FEASIBILITY_MD = ROOT / "proofs/v014/same_state_wrf_savepoint_feasibility.md"
OUT_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"
OUT_MD = ROOT / "proofs/v014/same_state_savepoint_request.md"

SCHEMA = "wrf_gpu2.v014.same_state_savepoint_request.v1"
EXPECTED_CELL_COUNT = 24
EXPECTED_LEAD_H = 10

TERM_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "stage_input",
        "name": "stage input",
        "timing": "before each RK stage tendency assembly",
        "native_grids": ["mass", "U", "V", "W/PH", "surface"],
        "requested_terms": [
            "U",
            "V",
            "W",
            "T",
            "QVAPOR and other active moisture scalars",
            "P",
            "PB",
            "PH",
            "PHB",
            "MU",
            "MUB",
            "static metrics and map factors",
            "Coriolis/map-rotation fields",
            "boundary/spec-relax leaves active for the sampled step",
        ],
    },
    {
        "id": "mass_coupling",
        "name": "mass coupling",
        "timing": "large-step dry tendency assembly",
        "native_grids": ["mass", "U", "V", "W"],
        "requested_terms": [
            "mass_u",
            "mass_v",
            "mass_h",
            "mass_f",
            "muu",
            "muv",
            "mut",
            "ru",
            "rv",
            "rom",
            "coupled velocity intermediates used by advection",
        ],
    },
    {
        "id": "momentum_advection",
        "name": "momentum advection",
        "timing": "large-step advection before source folding",
        "native_grids": ["U", "V", "W"],
        "requested_terms": [
            "ru_adv",
            "rv_adv",
            "rw_adv",
            "momentum flux components and final advection tendency arrays",
        ],
    },
    {
        "id": "scalar_theta_mu_advection",
        "name": "scalar/theta/mu advection",
        "timing": "large-step scalar and dry-mass advection",
        "native_grids": ["mass", "W"],
        "requested_terms": [
            "theta/t advection tendency",
            "mu advection tendency",
            "QVAPOR and active scalar advection tendencies",
            "scalar flux components used by the sampled configuration",
        ],
    },
    {
        "id": "diffusion",
        "name": "diffusion",
        "timing": "large-step diffusion according to active diff_opt/km_opt",
        "native_grids": ["mass", "U", "V", "W"],
        "requested_terms": [
            "sixth-order diffusion terms",
            "constant-K diffusion terms",
            "deformation/Smagorinsky terms if active",
            "ru_diff",
            "rv_diff",
            "rw_diff",
            "theta/scalar diffusion tendencies",
        ],
    },
    {
        "id": "horizontal_pgf",
        "name": "horizontal PGF",
        "timing": "large-step horizontal pressure-gradient calculation",
        "native_grids": ["U", "V"],
        "requested_terms": [
            "ru_pgf",
            "rv_pgf",
            "named pressure-gradient pieces before final sum where available",
        ],
    },
    {
        "id": "coriolis",
        "name": "Coriolis",
        "timing": "large-step Coriolis calculation",
        "native_grids": ["U", "V", "W"],
        "requested_terms": [
            "ru_cor",
            "rv_cor",
            "vertical coupling terms or rw_cor if present/active",
        ],
    },
    {
        "id": "source_tendency_folding",
        "name": "source-tendency folding",
        "timing": "before and after WRF folds external/source tendencies into RK tendencies",
        "native_grids": ["mass", "U", "V", "W/PH"],
        "requested_terms": [
            "raw ru_tendf/rv_tendf/rw_tendf/ph_tendf/t_tendf/mu_tendf",
            "RUBLTEN",
            "RVBLTEN",
            "RTHRATEN",
            "RTHBLTEN",
            "RTHCUTEN",
            "RQ* scalar tendencies",
            "MUTTEN",
            "total u/v/w/theta/mu/ph tendencies before acoustic",
        ],
    },
    {
        "id": "small_step_prep",
        "name": "small-step prep",
        "timing": "after large-step tendency assembly, before acoustic scan",
        "native_grids": ["mass", "U", "V", "W/PH"],
        "requested_terms": [
            "prep work arrays",
            "p",
            "alt",
            "al",
            "php",
            "cqu",
            "cqv",
            "cqw",
            "time weights and RK weights used by the acoustic step",
        ],
    },
    {
        "id": "acoustic_uv",
        "name": "acoustic U/V",
        "timing": "requested first and last acoustic substeps for each RK stage",
        "native_grids": ["U", "V"],
        "requested_terms": [
            "large-step U/V add",
            "small-step pressure-gradient components",
            "emdiv/damping components",
            "normal boundary work",
            "u_work",
            "v_work",
            "ru_m",
            "rv_m",
        ],
    },
    {
        "id": "mu_theta",
        "name": "MU/theta",
        "timing": "requested first and last acoustic substeps for each RK stage",
        "native_grids": ["mass", "W"],
        "requested_terms": [
            "dvdxi",
            "dmdt",
            "mu_tendency",
            "mu_work_new",
            "muts",
            "muave",
            "mudf",
            "ww",
            "theta tendency/update",
        ],
    },
    {
        "id": "w_ph",
        "name": "W/PH",
        "timing": "requested first and last acoustic substeps for each RK stage",
        "native_grids": ["W/PH"],
        "requested_terms": [
            "rw_tend_pg_buoy",
            "advance_w pressure pieces",
            "advance_w buoyancy pieces",
            "w_work/w_next",
            "ph_tend",
            "PH update pieces",
        ],
    },
    {
        "id": "pressure_rho_refresh",
        "name": "pressure/rho refresh",
        "timing": "after MU/theta/W/PH updates inside acoustic path and at final stage reconstruction",
        "native_grids": ["mass", "W/PH"],
        "requested_terms": [
            "p",
            "pb",
            "rho",
            "alt",
            "al",
            "pressure/rho refresh inputs and outputs",
        ],
    },
    {
        "id": "boundary_spec_relax",
        "name": "boundary/spec-relax",
        "timing": "before and after in-loop normal boundary work and end-step lateral boundary application",
        "native_grids": ["mass", "U", "V", "W/PH"],
        "requested_terms": [
            "specified-boundary input leaves",
            "relax-zone input leaves",
            "in-loop normal U/V boundary work",
            "PH/W relax deltas if active",
            "end-step apply_lateral_boundaries before/after deltas",
        ],
    },
    {
        "id": "final_stage_state",
        "name": "final stage state",
        "timing": "after each RK stage and after the final stage carry/finish",
        "native_grids": ["mass", "U", "V", "W/PH", "surface"],
        "requested_terms": [
            "U",
            "V",
            "W",
            "T",
            "QVAPOR and active scalars",
            "P",
            "PB",
            "PH",
            "PHB",
            "MU",
            "MUB",
            "surface diagnostics if already produced by WRF at this point",
        ],
    },
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path.relative_to(ROOT) if path.exists() and path.is_relative_to(ROOT) else path),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime_utc": (
            datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None
        ),
        "sha256": sha256_file(path),
    }


def require_cpu_only() -> dict[str, Any]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    jax_platforms = os.environ.get("JAX_PLATFORMS")
    if cuda_visible not in ("", "-1"):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be empty or -1 for this CPU-only request proof"
        )
    if jax_platforms not in (None, "", "cpu"):
        raise RuntimeError("JAX_PLATFORMS must be unset, empty, or cpu for this proof")
    return {
        "cpu_only": True,
        "gpu_used": False,
        "CUDA_VISIBLE_DEVICES": cuda_visible,
        "JAX_PLATFORMS": jax_platforms,
        "note": "This script uses only the Python standard library and does not import JAX.",
    }


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def bound(axis: str, start: int, stop_exclusive: int) -> dict[str, Any]:
    return {
        "axis": axis,
        "zero_based": {
            "start": int(start),
            "stop_exclusive": int(stop_exclusive),
            "count": int(stop_exclusive - start),
        },
        "fortran_1based_inclusive": {
            "start": int(start + 1),
            "end": int(stop_exclusive),
            "count": int(stop_exclusive - start),
        },
    }


def vertical_bound(axis: str, count: int | None) -> dict[str, Any]:
    return {
        "axis": axis,
        "all_levels": True,
        "zero_based": {
            "start": 0,
            "stop_exclusive": count,
            "count": count,
        },
        "fortran_1based_inclusive": {
            "start": 1 if count is not None else None,
            "end": count,
            "count": count,
        },
    }


def vertical_counts(dynamic: Mapping[str, Any]) -> dict[str, int | None]:
    compared = dynamic.get("compatibility", {}).get("compared", {})
    p_shape = compared.get("P", {}).get("shape", [])
    ph_shape = compared.get("PH", {}).get("shape", [])
    return {
        "bottom_top": int(p_shape[0]) if p_shape else None,
        "bottom_top_stag": int(ph_shape[0]) if ph_shape else None,
    }


def native_patch_bounds(
    patch: Mapping[str, Any], counts: Mapping[str, int | None]
) -> dict[str, Any]:
    y0 = int(patch["south_north_start"])
    y1 = int(patch["south_north_stop_exclusive"])
    x0 = int(patch["west_east_start"])
    x1 = int(patch["west_east_stop_exclusive"])
    return {
        "mass_scalar_3d": {
            "bottom_top": vertical_bound("bottom_top", counts.get("bottom_top")),
            "south_north": bound("south_north", y0, y1),
            "west_east": bound("west_east", x0, x1),
        },
        "surface_2d": {
            "south_north": bound("south_north", y0, y1),
            "west_east": bound("west_east", x0, x1),
        },
        "u_staggered": {
            "bottom_top": vertical_bound("bottom_top", counts.get("bottom_top")),
            "south_north": bound("south_north", y0, y1),
            "west_east_stag": bound("west_east_stag", x0, x1 + 1),
        },
        "v_staggered": {
            "bottom_top": vertical_bound("bottom_top", counts.get("bottom_top")),
            "south_north_stag": bound("south_north_stag", y0, y1 + 1),
            "west_east": bound("west_east", x0, x1),
        },
        "w_ph_staggered": {
            "bottom_top_stag": vertical_bound(
                "bottom_top_stag", counts.get("bottom_top_stag")
            ),
            "south_north": bound("south_north", y0, y1),
            "west_east": bound("west_east", x0, x1),
        },
    }


def compact_top_components(cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in cell.get("top_components", []):
        out.append(
            {
                "name": item.get("name"),
                "severity_ratio": finite_float(item.get("severity_ratio")),
            }
        )
    return out


def compact_cell(cell: Mapping[str, Any], counts: Mapping[str, int | None]) -> dict[str, Any]:
    stagger = cell["stagger_context"]
    patch = stagger["patch_bounds_mass_grid"]
    mass = cell["mass_index"]
    return {
        "selection_rank": int(cell["selection_rank"]),
        "candidate_rank_in_lead": int(cell["candidate_rank_in_lead"]),
        "lead_h": int(cell["lead_h"]),
        "valid_time_utc": cell["valid_time_utc"],
        "mass_index_zero_based": {
            "south_north": int(mass["south_north"]),
            "west_east": int(mass["west_east"]),
        },
        "mass_index_fortran_1based": {
            "south_north": int(mass["south_north"]) + 1,
            "west_east": int(mass["west_east"]) + 1,
        },
        "lat": finite_float(cell.get("lat")),
        "lon": finite_float(cell.get("lon")),
        "hgt_m": finite_float(cell.get("hgt_m")),
        "landmask": finite_float(cell.get("landmask")),
        "region_bins": cell.get("region_bins", {}),
        "diagnostic_context": {
            "composite_score": finite_float(cell.get("composite_score")),
            "co_located_component_hit_count": int(cell["co_located_component_hit_count"]),
            "top_components": compact_top_components(cell),
            "selected_field_diffs": {
                name: finite_float(value)
                for name, value in sorted(cell.get("field_diffs", {}).items())
                if name
                in {
                    "dU10",
                    "dV10",
                    "dPSFC",
                    "dMU",
                    "dP_k0",
                    "dPH_k0",
                    "dU_k0",
                    "dV_k0",
                    "dW_k0",
                    "dT_k0",
                    "dQVAPOR_k0",
                    "dPBLH",
                }
            },
        },
        "patch_bounds_mass_grid": patch,
        "native_patch_bounds": native_patch_bounds(patch, counts),
        "native_stagger_context_k0": {
            "mass_cell": stagger["mass_cell"],
            "adjacent_native_faces_k0": stagger["adjacent_native_faces_k0"],
            "adjacent_native_diff_values_k0": stagger.get("adjacent_native_diff_values_k0", {}),
        },
        "full_column_native_columns": stagger["vertical_column_context"],
    }


def validate_selected_cells(cells: list[dict[str, Any]]) -> None:
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(f"expected {EXPECTED_CELL_COUNT} selected cells, found {len(cells)}")
    ranks = [cell["selection_rank"] for cell in cells]
    if ranks != list(range(1, EXPECTED_CELL_COUNT + 1)):
        raise ValueError(f"selected-cell ranks are not exactly 1..{EXPECTED_CELL_COUNT}: {ranks}")
    leads = {cell["lead_h"] for cell in cells}
    if leads != {EXPECTED_LEAD_H}:
        raise ValueError(f"selected cells must all be h{EXPECTED_LEAD_H}, got {sorted(leads)}")
    keys = {
        (
            cell["mass_index_zero_based"]["south_north"],
            cell["mass_index_zero_based"]["west_east"],
        )
        for cell in cells
    }
    if len(keys) != EXPECTED_CELL_COUNT:
        raise ValueError("selected cells contain duplicate mass-grid indices")
    for cell in cells:
        full_column = cell["full_column_native_columns"].get(
            "all_native_vertical_levels_required_for_operator_probe"
        )
        if full_column is not True:
            raise ValueError(f"cell rank {cell['selection_rank']} lacks full-column flag")


def selected_vertical_levels(dynamic: Mapping[str, Any]) -> list[int]:
    levels = dynamic["localization_manifest"]["recommended_vertical_levels_for_first_probe"]
    return [int(level) for level in levels]


def artifact_schema() -> dict[str, Any]:
    return {
        "preferred_formats": ["NetCDF4", "Zarr", "HDF5"],
        "preferred_file_name": (
            "wrf_same_state_savepoints_20260501_18z_l2_72h_"
            "20260519T173026Z_d02_h10.<nc|zarr|h5>"
        ),
        "indexing_contract": {
            "manifest_indices": "zero-based Python indices with stop-exclusive patch bounds",
            "fortran_translation": (
                "for a zero-based [start, stop_exclusive) range, instrument WRF with "
                "1-based inclusive [start+1, stop_exclusive]"
            ),
            "native_staggering_required": True,
            "destaggered_values_are_not_sufficient": True,
        },
        "required_global_attributes": [
            "schema",
            "run_id",
            "domain",
            "selected_lead_h",
            "selected_valid_time_utc",
            "wrf_git_hash",
            "wrf_build_id",
            "compiler_id",
            "namelist_hash",
            "instrumentation_patch_hash",
            "model_step_number",
            "wrf_time_string",
            "dt_s",
            "rk_stage_count",
            "acoustic_substep_count",
            "source_request_sha256",
        ],
        "required_groups": {
            "/metadata": [
                "domain_shape_by_native_grid",
                "namelist_switches",
                "time_step_weights",
                "static/base field metadata",
                "instrumented WRF routine/file list",
            ],
            "/selection": [
                "selected_cells",
                "patch_bounds_mass_grid",
                "native_patch_bounds",
                "recommended_vertical_levels",
                "full_column_required flag",
            ],
            "/stage_<rk_stage>/stage_input": [
                "native-grid input state arrays",
                "static metrics",
                "boundary leaves active at stage entry",
            ],
            "/stage_<rk_stage>/large_step/<term_group>": [
                "named term arrays on native staggering",
                "routine input arrays needed to reproduce each term",
                "post-term total tendency arrays",
            ],
            "/stage_<rk_stage>/acoustic/<sample_selector>": [
                "substep index",
                "pre-substep state/work arrays",
                "named acoustic term arrays",
                "post-substep state/work arrays",
            ],
            "/stage_<rk_stage>/final_stage_state": [
                "state immediately after stage carry/finish",
                "end-step boundary/spec-relax before/after arrays where applied",
            ],
        },
        "required_dataset_attributes": [
            "term_group",
            "wrf_routine",
            "native_grid",
            "dimensions",
            "units",
            "rk_stage",
            "acoustic_substep_index",
            "sample_selector",
            "patch_rank",
            "zero_based_origin",
            "fortran_1based_bounds",
            "dtype",
            "finite_count",
        ],
        "required_companion_artifacts": [
            "instrumentation patch diff or commit hash",
            "WRF build/configuration log",
            "exact run command and namelist files",
            "short manifest echo proving these 24 cells and h10 were used",
        ],
    }


def wrf_source_build_feasibility_dependency() -> dict[str, Any]:
    if not FEASIBILITY_JSON.exists():
        return {
            "dependency": "Sartre WRF source/build feasibility",
            "status": "missing_from_workspace",
            "impact": (
                "This request is implementation-ready at the manifest level, "
                "but exact WRF checkout/build paths must come from the separate "
                "WRF feasibility sprint before instrumentation."
            ),
        }

    feasibility = load_json(FEASIBILITY_JSON)
    verdict = feasibility.get("verdict", {})
    return {
        "dependency": "Sartre WRF source/build feasibility",
        "status": "available_in_workspace",
        "proof_json": file_metadata(FEASIBILITY_JSON),
        "proof_md": file_metadata(FEASIBILITY_MD),
        "fastest_reliable_path": verdict.get("fastest_reliable_path"),
        "expected_first_target": verdict.get("expected_first_target"),
        "source_truth_quality": verdict.get("source_truth_quality"),
        "impact": (
            "Use this feasibility proof for WRF source/build choice and provenance "
            "risks. The selected h10 cells, levels, and patch bounds in this "
            "request remain sourced from dynamic_field_attribution.json."
        ),
    }


def build_manifest() -> dict[str, Any]:
    env = require_cpu_only()
    dynamic = load_json(DYNAMIC_JSON)
    loc = dynamic["localization_manifest"]
    counts = vertical_counts(dynamic)
    cells = [compact_cell(cell, counts) for cell in loc["selected_cells"]]
    validate_selected_cells(cells)

    valid_times = sorted({cell["valid_time_utc"] for cell in cells})
    if len(valid_times) != 1:
        raise ValueError(f"selected cells have multiple valid times: {valid_times}")
    lead_info = dynamic.get("lead_analysis", {}).get("selected_localization_lead", {})
    if int(lead_info.get("lead_h", EXPECTED_LEAD_H)) != EXPECTED_LEAD_H:
        raise ValueError(f"dynamic source selected lead is not h{EXPECTED_LEAD_H}: {lead_info}")

    feasibility_dependency = wrf_source_build_feasibility_dependency()
    unresolved_dependencies = (
        [] if feasibility_dependency["status"] == "available_in_workspace" else [feasibility_dependency]
    )

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "equivalence_claim": False,
            "root_cause_claim": False,
            "scope": (
                "Request manifest only. It packages selected h10 cells, native "
                "stagger context, patch bounds, full-column requirements, and "
                "WRF term groups for future CPU-WRF instrumentation."
            ),
        },
        "environment": env,
        "source_proofs": {
            "authoritative_selected_cell_manifest": file_metadata(DYNAMIC_JSON),
            "dynamic_field_attribution_md": file_metadata(DYNAMIC_MD),
            "same_state_tendency_localization_plan_md": file_metadata(LOCALIZATION_PLAN_MD),
            "grid_parity_first_handoff_md": file_metadata(HANDOFF_MD),
            "wrf_source_build_feasibility_json": file_metadata(FEASIBILITY_JSON),
            "wrf_source_build_feasibility_md": file_metadata(FEASIBILITY_MD),
        },
        "run_request": {
            "run_id": dynamic["run_id"],
            "domain": dynamic["inputs"]["domain"],
            "selected_lead_h": EXPECTED_LEAD_H,
            "selected_valid_time_utc": valid_times[0],
            "cpu_wrf_truth_dir": dynamic["inputs"].get("cpu_dir"),
            "retained_jax_output_dir_for_selection_only": dynamic["inputs"].get("gpu_dir"),
            "selection_source": (
                "proofs/v014/dynamic_field_attribution.json "
                "localization_manifest.selected_cells"
            ),
        },
        "same_state_sampling_request": {
            "model_step": (
                "Instrument one real CPU-WRF large timestep at the h10 same-state "
                "target. Record the exact WRF model step number, WRF Times value, "
                "dt, and namelist switches; do not infer cadence from hourly output alone."
            ),
            "rk_stages": [
                {"wrf_stage_index_1based": 1, "selector": "first RK stage"},
                {"wrf_stage_index_1based": 2, "selector": "second RK stage"},
                {"wrf_stage_index_1based": 3, "selector": "third/final RK stage"},
            ],
            "acoustic_substep_samples": [
                {
                    "selector": "first",
                    "wrf_substep_index_1based": 1,
                    "note": "Record for every requested RK stage.",
                },
                {
                    "selector": "last",
                    "wrf_substep_index_1based": "n_acoustic_substeps",
                    "note": "Record for every requested RK stage; de-duplicate if n_acoustic_substeps == 1.",
                },
            ],
            "stage_boundaries": [
                "stage input",
                "post large-step total tendencies before acoustic",
                "first requested acoustic sample input/output",
                "last requested acoustic sample input/output",
                "final stage state",
            ],
        },
        "selection": {
            "selected_cell_count": len(cells),
            "candidate_filters": loc.get("candidate_filters", {}),
            "candidate_source": loc.get("candidate_source"),
            "recommended_vertical_levels_for_first_probe": selected_vertical_levels(dynamic),
            "full_column_required": True,
            "full_column_reason": (
                "The listed levels are reporting probes only. WRF savepoints must "
                "write full native vertical columns for every selected cell/face "
                "because horizontal stencils, vertical coupling, pressure/geopotential "
                "refreshes, W/PH updates, and acoustic work arrays can depend on "
                "levels outside the first-probe list."
            ),
            "vertical_counts_from_source": counts,
            "native_staggering": {
                "mass_scalar": "bottom_top, south_north, west_east",
                "surface": "south_north, west_east",
                "U": "bottom_top, south_north, west_east_stag",
                "V": "bottom_top, south_north_stag, west_east",
                "W": "bottom_top_stag, south_north, west_east",
                "PH": "bottom_top_stag, south_north, west_east",
            },
            "selected_cells": cells,
        },
        "requested_wrf_source_term_groups": list(TERM_GROUPS),
        "expected_wrf_savepoint_artifact_schema": artifact_schema(),
        "wrf_source_build_feasibility_dependency": feasibility_dependency,
        "unresolved_dependencies": unresolved_dependencies,
    }


def write_markdown(path: Path, manifest: Mapping[str, Any]) -> None:
    run = manifest["run_request"]
    selection = manifest["selection"]
    source = manifest["source_proofs"]["authoritative_selected_cell_manifest"]
    cells = selection["selected_cells"]
    cell_inline = ", ".join(
        f"{cell['selection_rank']}:(y={cell['mass_index_zero_based']['south_north']},"
        f"x={cell['mass_index_zero_based']['west_east']})"
        for cell in cells
    )
    term_groups = ", ".join(group["id"] for group in manifest["requested_wrf_source_term_groups"])
    feasibility = manifest["wrf_source_build_feasibility_dependency"]
    if feasibility["status"] == "available_in_workspace":
        dependency_lines = [
            (
                "Sartre's WRF source/build feasibility artifact is present and should "
                "be used for the instrumentation tree/build choice."
            ),
            "",
            f"- Feasibility JSON: `{feasibility['proof_json']['path']}`",
            f"- Recommended path summary: {feasibility['fastest_reliable_path']}",
        ]
    else:
        dependency_lines = [
            (
                "Sartre's WRF source/build feasibility artifact was not present in this "
                "workspace. The manifest is ready for that worker, but exact WRF "
                "checkout/build paths still depend on that feasibility closeout."
            )
        ]
    lines = [
        "# V0.14 Same-State Savepoint Request",
        "",
        "## Verdict",
        "",
        (
            "Generated a CPU-WRF instrumentation request for h10 same-state "
            f"savepoints: domain `{run['domain']}`, run `{run['run_id']}`, "
            f"valid time `{run['selected_valid_time_utc']}`, "
            f"{selection['selected_cell_count']} selected mass-grid cells."
        ),
        "",
        "This is only a request manifest. It makes no equivalence or root-cause claim.",
        "",
        "## Source",
        "",
        (
            f"- Authoritative selected-cell proof: `{source['path']}` "
            f"sha256 `{source['sha256']}`"
        ),
        (
            f"- Selection source: `{run['selection_source']}`; retained JAX output is "
            "used only for choosing cells, not as same-state truth."
        ),
        "",
        "## Request",
        "",
        "- Instrument one real CPU-WRF large timestep at the h10 target and record the exact WRF model-step number, `Times`, `dt`, RK weights, and namelist switches.",
        "- Save all three RK stages and the first plus last acoustic substep for each stage.",
        "- Save native-staggered arrays, not destaggered diagnostics.",
        (
            "- Write full native vertical columns for each selected mass cell and "
            "adjacent U/V/W/PH faces. The first-probe reporting levels are "
            f"`{selection['recommended_vertical_levels_for_first_probe']}`, but "
            "they are not sufficient for stencil or vertical-coupling terms."
        ),
        "- Use the per-cell patch bounds in JSON. Bounds are zero-based stop-exclusive, with one-based inclusive WRF/Fortran translations included.",
        "",
        "## Cells",
        "",
        cell_inline,
        "",
        "Full lat/lon, patch bounds, native faces, and diagnostic context are in the JSON artifact.",
        "",
        "## Term Groups",
        "",
        term_groups,
        "",
        "The JSON expands each group into timing, native grid, and requested term arrays.",
        "",
        "## Expected Artifact",
        "",
        (
            "Preferred output is one compact NetCDF4/Zarr/HDF5 artifact named like "
            "`wrf_same_state_savepoints_20260501_18z_l2_72h_20260519T173026Z_d02_h10.<nc|zarr|h5>`."
        ),
        (
            "It must carry global WRF build/namelist/instrumentation identifiers, "
            "the selected-cell manifest echo, native patch bounds, per-term dataset "
            "metadata, and companion build/run logs."
        ),
        "",
        "## Dependency",
        "",
        *dependency_lines,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    manifest = build_manifest()
    write_json(OUT_JSON, manifest)
    write_markdown(OUT_MD, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "json": str(OUT_JSON.relative_to(ROOT)),
                "md": str(OUT_MD.relative_to(ROOT)),
                "selected_lead_h": manifest["run_request"]["selected_lead_h"],
                "selected_cell_count": manifest["selection"]["selected_cell_count"],
                "term_group_count": len(manifest["requested_wrf_source_term_groups"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
