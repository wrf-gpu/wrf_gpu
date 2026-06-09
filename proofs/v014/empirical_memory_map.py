#!/usr/bin/env python3
"""Generate the v0.14 empirical/static memory map.

This proof is intentionally CPU/static only. It reads current-branch source and
prior proof artifacts, verifies the source patterns the analysis depends on,
computes fp64 target-grid byte estimates, and writes JSON plus a concise report.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs/v014/empirical_memory_map.json"
OUT_MD = ROOT / "proofs/v014/empirical_memory_map.md"

TARGET = {
    "nx": 641,
    "ny": 321,
    "nz": 50,
    "dtype": "fp64",
    "bytes_per_value": 8,
}


def _run_git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    return out.strip()


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _line_matches(path: str, pattern: str) -> list[dict[str, Any]]:
    rx = re.compile(pattern)
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(_read(path).splitlines(), 1):
        if rx.search(line):
            rows.append({"line": lineno, "text": line.strip()})
    return rows


def _pattern(path: str, pattern: str, *, min_count: int = 1) -> dict[str, Any]:
    rows = _line_matches(path, pattern)
    return {
        "path": path,
        "pattern": pattern,
        "min_count": min_count,
        "count": len(rows),
        "ok": len(rows) >= min_count,
        "matches": rows,
        "sha256": _sha256(path),
    }


def _load_json(path: str) -> Any:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _bytes(values: int) -> int:
    return int(values) * int(TARGET["bytes_per_value"])


def _fmt_bytes(nbytes: int) -> dict[str, Any]:
    return {
        "bytes": int(nbytes),
        "mib": round(nbytes / (1024**2), 3),
        "gib": round(nbytes / (1024**3), 6),
    }


def _shape_estimates() -> dict[str, Any]:
    nx = int(TARGET["nx"])
    ny = int(TARGET["ny"])
    nz = int(TARGET["nz"])
    return {
        "surface_2d": {
            "formula": "ny*nx",
            "shape": [ny, nx],
            **_fmt_bytes(_bytes(ny * nx)),
        },
        "mass_3d": {
            "formula": "nz*ny*nx",
            "shape": [nz, ny, nx],
            **_fmt_bytes(_bytes(nz * ny * nx)),
        },
        "w_face_3d": {
            "formula": "(nz+1)*ny*nx",
            "shape": [nz + 1, ny, nx],
            **_fmt_bytes(_bytes((nz + 1) * ny * nx)),
        },
        "u_face_3d": {
            "formula": "nz*ny*(nx+1)",
            "shape": [nz, ny, nx + 1],
            **_fmt_bytes(_bytes(nz * ny * (nx + 1))),
        },
        "v_face_3d": {
            "formula": "nz*(ny+1)*nx",
            "shape": [nz, ny + 1, nx],
            **_fmt_bytes(_bytes(nz * (ny + 1) * nx)),
        },
        "dense_column_pair": {
            "formula": "ny*nx*nz*nz",
            "shape": [ny * nx, nz, nz],
            **_fmt_bytes(_bytes(ny * nx * nz * nz)),
        },
    }


def _sum_gib(shape_bytes: dict[str, Any], items: list[tuple[str, float]]) -> float:
    total = 0.0
    for key, count in items:
        total += shape_bytes[key]["gib"] * count
    return round(total, 6)


def _arrays(shape_bytes: dict[str, Any], rows: list[tuple[str, str, str, float]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, shape_key, role, count in rows:
        unit = shape_bytes[shape_key]
        out.append(
            {
                "name": name,
                "shape_key": shape_key,
                "shape_formula": unit["formula"],
                "role": role,
                "count": count,
                "fp64_gib": round(unit["gib"] * count, 6),
            }
        )
    return out


def _prior_evidence() -> dict[str, Any]:
    exact = _load_json("proofs/v014/exact_branch_memory_preflight.json")
    rrtmg_tile = _load_json("proofs/v013/rrtmg_column_tile_vram_suite.json")
    gpoint = _load_json("proofs/v013/gpoint_chunk_rrtmg.json")
    optics = _load_json("proofs/v013/optics_taumol_chunk.json")
    return {
        "exact_branch_memory_preflight": {
            "path": "proofs/v014/exact_branch_memory_preflight.json",
            "loaded": exact is not None,
            "branch_controls_ok": bool(exact and exact.get("branch_controls", {}).get("ok")),
            "rrtmg_column_tiling_present": bool(
                exact and exact.get("branch_controls", {}).get("rrtmg_column_tiling_present")
            ),
            "nested_allocator_controls_present": bool(
                exact and exact.get("branch_controls", {}).get("nested_allocator_controls_present")
            ),
            "verdict": exact.get("verdict") if isinstance(exact, dict) else None,
        },
        "rrtmg_column_tile_vram_suite": {
            "path": "proofs/v013/rrtmg_column_tile_vram_suite.json",
            "loaded": rrtmg_tile is not None,
            "summary": [
                {
                    "kind": row.get("kind"),
                    "mode": row.get("column_mode"),
                    "result": row.get("result", "OK"),
                    "peak_mib": row.get("peak_mib"),
                }
                for row in (rrtmg_tile or {}).get("rows", [])
            ],
        },
        "gpoint_chunk_rrtmg": {
            "path": "proofs/v013/gpoint_chunk_rrtmg.json",
            "loaded": gpoint is not None,
            "sw_bit_identical": bool(gpoint and gpoint.get("inertness", {}).get("sw", {}).get("all_bit_identical")),
            "lw_bit_identical": bool(gpoint and gpoint.get("inertness", {}).get("lw", {}).get("all_bit_identical")),
        },
        "optics_taumol_chunk": {
            "path": "proofs/v013/optics_taumol_chunk.json",
            "loaded": optics is not None,
            "sw_bit_identical": bool(optics and optics.get("verdict", {}).get("sw_construction_inert_bit_identical")),
            "lw_bit_identical": bool(optics and optics.get("verdict", {}).get("lw_construction_inert_bit_identical")),
        },
    }


def _source_evidence() -> dict[str, Any]:
    checks = {
        "moisture_velocity_two_build_sites": _pattern(
            "src/gpuwrf/runtime/operational_mode.py",
            r"vel = couple_velocities_periodic\(",
            min_count=2,
        ),
        "moisture_active_gate": _pattern(
            "src/gpuwrf/runtime/operational_mode.py",
            r"moisture_advected = \(",
        ),
        "moisture_species_loop": _pattern(
            "src/gpuwrf/dynamics/flux_advection.py",
            r"def advect_moisture_scalars\(",
        ),
        "limited_scalar_workspace": _pattern(
            "src/gpuwrf/dynamics/flux_advection.py",
            r"def advect_scalar_flux_limited\(",
        ),
        "wdm6_full_column_slmsk_broadcast": _pattern(
            "src/gpuwrf/coupling/scan_adapters.py",
            r"slmsk = _mp_in\(jnp\.broadcast_to\(slmsk_2d\[None, :, :\], state\.theta\.shape\)",
        ),
        "wdm6_kernel_expects_column_slmsk": _pattern(
            "src/gpuwrf/physics/microphysics_wdm6.py",
            r"pass a \(ncol,\) array",
        ),
        "nonradiation_mp_column_flatten": _pattern(
            "src/gpuwrf/coupling/scan_adapters.py",
            r"def _mp_in\(field3d: jax\.Array, ny: int, nx: int, nz: int\)",
        ),
        "thompson_column_inputs": _pattern(
            "src/gpuwrf/coupling/physics_couplers.py",
            r"def _thompson_column_from_state",
        ),
        "thompson_output_replacements": _pattern(
            "src/gpuwrf/coupling/physics_couplers.py",
            r"updates = dict\(",
        ),
        "pbl_surface_forcing_rederived": _pattern(
            "src/gpuwrf/coupling/scan_adapters.py",
            r"diag = surface_layer_with_diagnostics\(_surface_column_view\(state\)\)",
        ),
        "surface_column_view_full_profiles": _pattern(
            "src/gpuwrf/coupling/physics_couplers.py",
            r"def _surface_column_view\(state: State\)",
        ),
        "mynn_boulac_dense_source": _pattern(
            "src/gpuwrf/physics/mynn_pbl.py",
            r"\(nz x nz\) PE-accumulation matrices",
        ),
        "mynn_boulac_dense_broadcast": _pattern(
            "src/gpuwrf/physics/mynn_pbl.py",
            r"theta_i = theta\[\.\.\., :, None\]",
        ),
        "post_physics_mynn_increment": _pattern(
            "src/gpuwrf/coupling/physics_couplers.py",
            r"return state\.replace\(",
            min_count=4,
        ),
        "acoustic_reverted_carry_split_note": _pattern(
            "src/gpuwrf/runtime/operational_mode.py",
            r"reverted carry-split",
        ),
        "acoustic_scan_full_pytree": _pattern(
            "src/gpuwrf/runtime/operational_mode.py",
            r"acoustic, _ = jax\.lax\.scan\(",
        ),
        "acoustic_core_state": _pattern(
            "src/gpuwrf/dynamics/core/acoustic.py",
            r"class AcousticCoreState",
        ),
        "acoustic_full_state_fields": _pattern(
            "src/gpuwrf/dynamics/core/acoustic.py",
            r"FULL_STATE_FIELDS = \(",
        ),
        "state_total_perturbation_aliases": _pattern(
            "src/gpuwrf/contracts/state.py",
            r"p_total|ph_total|mu_total",
            min_count=6,
        ),
        "small_step_prep_large_stage_bundle": _pattern(
            "src/gpuwrf/dynamics/core/small_step_prep.py",
            r"class SmallStepPrepState",
        ),
    }
    return {
        "checks": checks,
        "all_required_patterns_ok": all(item["ok"] for item in checks.values()),
    }


def _candidates(shape_bytes: dict[str, Any]) -> list[dict[str, Any]]:
    mass = shape_bytes["mass_3d"]["gib"]
    surface = shape_bytes["surface_2d"]["gib"]
    wface = shape_bytes["w_face_3d"]["gib"]

    moisture_lower = _sum_gib(
        shape_bytes,
        [("mass_3d", 2), ("w_face_3d", 1), ("surface_2d", 4)],
    )
    moisture_upper = _sum_gib(
        shape_bytes,
        [
            ("mass_3d", 2),
            ("w_face_3d", 1),
            ("surface_2d", 4),
            ("mass_3d", 5),
        ],
    )
    post_merge_outputs = round(17.0 * mass, 6)
    post_merge_with_deltas = round(34.0 * mass, 6)

    return [
        {
            "id": "wdm6_slmsk_full_column_broadcast",
            "rank": 1,
            "recommendation": "FIX_NOW_BIT_IDENTICAL",
            "recommendation_scope": "smallest safe post-grid-parity memory-only source sprint",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static with kernel API source evidence",
            "current_source_shape": "(nz,ny,nx) broadcast from xland, then _mp_in to (ncol,nz)",
            "target_shape": "(ncol,) scalar per column; preserve current 1/0 values for a pure layout proof",
            "fp64_estimate": {
                "current_gib": round(mass, 6),
                "target_gib": round(surface, 6),
                "recoverable_gib": round(mass - surface, 6),
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("slmsk_2d", "surface_2d", "land/sea mask before broadcast", 1),
                    ("slmsk_broadcast_current", "mass_3d", "full-column temporary in current adapter", 1),
                    ("slmsk_column_target", "surface_2d", "per-column replacement target", 1),
                ],
            ),
            "correctness_risk": "low for shape-only/current-value-preserving change; medium if also changing sea value from 0 to WRF 2",
            "required_proof_gate": [
                "WDM6 adapter shape-only exact-output proof on CPU",
                "WDM6 scheme oracle/smoke with xland containing both land and sea",
                "default suite unchanged because mp_physics=16 is opt-in",
            ],
            "source_refs": [
                "src/gpuwrf/coupling/scan_adapters.py:329",
                "src/gpuwrf/coupling/scan_adapters.py:330",
                "src/gpuwrf/physics/microphysics_wdm6.py:1337",
            ],
        },
        {
            "id": "moisture_advection_duplicate_transport_velocity",
            "rank": 2,
            "recommendation": "FIX_NOW_BIT_IDENTICAL",
            "recommendation_scope": "safe only after grid parity, or adjacent to flux-advection work",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static, not measured peak",
            "activation": "only when use_flux_advection is true and moist_adv_opt != 0",
            "current_source_shape": "couple_velocities_periodic is built once in _augment_large_step_tendencies and again in _moisture_coupled_tendencies",
            "fp64_estimate": {
                "recoverable_lower_gib": moisture_lower,
                "recoverable_upper_source_static_gib": moisture_upper,
                "roadmap_static_range_gib": [0.45, 0.65],
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("ru", "mass_3d", "x-face coupled transport velocity collapsed to mass-column count", 1),
                    ("rv", "mass_3d", "y-face coupled transport velocity collapsed to mass-column count", 1),
                    ("rom", "w_face_3d", "vertical mass-coupled omega at w faces", 1),
                    ("msftx/msfux/msfvy/msfvx", "surface_2d", "map-factor leaves returned in CoupledVelocities", 4),
                    ("divv/increments/cum/mass_u/mass_v", "mass_3d", "construction transients that may overlap", 5),
                ],
            ),
            "correctness_risk": "low/medium: reuse must preserve exact WRF scalar cadence and active moisture limiter ordering",
            "required_proof_gate": [
                "default moist_adv_opt=0 bit identity",
                "active moist_adv_opt=1/2 conservation, positivity, and cadence proof",
                "rerun proofs/v013/moisture_advection_wiring.json or successor",
                "no new host/device transfers in timestep loop",
            ],
            "source_refs": [
                "src/gpuwrf/runtime/operational_mode.py:1770",
                "src/gpuwrf/runtime/operational_mode.py:2045",
                "src/gpuwrf/runtime/operational_mode.py:2221",
                "src/gpuwrf/dynamics/flux_advection.py:171",
            ],
        },
        {
            "id": "mynn_boulac_dense_column_pair_matrices",
            "rank": 3,
            "recommendation": "MEASURE_FIRST",
            "recommendation_scope": "first HLO/RSS/short VRAM target if MYNN is the selected PBL path",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static dense-shape evidence; no current exact-branch peak measurement",
            "current_source_shape": "MYNN _boulac_length vectorizes nested searches into (...,nz,nz) matrices",
            "fp64_estimate": {
                "one_dense_matrix_gib": shape_bytes["dense_column_pair"]["gib"],
                "three_dense_matrices_gib": round(3 * shape_bytes["dense_column_pair"]["gib"], 6),
                "six_dense_matrices_gib": round(6 * shape_bytes["dense_column_pair"]["gib"], 6),
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("up_incr/zup/rad_up/etc.", "dense_column_pair", "upward BouLac displacement dense intermediates", 3),
                    ("do_incr/zdo/rad_do/etc.", "dense_column_pair", "downward BouLac displacement dense intermediates", 3),
                ],
            ),
            "correctness_risk": "high: changing this alters MYNN length-scale search structure and can change PBL mixing",
            "required_proof_gate": [
                "CPU HLO/static liveness or RSS proof that dense arrays materialize",
                "MYNN WRF oracle and column exactness/tolerance proof",
                "coupled PBL real-case smoke before any long validation reuse",
            ],
            "source_refs": [
                "src/gpuwrf/physics/mynn_pbl.py:381",
                "src/gpuwrf/physics/mynn_pbl.py:388",
                "src/gpuwrf/physics/mynn_pbl.py:410",
            ],
        },
        {
            "id": "non_radiation_column_physics_tiling",
            "rank": 4,
            "recommendation": "MEASURE_FIRST",
            "recommendation_scope": "one scheme at a time after a measured offender is identified",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static adapter shape inventory, inferred peak",
            "current_source_shape": "whole-domain column batches (ncol,nz) for microphysics/PBL/cumulus adapters",
            "fp64_estimate": {
                "one_profile_leaf_gib": mass,
                "typical_visible_input_output_range_gib": [1.0, 3.0],
                "thompson_visible_input_approx_gib": round(15 * mass, 6),
                "thompson_output_replacements_approx_gib": round(11 * mass, 6),
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("microphysics profile inputs", "mass_3d", "8-15 full-domain column leaves depending on scheme", 12),
                    ("microphysics replacements", "mass_3d", "scheme output replacement leaves", 6),
                    ("surface accumulators", "surface_2d", "precip/cumulus increments", 4),
                ],
            ),
            "correctness_risk": "medium/high due per-scheme output shape, precipitation accumulators, and coupling order",
            "required_proof_gate": [
                "CPU tile-vs-untiled exact-output proof for selected scheme",
                "short GPU peak-VRAM suite for selected scheme",
                "real-case smoke with that scheme enabled",
            ],
            "source_refs": [
                "src/gpuwrf/coupling/scan_adapters.py:118",
                "src/gpuwrf/coupling/physics_couplers.py:1076",
                "src/gpuwrf/coupling/physics_couplers.py:1190",
            ],
        },
        {
            "id": "post_physics_non_dry_sparse_donated_merge",
            "rank": 5,
            "recommendation": "MEASURE_FIRST",
            "recommendation_scope": "defer source work until donation/liveness measurement exists",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "inferred from State.replace replacement leaves and roadmap arithmetic",
            "current_source_shape": "adapters build replacement leaves, then State.replace returns a full next State pytree",
            "fp64_estimate": {
                "output_leaves_static_gib": post_merge_outputs,
                "with_separate_deltas_static_gib": post_merge_with_deltas,
                "roadmap_static_range_gib": [1.33, 2.64],
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("non-dry replacement leaves", "mass_3d", "theta/moisture/number/qke-style outputs", 17),
                    ("optional deltas", "mass_3d", "increment or pre-merge deltas if not donated/fused", 17),
                ],
            ),
            "correctness_risk": "medium/high: changes coupling liveness, donation behavior, and exact leaf ownership",
            "required_proof_gate": [
                "exact default output proof over selected physics schemes",
                "JAX donation/alias audit and transfer audit",
                "short coupled GPU peak-VRAM run",
            ],
            "source_refs": [
                "src/gpuwrf/coupling/physics_couplers.py:1122",
                "src/gpuwrf/coupling/physics_couplers.py:1333",
                "src/gpuwrf/coupling/scan_adapters.py:1048",
            ],
        },
        {
            "id": "moisture_limiter_and_species_workspace",
            "rank": 6,
            "recommendation": "MEASURE_FIRST",
            "recommendation_scope": "do not rewrite until active moist_adv_opt path is a validation target",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static workspace inventory, inferred peak",
            "activation": "moist_adv_opt in {1,2} on final RK stage; plain path otherwise",
            "fp64_estimate": {
                "six_plain_outputs_gib": round(6 * mass, 6),
                "one_limited_scalar_visible_workspace_floor_gib": round(8 * mass + 3 * wface, 6),
                "roadmap_limited_workspace_range_gib": [1.0, 3.0],
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("q_tendencies for six species", "mass_3d", "tuple outputs retained before moisture large-step update", 6),
                    ("fqx/fqy/fqz low/high/limited", "mass_3d", "horizontal flux and scale work per limited scalar", 8),
                    ("vertical flux faces", "w_face_3d", "fqzl/fqz/fqz_lim style work per limited scalar", 3),
                ],
            ),
            "correctness_risk": "medium: limiter order, positivity, monotonicity, and conservation are semantic",
            "required_proof_gate": [
                "per-species WRF transcription parity",
                "total water conservation and positivity/monotonicity proof",
                "active real-case smoke with moist_adv_opt=1/2",
            ],
            "source_refs": [
                "src/gpuwrf/dynamics/flux_advection.py:504",
                "src/gpuwrf/dynamics/flux_advection.py:579",
                "src/gpuwrf/dynamics/flux_advection.py:812",
                "src/gpuwrf/runtime/operational_mode.py:2224",
            ],
        },
        {
            "id": "pbl_surface_bottom_only_prep_and_duplicate_diagnostics",
            "rank": 7,
            "recommendation": "DEFER_SEMANTIC_OR_DYCORE",
            "recommendation_scope": "treat as PBL/surface correctness plumbing, not memory-only cleanup",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static plus roadmap estimate",
            "current_source_shape": "surface diagnostics and PBL surface forcing are rederived from full column views",
            "fp64_estimate": {
                "full_surface_column_view_six_profiles_gib": round(6 * mass, 6),
                "roadmap_recoverable_range_gib": [0.3, 0.8],
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("surface_column_view profiles", "mass_3d", "u/v/theta/qv/p/dz-style profiles for surface layer", 6),
                    ("pbl_surface_forcing profiles", "mass_3d", "u/v/theta/T/qv/p/pii/rho/dz/z views for PBL", 10),
                    ("surface diagnostics", "surface_2d", "hfx/lh/br/psim/psih/u10/v10/znt/etc.", 10),
                ],
            ),
            "correctness_risk": "high: selected sfclay diagnostics and PBL handoff semantics are coupled",
            "required_proof_gate": [
                "WRF surface-driver to PBL-driver contract proof for selected pairs",
                "fail-close unproven pairs",
                "coupled real-case PBL/surface gate",
            ],
            "source_refs": [
                "src/gpuwrf/coupling/scan_adapters.py:969",
                "src/gpuwrf/coupling/scan_adapters.py:979",
                "src/gpuwrf/coupling/physics_couplers.py:1436",
                "src/gpuwrf/coupling/physics_couplers.py:1490",
            ],
        },
        {
            "id": "acoustic_scan_carry_split_evolving_only",
            "rank": 8,
            "recommendation": "DO_NOT_DO_BEFORE_GRID_PARITY",
            "recommendation_scope": "dycore-adjacent; co-design with later acoustic precision/base-state work",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static plus prior reverted-attempt note",
            "current_source_shape": "full AcousticCoreState pytree is carried through lax.scan; source note says ~19 evolving and ~50 stage-constant leaves",
            "fp64_estimate": {
                "roadmap_recoverable_static_gib": 1.56,
                "twenty_mass_grid_arrays_gib": round(20 * mass, 6),
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("evolving acoustic leaves", "mass_3d", "mu/theta/u/v/w/ph/p/t_2ave family, staggered in real code", 19),
                    ("stage-constant leaves in carry", "mass_3d", "prep/metric/reference fields closed over instead of carried by split design", 20),
                ],
            ),
            "correctness_risk": "high: acoustic substep liveness and prior split attempt were reverted",
            "required_proof_gate": [
                "default fp64 bit identity",
                "acoustic savepoint parity",
                "warm-bubble/Straka/terrain-rest gates",
                "short GPU smoke with corrected cache-hit timing and transfer audit",
            ],
            "source_refs": [
                "src/gpuwrf/runtime/operational_mode.py:1667",
                "src/gpuwrf/runtime/operational_mode.py:1686",
                "src/gpuwrf/dynamics/core/acoustic.py:104",
            ],
        },
        {
            "id": "state_total_perturbation_base_alias_reduction",
            "rank": 9,
            "recommendation": "DEFER_SEMANTIC_OR_DYCORE",
            "recommendation_scope": "ADR-required ABI/schema work, not a v0.14 memory cleanup",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "source-static State schema",
            "current_source_shape": "State carries total and perturbation/base-adjacent p/ph/mu families",
            "fp64_estimate": {
                "one_mass_3d_leaf_gib": mass,
                "one_w_face_leaf_gib": wface,
                "roadmap_recoverable_range_gib": [0.16, 0.32],
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("p/p_total/p_perturbation family", "mass_3d", "state ABI duplicate family", 1),
                    ("ph/ph_total/ph_perturbation family", "w_face_3d", "state ABI duplicate family", 1),
                    ("mu/mu_total/mu_perturbation family", "surface_2d", "state ABI duplicate family", 1),
                ],
            ),
            "correctness_risk": "high ABI risk across init, restart, wrfout, boundaries, and comparators",
            "required_proof_gate": [
                "ADR first",
                "restart roundtrip and wrfout compatibility",
                "boundary/savepoint parity",
            ],
            "source_refs": [
                "src/gpuwrf/contracts/state.py:397",
                "src/gpuwrf/contracts/state.py:462",
            ],
        },
        {
            "id": "small_dycore_masks_and_pad_helpers",
            "rank": 10,
            "recommendation": "NOT_WORTH_STANDALONE",
            "recommendation_scope": "only do adjacent to acoustic/dycore correctness work",
            "blocks_v014_long_validation_after_grid_parity": False,
            "evidence_strength": "roadmap/static only",
            "current_source_shape": "small full-column dycore helper masks/pad buffers such as dry cqw paths",
            "fp64_estimate": {
                "one_mass_3d_leaf_gib": mass,
                "roadmap_individual_range_gib": [0.078, 0.30],
            },
            "likely_concurrent_arrays": _arrays(
                shape_bytes,
                [
                    ("dry_cqw/full face masks", "mass_3d", "small dycore helper buffers", 1),
                    ("pad-based face helper transient", "mass_3d", "legacy dycore helper transient", 2),
                ],
            ),
            "correctness_risk": "medium because even tiny dycore buffers sit in implicit-w/acoustic paths",
            "required_proof_gate": [
                "focused dycore unit tests",
                "warm-bubble/Straka/terrain-rest CPU gates",
            ],
            "source_refs": [
                "src/gpuwrf/runtime/operational_mode.py:1623",
                "src/gpuwrf/dynamics/core/acoustic.py:927",
            ],
        },
    ]


def _ranked_recommendation(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "verdict": "NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY",
        "smallest_safe_memory_source_sprint": "WDM6 slmsk shape-only cleanup, preserving current values and proving exact WDM6 output equality",
        "only_material_bit_identical_cleanup": "moisture transport velocity reuse when active moisture advection matters",
        "must_measure_before_rewrite": [
            "MYNN BouLac dense matrices",
            "non-radiation whole-domain column physics tiling",
            "post-physics donated/sparse merge",
            "moisture limiter workspace",
        ],
        "do_not_start_before_grid_parity": [
            "acoustic carry split / evolving-only carry",
            "FP32 acoustic source work",
            "PBL/surface semantic diagnostic threading",
        ],
        "ranked_ids": [c["id"] for c in sorted(candidates, key=lambda row: row["rank"])],
    }


def _build() -> dict[str, Any]:
    shape_bytes = _shape_estimates()
    source = _source_evidence()
    candidates = _candidates(shape_bytes)
    prior = _prior_evidence()
    status_short = _run_git(["status", "--short", "--branch"]).splitlines()
    data = {
        "proof": "v0.14 empirical/static memory map for remaining non-radiation memory risks",
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "execution_constraints": {
            "cpu_only": True,
            "gpu_used": False,
            "tost_run": False,
            "switzerland_validation_run": False,
            "production_src_edited": False,
            "fp32_source_work_started": False,
        },
        "git": {
            "branch": _run_git(["branch", "--show-current"]),
            "head": _run_git(["rev-parse", "HEAD"]),
            "dirty": bool([line for line in status_short if line and not line.startswith("## ")]),
            "status_short": status_short,
        },
        "target_geometry": TARGET,
        "shape_byte_estimates": shape_bytes,
        "prior_radiation_evidence_treated_as_fixed": prior,
        "source_evidence": source,
        "candidates": candidates,
        "recommendation": _ranked_recommendation(candidates),
        "validation": {
            "script": "proofs/v014/empirical_memory_map.py",
            "json": "proofs/v014/empirical_memory_map.json",
            "markdown": "proofs/v014/empirical_memory_map.md",
            "source_patterns_ok": source["all_required_patterns_ok"],
            "required_commands": [
                "python -m py_compile proofs/v014/empirical_memory_map.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/empirical_memory_map.py",
                "python -m json.tool proofs/v014/empirical_memory_map.json >/tmp/empirical_memory_map.validated.json",
            ],
        },
    }
    return data


def _write_md(data: dict[str, Any]) -> None:
    rec = data["recommendation"]
    candidates = sorted(data["candidates"], key=lambda row: row["rank"])
    lines = [
        "# v0.14 Empirical/Static Memory Map",
        "",
        f"- Verdict: `{rec['verdict']}`",
        f"- Branch: `{data['git']['branch']}`",
        f"- HEAD: `{data['git']['head']}`",
        f"- Dirty worktree: `{data['git']['dirty']}`",
        "- GPU/TOST/Switzerland/FP32 source work: not run",
        "",
        "## Decision",
        "",
        "RRTMG column/band/optics tiling remains prior fixed evidence. On the exact current branch, the remaining non-radiation memory items are not blockers for the first long validation after grid-cell parity closes. Run the short exact-branch memory preflight again for the selected long-run configuration, but do not hold validation for new broad memory rewrites.",
        "",
        f"Smallest safe memory-only source sprint: `{rec['smallest_safe_memory_source_sprint']}`.",
        f"Only material bit-identical cleanup: `{rec['only_material_bit_identical_cleanup']}`.",
        "",
        "## Ranked Map",
        "",
        "| Rank | Candidate | Recommendation | Target fp64 estimate | Blocks long validation? |",
        "|---:|---|---|---:|---|",
    ]
    for row in candidates:
        estimate = row.get("fp64_estimate", {})
        if "recoverable_gib" in estimate:
            est = f"{estimate['recoverable_gib']} GiB recoverable"
        elif "recoverable_upper_source_static_gib" in estimate:
            est = f"{estimate['recoverable_lower_gib']}-{estimate['recoverable_upper_source_static_gib']} GiB"
        elif "one_dense_matrix_gib" in estimate:
            est = f"{estimate['one_dense_matrix_gib']} GiB per dense matrix"
        elif "output_leaves_static_gib" in estimate:
            est = f"{estimate['output_leaves_static_gib']}-{estimate['with_separate_deltas_static_gib']} GiB"
        elif "six_plain_outputs_gib" in estimate:
            est = f"{estimate['six_plain_outputs_gib']} GiB outputs; limiter higher"
        else:
            est = "see JSON"
        lines.append(
            f"| {row['rank']} | `{row['id']}` | `{row['recommendation']}` | {est} | `{row['blocks_v014_long_validation_after_grid_parity']}` |"
        )
    lines.extend(
        [
            "",
            "## Proof Notes",
            "",
            "- `FIX_NOW_BIT_IDENTICAL` here means safe as a post-grid-parity memory-only sprint with exact-output proof, not a reason to interrupt grid parity.",
            "- `MEASURE_FIRST` items have plausible GiB-scale upside but need HLO/RSS or short GPU peak evidence before source work.",
            "- `DEFER_SEMANTIC_OR_DYCORE` and `DO_NOT_DO_BEFORE_GRID_PARITY` items touch PBL/dycore semantics or previous reverted acoustic work.",
            "- Detailed array formulas, source-pattern checks, proof references, and proof gates are in `proofs/v014/empirical_memory_map.json`.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    data = _build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    _write_md(data)
    print(
        "empirical_memory_map wrote "
        f"{OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}; "
        f"source_patterns_ok={data['validation']['source_patterns_ok']}"
    )
    return 0 if data["validation"]["source_patterns_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
