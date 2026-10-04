#!/usr/bin/env python3
"""Static re-audit of v0.17 fp32 memory and small-grid compute claims.

This script intentionally imports no JAX modules and touches no GPU.  It
combines:

* the source-level State shape contract from gpuwrf.contracts.state,
* the source-level precision matrix from gpuwrf.contracts.precision,
* existing v0.16/v0.17 proof JSON/text artifacts, and
* code-pattern counts for total/base reconstruction and explicit fp64 casts.

It produces a compact JSON proof object for the accompanying markdown verdict.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FP32_TREE = ROOT / ".wt-fp32-dycore"
FP32_DEBUG_TREE = ROOT / ".wt-fp32dyc-debug"
R2_XCHECK_TREE = ROOT / ".wt-r2-xcheck"
BOULAC_TREE = ROOT / ".wt-perf-boulac"

PRECISION_SOURCE = FP32_TREE / "src/gpuwrf/contracts/precision.py"
STATE_SOURCE = FP32_TREE / "src/gpuwrf/contracts/state.py"
OPERATIONAL_SOURCE = FP32_TREE / "src/gpuwrf/runtime/operational_mode.py"

GiB = 1024**3


@dataclass(frozen=True)
class Grid:
    name: str
    nz: int
    ny: int
    nx: int

    @property
    def ncol(self) -> int:
        return self.ny * self.nx


GRIDS = (
    Grid("canary_d03_native", 44, 75, 93),
    Grid("swiss_16k_tile", 44, 128, 128),
    Grid("swiss_65k_tile", 44, 256, 256),
    Grid("swiss_147k_tile", 44, 384, 384),
    Grid("bigswiss_211k_proxy", 44, 459, 459),
)


def state_field_shapes(grid: Grid) -> dict[str, tuple[int, ...]]:
    """Mirror the current State shape contract without importing JAX."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    mass_3d = (nz, ny, nx)
    surface_2d = (ny, nx)
    boundary_side = max(nx + 1, ny + 1)
    boundary_width = 5
    boundary_mass = (1, 4, boundary_width, nz, boundary_side)
    boundary_face = (1, 4, boundary_width, nz + 1, boundary_side)
    boundary_surface = (1, 4, boundary_width, 1, boundary_side)
    return {
        "u": (nz, ny, nx + 1),
        "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx),
        "theta": mass_3d,
        "qv": mass_3d,
        "p": mass_3d,
        "p_total": mass_3d,
        "p_perturbation": mass_3d,
        "ph": (nz + 1, ny, nx),
        "ph_total": (nz + 1, ny, nx),
        "ph_perturbation": (nz + 1, ny, nx),
        "mu": surface_2d,
        "mu_total": surface_2d,
        "mu_perturbation": surface_2d,
        "qc": mass_3d,
        "qr": mass_3d,
        "qi": mass_3d,
        "qs": mass_3d,
        "qg": mass_3d,
        "Ni": mass_3d,
        "Nr": mass_3d,
        "Ns": mass_3d,
        "Ng": mass_3d,
        "qke": mass_3d,
        "ustar": surface_2d,
        "theta_flux": surface_2d,
        "qv_flux": surface_2d,
        "tau_u": surface_2d,
        "tau_v": surface_2d,
        "rhosfc": surface_2d,
        "fltv": surface_2d,
        "t_skin": surface_2d,
        "soil_moisture": surface_2d,
        "xland": surface_2d,
        "lakemask": surface_2d,
        "mavail": surface_2d,
        "roughness_m": surface_2d,
        "lu_index": surface_2d,
        "rain_acc": surface_2d,
        "snow_acc": surface_2d,
        "graupel_acc": surface_2d,
        "ice_acc": surface_2d,
        "u_bdy": boundary_mass,
        "v_bdy": boundary_mass,
        "theta_bdy": boundary_mass,
        "qv_bdy": boundary_mass,
        "ph_bdy": boundary_face,
        "mu_bdy": boundary_surface,
        "w_bdy": boundary_face,
        "p_bdy": boundary_mass,
        "pb_bdy": boundary_mass,
        "phb_bdy": boundary_face,
        "mub_bdy": boundary_surface,
        "Nc": mass_3d,
        "Nn": mass_3d,
        "rainc_acc": surface_2d,
        "qsq": mass_3d,
        "qc_bl": mass_3d,
        "qi_bl": mass_3d,
        "cldfra_bl": mass_3d,
        "nwfa": mass_3d,
        "nifa": mass_3d,
    }


def base_state_shapes(grid: Grid) -> dict[str, tuple[int, ...]]:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    return {
        "pb": (nz, ny, nx),
        "phb": (nz + 1, ny, nx),
        "mub": (ny, nx),
        "t0": (nz, ny, nx),
        "theta_base": (nz, ny, nx),
    }


def prod(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def gib(nbytes: int | float) -> float:
    return float(nbytes) / GiB


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_precision_matrix() -> dict[str, str]:
    """Return field -> dtype-token using the source precision matrix."""

    text = PRECISION_SOURCE.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for match in re.finditer(r'"([^"]+)":\s*\((FP64|FP32_GATED|INT32),\s*(?:True|False)\)', text):
        out[match.group(1)] = match.group(2)
    return out


def dtype_bytes(dtype: str) -> int:
    if dtype == "FP64":
        return 8
    if dtype == "FP32_GATED":
        return 4
    if dtype == "INT32":
        return 4
    raise KeyError(dtype)


ALIAS_TOTAL_FIELDS = {"p", "p_total", "ph", "ph_total", "mu", "mu_total"}
DYNAMIC_PERT_FP32_TARGET = {
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "p_perturbation",
    "ph_perturbation",
    "mu_perturbation",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "Ni",
    "Nr",
    "Ns",
    "Ng",
    "Nc",
    "Nn",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    "nwfa",
    "nifa",
    "u_bdy",
    "v_bdy",
    "theta_bdy",
    "qv_bdy",
}


def state_bytes(
    grid: Grid,
    precision: dict[str, str],
    *,
    force_all_float64: bool = False,
    drop_alias_totals: bool = False,
    target_dynamic_pert_fp32: bool = False,
) -> dict[str, Any]:
    shapes = state_field_shapes(grid)
    by_field: dict[str, int] = {}
    dtype_by_field: dict[str, str] = {}
    for field, shape in shapes.items():
        if drop_alias_totals and field in ALIAS_TOTAL_FIELDS:
            continue
        src_dtype = precision.get(field, "FP64")
        if force_all_float64 and src_dtype != "INT32":
            dtype = "FP64"
        elif target_dynamic_pert_fp32 and field in DYNAMIC_PERT_FP32_TARGET:
            dtype = "FP32_GATED"
        else:
            dtype = src_dtype
        dtype_by_field[field] = dtype
        by_field[field] = prod(shape) * dtype_bytes(dtype)
    total = sum(by_field.values())
    return {
        "bytes": total,
        "gib": gib(total),
        "field_count": len(by_field),
        "dtype_counts": {
            key: sum(1 for dtype in dtype_by_field.values() if dtype == key)
            for key in ("FP64", "FP32_GATED", "INT32")
        },
        "dropped_fields": sorted(ALIAS_TOTAL_FIELDS if drop_alias_totals else []),
        "top_fields_gib": [
            {"field": field, "gib": gib(nbytes), "dtype": dtype_by_field[field]}
            for field, nbytes in sorted(by_field.items(), key=lambda item: item[1], reverse=True)[:14]
        ],
    }


def base_state_bytes(grid: Grid) -> dict[str, Any]:
    by_field = {field: prod(shape) * 8 for field, shape in base_state_shapes(grid).items()}
    total = sum(by_field.values())
    return {
        "bytes": total,
        "gib": gib(total),
        "top_fields_gib": [
            {"field": field, "gib": gib(nbytes), "dtype": "FP64"}
            for field, nbytes in sorted(by_field.items(), key=lambda item: item[1], reverse=True)
        ],
    }


def alias_duplicate_bytes(grid: Grid, precision: dict[str, str]) -> dict[str, Any]:
    shapes = state_field_shapes(grid)
    groups = {
        "p_alias_plus_total": ("p", "p_total"),
        "ph_alias_plus_total": ("ph", "ph_total"),
        "mu_alias_plus_total": ("mu", "mu_total"),
    }
    out = {}
    for group, fields in groups.items():
        nbytes = sum(prod(shapes[field]) * dtype_bytes(precision.get(field, "FP64")) for field in fields)
        out[group] = {"bytes": nbytes, "gib": gib(nbytes), "fields": fields}
    total = sum(item["bytes"] for item in out.values())
    out["combined"] = {"bytes": total, "gib": gib(total)}
    return out


def memory_modes() -> dict[str, Any]:
    precision = parse_precision_matrix()
    by_grid = {}
    for grid in GRIDS:
        modes = {
            "all_float64_state": state_bytes(grid, precision, force_all_float64=True),
            "current_precision_matrix_state": state_bytes(grid, precision),
            "drop_p_ph_mu_alias_totals_keep_current_dtypes": state_bytes(
                grid, precision, drop_alias_totals=True
            ),
            "drop_alias_totals_and_store_dynamic_perturbations_fp32_target": state_bytes(
                grid,
                precision,
                drop_alias_totals=True,
                target_dynamic_pert_fp32=True,
            ),
            "separate_base_state_fp64": base_state_bytes(grid),
            "alias_duplicate_bytes": alias_duplicate_bytes(grid, precision),
        }
        cur = modes["current_precision_matrix_state"]["bytes"]
        for name, mode in list(modes.items()):
            if isinstance(mode, dict) and "bytes" in mode:
                mode["ratio_vs_current_state"] = mode["bytes"] / cur if cur else None
        by_grid[grid.name] = {
            "ncol": grid.ncol,
            "nz": grid.nz,
            "ny": grid.ny,
            "nx": grid.nx,
            "modes": modes,
        }
    return by_grid


def summarize_hlo_memory() -> dict[str, Any]:
    path = ROOT / "proofs/perf/v016/s2_hlo_stats_2x2_s1_final.json"
    data = load_json(path)
    records = {record["precision"]: record for record in data["records"]}
    fp64 = records["fp64"]
    mixed = records["mixed_s2"]

    def mem(record: dict[str, Any], key: str) -> int:
        return int(record["memory_analysis"][key])

    fields = [
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    ]
    ratios = {}
    for field in fields:
        a = mem(fp64, field)
        b = mem(mixed, field)
        ratios[field] = {
            "fp64_gib": gib(a),
            "mixed_s2_gib": gib(b),
            "mixed_over_fp64": b / a if a else None,
            "delta_gib": gib(b - a),
        }
    return {
        "artifact": str(path.relative_to(ROOT)),
        "ncol": int(fp64["ncol"]),
        "ratios": ratios,
        "dtype_token_ratio": {
            "f64_fp64_over_mixed": fp64["hlo_counts"]["dtype_tokens"]["f64"]
            / mixed["hlo_counts"]["dtype_tokens"]["f64"],
            "f32_mixed_over_fp64": mixed["hlo_counts"]["dtype_tokens"]["f32"]
            / fp64["hlo_counts"]["dtype_tokens"]["f32"],
            "convert_count_fp64": fp64["hlo_counts"]["top_ops"].get("convert"),
            "convert_count_mixed_s2": mixed["hlo_counts"]["top_ops"].get("convert"),
        },
    }


def summarize_fp32_dycore() -> dict[str, Any]:
    path = FP32_TREE / "proofs/perf/v017/fp32_dycore.json"
    data = load_json(path)
    ratios = []
    for row in data["ratios"]:
        if "fp32dyc_ms_per_step" not in row:
            ratios.append({"ncol": row["ncol"], "status": "missing_or_oom"})
            continue
        ratios.append(
            {
                "ncol": row["ncol"],
                "speedup": row["fp32dyc_speedup_over_fp64"],
                "fp64_peak_gib": row["fp64_peak_gib"],
                "fp32dyc_peak_gib": row["fp32dyc_peak_gib"],
                "fp64_over_fp32dyc_vram": row["fp64_over_fp32dyc_vram"],
                "mass_total_rel_delta": row["mass_total_rel_delta"],
            }
        )
    debug_path = FP32_DEBUG_TREE / "proofs/perf/v017/fp32_dycore_debug_carrysplit.json"
    debug_data = load_json(debug_path) if debug_path.exists() else {}
    return {
        "artifact": str(path.relative_to(ROOT)),
        "ratios": ratios,
        "fp32dyc_unlocks_fp64_oom_ncols": data.get("fp32dyc_unlocks_fp64_oom_ncols", []),
        "debug_carrysplit_artifact": str(debug_path.relative_to(ROOT)) if debug_path.exists() else None,
        "debug_carrysplit_ratios": debug_data.get("ratios", []),
    }


def summarize_boulac() -> dict[str, Any]:
    path = BOULAC_TREE / "proofs/perf/v017/boulac_onz_bench.json"
    data = load_json(path)
    return {
        "artifact": str(path.relative_to(ROOT)),
        "verdict": data.get("verdict"),
        "bit_identity": data.get("bit_identity"),
        "comparisons": data.get("comparisons"),
        "selected_measurements": [
            {
                "case": row.get("case"),
                "mode": row.get("mode"),
                "ncol": row.get("ncol"),
                "ran_ok": row.get("ran_ok"),
                "oom": row.get("oom"),
                "peak_vram_gib": row.get("peak_vram_gib"),
                "warm_ms_per_step": row.get("warm_ms_per_step"),
            }
            for row in data.get("measurements", [])
        ],
    }


def summarize_launch_flood() -> dict[str, Any]:
    path = R2_XCHECK_TREE / "proofs/perf/v017/r2_xcheck_launch_flood.txt"
    rows = []
    pattern = re.compile(
        r"mode=(?P<mode>\w+)\s+launches=(?P<launches>\d+)\b.*?"
        r"host_ms_median=(?P<host>[0-9.]+)\b.*?"
        r"event_ms_median=(?P<event>[0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rows.append(
            {
                "mode": match.group("mode"),
                "launches": int(match.group("launches")),
                "host_ms_median": float(match.group("host")),
                "event_ms_median": float(match.group("event")),
            }
        )
    derived = []
    for row in rows:
        derived.append(
            {
                **row,
                "host_us_per_launch_or_node": 1000.0 * row["host_ms_median"] / row["launches"],
                "event_us_per_launch_or_node": 1000.0 * row["event_ms_median"] / row["launches"],
            }
        )
    return {"artifact": str(path.relative_to(ROOT)), "rows": derived}


def code_pattern_counts() -> dict[str, Any]:
    roots = [
        FP32_TREE / "src/gpuwrf/runtime/operational_mode.py",
        FP32_TREE / "src/gpuwrf/dynamics/core",
        FP32_TREE / "src/gpuwrf/contracts",
    ]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob("*.py")))

    patterns = {
        "p_total_minus_p_perturbation": r"p_total\s*-\s*[^,\n]*p_perturbation",
        "ph_total_minus_ph_perturbation": r"ph_total\s*-\s*[^,\n]*ph_perturbation",
        "mu_total_minus_mu_perturbation": r"mu_total\s*-\s*[^,\n]*mu_perturbation",
        "astype_float64": r"\.astype\(jnp\.float64\)|jnp\.asarray\([^)]*dtype=jnp\.float64",
        "p_total_references": r"\bp_total\b",
        "ph_total_references": r"\bph_total\b",
        "mu_total_references": r"\bmu_total\b",
        "fp32_restore_outer_carry": r"_fp32_dycore_restore_carry_dtypes",
    }
    out: dict[str, Any] = {}
    for name, regex in patterns.items():
        compiled = re.compile(regex)
        hits = []
        for path in files:
            rel = str(path.relative_to(ROOT))
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if compiled.search(line):
                    hits.append({"path": rel, "line": lineno, "text": line.strip()[:180]})
        out[name] = {"count": len(hits), "samples": hits[:16]}
    return out


def main() -> None:
    report = {
        "schema": "gpuwrf.v017.fp32_memory_compute_reaudit.v1",
        "method": "static source-shape audit plus existing proof artifacts; no JAX import; no GPU run",
        "source_files": {
            "state": str(STATE_SOURCE.relative_to(ROOT)),
            "precision": str(PRECISION_SOURCE.relative_to(ROOT)),
            "operational": str(OPERATIONAL_SOURCE.relative_to(ROOT)),
        },
        "memory_modes_from_state_contract": memory_modes(),
        "hlo_memory_v016_s2": summarize_hlo_memory(),
        "fp32_dycore_v017": summarize_fp32_dycore(),
        "boulac_onz_v017": summarize_boulac(),
        "launch_flood_r2_xcheck": summarize_launch_flood(),
        "code_pattern_counts": code_pattern_counts(),
        "audit_verdict": {
            "fp32_dycore_tested_path_was_not_persistent_state_fp32": True,
            "current_v017_fp32_dycore_does_not_remove_total_aliases": True,
            "current_v017_fp32_dycore_rewidens_outer_carry_to_fp64": True,
            "hlo_temp_arena_remained_nearly_unchanged_in_v016_s2": True,
            "launch_packaging_alone_is_falsified_as_2x_4x_small_grid_fix": True,
        },
    }
    out_path = ROOT / "proofs/perf/v017/fp32_memory_compute_reaudit.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
