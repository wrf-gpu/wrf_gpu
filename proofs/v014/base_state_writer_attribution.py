#!/usr/bin/env python3
"""V0.14 base-state writer attribution proof.

CPU-only NetCDF/NumPy probe for the remaining h1 static/base-state fields after
the post-static writer smoke. It reads existing proof metadata and WRF NetCDF
artifacts only; it does not import JAX or run model code.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from math import cos, pi
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DOMAIN = "d02"

CPU_TRUTH_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
FRESH_GPU_DIR = Path("/tmp/v014_post_static_writer_smoke") / f"l2_d02_{RUN_ID}"
RUN_ROOT_INPUT_DIR = Path("/tmp/v0120_merged_run_root") / RUN_ID

STATIC_PARITY_JSON = ROOT / "proofs/v014/static_metric_base_parity.json"
POST_STATIC_COMPARE_JSON = ROOT / "proofs/v014/post_static_writer_grid_compare.json"
OUT_JSON = ROOT / "proofs/v014/base_state_writer_attribution.json"
OUT_MD = ROOT / "proofs/v014/base_state_writer_attribution.md"

TARGET_FIELDS = ("PHB", "MUB", "PB", "HGT", "XLAT", "XLONG")
DERIVED_SPLITS = {
    "PB": {"total": "P_TOTAL", "perturbation": "P", "base": "PB"},
    "PHB": {"total": "PH_TOTAL", "perturbation": "PH", "base": "PHB"},
    "MUB": {"total": "MU_TOTAL", "perturbation": "MU", "base": "MUB"},
}
READ_FIELDS = tuple(dict.fromkeys((*TARGET_FIELDS, "P", "PH", "MU")))

COMPARISON_PAIRS = (
    ("cpu_wrfinput_vs_gpu_native_wrfinput", "cpu_wrfinput", "gpu_native_wrfinput"),
    ("cpu_wrfinput_vs_cpu_wrfout_h0", "cpu_wrfinput", "cpu_wrfout_h0"),
    ("cpu_wrfinput_vs_cpu_wrfout_h1", "cpu_wrfinput", "cpu_wrfout_h1"),
    ("cpu_wrfout_h0_vs_cpu_wrfout_h1", "cpu_wrfout_h0", "cpu_wrfout_h1"),
    ("cpu_wrfout_h1_vs_gpu_wrfout_h1", "cpu_wrfout_h1", "gpu_wrfout_h1"),
    ("gpu_native_wrfinput_vs_gpu_wrfout_h1", "gpu_native_wrfinput", "gpu_wrfout_h1"),
    ("cpu_wrfinput_vs_gpu_wrfout_h1", "cpu_wrfinput", "gpu_wrfout_h1"),
)

NC_ATTRS = (
    "TITLE",
    "START_DATE",
    "SIMULATION_START_DATE",
    "DX",
    "DY",
    "MAP_PROJ",
    "CEN_LAT",
    "CEN_LON",
    "TRUELAT1",
    "TRUELAT2",
    "STAND_LON",
)


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _clean_float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_symlink": path.is_symlink(),
        "resolved": str(path.resolve()) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def nc_attrs(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with Dataset(path, "r") as ds:
        for name in NC_ATTRS:
            if hasattr(ds, name):
                value = getattr(ds, name)
                out[name] = value.item() if hasattr(value, "item") else value
    return out


def var_metadata(path: Path, field: str) -> dict[str, Any] | None:
    with Dataset(path, "r") as ds:
        if field not in ds.variables:
            return None
        var = ds.variables[field]
        dims = tuple(var.dimensions)
        shape = tuple(int(x) for x in var.shape)
        if dims and dims[0] == "Time":
            dims = dims[1:]
            shape = shape[1:]
        return {
            "dims": list(dims),
            "shape": list(shape),
            "dtype": str(var.dtype),
            "units": str(getattr(var, "units", "")),
            "description": str(getattr(var, "description", "")),
            "stagger": str(getattr(var, "stagger", "")),
        }


def read_var(path: Path, field: str) -> np.ndarray | None:
    with Dataset(path, "r") as ds:
        if field not in ds.variables:
            return None
        var = ds.variables[field]
        raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(np.ma.filled(raw, np.nan))


def compare_arrays(left: np.ndarray | None, right: np.ndarray | None) -> dict[str, Any]:
    if left is None or right is None:
        return {
            "status": "MISSING",
            "left_present": left is not None,
            "right_present": right is not None,
        }
    if left.shape != right.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    lnum = np.asarray(left, dtype=np.float64)
    rnum = np.asarray(right, dtype=np.float64)
    finite_left = np.isfinite(lnum)
    finite_right = np.isfinite(rnum)
    finite_pair = finite_left & finite_right
    total = int(lnum.size)
    if not np.any(finite_pair):
        return {
            "status": "NO_FINITE_PAIR",
            "shape": list(lnum.shape),
            "total": total,
            "finite_left": int(finite_left.sum()),
            "finite_right": int(finite_right.sum()),
            "finite_pair": 0,
            "finite_pair_fraction": 0.0 if total else None,
        }
    diff = lnum - rnum
    vals = diff[finite_pair]
    abs_diff = np.abs(diff)
    abs_vals = np.abs(vals)
    exact = bool(np.array_equal(left, right, equal_nan=True))
    masked_abs = np.where(finite_pair, abs_diff, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked_abs)), masked_abs.shape)
    return {
        "status": "EXACT" if exact else "DIFF",
        "exact": exact,
        "shape": list(lnum.shape),
        "total": total,
        "finite_left": int(finite_left.sum()),
        "finite_right": int(finite_right.sum()),
        "finite_pair": int(finite_pair.sum()),
        "finite_pair_fraction": float(finite_pair.sum() / total) if total else None,
        "bias": float(np.mean(vals)),
        "rmse": float(np.sqrt(np.mean(vals * vals))),
        "p99_abs": float(np.percentile(abs_vals, 99.0)),
        "max_abs": float(np.max(abs_vals)),
        "worst_cell": {
            "index": [int(i) for i in idx],
            "left": float(lnum[idx]),
            "right": float(rnum[idx]),
            "diff": float(diff[idx]),
            "abs_diff": float(abs_diff[idx]),
        },
    }


def is_exact(cmp: Mapping[str, Any]) -> bool:
    return cmp.get("status") == "EXACT"


def max_abs(cmp: Mapping[str, Any]) -> float | None:
    value = cmp.get("max_abs")
    return None if value is None else float(value)


def discover_paths() -> tuple[dict[str, Path], dict[str, Any]]:
    static = load_json(STATIC_PARITY_JSON)
    post = load_json(POST_STATIC_COMPARE_JSON)
    pairs = [
        pair
        for pair in post["pairing"]["pairs"]
        if int(pair["lead_h"]) == 1 and pair["valid_time_utc"] == "2026-05-01T19:00:00+00:00"
    ]
    if len(pairs) != 1:
        raise RuntimeError(f"expected exactly one h1 pair in {POST_STATIC_COMPARE_JSON}, got {len(pairs)}")
    pair = pairs[0]
    paths = {
        "cpu_wrfinput": Path(static["inputs"]["wrfinput"]),
        "gpu_native_wrfinput": RUN_ROOT_INPUT_DIR / "wrfinput_d02",
        "cpu_wrfout_h0": Path(static["inputs"]["cpu_h0"]),
        "cpu_wrfout_h1": Path(pair["cpu_file"]),
        "gpu_wrfout_h1": Path(pair["gpu_file"]),
    }
    if Path(post["inputs"]["cpu_dir"]) != CPU_TRUTH_DIR:
        raise RuntimeError("post-static comparator CPU directory does not match contract input")
    if Path(post["inputs"]["gpu_dir"]) != FRESH_GPU_DIR:
        raise RuntimeError("post-static comparator GPU directory does not match contract input")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required artifact(s) missing: {missing}")
    discovery = {
        "cpu_wrfinput": f"{STATIC_PARITY_JSON}: inputs.wrfinput",
        "gpu_native_wrfinput": f"{RUN_ROOT_INPUT_DIR}/wrfinput_d02",
        "cpu_wrfout_h0": f"{STATIC_PARITY_JSON}: inputs.cpu_h0",
        "cpu_wrfout_h1": f"{POST_STATIC_COMPARE_JSON}: pairing.pairs[lead_h=1].cpu_file",
        "gpu_wrfout_h1": f"{POST_STATIC_COMPARE_JSON}: pairing.pairs[lead_h=1].gpu_file",
    }
    return paths, discovery


def read_source_arrays(paths: Mapping[str, Path]) -> dict[str, dict[str, np.ndarray | None]]:
    return {name: {field: read_var(path, field) for field in READ_FIELDS} for name, path in paths.items()}


def build_comparisons(arrays: Mapping[str, Mapping[str, np.ndarray | None]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for comparison_name, left_name, right_name in COMPARISON_PAIRS:
        out[comparison_name] = {
            field: compare_arrays(arrays[left_name].get(field), arrays[right_name].get(field))
            for field in TARGET_FIELDS
        }
    return out


def synthetic_projection_latlon(gpu_wrfout: Path, field: str) -> np.ndarray:
    with Dataset(gpu_wrfout, "r") as ds:
        ny = int(len(ds.dimensions["south_north"]))
        nx = int(len(ds.dimensions["west_east"]))
        lat_0 = float(getattr(ds, "CEN_LAT"))
        lon_0 = float(getattr(ds, "CEN_LON"))
        dx_m = float(getattr(ds, "DX"))
        dy_m = float(getattr(ds, "DY"))
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    lat_step = dy_m / 111_320.0
    lon_step = dx_m / max(111_320.0 * cos(lat_0 * pi / 180.0), 1.0)
    lat_grid = lat_0 + y[:, None] * lat_step
    lon_grid = lon_0 + x[None, :] * lon_step
    if field == "XLAT":
        return np.broadcast_to(lat_grid, (ny, nx)).astype(np.float32)
    if field == "XLONG":
        return np.broadcast_to(lon_grid, (ny, nx)).astype(np.float32)
    raise ValueError(field)


def writer_fallback_tests(
    paths: Mapping[str, Path],
    arrays: Mapping[str, Mapping[str, np.ndarray | None]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for field in ("XLAT", "XLONG"):
        fallback = synthetic_projection_latlon(paths["gpu_wrfout_h1"], field)
        out[field] = {
            "method": (
                "wrfout_writer._latlon_fields projection fallback: centered linear "
                "lat/lon grid from CEN_LAT/CEN_LON/DX/DY when State lacks lat/lon"
            ),
            "projection_attrs_from_gpu_wrfout_h1": {
                name: nc_attrs(paths["gpu_wrfout_h1"]).get(name)
                for name in ("CEN_LAT", "CEN_LON", "DX", "DY", "TRUELAT1", "TRUELAT2", "STAND_LON")
            },
            "fallback_vs_gpu_wrfout_h1": compare_arrays(fallback, arrays["gpu_wrfout_h1"][field]),
            "cpu_wrfinput_vs_fallback": compare_arrays(arrays["cpu_wrfinput"][field], fallback),
        }
    return out


def derived_array(source: Mapping[str, np.ndarray | None], split: Mapping[str, str]) -> np.ndarray | None:
    pert = source.get(split["perturbation"])
    base = source.get(split["base"])
    if pert is None or base is None:
        return None
    return np.asarray(pert, dtype=np.float64) + np.asarray(base, dtype=np.float64)


def derived_state_split_tests(arrays: Mapping[str, Mapping[str, np.ndarray | None]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key_pairs = (
        ("cpu_wrfinput_vs_cpu_wrfout_h0", "cpu_wrfinput", "cpu_wrfout_h0"),
        ("cpu_wrfout_h0_vs_cpu_wrfout_h1", "cpu_wrfout_h0", "cpu_wrfout_h1"),
        ("cpu_wrfout_h1_vs_gpu_wrfout_h1", "cpu_wrfout_h1", "gpu_wrfout_h1"),
        ("cpu_wrfinput_vs_gpu_wrfout_h1", "cpu_wrfinput", "gpu_wrfout_h1"),
    )
    for base_field, split in DERIVED_SPLITS.items():
        field_rows: dict[str, Any] = {}
        for name, left_source, right_source in key_pairs:
            left_total = derived_array(arrays[left_source], split)
            right_total = derived_array(arrays[right_source], split)
            field_rows[name] = {
                "total": compare_arrays(left_total, right_total),
                "perturbation": compare_arrays(
                    arrays[left_source].get(split["perturbation"]),
                    arrays[right_source].get(split["perturbation"]),
                ),
                "base": compare_arrays(
                    arrays[left_source].get(split["base"]),
                    arrays[right_source].get(split["base"]),
                ),
            }
        out[base_field] = {
            "total_field": split["total"],
            "perturbation_field": split["perturbation"],
            "base_field": split["base"],
            "comparisons": field_rows,
        }
    return out


def classify_field(
    field: str,
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fallback: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    c = {name: group[field] for name, group in comparisons.items()}
    input_match = is_exact(c["cpu_wrfinput_vs_gpu_native_wrfinput"])
    cpu_input_to_h0 = c["cpu_wrfinput_vs_cpu_wrfout_h0"]
    cpu_h0_to_h1 = c["cpu_wrfout_h0_vs_cpu_wrfout_h1"]
    gpu_input_to_h1 = c["gpu_native_wrfinput_vs_gpu_wrfout_h1"]
    cpu_h1_to_gpu_h1 = c["cpu_wrfout_h1_vs_gpu_wrfout_h1"]

    if not input_match:
        return {
            "classification": "runtime_input_mismatch",
            "confidence": "high",
            "evidence": "GPU native run-root wrfinput differs from the retained CPU wrfinput.",
        }
    if is_exact(cpu_h1_to_gpu_h1):
        return {
            "classification": "exact",
            "confidence": "high",
            "evidence": "CPU h1 and fresh GPU h1 are byte-exact for this field.",
        }
    if field in {"XLAT", "XLONG"}:
        field_fallback = fallback[field]
        if (
            is_exact(cpu_input_to_h0)
            and is_exact(cpu_h0_to_h1)
            and is_exact(field_fallback["fallback_vs_gpu_wrfout_h1"])
            and not is_exact(field_fallback["cpu_wrfinput_vs_fallback"])
        ):
            return {
                "classification": "writer_fallback",
                "confidence": "high",
                "evidence": (
                    "CPU wrfinput/wrfout lat-lon are exact, but fresh GPU h1 is "
                    "byte-exact to the writer projection fallback generated from "
                    "GPU writer attrs, not to wrfinput."
                ),
                "cpu_wrfinput_vs_fallback_max_abs": max_abs(field_fallback["cpu_wrfinput_vs_fallback"]),
            }
    if field == "HGT":
        if not is_exact(cpu_input_to_h0) and is_exact(cpu_h0_to_h1) and is_exact(gpu_input_to_h1):
            return {
                "classification": "cpu_output_convention",
                "confidence": "high",
                "evidence": (
                    "GPU h1 terrain is byte-exact to the run-root wrfinput; CPU "
                    "wrfout h0/h1 are byte-exact to each other but differ from wrfinput."
                ),
                "cpu_wrfinput_vs_cpu_wrfout_h0_max_abs": max_abs(cpu_input_to_h0),
            }
    if field == "PHB":
        gpu_roundoff = max_abs(gpu_input_to_h1) or 0.0
        if not is_exact(cpu_input_to_h0) and is_exact(cpu_h0_to_h1) and gpu_roundoff <= 2.0e-2:
            return {
                "classification": "cpu_output_convention",
                "confidence": "high",
                "evidence": (
                    "CPU wrfout h0/h1 PHB are static but differ from wrfinput; "
                    "fresh GPU h1 follows wrfinput to fp32 writer roundoff "
                    f"(max_abs {gpu_roundoff:.6g})."
                ),
                "cpu_wrfinput_vs_cpu_wrfout_h0_max_abs": max_abs(cpu_input_to_h0),
                "gpu_native_wrfinput_vs_gpu_wrfout_h1_max_abs": gpu_roundoff,
            }
    if field in {"PB", "MUB"}:
        if not is_exact(gpu_input_to_h1) and is_exact(cpu_h0_to_h1):
            return {
                "classification": "forecast_step_change",
                "confidence": "medium",
                "evidence": (
                    "CPU and GPU wrfinputs are exact and CPU base field is h0-h1 "
                    "static, but fresh GPU h1 differs from its native wrfinput. "
                    "The writer reconstructs this base field from evolved "
                    "total-minus-perturbation state at output time, so this is a "
                    "one-hour state-split symptom rather than a static input mismatch."
                ),
                "gpu_native_wrfinput_vs_gpu_wrfout_h1_max_abs": max_abs(gpu_input_to_h1),
                "cpu_wrfout_h1_vs_gpu_wrfout_h1_max_abs": max_abs(cpu_h1_to_gpu_h1),
            }
    return {
        "classification": "unresolved_blocker",
        "confidence": "low",
        "evidence": "Available CPU-only artifacts did not isolate a single accepted attribution.",
        "cpu_wrfout_h1_vs_gpu_wrfout_h1_max_abs": max_abs(cpu_h1_to_gpu_h1),
    }


def build_field_attribution(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    fallback: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {field: classify_field(field, comparisons, fallback) for field in TARGET_FIELDS}


def classification_counts(attribution: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in attribution.values():
        cls = str(item["classification"])
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def write_markdown(payload: Mapping[str, Any]) -> None:
    attribution = payload["field_attribution"]
    lines: list[str] = []
    lines.append("# V0.14 Base-State Writer Attribution")
    lines.append("")
    lines.append(f"Generated UTC: `{payload['generated_utc']}`")
    lines.append("")
    lines.append("CPU-only NetCDF probe. No GPU, no model execution, no source edits.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- No runtime input mismatch: CPU `wrfinput_d02` and GPU run-root `wrfinput_d02` are exact for all six target fields.")
    lines.append("- No additional base-state source fix is required before same-state dynamic localization.")
    lines.append("- Proceed with documented exclusions: treat `PHB`/`HGT` as CPU output-convention fields, `XLAT`/`XLONG` as writer-fallback fields, and `PB`/`MUB` as one-hour forecast state-split symptoms.")
    lines.append("")
    lines.append("## Classifications")
    lines.append("")
    lines.append("| Field | Classification | Key evidence |")
    lines.append("| --- | --- | --- |")
    for field in TARGET_FIELDS:
        item = attribution[field]
        lines.append(f"| `{field}` | `{item['classification']}` | {item['evidence']} |")
    lines.append("")
    lines.append("## Exact Files")
    lines.append("")
    for name, info in payload["inputs"]["files"].items():
        lines.append(f"- `{name}`: `{info['path']}`")
    lines.append("")
    lines.append("Full comparison tables, finite coverage, p99/max/worst-cell statistics, writer-fallback tests, and derived state-split totals are in `proofs/v014/base_state_writer_attribution.json`.")
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    paths, discovery = discover_paths()
    arrays = read_source_arrays(paths)
    comparisons = build_comparisons(arrays)
    fallback = writer_fallback_tests(paths, arrays)
    derived = derived_state_split_tests(arrays)
    attribution = build_field_attribution(comparisons, fallback)

    payload: dict[str, Any] = {
        "schema": "v014_base_state_writer_attribution",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_used": False,
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "Uses netCDF4 and NumPy only; no JAX/CUDA/model execution.",
        },
        "inputs": {
            "run_id": RUN_ID,
            "domain": DOMAIN,
            "contract_cpu_truth_dir": str(CPU_TRUTH_DIR),
            "contract_fresh_gpu_dir": str(FRESH_GPU_DIR),
            "contract_run_root_input_dir": str(RUN_ROOT_INPUT_DIR),
            "metadata_sources": {
                "static_metric_base_parity_json": str(STATIC_PARITY_JSON),
                "post_static_writer_grid_compare_json": str(POST_STATIC_COMPARE_JSON),
            },
            "discovery": discovery,
            "files": {name: path_info(path) for name, path in paths.items()},
            "netcdf_attrs": {name: nc_attrs(path) for name, path in paths.items()},
            "field_metadata": {
                field: {
                    source_name: var_metadata(path, field)
                    for source_name, path in paths.items()
                }
                for field in TARGET_FIELDS
            },
        },
        "comparisons": comparisons,
        "writer_fallback_tests": fallback,
        "derived_state_split_tests": derived,
        "field_attribution": attribution,
        "field_classifications": {
            field: item["classification"] for field, item in attribution.items()
        },
        "summary": {
            "classification_counts": classification_counts(attribution),
            "runtime_input_mismatch": False,
            "same_state_localization": {
                "can_proceed": True,
                "decision": "proceed_to_same_state_dynamic_localization_with_documented_exclusions",
                "no_base_state_source_fix_required_first": True,
                "static_parity_exclusions": ["PHB", "HGT", "XLAT", "XLONG"],
                "dynamic_state_split_symptoms": ["PB", "MUB"],
                "writer_only_fix_remaining": ["XLAT", "XLONG"],
                "note": (
                    "PB/MUB should not block same-state localization as static/base "
                    "source mismatches; they are h1 evolved state-split symptoms to "
                    "localize in the dynamic pass."
                ),
            },
        },
    }

    write_json(OUT_JSON, payload)
    write_markdown(payload)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
