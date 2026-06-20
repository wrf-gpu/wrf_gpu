#!/usr/bin/env python3
"""Build the M7 wrfout I/O compatibility inventory and matrix.

This script is intentionally CPU-only.  It reads NetCDF metadata from one Gen2
WRF reference file and statically parses ``write_wrfout_gpu`` without importing
the coupling module, because importing that module initializes JAX.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO_ROOT / ".agent/sprints/2026-05-27-m7-wrfout-io-compat"
DRIVER_PATH = REPO_ROOT / "src/gpuwrf/coupling/driver.py"
D02_REPLAY_PATH = REPO_ROOT / "src/gpuwrf/integration/d02_replay.py"
GEN2_ACCESSOR_PATH = REPO_ROOT / "src/gpuwrf/io/gen2_accessor.py"
BOUNDARY_REPLAY_PATH = REPO_ROOT / "src/gpuwrf/io/boundary_replay.py"
DEFAULT_REFERENCE = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260525_18z_l3_24h_20260526T221207Z/"
    "wrfout_d02_2026-05-25_18:00:00"
)


DOWNSTREAM_CONSUMED: dict[str, str] = {
    "Times": "WRF time coordinate used by raw wrfout and thin-NetCDF consumers.",
    "XTIME": "WRF elapsed-time coordinate used by thin gridded extraction.",
    "XLAT": "Gen2 geolocation, station extraction, 3dweather, and ML gridded products.",
    "XLONG": "Gen2 geolocation, station extraction, 3dweather, and ML gridded products.",
    "HGT": "Terrain/georef audits and pressure-derived product proxies.",
    "LANDMASK": "Surface/geography and cloud/terrain diagnostics.",
    "LU_INDEX": "Surface/geography diagnostics.",
    "U10": "Binding M7 surface wind and station verification field.",
    "V10": "Binding M7 surface wind and station verification field.",
    "T2": "Binding M7 2m temperature and station verification field.",
    "Q2": "Surface humidity and validation loader field.",
    "PSFC": "Surface pressure, QA, and MSLP proxy input.",
    "RAINC": "Convective accumulated precipitation for precip verification.",
    "RAINNC": "Grid-scale accumulated precipitation for precip verification.",
    "RAINSH": "Shallow-convective accumulated precipitation for precip verification.",
    "SWDOWN": "Solar/cloud feature and point-shadow products.",
    "GLW": "Radiation/cloud feature and point-shadow products.",
    "PBLH": "PBL diagnostics and point-shadow products.",
    "UST": "Surface-layer diagnostics and point-shadow products.",
    "HFX": "Surface heat-flux diagnostics.",
    "LH": "Latent heat-flux diagnostics.",
    "TSK": "Skin-temperature diagnostics and surface-state QA.",
    "CLDFRA": "Cloud products; thin extractor derives TCC/CLDLOW/CLDMID/CLDHIGH.",
    "QCLOUD": "Cloud-cap diagnostics and cloud feature products.",
    "QICE": "Cloud-cap diagnostics and cloud feature products.",
    "QRAIN": "Cloud/precip diagnostics.",
    "TCC": "Thin-NetCDF/3dweather rich cloud field derived from CLDFRA.",
    "CLDLOW": "Thin-NetCDF/3dweather rich low-cloud field derived from CLDFRA.",
    "CLDMID": "Thin-NetCDF/3dweather rich mid-cloud field derived from CLDFRA.",
    "CLDHIGH": "Thin-NetCDF/3dweather rich high-cloud field derived from CLDFRA.",
}


GPU_FIELD_METADATA: dict[str, dict[str, Any]] = {
    "U": {
        "dimensions": ["bottom_top", "south_north", "west_east_stag"],
        "dtype": "float32",
        "units": "m s-1",
        "source_fields": ["state.u"],
        "semantic_agreement": True,
        "note": "WRF-shaped staggered U array; writer omits Time dimension and NetCDF metadata.",
    },
    "V": {
        "dimensions": ["bottom_top", "south_north_stag", "west_east"],
        "dtype": "float32",
        "units": "m s-1",
        "source_fields": ["state.v"],
        "semantic_agreement": True,
        "note": "WRF-shaped staggered V array; writer omits Time dimension and NetCDF metadata.",
    },
    "W": {
        "dimensions": ["bottom_top_stag", "south_north", "west_east"],
        "dtype": "float32",
        "units": "m s-1",
        "source_fields": ["state.w"],
        "semantic_agreement": True,
        "note": "WRF-shaped staggered W array; writer omits Time dimension and NetCDF metadata.",
    },
    "T": {
        "dimensions": ["bottom_top", "south_north", "west_east"],
        "dtype": "float32",
        "units": "K",
        "source_fields": ["state.theta"],
        "semantic_agreement": True,
        "note": "Writes state.theta - 300 K, matching WRF perturbation-theta convention; Time dimension omitted.",
    },
    "QVAPOR": {
        "dimensions": ["bottom_top", "south_north", "west_east"],
        "dtype": "float32",
        "units": "kg kg-1",
        "source_fields": ["state.qv"],
        "semantic_agreement": True,
        "note": "WRF-shaped water-vapor field; writer omits Time dimension and NetCDF metadata.",
    },
    "P": {
        "dimensions": ["bottom_top", "south_north", "west_east"],
        "dtype": "float32",
        "units": "Pa",
        "source_fields": ["state.p"],
        "semantic_agreement": False,
        "note": "Current State.p is aligned with p_total; WRF P is perturbation pressure and PB is omitted.",
    },
    "PH": {
        "dimensions": ["bottom_top_stag", "south_north", "west_east"],
        "dtype": "float32",
        "units": "m2 s-2",
        "source_fields": ["state.ph"],
        "semantic_agreement": False,
        "note": "Current State.ph is aligned with ph_total; WRF PH is perturbation geopotential and PHB is omitted.",
    },
    "MU": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "Pa",
        "source_fields": ["state.mu"],
        "semantic_agreement": False,
        "note": "Current State.mu is aligned with mu_total; WRF MU is perturbation dry-column mass and MUB is omitted.",
    },
    "U10": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "m s-1",
        "source_fields": ["surface.u10"],
        "semantic_agreement": True,
        "note": "Surface diagnostic present; Time dimension and NetCDF attributes omitted.",
    },
    "V10": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "m s-1",
        "source_fields": ["surface.v10"],
        "semantic_agreement": True,
        "note": "Surface diagnostic present; Time dimension and NetCDF attributes omitted.",
    },
    "T2": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "K",
        "source_fields": ["surface.t2"],
        "semantic_agreement": True,
        "note": "Surface diagnostic present; Time dimension and NetCDF attributes omitted.",
    },
    "Q2": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "kg kg-1",
        "source_fields": ["surface.q2"],
        "semantic_agreement": True,
        "note": "Surface diagnostic present; Time dimension and NetCDF attributes omitted.",
    },
    "UST": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "m s-1",
        "source_fields": ["surface.fluxes.ustar"],
        "semantic_agreement": True,
        "note": "Surface friction velocity present; Time dimension and NetCDF attributes omitted.",
    },
    "HFX_KIN": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "K m s-1",
        "source_fields": ["surface.fluxes.theta_flux"],
        "semantic_agreement": False,
        "note": "GPU-only kinematic heat-flux diagnostic; WRF HFX is W m-2.",
    },
    "QFX_KIN": {
        "dimensions": ["south_north", "west_east"],
        "dtype": "float32",
        "units": "kg kg-1 m s-1",
        "source_fields": ["surface.fluxes.qv_flux"],
        "semantic_agreement": False,
        "note": "GPU-only kinematic moisture-flux diagnostic; WRF QFX uses mass flux units.",
    },
    "lead_hours": {
        "dimensions": [],
        "dtype": "float32",
        "units": "h",
        "source_fields": ["lead_hours"],
        "semantic_agreement": False,
        "note": "GPU-only scalar metadata replacing WRF Times/XTIME.",
    },
    "mass_shape": {
        "dimensions": ["shape_component"],
        "dtype": "int32",
        "units": "1",
        "source_fields": ["grid.nz", "grid.ny", "grid.nx"],
        "semantic_agreement": False,
        "note": "GPU-only shape metadata.",
    },
    "wrf_staggered_extent": {
        "dimensions": ["shape_component"],
        "dtype": "int32",
        "units": "1",
        "source_fields": ["grid.nz + 1", "grid.ny + 1", "grid.nx + 1"],
        "semantic_agreement": False,
        "note": "GPU-only shape metadata.",
    },
    "run_start_label": {
        "dimensions": [],
        "dtype": "str",
        "units": "",
        "source_fields": ["run_start_label"],
        "semantic_agreement": False,
        "note": "GPU-only string metadata replacing WRF Times/XTIME.",
    },
    "container_note": {
        "dimensions": [],
        "dtype": "str",
        "units": "",
        "source_fields": ["literal"],
        "semantic_agreement": False,
        "note": "GPU-only string metadata documenting the NPZ container.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return [item.decode("utf-8", errors="replace") for item in value.reshape(-1).tolist()]
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def netcdf_attrs(obj: Any) -> dict[str, Any]:
    return {name: jsonable(obj.getncattr(name)) for name in obj.ncattrs()}


def inventory_cpu_wrfout(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with Dataset(source, "r") as dataset:
        dimensions = {
            name: {"size": int(len(dim)), "isunlimited": bool(dim.isunlimited())}
            for name, dim in dataset.dimensions.items()
        }
        variables: dict[str, Any] = {}
        for name, variable in dataset.variables.items():
            attrs = netcdf_attrs(variable)
            variables[name] = {
                "dimensions": list(variable.dimensions),
                "shape": [int(size) for size in variable.shape],
                "dtype": str(np.dtype(variable.dtype)),
                "attrs": attrs,
                "units": str(attrs.get("units", "")),
                "description": str(attrs.get("description", "")),
                "memory_order": str(attrs.get("MemoryOrder", "")),
                "stagger": str(attrs.get("stagger", "")),
            }
        global_attrs = netcdf_attrs(dataset)
        return {
            "schema": "m7_cpu_wrfout_reference_inventory_v1",
            "generated_utc": utc_now(),
            "reference_file": str(source),
            "reference_selection_reason": (
                "single representative recent successful Gen2 3km l3 d02 wrfout; "
                "selected explicitly to avoid bulk backfill iteration"
            ),
            "file_format": dataset.file_format,
            "dimensions": dimensions,
            "global_attrs": global_attrs,
            "variable_count": len(variables),
            "variables": variables,
        }


def find_write_wrfout_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "write_wrfout_gpu":
            return node
    raise ValueError("write_wrfout_gpu function not found")


def extract_payload_keys(driver_path: Path = DRIVER_PATH) -> tuple[list[str], dict[str, str], tuple[int, int]]:
    source = driver_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(driver_path))
    function = find_write_wrfout_function(tree)
    keys: list[str] = []
    expressions: dict[str, str] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values, strict=True):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise ValueError("write_wrfout_gpu payload has a non-literal key")
            key = key_node.value
            keys.append(key)
            expressions[key] = ast.unparse(value_node)
        break
    if not keys:
        raise ValueError("write_wrfout_gpu payload dict not found")
    return keys, expressions, (int(function.lineno), int(function.end_lineno or function.lineno))


def inventory_gpu_writer(driver_path: Path = DRIVER_PATH) -> dict[str, Any]:
    keys, expressions, line_span = extract_payload_keys(driver_path)
    variables: dict[str, Any] = {}
    for key in keys:
        base = dict(GPU_FIELD_METADATA.get(key, {}))
        base.setdefault("dimensions", [])
        base.setdefault("dtype", "unknown")
        base.setdefault("units", "")
        base.setdefault("source_fields", [])
        base.setdefault("semantic_agreement", False)
        base.setdefault("note", "Payload key lacks hand-authored metadata mapping.")
        base["source_expression"] = expressions[key]
        base["container_format"] = "npz"
        base["netcdf_dimensions_present"] = False
        base["time_dimension_present"] = False
        variables[key] = base
    missing_metadata = [key for key in keys if key not in GPU_FIELD_METADATA]
    return {
        "schema": "m7_gpu_wrfout_writer_inventory_v1",
        "generated_utc": utc_now(),
        "source_file": str(driver_path.resolve()),
        "function": "write_wrfout_gpu",
        "function_lines": {"start": line_span[0], "end": line_span[1]},
        "no_gpu_runtime": True,
        "parse_method": "ast_static_payload_dict_parse",
        "container_format": "npz",
        "container_writer": "numpy.savez",
        "writer_summary": "compact WRF-shaped NumPy proof container, not NetCDF wrfout",
        "payload_key_count": len(keys),
        "payload_keys": keys,
        "missing_hand_metadata": missing_metadata,
        "variables": variables,
    }


def normalize_units(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def dtype_matches(cpu_dtype: str, gpu_dtype: str) -> bool:
    aliases = {
        "float32": {"float32", "f4", "<f4"},
        "int32": {"int32", "i4", "<i4"},
        "|S1": {"|S1", "S1"},
    }
    if cpu_dtype == gpu_dtype:
        return True
    return gpu_dtype in aliases.get(cpu_dtype, set()) or cpu_dtype in aliases.get(gpu_dtype, set())


def dim_status(cpu_dims: list[str], gpu_dims: list[str]) -> tuple[bool, str]:
    if cpu_dims == gpu_dims:
        return True, "YES"
    if cpu_dims and cpu_dims[0] == "Time" and cpu_dims[1:] == gpu_dims:
        return False, "NO: GPU omits singleton Time dimension"
    return False, "NO"


def units_status(cpu_units: str, gpu_units: str) -> tuple[bool, str]:
    cpu_norm = normalize_units(cpu_units)
    gpu_norm = normalize_units(gpu_units)
    if cpu_norm == gpu_norm:
        return True, "YES"
    if cpu_norm == "" and gpu_norm == "":
        return True, "YES"
    return False, "NO"


def build_compat_rows(cpu_inventory: dict[str, Any], gpu_inventory: dict[str, Any]) -> list[dict[str, Any]]:
    cpu_vars: dict[str, Any] = cpu_inventory["variables"]
    gpu_vars: dict[str, Any] = gpu_inventory["variables"]
    names = sorted(set(cpu_vars) | set(gpu_vars))
    rows: list[dict[str, Any]] = []
    for name in names:
        cpu = cpu_vars.get(name)
        gpu = gpu_vars.get(name)
        if cpu and gpu:
            dim_ok, dim_text = dim_status(cpu["dimensions"], gpu["dimensions"])
            dtype_ok = dtype_matches(cpu["dtype"], gpu["dtype"])
            units_ok, units_text = units_status(cpu.get("units", ""), gpu.get("units", ""))
            semantic_ok = bool(gpu.get("semantic_agreement"))
            if dim_ok and dtype_ok and units_ok and semantic_ok and gpu.get("container_format") == "netcdf":
                classification = "MATCH"
            else:
                classification = "DEVIATION_DOCUMENTED"
            notes = gpu.get("note", "")
        elif cpu and not gpu:
            dim_text = dtype_text = units_text = "N/A"
            classification = "MISSING_GPU"
            notes = DOWNSTREAM_CONSUMED.get(name, "CPU WRF variable not emitted by compact GPU writer.")
            dtype_ok = units_ok = False
        else:
            dim_text = dtype_text = units_text = "N/A"
            classification = "EXTRA_GPU"
            notes = gpu.get("note", "GPU-only payload key.") if gpu else ""
            dtype_ok = units_ok = False
        if cpu and gpu:
            dtype_text = "YES" if dtype_ok else f"NO: CPU {cpu['dtype']} vs GPU {gpu['dtype']}"
            if units_text == "NO":
                units_text = f"NO: CPU {cpu.get('units', '')!r} vs GPU {gpu.get('units', '')!r}"
        row = {
            "variable": name,
            "cpu_has": bool(cpu),
            "gpu_writes": bool(gpu),
            "cpu_dimensions": cpu.get("dimensions", []) if cpu else [],
            "gpu_dimensions": gpu.get("dimensions", []) if gpu else [],
            "dim_agreement": dim_text,
            "cpu_dtype": cpu.get("dtype", "") if cpu else "",
            "gpu_dtype": gpu.get("dtype", "") if gpu else "",
            "dtype_agreement": dtype_text,
            "cpu_units": cpu.get("units", "") if cpu else "",
            "gpu_units": gpu.get("units", "") if gpu else "",
            "units_agreement": units_text,
            "classification": classification,
            "downstream_consumed": name in DOWNSTREAM_CONSUMED,
            "downstream_note": DOWNSTREAM_CONSUMED.get(name, ""),
            "notes": notes,
        }
        rows.append(row)
    return rows


def md_escape(value: Any) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("|", "\\|")
    return text


def bool_mark(value: bool) -> str:
    return "YES" if value else "NO"


def write_compat_matrix(path: Path, rows: list[dict[str, Any]], cpu_inventory: dict[str, Any], gpu_inventory: dict[str, Any]) -> None:
    counts = Counter(row["classification"] for row in rows)
    critical = [row for row in rows if row["downstream_consumed"]]
    critical_counts = Counter(row["classification"] for row in critical)
    lines = [
        "# M7 wrfout I/O Compatibility Matrix",
        "",
        f"Generated UTC: {utc_now()}",
        f"CPU reference: `{cpu_inventory['reference_file']}`",
        f"GPU writer: `{gpu_inventory['source_file']}:{gpu_inventory['function_lines']['start']}`",
        "",
        "## Summary",
        "",
        f"- CPU WRF variables: {cpu_inventory['variable_count']}",
        f"- GPU payload keys: {gpu_inventory['payload_key_count']}",
        f"- Classification counts: {dict(sorted(counts.items()))}",
        f"- Downstream-consumed classification counts: {dict(sorted(critical_counts.items()))}",
        "- Compatibility verdict: structural audit complete; GPU output is not yet a drop-in NetCDF wrfout.",
        "",
        "## Downstream-Critical Rows",
        "",
        "| Variable | Classification | CPU has | GPU writes | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in critical:
        notes = row["notes"] or row["downstream_note"]
        lines.append(
            f"| `{md_escape(row['variable'])}` | {row['classification']} | "
            f"{bool_mark(row['cpu_has'])} | {bool_mark(row['gpu_writes'])} | {md_escape(notes)} |"
        )
    lines.extend(
        [
            "",
            "## Full Matrix",
            "",
            "| Variable | CPU has | GPU writes | Dim agreement | Dtype agreement | Units agreement | Classification | Notes |",
            "|---|---:|---:|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        notes = row["notes"]
        if row["downstream_consumed"]:
            notes = f"{notes} Downstream: {row['downstream_note']}"
        lines.append(
            f"| `{md_escape(row['variable'])}` | {bool_mark(row['cpu_has'])} | "
            f"{bool_mark(row['gpu_writes'])} | {md_escape(row['dim_agreement'])} | "
            f"{md_escape(row['dtype_agreement'])} | {md_escape(row['units_agreement'])} | "
            f"{row['classification']} | {md_escape(notes)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wrap_names(names: list[str], width: int = 100) -> list[str]:
    joined = ", ".join(f"`{name}`" for name in names)
    return textwrap.wrap(joined, width=width, break_long_words=False, break_on_hyphens=False)


def write_explicit_deviations(path: Path, rows: list[dict[str, Any]], gpu_inventory: dict[str, Any]) -> None:
    missing = [row["variable"] for row in rows if row["classification"] == "MISSING_GPU"]
    extra = [row["variable"] for row in rows if row["classification"] == "EXTRA_GPU"]
    documented = [row for row in rows if row["classification"] == "DEVIATION_DOCUMENTED"]
    missing_critical = [name for name in missing if name in DOWNSTREAM_CONSUMED]
    lines = [
        "# Explicit wrfout Schema Deviations",
        "",
        f"Generated UTC: {utc_now()}",
        "",
        "This document enumerates intentional schema differences found by the M7 audit. "
        "The complete row-level matrix is `compat_matrix.md`.",
        "",
        "## Global Deviations",
        "",
        "| Difference | Why | Downstream consumers care? | Action required |",
        "|---|---|---|---|",
        "| GPU writer emits `.npz` via `numpy.savez`, not WRF NetCDF4 `wrfout_d02_YYYY-MM-DD_HH:MM:SS`. | Compact proof container from the current forecast driver. | Yes; Gen2 raw-wrfout readers expect NetCDF variables, dimensions, attrs, and WRF filenames. | Re-implement NetCDF wrfout writer or provide a tested adapter before claiming drop-in compatibility. |",
        "| GPU arrays omit the singleton `Time` dimension and named NetCDF dimensions. | Simplified direct array serialization. | Yes; xarray/netCDF4 consumers index `Time=0` and use named dimensions. | Add WRF-style dimensions or adapter. |",
        "| GPU output omits WRF global attributes and per-variable attrs such as `units`, `description`, `MemoryOrder`, and `stagger`. | Simplified proof artifact. | Yes for auditability and some metadata-driven consumers. | Populate WRF-compatible attrs in NetCDF output. |",
        "| GPU output stores `lead_hours` and `run_start_label` instead of `Times`/`XTIME`. | Simplified metadata. | Yes; Gen2 thin extraction and raw readers use WRF time coordinates. | Write `Times` and `XTIME`. |",
        "",
        "## GPU-Written WRF Variables With Documented Deviations",
        "",
        "| Variable | What is different | Why | Downstream consumers care? | Action required |",
        "|---|---|---|---|---|",
    ]
    for row in documented:
        reason = "compact proof writer"
        action = "document only for proof artifacts; re-implement NetCDF path for M7 drop-in use"
        if row["variable"] in {"P", "PH", "MU"}:
            action = "write WRF perturbation variable and companion base-state variable, or rename total-state output"
        cares = "yes" if row["downstream_consumed"] else "maybe"
        lines.append(
            f"| `{row['variable']}` | {md_escape(row['notes'])} | {reason} | {cares} | {action} |"
        )
    lines.extend(
        [
            "",
            "## GPU-Only Payload Keys",
            "",
            "| Key | Why | Downstream consumers care? | Action required |",
            "|---|---|---|---|",
        ]
    )
    for name in extra:
        meta = gpu_inventory["variables"].get(name, {})
        lines.append(
            f"| `{name}` | {md_escape(meta.get('note', 'GPU-only key'))} | No direct WRF consumer. | "
            "Keep only in proof artifacts; omit or map in NetCDF wrfout. |"
        )
    lines.extend(
        [
            "",
            "## Downstream-Critical CPU Variables Missing From GPU Output",
            "",
            "| Variable | Why consumers care | Action required |",
            "|---|---|---|",
        ]
    )
    for name in missing_critical:
        lines.append(f"| `{name}` | {md_escape(DOWNSTREAM_CONSUMED[name])} | Re-implement or explicitly replace in downstream adapter. |")
    lines.extend(
        [
            "",
            "## Complete CPU Variable Omission List",
            "",
            f"The compact GPU writer omits {len(missing)} CPU WRF variables from the selected reference file:",
            "",
        ]
    )
    lines.extend(wrap_names(missing))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def grep_lines(path: Path, needles: list[str]) -> list[str]:
    if not path.is_file():
        return [f"{path}: missing"]
    out: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if any(needle in line for needle in needles):
            out.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    return out


def write_endpoint_audit(path: Path, reference: Path) -> None:
    d02_refs = grep_lines(
        D02_REPLAY_PATH,
        ["build_replay_case", "wrfinput_file", "load_history_boundary_leaves", "history_files", "wrfout", "wrfbdy"],
    )
    boundary_refs = grep_lines(BOUNDARY_REPLAY_PATH, ["decode_wrfbdy", "wrfbdy_path_for_run"])
    accessor_refs = grep_lines(GEN2_ACCESSOR_PATH, ["wrfinput_file", "history_files", "wrfbdy", "wrfout", "restart_compatible"])
    run_dir = reference.resolve().parent
    lines = [
        "# M7 I/O Endpoint Footprint Audit",
        "",
        f"Generated UTC: {utc_now()}",
        f"Reference run directory: `{run_dir}`",
        "",
        "## Verdict",
        "",
        "The current GPU forecast path consumes Gen2 `wrfinput_d02` plus d02 hourly `wrfout` history for replay, "
        "and writes only compact `wrfout_gpu_d02_p###h.npz` proof containers. It does not currently produce a "
        "drop-in NetCDF `wrfout`, does not consume or produce native `wrfbdy` in `build_replay_case`, and has no "
        "observed `wrfrst` restart endpoint.",
        "",
        "## Endpoint Matrix",
        "",
        "| Endpoint | Current GPU role | Evidence | M7 daily-pipeline implication | Action |",
        "|---|---|---|---|---|",
        "| `wrfinput_d02` | Consumed for grid metrics, static fields, land state, and initial state context. Not produced. | `build_replay_case` calls `run.wrfinput_file(domain)` and land/metric loaders. | Acceptable as a read-only Gen2 IC source for M7 if documented; not a WRF-compatible producer. | Keep read path; document that GPU v0 does not emit wrfinput. |",
        "| `wrfbdy` | Not consumed by `build_replay_case`; validation code can decode `wrfbdy_d01` separately. | `boundary_replay.decode_wrfbdy`; `build_replay_case` uses `load_history_boundary_leaves` from d02 wrfout history. | Native WRF boundary compatibility is not satisfied by the forecast path. | Either implement wrfbdy consumption for forecast forcing or document d02-history replay as an explicit M7 deviation. |",
        "| `wrfout` | Consumed from d02 hourly history for initial/boundary replay; produced as `.npz`, not NetCDF. | `run_to_output_leads` calls `write_wrfout_gpu(...wrfout_gpu_d02_p###h.npz)`. | Downstream raw-wrfout consumers cannot read GPU output unchanged. | Implement NetCDF wrfout writer or adapter. |",
        "| `wrfrst` | No consumer or producer found in allowed audit surface. | No `wrfrst` references in `src/gpuwrf` endpoint code. | M7 restart-continuity gate remains structurally open. | Add WRF-compatible restart or explicit GPU checkpoint format plus continuity test and deviation document. |",
        "",
        "## Static Evidence Snippets",
        "",
        "### `d02_replay.py`",
        "",
    ]
    lines.extend(f"- `{md_escape(line)}`" for line in d02_refs[:40])
    lines.extend(["", "### `boundary_replay.py`", ""])
    lines.extend(f"- `{md_escape(line)}`" for line in boundary_refs[:40])
    lines.extend(["", "### `gen2_accessor.py`", ""])
    lines.extend(f"- `{md_escape(line)}`" for line in accessor_refs[:40])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_outputs(reference: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu_inventory = inventory_cpu_wrfout(reference)
    gpu_inventory = inventory_gpu_writer()
    rows = build_compat_rows(cpu_inventory, gpu_inventory)
    outputs = {
        "cpu_inventory": output_dir / "cpu_wrfout_reference_inventory.json",
        "gpu_inventory": output_dir / "gpu_wrfout_writer_inventory.json",
        "compat_matrix": output_dir / "compat_matrix.md",
        "endpoint_audit": output_dir / "io_endpoint_audit.md",
        "explicit_deviations": output_dir / "explicit_deviations.md",
    }
    write_json(outputs["cpu_inventory"], cpu_inventory)
    write_json(outputs["gpu_inventory"], gpu_inventory)
    write_compat_matrix(outputs["compat_matrix"], rows, cpu_inventory, gpu_inventory)
    write_endpoint_audit(outputs["endpoint_audit"], reference)
    write_explicit_deviations(outputs["explicit_deviations"], rows, gpu_inventory)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Single Gen2 wrfout_d02 reference file.")
    parser.add_argument("--output-dir", type=Path, default=SPRINT_DIR, help="Sprint output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_outputs(args.reference, args.output_dir)
    print("M7 wrfout I/O compatibility audit complete")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
