"""d02 lateral-boundary replay extraction from Gen2 parent-domain history."""

from __future__ import annotations

from datetime import timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset
import yaml
import zarr

from gpuwrf.io.gen2_accessor import GEN2_READ_ONLY_ROOT, Gen2Run


BOUNDARY_VARIABLES = ("U", "V", "T", "QVAPOR", "PH")
SIDES = ("W", "E", "S", "N")
WRFBDY_VARIABLES = ("U", "V", "W", "T", "QVAPOR", "PH", "MU")
WRFBDY_STATE_FIELDS = {
    "U": "u",
    "V": "v",
    "W": "w",
    "T": "theta",
    "QVAPOR": "qv",
    "PH": "ph",
    "MU": "mu",
}
WRFBDY_BASE_SUFFIXES = {"W": "BXS", "E": "BXE", "S": "BYS", "N": "BYE"}
WRFBDY_TENDENCY_SUFFIXES = {"W": "BTXS", "E": "BTXE", "S": "BTYS", "N": "BTYE"}
COORDS = {
    "U": ("XLAT_U", "XLONG_U", "ZNU"),
    "V": ("XLAT_V", "XLONG_V", "ZNU"),
    "W": ("XLAT", "XLONG", "ZNW"),
    "T": ("XLAT", "XLONG", "ZNU"),
    "QVAPOR": ("XLAT", "XLONG", "ZNU"),
    "P": ("XLAT", "XLONG", "ZNU"),
    "PB": ("XLAT", "XLONG", "ZNU"),
    "PH": ("XLAT", "XLONG", "ZNW"),
    "PHB": ("XLAT", "XLONG", "ZNW"),
    "MU": ("XLAT", "XLONG", "ZNU"),
    "MUB": ("XLAT", "XLONG", "ZNU"),
}
TOLERANCES = {
    "U": {"rel_mae_max": 0.03, "rmse_max": 0.5, "units": "m s-1"},
    "V": {"rel_mae_max": 0.03, "rmse_max": 0.5, "units": "m s-1"},
    "W": {"rmse_max": 0.2, "units": "m s-1"},
    "T": {"rmse_max": 0.5, "units": "K"},
    "QVAPOR": {"rel_mae_max": 0.03, "rmse_max": 1.0e-4, "units": "kg kg-1"},
    "P": {"rmse_max": 50.0, "units": "Pa"},
    "PB": {"rmse_max": 50.0, "units": "Pa"},
    "PH": {"rel_mae_max": 0.005, "rmse_max": 20.0, "units": "m2 s-2"},
    "PHB": {"rel_mae_max": 0.005, "rmse_max": 20.0, "units": "m2 s-2"},
    "MU": {"rmse_max": 50.0, "units": "Pa"},
    "MUB": {"rmse_max": 50.0, "units": "Pa"},
}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_gen2_write_target(path: Path) -> None:
    if _is_under(path, GEN2_READ_ONLY_ROOT):
        raise PermissionError(f"refusing to write boundary fixture inside read-only Gen2 path: {path}")


def _lambert_xy(lat, lon, truelat1: float, truelat2: float, stand_lon: float, ref_lat: float = 0.0):
    """Project WRF Lambert lat/lon to local x/y meters for fractional-index weights."""

    radius_m = 6_370_000.0
    lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))
    phi1 = math.radians(float(truelat1))
    phi2 = math.radians(float(truelat2))
    lambda0 = math.radians(float(stand_lon))
    if abs(phi1 - phi2) < 1.0e-12:
        cone = math.sin(phi1)
    else:
        cone = math.log(math.cos(phi1) / math.cos(phi2)) / math.log(
            math.tan(math.pi / 4.0 + phi2 / 2.0) / math.tan(math.pi / 4.0 + phi1 / 2.0)
        )
    factor = math.cos(phi1) * (math.tan(math.pi / 4.0 + phi1 / 2.0) ** cone) / cone
    rho = radius_m * factor / (np.tan(math.pi / 4.0 + lat_rad / 2.0) ** cone)
    rho0 = radius_m * factor / (math.tan(math.pi / 4.0 + math.radians(ref_lat) / 2.0) ** cone)
    theta = cone * (lon_rad - lambda0)
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    return x, y


def _weights(source: Dataset, target: Dataset, lat_name: str, lon_name: str) -> dict[str, np.ndarray]:
    lat_src = np.asarray(source.variables[lat_name][0], dtype=np.float64)
    lon_src = np.asarray(source.variables[lon_name][0], dtype=np.float64)
    lat_dst = np.asarray(target.variables[lat_name][0], dtype=np.float64)
    lon_dst = np.asarray(target.variables[lon_name][0], dtype=np.float64)
    xs, ys = _lambert_xy(lat_src, lon_src, source.TRUELAT1, source.TRUELAT2, source.STAND_LON)
    xt, yt = _lambert_xy(lat_dst, lon_dst, source.TRUELAT1, source.TRUELAT2, source.STAND_LON)
    origin = np.array([xs[0, 0], ys[0, 0]], dtype=np.float64)
    vi = np.array([xs[0, 1] - xs[0, 0], ys[0, 1] - ys[0, 0]], dtype=np.float64)
    vj = np.array([xs[1, 0] - xs[0, 0], ys[1, 0] - ys[0, 0]], dtype=np.float64)
    inverse = np.linalg.inv(np.column_stack([vi, vj]))
    fractional = inverse @ np.vstack([(xt - origin[0]).ravel(), (yt - origin[1]).ravel()])
    ii = fractional[0].reshape(lat_dst.shape)
    jj = fractional[1].reshape(lat_dst.shape)
    ny, nx = lat_src.shape
    i0 = np.clip(np.floor(ii).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(jj).astype(np.int64), 0, ny - 2)
    return {
        "i0": i0,
        "j0": j0,
        "wi": ii - i0,
        "wj": jj - j0,
    }


def _interp_horizontal(field: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    i0 = weights["i0"]
    j0 = weights["j0"]
    wi = weights["wi"]
    wj = weights["wj"]
    f00 = field[:, j0, i0]
    f10 = field[:, j0, i0 + 1]
    f01 = field[:, j0 + 1, i0]
    f11 = field[:, j0 + 1, i0 + 1]
    return f00 * (1.0 - wi) * (1.0 - wj) + f10 * wi * (1.0 - wj) + f01 * (1.0 - wi) * wj + f11 * wi * wj


def _interp_vertical(field: np.ndarray, src_eta: np.ndarray, dst_eta: np.ndarray) -> np.ndarray:
    if field.shape[0] == dst_eta.size and np.allclose(src_eta, dst_eta, rtol=0.0, atol=1.0e-12):
        return field
    src_inc = src_eta[::-1]
    flat = field.reshape(field.shape[0], -1)[::-1]
    out = np.empty((dst_eta.size, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        out[:, column] = np.interp(dst_eta[::-1], src_inc, flat[:, column])[::-1]
    return out.reshape((dst_eta.size,) + field.shape[1:])


def _sides(field: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "W": field[:, :, 0],
        "E": field[:, :, -1],
        "S": field[:, 0, :],
        "N": field[:, -1, :],
    }


def _read_var(dataset: Dataset, name: str) -> np.ndarray:
    data = dataset.variables[name][0]
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def _decode_char_times(dataset: Dataset) -> list[str]:
    if "Times" not in dataset.variables:
        return []
    chars = np.asarray(dataset.variables["Times"][:])
    decoded = []
    for row in chars:
        if row.dtype.kind == "S":
            decoded.append(b"".join(row.tolist()).decode("ascii", errors="replace").strip())
        else:
            decoded.append("".join(row.astype(str).tolist()).strip())
    return decoded


def _read_wrfbdy_array(dataset: Dataset, name: str, time_index: int) -> np.ndarray:
    if name not in dataset.variables:
        raise KeyError(f"{name!r} not present in wrfbdy file")
    variable = dataset.variables[name]
    data = variable[time_index] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def decode_wrfbdy(
    path: str | Path,
    *,
    variables: tuple[str, ...] = WRFBDY_VARIABLES,
    time_index: int = 0,
) -> dict[str, Any]:
    """Decode WRF `wrfbdy` boundary values and prescribed tendency increments.

    Returned side arrays keep WRF's `(bdy_width, z?, side_index)` order. The
    `_BT*` arrays are independent forcing tendencies consumed by WRF lateral
    boundary interpolation, not quantities reconstructed from GPU output.
    """

    source = Path(path)
    with Dataset(source, "r") as dataset:
        bdy_width = int(len(dataset.dimensions["bdy_width"])) if "bdy_width" in dataset.dimensions else 1
        times = _decode_char_times(dataset)
        decoded: dict[str, Any] = {
            "schema": "wrfbdy_decoder_v1",
            "path": str(source),
            "time_index": int(time_index),
            "times": times,
            "bdy_width": bdy_width,
            "variables": {},
        }
        for var in variables:
            var_payload: dict[str, Any] = {"sides": {}}
            for side in SIDES:
                base_name = f"{var}_{WRFBDY_BASE_SUFFIXES[side]}"
                tendency_name = f"{var}_{WRFBDY_TENDENCY_SUFFIXES[side]}"
                base = _read_wrfbdy_array(dataset, base_name, time_index)
                tendency = _read_wrfbdy_array(dataset, tendency_name, time_index)
                var_payload["sides"][side] = {
                    "boundary": base,
                    "tendency": tendency,
                    "boundary_variable": base_name,
                    "tendency_variable": tendency_name,
                    "units": getattr(dataset.variables[base_name], "units", ""),
                    "tendency_units": getattr(dataset.variables[tendency_name], "units", ""),
                    "shape": list(base.shape),
                }
            decoded["variables"][var] = var_payload
    return decoded


def wrfbdy_path_for_run(run: Gen2Run, domain: str = "d01") -> Path:
    """Return the Gen2 wrfbdy path for the requested parent domain."""

    path = run.path / f"wrfbdy_{domain}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _snapshot_field(snapshot: Any, field: str):
    if isinstance(snapshot, dict):
        return snapshot.get(field)
    return getattr(snapshot, field, None)


def _reverse_last_axis(array: np.ndarray) -> np.ndarray:
    return np.flip(array, axis=-1)


def _gpu_side_strip(field: np.ndarray, side: str, width: int) -> np.ndarray:
    """Extract GPU boundary strips in WRF wrfbdy `(width, z?, side_index)` order."""

    data = np.asarray(field, dtype=np.float64)
    if data.ndim == 3:
        if side == "W":
            return np.moveaxis(data[:, :, :width], 2, 0)
        if side == "E":
            return np.moveaxis(_reverse_last_axis(data[:, :, -width:]), 2, 0)
        if side == "S":
            return np.moveaxis(data[:, :width, :], 1, 0)
        return np.moveaxis(np.flip(data[:, -width:, :], axis=1), 1, 0)
    if data.ndim == 2:
        if side == "W":
            return data[:, :width].T
        if side == "E":
            return _reverse_last_axis(data[:, -width:]).T
        if side == "S":
            return data[:width, :]
        return np.flip(data[-width:, :], axis=0)
    raise ValueError(f"boundary snapshot field must be 2D or 3D, got shape {data.shape}")


def _common_extent(left: np.ndarray, right: np.ndarray) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    shape = tuple(min(int(a), int(b)) for a, b in zip(left.shape, right.shape, strict=False))
    index = tuple(slice(0, size) for size in shape)
    return index, index


def compare_boundary_tendency_to_wrfbdy(
    gpu_tendency: Any,
    decoded: dict[str, Any],
    *,
    variables: tuple[str, ...] = WRFBDY_VARIABLES,
    width: int | None = None,
    trim_corners: int = 0,
) -> dict[str, Any]:
    """Compare GPU boundary-apply tendencies with decoded wrfbdy `_BT*` terms."""

    bdy_width = int(decoded.get("bdy_width", 1) if width is None else width)
    per_variable: dict[str, Any] = {}
    for var in variables:
        state_field = WRFBDY_STATE_FIELDS[var]
        candidate = _snapshot_field(gpu_tendency, state_field)
        if candidate is None:
            continue
        side_records = {}
        for side in SIDES:
            truth = np.asarray(decoded["variables"][var]["sides"][side]["tendency"], dtype=np.float64)
            gpu_side = _gpu_side_strip(np.asarray(candidate), side, min(bdy_width, truth.shape[0]))
            left_index, right_index = _common_extent(gpu_side, truth)
            left = gpu_side[left_index]
            right = truth[right_index]
            if trim_corners > 0 and left.shape[-1] > 2 * int(trim_corners):
                trim = int(trim_corners)
                left = left[..., trim:-trim]
                right = right[..., trim:-trim]
            stats = _stats(left, right)
            stats["compared_shape"] = list(left.shape)
            stats["wrfbdy_variable"] = decoded["variables"][var]["sides"][side]["tendency_variable"]
            side_records[side] = stats
        aggregate = {
            "mae_max": max(record["mae"] for record in side_records.values()),
            "rmse_max": max(record["rmse"] for record in side_records.values()),
            "max_abs_max": max(record["max_abs"] for record in side_records.values()),
            "rel_mae_max": max(record["rel_mae"] for record in side_records.values()),
        }
        per_variable[var] = {"aggregate": aggregate, "sides": side_records}
    return {
        "schema": "wrfbdy_boundary_tendency_comparison_v1",
        "source": decoded.get("path"),
        "time_index": int(decoded.get("time_index", 0)),
        "bdy_width": bdy_width,
        "variables": per_variable,
    }


def _stats(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    diff = predicted - truth
    abs_diff = np.abs(diff)
    denom = float(np.mean(np.abs(truth))) + 1.0e-12
    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(abs_diff)),
        "rel_mae": float(np.mean(abs_diff) / denom),
    }


def _passes(var: str, aggregate: dict[str, float]) -> bool:
    tolerance = TOLERANCES[var]
    if "rmse_max" in tolerance and aggregate["rmse_max"] > tolerance["rmse_max"]:
        return False
    if "rel_mae_max" in tolerance and aggregate["rel_mae_max"] > tolerance["rel_mae_max"]:
        return False
    return True


def extract_d02_boundary(gen2_run: Gen2Run, output_path: str) -> dict[str, Any]:
    """Extract replayable d02 lateral boundaries from d01 Gen2 history files.

    The fixture is written as a local zarr group. Gen2 source files are only
    opened in NetCDF read mode; attempts to target the Gen2 tree for output are
    rejected before any zarr writer is created.
    """

    output = Path(output_path)
    _reject_gen2_write_target(output)
    d01_files = gen2_run.history_files("d01")
    d02_files = gen2_run.history_files("d02")
    if len(d01_files) != len(d02_files):
        raise ValueError(f"d01/d02 history count mismatch: {len(d01_files)} vs {len(d02_files)}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(d01_files[0], "r") as first_parent, Dataset(d02_files[0], "r") as first_child:
        weights_by_var = {
            var: _weights(first_parent, first_child, COORDS[var][0], COORDS[var][1]) for var in BOUNDARY_VARIABLES
        }
        eta_parent = {var: np.asarray(first_parent.variables[COORDS[var][2]][0], dtype=np.float64) for var in BOUNDARY_VARIABLES}
        eta_child = {var: np.asarray(first_child.variables[COORDS[var][2]][0], dtype=np.float64) for var in BOUNDARY_VARIABLES}

    arrays: dict[str, dict[str, list[np.ndarray]]] = {
        var: {side: [] for side in SIDES} for var in BOUNDARY_VARIABLES
    }
    validation: dict[str, Any] = {var: {"sides": {side: [] for side in SIDES}} for var in BOUNDARY_VARIABLES}

    for parent_path, child_path in zip(d01_files, d02_files, strict=True):
        with Dataset(parent_path, "r") as parent, Dataset(child_path, "r") as child:
            for var in BOUNDARY_VARIABLES:
                source = _read_var(parent, var)
                horizontal = _interp_horizontal(source, weights_by_var[var])
                replay = _interp_vertical(horizontal, eta_parent[var], eta_child[var])
                truth = _read_var(child, var)
                replay_sides = _sides(replay)
                truth_sides = _sides(truth)
                for side in SIDES:
                    arrays[var][side].append(replay_sides[side].astype(np.float32))
                    validation[var]["sides"][side].append(_stats(replay_sides[side], truth_sides[side]))

    aggregate_validation: dict[str, Any] = {}
    for var in BOUNDARY_VARIABLES:
        side_summary = {}
        for side in SIDES:
            samples = validation[var]["sides"][side]
            side_summary[side] = {
                "mae_max": max(sample["mae"] for sample in samples),
                "rmse_max": max(sample["rmse"] for sample in samples),
                "max_abs_max": max(sample["max_abs"] for sample in samples),
                "rel_mae_max": max(sample["rel_mae"] for sample in samples),
            }
        aggregate = {
            "mae_max": max(side_summary[side]["mae_max"] for side in SIDES),
            "rmse_max": max(side_summary[side]["rmse_max"] for side in SIDES),
            "max_abs_max": max(side_summary[side]["max_abs_max"] for side in SIDES),
            "rel_mae_max": max(side_summary[side]["rel_mae_max"] for side in SIDES),
        }
        aggregate["passed"] = _passes(var, aggregate)
        aggregate_validation[var] = {"aggregate": aggregate, "sides": side_summary, "tolerance": TOLERANCES[var]}

    times = [time.astimezone(timezone.utc).isoformat() for time in gen2_run.time_axis("d02")]
    group = zarr.open_group(str(output), mode="w")
    group.attrs.update(
        {
            "schema": "d02_boundary_replay_v1",
            "run_id": gen2_run.run_id,
            "source_parent_domain": "d01",
            "target_domain": "d02",
            "method": "WRF Lambert parent-to-child bilinear horizontal interpolation plus linear eta vertical interpolation",
            "times_utc": times,
            "variables": list(BOUNDARY_VARIABLES),
            "sides": list(SIDES),
            "validation": aggregate_validation,
            "source_files": {
                "parent": [path.name for path in d01_files],
                "child_truth": [path.name for path in d02_files],
            },
        }
    )
    group.create_array("lead_hours", data=np.arange(len(times), dtype=np.int32), overwrite=True)
    for var in BOUNDARY_VARIABLES:
        var_group = group.create_group(var)
        for side in SIDES:
            data = np.stack(arrays[var][side], axis=0)
            var_group.create_array(side, data=data, overwrite=True)

    manifest = {
        "fixture": str(output),
        "schema": "d02_boundary_replay_v1",
        "run_id": gen2_run.run_id,
        "source_path": str(gen2_run.path),
        "read_only_source": True,
        "variables": list(BOUNDARY_VARIABLES),
        "sides": list(SIDES),
        "times_utc": times,
        "validation": aggregate_validation,
        "tolerance_rationale": (
            "Nested d02 history differs from parent-regridded d01 by interpolation and feedback deltas; "
            "U/V/QVAPOR use few-percent relative MAE gates, T uses 0.5 K RMSE, PH uses a tight relative geopotential gate."
        ),
    }
    manifest_path = Path("fixtures/manifests/m6_d02_boundary_replay.yaml")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    summary_path = output / "validation_summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "BOUNDARY_VARIABLES",
    "SIDES",
    "TOLERANCES",
    "WRFBDY_STATE_FIELDS",
    "WRFBDY_VARIABLES",
    "compare_boundary_tendency_to_wrfbdy",
    "decode_wrfbdy",
    "extract_d02_boundary",
    "wrfbdy_path_for_run",
]
