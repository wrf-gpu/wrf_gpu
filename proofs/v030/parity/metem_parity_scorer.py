#!/usr/bin/env python3
"""WPS met_em parity scorer for v0.3.0 native metgrid-equivalent ingest.

This is an oracle-side harness only.  It compares a native-generated
metgrid-equivalent NetCDF artifact against real WPS/metgrid ``met_em`` output,
streaming one file at a time on CPU.  It does not call WRF, JAX, CUDA, or any
native ingest implementation.

Typical full oracle sanity run:

    PYTHONPATH=src taskset -c 0-3 python proofs/v030/parity/metem_parity_scorer.py \
      --oracle-root /mnt/data/canairy_meteo/runs/wps_cases \
      --native-root /mnt/data/canairy_meteo/runs/wps_cases \
      --output proofs/v030/parity/metem_self_parity.json

Generate a placeholder native tree of symlinks to the oracle, perturb one file,
and score it without failing the shell:

    STUB=$(mktemp -d /tmp/metem_stub.XXXXXX)
    PYTHONPATH=src taskset -c 0-3 python proofs/v030/parity/metem_parity_scorer.py \
      --oracle-root /mnt/data/canairy_meteo/runs/wps_cases \
      --make-stub-root "$STUB" \
      --cases 20260521_18z_72h \
      --domains d03 \
      --perturb-variable TT \
      --perturb-delta 1.0 \
      --allow-failures \
      --output proofs/v030/parity/metem_perturbed_parity.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

import numpy as np
from netCDF4 import Dataset


ORACLE_ROOT = Path("/mnt/data/canairy_meteo/runs/wps_cases")
DEFAULT_DOMAINS = ("d01", "d02", "d03")
METEM_RE = re.compile(r"^met_em\.(d\d{2})\.(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})\.nc$")


@dataclass(frozen=True)
class Tolerance:
    rmse: float
    bias: float
    max_abs: float
    exact: bool = False


@dataclass(frozen=True)
class VariablePolicy:
    source_class: str
    source: str
    derivation: str
    interpolation: str
    missing_policy: str
    tolerance: Tolerance


@dataclass(frozen=True)
class MetFile:
    case: str
    domain: str
    valid_time: str
    path: Path


class VarAccumulator:
    """Streaming numeric/exact comparison stats for one variable."""

    def __init__(self) -> None:
        self.files = 0
        self.numeric_count = 0
        self.sum_diff = 0.0
        self.sum_sq_diff = 0.0
        self.max_abs = 0.0
        self.nan_mismatch_count = 0
        self.exact_count = 0
        self.exact_mismatch_count = 0
        self.missing_native_count = 0
        self.shape_mismatch_count = 0
        self.metadata_mismatch_count = 0
        self.examples: list[dict[str, Any]] = []

    def add_example(self, item: dict[str, Any]) -> None:
        if len(self.examples) < 8:
            self.examples.append(item)

    def add_numeric(self, diff: np.ndarray, nan_mismatches: int) -> None:
        self.files += 1
        self.nan_mismatch_count += int(nan_mismatches)
        if diff.size == 0:
            return
        diff64 = np.asarray(diff, dtype=np.float64)
        self.numeric_count += int(diff64.size)
        self.sum_diff += float(np.sum(diff64))
        self.sum_sq_diff += float(np.sum(diff64 * diff64))
        self.max_abs = max(self.max_abs, float(np.max(np.abs(diff64))))

    def add_exact(self, total: int, mismatches: int) -> None:
        self.files += 1
        self.exact_count += int(total)
        self.exact_mismatch_count += int(mismatches)

    def metric_dict(self, policy: VariablePolicy) -> dict[str, Any]:
        if self.numeric_count:
            bias = self.sum_diff / self.numeric_count
            rmse = math.sqrt(self.sum_sq_diff / self.numeric_count)
        else:
            bias = None
            rmse = None

        tol = policy.tolerance
        reasons: list[str] = []
        if self.missing_native_count:
            reasons.append(f"missing_native={self.missing_native_count}")
        if self.shape_mismatch_count:
            reasons.append(f"shape_mismatch={self.shape_mismatch_count}")
        if self.metadata_mismatch_count:
            reasons.append(f"metadata_mismatch={self.metadata_mismatch_count}")
        if self.nan_mismatch_count:
            reasons.append(f"nan_mismatch={self.nan_mismatch_count}")
        if self.exact_mismatch_count:
            reasons.append(f"exact_mismatch={self.exact_mismatch_count}")
        if rmse is not None and rmse > tol.rmse:
            reasons.append(f"rmse>{tol.rmse:g}")
        if bias is not None and abs(bias) > tol.bias:
            reasons.append(f"abs_bias>{tol.bias:g}")
        if self.max_abs > tol.max_abs:
            reasons.append(f"max_abs>{tol.max_abs:g}")

        return {
            "status": "PASS" if not reasons else "FAIL",
            "fail_reasons": reasons,
            "files": self.files,
            "numeric_count": self.numeric_count,
            "rmse": rmse,
            "bias": bias,
            "max_abs_diff": self.max_abs if self.numeric_count else None,
            "nan_mismatch_count": self.nan_mismatch_count,
            "exact_count": self.exact_count,
            "exact_mismatch_count": self.exact_mismatch_count,
            "missing_native_count": self.missing_native_count,
            "shape_mismatch_count": self.shape_mismatch_count,
            "metadata_mismatch_count": self.metadata_mismatch_count,
            "tolerance": asdict(tol),
            "policy": asdict(policy),
            "examples": self.examples,
        }


EXACT = Tolerance(0.0, 0.0, 0.0, exact=True)
STATIC_FLOAT = Tolerance(1.0e-6, 1.0e-7, 1.0e-5)
COORD_FLOAT = Tolerance(1.0e-6, 1.0e-7, 2.0e-5)
MAP_FLOAT = Tolerance(1.0e-7, 1.0e-8, 1.0e-6)
STATIC_HGT = Tolerance(1.0e-3, 1.0e-4, 1.0e-2)
ATM_TEMP = Tolerance(5.0e-2, 1.0e-2, 5.0e-1)
ATM_WIND = Tolerance(5.0e-2, 1.0e-2, 5.0e-1)
ATM_HGT = Tolerance(1.0, 2.0e-1, 10.0)
ATM_Q = Tolerance(1.0e-5, 2.0e-6, 1.0e-4)
PRESSURE = Tolerance(2.0e-1, 5.0e-2, 2.0)
SURFACE_PRESSURE = Tolerance(5.0, 1.0, 50.0)
SURFACE_TEMP = Tolerance(1.0e-1, 2.0e-2, 1.0)
SOIL_TEMP = Tolerance(2.5e-1, 5.0e-2, 2.0)
SOIL_MOISTURE = Tolerance(2.0e-3, 5.0e-4, 2.0e-2)


def _build_policies() -> dict[str, VariablePolicy]:
    policies: dict[str, VariablePolicy] = {}

    def add(
        names: Iterable[str],
        source_class: str,
        source: str,
        derivation: str,
        interpolation: str,
        missing_policy: str,
        tolerance: Tolerance,
    ) -> None:
        for name in names:
            policies[name] = VariablePolicy(
                source_class=source_class,
                source=source,
                derivation=derivation,
                interpolation=interpolation,
                missing_policy=missing_policy,
                tolerance=tolerance,
            )

    add(
        ["Times"],
        "time",
        "case valid time encoded as WRF DateStrLen=19",
        "format YYYY-MM-DD_HH:MM:SS",
        "none",
        "must match oracle exactly",
        EXACT,
    )
    add(
        ["PRES"],
        "derived_from_aifs_plus_wps_levels",
        "AIFS pressure_surface for level 0; fixed WPS pressure levels for 1000..50 hPa",
        "assemble num_metgrid_levels: surface pressure, 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa",
        "horizontal interpolation to mass grid; pressure levels are not vertically interpolated",
        "no missing levels allowed; surface pressure source missing is FAIL",
        PRESSURE,
    )
    add(
        ["TT"],
        "partial_aifs_gap",
        "AIFS expanded temperature_925hpa and temperature_850hpa only; base temperature_2m is not a pressure-level substitute",
        "Celsius to Kelvin; all other pressure levels require additional AIFS fields or WPS-compatible fallback",
        "horizontal interpolation to mass grid",
        "unsupported pressure levels must be explicitly filled by approved policy; silent zeros/clamps are FAIL",
        ATM_TEMP,
    )
    add(
        ["UU", "VV"],
        "aifs_gap",
        "current AIFS archive has only wind_u/v_10m and wind_u/v_100m, not pressure-level winds",
        "none available for met_em pressure levels",
        "horizontal interpolation plus U/V staggering once pressure-level winds exist",
        "must add pressure-level winds or documented external source; 10m/100m cannot fill 14 WPS levels",
        ATM_WIND,
    )
    add(
        ["GHT"],
        "partial_aifs_gap",
        "AIFS expanded geopotential_height_925hpa, _850hpa, _500hpa only",
        "already in meters; remaining levels require additional AIFS fields or WPS-compatible fallback",
        "horizontal interpolation to mass grid",
        "unsupported pressure levels must be explicitly filled by approved policy",
        ATM_HGT,
    )
    add(
        ["SPECHUMD"],
        "aifs_gap",
        "current AIFS archive has dew_point_temperature_2m only, not pressure-level humidity",
        "2m dewpoint can derive near-surface q but cannot populate 14 pressure levels",
        "horizontal interpolation to mass grid once humidity source exists",
        "WPS sentinel -1 at surface level must remain exact; other missing levels are FAIL",
        ATM_Q,
    )
    add(
        ["PSFC"],
        "aifs_direct",
        "AIFS expanded pressure_surface",
        "direct Pa field",
        "horizontal interpolation to mass grid",
        "expanded-field coverage missing for 2026-05 cases after 2026-05-01 is a source gap",
        SURFACE_PRESSURE,
    )
    add(
        ["PMSL"],
        "aifs_direct",
        "AIFS base pressure_reduced_to_mean_sea_level",
        "direct Pa field",
        "horizontal interpolation to mass grid",
        "no missing allowed",
        SURFACE_PRESSURE,
    )
    add(
        ["DEWPT"],
        "aifs_direct",
        "AIFS expanded dew_point_temperature_2m",
        "Celsius to Kelvin",
        "horizontal interpolation to mass grid",
        "expanded-field coverage missing for 2026-05 cases after 2026-05-01 is a source gap",
        SURFACE_TEMP,
    )
    add(
        ["SKINTEMP"],
        "aifs_gap",
        "no skin temperature or SST field in current AIFS archive",
        "requires additional source or approved derivation from surface fields",
        "horizontal interpolation to mass grid after source/derivation",
        "cannot be replaced silently by 2m temperature",
        SURFACE_TEMP,
    )
    add(
        ["SM", "SM000010", "SM010040"],
        "aifs_gap",
        "no soil moisture fields in current AIFS base or expanded archive",
        "requires additional source, climatology, or approved fallback",
        "soil-layer mapping to WPS 0-10 and 10-40 cm layers",
        "missing soil water policy must be explicit; no zero/clamp fill",
        SOIL_MOISTURE,
    )
    add(
        ["ST", "ST000010", "ST010040"],
        "aifs_gap",
        "no soil temperature fields in current AIFS base or expanded archive",
        "requires additional source, climatology, or approved fallback",
        "soil-layer mapping to WPS 0-10 and 10-40 cm layers",
        "missing soil temperature policy must be explicit",
        SOIL_TEMP,
    )
    add(
        ["SOIL_LAYERS"],
        "wps_constant",
        "WPS/metgrid soil-layer coordinate",
        "2-layer coordinate matching met_em values 40 and 10 cm in these files",
        "none",
        "must match oracle exactly",
        EXACT,
    )
    add(
        ["SOILHGT"],
        "aifs_gap",
        "source-model surface height is present in met_em but not in current AIFS archive",
        "requires source or documented fallback to geogrid HGT_M",
        "horizontal interpolation to mass grid after source selection",
        "fallback choice must be recorded because real.exe uses it in initialization",
        STATIC_HGT,
    )
    add(
        ["LANDSEA"],
        "static_geog",
        "geo_em / geogrid land-water classification",
        "copied or derived from LANDMASK / land category",
        "mass-grid static field",
        "categorical exact match required",
        EXACT,
    )
    add(
        ["LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM"],
        "static_geog",
        "geo_em geogrid static categorical fields",
        "copy from geo_em into metgrid-equivalent artifact",
        "mass-grid static field",
        "categorical exact match required",
        EXACT,
    )
    add(
        ["LANDUSEF", "SOILCTOP", "SOILCBOT"],
        "static_geog",
        "geo_em geogrid categorical-fraction fields",
        "copy from geo_em into metgrid-equivalent artifact",
        "mass-grid static 3D category field",
        "near-exact float/category match required",
        STATIC_FLOAT,
    )
    add(
        ["SNOALB", "LAI12M", "GREENFRAC", "ALBEDO12M", "SOILTEMP"],
        "static_geog",
        "geo_em monthly/static geogrid fields",
        "copy from geo_em into metgrid-equivalent artifact",
        "mass-grid static field; 12-month fields retain z-dimension0012",
        "near-exact match required",
        STATIC_FLOAT,
    )
    add(
        ["HGT_M"],
        "static_geog",
        "geo_em terrain height",
        "copy from geo_em into metgrid-equivalent artifact",
        "mass-grid static field",
        "near-exact terrain match required",
        STATIC_HGT,
    )
    add(
        ["OL1", "OL2", "OL3", "OL4", "OA1", "OA2", "OA3", "OA4", "VAR", "CON"],
        "static_geog",
        "geo_em sub-grid orography/GWD descriptor fields",
        "copy from geo_em into metgrid-equivalent artifact",
        "mass-grid static field",
        "near-exact match required; WRF metadata labels these as whoknows/something",
        STATIC_FLOAT,
    )
    add(
        ["XLAT_M", "XLONG_M", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "XLAT_C", "XLONG_C", "CLAT", "CLONG"],
        "projection_static",
        "geo_em / geogrid projection coordinates",
        "copy from geo_em or recompute bit-faithfully from WPS projection metadata",
        "mass/U/V/corner staggered grids as encoded by variable dimensions",
        "coordinate mismatch beyond tolerance is FAIL",
        COORD_FLOAT,
    )
    add(
        ["MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX", "MAPFAC_UX", "MAPFAC_VX", "MAPFAC_MY", "MAPFAC_UY", "MAPFAC_VY"],
        "projection_static",
        "geo_em / geogrid map factors",
        "copy from geo_em or recompute bit-faithfully from WPS projection metadata",
        "mass/U/V staggered grids",
        "map-factor mismatch beyond tolerance is FAIL",
        MAP_FLOAT,
    )
    add(
        ["COSALPHA", "SINALPHA", "COSALPHA_U", "SINALPHA_U", "COSALPHA_V", "SINALPHA_V", "F", "E"],
        "projection_static",
        "geo_em / geogrid rotation and Coriolis fields",
        "copy from geo_em or recompute bit-faithfully from WPS projection metadata",
        "mass/U/V staggered grids",
        "near-exact match required",
        MAP_FLOAT,
    )
    return policies


VARIABLE_POLICIES = _build_policies()


def _unknown_policy() -> VariablePolicy:
    return VariablePolicy(
        source_class="unknown",
        source="not predeclared in v0.3.0 policy table",
        derivation="unknown",
        interpolation="unknown",
        missing_policy="unknown source is FAIL until mapped",
        tolerance=STATIC_FLOAT,
    )

VAR_METADATA_ATTRS = (
    "FieldType",
    "MemoryOrder",
    "units",
    "description",
    "stagger",
    "sr_x",
    "sr_y",
)

GLOBAL_ATTRS = (
    "TITLE",
    "SIMULATION_START_DATE",
    "WEST-EAST_GRID_DIMENSION",
    "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION",
    "WEST-EAST_PATCH_START_UNSTAG",
    "WEST-EAST_PATCH_END_UNSTAG",
    "WEST-EAST_PATCH_START_STAG",
    "WEST-EAST_PATCH_END_STAG",
    "SOUTH-NORTH_PATCH_START_UNSTAG",
    "SOUTH-NORTH_PATCH_END_UNSTAG",
    "SOUTH-NORTH_PATCH_START_STAG",
    "SOUTH-NORTH_PATCH_END_STAG",
    "GRIDTYPE",
    "DX",
    "DY",
    "DYN_OPT",
    "CEN_LAT",
    "CEN_LON",
    "TRUELAT1",
    "TRUELAT2",
    "MOAD_CEN_LAT",
    "STAND_LON",
    "POLE_LAT",
    "POLE_LON",
    "corner_lats",
    "corner_lons",
    "MAP_PROJ",
    "MMINLU",
    "NUM_LAND_CAT",
    "ISWATER",
    "ISLAKE",
    "ISICE",
    "ISURBAN",
    "ISOILWATER",
    "grid_id",
    "parent_id",
    "i_parent_start",
    "j_parent_start",
    "i_parent_end",
    "j_parent_end",
    "parent_grid_ratio",
    "sr_x",
    "sr_y",
    "NUM_METGRID_SOIL_LEVELS",
    "FLAG_METGRID",
    "FLAG_EXCLUDED_MIDDLE",
    "FLAG_SOIL_LAYERS",
    "FLAG_PSFC",
    "FLAG_SM000010",
    "FLAG_SM010040",
    "FLAG_ST000010",
    "FLAG_ST010040",
    "FLAG_SLP",
    "FLAG_SH",
    "FLAG_SOILHGT",
    "FLAG_MF_XY",
    "FLAG_LAI12M",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None or value.strip().lower() in ("", "all"):
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def discover_cases(root: Path) -> tuple[str, ...]:
    return tuple(sorted(p.name for p in root.iterdir() if p.is_dir()))


def discover_oracle_files(root: Path, cases: Iterable[str], domains: Iterable[str]) -> list[MetFile]:
    selected_domains = set(domains)
    out: list[MetFile] = []
    for case in cases:
        case_l3 = root / case / "l3"
        if not case_l3.is_dir():
            continue
        for path in sorted(case_l3.glob("met_em.d*.nc")):
            match = METEM_RE.match(path.name)
            if not match:
                continue
            domain, valid_time = match.groups()
            if domain not in selected_domains:
                continue
            out.append(MetFile(case=case, domain=domain, valid_time=valid_time, path=path))
    return out


def resolve_native_path(native_root: Path, met_file: MetFile) -> Path | None:
    if native_root.resolve() == ORACLE_ROOT.resolve() and met_file.path.exists():
        return met_file.path
    candidates = (
        native_root / met_file.case / "l3" / met_file.path.name,
        native_root / met_file.case / met_file.path.name,
        native_root / "l3" / met_file.path.name,
        native_root / met_file.path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _attr_value(ds: Dataset, name: str) -> Any:
    if not hasattr(ds, name):
        return None
    return getattr(ds, name)


def _attrs_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.dtype.kind in "SUO" or right_arr.dtype.kind in "SUO":
        return _jsonable(left) == _jsonable(right)
    try:
        return bool(np.allclose(left_arr, right_arr, rtol=0.0, atol=1.0e-6, equal_nan=True))
    except TypeError:
        return _jsonable(left) == _jsonable(right)


def _to_float_array(value: Any) -> np.ndarray:
    arr = np.ma.asarray(value)
    return np.asarray(np.ma.filled(arr, np.nan), dtype=np.float64)


def _is_numeric_var(var: Any) -> bool:
    return np.issubdtype(var.dtype, np.number)


def _compare_variable_metadata(
    name: str,
    oracle_var: Any,
    native_var: Any,
    acc: VarAccumulator,
    file_key: dict[str, str],
) -> None:
    issues = []
    if tuple(oracle_var.dimensions) != tuple(native_var.dimensions):
        issues.append({
            "field": "dimensions",
            "oracle": tuple(oracle_var.dimensions),
            "native": tuple(native_var.dimensions),
        })
    if str(oracle_var.dtype) != str(native_var.dtype):
        issues.append({
            "field": "dtype",
            "oracle": str(oracle_var.dtype),
            "native": str(native_var.dtype),
        })
    for attr in VAR_METADATA_ATTRS:
        oracle_has = hasattr(oracle_var, attr)
        native_has = hasattr(native_var, attr)
        if oracle_has != native_has:
            issues.append({"field": attr, "oracle": oracle_has, "native": native_has})
            continue
        if oracle_has:
            left = getattr(oracle_var, attr)
            right = getattr(native_var, attr)
            if _jsonable(left) != _jsonable(right):
                issues.append({"field": attr, "oracle": _jsonable(left), "native": _jsonable(right)})
    if issues:
        acc.metadata_mismatch_count += 1
        acc.add_example({**file_key, "variable": name, "metadata_issues": issues[:4]})


def _compare_arrays(
    name: str,
    oracle_var: Any,
    native_var: Any,
    acc: VarAccumulator,
    file_key: dict[str, str],
) -> None:
    if tuple(oracle_var.shape) != tuple(native_var.shape):
        acc.shape_mismatch_count += 1
        acc.add_example({
            **file_key,
            "variable": name,
            "oracle_shape": tuple(int(x) for x in oracle_var.shape),
            "native_shape": tuple(int(x) for x in native_var.shape),
        })
        return

    oracle_numeric = _is_numeric_var(oracle_var)
    native_numeric = _is_numeric_var(native_var)
    if oracle_numeric != native_numeric:
        acc.metadata_mismatch_count += 1
        acc.add_example({**file_key, "variable": name, "issue": "numeric_vs_nonnumeric"})
        return

    if not oracle_numeric:
        oracle_arr = np.asarray(oracle_var[:])
        native_arr = np.asarray(native_var[:])
        mismatches = int(np.count_nonzero(oracle_arr != native_arr))
        acc.add_exact(total=int(oracle_arr.size), mismatches=mismatches)
        if mismatches:
            acc.add_example({**file_key, "variable": name, "exact_mismatches": mismatches})
        return

    oracle_arr = _to_float_array(oracle_var[:])
    native_arr = _to_float_array(native_var[:])
    oracle_finite = np.isfinite(oracle_arr)
    native_finite = np.isfinite(native_arr)
    both = oracle_finite & native_finite
    nan_mismatches = int(np.count_nonzero(oracle_finite ^ native_finite))
    diff = native_arr[both] - oracle_arr[both]
    acc.add_numeric(diff=diff, nan_mismatches=nan_mismatches)
    if nan_mismatches:
        acc.add_example({**file_key, "variable": name, "nan_mismatches": nan_mismatches})


def _compare_global_attrs(
    oracle_ds: Dataset,
    native_ds: Dataset,
    file_key: dict[str, str],
    examples: list[dict[str, Any]],
) -> int:
    mismatches = 0
    for attr in GLOBAL_ATTRS:
        left = _attr_value(oracle_ds, attr)
        right = _attr_value(native_ds, attr)
        if not _attrs_equal(left, right):
            mismatches += 1
            if len(examples) < 20:
                examples.append({
                    **file_key,
                    "attribute": attr,
                    "oracle": _jsonable(left),
                    "native": _jsonable(right),
                })
    return mismatches


def compare(
    oracle_root: Path,
    native_root: Path,
    cases: tuple[str, ...],
    domains: tuple[str, ...],
) -> dict[str, Any]:
    oracle_files = discover_oracle_files(oracle_root, cases=cases, domains=domains)
    aggregate: dict[str, VarAccumulator] = defaultdict(VarAccumulator)
    per_case: dict[str, dict[str, VarAccumulator]] = defaultdict(lambda: defaultdict(VarAccumulator))
    missing_files: list[dict[str, str]] = []
    global_attr_examples: list[dict[str, Any]] = []
    global_attr_mismatches = 0
    dim_mismatches: list[dict[str, Any]] = []
    compared_files = 0

    for met_file in oracle_files:
        file_key = {
            "case": met_file.case,
            "domain": met_file.domain,
            "valid_time": met_file.valid_time,
            "file": met_file.path.name,
        }
        native_path = resolve_native_path(native_root, met_file)
        if native_path is None:
            missing_files.append(file_key)
            for name in VARIABLE_POLICIES:
                aggregate[name].missing_native_count += 1
                per_case[met_file.case][name].missing_native_count += 1
            continue

        compared_files += 1
        with Dataset(met_file.path) as oracle_ds, Dataset(native_path) as native_ds:
            oracle_dims = {name: len(dim) for name, dim in oracle_ds.dimensions.items()}
            native_dims = {name: len(dim) for name, dim in native_ds.dimensions.items()}
            if oracle_dims != native_dims:
                if len(dim_mismatches) < 20:
                    dim_mismatches.append({
                        **file_key,
                        "oracle_dims": oracle_dims,
                        "native_dims": native_dims,
                    })

            global_attr_mismatches += _compare_global_attrs(
                oracle_ds, native_ds, file_key=file_key, examples=global_attr_examples
            )

            for name, oracle_var in oracle_ds.variables.items():
                if name not in VARIABLE_POLICIES:
                    # The policy table is expected to be complete, but keep the
                    # scorer defensive if future WPS adds a variable.
                    VARIABLE_POLICIES[name] = _unknown_policy()
                agg = aggregate[name]
                case_acc = per_case[met_file.case][name]
                if name not in native_ds.variables:
                    agg.missing_native_count += 1
                    case_acc.missing_native_count += 1
                    agg.add_example({**file_key, "variable": name, "issue": "missing_native_variable"})
                    case_acc.add_example({**file_key, "variable": name, "issue": "missing_native_variable"})
                    continue
                native_var = native_ds.variables[name]

                before = agg.metadata_mismatch_count
                _compare_variable_metadata(name, oracle_var, native_var, agg, file_key)
                if agg.metadata_mismatch_count > before:
                    case_acc.metadata_mismatch_count += agg.metadata_mismatch_count - before
                    case_acc.add_example(agg.examples[-1])

                _compare_arrays(name, oracle_var, native_var, agg, file_key)
                # Re-run on the case accumulator.  This keeps the logic simple
                # and arrays small enough for the duplicated read to be
                # acceptable on the v0.3 CPU-only oracle lane.
                _compare_arrays(name, oracle_var, native_var, case_acc, file_key)

            for name in set(native_ds.variables) - set(oracle_ds.variables):
                if name not in VARIABLE_POLICIES:
                    VARIABLE_POLICIES[name] = _unknown_policy()
                aggregate[name].metadata_mismatch_count += 1
                aggregate[name].add_example({**file_key, "variable": name, "issue": "extra_native_variable"})
                per_case[met_file.case][name].metadata_mismatch_count += 1
                per_case[met_file.case][name].add_example({**file_key, "variable": name, "issue": "extra_native_variable"})

    variable_results = {
        name: aggregate[name].metric_dict(VARIABLE_POLICIES[name])
        for name in sorted(aggregate)
    }
    case_results: dict[str, Any] = {}
    for case, vars_for_case in sorted(per_case.items()):
        rendered = {
            name: vars_for_case[name].metric_dict(VARIABLE_POLICIES[name])
            for name in sorted(vars_for_case)
        }
        case_status = "PASS" if all(v["status"] == "PASS" for v in rendered.values()) else "FAIL"
        case_results[case] = {
            "status": case_status,
            "variables": rendered,
        }

    variable_failures = sorted(name for name, item in variable_results.items() if item["status"] != "PASS")
    status = "PASS"
    fail_reasons = []
    if missing_files:
        status = "FAIL"
        fail_reasons.append(f"missing_files={len(missing_files)}")
    if dim_mismatches:
        status = "FAIL"
        fail_reasons.append(f"dimension_mismatches={len(dim_mismatches)}")
    if global_attr_mismatches:
        status = "FAIL"
        fail_reasons.append(f"global_attr_mismatches={global_attr_mismatches}")
    if variable_failures:
        status = "FAIL"
        fail_reasons.append(f"variable_failures={len(variable_failures)}")

    return {
        "artifact_type": "v030_metem_parity_score",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "oracle_root": str(oracle_root),
        "native_root": str(native_root),
        "cases": list(cases),
        "domains": list(domains),
        "oracle_file_count": len(oracle_files),
        "compared_file_count": compared_files,
        "status": status,
        "fail_reasons": fail_reasons,
        "missing_files": missing_files[:50],
        "missing_file_count": len(missing_files),
        "dimension_mismatches": dim_mismatches,
        "global_attr_mismatch_count": global_attr_mismatches,
        "global_attr_mismatch_examples": global_attr_examples,
        "variable_failures": variable_failures,
        "variable_results": variable_results,
        "case_results": case_results,
        "variable_policies": {name: asdict(policy) for name, policy in sorted(VARIABLE_POLICIES.items())},
    }


def build_stub_tree(
    oracle_root: Path,
    stub_root: Path,
    cases: tuple[str, ...],
    domains: tuple[str, ...],
) -> list[MetFile]:
    if stub_root.exists() and any(stub_root.iterdir()):
        raise RuntimeError(f"stub root exists and is not empty: {stub_root}")
    stub_root.mkdir(parents=True, exist_ok=True)
    files = discover_oracle_files(oracle_root, cases=cases, domains=domains)
    for met_file in files:
        out_dir = stub_root / met_file.case / "l3"
        out_dir.mkdir(parents=True, exist_ok=True)
        link = out_dir / met_file.path.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(met_file.path)
    return files


def perturb_stub_file(
    stub_root: Path,
    met_file: MetFile,
    variable: str,
    delta: float,
) -> dict[str, Any]:
    target = stub_root / met_file.case / "l3" / met_file.path.name
    if target.is_symlink():
        target.unlink()
        shutil.copy2(met_file.path, target)
    elif not target.exists():
        shutil.copy2(met_file.path, target)

    with Dataset(target, "r+") as ds:
        if variable not in ds.variables:
            raise RuntimeError(f"variable {variable} not found in {target}")
        var = ds.variables[variable]
        if not _is_numeric_var(var):
            raise RuntimeError(f"variable {variable} is not numeric and cannot be perturbed")
        data = np.asarray(var[:])
        before = float(np.nanmean(data.astype(np.float64)))
        var[:] = data + np.asarray(delta, dtype=data.dtype)
        after = float(np.nanmean(np.asarray(var[:], dtype=np.float64)))

    return {
        "case": met_file.case,
        "domain": met_file.domain,
        "valid_time": met_file.valid_time,
        "file": target.name,
        "variable": variable,
        "delta": delta,
        "mean_before": before,
        "mean_after": after,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--native-root", type=Path, default=None)
    parser.add_argument("--cases", default=None, help="Comma-separated case ids; default/all = all cases.")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS), help="Comma-separated domains; default d01,d02,d03.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-failures", action="store_true", help="Write FAIL result but return exit code 0.")
    parser.add_argument("--make-stub-root", type=Path, default=None, help="Create a symlink placeholder native tree before scoring.")
    parser.add_argument("--perturb-variable", default=None, help="Numeric variable to perturb in the stub tree.")
    parser.add_argument("--perturb-delta", type=float, default=0.0)
    parser.add_argument("--perturb-case", default=None)
    parser.add_argument("--perturb-domain", default=None)
    parser.add_argument("--perturb-valid-time", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    oracle_root = args.oracle_root
    cases = _parse_csv(args.cases) or discover_cases(oracle_root)
    domains = _parse_csv(args.domains) or DEFAULT_DOMAINS

    perturb_info = None
    if args.make_stub_root is not None:
        files = build_stub_tree(oracle_root, args.make_stub_root, cases=cases, domains=domains)
        if args.perturb_variable:
            candidates = [
                item for item in files
                if (args.perturb_case is None or item.case == args.perturb_case)
                and (args.perturb_domain is None or item.domain == args.perturb_domain)
                and (args.perturb_valid_time is None or item.valid_time == args.perturb_valid_time)
            ]
            if not candidates:
                raise RuntimeError("no oracle file matched perturb selection")
            perturb_info = perturb_stub_file(
                args.make_stub_root,
                candidates[0],
                variable=args.perturb_variable,
                delta=args.perturb_delta,
            )
        native_root = args.make_stub_root
    else:
        if args.native_root is None:
            raise RuntimeError("--native-root is required unless --make-stub-root is used")
        native_root = args.native_root

    result = compare(oracle_root=oracle_root, native_root=native_root, cases=cases, domains=domains)
    if perturb_info is not None:
        result["perturbation"] = perturb_info

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")

    summary = {
        "status": result["status"],
        "oracle_file_count": result["oracle_file_count"],
        "compared_file_count": result["compared_file_count"],
        "fail_reasons": result["fail_reasons"],
        "variable_failures": result["variable_failures"][:20],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))

    if result["status"] != "PASS" and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
