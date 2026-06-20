"""Normalize decoded AIFS GRIB fields into the S1 source-field container."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Literal

import numpy as np

from gpuwrf.init.aifs_grib import (
    DEFAULT_AIFS_VTABLE,
    ISOBARIC_LEVELS_PA,
    WPS_MEAN_SEA_LEVEL,
    WPS_SURFACE_LEVEL,
    AIFSGrid,
    AIFSGribFile,
    AIFSMessage,
    read_aifs_grib,
    read_wps_intermediate,
)


SurfaceSpechumdPolicy = Literal["wps_fill", "dewpoint"]

SURFACE_SPECHUMD_WPS_FILL = np.float32(-1.0)
METEM_SOIL_LAYER_ORDER_CM = (40.0, 10.0)
SOIL_LAYER_FIELD_ORDER = ("010040", "000010")

EPSILON = 0.622
SVP1_PA = 611.2
SVP2 = 17.67
SVP3 = 29.65
SVPT0_K = 273.15


@dataclass(frozen=True)
class DecodedForcing:
    """AIFS source arrays normalized to met_em names on the native AIFS grid.

    Arrays use Python order ``(south_north, west_east)`` for 2D fields and
    ``(level, south_north, west_east)`` for 3D/soil stacks. Horizontal
    interpolation and target-grid staggering are S3 responsibilities.
    """

    source_path: Path
    source_sha256: str
    valid_time: str
    source_grid: AIFSGrid
    isobaric_levels_pa: tuple[float, ...]
    soil_layer_order_cm: tuple[float, ...]
    surface_spechumd_policy: SurfaceSpechumdPolicy
    arrays: dict[str, np.ndarray]
    provenance: dict[str, object]

    def validate(self) -> None:
        ny, nx = self.source_grid.shape
        for name in ("TT", "UU", "VV", "GHT", "SPECHUMD", "PRES"):
            _require_shape(name, self.arrays[name], (1 + len(self.isobaric_levels_pa), ny, nx))
        for name in (
            "PSFC",
            "PMSL",
            "SOILHGT",
            "SKINTEMP",
            "LANDSEA",
            "DEWPT",
            "ST000010",
            "ST010040",
            "SM000010",
            "SM010040",
        ):
            _require_shape(name, self.arrays[name], (ny, nx))
        for name in ("ST", "SM", "SOIL_LAYERS"):
            _require_shape(name, self.arrays[name], (2, ny, nx))
        if self.isobaric_levels_pa != tuple(ISOBARIC_LEVELS_PA):
            raise ValueError(f"isobaric level order drifted: {self.isobaric_levels_pa}")
        if self.soil_layer_order_cm != METEM_SOIL_LAYER_ORDER_CM:
            raise ValueError(f"soil layer order drifted: {self.soil_layer_order_cm}")


def decode_forcing(
    grib_path: str | Path,
    *,
    vtable_path: str | Path = DEFAULT_AIFS_VTABLE,
    surface_spechumd_policy: SurfaceSpechumdPolicy = "wps_fill",
) -> DecodedForcing:
    """Decode and normalize one AIFS GRIB2 step.

    ``surface_spechumd_policy="wps_fill"`` is the oracle-faithful default:
    ``METGRID.TBL.ARW`` fills ``SPECHUMD(200100)`` with -1 for this AIFS Vtable,
    and the checked met_em oracle contains -1 at k=0. ``"dewpoint"`` implements
    the sprint-contract recipe for experimental use and records distinct
    provenance.
    """

    raw = read_aifs_grib(grib_path, vtable_path=vtable_path)
    ny, nx = raw.grid.shape
    nlevels = 1 + len(ISOBARIC_LEVELS_PA)

    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, object] = {
        "source_path": str(raw.path),
        "source_sha256": raw.sha256,
        "valid_time": raw.valid_time,
        "vtable_path": str(raw.vtable_path),
        "source_grid": {
            "grid_type": raw.grid.grid_type,
            "shape": list(raw.grid.shape),
            "latitude_first": raw.grid.latitude_first,
            "longitude_first": raw.grid.longitude_first,
            "latitude_last": raw.grid.latitude_last,
            "longitude_last": raw.grid.longitude_last,
            "di": raw.grid.di,
            "dj": raw.grid.dj,
            "j_scans_positively": raw.grid.j_scans_positively,
            "i_scans_negatively": raw.grid.i_scans_negatively,
        },
        "surface_assembly": surface_assembly_recipe(surface_spechumd_policy),
        "fields": {},
    }

    # Direct surface/native fields.
    tt_2m = _surface_height(raw, "TT", 2.0)
    uu_10m = _surface_height(raw, "UU", 10.0)
    vv_10m = _surface_height(raw, "VV", 10.0)
    dewpt = _surface_height(raw, "DEWPT", 2.0)
    psfc = raw.get("PSFC", wps_level=WPS_SURFACE_LEVEL)
    pmsl = raw.get("PMSL", wps_level=WPS_MEAN_SEA_LEVEL)
    landsea = raw.get("LANDSEA", wps_level=WPS_SURFACE_LEVEL)
    skintemp = raw.get("SKINTEMP", wps_level=WPS_SURFACE_LEVEL)
    soilhgt = raw.get("SOILHGT", wps_level=WPS_SURFACE_LEVEL)
    st000010 = raw.get("ST000010", soil_depth_mm=(0, 1000))
    st010040 = raw.get("ST010040", soil_depth_mm=(1000, 4000))
    sm000010 = raw.get("SM000010", soil_depth_mm=(0, 1000))
    sm010040 = raw.get("SM010040", soil_depth_mm=(1000, 4000))

    arrays["PSFC"] = psfc.data.copy()
    arrays["PMSL"] = pmsl.data.copy()
    arrays["LANDSEA"] = landsea_flag_from_fraction(landsea.data)
    arrays["SKINTEMP"] = skintemp.data.copy()
    arrays["SOILHGT"] = soilhgt.data.copy()
    arrays["DEWPT"] = dewpt.data.copy()
    arrays["ST000010"] = st000010.data.copy()
    arrays["ST010040"] = st010040.data.copy()
    arrays["SM000010"] = sm000010.data.copy()
    arrays["SM010040"] = sm010040.data.copy()

    arrays["ST"] = np.stack([st010040.data, st000010.data], axis=0).astype(np.float32, copy=True)
    arrays["SM"] = np.stack([sm010040.data, sm000010.data], axis=0).astype(np.float32, copy=True)
    soil_layers = np.empty((2, ny, nx), dtype=np.float32)
    soil_layers[0, :, :] = METEM_SOIL_LAYER_ORDER_CM[0]
    soil_layers[1, :, :] = METEM_SOIL_LAYER_ORDER_CM[1]
    arrays["SOIL_LAYERS"] = soil_layers

    # 14-level atmosphere: index 0 is the metgrid surface level; 1..13 are
    # isobaric levels in met_em order 1000 -> 50 hPa.
    arrays["TT"] = _assemble_14(raw, "TT", tt_2m.data)
    arrays["UU"] = _assemble_14(raw, "UU", uu_10m.data)
    arrays["VV"] = _assemble_14(raw, "VV", vv_10m.data)
    arrays["GHT"] = _assemble_14(raw, "GHT", soilhgt.data)
    arrays["SPECHUMD"] = _assemble_surface_spechumd(raw, dewpt.data, psfc.data, surface_spechumd_policy)

    pres = np.empty((nlevels, ny, nx), dtype=np.float32)
    pres[0] = psfc.data
    for idx, level in enumerate(ISOBARIC_LEVELS_PA, start=1):
        pres[idx, :, :] = np.float32(level)
    arrays["PRES"] = pres

    field_provenance = provenance["fields"]
    assert isinstance(field_provenance, dict)
    for name, arr in arrays.items():
        field_provenance[name] = _field_provenance(name, arr, raw, surface_spechumd_policy)

    decoded = DecodedForcing(
        source_path=raw.path,
        source_sha256=raw.sha256,
        valid_time=raw.valid_time,
        source_grid=raw.grid,
        isobaric_levels_pa=tuple(ISOBARIC_LEVELS_PA),
        soil_layer_order_cm=METEM_SOIL_LAYER_ORDER_CM,
        surface_spechumd_policy=surface_spechumd_policy,
        arrays=arrays,
        provenance=provenance,
    )
    decoded.validate()
    return decoded


def specific_humidity_from_dewpoint(dewpoint_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """Specific humidity from dewpoint and pressure using WRF/WPS saturation constants.

    The vapor pressure is saturation vapor pressure at the dewpoint. The returned
    value is true specific humidity, so ``real.exe`` can convert it to vapor
    mixing ratio with ``qv = sh / (1 - sh)`` when ``FLAG_SH=1``.
    """

    td = np.asarray(dewpoint_k, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    vapor_pressure = SVP1_PA * np.exp(SVP2 * (td - SVPT0_K) / (td - SVP3))
    if np.any(vapor_pressure >= p):
        raise ValueError("dewpoint vapor pressure exceeds total pressure")
    q = EPSILON * vapor_pressure / (p - (1.0 - EPSILON) * vapor_pressure)
    return q.astype(np.float32)


def landsea_flag_from_fraction(landsea_fraction: np.ndarray) -> np.ndarray:
    """Match WPS rrpr.F ``make_zero_or_one`` for ECMWF LANDSEA fractions."""

    return np.where(np.asarray(landsea_fraction) > 0.5, np.float32(1.0), np.float32(0.0)).astype(np.float32)


def surface_assembly_recipe(policy: SurfaceSpechumdPolicy = "wps_fill") -> dict[str, object]:
    spechumd_surface: dict[str, object]
    if policy == "wps_fill":
        spechumd_surface = {
            "policy": "wps_fill",
            "assignment": "SPECHUMD[0] = -1.0",
            "rationale": (
                "Oracle met_em files have SPECHUMD k=0 equal to -1 everywhere; "
                "METGRID.TBL.ARW fills level 200100 with const(-1) for SPECHUMD."
            ),
            "source_refs": [
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/METGRID.TBL.ARW:691",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/METGRID.TBL.ARW:695",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:1144",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:1163",
            ],
        }
    elif policy == "dewpoint":
        spechumd_surface = {
            "policy": "dewpoint",
            "assignment": "SPECHUMD[0] = q_specific(DEWPT, PSFC)",
            "formula": (
                "e = 611.2 * exp(17.67 * (Td - 273.15) / (Td - 29.65)); "
                "q = 0.622 * e / (p - (1 - 0.622) * e)"
            ),
            "source_refs": [
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:1163",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:1167",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:7415",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:7421",
            ],
        }
    else:
        raise ValueError(f"unknown surface SPECHUMD policy {policy!r}")

    return {
        "TT[0]": "2t at heightAboveGround 2 m",
        "UU[0]": "10u at heightAboveGround 10 m",
        "VV[0]": "10v at heightAboveGround 10 m",
        "GHT[0]": "orog/SOILHGT at surface, already meters",
        "PRES[0]": "PSFC at surface",
        "SPECHUMD[0]": spechumd_surface,
        "isobaric_order": "indices 1..13 = 100000..5000 Pa",
        "wps_level_refs": [
            "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/ungrib/src/rd_grib2.F:755",
            "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/ungrib/src/rd_grib2.F:792",
            "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/src/process_domain_module.F:1070",
            "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/src/process_domain_module.F:1172",
        ],
        "soil_order": {
            "ST[0]/SM[0]": "010040, met_em SOIL_LAYERS[0] = 40 cm",
            "ST[1]/SM[1]": "000010, met_em SOIL_LAYERS[1] = 10 cm",
            "source_refs": [
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/METGRID.TBL.ARW:2",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/METGRID.TBL.ARW:29",
            ],
        },
    }


def field_ranges(decoded: DecodedForcing) -> dict[str, dict[str, object]]:
    ranges: dict[str, dict[str, object]] = {}
    for name, arr in decoded.arrays.items():
        finite = np.asarray(arr)[np.isfinite(arr)]
        ranges[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
        if arr.ndim == 3:
            ranges[name]["per_level_minmax"] = [
                [float(np.nanmin(arr[level])), float(np.nanmax(arr[level]))] for level in range(arr.shape[0])
            ]
    return ranges


def physical_range_checks(decoded: DecodedForcing) -> dict[str, dict[str, object]]:
    arrays = decoded.arrays
    checks = {
        "TT_all_180_330_K": _range_check(arrays["TT"], 180.0, 330.0),
        "PSFC_50000_105000_Pa": _range_check(arrays["PSFC"], 50000.0, 105000.0),
        "UU_abs_le_120_mps": _abs_check(arrays["UU"], 120.0),
        "VV_abs_le_120_mps": _abs_check(arrays["VV"], 120.0),
        "ST_220_330_K": _range_check(arrays["ST"], 220.0, 330.0),
        "SM_0_1_fraction": _range_check(arrays["SM"], 0.0, 1.0),
    }
    if decoded.surface_spechumd_policy == "wps_fill":
        checks["SPECHUMD_surface_wps_sentinel"] = {
            "pass": bool(np.all(arrays["SPECHUMD"][0] == SURFACE_SPECHUMD_WPS_FILL)),
            "expected": float(SURFACE_SPECHUMD_WPS_FILL),
        }
        checks["SPECHUMD_isobaric_0_0p03_kgkg"] = _range_check(arrays["SPECHUMD"][1:], 0.0, 0.03)
    else:
        checks["SPECHUMD_all_0_0p03_kgkg"] = _range_check(arrays["SPECHUMD"], 0.0, 0.03)
    return checks


def compare_to_wps_intermediate(
    decoded: DecodedForcing,
    intermediate_path: str | Path,
    *,
    tolerance_abs: float = 0.0,
) -> dict[str, object]:
    """Compare every directly decoded field to the ungrib WPS intermediate oracle."""

    records = read_wps_intermediate(intermediate_path)
    index: dict[tuple[str, float], object] = {(rec.field_name, rec.level): rec for rec in records}
    comparisons = []

    def add(field: str, level: float, actual: np.ndarray, *, wps_field: str | None = None) -> None:
        key = (wps_field or ("HGT" if field == "GHT" else field), float(level))
        if key not in index:
            raise KeyError(f"WPS intermediate missing {key}")
        rec = index[key]
        expected = rec.data
        diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        max_abs = float(np.max(diff))
        comparisons.append(
            {
                "field": field,
                "wps_field": key[0],
                "level": float(level),
                "max_abs": max_abs,
                "tolerance_abs": tolerance_abs,
                "bit_equal": bool(np.array_equal(actual.astype(np.float32), expected.astype(np.float32))),
                "pass": bool(max_abs <= tolerance_abs),
            }
        )

    for name in ("TT", "UU", "VV"):
        add(name, WPS_SURFACE_LEVEL, decoded.arrays[name][0])
        for idx, level in enumerate(ISOBARIC_LEVELS_PA, start=1):
            add(name, level, decoded.arrays[name][idx])
    for idx, level in enumerate(ISOBARIC_LEVELS_PA, start=1):
        add("GHT", level, decoded.arrays["GHT"][idx], wps_field="HGT")
        add("SPECHUMD", level, decoded.arrays["SPECHUMD"][idx])
    for name in ("PSFC", "LANDSEA", "SKINTEMP", "SOILHGT", "DEWPT", "ST000010", "ST010040", "SM000010", "SM010040"):
        add(name, WPS_SURFACE_LEVEL, decoded.arrays[name])
    add("PMSL", WPS_MEAN_SEA_LEVEL, decoded.arrays["PMSL"])

    return {
        "intermediate_path": str(intermediate_path),
        "record_count": len(records),
        "tolerance_abs": tolerance_abs,
        "comparisons": comparisons,
        "all_pass": bool(all(item["pass"] for item in comparisons)),
        "all_bit_equal": bool(all(item["bit_equal"] for item in comparisons)),
    }


def npz_roundtrip_bit_stable(decoded: DecodedForcing) -> dict[str, object]:
    """Round-trip the decoded container arrays through NPZ and verify bit stability."""

    buf = io.BytesIO()
    np.savez(buf, **decoded.arrays)
    buf.seek(0)
    out: dict[str, bool] = {}
    with np.load(buf) as loaded:
        for name, arr in decoded.arrays.items():
            out[name] = bool(np.array_equal(arr, loaded[name]))
    return {"all_bit_equal": bool(all(out.values())), "fields": out}


def build_forcing_decode_report(
    grib_path: str | Path,
    *,
    intermediate_path: str | Path | None = None,
    vtable_path: str | Path = DEFAULT_AIFS_VTABLE,
    surface_spechumd_policy: SurfaceSpechumdPolicy = "wps_fill",
) -> dict[str, object]:
    decoded = decode_forcing(
        grib_path,
        vtable_path=vtable_path,
        surface_spechumd_policy=surface_spechumd_policy,
    )
    dewpoint_surface_q = specific_humidity_from_dewpoint(decoded.arrays["DEWPT"], decoded.arrays["PSFC"])
    report: dict[str, object] = {
        "objective": "v0.3.0 S1 native AIFS GRIB2 forcing decode/normalize",
        "surface_spechumd_policy": surface_spechumd_policy,
        "source_path": str(decoded.source_path),
        "source_sha256": decoded.source_sha256,
        "valid_time": decoded.valid_time,
        "isobaric_levels_pa": list(decoded.isobaric_levels_pa),
        "soil_layer_order_cm": list(decoded.soil_layer_order_cm),
        "array_ranges": field_ranges(decoded),
        "physical_range_checks": physical_range_checks(decoded),
        "npz_roundtrip": npz_roundtrip_bit_stable(decoded),
        "surface_assembly": surface_assembly_recipe(surface_spechumd_policy),
        "dewpoint_specific_humidity_subcheck": {
            "formula": "q = eps*e/(p-(1-eps)*e), e=611.2*exp(17.67*(Td-273.15)/(Td-29.65))",
            "source_refs": [
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:1163",
                "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/dyn_em/module_initialize_real.F:1167",
            ],
            "min": float(np.min(dewpoint_surface_q)),
            "max": float(np.max(dewpoint_surface_q)),
            "sample_0_0": float(dewpoint_surface_q[0, 0]),
            "tolerance_abs": 1.0e-6,
        },
        "provenance": decoded.provenance,
        "oracle_findings": {
            "surface_spechumd": (
                "Checked met_em oracle has SPECHUMD[0] == -1 everywhere for the sampled case; "
                "this follows METGRID.TBL.ARW fill_lev=200100:const(-1)."
            ),
            "soil_stack_order": (
                "Checked met_em oracle stores ST/SM layer 0 as 010040 and layer 1 as 000010, "
                "with SOIL_LAYERS [40, 10]."
            ),
            "global_range_sanity": (
                "The raw AIFS source is a global grid. The sampled GRIB and matching ungrib "
                "intermediate contain PSFC below 50000 Pa and ST below 220 K at global "
                "points, so those contract sanity ranges are recorded but not used as the "
                "oracle-fidelity verdict."
            ),
        },
        "contract_discrepancies": [
            {
                "item": "SPECHUMD surface level",
                "contract": "derive SPECHUMD[0] from 2 m dewpoint and PSFC",
                "oracle": "met_em SPECHUMD[0] is -1 everywhere; METGRID.TBL fills const(-1)",
                "implementation": "default policy follows oracle; optional 'dewpoint' policy is implemented and tested",
            },
            {
                "item": "SOIL_LAYER_DEPTHS_CM",
                "contract_schema_comment": "schema constant lists (10, 40) while noting met_em SOIL_LAYERS=[40,10]",
                "oracle": "met_em SOIL_LAYERS values are [40, 10] and ST/SM stack is [010040, 000010]",
                "implementation": "DecodedForcing uses [40, 10] for oracle parity",
            },
        ],
    }
    if intermediate_path is not None:
        report["wps_intermediate_compare"] = compare_to_wps_intermediate(decoded, intermediate_path)
    report["strict_contract_physical_ranges_pass"] = bool(
        all(item["pass"] for item in physical_range_checks(decoded).values())
    )
    report["oracle_fidelity_pass"] = bool(
        report["npz_roundtrip"]["all_bit_equal"]  # type: ignore[index]
        and (
            "wps_intermediate_compare" not in report
            or report["wps_intermediate_compare"]["all_pass"]  # type: ignore[index]
        )
    )
    report["overall_pass"] = report["oracle_fidelity_pass"]
    return report


def write_forcing_decode_report(
    output_path: str | Path,
    grib_path: str | Path,
    *,
    intermediate_path: str | Path | None = None,
    vtable_path: str | Path = DEFAULT_AIFS_VTABLE,
    surface_spechumd_policy: SurfaceSpechumdPolicy = "wps_fill",
) -> dict[str, object]:
    report = build_forcing_decode_report(
        grib_path,
        intermediate_path=intermediate_path,
        vtable_path=vtable_path,
        surface_spechumd_policy=surface_spechumd_policy,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _surface_height(raw: AIFSGribFile, field_name: str, height_m: float) -> AIFSMessage:
    return raw.get(field_name, wps_level=WPS_SURFACE_LEVEL, height_m=height_m)


def _assemble_14(raw: AIFSGribFile, field_name: str, surface: np.ndarray) -> np.ndarray:
    isobaric = raw.stack_isobaric(field_name)
    out = np.empty((1 + len(ISOBARIC_LEVELS_PA), *surface.shape), dtype=np.float32)
    out[0] = surface
    out[1:] = isobaric
    return out


def _assemble_surface_spechumd(
    raw: AIFSGribFile,
    dewpt: np.ndarray,
    psfc: np.ndarray,
    policy: SurfaceSpechumdPolicy,
) -> np.ndarray:
    isobaric = raw.stack_isobaric("SPECHUMD")
    out = np.empty((1 + len(ISOBARIC_LEVELS_PA), *dewpt.shape), dtype=np.float32)
    if policy == "wps_fill":
        out[0, :, :] = SURFACE_SPECHUMD_WPS_FILL
    elif policy == "dewpoint":
        out[0] = specific_humidity_from_dewpoint(dewpt, psfc)
    else:
        raise ValueError(f"unknown surface SPECHUMD policy {policy!r}")
    out[1:] = isobaric
    return out


def _field_provenance(
    name: str,
    arr: np.ndarray,
    raw: AIFSGribFile,
    surface_policy: SurfaceSpechumdPolicy,
) -> dict[str, object]:
    base = {
        "source_path": str(raw.path),
        "grib_sha256": raw.sha256,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if name == "PRES":
        return {
            **base,
            "source": "derived",
            "recipe": "PRES[0]=PSFC; PRES[1..13]=isobaric level constants",
        }
    if name == "SOIL_LAYERS":
        return {
            **base,
            "source": "derived",
            "recipe": "constant met_em soil layer order [40, 10] cm",
        }
    if name in ("ST", "SM"):
        prefix = "ST" if name == "ST" else "SM"
        return {
            **base,
            "source": "derived_stack",
            "recipe": f"{name}[0]={prefix}010040; {name}[1]={prefix}000010",
            "layers": [
                raw.get(f"{prefix}010040", soil_depth_mm=(1000, 4000)).provenance,
                raw.get(f"{prefix}000010", soil_depth_mm=(0, 1000)).provenance,
            ],
        }
    if name in ("TT", "UU", "VV", "GHT"):
        surface_msg = {
            "TT": raw.get("TT", height_m=2.0),
            "UU": raw.get("UU", height_m=10.0),
            "VV": raw.get("VV", height_m=10.0),
            "GHT": raw.get("SOILHGT", wps_level=WPS_SURFACE_LEVEL),
        }[name]
        return {
            **base,
            "source": "aifs_grib",
            "surface": surface_msg.provenance,
            "isobaric": [raw.get(name, level_pa=level).provenance for level in ISOBARIC_LEVELS_PA],
        }
    if name == "SPECHUMD":
        surface = (
            {
                "source": "METGRID.TBL fill",
                "value": float(SURFACE_SPECHUMD_WPS_FILL),
                "policy": surface_policy,
            }
            if surface_policy == "wps_fill"
            else {
                "source": "derived",
                "policy": surface_policy,
                "recipe": "specific_humidity_from_dewpoint(DEWPT, PSFC)",
                "dewpt": raw.get("DEWPT", height_m=2.0).provenance,
                "psfc": raw.get("PSFC", wps_level=WPS_SURFACE_LEVEL).provenance,
            }
        )
        return {
            **base,
            "source": "aifs_grib+surface_policy",
            "surface": surface,
            "isobaric": [raw.get("SPECHUMD", level_pa=level).provenance for level in ISOBARIC_LEVELS_PA],
        }
    if name in ("ST000010", "SM000010"):
        return {**base, "source": "aifs_grib", "message": raw.get(name, soil_depth_mm=(0, 1000)).provenance}
    if name in ("ST010040", "SM010040"):
        return {**base, "source": "aifs_grib", "message": raw.get(name, soil_depth_mm=(1000, 4000)).provenance}
    if name == "DEWPT":
        return {**base, "source": "aifs_grib", "message": raw.get(name, height_m=2.0).provenance}
    if name in ("PSFC", "LANDSEA", "SKINTEMP", "SOILHGT"):
        message = raw.get(name, wps_level=WPS_SURFACE_LEVEL).provenance
        if name == "LANDSEA":
            return {
                **base,
                "source": "aifs_grib_normalized",
                "message": message,
                "recipe": "ECMWF fractional LANDSEA > 0.5 -> 1, else 0",
                "source_refs": [
                    "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/ungrib/src/rrpr.F:869",
                    "<USER_HOME>/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/ungrib/src/rrpr.F:978",
                ],
            }
        return {**base, "source": "aifs_grib", "message": message}
    if name == "PMSL":
        return {**base, "source": "aifs_grib", "message": raw.get(name, wps_level=WPS_MEAN_SEA_LEVEL).provenance}
    raise KeyError(f"no provenance recipe for {name}")


def _range_check(arr: np.ndarray, lower: float, upper: float) -> dict[str, object]:
    amin = float(np.nanmin(arr))
    amax = float(np.nanmax(arr))
    return {"pass": bool(amin >= lower and amax <= upper), "min": amin, "max": amax, "lower": lower, "upper": upper}


def _abs_check(arr: np.ndarray, upper_abs: float) -> dict[str, object]:
    max_abs = float(np.nanmax(np.abs(arr)))
    return {"pass": bool(max_abs <= upper_abs), "max_abs": max_abs, "upper_abs": upper_abs}


def _require_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if tuple(arr.shape) != expected:
        raise ValueError(f"{name} shape {tuple(arr.shape)} != {expected}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode one AIFS GRIB2 step and write the S1 proof report.")
    parser.add_argument("--grib", required=True, help="Path to step_NNN.grib2")
    parser.add_argument("--wps-intermediate", help="Path to matching AIFS:YYYY-MM-DD_HH intermediate")
    parser.add_argument("--vtable", default=str(DEFAULT_AIFS_VTABLE), help="Path to Vtable.AIFS_PURE")
    parser.add_argument("--output", required=True, help="JSON proof output")
    parser.add_argument(
        "--surface-spechumd-policy",
        choices=("wps_fill", "dewpoint"),
        default="wps_fill",
        help="Surface SPECHUMD assembly policy",
    )
    args = parser.parse_args(argv)
    report = write_forcing_decode_report(
        args.output,
        args.grib,
        intermediate_path=args.wps_intermediate,
        vtable_path=args.vtable,
        surface_spechumd_policy=args.surface_spechumd_policy,
    )
    print(json.dumps({"output": args.output, "overall_pass": report["overall_pass"]}, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
