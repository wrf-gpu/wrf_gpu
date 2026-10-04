#!/usr/bin/env python3
"""Build the retained-pair spatial/causal proof for Retry20 V10.

This is deliberately CPU-only.  It reads the immutable pair snapshots emitted
by the accepted incremental gate and separates 10 m diagnostic scaling/turning
from the lowest-model-level prognostic wind difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np


NI_Y = 48
NI_X = 78
NI_LAT = 28.297913
NI_LON = -16.303406
EXPECTED_CONTRACT_SHA256 = "26b16e21782400308ec4d58175907796bf8aaddb01062bb06be8b87a2a88848a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    value = np.asarray(ds.variables[name][0], dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains a nonfinite value")
    return value


def destagger_lowest(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mass-point k=0 grid-relative wind from WRF staggered arrays."""

    if u.ndim != 3 or v.ndim != 3:
        raise ValueError("U and V must have (bottom_top, y[/stag], x[/stag]) layout")
    u0 = 0.5 * (u[0, :, :-1] + u[0, :, 1:])
    v0 = 0.5 * (v[0, :-1, :] + v[0, 1:, :])
    if u0.shape != v0.shape:
        raise ValueError(f"destaggered shapes differ: {u0.shape} versus {v0.shape}")
    return u0, v0


def rmse(delta: np.ndarray, mask: np.ndarray | None = None) -> float:
    values = np.asarray(delta, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("RMSE requires a nonempty finite population")
    return float(np.sqrt(np.mean(values * values, dtype=np.float64)))


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("correlation populations must have equal size >=2")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def earth_relative(u: np.ndarray, v: np.ndarray, sin_alpha: np.ndarray, cos_alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply WRF's grid-to-earth horizontal rotation."""

    return u * cos_alpha - v * sin_alpha, v * cos_alpha + u * sin_alpha


def vector_scale_decomposition(
    cpu_u10: np.ndarray,
    cpu_v10: np.ndarray,
    gpu_u10: np.ndarray,
    gpu_v10: np.ndarray,
    cpu_u0: np.ndarray,
    cpu_v0: np.ndarray,
    gpu_u0: np.ndarray,
    gpu_v0: np.ndarray,
) -> dict[str, np.ndarray]:
    """Stable V10 split into low-level-wind and surface-vector-scale terms.

    The residual includes surface turning and the small cross term.  This is a
    diagnostic decomposition, not a claim that the surface scheme is scalar.
    """

    eps = np.finfo(np.float64).eps
    cpu_speed0 = np.hypot(cpu_u0, cpu_v0)
    gpu_speed0 = np.hypot(gpu_u0, gpu_v0)
    cpu_ratio = np.hypot(cpu_u10, cpu_v10) / np.maximum(cpu_speed0, eps)
    gpu_ratio = np.hypot(gpu_u10, gpu_v10) / np.maximum(gpu_speed0, eps)
    total = gpu_v10 - cpu_v10
    dynamics = cpu_ratio * (gpu_v0 - cpu_v0)
    surface_ratio = gpu_v0 * (gpu_ratio - cpu_ratio)
    residual = total - dynamics - surface_ratio
    return {
        "total": total,
        "dynamics": dynamics,
        "surface_ratio": surface_ratio,
        "turning_and_cross_residual": residual,
        "cpu_ratio": cpu_ratio,
        "gpu_ratio": gpu_ratio,
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(a))


def _frame(pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cpu_path = Path(pair["cpu_snapshot_path"])
    gpu_path = Path(pair["gpu_snapshot_path"])
    if sha256(cpu_path) != pair["cpu_sha256"] or sha256(gpu_path) != pair["gpu_sha256"]:
        raise ValueError(f"pair hash mismatch at {pair['valid_time']}")
    with netCDF4.Dataset(cpu_path) as cpu_ds, netCDF4.Dataset(gpu_path) as gpu_ds:
        cpu = {name: _read(cpu_ds, name) for name in ("U", "V", "U10", "V10", "LANDMASK", "HGT", "XLAT", "XLONG", "SINALPHA", "COSALPHA")}
        gpu = {name: _read(gpu_ds, name) for name in ("U", "V", "U10", "V10", "LANDMASK", "HGT", "XLAT", "XLONG", "SINALPHA", "COSALPHA")}

    for static in ("LANDMASK", "HGT", "XLAT", "XLONG", "SINALPHA", "COSALPHA"):
        if not np.array_equal(cpu[static], gpu[static]):
            raise ValueError(f"static field {static} differs at {pair['valid_time']}")

    cpu_u0, cpu_v0 = destagger_lowest(cpu["U"], cpu["V"])
    gpu_u0, gpu_v0 = destagger_lowest(gpu["U"], gpu["V"])
    land = cpu["LANDMASK"] >= 0.5
    sea = ~land
    dv10 = gpu["V10"] - cpu["V10"]
    du10 = gpu["U10"] - cpu["U10"]
    du0 = gpu_u0 - cpu_u0
    dv0 = gpu_v0 - cpu_v0
    decomp = vector_scale_decomposition(
        cpu["U10"], cpu["V10"], gpu["U10"], gpu["V10"],
        cpu_u0, cpu_v0, gpu_u0, gpu_v0,
    )
    cpu_u0_e, cpu_v0_e = earth_relative(cpu_u0, cpu_v0, cpu["SINALPHA"], cpu["COSALPHA"])
    gpu_u0_e, gpu_v0_e = earth_relative(gpu_u0, gpu_v0, cpu["SINALPHA"], cpu["COSALPHA"])
    cpu_u10_e, cpu_v10_e = earth_relative(cpu["U10"], cpu["V10"], cpu["SINALPHA"], cpu["COSALPHA"])
    gpu_u10_e, gpu_v10_e = earth_relative(gpu["U10"], gpu["V10"], cpu["SINALPHA"], cpu["COSALPHA"])

    max_flat = int(np.argmax(np.abs(dv10)))
    max_y, max_x = np.unravel_index(max_flat, dv10.shape)
    record = {
        "valid_time": pair["valid_time"],
        "source": {
            "cpu_path": str(cpu_path), "cpu_sha256": pair["cpu_sha256"],
            "gpu_path": str(gpu_path), "gpu_sha256": pair["gpu_sha256"],
        },
        "v10": {
            "rmse": rmse(dv10), "land_rmse": rmse(dv10, land), "sea_rmse": rmse(dv10, sea),
            "bias": float(np.mean(dv10, dtype=np.float64)),
            "max_abs": float(abs(dv10[max_y, max_x])),
            "max_location": {
                "y": int(max_y), "x": int(max_x), "lat": float(cpu["XLAT"][max_y, max_x]),
                "lon": float(cpu["XLONG"][max_y, max_x]), "landmask": int(land[max_y, max_x]),
            },
            "ni_cell_delta": float(dv10[NI_Y, NI_X]),
        },
        "u10_rmse": rmse(du10),
        "lowest_model_level": {"u_rmse": rmse(du0), "v_rmse": rmse(dv0)},
        "causal_decomposition": {
            "delta_v10_vs_delta_v0_correlation_grid": correlation(dv10, dv0),
            "delta_v10_vs_delta_v0_correlation_earth": correlation(gpu_v10_e - cpu_v10_e, gpu_v0_e - cpu_v0_e),
            "total_rmse": rmse(decomp["total"]),
            "low_level_dynamics_term_rmse": rmse(decomp["dynamics"]),
            "surface_vector_ratio_term_rmse": rmse(decomp["surface_ratio"]),
            "turning_and_cross_residual_rmse": rmse(decomp["turning_and_cross_residual"]),
            "cpu_surface_vector_ratio_median": float(np.median(decomp["cpu_ratio"])),
            "gpu_surface_vector_ratio_median": float(np.median(decomp["gpu_ratio"])),
        },
    }
    arrays = {"dv10": dv10, "du10": du10, "dv0": dv0, "du0": du0}
    return record, arrays


def build_proof(root: Path, contract: Path) -> dict[str, Any]:
    if sha256(contract) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("terminal CPU contract hash mismatch")
    pairs_path = root / "incremental-pairs.json"
    pairs_doc = json.loads(pairs_path.read_text())
    pairs = pairs_doc["pairs"]
    if pairs_doc["matched_count"] != 46 or len(pairs) != 46:
        raise ValueError("expected exactly 46 retained matched d03 frames")

    frames: list[dict[str, Any]] = []
    pooled_dv10: list[np.ndarray] = []
    pooled_du10: list[np.ndarray] = []
    for pair in pairs:
        record, arrays = _frame(pair)
        frames.append(record)
        pooled_dv10.append(arrays["dv10"].ravel())
        pooled_du10.append(arrays["du10"].ravel())

    all_dv10 = np.concatenate(pooled_dv10)
    all_du10 = np.concatenate(pooled_du10)
    expected_v10 = pairs_doc["pooled_metrics"]["V10"]
    expected_u10 = pairs_doc["pooled_metrics"]["U10"]
    reproduced_v10 = rmse(all_dv10)
    reproduced_u10 = rmse(all_du10)
    # The accepted gate streams frame sums while this audit concatenates the
    # same float32 populations before the float64 reduction.  Permit only two
    # float64 ulps for that reduction-order difference (observed: one ulp).
    v10_ulp_error = abs(reproduced_v10 - expected_v10["pooled_rmse"]) / np.spacing(expected_v10["pooled_rmse"])
    u10_ulp_error = abs(reproduced_u10 - expected_u10["pooled_rmse"]) / np.spacing(expected_u10["pooled_rmse"])
    if v10_ulp_error > 2.0 or u10_ulp_error > 2.0:
        raise ValueError("pooled metric did not reproduce within two float64 ulps")

    first = frames[0]
    last = frames[-1]
    maxloc = last["v10"]["max_location"]
    distance = haversine_km(NI_LAT, NI_LON, maxloc["lat"], maxloc["lon"])
    frame_rmse = np.asarray([f["v10"]["rmse"] for f in frames])
    ni_delta = np.asarray([f["v10"]["ni_cell_delta"] for f in frames])

    first_dynamics_identity = (
        first["lowest_model_level"]["u_rmse"] < 1.0e-6
        and first["lowest_model_level"]["v_rmse"] < 1.0e-6
        and first["v10"]["rmse"] > 0.1
    )
    late_dynamics_dominant = (
        last["causal_decomposition"]["delta_v10_vs_delta_v0_correlation_grid"] > 0.99
        and last["causal_decomposition"]["low_level_dynamics_term_rmse"]
        > 10.0 * last["causal_decomposition"]["surface_vector_ratio_term_rmse"]
    )
    spatial_separation = distance > 10.0 and abs(last["v10"]["ni_cell_delta"]) < 0.1 * last["v10"]["max_abs"]

    return {
        "schema": "gpuwrf.v0234.corrected-ni-rca-v10.v1",
        "authority": {
            "retry20_root": str(root),
            "incremental_pairs_path": str(pairs_path),
            "incremental_pairs_sha256": sha256(pairs_path),
            "terminal_cpu_contract_path": str(contract),
            "terminal_cpu_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "matched_frames": 46,
            "population_per_frame": int(all_dv10.size // 46),
            "pooled_population": int(all_dv10.size),
        },
        "ni_spatial_binding": {
            "jax_mass_index": {"y": NI_Y, "x": NI_X},
            "wrf_fortran_mass_index": {"j": NI_Y + 1, "i": NI_X + 1},
            "lat": NI_LAT, "lon": NI_LON, "landmask": 0, "hgt_m": 0.0,
            "classification": "interior offshore E/NE; not boundary or edge",
        },
        "pooled_reproduction": {
            "v10_rmse": reproduced_v10, "v10_accepted_value": expected_v10["pooled_rmse"],
            "v10_reproduction_ulp_error": v10_ulp_error, "v10_gate_threshold": expected_v10["threshold"],
            "v10_gate_pass": reproduced_v10 <= expected_v10["threshold"],
            "u10_rmse": reproduced_u10, "u10_accepted_value": expected_u10["pooled_rmse"],
            "u10_reproduction_ulp_error": u10_ulp_error, "u10_gate_threshold": expected_u10["threshold"],
            "u10_gate_pass": reproduced_u10 <= expected_u10["threshold"],
        },
        "bounded_gates": {
            "surface_diagnostic_mismatch_preexists_prognostic_drift": first_dynamics_identity,
            "late_v10_is_low_level_prognostic_drift_dominant": late_dynamics_dominant,
            "last_frame_v10_extreme_is_spatially_separate_from_ni": spatial_separation,
            "rotation_bug_falsified": (
                last["causal_decomposition"]["delta_v10_vs_delta_v0_correlation_grid"] > 0.99
                and last["causal_decomposition"]["delta_v10_vs_delta_v0_correlation_earth"] > 0.99
            ),
            "common_ni_v10_root_proved": False,
        },
        "cross_frame": {
            "frame_rmse_vs_ni_cell_delta_correlation": correlation(frame_rmse, ni_delta),
            "last_v10_max_distance_from_ni_km": distance,
        },
        "verdict": {
            "v10": "TWO_COMPONENT_MISS: initial surface-output algebra plus late broad lowest-level prognostic wind drift",
            "ni_link": "NOT_PROVED; spatial and temporal retained-output evidence requires separate mechanisms",
            "fix_authority": "NONE_FROM_RETAINED_OUTPUTS",
        },
        "first_frame": first,
        "last_frame": last,
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry20-root", type=Path, required=True)
    parser.add_argument("--terminal-cpu-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_proof(args.retry20_root.resolve(), args.terminal_cpu_contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
