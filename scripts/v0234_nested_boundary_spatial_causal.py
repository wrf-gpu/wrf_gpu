#!/usr/bin/env python3
"""CPU-only spatial causal decomposition for the v0.23.4 nested-boundary run.

This module intentionally imports only NumPy/netCDF4 model-independent readers.
It never imports JAX or gpuwrf and never touches the GPU.  The three 15:00 WRF
files are authenticated before any metric is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO_ROOT / ".agent/sprints/2026-07-13-v0234-nested-boundary-science-repair"
DEFAULT_OUTPUT = SPRINT_DIR / "spatial-causal-proof.json"

CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
RETRY20_ROOT = CASE_ROOT / "gpu_validation_retry20_relative_rmse_3ee02c19"
CANDIDATE_ROOT = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a/nested_frozen_bundle_e0d0b05a"

INPUTS = {
    "cpu": {
        "path": RETRY20_ROOT / "pair-snapshots/20250301T150000/cpu.nc",
        "sha256": "1ea00bd68bfa4098abd14d49e031389bae53bd0c921c80b2af264cd9dcbe2f0c",
        "role": "pristine WRF v4.7.1 d03 15:00 CPU oracle snapshot",
    },
    "retry20": {
        "path": RETRY20_ROOT / "pair-snapshots/20250301T150000/gpu.nc",
        "sha256": "df389ae9bfedaa3e49adf8da3aaafcfbaa0ee706525121373ffb3324e6c1c928",
        "role": "ordinary Retry20 d03 15:00 GPU snapshot",
    },
    "candidate": {
        "path": CANDIDATE_ROOT / "output/wrfout_d03_2025-03-01_15:00:00",
        "sha256": "ddef3d8fd6feb5807c988be20a447f5850de6d0b40af4ec0618ccf94d40ec0dd",
        "role": "nested frozen-WRF bundle d03 15:00 GPU snapshot",
    },
    "failure_proof": {
        "path": CANDIDATE_ROOT / "failure/failure-proof.json",
        "sha256": "10336c9ed866aa88d3e8df3f27e43da4166d5f769d8e25225f1deefa6dd377e4",
        "role": "atomic CPU_METRIC_NO_WORSE failure proof",
    },
    "last_healthy_carry": {
        "path": CANDIDATE_ROOT / "failure/last-healthy-d03-step-8800.pkl",
        "sha256": "ccaf8a462506755c3d557f09c304a77e7a2722170c22fa466349cff643bca9aa",
        "role": "retained complete d03 step-8800 carry",
    },
    "first_failed_carry": {
        "path": CANDIDATE_ROOT / "failure/first-failed-d03-step-9000.pkl",
        "sha256": "4d57a964e802407b584b24a54833816b69a304e165d99f6ffdeb9b8931805d69",
        "role": "retained complete finite d03 step-9000 carry",
    },
}

FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
REGRESSED_FIELDS = ("T", "U", "V", "W", "T2", "V10", "PSFC")
EXPECTED_RMSE = {
    "candidate_minus_cpu": {
        "T": 0.5809053918509451,
        "U": 1.2897885518304513,
        "V": 1.2354672080455205,
        "W": 0.18907232651547176,
        "T2": 1.3785549111350748,
        "U10": 1.934738422796919,
        "V10": 2.219842224410643,
        "PSFC": 32.13450997887015,
    },
    "retry20_minus_cpu": {
        "T": 0.5421680888949643,
        "U": 1.1323567330094144,
        "V": 1.1318203205639872,
        "W": 0.18633111790197587,
        "T2": 1.352612988238872,
        "U10": 1.9737860008971808,
        "V10": 2.1128268857679338,
        "PSFC": 20.22747532736003,
    },
}

DIRECT_FORCING_MAX_DISTANCE = 3
BUFFER_DISTANCE = 4
DISTANCE_BINS = (
    ("ring_0", 0, 0),
    ("ring_1", 1, 1),
    ("ring_2", 2, 2),
    ("ring_3", 3, 3),
    ("buffer_ring_4", 4, 4),
    ("shell_5_7", 5, 7),
    ("shell_8_15", 8, 15),
    ("shell_16_31", 16, 31),
    ("deep_interior_32_plus", 32, None),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_inputs() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in INPUTS.items():
        path = Path(spec["path"])
        actual = sha256_file(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"input authentication failed for {name}: {actual}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "bytes": path.stat().st_size,
            "role": spec["role"],
        }
    return result


def _read_time_zero(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    value = np.asarray(variable[:], dtype=np.float64)
    if variable.dimensions and variable.dimensions[0] == "Time":
        value = value[0]
    return value


def horizontal_distances(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    ny, nx = shape
    y, x = np.indices((ny, nx), dtype=np.int64)
    sides = {
        "south": y,
        "north": ny - 1 - y,
        "west": x,
        "east": nx - 1 - x,
    }
    sides["nearest"] = np.minimum.reduce(tuple(sides.values()))
    return sides


def distance_bin_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for label, lower, upper in DISTANCE_BINS:
        mask = distance >= lower
        if upper is not None:
            mask &= distance <= upper
        masks[label] = mask
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise AssertionError("distance bins must be exhaustive and non-overlapping")
    return masks


def topology_masks(distances: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    active = {
        side: distances[side] <= DIRECT_FORCING_MAX_DISTANCE
        for side in ("south", "north", "west", "east")
    }
    active_count = sum(mask.astype(np.int8) for mask in active.values())
    direct = active_count > 0
    masks = {
        "south_side": active["south"] & (active_count == 1),
        "north_side": active["north"] & (active_count == 1),
        "west_side": active["west"] & (active_count == 1),
        "east_side": active["east"] & (active_count == 1),
        "corner": active_count >= 2,
        "interior": ~direct,
    }
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise AssertionError("topology masks must be exhaustive and non-overlapping")
    return masks


def _stagger_mass_field(mass: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return mean/minmax spread of a mass-grid field on a target staggering."""

    ny, nx = mass.shape
    if shape == (ny, nx):
        return mass, np.zeros_like(mass)
    if shape == (ny, nx + 1):
        left = np.pad(mass, ((0, 0), (1, 0)), mode="edge")
        right = np.pad(mass, ((0, 0), (0, 1)), mode="edge")
        return 0.5 * (left + right), np.abs(left - right)
    if shape == (ny + 1, nx):
        south = np.pad(mass, ((1, 0), (0, 0)), mode="edge")
        north = np.pad(mass, ((0, 1), (0, 0)), mode="edge")
        return 0.5 * (south + north), np.abs(south - north)
    raise ValueError(f"unsupported target staggering {shape} from mass {mass.shape}")


def surface_type_masks(landmask: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    fraction, spread = _stagger_mass_field(landmask, shape)
    mixed = spread > 0.0
    masks = {
        "sea": (~mixed) & (fraction < 0.5),
        "land": (~mixed) & (fraction >= 0.5),
        "mixed_coast": mixed,
    }
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise AssertionError("surface masks must be exhaustive and non-overlapping")
    return masks


def forcing_footprint_masks(distance: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "direct_spec_relax_rings_0_3": distance <= DIRECT_FORCING_MAX_DISTANCE,
        "reserved_package_buffer_ring_4": distance == BUFFER_DISTANCE,
        "propagated_interior_distance_5_plus": distance > BUFFER_DISTANCE,
    }
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise AssertionError("forcing footprint masks must be exhaustive and non-overlapping")
    return masks


def _expand_mask(mask: np.ndarray, ndim: int) -> np.ndarray:
    if ndim == 2:
        return mask
    if ndim == 3:
        return np.broadcast_to(mask, (1, *mask.shape))
    raise ValueError(f"unsupported field rank {ndim}")


def metrics(delta: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    if mask is not None:
        expanded = _expand_mask(mask, values.ndim)
        if values.ndim == 3:
            expanded = np.broadcast_to(expanded, values.shape)
        values = values[expanded]
    else:
        values = values.ravel()
    finite = np.isfinite(values)
    if not np.all(finite):
        raise AssertionError("nonfinite value in retained finite 15:00 evidence")
    n = int(values.size)
    if n == 0:
        return {"n": 0, "sum_sq": 0.0, "rmse": None, "mae": None, "bias": None, "max_abs": None}
    sum_sq = float(np.dot(values, values))
    return {
        "n": n,
        "sum_sq": sum_sq,
        "rmse": float(np.sqrt(sum_sq / n)),
        "mae": float(np.mean(np.abs(values))),
        "bias": float(np.mean(values)),
        "max_abs": float(np.max(np.abs(values))),
    }


def decompose_delta(
    delta: np.ndarray,
    *,
    landmask: np.ndarray,
) -> dict[str, Any]:
    shape = delta.shape[-2:]
    distances = horizontal_distances(shape)
    nearest_bins = distance_bin_masks(distances["nearest"])
    topology = topology_masks(distances)
    surface = surface_type_masks(landmask, shape)
    footprint = forcing_footprint_masks(distances["nearest"])

    result: dict[str, Any] = {
        "overall": metrics(delta),
        "nearest_boundary_distance": {name: metrics(delta, mask) for name, mask in nearest_bins.items()},
        "distance_from_each_side": {},
        "side_corner_interior": {name: metrics(delta, mask) for name, mask in topology.items()},
        "land_sea": {name: metrics(delta, mask) for name, mask in surface.items()},
        "parent_forcing_footprint": {name: metrics(delta, mask) for name, mask in footprint.items()},
        "forcing_footprint_by_surface": {},
        "vertical_levels": [],
    }
    for side in ("south", "north", "west", "east"):
        result["distance_from_each_side"][side] = {
            name: metrics(delta, mask)
            for name, mask in distance_bin_masks(distances[side]).items()
        }
    for footprint_name, footprint_mask in footprint.items():
        result["forcing_footprint_by_surface"][footprint_name] = {
            surface_name: metrics(delta, footprint_mask & surface_mask)
            for surface_name, surface_mask in surface.items()
        }
    if delta.ndim == 3:
        result["vertical_levels"] = [
            {"level_index": level, **metrics(delta[level])}
            for level in range(delta.shape[0])
        ]
    return result


def _energy(mask: np.ndarray, values: np.ndarray) -> float:
    expanded = mask if values.ndim == 2 else np.broadcast_to(mask, values.shape)
    selected = values[expanded]
    return float(np.dot(selected, selected))


def classify_regression(candidate_cpu: np.ndarray, retry_cpu: np.ndarray) -> dict[str, Any]:
    distance = horizontal_distances(candidate_cpu.shape[-2:])["nearest"]
    direct = distance <= DIRECT_FORCING_MAX_DISTANCE
    direct_c = _energy(direct, candidate_cpu)
    direct_r = _energy(direct, retry_cpu)
    interior_c = _energy(~direct, candidate_cpu)
    interior_r = _energy(~direct, retry_cpu)
    direct_excess = direct_c - direct_r
    interior_excess = interior_c - interior_r
    total_excess = direct_excess + interior_excess
    if total_excess <= 0.0:
        verdict = "not_regressed"
    elif direct_excess > 0.5 * total_excess:
        verdict = "boundary_local"
    elif interior_excess > 0.8 * total_excess:
        verdict = "propagated_interior"
    else:
        verdict = "mixed_boundary_and_interior"
    return {
        "verdict": verdict,
        "direct_spec_relax_excess_sum_sq": float(direct_excess),
        "outside_direct_footprint_excess_sum_sq": float(interior_excess),
        "total_excess_sum_sq": float(total_excess),
        "outside_direct_fraction_of_positive_excess": (
            float(interior_excess / total_excess) if total_excess > 0.0 else None
        ),
    }


def build_proof() -> dict[str, Any]:
    authority = authenticate_inputs()
    paths = {name: Path(INPUTS[name]["path"]) for name in ("cpu", "retry20", "candidate")}
    datasets = {name: Dataset(path, "r") for name, path in paths.items()}
    try:
        geometry = {
            name: _read_time_zero(datasets["cpu"], name)
            for name in ("XLAT", "XLONG", "HGT", "LANDMASK")
        }
        fields: dict[str, Any] = {}
        classifications: dict[str, Any] = {}
        variable_manifest: dict[str, Any] = {}
        for field in FIELDS:
            arrays = {name: _read_time_zero(dataset, field) for name, dataset in datasets.items()}
            shapes = {name: list(array.shape) for name, array in arrays.items()}
            if len({tuple(shape) for shape in shapes.values()}) != 1:
                raise AssertionError(f"shape mismatch for {field}: {shapes}")
            variable = datasets["cpu"].variables[field]
            variable_manifest[field] = {
                "wrf_name": field,
                "units": getattr(variable, "units", "unknown"),
                "dimensions_without_time": list(variable.dimensions[1:]),
                "shape": list(arrays["cpu"].shape),
                "source_dtype": str(variable.dtype),
                "analysis_dtype": "float64",
                "staggering": (
                    "u" if "west_east_stag" in variable.dimensions
                    else "v" if "south_north_stag" in variable.dimensions
                    else "w" if "bottom_top_stag" in variable.dimensions
                    else "mass"
                ),
                "absolute_tolerance": None,
                "relative_tolerance": None,
                "gate": "predeclared RMSE no worse than Retry20; decomposition is diagnostic",
            }
            deltas = {
                "candidate_minus_cpu": arrays["candidate"] - arrays["cpu"],
                "retry20_minus_cpu": arrays["retry20"] - arrays["cpu"],
                "candidate_minus_retry20": arrays["candidate"] - arrays["retry20"],
            }
            field_result = {
                name: decompose_delta(delta, landmask=geometry["LANDMASK"])
                for name, delta in deltas.items()
            }
            for comparison in ("candidate_minus_cpu", "retry20_minus_cpu"):
                expected = EXPECTED_RMSE[comparison][field]
                actual = field_result[comparison]["overall"]["rmse"]
                if not np.isclose(actual, expected, rtol=0.0, atol=5.0e-12):
                    raise AssertionError(f"{field} {comparison} RMSE {actual} != {expected}")
            classification = classify_regression(
                deltas["candidate_minus_cpu"], deltas["retry20_minus_cpu"]
            )
            classifications[field] = classification
            field_result["spatial_regression_classification"] = classification
            fields[field] = field_result

        regressed_classes = {classifications[field]["verdict"] for field in REGRESSED_FIELDS}
        propagated_count = sum(
            classifications[field]["verdict"] == "propagated_interior"
            for field in REGRESSED_FIELDS
        )
        overall_class = (
            "propagated_interior_after_boundary_operator_change"
            if propagated_count >= 5
            else "mixed_spatial_regression"
        )
        return {
            "schema": "gpuwrf.v0234.nested-boundary-spatial-causal-proof.v1",
            "fixture_id": "tenerife-d03-20250228T18Z-valid-20250301T15Z-nested-bundle-failure",
            "source": "authenticated retained CPU/Retry20/candidate WRF snapshots",
            "source_commit": "e0d0b05a20b83502bbfe41e7808e9387168d7976",
            "pristine_wrf_commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
            "scenario": "Tenerife operational 3-domain d03, 15-hour valid time, 1-km child",
            "created_utc": "2026-07-13T00:00:00Z",
            "license_notes": "Large WRF/GPU artifacts remain outside git; only hashes and metrics are committed.",
            "authority": authority,
            "files": authority,
            "variables": variable_manifest,
            "geometry": {
                "mass_shape": list(geometry["LANDMASK"].shape),
                "spec_zone": 1,
                "relax_zone": 4,
                "spec_bdy_width": 5,
                "direct_parent_forcing_definition": "nearest boundary distance 0..3 (spec ring 0 plus relax rings 1..3)",
                "reserved_package_buffer_definition": "nearest boundary distance 4",
                "land_sea_staggering": "mass cells exact; U/V faces land or sea only when both adjacent mass cells agree, otherwise mixed_coast",
            },
            "fields": fields,
            "causal_verdict": {
                "spatial_class": overall_class,
                "regressed_field_classes": {
                    field: classifications[field]["verdict"] for field in REGRESSED_FIELDS
                },
                "distinct_classes_seen": sorted(regressed_classes),
                "boundary_local_rejected": propagated_count >= 5,
                "output_surface_algebra_primary_rejected": propagated_count >= 5,
                "reason": (
                    "The positive excess CPU-error energy is predominantly outside the direct rings 0..3 "
                    "for the regressed three-dimensional and surface fields. T2/V10/PSFC therefore follow "
                    "an interior dynamic response rather than an isolated writer/surface diagnostic defect."
                ),
                "v10_scope": "V10 remains separate from the step-9315 Ni mechanism; this proof links only its 15:00 drift to the broad boundary-driven trajectory change.",
            },
            "generation": {
                "command": "python scripts/v0234_nested_boundary_spatial_causal.py --output .agent/sprints/2026-07-13-v0234-nested-boundary-science-repair/spatial-causal-proof.json",
                "jax_imported": False,
                "gpu_commands_run": 0,
            },
        }
    finally:
        for dataset in datasets.values():
            dataset.close()


def write_proof(path: Path) -> dict[str, Any]:
    proof = build_proof()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    return proof


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    proof = write_proof(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "verdict": proof["causal_verdict"]["spatial_class"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
