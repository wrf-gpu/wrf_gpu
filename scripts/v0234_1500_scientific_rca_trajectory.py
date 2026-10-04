#!/usr/bin/env python3
"""CPU-only three-way trajectory/spatial proof for the v0.23.4 15:00 RCA.

The analysis authenticates every common retained d03 frame before reading it.
It imports neither JAX nor gpuwrf and never touches CUDA.  The material-change
rule below is intentionally source-frozen before this script is first executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset

from scripts import v0234_nested_boundary_spatial_causal as spatial


REPO_ROOT = Path(__file__).resolve().parents[1]
SPRINT_DIR = REPO_ROOT / ".agent/sprints/2026-07-14-v0234-1500-scientific-rca"
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
CPU_CONTRACT = CASE_ROOT / "terminal_cpu_authority_contract_v1.json"
RETRY_ROOT = CASE_ROOT / "gpu_validation_retry20_relative_rmse_3ee02c19"
CANDIDATE_ROOT = (
    CASE_ROOT
    / "corrected_ni_rca_max_22c2bd7a"
    / "nested_boundary_final_985f5714_full18h_gatefix1_replay1"
)

CPU_CONTRACT_SHA256 = "26b16e21782400308ec4d58175907796bf8aaddb01062bb06be8b87a2a88848a"
CANDIDATE_COMMIT = "985f5714b533d2c38da9c03ba305d8f8b3c59ed7"
CANDIDATE_TREE = "38046d951b505197a316f6523f1d0d085142987f"
RETRY20_COMMIT = "16774bed70b26d465e3aa87e91060e6a17de7f92"
RETRY20_TREE = "10247ab5b1e017d8b6af2ecb6865f0c7ea04efd1"
PRISTINE_WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
PRISTINE_WRF_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")

FROZEN_FILES = {
    "cpu_terminal_contract": (CPU_CONTRACT, CPU_CONTRACT_SHA256),
    "retry20_pairs": (RETRY_ROOT / "incremental-pairs.json", "9436da45de29518abc25c7fbe7f458f0323057f5768dc2b38b6e96e4713b4ffd"),
    "retry20_final_accept": (RETRY_ROOT / "retry20-final-accept.json", "72e9e6df6e23dc3889586afdc27b782de19aba54a8c7f62060baf3d2087648c8"),
    "retry20_cache_authority": (RETRY_ROOT / "retry20-cache-source-authority.json", "35b03ad00f6b652c60fbc2c72a9e1228c8823c95f87bb4c13299ba5817ecdb5c"),
    "retry20_runtime_attestation": (RETRY_ROOT / "gpu-proof/retry20-runtime-source-attestation.json", "fec1ea5e1be9ca6f272dbbf955a7cb96ade1710becedf87c6cc5b194a45e8a26"),
    "candidate_failure": (CANDIDATE_ROOT / "failure/failure-proof.json", "cd642149edc41eeebe09d3f7e0234e077ae7e86139245579f161715f1bb5a620"),
    "candidate_blocker": (CANDIDATE_ROOT / "full-run-blocker.json", "39bb8a511bac549858f9953eebf95fb6a02fbe6ca6e2cc6ca4ece079dd1bf4be"),
    "candidate_step9000_pair": (CANDIDATE_ROOT / "frame-pairs/d03-step-09000.json", "85c6487abaa07e735bea47564877e1277a386cc9559566ade9bf2cef55f40d3d"),
    "candidate_step8800_carry": (CANDIDATE_ROOT / "failure/last-healthy-d03-step-8800.pkl", "4bfe21f007c30e44102715495916651fd97aea3c8b5f81329886c1258872e786"),
    "candidate_step9000_carry": (CANDIDATE_ROOT / "failure/first-failed-d03-step-9000.pkl", "5f568bb92e9172156f525dbff6c4b4b595387460ab19a7d5aeacbd6650824a58"),
}

STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
FROZEN_1500_LIMITS = {
    "T": 0.5421680888949643,
    "U": 1.1323567330094144,
    "V": 1.1318203205639872,
    "W": 0.18633111790197587,
    "T2": 1.352612988238872,
    "U10": 1.9737860008971808,
    "V10": 2.1128268857679338,
    "PSFC": 20.22747532736003,
}

# Frozen before the first all-frame analysis.  This ranks onset only; it never
# replaces the exact 15:00 no-worse limits above.
MATERIAL_RULE = {
    "relative_fraction_of_frozen_1500_limit": 0.01,
    "consecutive_frames": 2,
    "change_definition": "candidate_minus_retry20_rmse >= 1% of the frozen Retry20 15:00 RMSE for two consecutive retained frames",
    "degradation_definition": "candidate_cpu_rmse - retry20_cpu_rmse >= 1% of the frozen Retry20 15:00 RMSE for two consecutive retained frames",
    "first_nonzero_definition": "candidate-minus-Retry20 max_abs > 0 in retained float32 output",
}

CANDIDATE_SOURCE_PATHS = (
    "src/gpuwrf/coupling/boundary_apply.py",
    "src/gpuwrf/nesting/__init__.py",
    "src/gpuwrf/nesting/boundary_construction.py",
    "src/gpuwrf/nesting/interp.py",
    "src/gpuwrf/nesting/moving_driver.py",
    "src/gpuwrf/runtime/domain_tree.py",
    "src/gpuwrf/runtime/operational_mode.py",
)
WRF_SOURCE_PATHS = (
    "share/mediation_force_domain.F",
    "share/interp_fcn.F",
    "share/sint.F",
    "share/module_bc.F",
    "dyn_em/couple_or_uncouple_em.F",
    "dyn_em/module_bc_em.F",
    "dyn_em/module_em.F",
    "dyn_em/solve_em.F",
    "dyn_em/module_small_step_em.F",
    "dyn_em/module_big_step_utilities_em.F",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_proof(path: Path, payload: dict[str, Any]) -> str:
    payload["proof_sha256"] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload["proof_sha256"]


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(("git", "-C", str(cwd), *args), text=True).strip()


def authenticate_fixed_inputs() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, (path, expected) in FROZEN_FILES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{name} SHA256 changed: {actual} != {expected}")
        files[name] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    if _git("rev-parse", f"{CANDIDATE_COMMIT}^{{tree}}") != CANDIDATE_TREE:
        raise RuntimeError("candidate tree changed")
    if _git("rev-parse", f"{RETRY20_COMMIT}^{{tree}}") != RETRY20_TREE:
        raise RuntimeError("Retry20 tree changed")
    if _git("rev-parse", PRISTINE_WRF_COMMIT, cwd=PRISTINE_WRF_ROOT) != PRISTINE_WRF_COMMIT:
        raise RuntimeError("pristine WRF commit unavailable")
    return files


def build_authority_proof() -> dict[str, Any]:
    files = authenticate_fixed_inputs()
    candidate_blobs = {
        path: _git("rev-parse", f"{CANDIDATE_COMMIT}:{path}")
        for path in CANDIDATE_SOURCE_PATHS
    }
    retry_blobs = {
        path: _git("rev-parse", f"{RETRY20_COMMIT}:{path}")
        for path in CANDIDATE_SOURCE_PATHS
    }
    wrf_blobs = {
        path: _git("rev-parse", f"{PRISTINE_WRF_COMMIT}:{path}", cwd=PRISTINE_WRF_ROOT)
        for path in WRF_SOURCE_PATHS
    }
    candidate_diff = subprocess.check_output(
        (
            "git", "-C", str(REPO_ROOT), "diff", "--binary",
            RETRY20_COMMIT, CANDIDATE_COMMIT, "--", "src/gpuwrf",
        )
    )
    return {
        "schema": "gpuwrf.v0234.1500-scientific-rca-authority.v1",
        "verdict": "AUTHORITY_GREEN",
        "frozen_files": files,
        "candidate": {
            "commit": CANDIDATE_COMMIT,
            "tree": CANDIDATE_TREE,
            "source_blobs": candidate_blobs,
        },
        "retry20": {
            "commit": RETRY20_COMMIT,
            "tree": RETRY20_TREE,
            "source_blobs": retry_blobs,
        },
        "candidate_model_diff_from_retry20_sha256": hashlib.sha256(candidate_diff).hexdigest(),
        "pristine_wrf": {
            "commit": PRISTINE_WRF_COMMIT,
            "tag": _git("describe", "--tags", "--exact-match", PRISTINE_WRF_COMMIT, cwd=PRISTINE_WRF_ROOT),
            "source_blobs": wrf_blobs,
            "read_policy": "git objects at pinned commit; dirty working-tree files are never read",
        },
        "frozen_1500_limits": FROZEN_1500_LIMITS,
        "material_rule": MATERIAL_RULE,
        "commands": [
            "python scripts/v0234_1500_scientific_rca_trajectory.py",
        ],
        "gpu_commands": 0,
        "jax_imported": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _verify_pair_proof(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("proof_sha256") != canonical_hash(payload):
        raise RuntimeError(f"candidate pair canonical proof changed: {path}")
    return payload


def discover_frames() -> list[dict[str, Any]]:
    retry = _read_json(RETRY_ROOT / "incremental-pairs.json")
    retry_rows = {row["valid_time"]: row for row in retry["pairs"]}
    rows: list[dict[str, Any]] = []
    for pair_path in sorted((CANDIDATE_ROOT / "frame-pairs").glob("d03-step-*.json")):
        candidate = _verify_pair_proof(pair_path)
        stamp = candidate["valid_time"]
        retry_row = retry_rows.get(stamp)
        if retry_row is None:
            continue
        rows.append(
            {
                "valid_time": stamp,
                "own_step": int(candidate["own_step"]),
                "pair_path": pair_path,
                "pair_proof_sha256": candidate["proof_sha256"],
                "candidate_path": Path(candidate["candidate"]["path"]),
                "candidate_sha256": candidate["candidate"]["sha256"],
                "cpu_path": Path(candidate["cpu"]["path"]),
                "cpu_sha256": candidate["cpu"]["sha256"],
                "retry_path": Path(retry_row["gpu_snapshot_path"]),
                "retry_sha256": retry_row["gpu_sha256"],
                "candidate_pair": candidate,
                "retry_pair": retry_row,
            }
        )
    if len(rows) != 46 or rows[0]["own_step"] != 0 or rows[-1]["own_step"] != 9000:
        raise RuntimeError(f"unexpected common retained frame set: {len(rows)}")
    return rows


def authenticate_frames(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    roles = ("candidate", "cpu", "retry")
    for row in rows:
        for role in roles:
            path = row[f"{role}_path"]
            expected = row[f"{role}_sha256"]
            key = (str(path), expected)
            if key in unique:
                continue
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"{role} frame SHA changed at {row['valid_time']}: {actual}")
            unique[key] = {
                "role": role,
                "path": str(path.resolve()),
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
    manifest_rows = sorted(unique.values(), key=lambda item: (item["role"], item["path"]))
    manifest_sha = hashlib.sha256(
        json.dumps(manifest_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "common_frame_count": len(rows),
        "authenticated_file_count": len(manifest_rows),
        "manifest_sha256": manifest_sha,
        "files": manifest_rows,
    }


def _read(dataset: Dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    value = np.asarray(variable[:], dtype=np.float64)
    if variable.dimensions and variable.dimensions[0] == "Time":
        value = value[0]
    return value


def _mask_metrics(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return spatial.metrics(values, mask)


def compact_decomposition(delta: np.ndarray, landmask: np.ndarray) -> dict[str, Any]:
    shape = delta.shape[-2:]
    distances = spatial.horizontal_distances(shape)
    nearest = distances["nearest"]
    distance_masks = {
        "direct_0_3": nearest <= 3,
        "buffer_4": nearest == 4,
        "shell_5_7": (nearest >= 5) & (nearest <= 7),
        "shell_8_15": (nearest >= 8) & (nearest <= 15),
        "shell_16_31": (nearest >= 16) & (nearest <= 31),
        "deep_32_plus": nearest >= 32,
    }
    coverage = sum(mask.astype(np.int8) for mask in distance_masks.values())
    if not np.all(coverage == 1):
        raise AssertionError("distance partition is not exact")
    topology = spatial.topology_masks(distances)
    surfaces = spatial.surface_type_masks(landmask, shape)
    result = {
        "overall": spatial.metrics(delta),
        "distance": {name: _mask_metrics(delta, mask) for name, mask in distance_masks.items()},
        "topology": {name: _mask_metrics(delta, mask) for name, mask in topology.items()},
        "surface": {name: _mask_metrics(delta, mask) for name, mask in surfaces.items()},
    }
    if delta.ndim == 3:
        result["vertical_rmse"] = [spatial.metrics(delta[level])["rmse"] for level in range(delta.shape[0])]
    else:
        result["vertical_rmse"] = []
    direct = result["distance"]["direct_0_3"]["sum_sq"]
    total = result["overall"]["sum_sq"]
    result["direct_0_3_energy_fraction"] = float(direct / total) if total > 0.0 else 0.0
    return result


def _staggered_geometry(mass: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return spatial._stagger_mass_field(mass, shape)[0]


def max_location(delta: np.ndarray, geometry: Mapping[str, np.ndarray]) -> dict[str, Any]:
    index = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    y, x = int(index[-2]), int(index[-1])
    shape = delta.shape[-2:]
    lat = _staggered_geometry(geometry["XLAT"], shape)
    lon = _staggered_geometry(geometry["XLONG"], shape)
    land = _staggered_geometry(geometry["LANDMASK"], shape)
    hgt = _staggered_geometry(geometry["HGT"], shape)
    distance = spatial.horizontal_distances(shape)["nearest"]
    return {
        "index": [int(value) for value in index],
        "lat": float(lat[y, x]),
        "lon": float(lon[y, x]),
        "land_fraction": float(land[y, x]),
        "hgt_m": float(hgt[y, x]),
        "nearest_boundary_distance_cells": int(distance[y, x]),
        "signed_delta": float(delta[index]),
        "max_abs": float(abs(delta[index])),
    }


def _area_weights(mapfac: np.ndarray) -> np.ndarray:
    return 1.0 / np.maximum(np.asarray(mapfac, dtype=np.float64), 1.0e-12) ** 2


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _pressure_mass_summary(arrays: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    derived: dict[str, dict[str, np.ndarray]] = {}
    integrals: dict[str, Any] = {}
    for role, fields in arrays.items():
        ptop = float(np.asarray(fields["P_TOP"]).reshape(-1)[0])
        dry_surface = fields["MU"] + fields["MUB"] + ptop
        psfc_excess = fields["PSFC"] - dry_surface
        total_pressure = fields["P"] + fields["PB"]
        derived[role] = {
            "dry_surface": dry_surface,
            "psfc_excess": psfc_excess,
            "total_pressure": total_pressure,
        }
        weights = _area_weights(fields["MAPFAC_M"])
        integrals[role] = {
            "area_weighted_dry_surface_pa": _weighted_mean(dry_surface, weights),
            "area_weighted_psfc_pa": _weighted_mean(fields["PSFC"], weights),
            "area_weighted_psfc_minus_dry_pa": _weighted_mean(psfc_excess, weights),
        }
    comparisons: dict[str, Any] = {}
    for name, left, right in (
        ("candidate_minus_cpu", "candidate", "cpu"),
        ("retry20_minus_cpu", "retry", "cpu"),
        ("candidate_minus_retry20", "candidate", "retry"),
    ):
        p3 = derived[left]["total_pressure"] - derived[right]["total_pressure"]
        comparisons[name] = {
            "dry_surface": spatial.metrics(derived[left]["dry_surface"] - derived[right]["dry_surface"]),
            "psfc_excess": spatial.metrics(derived[left]["psfc_excess"] - derived[right]["psfc_excess"]),
            "total_pressure": spatial.metrics(p3),
            "total_pressure_vertical_rmse": [spatial.metrics(p3[k])["rmse"] for k in range(p3.shape[0])],
        }
    return {"area_weighted_state": integrals, "comparisons": comparisons}


def _first_sustained(values: list[float], threshold: float, count: int) -> int | None:
    if count <= 0:
        raise ValueError("count must be positive")
    for start in range(0, len(values) - count + 1):
        if all(values[index] >= threshold for index in range(start, start + count)):
            return start
    return None


def build_trajectory_proof(frame_authority: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    variable_manifest: dict[str, Any] = {}
    geometry_ref: dict[str, np.ndarray] | None = None
    for row in rows:
        datasets = {
            role: Dataset(row[f"{role}_path"], "r")
            for role in ("candidate", "cpu", "retry")
        }
        try:
            geometry = {name: _read(datasets["cpu"], name) for name in ("XLAT", "XLONG", "HGT", "LANDMASK")}
            if geometry_ref is None:
                geometry_ref = geometry
            static_exact = {
                role: {
                    name: bool(np.array_equal(_read(datasets[role], name), geometry[name]))
                    for name in geometry
                }
                for role in ("candidate", "retry")
            }
            if not all(all(values.values()) for values in static_exact.values()):
                raise AssertionError(f"static geometry changed at {row['valid_time']}")
            fields: dict[str, Any] = {}
            pressure_arrays: dict[str, dict[str, np.ndarray]] = {role: {} for role in datasets}
            for field in STRICT_FIELDS:
                values = {role: _read(dataset, field) for role, dataset in datasets.items()}
                if len({value.shape for value in values.values()}) != 1:
                    raise AssertionError(f"shape mismatch {field} at {row['valid_time']}")
                if not all(np.all(np.isfinite(value)) for value in values.values()):
                    raise AssertionError(f"nonfinite {field} at {row['valid_time']}")
                if field not in variable_manifest:
                    variable = datasets["cpu"].variables[field]
                    variable_manifest[field] = {
                        "units": getattr(variable, "units", "unknown"),
                        "dimensions": list(variable.dimensions[1:]),
                        "shape": list(values["cpu"].shape),
                        "source_dtype": str(variable.dtype),
                        "analysis_dtype": "float64",
                        "staggering": (
                            "u" if "west_east_stag" in variable.dimensions
                            else "v" if "south_north_stag" in variable.dimensions
                            else "w" if "bottom_top_stag" in variable.dimensions
                            else "mass"
                        ),
                        "frozen_1500_maximum_rmse": FROZEN_1500_LIMITS[field],
                    }
                deltas = {
                    "candidate_minus_cpu": values["candidate"] - values["cpu"],
                    "retry20_minus_cpu": values["retry"] - values["cpu"],
                    "candidate_minus_retry20": values["candidate"] - values["retry"],
                }
                decomposition = {
                    name: compact_decomposition(delta, geometry["LANDMASK"])
                    for name, delta in deltas.items()
                }
                candidate_expected = row["candidate_pair"]["d03_full_pair"]["strict_rmse"][field]
                retry_expected = row["retry_pair"]["metrics"][field]["rmse"]
                candidate_actual = decomposition["candidate_minus_cpu"]["overall"]["rmse"]
                retry_actual = decomposition["retry20_minus_cpu"]["overall"]["rmse"]
                if not np.isclose(candidate_actual, candidate_expected, rtol=0.0, atol=5.0e-12):
                    raise AssertionError(f"candidate metric drift {field} at {row['valid_time']}")
                if not np.isclose(retry_actual, retry_expected, rtol=0.0, atol=5.0e-12):
                    raise AssertionError(f"Retry20 metric drift {field} at {row['valid_time']}")
                fields[field] = {
                    "comparisons": decomposition,
                    "candidate_cpu_minus_retry20_cpu_rmse": float(candidate_actual - retry_actual),
                    "candidate_minus_retry20_max_location": max_location(
                        deltas["candidate_minus_retry20"], geometry
                    ),
                }
                if field == "PSFC":
                    pressure_arrays["candidate"]["PSFC"] = values["candidate"]
                    pressure_arrays["cpu"]["PSFC"] = values["cpu"]
                    pressure_arrays["retry"]["PSFC"] = values["retry"]
            for name in ("MU", "MUB", "P_TOP", "P", "PB", "MAPFAC_M"):
                for role, dataset in datasets.items():
                    pressure_arrays[role][name] = _read(dataset, name)
            table.append(
                {
                    "valid_time": row["valid_time"],
                    "own_step": row["own_step"],
                    "candidate_pair_proof_sha256": row["pair_proof_sha256"],
                    "static_exact": static_exact,
                    "fields": fields,
                    "pressure_mass": _pressure_mass_summary(pressure_arrays),
                }
            )
        finally:
            for dataset in datasets.values():
                dataset.close()

    onset: dict[str, Any] = {}
    for field in STRICT_FIELDS:
        threshold = MATERIAL_RULE["relative_fraction_of_frozen_1500_limit"] * FROZEN_1500_LIMITS[field]
        change = [
            frame["fields"][field]["comparisons"]["candidate_minus_retry20"]["overall"]["rmse"]
            for frame in table
        ]
        degradation = [
            frame["fields"][field]["candidate_cpu_minus_retry20_cpu_rmse"]
            for frame in table
        ]
        max_abs = [
            frame["fields"][field]["comparisons"]["candidate_minus_retry20"]["overall"]["max_abs"]
            for frame in table
        ]
        count = int(MATERIAL_RULE["consecutive_frames"])
        change_index = _first_sustained(change, threshold, count)
        degradation_index = _first_sustained(degradation, threshold, count)
        nonzero_index = next((index for index, value in enumerate(max_abs) if value > 0.0), None)

        def bind(index: int | None) -> dict[str, Any] | None:
            if index is None:
                return None
            frame = table[index]
            comparison = frame["fields"][field]["comparisons"]["candidate_minus_retry20"]
            return {
                "valid_time": frame["valid_time"],
                "own_step": frame["own_step"],
                "candidate_minus_retry20_rmse": comparison["overall"]["rmse"],
                "candidate_cpu_minus_retry20_cpu_rmse": frame["fields"][field]["candidate_cpu_minus_retry20_cpu_rmse"],
                "direct_0_3_energy_fraction": comparison["direct_0_3_energy_fraction"],
                "max_location": frame["fields"][field]["candidate_minus_retry20_max_location"],
                "topology_rmse": {
                    name: metrics["rmse"] for name, metrics in comparison["topology"].items()
                },
                "distance_rmse": {
                    name: metrics["rmse"] for name, metrics in comparison["distance"].items()
                },
            }

        onset[field] = {
            "material_threshold": threshold,
            "first_nonzero": bind(nonzero_index),
            "first_material_change": bind(change_index),
            "first_material_degradation": bind(degradation_index),
        }

    material_times = [
        item["first_material_change"]["valid_time"]
        for item in onset.values()
        if item["first_material_change"] is not None
    ]
    degradation_times = [
        item["first_material_degradation"]["valid_time"]
        for item in onset.values()
        if item["first_material_degradation"] is not None
    ]
    assert geometry_ref is not None
    return {
        "schema": "gpuwrf.v0234.1500-trajectory-spatial-proof.v1",
        "verdict": "TRAJECTORY_RECONSTRUCTED",
        "source": "authenticated candidate, Retry20, and pristine WRF CPU d03 frames",
        "frame_authority": frame_authority,
        "material_rule": MATERIAL_RULE,
        "variables": variable_manifest,
        "geometry": {
            "mass_shape": list(geometry_ref["LANDMASK"].shape),
            "spec_zone": 1,
            "relax_zone": 4,
            "direct_boundary_definition": "nearest horizontal stagger-grid distance 0..3",
            "static_exact_all_frames": True,
        },
        "onset": onset,
        "earliest_material_change_time": min(material_times) if material_times else None,
        "earliest_material_degradation_time": min(degradation_times) if degradation_times else None,
        "frames": table,
        "v10_causal_policy": "strict metric reported independently; no shared V10/Ni cause inferred",
        "commands": ["python scripts/v0234_1500_scientific_rca_trajectory.py"],
        "gpu_commands": 0,
        "jax_imported": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-output", type=Path, default=SPRINT_DIR / "authority-proof.json")
    parser.add_argument("--trajectory-output", type=Path, default=SPRINT_DIR / "trajectory-spatial-proof.json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    authority = build_authority_proof()
    authority_sha = write_proof(args.authority_output, authority)
    rows = discover_frames()
    frame_authority = authenticate_frames(rows)
    trajectory = build_trajectory_proof(frame_authority, rows)
    trajectory["authority_proof_sha256"] = authority_sha
    trajectory_sha = write_proof(args.trajectory_output, trajectory)
    print(
        json.dumps(
            {
                "authority_proof_sha256": authority_sha,
                "trajectory_proof_sha256": trajectory_sha,
                "frames": len(rows),
                "earliest_material_change": trajectory["earliest_material_change_time"],
                "earliest_material_degradation": trajectory["earliest_material_degradation_time"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
