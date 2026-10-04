"""CPU-only attribution of the H5 Step200 PSFC scientific red.

Reads the authenticated CPU-WRF, pinned GPU reference, and H5 GPU Step200
frames.  It reconstructs WRF's moist hydrostatic PSFC diagnostic and separates
the dry-column-mass and moisture-column errors.  No model is imported, compiled,
or run; no GPU command/query or WRF/MPI execution occurs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "H5_STEP200_PSFC_ATTRIBUTION.json"
BLOCKER = SPRINT / "H5_STEP200_PSFC_ATTRIBUTION_BLOCKER.json"
RESULT = SPRINT / "GPU_STEP200_RESULT_fbb2716e855e4fe5.json"
RESULT_FILE_SHA256 = (
    "2b6ac8b3c0b3fef804363066f2d7a1c51dc41f94928119f88dd723955a642fc0"
)
RESULT_PROOF_SHA256 = (
    "5d58ff481fa869e348a5b955e66d57f4106fe3023d0a87449e554d70f04f479e"
)
CASE_ROOT = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
LINEAGE = CASE_ROOT / "corrected_ni_rca_max_22c2bd7a"
CPU = CASE_ROOT / "run/wrf/wrfout_d03_2025-03-01_00:20:00"
REFERENCE = (
    LINEAGE
    / "v0234_gpt_v10_replay_c17fca1e201ae106_reference/"
    "gpu-output/wrfout_d03_2025-03-01_00:20:00"
)
H5 = (
    LINEAGE
    / "v0234_gpt_v10_rootcause_fbb2716e855e4fe5_step200/"
    "gpu-output/wrfout_d03_2025-03-01_00:20:00"
)
FRAME_SHA256 = {
    "cpu": "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1",
    "reference": "441395eb1384e1515f4135d34b858dbd60881581711253ed2360a701dbbaa301",
    "h5": "2beaeafe9e12fdb745b04edac8c8b2b771d7c09fd077bc86b9ab6ee901d32ff1",
}
SOURCE_SHA256 = {
    "src/gpuwrf/integration/nested_pipeline.py": (
        "77b6a690647ec4b0b615f425338ca41f3e594c29566eeafc0f9e884e1f0e98c4"
    ),
    "src/gpuwrf/runtime/operational_mode.py": (
        "d5ce22236188c4264ab13e3108f163830d9c5bce1560b0090db1c927e8e1a19d"
    ),
}
SPECIES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _rms(np: Any, value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def _stats(np: Any, error: Any, mask: Any) -> dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)[mask]
    return {
        "cells": int(values.size),
        "rmse": _rms(np, values),
        "bias": float(np.mean(values, dtype=np.float64)),
        "mae": float(np.mean(np.abs(values), dtype=np.float64)),
        "max_abs": float(np.max(np.abs(values))),
    }


def main() -> int:
    stage = "startup"
    try:
        if sorted(os.sched_getaffinity(0)) != [13, 14, 15, 29, 30, 31]:
            raise RuntimeError("CPU affinity changed")
        if _sha256(RESULT) != RESULT_FILE_SHA256:
            raise RuntimeError("sealed H5 result file changed")
        result = json.loads(RESULT.read_text())
        if (
            result.get("proof_sha256") != RESULT_PROOF_SHA256
            or result.get("verdict") != "V10_H5_STEP200_SCIENTIFIC_RED"
            or result["score"]["decision"]["frozen_field_regressions"]
            != {
                "PSFC": {
                    "candidate": 13.065055589548484,
                    "reference": 12.354054650535346,
                }
            }
        ):
            raise RuntimeError("sealed H5 result semantics changed")
        paths = {"cpu": CPU, "reference": REFERENCE, "h5": H5}
        for name, path in paths.items():
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"{name} frame absent or symlinked")
            actual = _sha256(path)
            if actual != FRAME_SHA256[name]:
                raise RuntimeError(f"{name} frame hash changed: {actual}")
        for relative, expected in SOURCE_SHA256.items():
            if _sha256(ROOT / relative) != expected:
                raise RuntimeError(f"candidate source changed: {relative}")

        stage = "read_frames"
        import numpy as np
        from netCDF4 import Dataset

        variables = (
            STRICT_FIELDS
            + SPECIES
            + ("MU", "MUB", "C1H", "C2H", "DNW", "P_TOP", "LANDMASK", "HGT")
        )
        data: dict[str, dict[str, Any]] = {}
        for name, path in paths.items():
            with Dataset(path) as dataset:
                missing = [var for var in variables if var not in dataset.variables]
                if missing:
                    raise RuntimeError(f"{name} missing variables: {missing}")
                data[name] = {
                    var: np.asarray(dataset.variables[var][0], dtype=np.float64)
                    for var in variables
                }

        stage = "geometry"
        land = data["cpu"]["LANDMASK"] > 0.5
        hgt = data["cpu"]["HGT"]
        ny, nx = land.shape
        yy, xx = np.indices((ny, nx))
        ring = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
        gy, gx = np.gradient(hgt)
        slope = np.hypot(gx, gy)
        masks = {
            "all": np.ones_like(land, dtype=bool),
            "interior5": ring >= 5,
            "outer5": ring < 5,
            "land": land,
            "sea": ~land,
            "high_land_ge_1000m": land & (hgt >= 1000.0),
            "steep_land_top_quartile": land
            & (slope >= np.quantile(slope[land], 0.75)),
        }

        stage = "strict_metrics"
        strict = {}
        for arm in ("reference", "h5"):
            strict[arm] = {
                field: _rms(np, data[arm][field] - data["cpu"][field])
                for field in STRICT_FIELDS
            }
        recorded = result["score"]["decision"]
        for field in STRICT_FIELDS:
            if not math.isclose(
                strict["reference"][field],
                recorded["reference_rmse"][field],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ) or not math.isclose(
                strict["h5"][field],
                recorded["strict_rmse"][field],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(f"strict metric reproduction failed: {field}")
        strict_change = {
            field: {
                "reference_rmse": strict["reference"][field],
                "h5_rmse": strict["h5"][field],
                "h5_minus_reference": (
                    strict["h5"][field] - strict["reference"][field]
                ),
                "relative_change_percent": 100.0
                * (
                    strict["h5"][field] / strict["reference"][field] - 1.0
                ),
                "improved": strict["h5"][field] < strict["reference"][field],
            }
            for field in STRICT_FIELDS
        }

        stage = "psfc_reconstruction"
        c1h = data["cpu"]["C1H"]
        c2h = data["cpu"]["C2H"]
        dnw = data["cpu"]["DNW"]
        p_top = float(data["cpu"]["P_TOP"].reshape(-1)[0])
        for arm in ("reference", "h5"):
            for metric in ("C1H", "C2H", "DNW", "P_TOP"):
                if not np.array_equal(data[arm][metric], data["cpu"][metric]):
                    raise RuntimeError(f"hybrid metric changed: {arm}/{metric}")
        qtot = {
            arm: sum(
                (data[arm][species] for species in SPECIES),
                start=np.zeros_like(data[arm]["QVAPOR"]),
            )
            for arm in data
        }
        mut = {arm: data[arm]["MU"] + data[arm]["MUB"] for arm in data}
        component = {}
        reconstructed = {}
        for arm in data:
            dp_dry = (
                c1h[:, None, None] * mut[arm][None, :, :]
                + c2h[:, None, None]
            ) * (-dnw[:, None, None])
            dry = np.sum(dp_dry, axis=0, dtype=np.float64)
            moist = np.sum(qtot[arm] * dp_dry, axis=0, dtype=np.float64)
            component[arm] = {"dry": dry, "moist": moist, "dp_dry": dp_dry}
            reconstructed[arm] = p_top + dry + moist
        reconstruction_residual = {
            arm: {
                "rmse": _rms(np, reconstructed[arm] - data[arm]["PSFC"]),
                "max_abs": float(
                    np.max(np.abs(reconstructed[arm] - data[arm]["PSFC"]))
                ),
            }
            for arm in data
        }

        spatial = {}
        for mask_name, mask in masks.items():
            spatial[mask_name] = {}
            for arm in ("reference", "h5"):
                spatial[mask_name][arm] = _stats(
                    np, data[arm]["PSFC"] - data["cpu"]["PSFC"], mask
                )
            spatial[mask_name]["h5_minus_reference_rmse"] = (
                spatial[mask_name]["h5"]["rmse"]
                - spatial[mask_name]["reference"]["rmse"]
            )
            spatial[mask_name]["relative_change_percent"] = 100.0 * (
                spatial[mask_name]["h5"]["rmse"]
                / spatial[mask_name]["reference"]["rmse"]
                - 1.0
            )

        stage = "component_attribution"
        components = {}
        for arm in ("reference", "h5"):
            dry_error = component[arm]["dry"] - component["cpu"]["dry"]
            moist_error = component[arm]["moist"] - component["cpu"]["moist"]
            combined = dry_error + moist_error
            components[arm] = {
                "dry_column_error": _stats(np, dry_error, masks["all"]),
                "moist_column_error": _stats(np, moist_error, masks["all"]),
                "combined_component_error": _stats(np, combined, masks["all"]),
                "dry_moist_error_correlation": float(
                    np.corrcoef(dry_error.ravel(), moist_error.ravel())[0, 1]
                ),
                "twice_mean_cross_term_pa2": float(
                    np.mean(2.0 * dry_error * moist_error, dtype=np.float64)
                ),
                "mean_square_pa2": {
                    "dry": float(np.mean(dry_error * dry_error, dtype=np.float64)),
                    "moist": float(
                        np.mean(moist_error * moist_error, dtype=np.float64)
                    ),
                    "combined": float(np.mean(combined * combined, dtype=np.float64)),
                },
            }

        hybrid = {}
        for mass_arm in ("cpu", "reference", "h5"):
            for moisture_arm in ("cpu", "reference", "h5"):
                pressure = (
                    p_top
                    + component[mass_arm]["dry"]
                    + np.sum(
                        qtot[moisture_arm] * component[mass_arm]["dp_dry"],
                        axis=0,
                        dtype=np.float64,
                    )
                )
                hybrid[f"{mass_arm}_mass__{moisture_arm}_moisture"] = _stats(
                    np, pressure - data["cpu"]["PSFC"], masks["all"]
                )

        species_rows = {}
        cpu_dp = component["cpu"]["dp_dry"]
        for species in SPECIES:
            species_rows[species] = {}
            for arm in ("reference", "h5"):
                q_error = data[arm][species] - data["cpu"][species]
                pressure_error = np.sum(
                    q_error * cpu_dp, axis=0, dtype=np.float64
                )
                species_rows[species][arm] = {
                    "mixing_ratio_rmse": _rms(np, q_error),
                    "mixing_ratio_bias": float(
                        np.mean(q_error, dtype=np.float64)
                    ),
                    "surface_pressure_contribution_rmse_pa": _rms(
                        np, pressure_error
                    ),
                    "surface_pressure_contribution_bias_pa": float(
                        np.mean(pressure_error, dtype=np.float64)
                    ),
                }

        mu_spatial = {}
        for mask_name, mask in masks.items():
            mu_spatial[mask_name] = {
                arm: _stats(np, data[arm]["MU"] - data["cpu"]["MU"], mask)
                for arm in ("reference", "h5")
            }
            mu_spatial[mask_name]["h5_minus_reference_rmse"] = (
                mu_spatial[mask_name]["h5"]["rmse"]
                - mu_spatial[mask_name]["reference"]["rmse"]
            )

        checks = {
            "all_authorities_authenticated": True,
            "strict_metrics_reproduce_sealed_result": True,
            "H5_improves_all_seven_non_PSFC_strict_fields": all(
                strict_change[field]["improved"]
                for field in STRICT_FIELDS
                if field != "PSFC"
            ),
            "H5_PSFC_full_grid_regresses": not strict_change["PSFC"]["improved"],
            "H5_PSFC_land_and_terrain_improve": bool(
                spatial["land"]["h5_minus_reference_rmse"] < 0.0
                and spatial["high_land_ge_1000m"]["h5_minus_reference_rmse"] < 0.0
                and spatial["steep_land_top_quartile"][
                    "h5_minus_reference_rmse"
                ]
                < 0.0
            ),
            "H5_PSFC_sea_and_outer5_regress": bool(
                spatial["sea"]["h5_minus_reference_rmse"] > 0.0
                and spatial["outer5"]["h5_minus_reference_rmse"] > 0.0
            ),
            "H5_dry_column_mass_component_improves": bool(
                components["h5"]["dry_column_error"]["rmse"]
                < components["reference"]["dry_column_error"]["rmse"]
            ),
            "H5_QV_and_moist_pressure_component_improve": bool(
                species_rows["QVAPOR"]["h5"]["mixing_ratio_rmse"]
                < species_rows["QVAPOR"]["reference"]["mixing_ratio_rmse"]
                and components["h5"]["moist_column_error"]["rmse"]
                < components["reference"]["moist_column_error"]["rmse"]
            ),
            "reference_PSFC_contains_large_negative_error_cancellation": bool(
                components["reference"]["twice_mean_cross_term_pa2"] < -50.0
                and components["reference"]["moist_column_error"]["bias"] < -2.0
            ),
            "H5_removes_that_error_cancellation": bool(
                abs(components["h5"]["twice_mean_cross_term_pa2"]) < 1.0
                and abs(components["h5"]["moist_column_error"]["bias"]) < 0.01
            ),
            "all_reconstructions_bounded": bool(
                reconstruction_residual["reference"]["max_abs"] < 0.02
                and reconstruction_residual["h5"]["max_abs"] < 0.02
                and reconstruction_residual["cpu"]["max_abs"] < 3.0
            ),
        }
        verdict = (
            "PSFC_RED_IS_SEA_BOUNDARY_ERROR_CANCELLATION_LOSS__"
            "H5_DRY_AND_MOIST_COMPONENTS_INDIVIDUALLY_IMPROVE"
            if all(checks.values())
            else "PSFC_ATTRIBUTION_RED"
        )
        proof = {
            "schema": "gpuwrf.v0234.h5-step200-psfc-attribution.v1",
            "verdict": verdict,
            "scope": (
                "CPU-only read of authenticated Step200 frames; exact moist-"
                "hydrostatic PSFC decomposition; no model import/compile/step, "
                "WRF/MPI execution, GPU command, or GPU query"
            ),
            "authority": {
                "sealed_result": {
                    "path": str(RESULT),
                    "file_sha256": RESULT_FILE_SHA256,
                    "proof_sha256": RESULT_PROOF_SHA256,
                },
                "frames": {
                    name: {"path": str(paths[name]), "sha256": FRAME_SHA256[name]}
                    for name in paths
                },
                "candidate_source_sha256": SOURCE_SHA256,
            },
            "strict_field_change": strict_change,
            "psfc_spatial_localization": spatial,
            "mu_spatial_localization": mu_spatial,
            "psfc_reconstruction_residual": reconstruction_residual,
            "dry_moist_component_attribution": components,
            "mass_moisture_hybrid_table": hybrid,
            "species_attribution_on_CPU_dry_mass": species_rows,
            "checks": checks,
            "interpretation": {
                "gate_disposition": (
                    "The preregistered all-fields gate remains scientifically red; "
                    "this proof does not waive or retune it."
                ),
                "causal_result": (
                    "H5 improves dry column mass and moisture independently. The "
                    "pinned reference's QV deficit contributes a negative pressure "
                    "error that cancels part of its positive dry-mass error; H5 "
                    "removes that compensating QV error and exposes the remaining "
                    "sea/boundary dry-mass bias."
                ),
                "next_target": (
                    "Retain the H5 mechanism for adversarial review and target the "
                    "pre-existing positive dry-column-mass bias over sea/boundaries; "
                    "do not restore the inaccurate QV field merely to recover PSFC "
                    "error cancellation."
                ),
            },
            "gpu_commands": 0,
            "gpu_queries": 0,
            "model_compiles": 0,
            "model_steps": 0,
            "wrf_or_mpi_executions": 0,
        }
        proof["proof_sha256"] = _canonical(proof)
        _atomic_json(OUT, proof)
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "proof_sha256": proof["proof_sha256"],
                    "PSFC_relative_change_percent": strict_change["PSFC"][
                        "relative_change_percent"
                    ],
                    "reference_cross_term_pa2": components["reference"][
                        "twice_mean_cross_term_pa2"
                    ],
                    "h5_cross_term_pa2": components["h5"][
                        "twice_mean_cross_term_pa2"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if all(checks.values()) else 3
    except Exception as exc:
        blocker = {
            "schema": "gpuwrf.v0234.h5-step200-psfc-attribution-blocker.v1",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "gpu_commands": 0,
            "gpu_queries": 0,
            "model_compiles": 0,
            "model_steps": 0,
            "wrf_or_mpi_executions": 0,
        }
        blocker["proof_sha256"] = _canonical(blocker)
        _atomic_json(BLOCKER, blocker)
        print(json.dumps(blocker, sort_keys=True), flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
