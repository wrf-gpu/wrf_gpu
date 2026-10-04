"""Read-only terrain/land localization of the earliest nocturnal V error.

This reducer authenticates selected retained frame pairs from the pinned V10
replay.  It distinguishes the nearly exact Step0 prognostic V state from
diagnostic initialization differences, then measures where the first evolved
lowest-level V error appears relative to land, height, slope, and boundaries.
No model, WRF, MPI, JAX, or GPU code is executed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from scripts import v0234_nocturnal_localization as localization


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
OUT = SPRINT / "NOCTURNAL_ORIGIN_TERRAIN.json"
PRIOR = SPRINT / "NOCTURNAL_LOCALIZATION.json"
PRIOR_FILE_SHA256 = (
    "a6af0a4ebc8d5566a47d3ba41365505bf8c889f34ff5a22021ef2bb16e9e6ce9"
)
PRIOR_PROOF_SHA256 = (
    "c4aba34884fa3723070f5ee42bcadbbda6541605e83a6c7be0675bfc7f7d9673"
)
STEPS = (0, 200, 400, 800, 3800)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def _mask_row(error: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)[mask]
    if not values.size:
        raise RuntimeError("empty terrain mask")
    return {
        "cells": int(values.size),
        "rms": _rms(values),
        "mean_abs": float(np.mean(np.abs(values), dtype=np.float64)),
        "max_abs": float(np.max(np.abs(values))),
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def main() -> int:
    if OUT.exists() or OUT.is_symlink():
        raise RuntimeError(f"refusing to overwrite proof: {OUT}")
    if _sha256(PRIOR) != PRIOR_FILE_SHA256:
        raise RuntimeError("prior nocturnal localization file changed")
    prior = json.loads(PRIOR.read_text())
    if (
        prior.get("proof_sha256") != PRIOR_PROOF_SHA256
        or localization._canonical(prior) != PRIOR_PROOF_SHA256
        or prior.get("verdict")
        != "NOCTURNAL_DYNAMICS_DOMINANT_SHORTWAVE_CAUSAL_CHAIN_WITHDRAWN"
    ):
        raise RuntimeError("prior nocturnal localization semantics changed")

    frames: list[dict[str, Any]] = []
    authority: dict[str, Any] = {}
    static_sha: str | None = None
    for step in STEPS:
        payload, receipt = localization._frame_receipt("d03", step)
        key = f"d03-{step:05d}"
        if receipt != prior["authority"]["frame_receipts"][key]:
            raise RuntimeError(f"frame receipt differs from prior: {key}")
        authority[key] = receipt
        arrays = localization._read_pair(payload)
        candidate = arrays["candidate"]
        cpu = arrays["cpu"]
        with Dataset(payload["candidate"]["path"]) as candidate_ds, Dataset(
            payload["cpu"]["path"]
        ) as cpu_ds:
            candidate_hgt = localization._array(candidate_ds, "HGT")
            cpu_hgt = localization._array(cpu_ds, "HGT")
        if not np.array_equal(candidate_hgt, cpu_hgt):
            raise RuntimeError(f"HGT static identity failed at step {step}")
        hgt = cpu_hgt
        hgt_sha = _array_sha256(hgt)
        if static_sha is None:
            static_sha = hgt_sha
        elif hgt_sha != static_sha:
            raise RuntimeError("HGT changed across retained frames")
        land = cpu["LANDMASK"] > 0.5
        if not np.array_equal(candidate["LANDMASK"], cpu["LANDMASK"]):
            raise RuntimeError(f"LANDMASK static identity failed at step {step}")
        sea = ~land
        grad_y, grad_x = np.gradient(hgt)
        slope_index = np.hypot(grad_x, grad_y)
        steep_threshold = float(np.quantile(slope_index[land], 0.9))
        masks = {
            "sea": sea,
            "land_all": land,
            "land_below_500m": land & (hgt < 500.0),
            "land_500_to_1500m": land & (hgt >= 500.0) & (hgt < 1500.0),
            "land_at_or_above_1500m": land & (hgt >= 1500.0),
            "steepest_land_decile": land & (slope_index >= steep_threshold),
        }
        v_low_candidate = 0.5 * (
            candidate["V"][0, :-1, :] + candidate["V"][0, 1:, :]
        )
        v_low_cpu = 0.5 * (cpu["V"][0, :-1, :] + cpu["V"][0, 1:, :])
        v_low_error = v_low_candidate - v_low_cpu
        maximum = np.unravel_index(
            int(np.argmax(np.abs(v_low_error))), v_low_error.shape
        )
        vertical = np.sqrt(
            np.mean(
                (candidate["V"] - cpu["V"]) ** 2,
                axis=(1, 2),
                dtype=np.float64,
            )
        )
        outer5 = localization._outer_mask(*v_low_error.shape, 5)
        terrain_bins = {
            name: _mask_row(v_low_error, mask) for name, mask in masks.items()
        }
        frames.append(
            {
                "step": step,
                "valid_time": payload["valid_time"],
                "prognostic_V_full_rmse": localization._stats(
                    candidate["V"] - cpu["V"]
                )["rmse"],
                "diagnostics": {
                    "V10_rmse": localization._stats(
                        candidate["V10"] - cpu["V10"]
                    )["rmse"],
                    "PBLH_rmse": localization._stats(
                        candidate["PBLH"] - cpu["PBLH"]
                    )["rmse"],
                    "UST_rmse": localization._stats(
                        candidate["UST"] - cpu["UST"]
                    )["rmse"],
                },
                "lowest_mass_V": {
                    "rmse": _rms(v_low_error),
                    "interior5_rms": _rms(v_low_error[~outer5]),
                    "outer5_rms": _rms(v_low_error[outer5]),
                    "terrain_bins": terrain_bins,
                    "max_abs": float(np.max(np.abs(v_low_error))),
                    "max_location": {
                        "y": int(maximum[0]),
                        "x": int(maximum[1]),
                        "error": float(v_low_error[maximum]),
                        "landmask": int(land[maximum]),
                        "hgt_m": float(hgt[maximum]),
                        "slope_index_m_per_cell": float(slope_index[maximum]),
                    },
                    "abs_error_correlation_on_land": {
                        "HGT": localization._corr(
                            np.abs(v_low_error[land]), hgt[land]
                        ),
                        "slope_index": localization._corr(
                            np.abs(v_low_error[land]), slope_index[land]
                        ),
                    },
                },
                "V_vertical_rmse": [float(value) for value in vertical],
                "static": {
                    "HGT_sha256": hgt_sha,
                    "steepest_land_decile_threshold_m_per_cell": steep_threshold,
                },
            }
        )

    by_step = {row["step"]: row for row in frames}
    step0 = by_step[0]
    step200 = by_step[200]
    step800 = by_step[800]
    step3800 = by_step[3800]
    bins200 = step200["lowest_mass_V"]["terrain_bins"]
    bins3800 = step3800["lowest_mass_V"]["terrain_bins"]
    summary = {
        "step0_prognostic_V_nearly_exact_rmse": step0["prognostic_V_full_rmse"],
        "step0_diagnostic_V10_rmse": step0["diagnostics"]["V10_rmse"],
        "step0_diagnostic_PBLH_rmse": step0["diagnostics"]["PBLH_rmse"],
        "step200_land_to_sea_rms_ratio": (
            bins200["land_all"]["rms"] / bins200["sea"]["rms"]
        ),
        "step200_high_land_to_sea_rms_ratio": (
            bins200["land_at_or_above_1500m"]["rms"] / bins200["sea"]["rms"]
        ),
        "step200_steep_land_to_sea_rms_ratio": (
            bins200["steepest_land_decile"]["rms"] / bins200["sea"]["rms"]
        ),
        "step200_interior_to_outer5_rms_ratio": (
            step200["lowest_mass_V"]["interior5_rms"]
            / step200["lowest_mass_V"]["outer5_rms"]
        ),
        "step800_sea_to_land_rms_ratio": (
            step800["lowest_mass_V"]["terrain_bins"]["sea"]["rms"]
            / step800["lowest_mass_V"]["terrain_bins"]["land_all"]["rms"]
        ),
        "step3800_high_land_to_sea_rms_ratio": (
            bins3800["land_at_or_above_1500m"]["rms"]
            / bins3800["sea"]["rms"]
        ),
        "step3800_steep_land_to_sea_rms_ratio": (
            bins3800["steepest_land_decile"]["rms"] / bins3800["sea"]["rms"]
        ),
        "interpretation": (
            "Step0 V is effectively identical while Step0 V10/PBLH are not, so "
            "initial diagnostic differences are not an evolved-wind cause. The "
            "first prognostic V error is interior and land/terrain concentrated, "
            "then spreads over sea. This demotes direct boundary injection and "
            "ranks terrain-following interior dynamics and surface/PBL coupling; "
            "it does not implicate daylight terrain radiation."
        ),
    }
    proof = {
        "schema": "gpuwrf.v0234.v10-nocturnal-origin-terrain.v1",
        "verdict": (
            "EARLIEST_PROGNOSTIC_V_ERROR_INTERIOR_TERRAIN_LAND_ORIGIN__"
            "BOUNDARY_AND_STEP0_DIAGNOSTICS_DEMOTED"
        ),
        "authority": {
            "prior_localization": {
                "path": str(PRIOR),
                "file_sha256": PRIOR_FILE_SHA256,
                "proof_sha256": PRIOR_PROOF_SHA256,
            },
            "selected_frame_receipts": authority,
            "static_HGT_sha256": static_sha,
        },
        "method": {
            "type": "read-only retained NetCDF terrain/land reduction",
            "slope_index": "hypot(numpy.gradient(HGT)) in metres per grid cell",
            "model_executions": 0,
            "gpu_commands": 0,
            "gpu_queries": 0,
            "wrf_or_mpi_executions": 0,
        },
        "frames": frames,
        "summary": summary,
        "limitations": {
            "spatial_association_is_not_operator_causality": True,
            "terrain_localization_does_not_select_dynamics_vs_PBL": True,
            "no_C3_radiative_inference_before_sunrise": True,
        },
    }
    if not _finite(proof):
        raise RuntimeError("nonfinite terrain proof")
    proof["proof_sha256"] = localization._canonical(proof)
    localization._atomic_json(OUT, proof)
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof_sha256": proof["proof_sha256"],
                "summary": summary,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
