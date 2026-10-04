"""Read-only retained-frame localization of the pre-sunrise V/V10 error.

This utility executes no model code.  It authenticates the completed pinned
replay's frame-pair receipts, reads the existing GPU and CPU-WRF NetCDF frames,
and reduces the nocturnal wind/error geometry needed to re-rank V10 causes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-21-v0234-gpt-v10-rootcause"
REPLAY = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "v0234_gpt_v10_replay_c17fca1e201ae106_reference"
)
FRAME_PAIRS = REPLAY / "frame-pairs"
REFERENCE_WAKE = (
    ROOT.parent
    / "v0234-gpt-v10-replay"
    / ".agent/sprints/2026-07-21-v0234-gpt-v10-replay/REFERENCE_WAKE_ANALYSIS.json"
)
REFERENCE_WAKE_SHA256 = (
    "3cb55e89f6536a510a72a9cfa25c96df72175ad098e137de48952fcb28dfcc6a"
)
D03_STEPS = (0, 200, 400, 800, 1200, 1600, 2000, 2400, 2800, 3000, 3200, 3400, 3600, 3800)
ALIGNED_0600 = {"d01": 400, "d02": 1200, "d03": 3600}
STRICT_FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")
CONTEXT_FIELDS = ("QVAPOR", "SWDOWN", "TSK", "HFX", "LH", "UST", "PBLH")
TERMINAL_RMSE = {"V": 1.1644191535446722, "V10": 2.3086370774842995}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(
            clean,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _frame_receipt(domain: str, step: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = FRAME_PAIRS / f"{domain}-step-{step:05d}.json"
    payload = json.loads(path.read_text())
    declared = payload.get("proof_sha256")
    if declared != _canonical(payload):
        raise RuntimeError(f"frame-pair canonical proof drift: {path}")
    if (
        payload.get("domain") != domain
        or int(payload.get("own_step", -1)) != step
        or payload.get("passed") is not True
        or payload.get("finite_identity", {}).get("passed") is not True
    ):
        raise RuntimeError(f"frame-pair admission drift: {path}")
    row = {
        "path": str(path),
        "file_sha256": _sha256(path),
        "canonical_sha256": declared,
        "valid_time": payload["valid_time"],
    }
    for arm in ("candidate", "cpu"):
        frame = Path(payload[arm]["path"])
        expected = payload[arm].get("sha256")
        if not frame.is_file() or frame.is_symlink() or _sha256(frame) != expected:
            raise RuntimeError(f"{domain} step {step} {arm} frame authority drift")
        row[f"{arm}_frame"] = {
            "path": str(frame),
            "bytes": frame.stat().st_size,
            "sha256": expected,
        }
    return payload, row


def _array(dataset: Dataset, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise RuntimeError(f"required retained field missing: {name}")
    value = np.asarray(dataset.variables[name][0], dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"nonfinite retained field: {name}")
    return value


def _stats(error: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "bias": float(np.mean(error, dtype=np.float64)),
        "max_abs": float(np.max(np.abs(error))),
    }


def _outer_mask(ny: int, nx: int, width: int) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask


def _masked_rms(error: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(error)[mask]
    return float(np.sqrt(np.mean(values * values, dtype=np.float64)))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _read_pair(payload: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {"candidate": {}, "cpu": {}}
    with Dataset(payload["candidate"]["path"]) as candidate, Dataset(
        payload["cpu"]["path"]
    ) as cpu:
        for field in STRICT_FIELDS + CONTEXT_FIELDS + ("LANDMASK",):
            arrays["candidate"][field] = _array(candidate, field)
            arrays["cpu"][field] = _array(cpu, field)
    return arrays


def _spatial_wind(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    candidate = arrays["candidate"]
    cpu = arrays["cpu"]
    v_low_candidate = 0.5 * (candidate["V"][0, :-1, :] + candidate["V"][0, 1:, :])
    v_low_cpu = 0.5 * (cpu["V"][0, :-1, :] + cpu["V"][0, 1:, :])
    u_low_candidate = 0.5 * (candidate["U"][0, :, :-1] + candidate["U"][0, :, 1:])
    u_low_cpu = 0.5 * (cpu["U"][0, :, :-1] + cpu["U"][0, :, 1:])
    v_error = v_low_candidate - v_low_cpu
    u_error = u_low_candidate - u_low_cpu
    v10_error = candidate["V10"] - cpu["V10"]
    u10_error = candidate["U10"] - cpu["U10"]
    ny, nx = v10_error.shape
    outer5 = _outer_mask(ny, nx, 5)
    interior5 = ~outer5
    wake = np.zeros((ny, nx), dtype=bool)
    wake[32:50, 10:30] = True
    land = cpu["LANDMASK"] > 0.5
    sea = ~land
    v_location = np.unravel_index(int(np.argmax(np.abs(v_error))), v_error.shape)
    v10_location = np.unravel_index(int(np.argmax(np.abs(v10_error))), v10_error.shape)
    return {
        "lowest_mass_u": _stats(u_error),
        "lowest_mass_v": _stats(v_error),
        "U10": _stats(u10_error),
        "V10": _stats(v10_error),
        "lowest_v_vs_V10_error_correlation": _corr(v_error, v10_error),
        "lowest_u_vs_U10_error_correlation": _corr(u_error, u10_error),
        "lowest_v_outer5_rms": _masked_rms(v_error, outer5),
        "lowest_v_interior5_rms": _masked_rms(v_error, interior5),
        "V10_outer5_rms": _masked_rms(v10_error, outer5),
        "V10_interior5_rms": _masked_rms(v10_error, interior5),
        "lowest_v_wake_box_rms": _masked_rms(v_error, wake),
        "V10_wake_box_rms": _masked_rms(v10_error, wake),
        "lowest_v_land_rms": _masked_rms(v_error, land),
        "lowest_v_sea_rms": _masked_rms(v_error, sea),
        "V10_land_rms": _masked_rms(v10_error, land),
        "V10_sea_rms": _masked_rms(v10_error, sea),
        "lowest_v_max_abs_location_yx": [int(v_location[0]), int(v_location[1])],
        "V10_max_abs_location_yx": [int(v10_location[0]), int(v10_location[1])],
        "lowest_v_error": v_error,
        "V10_error": v10_error,
    }


def _frame_reduction(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    candidate = arrays["candidate"]
    cpu = arrays["cpu"]
    strict = {
        field: _stats(candidate[field] - cpu[field])
        for field in STRICT_FIELDS
    }
    context = {
        field: {
            **_stats(candidate[field] - cpu[field]),
            "candidate_min": float(np.min(candidate[field])),
            "candidate_max": float(np.max(candidate[field])),
            "cpu_min": float(np.min(cpu[field])),
            "cpu_max": float(np.max(cpu[field])),
        }
        for field in CONTEXT_FIELDS
    }
    wind = _spatial_wind(arrays)
    v_vertical = np.sqrt(
        np.mean(
            (candidate["V"] - cpu["V"]) ** 2,
            axis=(1, 2),
            dtype=np.float64,
        )
    )
    return {
        "strict": strict,
        "context": context,
        "wind": {
            key: value
            for key, value in wind.items()
            if key not in {"lowest_v_error", "V10_error"}
        },
        "V_vertical_rmse": [float(value) for value in v_vertical],
        "_lowest_v_error": wind["lowest_v_error"],
        "_V10_error": wind["V10_error"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite proof: {output}")
    if _sha256(REFERENCE_WAKE) != REFERENCE_WAKE_SHA256:
        raise RuntimeError("reference wake authority drift")

    receipts: dict[str, Any] = {}
    timeline: list[dict[str, Any]] = []
    retained_errors: dict[int, dict[str, np.ndarray]] = {}
    max_presun_swdown = {"candidate": 0.0, "cpu": 0.0}
    for step in D03_STEPS:
        pair, receipt = _frame_receipt("d03", step)
        receipts[f"d03-{step:05d}"] = receipt
        arrays = _read_pair(pair)
        reduced = _frame_reduction(arrays)
        retained_errors[step] = {
            "lowest_v": reduced.pop("_lowest_v_error"),
            "V10": reduced.pop("_V10_error"),
        }
        max_presun_swdown["candidate"] = max(
            max_presun_swdown["candidate"],
            float(np.max(np.abs(arrays["candidate"]["SWDOWN"]))),
        )
        max_presun_swdown["cpu"] = max(
            max_presun_swdown["cpu"],
            float(np.max(np.abs(arrays["cpu"]["SWDOWN"]))),
        )
        timeline.append(
            {
                "step": step,
                "valid_time": pair["valid_time"],
                **reduced,
            }
        )

    reference_low_v = retained_errors[3800]["lowest_v"]
    reference_v10 = retained_errors[3800]["V10"]
    for row in timeline:
        step = int(row["step"])
        row["pattern_correlation_to_step3800"] = {
            "lowest_v": _corr(retained_errors[step]["lowest_v"], reference_low_v),
            "V10": _corr(retained_errors[step]["V10"], reference_v10),
        }

    aligned: dict[str, Any] = {}
    for domain, step in ALIGNED_0600.items():
        if domain == "d03":
            row = next(item for item in timeline if item["step"] == step)
            aligned[domain] = {
                "step": step,
                "valid_time": row["valid_time"],
                "strict": row["strict"],
                "wind": row["wind"],
            }
            continue
        pair, receipt = _frame_receipt(domain, step)
        receipts[f"{domain}-{step:05d}"] = receipt
        reduced = _frame_reduction(_read_pair(pair))
        reduced.pop("_lowest_v_error")
        reduced.pop("_V10_error")
        aligned[domain] = {
            "step": step,
            "valid_time": pair["valid_time"],
            "strict": reduced["strict"],
            "wind": reduced["wind"],
        }

    step3800 = next(item for item in timeline if item["step"] == 3800)
    step200 = next(item for item in timeline if item["step"] == 200)
    summary = {
        "step3800_utc": step3800["valid_time"],
        "sunrise_utc_approx": "2025-03-01T07:20:00+00:00",
        "step3800_fraction_of_terminal_rmse": {
            field: step3800["strict"][field]["rmse"] / TERMINAL_RMSE[field]
            for field in ("V", "V10")
        },
        "step200_to_step3800_growth_fraction_of_terminal": {
            field: (
                step3800["strict"][field]["rmse"]
                - step200["strict"][field]["rmse"]
            )
            / TERMINAL_RMSE[field]
            for field in ("V", "V10")
        },
        "max_abs_SWDOWN_through_step3800": max_presun_swdown,
        "shortwave_path_active_before_step3800": bool(
            max(max_presun_swdown.values()) > 0.0
        ),
        "step3800_lowest_v_vs_V10_error_correlation": step3800["wind"][
            "lowest_v_vs_V10_error_correlation"
        ],
        "step3800_interior_to_outer5_rms_ratio": {
            "lowest_v": step3800["wind"]["lowest_v_interior5_rms"]
            / step3800["wind"]["lowest_v_outer5_rms"],
            "V10": step3800["wind"]["V10_interior5_rms"]
            / step3800["wind"]["V10_outer5_rms"],
        },
        "interpretation": (
            "The majority error is present with SWDOWN exactly inactive and V10 "
            "tracks prognostic lowest-level V. Re-rank nocturnal dynamics/advection/"
            "boundary and direct moisture-density pathways above any shortwave chain."
        ),
    }
    proof = {
        "schema": "gpuwrf.v0234.v10-nocturnal-localization.v1",
        "authority": {
            "reference_wake_analysis": str(REFERENCE_WAKE),
            "reference_wake_analysis_sha256": REFERENCE_WAKE_SHA256,
            "replay_root": str(REPLAY),
            "frame_receipts": receipts,
        },
        "method": {
            "type": "read-only authenticated retained NetCDF reduction",
            "model_executions": 0,
            "gpu_commands": 0,
            "gpu_queries": 0,
            "wrf_or_mpi_executions": 0,
        },
        "aligned_0600_domain_metrics": aligned,
        "d03_nocturnal_timeline": timeline,
        "summary": summary,
        "verdict": "NOCTURNAL_DYNAMICS_DOMINANT_SHORTWAVE_CAUSAL_CHAIN_WITHDRAWN",
    }
    if not all(math.isfinite(value) for value in summary["step3800_fraction_of_terminal_rmse"].values()):
        raise RuntimeError("nonfinite summary")
    proof["proof_sha256"] = _canonical(proof)
    _atomic_json(output, proof)
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
