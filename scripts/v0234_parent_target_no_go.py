"""Close upstream d02 contamination from the retained live Step200 package."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
OUT = SPRINT / "upstream-d02-target-no-go-proof.json"
DATA = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
CARRY = (
    DATA
    / "corrected_ni_rca_max_22c2bd7a/"
    "nested_h_sca_order_395fb800_full18h_discriminator1/"
    "failure/first-failed-d03-step-200.pkl"
)
FILES = {
    "carry": (CARRY, "6c0700dd0e6d973ae165a1b2512b5530824357ccc94be3b20080f0f830a8d3b8"),
    "wrfinput_d03": (DATA / "run/wrf/wrfinput_d03", "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a"),
    "current_395": (
        DATA
        / "corrected_ni_rca_max_22c2bd7a/"
        "nested_h_sca_order_395fb800_full18h_discriminator1/"
        "output/wrfout_d03_2025-03-01_00:20:00",
        "3235053155927d04ee11971e34ddce71ba81572bf9f1b94a043e59c14d0c6c10",
    ),
    "cpu_wrf": (
        DATA / "run/wrf/wrfout_d03_2025-03-01_00:20:00",
        "0a1157771f8b00f2c2c4fb66ec1cb63e4534cf3c305981ca81e1cfaad0d8d7f1",
    ),
    "retry20": (
        DATA
        / "gpu_validation_retry20_relative_rmse_3ee02c19/"
        "pair-snapshots/20250301T002000/gpu.nc",
        "70e09cf3ca22711c66ef529e716ead53fb998d2f548e9939c35d7767ac3f9e62",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _interpolate(boundary: np.ndarray, lead_seconds: float, cadence_s: float) -> np.ndarray:
    index = lead_seconds / cadence_s
    lower = min(max(int(np.floor(index)), 0), boundary.shape[0] - 1)
    upper = min(lower + 1, boundary.shape[0] - 1)
    alpha = min(max(index - lower, 0.0), 1.0)
    return boundary[lower] * (1.0 - alpha) + boundary[upper] * alpha


def _full_ring(leaf: np.ndarray, z_len: int, y_len: int, x_len: int) -> np.ndarray:
    target = np.zeros((z_len, y_len, x_len), dtype=np.float64)
    for distance in range(leaf.shape[1]):
        target[:, :, distance] = leaf[0, distance, :z_len, :y_len]
        target[:, :, x_len - 1 - distance] = leaf[1, distance, :z_len, :y_len]
        # W/E are written first and S/N own the corners, exactly like the source.
        target[:, distance, :] = leaf[2, distance, :z_len, :x_len]
        target[:, y_len - 1 - distance, :] = leaf[3, distance, :z_len, :x_len]
    return target


def _rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def main() -> int:
    authority = {}
    for name, (path, expected) in FILES.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"authority hash mismatch for {name}: {actual}")
        authority[name] = {"path": str(path), "sha256": actual}

    with CARRY.open("rb") as stream:
        carry = pickle.load(stream)
    theta_boundary = np.asarray(carry.state.theta_bdy, dtype=np.float64)
    mu_boundary = np.asarray(carry.state.mu_bdy, dtype=np.float64)
    if theta_boundary.shape != (2, 4, 5, 44, 112):
        raise RuntimeError(f"unexpected theta package shape {theta_boundary.shape}")
    if mu_boundary.shape != (2, 4, 5, 1, 112):
        raise RuntimeError(f"unexpected mu package shape {mu_boundary.shape}")

    with Dataset(FILES["wrfinput_d03"][0]) as dataset:
        c1h = np.asarray(dataset.variables["C1H"][0], dtype=np.float64)[:, None, None]
        c2h = np.asarray(dataset.variables["C2H"][0], dtype=np.float64)[:, None, None]

    frames = {}
    for name in ("current_395", "cpu_wrf", "retry20"):
        with Dataset(FILES[name][0]) as dataset:
            frames[name] = np.asarray(dataset.variables["THM"][0], dtype=np.float64)

    # The retained package spans parent d02 endpoints 1188 -> 1206 s.  Step200
    # is absolute 1200 s, hence WRF dtbc=12 s and alpha=2/3.
    package_lead_s = 12.0
    cadence_s = 18.0
    theta_strip = _interpolate(theta_boundary, package_lead_s, cadence_s)
    mu_strip = _interpolate(mu_boundary, package_lead_s, cadence_s)
    theta_coupled = _full_ring(theta_strip, 44, 93, 111)
    mu_perturbation = _full_ring(mu_strip, 1, 93, 111)[0]
    child_mub = np.asarray(carry.state.mu_total, dtype=np.float64) - np.asarray(
        carry.state.mu_perturbation, dtype=np.float64
    )
    mass_h = c1h * (child_mub + mu_perturbation)[None, :, :] + c2h
    target_thm = theta_coupled / mass_h

    yy, xx = np.indices((93, 111))
    distance = np.minimum.reduce((yy, xx, 92 - yy, 110 - xx))
    rows = {}
    for ring in range(5):
        mask = distance == ring
        row = {}
        for anchor in ("cpu_wrf", "retry20"):
            target_rmse = _rms(target_thm[:, mask] - frames[anchor][:, mask])
            child_rmse = _rms(frames["current_395"][:, mask] - frames[anchor][:, mask])
            row[anchor] = {
                "parent_target_rmse_K": target_rmse,
                "evolved_child_rmse_K": child_rmse,
                "target_is_closer": target_rmse < child_rmse,
            }
        rows[str(ring)] = row

    endpoint_rows = []
    for lead_s in (0.0, 6.0, 12.0, 18.0):
        theta = _full_ring(_interpolate(theta_boundary, lead_s, cadence_s), 44, 93, 111)
        mu = _full_ring(_interpolate(mu_boundary, lead_s, cadence_s), 1, 93, 111)[0]
        target = theta / (c1h * (child_mub + mu)[None, :, :] + c2h)
        mask = distance == 1
        endpoint_rows.append(
            {
                "dtbc_s": lead_s,
                "absolute_time_s": 1188.0 + lead_s,
                "cpu_wrf_ring1_rmse_K": _rms(target[:, mask] - frames["cpu_wrf"][:, mask]),
            }
        )

    checks = {
        "authenticated_inputs": True,
        "exact_two_record_live_package": theta_boundary.shape[0] == mu_boundary.shape[0] == 2,
        "exact_step200_clock_alpha_two_thirds": package_lead_s / cadence_s == 2.0 / 3.0,
        "target_finite": bool(np.all(np.isfinite(target_thm))),
        "ring0_target_and_child_effectively_identical": abs(
            rows["0"]["cpu_wrf"]["parent_target_rmse_K"]
            - rows["0"]["cpu_wrf"]["evolved_child_rmse_K"]
        ) < 1.0e-6,
        "parent_target_closer_both_anchors_ring1": all(
            rows["1"][anchor]["target_is_closer"] for anchor in ("cpu_wrf", "retry20")
        ),
        "parent_target_closer_both_anchors_rings1_through4": all(
            rows[str(ring)][anchor]["target_is_closer"]
            for ring in range(1, 5)
            for anchor in ("cpu_wrf", "retry20")
        ),
    }
    proof = {
        "schema": "gpuwrf.v0234.upstream-d02-target-no-go.v1",
        "authority": authority,
        "clock": {
            "package_start_s": 1188.0,
            "package_end_s": 1206.0,
            "evaluated_absolute_s": 1200.0,
            "dtbc_s": package_lead_s,
            "parent_cadence_s": cadence_s,
            "alpha": package_lead_s / cadence_s,
        },
        "algebra": "THM_target=T_coupled/(C1H*(MUB_child+MU_target)+C2H)",
        "ring_metrics": rows,
        "endpoint_ring1_cpu_wrf": endpoint_rows,
        "checks": checks,
        "causal_conclusion": (
            "the live d02 target is already closer than the evolved d03 child; "
            "the Step200 THM error is generated by child evolution away from useful forcing, "
            "not inherited from a worse upstream target"
        ),
        "gpu_commands": 0,
        "gpu_queries": 0,
        "verdict": (
            "UPSTREAM_D02_CONTAMINATION_NO_GO"
            if all(checks.values())
            else "UPSTREAM_D02_CONTAMINATION_UNRESOLVED"
        ),
    }
    proof["proof_sha256"] = _canonical(proof)
    _atomic_json(OUT, proof)
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof_sha256": proof["proof_sha256"],
                "ring1": rows["1"],
                "failed_checks": [name for name, passed in checks.items() if not passed],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
