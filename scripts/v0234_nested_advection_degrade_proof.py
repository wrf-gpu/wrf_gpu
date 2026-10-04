"""Independent source/spatial proof for the nested advection-stencil candidate.

This tool imports neither JAX nor gpuwrf.  It authenticates the retained first
post-step frame, evaluates the ring-one signature, and mirrors pristine WRF's
one-dimensional order-5 degraded flux tiers in NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")
BASE = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z"
)
CANDIDATE = BASE / (
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_boundary_final_4484be85_full18h_owner_override1/output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
PAIR = BASE / (
    "gpu_validation_retry20_relative_rmse_3ee02c19/"
    "pair-snapshots/20250301T002000"
)
RETRY = PAIR / "gpu.nc"
CPU = PAIR / "cpu.nc"
FIELDS = ("T", "U", "V", "W", "T2", "U10", "V10", "PSFC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ring_mask(shape: tuple[int, ...], ring: int) -> np.ndarray:
    ny, nx = shape[-2:]
    y, x = np.ogrid[:ny, :nx]
    distance = np.minimum.reduce(
        (
            np.broadcast_to(y, (ny, nx)),
            np.broadcast_to(x, (ny, nx)),
            np.broadcast_to(ny - 1 - y, (ny, nx)),
            np.broadcast_to(nx - 1 - x, (ny, nx)),
        )
    )
    mask = distance == int(ring)
    return np.broadcast_to(mask, shape) if len(shape) == 3 else mask


def rmse(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def frame_metrics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    with Dataset(CANDIDATE) as candidate, Dataset(RETRY) as retry, Dataset(CPU) as cpu:
        for name in FIELDS:
            c = np.asarray(candidate.variables[name][0], dtype=np.float64)
            r = np.asarray(retry.variables[name][0], dtype=np.float64)
            truth = np.asarray(cpu.variables[name][0], dtype=np.float64)
            ring1 = ring_mask(c.shape, 1)
            result[name] = {
                "candidate_vs_cpu_full_rmse": rmse(c - truth),
                "retry20_vs_cpu_full_rmse": rmse(r - truth),
                "candidate_vs_cpu_ring1_rmse": rmse((c - truth)[ring1]),
                "retry20_vs_cpu_ring1_rmse": rmse((r - truth)[ring1]),
                "candidate_minus_retry20_ring1_rmse": rmse((c - r)[ring1]),
            }
    return result


def wrf_degraded_face(q: np.ndarray, m: int, velocity: float) -> float:
    """Literal positive-timestep WRF h5 degraded face formula."""

    n = int(q.size)
    v = float(velocity)
    if 3 <= m <= n - 3:
        flux6 = (
            37.0 * (q[m] + q[m - 1])
            - 8.0 * (q[m + 1] + q[m - 2])
            + q[m + 2]
            + q[m - 3]
        ) / 60.0
        corr = (
            (q[m + 2] - q[m - 3])
            - 5.0 * (q[m + 1] - q[m - 2])
            + 10.0 * (q[m] - q[m - 1])
        ) / 60.0
        return v * (flux6 - np.sign(v) * corr)
    if m in (2, n - 2):
        flux4 = (
            7.0 * (q[m] + q[m - 1]) - q[m + 1] - q[m - 2]
        ) / 12.0
        corr = (
            q[m + 1] - q[m - 2] - 3.0 * (q[m] - q[m - 1])
        ) / 12.0
        return v * (flux4 + np.sign(v) * corr)
    if m in (1, n - 1):
        return v * 0.5 * (q[m] + q[m - 1])
    return 0.0


def periodic_face(q: np.ndarray, m: int, velocity: float) -> float:
    n = int(q.size)
    at = lambda offset: q[(m + offset) % n]
    flux6 = (
        37.0 * (at(0) + at(-1))
        - 8.0 * (at(1) + at(-2))
        + at(2)
        + at(-3)
    ) / 60.0
    corr = (
        (at(2) - at(-3))
        - 5.0 * (at(1) - at(-2))
        + 10.0 * (at(0) - at(-1))
    ) / 60.0
    return float(velocity) * (flux6 - np.sign(velocity) * corr)


def algebra_oracle() -> dict[str, Any]:
    n = 12
    q = np.zeros(n, dtype=np.float64)
    q[-1] = 1.0
    degraded_faces = np.array(
        [wrf_degraded_face(q, m, 1.0) for m in range(n)], dtype=np.float64
    )
    periodic_faces = np.array(
        [periodic_face(q, m, 1.0) for m in range(n)], dtype=np.float64
    )
    degraded_tendency = np.zeros(n, dtype=np.float64)
    periodic_tendency = -(np.roll(periodic_faces, -1) - periodic_faces)
    degraded_tendency[1:-1] = -(
        degraded_faces[2:] - degraded_faces[1:-1]
    )
    conservation_residual = float(
        np.sum(degraded_tendency[1:-1])
        + degraded_faces[-1]
        - degraded_faces[1]
    )
    if periodic_tendency[1] != -0.25:
        raise RuntimeError(f"periodic wrap oracle changed: {periodic_tendency[1]}")
    if degraded_tendency[1] != 0.0:
        raise RuntimeError(f"nested west ring received east signal: {degraded_tendency[1]}")
    if degraded_tendency[-2] != -0.5:
        raise RuntimeError("nested east-local transport was unexpectedly disabled")
    if abs(conservation_residual) > 1.0e-15:
        raise RuntimeError(f"boundary-flux conservation failed: {conservation_residual}")
    return {
        "east_impulse_periodic_west_ring1_tendency": float(periodic_tendency[1]),
        "east_impulse_nested_west_ring1_tendency": float(degraded_tendency[1]),
        "east_impulse_nested_east_ring1_tendency": float(degraded_tendency[-2]),
        "nested_interior_sum_plus_boundary_flux_residual": conservation_residual,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=SPRINT / "nested-advection-degrade-proof.json",
    )
    args = parser.parse_args()

    pristine = PRISTINE.read_text()
    port_path = ROOT / "src/gpuwrf/runtime/operational_mode.py"
    port = port_path.read_text()
    selector = (
        "if(config_flags%specified .or. config_flags%nested) "
        "specified = .true."
    )
    candidate_gate = (
        "if _nested_frozen_wrf_boundary_active(namelist):\n"
        "        return True\n"
        "    if not bool(getattr(namelist, \"specified_adv_degrade\", False))"
    )
    if pristine.count(selector) != 8:
        raise RuntimeError("pristine WRF specified-or-nested selector count changed")
    if candidate_gate not in port:
        raise RuntimeError("candidate gate is missing or no longer precedes the legacy gate")

    metrics = frame_metrics()
    if not all(
        metrics[name]["candidate_vs_cpu_ring1_rmse"]
        > metrics[name]["retry20_vs_cpu_ring1_rmse"]
        for name in ("T", "U", "V", "W")
    ):
        raise RuntimeError("retained earliest dry-dynamics ring-one signature changed")

    payload: dict[str, Any] = {
        "schema": "gpuwrf.v0234.nested-advection-degrade-proof.v1",
        "source": {
            "pristine_path": str(PRISTINE),
            "pristine_sha256": sha256_file(PRISTINE),
            "specified_or_nested_selector_count": pristine.count(selector),
            "families": [
                "U", "V", "scalar", "W", "scalar_pd", "scalar_weno",
                "scalar_wenopd", "scalar_mono",
            ],
            "port_path": str(port_path),
            "port_sha256": sha256_file(port_path),
            "candidate_gate_precedes_legacy_gate": True,
        },
        "retained_00_20_authority": {
            "candidate_4484be85": {
                "path": str(CANDIDATE), "sha256": sha256_file(CANDIDATE)
            },
            "retry20": {"path": str(RETRY), "sha256": sha256_file(RETRY)},
            "cpu_wrf": {"path": str(CPU), "sha256": sha256_file(CPU)},
        },
        "retained_00_20_metrics": metrics,
        "independent_numpy_oracle": algebra_oracle(),
        "causal_prediction": {
            "earliest_gate": "d03 00:20 after exactly 200 child dispatches",
            "required": (
                "finite; ring-one U/T move toward Retry20 and CPU WRF; no "
                "opposite-edge periodic communication"
            ),
            "non_claims": (
                "an early boundary improvement does not prove the late offshore "
                "Ni onset or broad 15:00 V10 drift fixed"
            ),
        },
        "ranked_verdict": [
            "nested advection stencil selection: source discrepancy proven and causal discriminator admitted",
            "W relax map-factor cancellation: real but <=0.0934% and insufficient for the order-one ring drift",
            "target/cadence/corners: falsified by ring-0 target agreement and negative ring-1 projection",
            "V10: separate mechanism absent temporal+causal linkage",
        ],
        "gpu_commands": 0,
        "verdict": "NESTED_ADVECTION_DEGRADE_OFFLINE_CAUSAL_GATE_GREEN",
    }
    payload["proof_sha256"] = canonical_hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["verdict"])
    print(payload["proof_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

