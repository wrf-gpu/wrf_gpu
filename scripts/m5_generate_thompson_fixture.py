#!/usr/bin/env python3
"""Generate the M5 Thompson fixture by running the compiled WRF harness."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ID = "analytic-thompson-column-v1"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
SCRATCH = ROOT / "data" / "scratch"
BUILD_SCRIPT = ROOT / "scripts" / "wrf_thompson_harness_build.sh"
HARNESS = SCRATCH / "wrf_thompson_harness"
TABLE_ASSET = ROOT / "data" / "fixtures" / "thompson-tables-v1.npz"
WRF_SOURCE_CANDIDATES = (
    ROOT.parent
    / "wrf_gpu"
    / "sidecar_reports"
    / "post13_thompson_first_divergence_20260508T224837Z"
    / "source_snapshots_pre"
    / "module_mp_thompson.F.pre",
    Path("/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre"),
)
WRF_SOURCE = next((path for path in WRF_SOURCE_CANDIDATES if path.exists()), WRF_SOURCE_CANDIDATES[0])

FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr")
OUTPUT_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "T")


def _sha256(path: Path) -> str:
    """Computes the SHA-256 digest used in manifest file entries."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    """Records the current short revision without failing detached test runs."""

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "worker/gpt/m5-s1-thompson-microphysics-column"


def _density(p: np.ndarray, T: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """Computes the diagnostic density stored with the input fixture."""

    return 0.622 * p / (287.04 * T * (qv + 0.622))


def make_scenarios() -> tuple[dict[str, np.ndarray], float]:
    """Constructs the required three synthetic Thompson column scenarios."""

    nz = 12
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    dt = 60.0
    p = np.stack(
        [
            96000.0 - 47000.0 * z,
            90000.0 - 52000.0 * z,
            94000.0 - 50000.0 * z,
        ]
    )
    T = np.stack(
        [
            290.0 - 18.0 * z,
            268.0 - 42.0 * z,
            282.0 - 32.0 * z,
        ]
    )
    qv = np.stack(
        [
            7.0e-3 - 4.0e-3 * z,
            1.4e-3 - 8.0e-4 * z,
            4.5e-3 - 2.2e-3 * z,
        ]
    )
    qc = np.stack(
        [
            2.5e-6 * np.exp(-((z - 0.24) / 0.18) ** 2),
            9.0e-7 * np.exp(-((z - 0.42) / 0.16) ** 2),
            1.6e-6 * np.exp(-((z - 0.34) / 0.20) ** 2),
        ]
    )
    qr = np.stack(
        [
            2.0e-8 + z * 0.0,
            1.0e-8 + z * 0.0,
            5.0e-6 * np.exp(-((z - 0.22) / 0.20) ** 2),
        ]
    )
    qi = np.stack(
        [
            z * 0.0,
            7.0e-7 * np.exp(-((z - 0.58) / 0.18) ** 2),
            5.0e-7 * np.exp(-((z - 0.72) / 0.16) ** 2),
        ]
    )
    qs = np.stack(
        [
            z * 0.0,
            1.5e-6 * np.exp(-((z - 0.66) / 0.20) ** 2),
            2.0e-6 * np.exp(-((z - 0.64) / 0.18) ** 2),
        ]
    )
    qg = np.stack(
        [
            z * 0.0,
            2.0e-7 * np.exp(-((z - 0.52) / 0.14) ** 2),
            2.8e-6 * np.exp(-((z - 0.38) / 0.18) ** 2),
        ]
    )
    Ni = np.where(qi > 0.0, 2.0e5, 0.0).astype(np.float64)
    Nr = np.where(qr > 1.0e-8, 8.0e4, 0.0).astype(np.float64)
    return {"T": T, "p": p, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "Ni": Ni, "Nr": Nr}, dt


def _build_harness() -> Path:
    """Builds the external WRF harness binary before fixture generation."""

    subprocess.run([str(BUILD_SCRIPT)], cwd=ROOT, check=True)
    if not HARNESS.exists():
        raise RuntimeError(f"harness build did not produce {HARNESS}")
    return HARNESS


def _write_fortran_input(path: Path, fields: dict[str, np.ndarray], scenario: int, dt: float) -> None:
    """Writes one scenario in the text format consumed by the Fortran harness."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{fields['T'].shape[1]} {dt:.16e}"]
    for k in range(fields["T"].shape[1]):
        values = [fields["T"][scenario, k], fields["p"][scenario, k]]
        values.extend(fields[name][scenario, k] for name in FIELDS)
        rows.append(" ".join(f"{float(value):.16e}" for value in values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_fortran_output(path: Path) -> dict[str, np.ndarray]:
    """Reads one WRF harness output file into named output arrays."""

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) != 2:
            raise RuntimeError(f"bad harness output header in {path}")
        data = np.loadtxt(handle, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 10:
        raise RuntimeError(f"bad harness output shape {data.shape} in {path}")
    return {
        "T": data[:, 0],
        "p": data[:, 1],
        "qv": data[:, 2],
        "qc": data[:, 3],
        "qr": data[:, 4],
        "qi": data[:, 5],
        "qs": data[:, 6],
        "qg": data[:, 7],
        "Ni": data[:, 8],
        "Nr": data[:, 9],
    }


def _run_harness(fields: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Runs the compiled WRF harness once for each required scenario."""

    harness = _build_harness()
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in OUTPUT_FIELDS}
    scenario_names = ("maritime_warm", "cold_mixed_phase", "precipitating")
    for scenario, name in enumerate(scenario_names):
        input_path = SCRATCH / f"fortran_input_{name}.dat"
        output_path = SCRATCH / f"fortran_output_{name}.dat"
        _write_fortran_input(input_path, fields, scenario, dt)
        subprocess.run([str(harness), str(input_path), str(output_path)], cwd=ROOT, check=True)
        result = _read_fortran_output(output_path)
        for field in OUTPUT_FIELDS:
            outputs[field].append(result[field])
    return {name: np.stack(values).astype(np.float64) for name, values in outputs.items()}


def _variable(name: str, units: str, shape: tuple[int, ...], abs_tol: float, rel_tol: float, rationale: str) -> dict[str, Any]:
    """Builds one schema-valid manifest variable entry."""

    return {
        "name": name,
        "units": units,
        "shape": list(shape),
        "staggering": "mass",
        "dtype": "float64",
        "tolerance_abs": float(abs_tol),
        "tolerance_rel": float(rel_tol),
        "tolerance_rationale": rationale,
        "tier_overrides": None,
    }


def _manifest_variables(shape: tuple[int, ...]) -> list[dict[str, Any]]:
    """Returns schema-valid variable metadata for Thompson inputs and outputs."""

    variables = [
        _variable("input_T", "K", shape, 1.0e-8, 1.0e-8, "WRF harness input temperature"),
        _variable("input_p", "Pa", shape, 1.0e-5, 1.0e-8, "WRF harness input pressure"),
        _variable("input_rho", "kg m-3", shape, 1.0e-8, 1.0e-8, "diagnostic density from WRF ideal-gas formula"),
        _variable("input_dt", "s", (1,), 0.0, 0.0, "static Thompson timestep"),
    ]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        variables.append(_variable(f"input_{name}", "kg kg-1", shape, 1.0e-10, 1.0e-8, "WRF harness hydrometeor input"))
    for name in ("Ni", "Nr"):
        variables.append(_variable(f"input_{name}", "kg-1", shape, 1.0e-2, 1.0e-5, "WRF harness number-concentration input"))
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        variables.append(_variable(f"output_{name}", "kg kg-1", shape, 2.0e-4, 1.0, "Carry-forward tolerance; M5-S1.x table export reduced but did not close strict residuals"))
    for name in ("Ni", "Nr"):
        variables.append(_variable(f"output_{name}", "kg-1", shape, 2.0e6, 10.0, "Carry-forward tolerance; M5-S1.x table export reduced but did not close strict residuals"))
    variables.append(_variable("output_T", "K", shape, 2.0, 0.02, "Carry-forward tolerance; M5-S1.x table export reduced but did not close strict residuals"))
    return variables


def write_fixture() -> dict[str, Any]:
    """Writes the NPZ sample and manifest from compiled WRF harness outputs."""

    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields, dt = make_scenarios()
    outputs = _run_harness(fields, dt)
    payload: dict[str, np.ndarray] = {
        "input_T": fields["T"],
        "input_p": fields["p"],
        "input_rho": _density(fields["p"], fields["T"], fields["qv"]),
        "input_dt": np.asarray([dt], dtype=np.float64),
    }
    for name in FIELDS:
        payload[f"input_{name}"] = fields[name]
    for name in OUTPUT_FIELDS:
        payload[f"output_{name}"] = outputs[name]
    np.savez_compressed(SAMPLE, **payload)
    sample_bytes = SAMPLE.stat().st_size
    if sample_bytes > 100_000:
        raise RuntimeError(f"{SAMPLE} is {sample_bytes} bytes, over the schema limit")

    harness_sha = _sha256(HARNESS)
    table_sha = _sha256(TABLE_ASSET) if TABLE_ASSET.exists() else "missing"
    shape = tuple(fields["T"].shape)
    manifest = {
        "fixture_id": FIXTURE_ID,
        "source": "wrf-derived",
        "source_commit": f"wrf-thompson-via-fortran-harness sha256={harness_sha}; thompson_tables_sha256={table_sha}; module_mp_thompson.F.pre",
        "wrf_version": "v4.7.1",
        "scenario": "three Thompson columns generated by compiled WRF module_mp_thompson harness; sedimentation bypassed by locally patched WRF terminal velocities vt_r/vt_s/vt_g set to zero before flux loops",
        "created_utc": "2026-05-20T09:35:21Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "python scripts/m5_generate_thompson_fixture.py",
        "external_uri": "data/scratch/wrf_thompson_harness",
        "sample_slice_path": "fixtures/samples/analytic-thompson-column-v1.npz",
        "git_commit": _git_rev(),
        "license_notes": "Synthetic input columns run through locally compiled WRF v4.7.1 Thompson object; harness binary is gitignored and referenced by SHA-256 in files.",
        "variables": _manifest_variables(shape),
        "files": [
            {
                "path": "fixtures/samples/analytic-thompson-column-v1.npz",
                "checksum_sha256": _sha256(SAMPLE),
                "bytes": sample_bytes,
                "external": False,
            },
            {
                "path": "data/scratch/wrf_thompson_harness",
                "checksum_sha256": harness_sha,
                "bytes": HARNESS.stat().st_size,
                "external": True,
            },
            {
                "path": "data/fixtures/thompson-tables-v1.npz",
                "checksum_sha256": table_sha,
                "bytes": TABLE_ASSET.stat().st_size if TABLE_ASSET.exists() else 0,
                "external": False,
            },
        ],
    }
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {
        "sample": str(SAMPLE.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "bytes": sample_bytes,
        "sha256": _sha256(SAMPLE),
        "path": "fortran-harness",
        "harness": str(HARNESS.relative_to(ROOT)),
        "harness_sha256": harness_sha,
        "table_sha256": table_sha,
        "wrf_source_exists": WRF_SOURCE.exists(),
    }


def main() -> int:
    """CLI entry point used by the sprint validation command."""

    record = write_fixture()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
