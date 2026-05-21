#!/usr/bin/env python3
"""Generate the M5-S2 MYNN PBL column fixture."""

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
FIXTURE_ID = "analytic-mynn-pbl-column-v1"
SAMPLE = ROOT / "fixtures" / "samples" / f"{FIXTURE_ID}.npz"
MANIFEST = ROOT / "fixtures" / "manifests" / f"{FIXTURE_ID}.yaml"
FULL = ROOT / "data" / "fixtures" / FIXTURE_ID / "full.npz"
SCRATCH = ROOT / "data" / "scratch"
BUILD_SCRIPT = ROOT / "scripts" / "wrf_mynn_harness_build.sh"
HARNESS = SCRATCH / "wrf_mynn_harness"
WRF_SOURCE = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/misc/module_bl_mynn.F90")
WRF_EDMF_SOURCE = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/module_bl_mynnedmf.F90")
WRF_EDMF_OBJECT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf.o")
WRF_EDMF_COMMON_OBJECT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf_common.o")

INPUT_FIELDS = ("u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz")
OUTPUT_FIELDS = ("u", "v", "w", "theta", "qv", "tke", "km", "kh", "el")
TENDENCY_FIELDS = ("du", "dv", "dtheta", "dqv")


def _sha256(path: Path) -> str:
    """Computes the SHA-256 digest used by manifest file entries."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    """Records the current short revision without failing detached runs."""

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "worker/codex/m5-s2-mynn-pbl-column"


def _density(p: np.ndarray, theta: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """Computes a dry-ideal density diagnostic for the synthetic columns."""

    return 0.622 * p / (287.04 * theta * (qv + 0.622))


def make_scenarios() -> tuple[dict[str, np.ndarray], float]:
    """Builds three Canary-relevant MYNN2.5 column scenarios."""

    nz = 16
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    dz = np.ones((3, nz), dtype=np.float64) * 100.0
    dt = 30.0
    u = np.stack(
        [
            7.5 + 2.0 * z,
            4.0 + 8.0 * z,
            11.0 - 4.0 * z,
        ]
    )
    v = np.stack(
        [
            3.0 + 1.0 * z,
            -1.0 + 3.0 * z,
            4.0 + 2.5 * z,
        ]
    )
    w = np.zeros((3, nz), dtype=np.float64)
    theta = np.stack(
        [
            289.0 + 7.0 * z + 3.5 / (1.0 + np.exp(-(z - 0.32) / 0.035)),
            292.0 + 18.0 * z,
            286.0 + 5.0 * z + 7.0 / (1.0 + np.exp(-(z - 0.48) / 0.04)),
        ]
    )
    qv = np.stack(
        [
            8.5e-3 - 4.0e-3 * z,
            4.0e-3 - 2.8e-3 * z,
            7.0e-3 - 5.5e-3 * z,
        ]
    )
    tke = np.stack(
        [
            0.65 * np.exp(-2.8 * z) + 0.03,
            0.18 * np.exp(-1.8 * z) + 0.01,
            0.95 * np.exp(-3.5 * z) + 0.015,
        ]
    )
    p = np.stack(
        [
            97000.0 - 16000.0 * z,
            96000.0 - 17000.0 * z,
            94500.0 - 15500.0 * z,
        ]
    )
    rho = _density(p, theta, qv)
    return {"u": u, "v": v, "w": w, "theta": theta, "qv": qv, "tke": tke, "p": p, "rho": rho, "dz": dz}, dt


def _build_harness() -> Path:
    """Compiles the WRF-object-linked Fortran harness."""

    subprocess.run([str(BUILD_SCRIPT)], cwd=ROOT, check=True)
    if not HARNESS.exists():
        raise RuntimeError(f"harness build did not produce {HARNESS}")
    return HARNESS


def _write_fortran_input(path: Path, fields: dict[str, np.ndarray], scenario: int, dt: float) -> None:
    """Writes one column in the text format consumed by the Fortran harness."""

    rows = [f"{fields['u'].shape[1]} {dt:.16e}"]
    for k in range(fields["u"].shape[1]):
        rows.append(" ".join(f"{float(fields[name][scenario, k]):.16e}" for name in INPUT_FIELDS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_fortran_output(path: Path) -> dict[str, np.ndarray]:
    """Reads one MYNN harness output file."""

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) != 2:
            raise RuntimeError(f"bad harness output header in {path}")
        data = np.loadtxt(handle, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 22:
        raise RuntimeError(f"bad harness output shape {data.shape} in {path}")
    return {
        "u": data[:, 0],
        "v": data[:, 1],
        "w": data[:, 2],
        "theta": data[:, 3],
        "qv": data[:, 4],
        "tke": data[:, 5],
        "km": data[:, 9],
        "kh": data[:, 10],
        "el": data[:, 11],
        "shear": data[:, 12],
        "buoy": data[:, 13],
        "diss": data[:, 14],
        "transport": data[:, 15],
        "surface_theta_flux": data[:, 16],
        "surface_qv_flux": data[:, 17],
        "du": data[:, 18],
        "dv": data[:, 19],
        "dtheta": data[:, 20],
        "dqv": data[:, 21],
    }


def _run_harness(fields: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Runs the compiled harness for each fixture scenario."""

    harness = _build_harness()
    outputs: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "u",
            "v",
            "w",
            "theta",
            "qv",
            "tke",
            "km",
            "kh",
            "el",
            "shear",
            "buoy",
            "diss",
            "transport",
            "surface_theta_flux",
            "surface_qv_flux",
            "du",
            "dv",
            "dtheta",
            "dqv",
        )
    }
    for scenario, name in enumerate(("marine_trade_inversion", "stable_nocturnal", "windy_terrain_mixed")):
        input_path = SCRATCH / f"mynn_input_{name}.dat"
        output_path = SCRATCH / f"mynn_output_{name}.dat"
        _write_fortran_input(input_path, fields, scenario, dt)
        subprocess.run([str(harness), str(input_path), str(output_path)], cwd=ROOT, check=True)
        result = _read_fortran_output(output_path)
        for field, value in result.items():
            outputs[field].append(value)
    return {name: np.stack(values).astype(np.float64) for name, values in outputs.items()}


def _variable(name: str, units: str, shape: tuple[int, ...], abs_tol: float, rel_tol: float, rationale: str) -> dict[str, Any]:
    """Builds one schema-valid variable entry."""

    return {
        "name": name,
        "units": units,
        "shape": list(shape),
        "staggering": "mass",
        "dtype": "float64",
        "tolerance_abs": float(abs_tol),
        "tolerance_rel": float(rel_tol),
        "tolerance_rationale": rationale[:200],
        "tier_overrides": None,
    }


def _manifest_variables(shape: tuple[int, ...]) -> list[dict[str, Any]]:
    """Returns schema-valid metadata for MYNN inputs and outputs."""

    variables = [
        _variable("input_u", "m s-1", shape, 1.0e-10, 1.0e-10, "MYNN harness input U wind"),
        _variable("input_v", "m s-1", shape, 1.0e-10, 1.0e-10, "MYNN harness input V wind"),
        _variable("input_w", "m s-1", shape, 1.0e-10, 1.0e-10, "MYNN harness input W wind"),
        _variable("input_theta", "K", shape, 1.0e-10, 1.0e-10, "MYNN harness input potential temperature"),
        _variable("input_qv", "kg kg-1", shape, 1.0e-12, 1.0e-10, "MYNN harness input vapor mixing ratio"),
        _variable("input_tke", "m2 s-2", shape, 1.0e-12, 1.0e-10, "MYNN harness input prognostic TKE"),
        _variable("input_p", "Pa", shape, 1.0e-8, 1.0e-10, "MYNN harness input pressure"),
        _variable("input_rho", "kg m-3", shape, 1.0e-12, 1.0e-10, "MYNN harness input density"),
        _variable("input_dz", "m", shape, 1.0e-12, 1.0e-10, "MYNN harness input layer depth"),
        _variable("input_dt", "s", (1,), 0.0, 0.0, "static MYNN timestep"),
    ]
    variables.extend(
        [
            _variable("output_u", "m s-1", shape, 5.0e-2, 2.0e-2, "Carry-forward tolerance for WRF-object-linked dry MYNN residuals"),
            _variable("output_v", "m s-1", shape, 5.0e-2, 2.0e-2, "Carry-forward tolerance for WRF-object-linked dry MYNN residuals"),
            _variable("output_w", "m s-1", shape, 1.0e-12, 1.0e-10, "W is carried unchanged in M5-S2 column mode"),
            _variable("output_theta", "K", shape, 1.0e-1, 1.0e-3, "Carry-forward tolerance for WRF-object-linked dry MYNN residuals"),
            _variable("output_qv", "kg kg-1", shape, 5.0e-5, 5.0e-2, "Carry-forward tolerance for WRF-object-linked dry MYNN residuals"),
            _variable("output_tke", "m2 s-2", shape, 8.0e-1, 1.0, "Carry-forward tolerance for WRF-object-linked dry MYNN residuals"),
            _variable("output_km", "m2 s-1", shape, 5.0, 5.0e-1, "WRF MYNN object-linked exchange-coefficient diagnostic"),
            _variable("output_kh", "m2 s-1", shape, 5.0, 5.0e-1, "WRF MYNN object-linked exchange-coefficient diagnostic"),
            _variable("output_el", "m", shape, 20.0, 5.0e-1, "WRF MYNN object-linked master-length diagnostic"),
            _variable("output_du", "m s-2", shape, 1.0e-3, 1.0e-2, "WRF mynn_tendencies U tendency oracle for Tier-2 independent budget"),
            _variable("output_dv", "m s-2", shape, 1.0e-3, 1.0e-2, "WRF mynn_tendencies V tendency oracle for Tier-2 independent budget"),
            _variable("output_dtheta", "K s-1", shape, 1.0e-3, 1.0e-2, "WRF mynn_tendencies theta tendency oracle for Tier-2 independent budget"),
            _variable("output_dqv", "kg kg-1 s-1", shape, 1.0e-3, 1.0e-2, "WRF mynn_tendencies qv tendency oracle for Tier-2 independent budget"),
        ]
    )
    return variables


def write_fixture() -> dict[str, Any]:
    """Writes the NPZ fixture and manifest."""

    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    FULL.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields, dt = make_scenarios()
    outputs = _run_harness(fields, dt)
    payload: dict[str, np.ndarray] = {"input_dt": np.asarray([dt], dtype=np.float64)}
    for name in INPUT_FIELDS:
        payload[f"input_{name}"] = fields[name]
    for name in OUTPUT_FIELDS:
        payload[f"output_{name}"] = outputs[name]
    for name in TENDENCY_FIELDS:
        payload[f"output_{name}"] = outputs[name]
    np.savez_compressed(SAMPLE, **payload)
    np.savez_compressed(FULL, **payload)
    sample_bytes = SAMPLE.stat().st_size
    if sample_bytes > 100_000:
        raise RuntimeError(f"{SAMPLE} is {sample_bytes} bytes, over the schema limit")

    harness_sha = _sha256(HARNESS)
    source_sha = _sha256(WRF_SOURCE) if WRF_SOURCE.exists() else "0" * 64
    edmf_source_sha = _sha256(WRF_EDMF_SOURCE) if WRF_EDMF_SOURCE.exists() else "0" * 64
    edmf_object_status = "present" if WRF_EDMF_OBJECT.exists() and WRF_EDMF_COMMON_OBJECT.exists() else "absent"
    manifest = {
        "fixture_id": FIXTURE_ID,
        "source": "wrf-derived",
        "source_commit": f"wrf-mynnedmf-object-linked-harness sha256={harness_sha}; module_bl_mynn_sha256={source_sha}; module_bl_mynnedmf_sha256={edmf_source_sha}; module_bl_mynnedmf_o={edmf_object_status}",
        "wrf_version": "v4.7.1-MYNN-EDMF-object-linked",
        "scenario": "three MYNN2.5 columns generated by nvfortran harness linked to WRF module_bl_mynnedmf/module_bl_mynnedmf_common objects",
        "created_utc": "2026-05-20T23:10:00Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "python scripts/m5_generate_mynn_fixture.py",
        "external_uri": "data/scratch/wrf_mynn_harness",
        "sample_slice_path": "fixtures/samples/analytic-mynn-pbl-column-v1.npz",
        "git_commit": _git_rev(),
        "license_notes": "Synthetic input columns run through an nvfortran harness linked against compiled WRF MYNN-EDMF module objects.",
        "variables": _manifest_variables(tuple(fields["u"].shape)),
        "files": [
            {"path": "fixtures/samples/analytic-mynn-pbl-column-v1.npz", "checksum_sha256": _sha256(SAMPLE), "bytes": sample_bytes, "external": False},
            {"path": "data/fixtures/analytic-mynn-pbl-column-v1/full.npz", "checksum_sha256": _sha256(FULL), "bytes": FULL.stat().st_size, "external": True},
            {"path": "data/scratch/wrf_mynn_harness", "checksum_sha256": harness_sha, "bytes": HARNESS.stat().st_size, "external": True},
        ],
    }
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return {
        "sample": str(SAMPLE.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "full": str(FULL.relative_to(ROOT)),
        "bytes": sample_bytes,
        "sha256": _sha256(SAMPLE),
        "harness": str(HARNESS.relative_to(ROOT)),
        "harness_sha256": harness_sha,
        "source_sha256": source_sha,
        "module_bl_mynnedmf_sha256": edmf_source_sha,
        "module_bl_mynnedmf_o": edmf_object_status,
    }


def main() -> int:
    """CLI entry point used by the sprint validation command."""

    record = write_fixture()
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
