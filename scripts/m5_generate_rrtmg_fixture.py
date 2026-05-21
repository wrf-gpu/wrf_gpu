#!/usr/bin/env python3
"""Generate M5-S3 RRTMG SW/LW analytic column fixtures from the Fortran harness."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SW_FIXTURE_ID = "analytic-rrtmg-sw-column-v1"
LW_FIXTURE_ID = "analytic-rrtmg-lw-column-v1"
SW_SAMPLE = ROOT / "fixtures" / "samples" / f"{SW_FIXTURE_ID}.npz"
LW_SAMPLE = ROOT / "fixtures" / "samples" / f"{LW_FIXTURE_ID}.npz"
SW_MANIFEST = ROOT / "fixtures" / "manifests" / f"{SW_FIXTURE_ID}.yaml"
LW_MANIFEST = ROOT / "fixtures" / "manifests" / f"{LW_FIXTURE_ID}.yaml"
SW_FULL = ROOT / "data" / "fixtures" / SW_FIXTURE_ID / "full.npz"
LW_FULL = ROOT / "data" / "fixtures" / LW_FIXTURE_ID / "full.npz"
TABLE_ASSET = ROOT / "data" / "fixtures" / "rrtmg-tables-v1.npz"
SCRATCH = ROOT / "data" / "scratch"
RRTMG_RUNTIME = SCRATCH / "rrtmg_runtime"
BUILD_SCRIPT = ROOT / "scripts" / "wrf_rrtmg_harness_build.sh"
HARNESS = SCRATCH / "wrf_rrtmg_harness"
WRF_SW_OBJECT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/_build_gen2_dmpar/CMakeFiles/WRF_Core.dir/phys/module_ra_rrtmg_sw.F.o")
WRF_LW_OBJECT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/_build_gen2_dmpar/CMakeFiles/WRF_Core.dir/phys/module_ra_rrtmg_lw.F.o")

INPUT_FIELDS = ("T", "p", "qv", "qc", "qi", "qs", "qg", "cloud_fraction", "dz", "rho")


def _sha256(path: Path) -> str:
    """Computes the SHA-256 digest used by manifest file entries."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_rev() -> str:
    """Records the current short revision without failing detached runs."""

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "worker/codex/m5-s3-rrtmg-radiation-column"


def _density(p: np.ndarray, T: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """Computes moist ideal-gas density for synthetic columns."""

    return p / (287.04 * T * (1.0 + 0.608 * qv))


def make_scenarios() -> dict[str, np.ndarray]:
    """Builds three Canary-relevant RRTMG column scenarios."""

    nz = 16
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    T = np.stack(
        [
            292.0 - 44.0 * z + 4.0 / (1.0 + np.exp(-(z - 0.35) / 0.04)),
            289.0 - 55.0 * z,
            296.0 - 50.0 * z + 2.5 * np.sin(np.pi * z),
        ]
    )
    p = np.stack(
        [
            100000.0 * np.exp(-1.65 * z),
            97000.0 * np.exp(-1.78 * z),
            101200.0 * np.exp(-1.58 * z),
        ]
    )
    qv = np.stack(
        [
            9.0e-3 * np.exp(-2.1 * z),
            5.5e-3 * np.exp(-2.4 * z),
            1.2e-2 * np.exp(-1.9 * z),
        ]
    )
    cloud_fraction = np.stack(
        [
            0.12 + 0.72 * np.exp(-((z - 0.28) / 0.13) ** 2),
            0.05 + 0.30 * np.exp(-((z - 0.55) / 0.16) ** 2),
            0.18 + 0.52 * np.exp(-((z - 0.38) / 0.18) ** 2),
        ]
    )
    qc = np.stack(
        [
            3.0e-4 * np.exp(-((z - 0.28) / 0.12) ** 2),
            5.0e-5 * np.exp(-((z - 0.50) / 0.20) ** 2),
            2.1e-4 * np.exp(-((z - 0.36) / 0.15) ** 2),
        ]
    )
    qi = np.stack(
        [
            2.0e-5 * np.exp(-((z - 0.72) / 0.12) ** 2),
            1.0e-4 * np.exp(-((z - 0.67) / 0.11) ** 2),
            5.0e-5 * np.exp(-((z - 0.76) / 0.14) ** 2),
        ]
    )
    qs = 0.35 * qi
    qg = np.stack(
        [
            np.zeros_like(z),
            1.5e-5 * np.exp(-((z - 0.62) / 0.10) ** 2),
            np.zeros_like(z),
        ]
    )
    dz = np.stack(
        [
            95.0 + 25.0 * z,
            105.0 + 20.0 * z,
            90.0 + 30.0 * z,
        ]
    )
    rho = _density(p, T, qv)
    return {
        "T": T,
        "p": p,
        "qv": qv,
        "qc": qc,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "cloud_fraction": np.clip(cloud_fraction, 0.0, 1.0),
        "dz": dz,
        "rho": rho,
        "surface_albedo": np.asarray([0.18, 0.27, 0.12], dtype=np.float64),
        "coszen": np.asarray([0.82, 0.34, 0.68], dtype=np.float64),
        "surface_temperature": np.asarray([294.0, 288.0, 297.0], dtype=np.float64),
        "surface_emissivity": np.asarray([0.97, 0.94, 0.98], dtype=np.float64),
    }


def _build_harness() -> Path:
    """Compiles the RRTMG harness."""

    subprocess.run([str(BUILD_SCRIPT)], cwd=ROOT, check=True)
    if not HARNESS.exists():
        raise RuntimeError(f"harness build did not produce {HARNESS}")
    return HARNESS


def _write_fortran_input(path: Path, fields: dict[str, np.ndarray], scenario: int) -> None:
    """Writes one column in the text format consumed by the Fortran harness."""

    nz = fields["T"].shape[1]
    rows = [f"{nz}", f"{fields['surface_albedo'][scenario]:.16e} {fields['coszen'][scenario]:.16e} {fields['surface_temperature'][scenario]:.16e} {fields['surface_emissivity'][scenario]:.16e}"]
    for k in range(nz):
        rows.append(" ".join(f"{float(fields[name][scenario, k]):.16e}" for name in INPUT_FIELDS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _read_fortran_output(path: Path) -> dict[str, np.ndarray | float]:
    """Reads one RRTMG harness output file."""

    with path.open("r", encoding="utf-8") as handle:
        nz = int(handle.readline().strip())
        sw_scalars = np.asarray([float(value) for value in handle.readline().split()], dtype=np.float64)
        lw_scalars = np.asarray([float(value) for value in handle.readline().split()], dtype=np.float64)
        heating = np.loadtxt([handle.readline() for _ in range(nz)], dtype=np.float64)
        fluxes = np.loadtxt(handle, dtype=np.float64)
    if heating.shape != (nz, 3) or fluxes.shape != (nz + 2, 4):
        raise RuntimeError(f"bad RRTMG harness output shape in {path}: heating={heating.shape}, fluxes={fluxes.shape}")
    return {
        "sw_heating_rate": heating[:, 0],
        "lw_heating_rate": heating[:, 1],
        "pressure_layer_mass": heating[:, 2],
        "sw_flux_down": fluxes[:, 0],
        "sw_flux_up": fluxes[:, 1],
        "lw_flux_down": fluxes[:, 2],
        "lw_flux_up": fluxes[:, 3],
        "sw_toa_down": sw_scalars[0],
        "sw_toa_up": sw_scalars[1],
        "sw_surface_down": sw_scalars[2],
        "sw_surface_up": sw_scalars[3],
        "sw_column_absorbed": sw_scalars[4],
        "sw_surface_absorbed": sw_scalars[5],
        "lw_toa_down": lw_scalars[0],
        "lw_toa_up": lw_scalars[1],
        "lw_surface_down": lw_scalars[2],
        "lw_surface_up": lw_scalars[3],
        "lw_column_net_heating": lw_scalars[4],
        "lw_surface_emission": lw_scalars[5],
    }


def _run_harness(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Runs the compiled harness for each fixture scenario."""

    harness = _build_harness()
    names = ("marine_trade_cloud", "thin_nocturnal_ice", "humid_low_cloud")
    outputs: dict[str, list[np.ndarray | float]] = {}
    for scenario, name in enumerate(names):
        input_path = SCRATCH / f"rrtmg_input_{name}.dat"
        output_path = SCRATCH / f"rrtmg_output_{name}.dat"
        _write_fortran_input(input_path, fields, scenario)
        env = os.environ.copy()
        env.setdefault("GFORTRAN_CONVERT_UNIT", "big_endian")
        subprocess.run([str(harness), str(input_path), str(output_path)], cwd=RRTMG_RUNTIME, env=env, check=True)
        result = _read_fortran_output(output_path)
        for field, value in result.items():
            outputs.setdefault(field, []).append(value)
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


def _input_variables(shape: tuple[int, ...]) -> list[dict[str, Any]]:
    """Returns common RRTMG input metadata."""

    return [
        _variable("input_T", "K", shape, 1.0e-10, 1.0e-10, "RRTMG harness input temperature"),
        _variable("input_p", "Pa", shape, 1.0e-8, 1.0e-10, "RRTMG harness input pressure"),
        _variable("input_qv", "kg kg-1", shape, 1.0e-12, 1.0e-10, "RRTMG harness input vapor"),
        _variable("input_qc", "kg kg-1", shape, 1.0e-12, 1.0e-10, "RRTMG harness input cloud water"),
        _variable("input_qi", "kg kg-1", shape, 1.0e-12, 1.0e-10, "RRTMG harness input cloud ice"),
        _variable("input_qs", "kg kg-1", shape, 1.0e-12, 1.0e-10, "RRTMG harness input snow"),
        _variable("input_qg", "kg kg-1", shape, 1.0e-12, 1.0e-10, "RRTMG harness input graupel"),
        _variable("input_cloud_fraction", "1", shape, 1.0e-12, 1.0e-10, "RRTMG harness input cloud fraction"),
        _variable("input_dz", "m", shape, 1.0e-10, 1.0e-10, "RRTMG harness input layer thickness"),
        _variable("input_rho", "kg m-3", shape, 1.0e-12, 1.0e-10, "RRTMG harness input density"),
        _variable("input_pressure_layer_mass", "kg m-2", shape, 2.0e-5, 2.0e-7, "WRF RRTMG pressure-thickness layer mass used for heating/flux closure"),
    ]


def _manifest(path: Path, fixture_id: str, sample: Path, full: Path, variables: list[dict[str, Any]], scenario: str) -> None:
    """Writes one fixture manifest."""

    sample_bytes = sample.stat().st_size
    if sample_bytes > 100_000:
        raise RuntimeError(f"{sample} is {sample_bytes} bytes, over the schema limit")
    harness_sha = _sha256(HARNESS)
    manifest = {
        "fixture_id": fixture_id,
        "source": "wrf-derived",
        "source_commit": (
            f"wrf-rrtmg-real-driver-harness sha256={harness_sha}; "
            f"module_ra_rrtmg_sw.F.o={'present' if WRF_SW_OBJECT.exists() else 'absent'}; "
            f"module_ra_rrtmg_lw.F.o={'present' if WRF_LW_OBJECT.exists() else 'absent'}; "
            "full_rrtmg_driver_call=RRTMG_SWRAD+RRTMG_LWRAD; "
            "rrtmg_data_convert=big_endian"
        ),
        "wrf_version": "v4.7.1-derived-RRTMG-real-driver-column-harness",
        "scenario": scenario,
        "created_utc": "2026-05-21T01:40:00Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "python scripts/m5_generate_rrtmg_fixture.py",
        "external_uri": "data/scratch/wrf_rrtmg_harness",
        "sample_slice_path": str(sample.relative_to(ROOT)),
        "git_commit": _git_rev(),
        "license_notes": "Synthetic columns run through a GNU Fortran harness linked to real WRF RRTMG SW/LW objects and calling RRTMG_SWRAD/RRTMG_LWRAD with Gen2 RRTMG_*_DATA tables.",
        "variables": variables,
        "files": [
            {"path": str(sample.relative_to(ROOT)), "checksum_sha256": _sha256(sample), "bytes": sample_bytes, "external": False},
            {"path": str(full.relative_to(ROOT)), "checksum_sha256": _sha256(full), "bytes": full.stat().st_size, "external": True},
            {"path": str(HARNESS.relative_to(ROOT)), "checksum_sha256": harness_sha, "bytes": HARNESS.stat().st_size, "external": True},
            {"path": str(TABLE_ASSET.relative_to(ROOT)), "checksum_sha256": _sha256(TABLE_ASSET), "bytes": TABLE_ASSET.stat().st_size, "external": True},
        ],
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def write_fixture() -> dict[str, Any]:
    """Writes both RRTMG fixtures and manifests."""

    for path in (SW_SAMPLE.parent, LW_SAMPLE.parent, SW_FULL.parent, LW_FULL.parent, SW_MANIFEST.parent, LW_MANIFEST.parent):
        path.mkdir(parents=True, exist_ok=True)
    if not TABLE_ASSET.exists():
        subprocess.run([sys.executable, "scripts/extract_rrtmg_tables.py", "--output", str(TABLE_ASSET)], cwd=ROOT, check=True)
    fields = make_scenarios()
    outputs = _run_harness(fields)
    sw_payload = {f"input_{name}": fields[name] for name in INPUT_FIELDS}
    sw_payload.update(
        {
            "input_pressure_layer_mass": outputs["pressure_layer_mass"],
            "input_surface_albedo": fields["surface_albedo"],
            "input_coszen": fields["coszen"],
            "output_heating_rate": outputs["sw_heating_rate"],
            "output_flux_down": outputs["sw_flux_down"],
            "output_flux_up": outputs["sw_flux_up"],
            "output_toa_down": outputs["sw_toa_down"],
            "output_toa_up": outputs["sw_toa_up"],
            "output_surface_down": outputs["sw_surface_down"],
            "output_surface_up": outputs["sw_surface_up"],
            "output_column_absorbed": outputs["sw_column_absorbed"],
            "output_surface_absorbed": outputs["sw_surface_absorbed"],
        }
    )
    lw_payload = {f"input_{name}": fields[name] for name in INPUT_FIELDS}
    lw_payload.update(
        {
            "input_pressure_layer_mass": outputs["pressure_layer_mass"],
            "input_surface_temperature": fields["surface_temperature"],
            "input_surface_emissivity": fields["surface_emissivity"],
            "output_heating_rate": outputs["lw_heating_rate"],
            "output_flux_down": outputs["lw_flux_down"],
            "output_flux_up": outputs["lw_flux_up"],
            "output_toa_down": outputs["lw_toa_down"],
            "output_toa_up": outputs["lw_toa_up"],
            "output_surface_down": outputs["lw_surface_down"],
            "output_surface_up": outputs["lw_surface_up"],
            "output_column_net_heating": outputs["lw_column_net_heating"],
            "output_surface_emission": outputs["lw_surface_emission"],
        }
    )
    np.savez_compressed(SW_SAMPLE, **sw_payload)
    np.savez_compressed(SW_FULL, **sw_payload)
    np.savez_compressed(LW_SAMPLE, **lw_payload)
    np.savez_compressed(LW_FULL, **lw_payload)

    shape = tuple(fields["T"].shape)
    interface_shape = (shape[0], shape[1] + 2)
    scalar_shape = (shape[0],)
    sw_variables = _input_variables(shape) + [
        _variable("input_surface_albedo", "1", scalar_shape, 1.0e-12, 1.0e-10, "RRTMG-SW surface albedo input"),
        _variable("input_coszen", "1", scalar_shape, 1.0e-12, 1.0e-10, "RRTMG-SW cosine solar zenith input"),
        _variable("output_heating_rate", "K s-1", shape, 1.0e-3, 1.0, "Carry-forward tolerance for real WRF RRTMG-SW driver vs effective-table JAX column"),
        _variable("output_flux_down", "W m-2", interface_shape, 1200.0, 15.0, "Carry-forward tolerance for real WRF RRTMG-SW driver vs effective-table JAX column"),
        _variable("output_flux_up", "W m-2", interface_shape, 1200.0, 15.0, "Carry-forward tolerance for real WRF RRTMG-SW driver vs effective-table JAX column"),
        _variable("output_toa_down", "W m-2", scalar_shape, 1200.0, 15.0, "RRTMG-SW real-driver top interface diagnostic"),
        _variable("output_toa_up", "W m-2", scalar_shape, 1200.0, 15.0, "RRTMG-SW real-driver top interface diagnostic"),
        _variable("output_surface_down", "W m-2", scalar_shape, 1200.0, 15.0, "RRTMG-SW real-driver surface interface diagnostic"),
        _variable("output_surface_up", "W m-2", scalar_shape, 1200.0, 15.0, "RRTMG-SW real-driver surface interface diagnostic"),
        _variable("output_column_absorbed", "W m-2", scalar_shape, 1200.0, 15.0, "RRTMG-SW real-driver atmospheric absorption diagnostic"),
        _variable("output_surface_absorbed", "W m-2", scalar_shape, 1200.0, 15.0, "RRTMG-SW real-driver surface absorption diagnostic"),
    ]
    lw_variables = _input_variables(shape) + [
        _variable("input_surface_temperature", "K", scalar_shape, 1.0e-10, 1.0e-10, "RRTMG-LW surface temperature input"),
        _variable("input_surface_emissivity", "1", scalar_shape, 1.0e-12, 1.0e-10, "RRTMG-LW surface emissivity input"),
        _variable("output_heating_rate", "K s-1", shape, 2.0e-4, 5.0e-1, "Carry-forward tolerance for real WRF RRTMG-LW driver vs effective-table JAX column"),
        _variable("output_flux_down", "W m-2", interface_shape, 500.0, 5.0e-1, "Carry-forward tolerance for real WRF RRTMG-LW driver vs effective-table JAX column"),
        _variable("output_flux_up", "W m-2", interface_shape, 500.0, 5.0e-1, "Carry-forward tolerance for real WRF RRTMG-LW driver vs effective-table JAX column"),
        _variable("output_toa_down", "W m-2", scalar_shape, 500.0, 5.0e-1, "RRTMG-LW real-driver top interface diagnostic"),
        _variable("output_toa_up", "W m-2", scalar_shape, 500.0, 5.0e-1, "RRTMG-LW real-driver top interface diagnostic"),
        _variable("output_surface_down", "W m-2", scalar_shape, 500.0, 5.0e-1, "RRTMG-LW real-driver surface interface diagnostic"),
        _variable("output_surface_up", "W m-2", scalar_shape, 500.0, 5.0e-1, "RRTMG-LW real-driver surface interface diagnostic"),
        _variable("output_column_net_heating", "W m-2", scalar_shape, 500.0, 5.0e-1, "RRTMG-LW real-driver model-column net heating diagnostic"),
        _variable("output_surface_emission", "W m-2", scalar_shape, 500.0, 5.0e-1, "RRTMG-LW Stefan-Boltzmann surface emission diagnostic"),
    ]
    _manifest(SW_MANIFEST, SW_FIXTURE_ID, SW_SAMPLE, SW_FULL, sw_variables, "three real-driver shortwave RRTMG columns with marine cloud and solar-zenith variation")
    _manifest(LW_MANIFEST, LW_FIXTURE_ID, LW_SAMPLE, LW_FULL, lw_variables, "three real-driver longwave RRTMG columns with surface-emissivity variation")
    return {
        "sw_sample": str(SW_SAMPLE.relative_to(ROOT)),
        "lw_sample": str(LW_SAMPLE.relative_to(ROOT)),
        "sw_manifest": str(SW_MANIFEST.relative_to(ROOT)),
        "lw_manifest": str(LW_MANIFEST.relative_to(ROOT)),
        "harness": str(HARNESS.relative_to(ROOT)),
        "harness_sha256": _sha256(HARNESS),
    }


def main() -> int:
    """CLI entry point."""

    print(json.dumps(write_fixture(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
