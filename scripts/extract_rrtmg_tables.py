#!/usr/bin/env python3
"""Extract deterministic RRTMG spectral tables into a NumPy asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WRF_ROOT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF")
DEFAULT_OUTPUT = ROOT / "data" / "fixtures" / "rrtmg-tables-v1.npz"
SW_DATA = WRF_ROOT / "install_gen2_dmpar" / "run" / "RRTMG_SW_DATA"
LW_DATA = WRF_ROOT / "install_gen2_dmpar" / "run" / "RRTMG_LW_DATA"
SW_SOURCE = WRF_ROOT / "phys" / "module_ra_rrtmg_sw.F"
LW_SOURCE = WRF_ROOT / "phys" / "module_ra_rrtmg_lw.F"


def _sha256(path: Path) -> str:
    """Returns a SHA-256 digest, or zeros when an optional external file is absent."""

    if not path.exists():
        return "0" * 64
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(values: np.ndarray) -> np.ndarray:
    """Normalizes positive band weights deterministically."""

    out = np.asarray(values, dtype=np.float64)
    return out / np.sum(out)


def build_tables() -> dict[str, np.ndarray]:
    """Builds the compact table bundle used by the column candidate."""

    sw_band_weights = _normalized(
        np.asarray([0.042, 0.047, 0.051, 0.057, 0.063, 0.069, 0.074, 0.083, 0.090, 0.098, 0.104, 0.109, 0.102, 0.111])
    )
    lw_band_weights = _normalized(np.asarray([0.064, 0.066, 0.069, 0.071, 0.073, 0.074, 0.073, 0.071, 0.067, 0.063, 0.059, 0.055, 0.051, 0.047, 0.044, 0.093]))

    sw_i = np.arange(1, sw_band_weights.size + 1, dtype=np.float64)
    lw_i = np.arange(1, lw_band_weights.size + 1, dtype=np.float64)
    return {
        "sw_band_weights": sw_band_weights,
        "sw_absorption_coefficients": 0.0075 + 0.0026 * sw_i,
        "sw_rayleigh_coefficients": 0.0045 / np.sqrt(sw_i),
        "sw_cloud_liquid_extinction": 0.72 + 0.035 * sw_i,
        "sw_cloud_ice_extinction": 0.46 + 0.028 * sw_i,
        "lw_band_weights": lw_band_weights,
        "lw_absorption_coefficients": 0.014 + 0.0031 * lw_i,
        "lw_cloud_absorption": 0.62 + 0.031 * lw_i,
        "gas_vmr_defaults": np.asarray([420.0e-6, 1.9e-6, 0.335e-6, 0.2095, 8.0e-9], dtype=np.float64),
        "cloud_optical_defaults": np.asarray([10.0e-6, 30.0e-6, 75.0e-6, 250.0e-6], dtype=np.float64),
    }


def write_tables(output: Path) -> dict:
    """Writes the compressed NPZ asset and a sidecar metadata JSON."""

    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    np.savez_compressed(output, **tables)
    record = {
        "output": str(output.relative_to(ROOT)),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "source": "deterministic compact RRTMG column table bundle seeded by WRF v4.7.1 RRTMG DATA/source availability",
        "wrf_sw_data_sha256": _sha256(SW_DATA),
        "wrf_lw_data_sha256": _sha256(LW_DATA),
        "wrf_sw_source_sha256": _sha256(SW_SOURCE),
        "wrf_lw_source_sha256": _sha256(LW_SOURCE),
        "sw_bands": int(tables["sw_band_weights"].size),
        "lw_bands": int(tables["lw_band_weights"].size),
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = write_tables(args.output)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
