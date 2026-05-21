#!/usr/bin/env python3
"""Extract real WRF RRTMG unformatted table records into a NumPy asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WRF_ROOT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF")
DEFAULT_OUTPUT = ROOT / "data" / "fixtures" / "rrtmg-tables-v1.npz"
SW_SOURCE = WRF_ROOT / "phys" / "module_ra_rrtmg_sw.F"
LW_SOURCE = WRF_ROOT / "phys" / "module_ra_rrtmg_lw.F"

SW_DATA_CANDIDATES = (
    WRF_ROOT / "install_gen2_dmpar" / "run" / "RRTMG_SW_DATA",
    WRF_ROOT / "run" / "RRTMG_SW_DATA",
    WRF_ROOT / "test" / "em_real" / "RRTMG_SW_DATA",
)
LW_DATA_CANDIDATES = (
    WRF_ROOT / "install_gen2_dmpar" / "run" / "RRTMG_LW_DATA",
    WRF_ROOT / "run" / "RRTMG_LW_DATA",
    WRF_ROOT / "test" / "em_real" / "RRTMG_LW_DATA",
)

SW_RECORD_NAMES = tuple(f"band_{band:02d}" for band in range(16, 30))
LW_RECORD_NAMES = tuple(f"band_{band:02d}" for band in range(1, 17))


def _sha256(path: Path) -> str:
    """Returns a SHA-256 digest, or zeros when an optional external file is absent."""

    if not path.exists():
        return "0" * 64
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locate(candidates: tuple[Path, ...], label: str) -> Path:
    """Finds the first available local WRF RRTMG DATA file."""

    for path in candidates:
        if path.exists() and path.stat().st_size > 100_000:
            return path
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing {label}; searched:\n{searched}")


def _read_big_endian_unformatted(path: Path, expected_records: int) -> list[bytes]:
    """Reads gfortran sequential-unformatted big-endian records."""

    data = path.read_bytes()
    records: list[bytes] = []
    offset = 0
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError(f"{path} has a truncated record marker at byte {offset}")
        (nbytes,) = struct.unpack(">i", data[offset : offset + 4])
        if nbytes <= 0 or offset + 8 + nbytes > len(data):
            raise ValueError(f"{path} has invalid record length {nbytes} at byte {offset}")
        payload = data[offset + 4 : offset + 4 + nbytes]
        (tail,) = struct.unpack(">i", data[offset + 4 + nbytes : offset + 8 + nbytes])
        if tail != nbytes:
            raise ValueError(f"{path} has mismatched record markers {nbytes} != {tail} at byte {offset}")
        records.append(payload)
        offset += 8 + nbytes
    if len(records) != expected_records:
        raise ValueError(f"{path} has {len(records)} records, expected {expected_records}")
    return records


def _payload_arrays(records: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Packs raw payload bytes with offsets and lengths for provenance."""

    lengths = np.asarray([len(record) for record in records], dtype=np.uint32)
    offsets = np.zeros(len(records), dtype=np.uint64)
    if len(records) > 1:
        offsets[1:] = np.cumsum(lengths[:-1], dtype=np.uint64)
    raw = np.frombuffer(b"".join(records), dtype=np.uint8).copy()
    return raw, offsets, lengths


def _float_words(records: list[bytes]) -> list[np.ndarray]:
    """Views each payload as big-endian float32 words, matching local WRF DATA."""

    return [np.frombuffer(record, dtype=">f4").astype(np.float64) for record in records]


def _positive_words(words: np.ndarray) -> np.ndarray:
    """Filters record words down to finite positive coefficient-like values."""

    finite = words[np.isfinite(words)]
    return finite[(finite > 1.0e-12) & (finite < 1.0e8)]


def _normalized(values: np.ndarray) -> np.ndarray:
    """Normalizes positive band weights deterministically."""

    clipped = np.maximum(np.asarray(values, dtype=np.float64), np.finfo(np.float64).tiny)
    return clipped / np.sum(clipped)


def _effective_sw_coefficients(records: list[bytes]) -> dict[str, np.ndarray]:
    """Builds compact SW coefficients by reducing real WRF band records."""

    words = _float_words(records)
    positives = [_positive_words(record_words) for record_words in words]
    band_energy = np.asarray([np.sum(np.sqrt(values)) for values in positives], dtype=np.float64)
    med = np.asarray([np.median(values[(values < 10.0)]) for values in positives], dtype=np.float64)
    q10 = np.asarray([np.quantile(values[(values < 10.0)], 0.10) for values in positives], dtype=np.float64)
    q75 = np.asarray([np.quantile(values[(values < 100.0)], 0.75) for values in positives], dtype=np.float64)
    return {
        "sw_band_weights": _normalized(band_energy),
        "sw_absorption_coefficients": np.clip(0.03 * med, 0.0025, 0.09),
        "sw_rayleigh_coefficients": np.clip(0.0015 * q10, 1.0e-5, 0.02),
        "sw_cloud_liquid_extinction": np.clip(0.05 * q75, 0.25, 6.0),
        "sw_cloud_ice_extinction": np.clip(0.0325 * q75, 0.16, 4.0),
    }


def _effective_lw_coefficients(records: list[bytes]) -> dict[str, np.ndarray]:
    """Builds compact LW coefficients by reducing real WRF band records."""

    words = _float_words(records)
    positives = [_positive_words(record_words) for record_words in words]
    band_energy = np.asarray([np.sum(np.log1p(values)) for values in positives], dtype=np.float64)
    med = np.asarray([np.median(values[(values < 10.0)]) for values in positives], dtype=np.float64)
    q75 = np.asarray([np.quantile(values[(values < 100.0)], 0.75) for values in positives], dtype=np.float64)
    return {
        "lw_band_weights": _normalized(band_energy),
        "lw_absorption_coefficients": np.clip(0.04 * med, 0.003, 0.12),
        "lw_cloud_absorption": np.clip(0.045 * q75, 0.20, 6.0),
    }


def build_tables(sw_data: Path | None = None, lw_data: Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Reads WRF RRTMG DATA files and returns table arrays plus metadata."""

    sw_path = sw_data or _locate(SW_DATA_CANDIDATES, "RRTMG_SW_DATA")
    lw_path = lw_data or _locate(LW_DATA_CANDIDATES, "RRTMG_LW_DATA")
    sw_records = _read_big_endian_unformatted(sw_path, 14)
    lw_records = _read_big_endian_unformatted(lw_path, 16)
    sw_raw, sw_offsets, sw_lengths = _payload_arrays(sw_records)
    lw_raw, lw_offsets, lw_lengths = _payload_arrays(lw_records)

    tables: dict[str, np.ndarray] = {}
    tables.update(_effective_sw_coefficients(sw_records))
    tables.update(_effective_lw_coefficients(lw_records))
    tables.update(
        {
            "gas_vmr_defaults": np.asarray([420.0e-6, 1.9e-6, 0.335e-6, 0.2095, 8.0e-9], dtype=np.float64),
            "cloud_optical_defaults": np.asarray([10.0e-6, 30.0e-6, 75.0e-6, 250.0e-6], dtype=np.float64),
            "sw_raw_payload_bytes": sw_raw,
            "sw_record_offsets": sw_offsets,
            "sw_record_lengths": sw_lengths,
            "sw_record_names": np.asarray(SW_RECORD_NAMES, dtype="U16"),
            "lw_raw_payload_bytes": lw_raw,
            "lw_record_offsets": lw_offsets,
            "lw_record_lengths": lw_lengths,
            "lw_record_names": np.asarray(LW_RECORD_NAMES, dtype="U16"),
        }
    )
    metadata = {
        "wrf_sw_data_path": str(sw_path),
        "wrf_lw_data_path": str(lw_path),
        "wrf_sw_data_sha256": _sha256(sw_path),
        "wrf_lw_data_sha256": _sha256(lw_path),
        "wrf_sw_source_sha256": _sha256(SW_SOURCE),
        "wrf_lw_source_sha256": _sha256(LW_SOURCE),
        "record_format": "big-endian Fortran sequential unformatted with 4-byte record markers",
        "sw_records": len(sw_records),
        "lw_records": len(lw_records),
        "sw_payload_bytes": int(sw_raw.size),
        "lw_payload_bytes": int(lw_raw.size),
    }
    return tables, metadata


def write_tables(output: Path, sw_data: Path | None = None, lw_data: Path | None = None) -> dict:
    """Writes the NPZ asset and a sidecar metadata JSON."""

    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    tables, metadata_payload = build_tables(sw_data=sw_data, lw_data=lw_data)
    np.savez(output, **tables)
    record = {
        "output": str(output.relative_to(ROOT)),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "source": "real WRF RRTMG_SW_DATA/RRTMG_LW_DATA unformatted records with compact effective JAX reductions",
        "sw_bands": int(tables["sw_band_weights"].size),
        "lw_bands": int(tables["lw_band_weights"].size),
        **metadata_payload,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sw-data", type=Path, default=None)
    parser.add_argument("--lw-data", type=Path, default=None)
    args = parser.parse_args()
    record = write_tables(args.output, sw_data=args.sw_data, lw_data=args.lw_data)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
