"""Extractor for the first Canary WRF-derived M1 fixture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import zipfile

import numpy as np
import yaml

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover - exercised only in missing-dep envs
    raise RuntimeError("netCDF4 is required to extract WRF NetCDF-4 fixture slices") from exc


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ID = "canary-wrf-d01-20260518T18-tslice-v1"
# Source-of-record is project-owned and immutable. We previously borrowed from a
# Gen2 nightly-run path, but Gen2 rotates wrfouts (its scheduler overwrites the
# directory on each new run). To make this fixture reproducible long-term, the
# project copied a representative wrfout into its own storage at the path below
# (sha256 5cb92e491d0d7ccf5ba1f4835cbca73a82c4e1b2db75fdcaa6fa12a9093301e1).
SOURCE_WRFOUT = Path("/mnt/data/wrf_gpu2/source_wrfouts/wrfout_d01_2026-05-18_18:00:00")
EXTERNAL_DIR = ROOT / "data/fixtures" / FIXTURE_ID
FULL_PATH = EXTERNAL_DIR / "full.npz"
CHECKSUM_PATH = EXTERNAL_DIR / "checksums.txt"
SAMPLE_PATH = ROOT / "fixtures/samples" / f"{FIXTURE_ID}.npz"
MANIFEST_PATH = ROOT / "fixtures/manifests" / f"{FIXTURE_ID}.yaml"
CREATED_UTC = datetime(2026, 5, 19, 0, 0, 0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
GENERATION_COMMAND = "python scripts/extract_canary_wrf_fixture.py"


@dataclass(frozen=True)
class VariableMeta:
    name: str
    units: str
    staggering: str
    tolerance_abs: float
    tolerance_rel: float
    tolerance_rationale: str


VARIABLES = (
    VariableMeta("T", "K", "mass", 1.0e-3, 1.0e-6, "WRF fp32 perturbation theta; interior z0:10 y24:32 x42:50"),
    VariableMeta("U", "m s-1", "u", 1.0e-4, 1.0e-6, "WRF fp32 u-face wind; interior z0:10 y24:32 x42:51"),
    VariableMeta("V", "m s-1", "v", 1.0e-4, 1.0e-6, "WRF fp32 v-face wind; interior z0:10 y24:33 x42:50"),
    VariableMeta("QVAPOR", "kg kg-1", "mass", 1.0e-8, 1.0e-6, "WRF fp32 water vapor mixing ratio; interior 10x8x8 slice"),
    VariableMeta("P", "Pa", "mass", 1.0e-2, 1.0e-6, "WRF fp32 perturbation pressure; interior 10x8x8 slice"),
    VariableMeta("PB", "Pa", "mass", 1.0e-2, 1.0e-6, "WRF fp32 base-state pressure; interior 10x8x8 slice"),
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(2026, 5, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _slice_arrays(source_wrfout: Path) -> dict[str, np.ndarray]:
    if not source_wrfout.is_file():
        raise FileNotFoundError(source_wrfout)

    z = slice(0, 10)
    y = slice(24, 32)
    x = slice(42, 50)
    with Dataset(source_wrfout, "r") as dataset:
        arrays = {
            "T": np.asarray(dataset.variables["T"][0, z, y, x], dtype=np.float64),
            "U": np.asarray(dataset.variables["U"][0, z, y, slice(42, 51)], dtype=np.float64),
            "V": np.asarray(dataset.variables["V"][0, z, slice(24, 33), x], dtype=np.float64),
            "QVAPOR": np.asarray(dataset.variables["QVAPOR"][0, z, y, x], dtype=np.float64),
            "P": np.asarray(dataset.variables["P"][0, z, y, x], dtype=np.float64),
            "PB": np.asarray(dataset.variables["PB"][0, z, y, x], dtype=np.float64),
        }
    return arrays


def _manifest(source_wrfout: Path, source_hash: str, full_hash: str, sample_hash: str, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    rel_full = FULL_PATH.relative_to(ROOT).as_posix()
    rel_sample = SAMPLE_PATH.relative_to(ROOT).as_posix()
    return {
        "fixture_id": FIXTURE_ID,
        "source": "wrf-derived",
        "source_commit": f"path={source_wrfout.as_posix()} sha256={source_hash}",
        "wrf_version": "4.7.1",
        "scenario": "Canary Islands WRF L3 d01 valid 2026-05-18 13:00 UTC, single interior timestep slice",
        "created_utc": CREATED_UTC,
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": GENERATION_COMMAND,
        "external_uri": rel_full,
        "sample_slice_path": rel_sample,
        "git_commit": "worker/gpt/m1-canary-wrf-derived-fixture",
        "license_notes": "Derived from local Gen2 WRF operational run for project validation; bulk payload remains under data/.",
        "variables": [
            {
                "name": meta.name,
                "units": meta.units,
                "shape": [int(dim) for dim in arrays[meta.name].shape],
                "staggering": meta.staggering,
                "dtype": "float32",
                "tolerance_abs": meta.tolerance_abs,
                "tolerance_rel": meta.tolerance_rel,
                "tolerance_rationale": meta.tolerance_rationale,
                "tier_overrides": None,
            }
            for meta in VARIABLES
        ],
        "files": [
            {
                "path": rel_full,
                "checksum_sha256": full_hash,
                "bytes": FULL_PATH.stat().st_size,
                "external": True,
            },
            {
                "path": rel_sample,
                "checksum_sha256": sample_hash,
                "bytes": SAMPLE_PATH.stat().st_size,
                "external": False,
            },
        ],
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _write_checksums(full_hash: str, sample_hash: str) -> None:
    CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_PATH.write_text(
        "\n".join(
            [
                f"{full_hash}  {FULL_PATH.name}",
                f"{sample_hash}  {SAMPLE_PATH.relative_to(ROOT).as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def extract_fixture(source_wrfout: Path = SOURCE_WRFOUT) -> list[Path]:
    """Regenerate the Canary WRF-derived fixture and return touched paths."""

    arrays64 = _slice_arrays(source_wrfout)
    arrays32 = {name: value.astype(np.float32) for name, value in arrays64.items()}

    _write_deterministic_npz(FULL_PATH, arrays64)
    _write_deterministic_npz(SAMPLE_PATH, arrays32)

    source_hash = sha256_file(source_wrfout)
    full_hash = sha256_file(FULL_PATH)
    sample_hash = sha256_file(SAMPLE_PATH)
    _write_checksums(full_hash, sample_hash)
    _write_manifest(MANIFEST_PATH, _manifest(source_wrfout, source_hash, full_hash, sample_hash, arrays32))

    return [FULL_PATH, CHECKSUM_PATH, SAMPLE_PATH, MANIFEST_PATH]


def extraction_summary(paths: list[Path]) -> str:
    records = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size} for path in paths]
    return json.dumps({"fixture_id": FIXTURE_ID, "files": records}, indent=2, sort_keys=True)
