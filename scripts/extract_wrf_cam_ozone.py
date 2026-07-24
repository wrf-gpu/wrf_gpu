#!/usr/bin/env python3
"""Extract WRF's CAM monthly ozone climatology into a deterministic NPZ asset."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read_f4(path: Path) -> np.ndarray:
    return np.asarray(
        [float(token) for token in path.read_text(encoding="ascii").split()],
        dtype=np.float32,
    )


def _npy_payload(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_payload(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    run = args.wrf_root.resolve() / "run"
    sources = {
        "latitude": run / "ozone_lat.formatted",
        "pressure": run / "ozone_plev.formatted",
        "mixing_ratio": run / "ozone.formatted",
    }
    if any(not path.is_file() for path in sources.values()):
        raise SystemExit(f"missing WRF ozone source: {sources}")

    latitude = _read_f4(sources["latitude"])
    pressure_hpa = _read_f4(sources["pressure"])
    mixing_ratio = _read_f4(sources["mixing_ratio"])
    if latitude.shape != (64,) or pressure_hpa.shape != (59,) or mixing_ratio.shape != (12 * 64 * 59,):
        raise SystemExit(
            "unexpected WRF ozone table dimensions: "
            f"lat={latitude.shape} pressure={pressure_hpa.shape} ozone={mixing_ratio.shape}"
        )
    arrays = {
        "latitude_deg": latitude,
        "pressure_pa": (pressure_hpa * np.float32(100.0)).astype(np.float32),
        # WRF read order: month outermost, then latitude, then pressure level.
        "ozone_vmr": mixing_ratio.reshape(12, 64, 59),
    }
    if not np.all(np.diff(arrays["latitude_deg"]) > 0.0) or not np.all(
        np.diff(arrays["pressure_pa"]) > 0.0
    ):
        raise SystemExit("WRF CAM ozone coordinates are not strictly increasing")
    if not all(np.isfinite(value).all() and value.dtype == np.float32 for value in arrays.values()):
        raise SystemExit("WRF CAM ozone asset contains invalid values")

    _write_deterministic_npz(args.output.resolve(), arrays)
    record = {
        "schema": "wrf-cam-ozone-v1",
        "description": (
            "WRF v4 ozone.formatted monthly VMR on its 64-latitude x 59-pressure grid; "
            "read order and float32 storage match module_ra_cam_support.F::oznini"
        ),
        "asset": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output.resolve()),
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "payload_sha256": _array_sha(value),
                }
                for name, value in arrays.items()
            },
        },
        "wrf_sources": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
    }
    args.manifest.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.manifest.resolve().write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record["asset"], sort_keys=True))


if __name__ == "__main__":
    main()
