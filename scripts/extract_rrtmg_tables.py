#!/usr/bin/env python3
"""Extract real WRF RRTMG unformatted table records into a NumPy asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _resolve_wrf_root() -> Path:
    """Resolve the WRF source tree that carries the RRTMG ``.F`` + DATA tables.

    The RRTMG coefficient tables are parsed from real WRF Fortran source. A private
    workstation path was hardcoded, which makes a clean clone (or a corpus with the
    artifacts tree purged) un-runnable -- the exact standalone out-of-the-box
    failure mode this guards. Precedence:
      1. ``$GPUWRF_WRF_SRC`` (explicit override),
      2. the historical workstation artifacts path (kept for the original layout),
      3. the pristine WRF build (``/home/enric/src/wrf_pristine/WRF``; project memory
         "WRF ground truth BUILT 2026-05-29"),
      4. a checkout-relative ``external/WRF``.
    Returns the first that has ``phys/module_ra_rrtmg_lw.F``; falls back to the
    historical path (so the error message still names a concrete location).
    """

    candidates = []
    env = os.environ.get("GPUWRF_WRF_SRC", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF"),
            Path("/home/enric/src/wrf_pristine/WRF"),
            ROOT / "external" / "WRF",
        ]
    )
    for cand in candidates:
        if (cand / "phys" / "module_ra_rrtmg_lw.F").is_file():
            return cand
    return candidates[1] if len(candidates) > 1 else candidates[0]


WRF_ROOT = _resolve_wrf_root()
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
ORIGINAL_GPOINT_WEIGHTS = np.asarray(
    [
        0.1527534276,
        0.1491729617,
        0.1420961469,
        0.1316886544,
        0.1181945205,
        0.1019300893,
        0.0832767040,
        0.0626720116,
        0.0424925000,
        0.0046269894,
        0.0038279891,
        0.0030260086,
        0.0022199750,
        0.0014140010,
        0.0005330000,
        0.0000750000,
    ],
    dtype=np.float64,
)
SW_REDUCED_GROUPS = (
    (2, 2, 2, 2, 4, 4),
    (1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2),
    (1, 1, 1, 1, 2, 2, 4, 4),
    (1, 1, 1, 1, 2, 2, 4, 4),
    (1, 1, 1, 1, 1, 1, 1, 1, 2, 6),
    (1, 1, 1, 1, 1, 1, 1, 1, 2, 6),
    (8, 8),
    (2, 2, 1, 1, 1, 1, 1, 1, 2, 4),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (1, 1, 2, 2, 4, 6),
    (1, 1, 2, 2, 4, 6),
    (1, 1, 1, 1, 1, 1, 4, 6),
    (1, 1, 2, 2, 4, 6),
    (1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1),
)
LW_REDUCED_GROUPS = (
    (1, 1, 2, 2, 2, 2, 2, 2, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    (2, 2, 2, 2, 4, 4),
    (1, 1, 2, 2, 2, 2, 3, 3),
    (1, 1, 1, 1, 2, 2, 4, 4),
    (3, 3, 4, 6),
    (8, 8),
    (8, 8),
    (4, 12),
)
LW_DELWAVE = np.asarray(
    [340.0, 150.0, 130.0, 70.0, 120.0, 160.0, 100.0, 100.0, 210.0, 90.0, 320.0, 280.0, 170.0, 130.0, 220.0, 650.0],
    dtype=np.float64,
)
REFERENCE_PRESSURE_MB = np.asarray(
    [
        1.05363e3,
        8.62642e2,
        7.06272e2,
        5.78246e2,
        4.73428e2,
        3.87610e2,
        3.17348e2,
        2.59823e2,
        2.12725e2,
        1.74164e2,
        1.42594e2,
        1.16746e2,
        9.55835e1,
        7.82571e1,
        6.40715e1,
        5.24573e1,
        4.29484e1,
        3.51632e1,
        2.87892e1,
        2.35706e1,
        1.92980e1,
        1.57998e1,
        1.29358e1,
        1.05910e1,
        8.67114,
        7.09933,
        5.81244,
        4.75882,
        3.89619,
        3.18993,
        2.61170,
        2.13828,
        1.75067,
        1.43333,
        1.17351,
        9.60789e-1,
        7.86628e-1,
        6.44036e-1,
        5.27292e-1,
        4.31710e-1,
        3.53455e-1,
        2.89384e-1,
        2.36928e-1,
        1.93980e-1,
        1.58817e-1,
        1.30029e-1,
        1.06458e-1,
        8.71608e-2,
        7.13612e-2,
        5.84256e-2,
        4.78349e-2,
        3.91639e-2,
        3.20647e-2,
        2.62523e-2,
        2.14936e-2,
        1.75975e-2,
        1.44076e-2,
        1.17959e-2,
        9.65769e-3,
    ],
    dtype=np.float64,
)
MAX_SW_GPOINTS = max(len(groups) for groups in SW_REDUCED_GROUPS)
MAX_LW_GPOINTS = max(len(groups) for groups in LW_REDUCED_GROUPS)

SW_READ_SPECS = (
    (16, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (3, 16)), ("sfluxrefo", "f", (16,)))),
    (17, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)), ("sfluxrefo", "f", (16, 5)))),
    (18, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (3, 16)), ("sfluxrefo", "f", (16, 9)))),
    (19, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (3, 16)), ("sfluxrefo", "f", (16, 9)))),
    (20, (("rayl", "f", ()), ("layreffr", "i", ()), ("absch4o", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)), ("sfluxrefo", "f", (16,)))),
    (21, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)), ("sfluxrefo", "f", (16, 9)))),
    (22, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (3, 16)), ("sfluxrefo", "f", (16, 9)))),
    (23, (("raylo", "f", (16,)), ("givfac", "f", ()), ("layreffr", "i", ()), ("kao", "f", (5, 13, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (3, 16)), ("sfluxrefo", "f", (16,)))),
    (24, (("raylao", "f", (16, 9)), ("raylbo", "f", (16,)), ("strrat", "f", ()), ("layreffr", "i", ()), ("abso3ao", "f", (16,)), ("abso3bo", "f", (16,)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (3, 16)), ("sfluxrefo", "f", (16, 9)))),
    (25, (("raylo", "f", (16,)), ("layreffr", "i", ()), ("abso3ao", "f", (16,)), ("abso3bo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("sfluxrefo", "f", (16,)))),
    (26, (("raylo", "f", (16,)), ("sfluxrefo", "f", (16,)))),
    (27, (("raylo", "f", (16,)), ("scalekur", "f", ()), ("layreffr", "i", ()), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("sfluxrefo", "f", (16,)))),
    (28, (("rayl", "f", ()), ("strrat", "f", ()), ("layreffr", "i", ()), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 5, 47, 16)), ("sfluxrefo", "f", (16, 5)))),
    (29, (("rayl", "f", ()), ("layreffr", "i", ()), ("absh2oo", "f", (16,)), ("absco2o", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)), ("sfluxrefo", "f", (16,)))),
)
LW_READ_SPECS = (
    (1, (("fracrefao", "f", (16,)), ("fracrefbo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("kao_mn2", "f", (19, 16)), ("kbo_mn2", "f", (19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (2, (("fracrefao", "f", (16,)), ("fracrefbo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (3, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16, 5)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 5, 47, 16)), ("kao_mn2o", "f", (9, 19, 16)), ("kbo_mn2o", "f", (5, 19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (4, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16, 5)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (5, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16, 5)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 5, 47, 16)), ("kao_mo3", "f", (9, 19, 16)), ("ccl4o", "f", (16,)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (6, (("fracrefao", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kao_mco2", "f", (19, 16)), ("cfc11adjo", "f", (16,)), ("cfc12o", "f", (16,)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (7, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16,)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("kao_mco2", "f", (9, 19, 16)), ("kbo_mco2", "f", (19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (8, (("fracrefao", "f", (16,)), ("fracrefbo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("kao_mco2", "f", (19, 16)), ("kbo_mco2", "f", (19, 16)), ("kao_mn2o", "f", (19, 16)), ("kbo_mn2o", "f", (19, 16)), ("kao_mo3", "f", (19, 16)), ("cfc12o", "f", (16,)), ("cfc22adjo", "f", (16,)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (9, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16,)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("kao_mn2o", "f", (9, 19, 16)), ("kbo_mn2o", "f", (19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (10, (("fracrefao", "f", (16,)), ("fracrefbo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (11, (("fracrefao", "f", (16,)), ("fracrefbo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("kao_mo2", "f", (19, 16)), ("kbo_mo2", "f", (19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (12, (("fracrefao", "f", (16, 9)), ("kao", "f", (9, 5, 13, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (13, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16,)), ("kao", "f", (9, 5, 13, 16)), ("kao_mco2", "f", (9, 19, 16)), ("kao_mco", "f", (9, 19, 16)), ("kbo_mo3", "f", (19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (14, (("fracrefao", "f", (16,)), ("fracrefbo", "f", (16,)), ("kao", "f", (5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (15, (("fracrefao", "f", (16, 9)), ("kao", "f", (9, 5, 13, 16)), ("kao_mn2", "f", (9, 19, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
    (16, (("fracrefao", "f", (16, 9)), ("fracrefbo", "f", (16,)), ("kao", "f", (9, 5, 13, 16)), ("kbo", "f", (5, 47, 16)), ("selfrefo", "f", (10, 16)), ("forrefo", "f", (4, 16)))),
)


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

    positive = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = np.sum(positive)
    if total <= 0.0:
        raise ValueError("cannot normalize all-zero spectral weights")
    return positive / total


class _RecordReader:
    """Typed reader for one big-endian WRF sequential-unformatted payload."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def read_real(self, shape: tuple[int, ...]) -> np.ndarray:
        """Reads one real(kind=rb) array using WRF DATA file byte order."""

        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = count * 4
        values = np.frombuffer(self.payload, dtype=">f4", count=count, offset=self.offset).astype(np.float64)
        self.offset += nbytes
        if shape:
            return values.reshape(shape, order="F")
        return values.reshape(())

    def read_int(self, shape: tuple[int, ...]) -> np.ndarray:
        """Reads one integer(kind=im) array using WRF DATA file byte order."""

        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        nbytes = count * 4
        values = np.frombuffer(self.payload, dtype=">i4", count=count, offset=self.offset).astype(np.int32)
        self.offset += nbytes
        if shape:
            return values.reshape(shape, order="F")
        return values.reshape(())

    def done(self) -> None:
        """Verifies that the record specification consumed the full payload."""

        if self.offset != len(self.payload):
            raise ValueError(f"record parse consumed {self.offset} bytes, expected {len(self.payload)}")


def _parse_record(payload: bytes, spec: tuple[tuple[str, str, tuple[int, ...]], ...]) -> dict[str, np.ndarray]:
    """Parses one WRF RRTMG DATA record from its source READ-list spec."""

    reader = _RecordReader(payload)
    parsed: dict[str, np.ndarray] = {}
    for name, kind, shape in spec:
        parsed[name] = reader.read_real(shape) if kind == "f" else reader.read_int(shape)
    reader.done()
    return parsed


def _reduce_gpoints(values: np.ndarray, groups: tuple[int, ...], *, weighted: bool) -> np.ndarray:
    """Applies WRF's reduced-g-point grouping to arrays with original g last."""

    reduced = []
    start = 0
    for group_size in groups:
        end = start + group_size
        segment = values[..., start:end]
        if weighted:
            weights = ORIGINAL_GPOINT_WEIGHTS[start:end]
            weights = weights / np.sum(weights)
            reduced.append(np.sum(segment * weights, axis=-1))
        else:
            reduced.append(np.sum(segment, axis=-1))
        start = end
    if start != 16:
        raise ValueError(f"g-point grouping consumed {start} original points, expected 16")
    return np.stack(reduced, axis=-1)


def _reduce_gpoints_first(values: np.ndarray, groups: tuple[int, ...], *, weighted: bool) -> np.ndarray:
    """Applies WRF's reduced-g-point grouping to arrays with original g first."""

    moved = np.moveaxis(values, 0, -1)
    reduced = _reduce_gpoints(moved, groups, weighted=weighted)
    return np.moveaxis(reduced, -1, 0)


def _pad_gpoints(values: np.ndarray, width: int) -> np.ndarray:
    """Pads a reduced-g-point array on the last axis with zeros."""

    pad = width - values.shape[-1]
    if pad < 0:
        raise ValueError(f"cannot pad width {values.shape[-1]} to {width}")
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, pad)])


def _pad_axis(values: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Pads one axis of an array with zeros up to `width`."""

    pad = width - values.shape[axis]
    if pad < 0:
        raise ValueError(f"cannot pad axis width {values.shape[axis]} to {width}")
    pads = [(0, 0)] * values.ndim
    pads[axis] = (0, pad)
    return np.pad(values, pads)


def _reduced_mask(groups_by_band: tuple[tuple[int, ...], ...], width: int) -> np.ndarray:
    """Builds a band-by-g-point validity mask for padded arrays."""

    mask = np.zeros((len(groups_by_band), width), dtype=np.float64)
    for band_index, groups in enumerate(groups_by_band):
        mask[band_index, : len(groups)] = 1.0
    return mask


def _reference_profile(kao: np.ndarray | None, kbo: np.ndarray | None, groups: tuple[int, ...]) -> np.ndarray:
    """Reduces WRF KAO/KBO data to reference-pressure profiles by g point."""

    ng = len(groups)
    profile = np.zeros((REFERENCE_PRESSURE_MB.size, ng), dtype=np.float64)
    if kao is not None:
        if kao.ndim == 4:
            lower = np.mean(kao[:, 2, :, :], axis=0)
        elif kao.ndim == 3:
            lower = kao[2, :, :]
        else:
            raise ValueError(f"unsupported KAO rank {kao.ndim}")
        profile[:13, :] = _reduce_gpoints(lower, groups, weighted=True)
    if kbo is not None:
        if kbo.ndim == 4:
            upper = np.mean(kbo[2, :, :, :], axis=0)
        elif kbo.ndim == 3:
            upper = kbo[2, :, :]
        else:
            raise ValueError(f"unsupported KBO rank {kbo.ndim}")
        profile[12:, :] = _reduce_gpoints(upper, groups, weighted=True)
    else:
        profile[13:, :] = profile[12:13, :]
    return profile


def _parse_source_array(source: Path, name: str, band: int) -> np.ndarray:
    """Extracts a static WRF source array assignment such as extliq1(:, 16)."""

    text = source.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(rf"{name}\(:,\s*{band}\)\s*=\s*\(/\s*&(?P<body>.*?)\s*/\)", re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing source array {name}(:, {band}) in {source}")
    body = re.sub(r"!.*", "", match.group("body"))
    body = body.replace("_rb", "").replace("D", "E").replace("d", "e")
    numbers = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", body)
    return np.asarray([float(number) for number in numbers], dtype=np.float64)


def _interp_table(values: np.ndarray, grid: np.ndarray, radius: float) -> float:
    """Interpolates WRF cloud-optical tables with endpoint guarding."""

    return float(np.interp(radius, grid, values))


def _parse_source_block_numbers(body: str) -> np.ndarray:
    """Parses a Fortran DATA/assignment body into float64 values."""

    body = re.sub(r"!.*", "", body)
    body = body.replace("_rb", "").replace("D", "E").replace("d", "e")
    numbers = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", body)
    return np.asarray([float(number) for number in numbers], dtype=np.float64)


def _parse_named_vector_assignment(source: Path, name: str) -> np.ndarray:
    """Parses a whole-array Fortran vector assignment such as `tref(:) = (/ ... /)`."""

    text = source.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(rf"{name}\(:\)\s*=\s*\(/\s*&(?P<body>.*?)\s*/\)", re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing source vector {name}(:) in {source}")
    return _parse_source_block_numbers(match.group("body"))


def _lw_planck_tables() -> tuple[np.ndarray, np.ndarray]:
    """Extracts WRF LW integrated Planck tables from source assignments."""

    text = LW_SOURCE.read_text(encoding="utf-8", errors="replace")
    totplnk = np.zeros((181, 16), dtype=np.float64)
    pattern = re.compile(r"totplnk\((\d+):(\d+),\s*(\d+)\)\s*=\s*\(/\s*&(?P<body>.*?)\s*/\)", re.DOTALL)
    for match in pattern.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2))
        band = int(match.group(3))
        values = _parse_source_block_numbers(match.group("body"))
        if values.size != end - start + 1:
            raise ValueError(f"bad totplnk({start}:{end},{band}) size {values.size}")
        totplnk[start - 1 : end, band - 1] = values
    if np.count_nonzero(totplnk) == 0:
        raise ValueError("failed to extract LW totplnk table")

    totplk16 = np.zeros(181, dtype=np.float64)
    pattern16 = re.compile(r"totplk16\((\d+):(\d+)\)\s*=\s*\(/\s*&(?P<body>.*?)\s*/\)", re.DOTALL)
    for match in pattern16.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2))
        values = _parse_source_block_numbers(match.group("body"))
        if values.size != end - start + 1:
            raise ValueError(f"bad totplk16({start}:{end}) size {values.size}")
        totplk16[start - 1 : end] = values
    if np.count_nonzero(totplk16) == 0:
        raise ValueError("failed to extract LW totplk16 table")
    return totplnk, totplk16


def _sw_cloud_coefficients() -> tuple[np.ndarray, ...]:
    """Builds SW cloud optical coefficients from WRF source tables."""

    liquid_grid = np.arange(2.5, 60.5, dtype=np.float64)
    ice_grid = 2.0 + 3.0 * np.arange(1, 47, dtype=np.float64)
    liquid_ext = []
    liquid_ssa = []
    liquid_asy = []
    ice_ext = []
    ice_ssa = []
    ice_asy = []
    ice_forw = []
    snow_ext = []
    snow_ssa = []
    snow_asy = []
    snow_forw = []
    for band in range(16, 30):
        ext_liq = _interp_table(_parse_source_array(SW_SOURCE, "extliq1", band), liquid_grid, 10.0)
        ssa_liq = _interp_table(_parse_source_array(SW_SOURCE, "ssaliq1", band), liquid_grid, 10.0)
        asy_liq = _interp_table(_parse_source_array(SW_SOURCE, "asyliq1", band), liquid_grid, 10.0)
        ext_ice = _interp_table(_parse_source_array(SW_SOURCE, "extice3", band), ice_grid, 30.0)
        ssa_ice = _interp_table(_parse_source_array(SW_SOURCE, "ssaice3", band), ice_grid, 30.0)
        asy_ice = _interp_table(_parse_source_array(SW_SOURCE, "asyice3", band), ice_grid, 30.0)
        fdl_ice = _interp_table(_parse_source_array(SW_SOURCE, "fdlice3", band), ice_grid, 30.0)
        ext_snow = _interp_table(_parse_source_array(SW_SOURCE, "extice3", band), ice_grid, 75.0)
        ssa_snow = _interp_table(_parse_source_array(SW_SOURCE, "ssaice3", band), ice_grid, 75.0)
        asy_snow = _interp_table(_parse_source_array(SW_SOURCE, "asyice3", band), ice_grid, 75.0)
        fdl_snow = _interp_table(_parse_source_array(SW_SOURCE, "fdlice3", band), ice_grid, 75.0)
        liquid_ext.append(ext_liq)
        ice_ext.append(ext_ice)
        liquid_ssa.append(ssa_liq)
        ice_ssa.append(ssa_ice)
        liquid_asy.append(asy_liq)
        ice_asy.append(asy_ice)
        ice_forw.append(min(asy_ice, fdl_ice + 0.5 / ssa_ice))
        snow_ext.append(ext_snow)
        snow_ssa.append(ssa_snow)
        snow_asy.append(asy_snow)
        snow_forw.append(min(asy_snow, fdl_snow + 0.5 / ssa_snow))
    return (
        np.asarray(liquid_ext, dtype=np.float64),
        np.asarray(ice_ext, dtype=np.float64),
        np.asarray(liquid_ssa, dtype=np.float64),
        np.asarray(ice_ssa, dtype=np.float64),
        np.asarray(liquid_asy, dtype=np.float64),
        np.asarray(ice_asy, dtype=np.float64),
        np.asarray(ice_forw, dtype=np.float64),
        np.asarray(snow_ext, dtype=np.float64),
        np.asarray(snow_ssa, dtype=np.float64),
        np.asarray(snow_asy, dtype=np.float64),
        np.asarray(snow_forw, dtype=np.float64),
    )


def _lw_cloud_absorption() -> np.ndarray:
    """Builds LW cloud absorption coefficients from WRF source tables."""

    liquid_grid = np.arange(1.5, 59.5, dtype=np.float64)
    ice_grid = 2.0 + 3.0 * np.arange(1, 47, dtype=np.float64)
    values = []
    for band in range(1, 17):
        liq = _interp_table(_parse_source_array(LW_SOURCE, "absliq1", band), liquid_grid, 10.0)
        ice = _interp_table(_parse_source_array(LW_SOURCE, "absice3", band), ice_grid, 30.0)
        values.append(0.5 * (liq + ice))
    return np.asarray(values, dtype=np.float64)


def _effective_sw_coefficients(records: list[bytes]) -> dict[str, np.ndarray]:
    """Builds SW coefficients using WRF's source-cited reduced g-point formula."""

    gas_profiles = []
    source_weights = []
    rayleigh = []
    for record, (_band, spec), groups in zip(records, SW_READ_SPECS, SW_REDUCED_GROUPS, strict=True):
        parsed = _parse_record(record, spec)
        gas_profiles.append(_pad_gpoints(_reference_profile(parsed.get("kao"), parsed.get("kbo"), groups), MAX_SW_GPOINTS))
        source = parsed["sfluxrefo"]
        if source.ndim > 1:
            source = np.mean(source, axis=tuple(range(1, source.ndim)))
        source_weights.append(_pad_gpoints(_reduce_gpoints(source, groups, weighted=False)[None, :], MAX_SW_GPOINTS)[0])
        if "raylo" in parsed:
            rayleigh.append(_pad_gpoints(_reduce_gpoints(parsed["raylo"], groups, weighted=True)[None, :], MAX_SW_GPOINTS)[0])
        elif "raylao" in parsed:
            ray = np.mean(parsed["raylao"], axis=1)
            rayleigh.append(_pad_gpoints(_reduce_gpoints(ray, groups, weighted=True)[None, :], MAX_SW_GPOINTS)[0])
        else:
            rayleigh.append(np.full(MAX_SW_GPOINTS, float(parsed.get("rayl", np.asarray(0.0))), dtype=np.float64) * _reduced_mask((groups,), MAX_SW_GPOINTS)[0])
    source_weights_array = np.asarray(source_weights, dtype=np.float64)
    sw_gpoint_weights = _normalized(source_weights_array)
    sw_gpoint_mask = _reduced_mask(SW_REDUCED_GROUPS, MAX_SW_GPOINTS)
    (
        liquid_ext,
        ice_ext,
        liquid_ssa,
        ice_ssa,
        liquid_asy,
        ice_asy,
        ice_forw,
        snow_ext,
        snow_ssa,
        snow_asy,
        snow_forw,
    ) = _sw_cloud_coefficients()
    band_weights = np.sum(sw_gpoint_weights, axis=1)
    return {
        "sw_reference_pressure_pa": REFERENCE_PRESSURE_MB * 100.0,
        "sw_preflog": np.log(REFERENCE_PRESSURE_MB),
        "sw_tref": _parse_named_vector_assignment(SW_SOURCE, "tref"),
        "sw_gpoint_mask": sw_gpoint_mask,
        "sw_gpoint_weights": sw_gpoint_weights,
        "sw_absorption_coefficients": np.asarray(gas_profiles, dtype=np.float64),
        "sw_rayleigh_coefficients": np.asarray(rayleigh, dtype=np.float64),
        "sw_cloud_liquid_extinction": liquid_ext,
        "sw_cloud_ice_extinction": ice_ext,
        "sw_cloud_liquid_ssa": liquid_ssa,
        "sw_cloud_ice_ssa": ice_ssa,
        "sw_cloud_liquid_asymmetry": liquid_asy,
        "sw_cloud_ice_asymmetry": ice_asy,
        "sw_cloud_ice_forward_fraction": ice_forw,
        "sw_cloud_snow_extinction": snow_ext,
        "sw_cloud_snow_ssa": snow_ssa,
        "sw_cloud_snow_asymmetry": snow_asy,
        "sw_cloud_snow_forward_fraction": snow_forw,
        "sw_band_weights": band_weights,
    }


def _full_sw_reduced_tables(records: list[bytes]) -> dict[str, np.ndarray]:
    """Builds native SW reduced-g tables used by WRF `taumol_sw`."""

    max_g = MAX_SW_GPOINTS
    max_absa = 9 * 5 * 13
    max_absb = 5 * 5 * 47
    max_source_param = 9
    nspa = np.asarray([9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1], dtype=np.int32)
    nspb = np.asarray([1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1], dtype=np.int32)

    absa = np.zeros((14, max_absa, max_g), dtype=np.float64)
    absb = np.zeros((14, max_absb, max_g), dtype=np.float64)
    selfref = np.zeros((14, 10, max_g), dtype=np.float64)
    forref = np.zeros((14, 4, max_g), dtype=np.float64)
    sfluxref = np.zeros((14, max_g, max_source_param), dtype=np.float64)
    rayl = np.zeros((14, max_g), dtype=np.float64)
    rayl_scalar = np.zeros(14, dtype=np.float64)
    rayla = np.zeros((14, max_g, max_source_param), dtype=np.float64)
    raylb = np.zeros((14, max_g), dtype=np.float64)
    abs_ch4 = np.zeros((14, max_g), dtype=np.float64)
    abs_o3a = np.zeros((14, max_g), dtype=np.float64)
    abs_o3b = np.zeros((14, max_g), dtype=np.float64)
    abs_h2o = np.zeros((14, max_g), dtype=np.float64)
    abs_co2 = np.zeros((14, max_g), dtype=np.float64)
    strrat = np.zeros(14, dtype=np.float64)
    layreffr = np.ones(14, dtype=np.int32)
    givfac = np.ones(14, dtype=np.float64)
    scalekur = np.ones(14, dtype=np.float64)

    for band_index, (record, (_band, spec), groups) in enumerate(zip(records, SW_READ_SPECS, SW_REDUCED_GROUPS, strict=True)):
        parsed = _parse_record(record, spec)
        ng = len(groups)
        if "kao" in parsed:
            kao = parsed["kao"]
            if kao.ndim == 3:
                kao = kao[None, :, :, :]
            kao_red = _reduce_gpoints(kao, groups, weighted=True)
            flat = kao_red.reshape((kao_red.shape[0] * kao_red.shape[1] * kao_red.shape[2], ng), order="F")
            absa[band_index, : flat.shape[0], :ng] = flat
        if "kbo" in parsed:
            kbo = parsed["kbo"]
            if kbo.ndim == 3:
                kbo = kbo[None, :, :, :]
            kbo_red = _reduce_gpoints(kbo, groups, weighted=True)
            flat = kbo_red.reshape((kbo_red.shape[0] * kbo_red.shape[1] * kbo_red.shape[2], ng), order="F")
            absb[band_index, : flat.shape[0], :ng] = flat
        if "selfrefo" in parsed:
            reduced = _reduce_gpoints(parsed["selfrefo"], groups, weighted=True)
            selfref[band_index, : reduced.shape[0], :ng] = reduced
        if "forrefo" in parsed:
            reduced = _reduce_gpoints(parsed["forrefo"], groups, weighted=True)
            forref[band_index, : reduced.shape[0], :ng] = reduced
        source = parsed["sfluxrefo"]
        if source.ndim == 1:
            reduced = _reduce_gpoints(source, groups, weighted=False)
            sfluxref[band_index, :ng, 0] = reduced
        else:
            reduced = _reduce_gpoints_first(source, groups, weighted=False)
            sfluxref[band_index, :ng, : reduced.shape[1]] = reduced
        if "rayl" in parsed:
            rayl_scalar[band_index] = float(parsed["rayl"])
            rayl[band_index, :ng] = float(parsed["rayl"])
        if "raylo" in parsed:
            rayl[band_index, :ng] = _reduce_gpoints(parsed["raylo"], groups, weighted=True)
        if "raylbo" in parsed:
            raylb[band_index, :ng] = _reduce_gpoints(parsed["raylbo"], groups, weighted=True)
        if "raylao" in parsed:
            reduced = _reduce_gpoints_first(parsed["raylao"], groups, weighted=True)
            rayla[band_index, :ng, : reduced.shape[1]] = reduced
        if "absch4o" in parsed:
            abs_ch4[band_index, :ng] = _reduce_gpoints(parsed["absch4o"], groups, weighted=True)
        if "abso3ao" in parsed:
            abs_o3a[band_index, :ng] = _reduce_gpoints(parsed["abso3ao"], groups, weighted=True)
        if "abso3bo" in parsed:
            abs_o3b[band_index, :ng] = _reduce_gpoints(parsed["abso3bo"], groups, weighted=True)
        if "absh2oo" in parsed:
            abs_h2o[band_index, :ng] = _reduce_gpoints(parsed["absh2oo"], groups, weighted=True)
        if "absco2o" in parsed:
            abs_co2[band_index, :ng] = _reduce_gpoints(parsed["absco2o"], groups, weighted=True)
        if "strrat" in parsed:
            strrat[band_index] = float(parsed["strrat"])
        layreffr[band_index] = int(parsed.get("layreffr", np.asarray(1)))
        if "givfac" in parsed:
            givfac[band_index] = float(parsed["givfac"])
        if "scalekur" in parsed:
            scalekur[band_index] = float(parsed["scalekur"])

    return {
        "sw_nspa": nspa,
        "sw_nspb": nspb,
        "sw_absa": absa,
        "sw_absb": absb,
        "sw_selfref": selfref,
        "sw_forref": forref,
        "sw_sfluxref": sfluxref,
        "sw_rayl": rayl,
        "sw_rayl_scalar": rayl_scalar,
        "sw_rayla": rayla,
        "sw_raylb": raylb,
        "sw_abs_ch4": abs_ch4,
        "sw_abs_o3a": abs_o3a,
        "sw_abs_o3b": abs_o3b,
        "sw_abs_h2o": abs_h2o,
        "sw_abs_co2": abs_co2,
        "sw_strrat": strrat,
        "sw_layreffr": layreffr,
        "sw_givfac": givfac,
        "sw_scalekur": scalekur,
    }


def _effective_lw_coefficients(records: list[bytes]) -> dict[str, np.ndarray]:
    """Builds LW coefficients using WRF's source-cited reduced g-point formula."""

    gas = []
    for record, (_band, spec), groups in zip(records, LW_READ_SPECS, LW_REDUCED_GROUPS, strict=True):
        parsed = _parse_record(record, spec)
        gas.append(_pad_gpoints(_reference_profile(parsed.get("kao"), parsed.get("kbo"), groups), MAX_LW_GPOINTS))
    weights = np.zeros((len(LW_REDUCED_GROUPS), MAX_LW_GPOINTS), dtype=np.float64)
    for band_index, groups in enumerate(LW_REDUCED_GROUPS):
        weights[band_index, : len(groups)] = LW_DELWAVE[band_index] / len(groups)
    lw_gpoint_weights = _normalized(weights)
    lw_gpoint_mask = _reduced_mask(LW_REDUCED_GROUPS, MAX_LW_GPOINTS)
    cloud = _lw_cloud_absorption()
    return {
        "lw_reference_pressure_pa": REFERENCE_PRESSURE_MB * 100.0,
        "lw_preflog": np.log(REFERENCE_PRESSURE_MB),
        "lw_tref": _parse_named_vector_assignment(LW_SOURCE, "tref"),
        "lw_gpoint_mask": lw_gpoint_mask,
        "lw_gpoint_weights": lw_gpoint_weights,
        "lw_absorption_coefficients": np.asarray(gas, dtype=np.float64),
        "lw_cloud_absorption": cloud,
        "lw_band_weights": np.sum(lw_gpoint_weights, axis=1),
    }


def _full_lw_planck_tables() -> dict[str, np.ndarray]:
    """Builds LW Planck-source tables from WRF source assignments."""

    totplnk, totplk16 = _lw_planck_tables()
    return {
        "lw_totplnk": totplnk,
        "lw_totplk16": totplk16,
        "lw_delwave": LW_DELWAVE.copy(),
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
    tables.update(_full_sw_reduced_tables(sw_records))
    tables.update(_effective_lw_coefficients(lw_records))
    tables.update(_full_lw_planck_tables())
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
        "source": "real WRF RRTMG_SW_DATA/RRTMG_LW_DATA records reduced with WRF g-point weights and source cloud optics",
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
