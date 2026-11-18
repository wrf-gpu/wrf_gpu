"""Grid lock for static per-grid surface-control files (z0bare Nightly path; host-only, no traced change).

A control file built for one grid carries ``provenance["grid_lock"]``: a per-domain fingerprint of the
wrfinput geometry and land classification it was derived from. Every consumer re-derives the fingerprint
from the run's own wrfinput files and refuses on any difference (grid change => refusal, never a silent
OFF or a control applied to the wrong cells). Domains named by the lock must exist in the run and in the
file; a locked file therefore cannot degrade to the default path on a missing nest.

Files without ``grid_lock`` keep the existing behaviour (shape/dtype/bounds checks in controls.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SCHEMA = "gpuwrf.surface_controls.grid_lock.v1"
KEY = "grid_lock"
# Geometry + land classification: everything a per-cell LU rule depends on.
ARRAYS = ("XLAT", "XLONG", "LU_INDEX", "IVGTYP", "LANDMASK", "HGT")
ATTRS = ("WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION", "DX", "DY", "MAP_PROJ", "TRUELAT1",
         "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON", "MOAD_CEN_LAT", "POLE_LAT", "POLE_LON",
         "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO",
         "MMINLU", "NUM_LAND_CAT", "ISWATER", "ISLAKE", "ISICE", "ISURBAN", "ISOILWATER")


class GridLockError(ValueError):
    pass


def _attr(value):
    if isinstance(value, (bytes, str)):
        return value.decode() if isinstance(value, bytes) else value
    arr = np.asarray(value)
    if arr.dtype.kind in "iu":
        return int(arr.reshape(-1)[0])
    # Exact bit pattern of the stored float32 (no decimal rounding question).
    return {"f32": arr.astype(np.float32).reshape(-1)[0].tobytes().hex()}


def domain_fingerprint(wrfinput: Path) -> dict:
    """Fingerprint of one wrfinput file (first time level; arrays hashed as stored dtype bytes)."""
    from gpuwrf.io.netcdf_lock import Dataset  # noqa: PLC0415 (process-wide netCDF lock)

    wrfinput = Path(wrfinput)
    with Dataset(wrfinput) as ds:
        missing = [k for k in ATTRS if k not in ds.ncattrs()] + [k for k in ARRAYS if k not in ds.variables]
        if missing:
            raise GridLockError(f"{wrfinput.name}: grid-lock fields absent: {missing}")
        attrs = {k: _attr(ds.getncattr(k)) for k in ATTRS}
        arrays = {}
        for k in ARRAYS:
            v = ds.variables[k]
            v.set_auto_maskandscale(False)
            a = np.ascontiguousarray(np.asarray(v[0]))
            arrays[k] = {"dtype": str(a.dtype), "shape": list(a.shape),
                         "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    return {"attrs": attrs, "arrays": arrays}


def fingerprint(input_dir: Path, domains: Iterable[str]) -> dict:
    input_dir = Path(input_dir)
    return {d: domain_fingerprint(input_dir / f"wrfinput_{d}") for d in sorted(domains)}


def make_lock(input_dir: Path, domains: Iterable[str], rule: Mapping) -> dict:
    fp = fingerprint(input_dir, domains)
    body = {"schema": SCHEMA, "domains": fp, "rule": dict(rule)}
    body["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def lock_of(manifest: Mapping) -> dict | None:
    lock = (manifest.get("provenance") or {}).get(KEY)
    if lock is None:
        return None
    if not isinstance(lock, Mapping) or lock.get("schema") != SCHEMA:
        raise GridLockError("surface controls: unknown grid_lock schema")
    body = {k: v for k, v in lock.items() if k != "digest"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != lock.get("digest"):
        raise GridLockError("surface controls: grid_lock digest mismatch (provenance altered)")
    return dict(lock)


def _structural_admission(manifest: Mapping, input_dir: Path | None, run_domains: Iterable[str]) -> dict | None:
    """Cheap per-manifest checks, run on EVERY admission: lock digest (lock_of), case dir given, every locked
    domain in this run and controlled by THIS file. Returns the verified lock (None for unlocked files)."""
    lock = lock_of(manifest)
    if lock is None:
        return None
    if input_dir is None:
        raise GridLockError("surface controls: grid-locked file needs the case input dir (--input-dir) to verify the grid")
    locked = sorted(lock["domains"])
    run_domains = set(run_domains)
    absent_run = [d for d in locked if d not in run_domains]
    if absent_run:
        raise GridLockError(f"surface controls: locked domains not in this run: {absent_run} (grid changed?)")
    absent_file = [d for d in locked if d not in manifest.get("domains", {})]
    if absent_file:
        raise GridLockError(f"surface controls: locked domains without controls in the file: {absent_file}")
    return lock


def _grid_comparison(lock: Mapping, input_dir: Path) -> None:
    """Expensive part: re-derive the wrfinput fingerprint of every locked domain and compare."""
    locked = sorted(lock["domains"])
    current = fingerprint(input_dir, locked)
    for d in locked:
        if current[d] != lock["domains"][d]:
            diff = sorted(k for part in ("attrs", "arrays") for k in lock["domains"][d][part]
                          if current[d][part].get(k) != lock["domains"][d][part][k])
            raise GridLockError(f"surface controls: {d} grid differs from the locked grid ({diff}); "
                                "rebuild the control file for this grid (no silent OFF)")


def require_grid_lock(manifest: Mapping, input_dir: Path | None, run_domains: Iterable[str]) -> dict | None:
    """Refuse unless the run's wrfinput grid equals the locked grid (no-op for unlocked files). Uncached."""
    lock = _structural_admission(manifest, input_dir, run_domains)
    if lock is None:
        return None
    _grid_comparison(lock, input_dir)
    return {"status": "GRID_LOCK_PASS", "digest": lock["digest"], "domains": sorted(lock["domains"])}


# Memo of the EXPENSIVE grid comparison only: keyed by (lock digest, resolved case dir, wrfinput stat identity
# of the locked domains). The per-manifest structural checks are never memoized (critic DELTA 1).
_GRID_PASSED: set = set()


def _wrfinput_identity(input_dir: Path, domains: Iterable[str]) -> tuple:
    out = []
    for d in sorted(domains):
        s = (Path(input_dir) / f"wrfinput_{d}").stat()  # missing file -> OSError (refusal)
        out.append((d, s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return tuple(out)


def admit_control_source(source, input_dir: Path | None, run_domains: Iterable[str]) -> dict | None:
    """Native admission of one control source (file path or in-memory mapping) for one actual case dir.

    Every model consumer of control files calls this with the case directory it really loads (B=1 dir,
    or each resolved B>1 lane dir) before any binding/initialization. Unlocked files and in-memory
    mappings keep the existing behaviour (None). A locked file is refused on any grid difference."""
    if source is None or isinstance(source, Mapping):
        return None
    from gpuwrf.surface_params.controls import load_controls  # noqa: PLC0415

    _, manifest = load_controls(source)
    return admit_manifest(manifest, input_dir, run_domains)


def admit_manifest(manifest: Mapping, input_dir: Path | None, run_domains: Iterable[str]) -> dict | None:
    """admit_control_source on an already loaded manifest (no second file read).

    Structural checks of THIS manifest run on every call; only the grid fingerprint comparison is reused."""
    lock = _structural_admission(manifest, input_dir, run_domains)
    if lock is None:
        return None
    key = (lock["digest"], str(Path(input_dir).resolve()), _wrfinput_identity(input_dir, lock["domains"]))
    if key not in _GRID_PASSED:
        _grid_comparison(lock, input_dir)  # raises on any difference; only a PASS is remembered
        _GRID_PASSED.add(key)
    return {"status": "GRID_LOCK_PASS", "digest": lock["digest"], "domains": sorted(lock["domains"])}
