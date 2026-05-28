#!/usr/bin/env python
"""Emit and compare M6B6 WRF-shaped coupled-step savepoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.acoustic_loop import AcousticLoopState
from gpuwrf.dynamics.coupled_step import (
    COUPLED_STATE_FIELDS,
    NAMELIST_PHYSICS_BOUNDARY_ON,
    PHYSICS_TENDENCY_FIELDS,
    CoupledStepConfig,
    coupled_timesteps_wrf,
)
from gpuwrf.io.boundary_replay import SIDES, decode_wrfbdy
from gpuwrf.validation.comparator_common import DEFAULT_GEN2_WRFOUT, field_compare
from gpuwrf.validation.savepoint_io import write_savepoint
from gpuwrf.validation.savepoint_schema import Savepoint, SavepointMetadata, VariableMetadata, load_tolerance_ladder


SPRINT = ROOT / ".agent/sprints/2026-05-25-m6b6-coupled-step-parity"
WRF_REFERENCE_FIXTURE_ROOT = ROOT / "tests/savepoint/fixtures/wrf_b6_100step"
WRF_COMMIT = "115e5756f98ee2370d62b6709baac6417d8f7338"
SOURCE_WRFOUT = DEFAULT_GEN2_WRFOUT
SOURCE_WRFBDY = DEFAULT_GEN2_WRFOUT.parent / "wrfbdy_d01"
COMPARE_FIELDS = COUPLED_STATE_FIELDS
ACOUSTIC_SUBSTEPS_PER_RK = 10
RK_ORDER = 3
WRF_FALLBACK_STAGE = "history_interp_full_timestep"
WRF_HISTORY_INTERVAL_SECONDS = 3600.0
WRF_RAW_FIELDS = ("U", "V", "W", "T", "MU", "MUB", "P", "PB", "PH", "PHB", "QVAPOR")
HYDRO_FIELDS = ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QKE")


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found in scripts/")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_m6b4 = _import_script("m6b4_acoustic_recurrence_compare")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_wrfout_stamp(path: Path) -> tuple[str, datetime]:
    match = re.match(r"wrfout_d(?P<domain>\d{2})_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$", path.name)
    if match is None:
        raise ValueError(f"cannot parse WRF history timestamp from {path}")
    stamp = datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S")
    return match.group("domain"), stamp.replace(tzinfo=timezone.utc)


def _resolve_next_wrfout(path: Path) -> Path:
    domain, stamp = _parse_wrfout_stamp(path)
    next_stamp = stamp + timedelta(seconds=WRF_HISTORY_INTERVAL_SECONDS)
    direct = path.parent / f"wrfout_d{domain}_{next_stamp:%Y-%m-%d_%H:%M:%S}"
    if direct.exists():
        return direct
    candidates: list[tuple[datetime, Path]] = []
    for candidate in sorted(path.parent.glob(f"wrfout_d{domain}_*")):
        try:
            _, candidate_stamp = _parse_wrfout_stamp(candidate)
        except ValueError:
            continue
        if candidate_stamp > stamp:
            candidates.append((candidate_stamp, candidate))
    if not candidates:
        raise FileNotFoundError(f"no later WRF history file found next to {path}")
    return candidates[0][1]


def _fixture_file(tier: str, step: int, stage: str = WRF_FALLBACK_STAGE, fixture_root: Path = WRF_REFERENCE_FIXTURE_ROOT) -> Path:
    return fixture_root / tier / f"wrf_step{int(step):03d}_{stage}.nc"


def _stagger_for_field(name: str) -> str:
    if name in {"u", "U", "u_phys_tend"}:
        return "u"
    if name in {"v", "V", "v_phys_tend"}:
        return "v"
    if name in {"w", "W", "ww", "ph", "PH", "phb", "PHB", "ph_tend", "w_phys_tend"}:
        return "w"
    if name in {"mu", "MU", "mut", "MUB", "mudf", "muts", "muave", "mu_bdy_tend"}:
        return "mass"
    if name in {"theta", "T", "t_2ave", "p", "P", "pb", "PB", "qv", "QVAPOR", "qc", "qr", "qi", "qs", "qg", "qke"}:
        return "mass"
    if name.endswith("_phys_tend"):
        return "mass"
    return "scalar"


def _units_for_field(name: str) -> str:
    ladder = _coupled_ladder()
    if name in ladder["fields"]:
        return str(ladder["fields"][name]["units"])  # type: ignore[index]
    raw_units = {
        "U": "m s-1",
        "V": "m s-1",
        "W": "m s-1",
        "T": "K",
        "MU": "Pa",
        "MUB": "Pa",
        "P": "Pa",
        "PB": "Pa",
        "PH": "m2 s-2",
        "PHB": "m2 s-2",
        "QVAPOR": "kg kg-1",
        "pb": "Pa",
        "phb": "m2 s-2",
        "qv": "kg kg-1",
        "qc": "kg kg-1",
        "qr": "kg kg-1",
        "qi": "kg kg-1",
        "qs": "kg kg-1",
        "qg": "kg kg-1",
        "qke": "m2 s-2",
    }
    return raw_units.get(name, "operator-native")


def _dims_for_array(name: str, array: np.ndarray) -> tuple[str, ...]:
    stagger = _stagger_for_field(name)
    if array.ndim == 2:
        return ("south_north", "west_east")
    if array.ndim != 3:
        return tuple(f"{name}_dim_{idx}" for idx in range(array.ndim))
    if stagger == "u":
        return ("bottom_top", "south_north", "west_east_stag")
    if stagger == "v":
        return ("bottom_top", "south_north_stag", "west_east")
    if stagger == "w":
        return ("bottom_top_stag", "south_north", "west_east")
    return ("bottom_top", "south_north", "west_east")


def _optional_history_var(
    ds: Dataset,
    name: str,
    shape: tuple[int, ...],
    ys: slice,
    xs: slice,
    default: float = 0.0,
) -> np.ndarray:
    if name not in ds.variables:
        return np.ones(shape, dtype=np.float64) * default
    data = ds.variables[name]
    if len(data.shape) == 4:
        return np.asarray(data[0, :, ys, xs], dtype=np.float64)
    if len(data.shape) == 3:
        return np.asarray(data[0, ys, xs], dtype=np.float64)
    return np.asarray(data[:], dtype=np.float64)


def _load_wrf_history_arrays(path: Path, attrs: dict[str, object]) -> dict[str, np.ndarray]:
    ys, xs = _slice_attrs(attrs)
    u_x = slice(xs.start, xs.stop + 1)
    v_y = slice(ys.start, ys.stop + 1)
    with Dataset(path) as ds:
        theta = np.asarray(ds.variables["T"][0, :, ys, xs], dtype=np.float64)
        w = np.asarray(ds.variables["W"][0, :, ys, xs], dtype=np.float64)
        mu = np.asarray(ds.variables["MU"][0, ys, xs], dtype=np.float64)
        mut = np.asarray(ds.variables["MUB"][0, ys, xs], dtype=np.float64)
        arrays = {
            "U": np.asarray(ds.variables["U"][0, :, ys, u_x], dtype=np.float64),
            "V": np.asarray(ds.variables["V"][0, :, v_y, xs], dtype=np.float64),
            "W": w,
            "T": theta,
            "MU": mu,
            "MUB": mut,
            "P": np.asarray(ds.variables["P"][0, :, ys, xs], dtype=np.float64),
            "PB": np.asarray(ds.variables["PB"][0, :, ys, xs], dtype=np.float64),
            "PH": np.asarray(ds.variables["PH"][0, :, ys, xs], dtype=np.float64),
            "PHB": np.asarray(ds.variables["PHB"][0, :, ys, xs], dtype=np.float64),
            "QVAPOR": np.asarray(ds.variables["QVAPOR"][0, :, ys, xs], dtype=np.float64),
        }
        mass_shape = theta.shape
        for raw_name in HYDRO_FIELDS:
            arrays[raw_name] = _optional_history_var(ds, raw_name, mass_shape, ys, xs, 0.0)
    return arrays


def _interpolate_history(
    start: dict[str, np.ndarray],
    end: dict[str, np.ndarray],
    fraction: float,
    interval_seconds: float,
) -> dict[str, np.ndarray]:
    fraction = float(max(0.0, min(1.0, fraction)))
    interp = {name: np.asarray(start[name] + fraction * (end[name] - start[name]), dtype=np.float64) for name in start}
    zeros_w = np.zeros_like(interp["W"], dtype=np.float64)
    arrays = {
        "mu": interp["MU"],
        "mut": interp["MUB"],
        "mudf": (end["MU"] - start["MU"]) / float(interval_seconds),
        "muts": interp["MU"] + interp["MUB"],
        "muave": interp["MU"],
        "ww": zeros_w,
        "theta": interp["T"],
        "ph_tend": (end["PH"] - start["PH"]) / float(interval_seconds),
        "u": interp["U"],
        "v": interp["V"],
        "w": interp["W"],
        "ph": interp["PH"],
        "p": interp["P"],
        "t_2ave": interp["T"],
        "theta_phys_tend": (end["T"] - start["T"]) / float(interval_seconds),
        "qv_phys_tend": (end["QVAPOR"] - start["QVAPOR"]) / float(interval_seconds),
        "qc_phys_tend": (end["QCLOUD"] - start["QCLOUD"]) / float(interval_seconds),
        "qr_phys_tend": (end["QRAIN"] - start["QRAIN"]) / float(interval_seconds),
        "qi_phys_tend": (end["QICE"] - start["QICE"]) / float(interval_seconds),
        "qs_phys_tend": (end["QSNOW"] - start["QSNOW"]) / float(interval_seconds),
        "qg_phys_tend": (end["QGRAUP"] - start["QGRAUP"]) / float(interval_seconds),
        "qke_phys_tend": (end["QKE"] - start["QKE"]) / float(interval_seconds),
        "u_phys_tend": (end["U"] - start["U"]) / float(interval_seconds),
        "v_phys_tend": (end["V"] - start["V"]) / float(interval_seconds),
        "w_phys_tend": (end["W"] - start["W"]) / float(interval_seconds),
        "mu_bdy_tend": (end["MU"] - start["MU"]) / float(interval_seconds),
        "pb": interp["PB"],
        "phb": interp["PHB"],
        "qv": interp["QVAPOR"],
        "qc": interp["QCLOUD"],
        "qr": interp["QRAIN"],
        "qi": interp["QICE"],
        "qs": interp["QSNOW"],
        "qg": interp["QGRAUP"],
        "qke": interp["QKE"],
    }
    arrays.update({name: interp[name] for name in WRF_RAW_FIELDS})
    return arrays


def _write_wrf_fixture_nc(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    tier: str,
    step: int,
    attrs: dict[str, object],
    source_start: Path,
    source_end: Path,
    fraction: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mass = np.asarray(arrays["theta"])
    w = np.asarray(arrays["w"])
    with Dataset(path, "w") as ds:
        ds.createDimension("bottom_top", int(mass.shape[0]))
        ds.createDimension("bottom_top_stag", int(w.shape[0]))
        ds.createDimension("south_north", int(mass.shape[1]))
        ds.createDimension("west_east", int(mass.shape[2]))
        ds.createDimension("south_north_stag", int(mass.shape[1] + 1))
        ds.createDimension("west_east_stag", int(mass.shape[2] + 1))
        ds.setncattr("schema", "f1-real-wrf-history-fallback-v1")
        ds.setncattr("tier", tier)
        ds.setncattr("step", int(step))
        ds.setncattr("stage", WRF_FALLBACK_STAGE)
        ds.setncattr("source_start", str(source_start))
        ds.setncattr("source_end", str(source_end))
        ds.setncattr("history_fraction", float(fraction))
        ds.setncattr("dt_seconds", float(attrs["dt"]))
        ds.setncattr("limitation", "Linear interpolation between real CPU WRF hourly wrfout files; not a Fortran RK/acoustic savepoint.")
        for name in sorted(arrays):
            array = np.asarray(arrays[name], dtype=np.float64)
            dims = _dims_for_array(name, array)
            var = ds.createVariable(name, "f8", dims, zlib=True, complevel=1)
            var[:] = array
            var.setncattr("units", _units_for_field(name))
            var.setncattr("stagger", _stagger_for_field(name))
            var.setncattr("dimension_order", ",".join(dims))
            var.setncattr("source", "real CPU WRF wrfout history fallback")


def ensure_wrf_reference_fixture(
    tier: str,
    steps: int,
    fixture_root: Path = WRF_REFERENCE_FIXTURE_ROOT,
) -> dict[str, object]:
    """Create or reuse the F1 fallback fixture from real CPU WRF history.

    This is intentionally a fallback, not a parity-grade RK/acoustic oracle. It
    keeps the comparator honest by separating the WRF reference path from JAX
    emissions when true Fortran savepoint hooks are unavailable.
    """

    fixture_dir = fixture_root / tier
    manifest_path = fixture_dir / "manifest.json"
    expected_files = [_fixture_file(tier, step, fixture_root=fixture_root) for step in range(1, int(steps) + 1)]
    if manifest_path.exists() and all(path.exists() for path in expected_files):
        try:
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("source_start") == str(SOURCE_WRFOUT)
                and manifest.get("steps") == int(steps)
                and manifest.get("stage") == WRF_FALLBACK_STAGE
                and manifest.get("self_compare") is False
            ):
                return manifest
        except json.JSONDecodeError:
            pass

    _state, attrs, _extras = _load_initial_state(tier)
    source_start = SOURCE_WRFOUT
    source_end = _resolve_next_wrfout(source_start)
    start_arrays = _load_wrf_history_arrays(source_start, attrs)
    end_arrays = _load_wrf_history_arrays(source_end, attrs)
    dt_seconds = float(attrs["dt"])
    files: list[str] = []
    for step in range(1, int(steps) + 1):
        fraction = min(float(step) * dt_seconds / WRF_HISTORY_INTERVAL_SECONDS, 1.0)
        arrays = _interpolate_history(start_arrays, end_arrays, fraction, WRF_HISTORY_INTERVAL_SECONDS)
        path = _fixture_file(tier, step, fixture_root=fixture_root)
        _write_wrf_fixture_nc(
            path,
            arrays,
            tier=tier,
            step=step,
            attrs=attrs,
            source_start=source_start,
            source_end=source_end,
            fraction=fraction,
        )
        files.append(str(path))

    ladder = _coupled_ladder()
    variables = {}
    sample_arrays = _interpolate_history(
        start_arrays,
        end_arrays,
        min(dt_seconds / WRF_HISTORY_INTERVAL_SECONDS, 1.0),
        WRF_HISTORY_INTERVAL_SECONDS,
    )
    for name, array in sample_arrays.items():
        entry = ladder["fields"].get(name, {"abs": None, "rel": None, "units": _units_for_field(name)})  # type: ignore[union-attr]
        variables[name] = {
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
            "units": str(entry["units"]),
            "stagger": _stagger_for_field(name),
            "dimension_order": list(_dims_for_array(name, np.asarray(array))),
            "abs_tolerance": entry.get("abs"),
            "rel_tolerance": entry.get("rel"),
        }

    manifest = {
        "fixture_id": f"f1-wrf-b6-100step-{tier}-history-fallback",
        "schema": "f1-real-wrf-history-fallback-v1",
        "source_type": "real_wrf_history_fallback",
        "self_compare": False,
        "stage": WRF_FALLBACK_STAGE,
        "tier": tier,
        "steps": int(steps),
        "dt_seconds": dt_seconds,
        "history_interval_seconds": WRF_HISTORY_INTERVAL_SECONDS,
        "source_start": str(source_start),
        "source_end": str(source_end),
        "source_start_sha256": _sha256_path(source_start),
        "source_end_sha256": _sha256_path(source_end),
        "source_wrf_commit": WRF_COMMIT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "file_sha256": {path: _sha256_path(Path(path)) for path in files},
        "variables": variables,
        "license_notes": "Derived, compact column/patch fixture from local real CPU WRF NetCDF history; original WRF outputs stay outside git.",
        "limitation": (
            "Fallback only: linear interpolation between hourly WRF history outputs. "
            "No per-RK-stage or per-acoustic-substep Fortran savepoints were available."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _set_source_paths(wrfout: Path, wrfbdy: Path | None = None) -> None:
    global SOURCE_WRFOUT, SOURCE_WRFBDY
    SOURCE_WRFOUT = wrfout
    SOURCE_WRFBDY = wrfbdy if wrfbdy is not None else wrfout.parent / "wrfbdy_d01"
    _m6b4._set_source_wrfout(wrfout)


def _cfg(attrs: dict[str, object]) -> CoupledStepConfig:
    return CoupledStepConfig(
        dt=float(attrs["dt"]),
        dx=float(attrs["dx"]),
        dy=float(attrs["dy"]),
        acoustic_substeps=ACOUSTIC_SUBSTEPS_PER_RK,
        rk_order=RK_ORDER,
        epssm=float(attrs["epssm"]),
        top_lid=bool(attrs.get("top_lid", False)),
        physics_enabled=True,
        boundary_enabled=True,
    )


def _slice_attrs(attrs: dict[str, object]) -> tuple[slice, slice]:
    y0, y1 = [int(v) for v in attrs["halo_slice_y"]]  # type: ignore[index]
    x0, x1 = [int(v) for v in attrs["halo_slice_x"]]  # type: ignore[index]
    return slice(y0, y1), slice(x0, x1)


def _optional_wrfout_var(ds: Dataset, name: str, attrs: dict[str, object], shape: tuple[int, ...], default: float = 0.0) -> np.ndarray:
    if name not in ds.variables:
        return np.ones(shape, dtype=np.float64) * default
    ys, xs = _slice_attrs(attrs)
    data = ds.variables[name]
    if len(data.shape) == 4:
        return np.asarray(data[0, :, ys, xs], dtype=np.float64)
    if len(data.shape) == 3:
        return np.asarray(data[0, ys, xs], dtype=np.float64)
    return np.asarray(data[:], dtype=np.float64)


def _pack_wrfbdy_leaf(decoded: dict[str, Any], var: str, z_len: int, side_len: int, cadence_s: float) -> np.ndarray:
    packed = np.zeros((2, 4, z_len, side_len), dtype=np.float64)
    for side_index, side in enumerate(SIDES):
        base = np.asarray(decoded["variables"][var]["sides"][side]["boundary"][0], dtype=np.float64)
        tendency = np.asarray(decoded["variables"][var]["sides"][side]["tendency"][0], dtype=np.float64)
        if base.ndim == 1:
            n = min(side_len, base.shape[0])
            packed[0, side_index, 0, :n] = base[:n]
            packed[1, side_index, 0, :n] = base[:n] + float(cadence_s) * tendency[:n]
        else:
            z = min(z_len, base.shape[0])
            n = min(side_len, base.shape[-1])
            packed[0, side_index, :z, :n] = base[:z, :n]
            packed[1, side_index, :z, :n] = base[:z, :n] + float(cadence_s) * tendency[:z, :n]
    return packed


def _physics_boundary_extras(attrs: dict[str, object], acoustic_state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    theta = acoustic_state["theta"]
    nz, ny, nx = theta.shape
    side_len = max(nx + 1, ny + 1)
    mass = theta.shape
    surface = (ny, nx)
    with Dataset(SOURCE_WRFOUT) as ds:
        extras = {
            "qv": _optional_wrfout_var(ds, "QVAPOR", attrs, mass, 0.010),
            "qc": _optional_wrfout_var(ds, "QCLOUD", attrs, mass, 0.0),
            "qr": _optional_wrfout_var(ds, "QRAIN", attrs, mass, 0.0),
            "qi": _optional_wrfout_var(ds, "QICE", attrs, mass, 0.0),
            "qs": _optional_wrfout_var(ds, "QSNOW", attrs, mass, 0.0),
            "qg": _optional_wrfout_var(ds, "QGRAUP", attrs, mass, 0.0),
            "qke": _optional_wrfout_var(ds, "QKE", attrs, mass, 0.20),
            "t_skin": _optional_wrfout_var(ds, "TSK", attrs, surface, 295.0),
            "xland": _optional_wrfout_var(ds, "XLAND", attrs, surface, 1.0),
            "lakemask": _optional_wrfout_var(ds, "LAKEMASK", attrs, surface, 0.0),
        }
    decoded = decode_wrfbdy(SOURCE_WRFBDY, variables=("U", "V", "T", "QVAPOR", "PH", "MU"), time_index=0)
    extras.update(
        {
            "u_bdy": _pack_wrfbdy_leaf(decoded, "U", nz, side_len, 3600.0),
            "v_bdy": _pack_wrfbdy_leaf(decoded, "V", nz, side_len, 3600.0),
            "theta_bdy": _pack_wrfbdy_leaf(decoded, "T", nz, side_len, 3600.0),
            "qv_bdy": _pack_wrfbdy_leaf(decoded, "QVAPOR", nz, side_len, 3600.0),
            "ph_bdy": _pack_wrfbdy_leaf(decoded, "PH", nz + 1, side_len, 3600.0),
            "mu_bdy": _pack_wrfbdy_leaf(decoded, "MU", 1, side_len, 3600.0),
        }
    )
    return extras


def _load_initial_state(tier: str) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, np.ndarray]]:
    state, attrs = _m6b4._load_initial_state(tier)
    acoustic = {name: np.asarray(value, dtype=np.float64) for name, value in state.items()}
    return acoustic, attrs, _physics_boundary_extras(attrs, acoustic)


def _snapshot(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(state[name], dtype=np.float64) for name in COMPARE_FIELDS}


def _coupled_steps(tier: str, steps: int) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    state, attrs, extras = _load_initial_state(tier)
    snapshots = coupled_timesteps_wrf(
        AcousticLoopState.from_mapping(state),
        _m6b4._metrics(),
        _cfg(attrs),
        steps=int(steps),
        extras=extras,
    )
    arrays = [{name: np.asarray(value) for name, value in snapshot.items()} for snapshot in snapshots]
    return arrays, {"attrs": attrs, "initial": state, "extras": extras}


def _run_id(run_id: str) -> str:
    return str(run_id).replace("m6b1", "m6b6", 1).replace("m6b4", "m6b6", 1).replace("m6b5", "m6b6", 1)


def _var_meta(arrays: dict[str, np.ndarray], roles: dict[str, str]) -> dict[str, VariableMetadata]:
    ladder = _coupled_ladder()
    meta = {}
    for name, array in arrays.items():
        arr = np.asarray(array)
        entry = ladder["fields"].get(name, {"units": "operator-native", "dtype": str(arr.dtype)})  # type: ignore[union-attr]
        stagger = "scalar"
        if name in {"mu", "mut", "mudf", "muts", "muave", "mu_bdy_tend"}:
            stagger = "mass"
        elif name in {"theta", "t_2ave", "theta_phys_tend", "qv_phys_tend", "qc_phys_tend", "qr_phys_tend", "qi_phys_tend", "qs_phys_tend", "qg_phys_tend", "qke_phys_tend"}:
            stagger = "mass"
        elif name in {"ww", "ph_tend", "w", "ph", "w_phys_tend"}:
            stagger = "w"
        elif name in {"u", "u_phys_tend"}:
            stagger = "u"
        elif name in {"v", "v_phys_tend"}:
            stagger = "v"
        meta[name] = VariableMetadata(
            name=name,
            dtype=str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            stagger=stagger,
            units=str(entry["units"]),
            provenance="WRF solve_em.F coupled timestep with M5 physics adapters and Gen2 wrfbdy boundary replay",
            role=roles.get(name, "expected"),
        )
    return meta


def _savepoint(
    *,
    tier: str,
    step: int,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, object],
    roles: dict[str, str],
) -> Savepoint:
    metadata_attrs = {k: v for k, v in attrs.items() if k != "run_id"}
    metadata_attrs["namelist_physics_boundary_on"] = NAMELIST_PHYSICS_BOUNDARY_ON
    metadata_attrs["wrfbdy_path"] = str(SOURCE_WRFBDY)
    return Savepoint(
        metadata=SavepointMetadata(
            run_id=f"{_run_id(str(attrs['run_id']))}-coupled-step{step:03d}",
            wrf_version="WRF-Gen2-artifact",
            wrf_commit=WRF_COMMIT,
            namelist_hash=hashlib.sha256(json.dumps(metadata_attrs, sort_keys=True).encode()).hexdigest(),
            source_path=str(SOURCE_WRFOUT),
            domain_index=2,
            tier=tier,
            operator="coupled_step",
            boundary="coupled_step_complete",
            dt_seconds=float(attrs["dt"]),
            rk_stage_index=RK_ORDER,
            acoustic_substep_index=ACOUSTIC_SUBSTEPS_PER_RK,
            map_factors={"MAPFAC_MY": {"min": attrs["mapfac_min"], "max": attrs["mapfac_max"]}},
            vertical_grid={
                "kind": "wrf-hybrid-eta",
                "nz": int(np.asarray(arrays["theta"]).shape[0]),
                "coupled_step_attrs": metadata_attrs,
                "wrf_source_order": [
                    "solve_em.F:1437-1704 non-timesplit physics setup and drivers",
                    "solve_em.F:2034-2285 specified lateral-boundary tendencies",
                    "solve_em.F:3065-4363 acoustic small_steps loop",
                    "solve_em.F:6765 Runge_Kutta_loop ends",
                ],
            },
            variables=_var_meta(arrays, roles),
            created_utc=datetime.now(timezone.utc).isoformat(),
            notes=(
                "Sanitizer-off M6B6 WRF-shaped coupled-step extraction from real Canary d02 wrfout and Gen2 wrfbdy. "
                "Expected arrays are generated through the validation-only B-direct lane; direct Fortran hook bodies "
                "remain empty pending hook-ABI follow-up."
            ),
        ),
        arrays=arrays,
    )


def emit_jax_savepoints(tier: str, steps: int, output: Path, snapshots: list[dict[str, np.ndarray]] | None = None) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    if snapshots is None:
        snapshots, context = _coupled_steps(tier, steps)
    else:
        _state, attrs, extras = _load_initial_state(tier)
        context = {"attrs": attrs, "extras": extras}
    attrs = dict(context["attrs"])
    roles = {name: "expected" for name in COMPARE_FIELDS}
    files = []
    for step, arrays in enumerate(snapshots, start=1):
        path = output / f"coupled_step_complete_step{step:03d}.h5"
        write_savepoint(path, _savepoint(tier=tier, step=step, arrays=arrays, attrs=attrs, roles=roles))
        files.append(path)

    manifest = {
        "tier": tier,
        "source_path": str(SOURCE_WRFOUT),
        "source_sha256": _sha256_path(SOURCE_WRFOUT),
        "wrfbdy_path": str(SOURCE_WRFBDY),
        "wrfbdy_sha256": _sha256_path(SOURCE_WRFBDY),
        "run_id": _run_id(str(attrs["run_id"])),
        "steps": list(range(1, int(steps) + 1)),
        "rk_order": RK_ORDER,
        "acoustic_substeps_per_rk": ACOUSTIC_SUBSTEPS_PER_RK,
        "files": [str(path) for path in files],
        "file_sha256": {str(path): _sha256_path(path) for path in files},
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "attrs": attrs,
        "namelist_physics_boundary_on": NAMELIST_PHYSICS_BOUNDARY_ON,
        "sanitizer_mode": "off",
        "cpu_operator_path": True,
        "direct_relinked_wrf": False,
        "ground_truth": "none; JAX candidate emission only",
        "composition_order": "dycore_step -> Thompson mp=8 -> MYNN bl=5 -> RRTMG LW/SW ra=4/4 -> Gen2 wrfbdy boundary",
        "tolerance_rationale": "M6B5 dycore-step tolerance plus 1e-10 abs cap for physics-tendency fields per ADR-007 fp64-strict exception.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def emit_tier(tier: str, steps: int, output: Path, snapshots: list[dict[str, np.ndarray]] | None = None) -> dict[str, object]:
    """Backward-compatible wrapper for old call sites.

    The F1 comparator treats this as candidate JAX emission only; it is never
    used as expected truth.
    """

    return emit_jax_savepoints(tier, steps, output, snapshots)


def _coupled_ladder() -> dict[str, object]:
    ladder = load_tolerance_ladder()
    coupled = dict(ladder["coupled_step_tolerances"])  # type: ignore[index]
    return {
        "schema_version": ladder["schema_version"],
        "perturbation_rule": ladder["perturbation_rule"],
        "fields": coupled["fields"],
    }


def _field_operator(name: str) -> str:
    if name in PHYSICS_TENDENCY_FIELDS:
        return "physics_or_boundary_tendency"
    if name in {"u", "v", "w", "theta", "mu", "mut", "muts", "muave", "mudf", "ww", "ph", "p", "ph_tend", "t_2ave"}:
        return "dycore_coupled_step"
    return "unknown"


def _compare_snapshot(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray], ladder: dict[str, object]) -> dict[str, object]:
    fields = {}
    for name in COMPARE_FIELDS:
        got = np.asarray(actual[name])
        exp = np.asarray(expected[name])
        item = field_compare(name, got, exp, ladder)
        common_shape = tuple(min(a, b) for a, b in zip(exp.shape, got.shape))
        slices = tuple(slice(0, dim) for dim in common_shape)
        delta = got[slices] - exp[slices]
        item["mean_abs_delta"] = float(np.nanmean(np.abs(delta))) if delta.size else float("nan")
        item["operator"] = _field_operator(name)
        fields[name] = item
    return {"passed": all(bool(item["passed"]) for item in fields.values()), "fields": fields}


def load_wrf_savepoints(
    step: int,
    stage: str = WRF_FALLBACK_STAGE,
    *,
    tier: str = "column",
    fixture_root: Path = WRF_REFERENCE_FIXTURE_ROOT,
) -> dict[str, np.ndarray]:
    path = _fixture_file(tier, int(step), stage, fixture_root)
    if not path.exists():
        raise FileNotFoundError(f"WRF reference savepoint missing: {path}")
    with Dataset(path) as ds:
        missing = [name for name in COMPARE_FIELDS if name not in ds.variables]
        if missing:
            raise ValueError(f"{path} missing WRF reference variables: {', '.join(missing)}")
        return {name: np.asarray(ds.variables[name][:], dtype=np.float64) for name in COMPARE_FIELDS}


def compare_jax_vs_wrf(
    *,
    tier: str,
    step: int,
    stage: str,
    jax_snapshot: dict[str, np.ndarray],
    fixture_root: Path,
    ladder: dict[str, object],
) -> dict[str, object]:
    expected = load_wrf_savepoints(step, stage, tier=tier, fixture_root=fixture_root)
    compared = _compare_snapshot(
        {name: np.asarray(value) for name, value in expected.items()},
        {name: np.asarray(value) for name, value in jax_snapshot.items()},
        ladder,
    )
    first_failed = next((name for name, item in compared["fields"].items() if not bool(item["passed"])), None)
    result = {
        "step": int(step),
        "tier": tier,
        "stage": stage,
        "path": str(_fixture_file(tier, int(step), stage, fixture_root)),
        "boundary": "real_wrf_reference",
        "first_failed_field": first_failed,
        "first_failed_operator": _field_operator(str(first_failed)) if first_failed is not None else None,
        **compared,
    }
    return result


def _first_divergence(results: list[dict[str, object]]) -> dict[str, object] | None:
    for result in results:
        if bool(result["passed"]):
            continue
        field = result.get("first_failed_field")
        if not field:
            continue
        field_result = result["fields"][field]  # type: ignore[index]
        return {
            "step": int(result["step"]),
            "stage": str(result["stage"]),
            "field": str(field),
            "operator": str(result["first_failed_operator"]),
            "max_abs_delta": float(field_result["max_abs_delta"]),  # type: ignore[index]
            "mean_abs_delta": float(field_result["mean_abs_delta"]),  # type: ignore[index]
            "tolerance": float(field_result["tolerance"]),  # type: ignore[index]
            "location": field_result["location"],  # type: ignore[index]
        }
    return None


def _delta_histogram(results: list[dict[str, object]]) -> dict[str, object]:
    values = [
        float(item["max_abs_delta"])
        for result in results
        for item in result["fields"].values()  # type: ignore[union-attr]
        if np.isfinite(float(item["max_abs_delta"]))
    ]
    if not values:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=10)
    return {"bins": [float(value) for value in edges], "counts": [int(value) for value in counts]}


def compare_tier(tier: str, steps: int, savepoint_root: Path) -> dict[str, object]:
    output = savepoint_root / tier / "jax"
    actual_steps, _context = _coupled_steps(tier, steps)
    jax_manifest = emit_jax_savepoints(tier, steps, output, actual_steps)
    wrf_manifest = ensure_wrf_reference_fixture(tier, int(steps), WRF_REFERENCE_FIXTURE_ROOT)
    ladder = _coupled_ladder()
    results = []
    for step, actual in enumerate(actual_steps, start=1):
        results.append(
            compare_jax_vs_wrf(
                tier=tier,
                step=step,
                stage=WRF_FALLBACK_STAGE,
                jax_snapshot=actual,
                fixture_root=WRF_REFERENCE_FIXTURE_ROOT,
                ladder=ladder,
            )
        )
    passed = all(bool(item["passed"]) for item in results)
    first = _first_divergence(results)
    if passed:
        outcome = f"REAL-WRF-PARITY-ACHIEVED-THROUGH-STEP-{int(steps)}"
    elif first is not None:
        outcome = (
            "REAL-WRF-DIVERGENCE-AT-"
            f"STEP-{first['step']}-STAGE-{first['stage']}-FIELD-{first['field']}-OPERATOR-{first['operator']}"
        )
    else:
        outcome = "REAL-WRF-DIVERGENCE-WITHOUT-FIELD-ATTRIBUTION"
    return {
        "operator": "coupled_step",
        "tier": tier,
        "passed": bool(passed),
        "outcome": outcome,
        "savepoint_count": int(steps),
        "manifest": jax_manifest,
        "wrf_manifest": wrf_manifest,
        "oracle": {
            "source_type": wrf_manifest["source_type"],
            "self_compare": False,
            "stage": WRF_FALLBACK_STAGE,
            "limitation": wrf_manifest["limitation"],
        },
        "first_divergence": first,
        "delta_histogram": _delta_histogram(results),
        "results": results,
        "tolerance_ladder_path": str(ROOT / "src/gpuwrf/validation/tolerance_ladder.json"),
        "sanitizer_mode": "off",
        "namelist_physics_boundary_on": NAMELIST_PHYSICS_BOUNDARY_ON,
        "transfer_audit": {
            "h2d_d2h_inside_timestep_loop_bytes": 0,
            "note": "Comparator reads HDF5 before isolated validation calls; no production timestep loop is executed.",
        },
    }


def _write_kill_gate(payload: dict[str, object]) -> dict[str, object]:
    diverging = 0
    for tier in payload["tiers"].values():  # type: ignore[union-attr]
        first = tier["results"][0]  # type: ignore[index]
        diverging += sum(1 for item in first["fields"].values() if not bool(item["passed"]))  # type: ignore[index]
    status = {
        "operator": "coupled_step",
        "step": 1,
        "diverging_field_count": int(diverging),
        "threshold": 15,
        "passed": bool(diverging <= 15),
        "decision": "PROCEED_TO_M6_PERF_DESIGN" if diverging <= 15 else "STOP_ESCALATE_M6B6",
    }
    (SPRINT / "proof_kill_gate_status.txt").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def synthetic_dryrun() -> dict[str, object]:
    actual_steps, _ = _coupled_steps("column", 1)
    expected = actual_steps[-1]
    ladder = _coupled_ladder()
    clean = _compare_snapshot(expected, actual_steps[-1], ladder)["fields"]
    perturbed = {}
    caught = True
    for name in COMPARE_FIELDS:
        bad = {field: np.array(value, copy=True) for field, value in expected.items()}
        tol = float(clean[name]["tolerance"])  # type: ignore[index]
        bad[name].flat[0] += 20.0 * tol
        result = field_compare(name, actual_steps[-1][name], bad[name], ladder)
        perturbed[name] = result
        caught = caught and not bool(result["passed"])
    payload = {
        "operator": "coupled_step",
        "clean_self_compare_passed": all(bool(item["passed"]) for item in clean.values()),
        "boundary_field_perturbations_caught": bool(caught),
        "clean": clean,
        "perturbed": perturbed,
        "passed": bool(all(bool(item["passed"]) for item in clean.values()) and caught),
        "source_path": str(SOURCE_WRFOUT),
        "wrfbdy_path": str(SOURCE_WRFBDY),
        "namelist_physics_boundary_on": NAMELIST_PHYSICS_BOUNDARY_ON,
        "sanitizer_mode": "off",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    (SPRINT / "proof_synthetic_dryrun_m6b6.json").write_text(text + "\n")
    (SPRINT / "proof_synthetic_dryrun_m6b6.txt").write_text(text + "\n")
    return payload


def _summary_text(payload: dict[str, object]) -> str:
    lines = [
        f"operator={payload['operator']}",
        f"outcome={payload['outcome']}",
        f"passed={payload['passed']}",
        "physics=Thompson mp=8, MYNN bl=5, RRTMG LW/SW ra=4/4",
        "boundary=Gen2 wrfbdy lateral replay",
    ]
    for tier, result in payload["tiers"].items():  # type: ignore[union-attr]
        max_delta = 0.0
        max_field = ""
        for step in result["results"]:  # type: ignore[index]
            for field, item in step["fields"].items():
                delta = float(item["max_abs_delta"])
                if delta >= max_delta:
                    max_delta = delta
                    max_field = f"{tier}/step{step['step']}/{field}"
        lines.append(f"{tier}: passed={result['passed']} max_abs_delta={max_delta} at {max_field}")  # type: ignore[index]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("column", "patch16", "golden", "all"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--savepoint-root", type=Path, default=SPRINT / "savepoints")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source-wrfout", type=Path, default=DEFAULT_GEN2_WRFOUT)
    parser.add_argument("--source-wrfbdy", type=Path, default=None)
    parser.add_argument("--synthetic-dryrun", action="store_true")
    args = parser.parse_args()

    _set_source_paths(args.source_wrfout, args.source_wrfbdy)
    if args.synthetic_dryrun:
        payload = synthetic_dryrun()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 2
    if args.tier is None:
        parser.error("--tier is required unless --synthetic-dryrun is set")

    tiers = ("column", "patch16", "golden") if args.tier == "all" else (args.tier,)
    tier_results = {tier: compare_tier(tier, int(args.steps), args.savepoint_root) for tier in tiers}
    passed = all(bool(item["passed"]) for item in tier_results.values())
    outcomes = {str(item["outcome"]) for item in tier_results.values()}
    outcome = "SEVENTH-COUPLED-STEP-PARITY-ACHIEVED" if passed else sorted(outcomes)[0]
    payload: dict[str, object] = {
        "operator": "coupled_step",
        "passed": bool(passed),
        "outcome": outcome,
        "tiers": tier_results,
        "operational_compatibility": {
            "sp_coupled_step_complete hook": "validation-only",
            "coupled_step.py callable": "validation-only",
            "M5 physics adapter invocations": "validation-only",
            "Gen2 wrfbdy boundary replay": "validation-only",
            "per-coupled-step tolerance entries": "validation-only",
            "schema v7 extension": "validation-only",
        },
    }
    if args.tier == "all":
        payload["kill_gate"] = _write_kill_gate(payload)
    output = args.output
    if output is None:
        suffix = "all" if args.tier == "all" else str(args.tier)
        output = SPRINT / ("proof_coupled_step_parity.json" if suffix == "all" else f"proof_coupled_step_parity_{suffix}.json")
    text = json.dumps(payload, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    if args.tier == "all":
        (SPRINT / "proof_coupled_step_parity.txt").write_text(_summary_text(payload))
    print(text)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
