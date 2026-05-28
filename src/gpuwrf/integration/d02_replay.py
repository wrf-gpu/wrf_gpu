"""ADR-023 d02 boundary-replay integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import BaseState, State, Tendencies
from gpuwrf.coupling.boundary_apply import BoundaryConfig, SIDE_INDEX, SIDES, apply_lateral_boundaries
from gpuwrf.coupling.driver import (
    DEFAULT_RADIATION_CADENCE_STEPS,
    _surface_diagnostics_for_output,
    run_start_label,
    sanitize_state_with_stats,
    state_diagnostics,
)
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig, run_acoustic_scan
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.damping import RayleighConfig
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.io.boundary_replay import decode_wrfbdy, wrfbdy_path_for_run
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.gen2_wrfout_loader import normalize_valid_time
from gpuwrf.io.land_state import load_prescribed_land_state
from gpuwrf.paths import cache_root, reference_path
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name


config.update("jax_enable_x64", True)

_DEBUG = os.environ.get("GPUWRF_D02_REPLAY_DEBUG", "").lower() not in {"", "0", "false", "no", "off"}
_DEBUG_START = time.perf_counter()


def _debug(message: str) -> None:
    if _DEBUG:
        print(f"[d02-replay +{time.perf_counter() - _DEBUG_START:8.3f}s] {message}", flush=True)


def _shape_dtype(value: Any) -> str:
    shape = getattr(value, "shape", np.shape(value))
    dtype = getattr(value, "dtype", None)
    return f"shape={tuple(shape)} dtype={dtype}"


_debug("module import complete")

_TRACE_SIZE_RE = re.compile(r"(?:^|[\s,{])(?:bytes|byte_size|size|num_bytes|NumBytes)\s*[:=]\s*(\d+)", re.IGNORECASE)
_TRACE_DIRECTION_RE = re.compile(
    r"(host_to_device|device_to_host|memcpyh2d|memcpyd2h|\bh2d\b|\bd2h\b)",
    re.IGNORECASE,
)


def _flatten_trace_args(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key}:{_flatten_trace_args(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_trace_args(item) for item in value)
    return str(value)


def _trace_event_size(value: Any) -> int:
    size = 0
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"bytes", "byte size", "byte_size", "size", "numbytes", "num_bytes"}:
                try:
                    size = max(size, int(item))
                except (TypeError, ValueError):
                    pass
            size = max(size, _trace_event_size(item))
        return size
    if isinstance(value, (list, tuple)):
        for item in value:
            size = max(size, _trace_event_size(item))
        return size
    if isinstance(value, str):
        for match in _TRACE_SIZE_RE.finditer(value):
            size = max(size, int(match.group(1)))
    return size


def _read_trace_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return json.loads(handle.read())
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_trace_text(path: Path) -> str:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _count_replay_transfer_bytes(trace_dir: Path) -> tuple[int, int, list[str]]:
    """Count actual H2D/D2H JSON trace events while ignoring D2D and xplane text hits."""

    h2d = 0
    d2h = 0
    matched: list[str] = []
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if not (path.name.endswith(".json") or path.name.endswith(".trace.json.gz")):
            continue
        text = _read_trace_text(path)
        if not _TRACE_DIRECTION_RE.search(text):
            continue
        payload = _read_trace_json(path)
        if not isinstance(payload, dict):
            continue
        file_matched = False
        for event in payload.get("traceEvents", []):
            if not isinstance(event, dict):
                continue
            name = str(event.get("name", ""))
            args = event.get("args", {})
            detail = f"{name} {_flatten_trace_args(args)}".lower()
            if "d2d" in detail or "device-to-device" in detail:
                continue
            size = _trace_event_size(args)
            if "host_to_device" in detail or "memcpyh2d" in detail or re.search(r"\bh2d\b", detail):
                h2d += size
                file_matched = True
            elif "device_to_host" in detail or "memcpyd2h" in detail or re.search(r"\bd2h\b", detail):
                d2h += size
                file_matched = True
        if file_matched:
            matched.append(str(path))
    return h2d, d2h, matched

P0_THETA_OFFSET_K = 300.0
DEFAULT_REPLAY_RUN_DIR = reference_path("runs", "wrf_l3", "20260521_18z_l3_24h_20260522T133443Z")
DEFAULT_OUTPUT_FIELD_PATH = cache_root() / "outputs" / "m6x_d02_replay" / "proof_d02_replay_fields.npz"
DEFAULT_TRACE_ROOT = Path(os.environ.get("GPUWRF_TMPDIR", str(cache_root() / "tmp")))
EXPECTED_L2_D02_MASS_SHAPE_YX = (66, 159)


@dataclass(frozen=True)
class ReplayConfig:
    """Static runtime knobs for the d02 ADR-023 replay."""

    dt_s: float = 1.0
    duration_s: float = 3600.0
    n_acoustic: int = 4
    radiation_cadence_steps: int = DEFAULT_RADIATION_CADENCE_STEPS
    final_radiation: bool = True
    boundary_config: BoundaryConfig = BoundaryConfig(update_cadence_s=3600.0)
    rayleigh_coefficient: float = 0.0


class ReplayCase(NamedTuple):
    """Device-resident inputs for one Gen2 d02 replay."""

    run: Gen2Run
    state: State
    tendencies: Tendencies
    grid: GridSpec
    metrics: DycoreMetrics
    base_state: BaseState
    previous_pressure: object
    metadata: dict[str, Any]


class StepDiagnostics(NamedTuple):
    """Scalar per-step diagnostics emitted from the timestep scan."""

    finite_after_sanitize: object
    candidate_nonfinite_count: object
    candidate_clip_count: object
    candidate_changed_count: object
    w_abs_max_m_s: object
    theta_min_k: object
    theta_max_k: object


def _load(run: Gen2Run, domain: str, var: str, time_index: int):
    start = time.perf_counter()
    _debug(f"load start {domain}:{var}[{time_index}]")
    value = run.load(domain, var, time=time_index, lazy=False)
    _debug(f"load done {domain}:{var}[{time_index}] {_shape_dtype(value)} elapsed_s={time.perf_counter() - start:.3f}")
    return value


def _optional_load(run: Gen2Run, domain: str, var: str, time: int, fallback):
    try:
        return _load(run, domain, var, time)
    except KeyError:
        _debug(f"optional load missing {domain}:{var}[{time}], using fallback {_shape_dtype(fallback)}")
        return fallback


def _wrfbdy_width_for_run(run: Gen2Run, fallback: int = 5) -> tuple[int, str | None, str]:
    try:
        path = wrfbdy_path_for_run(run, "d01")
        decoded = decode_wrfbdy(path, variables=("U",), time_index=0)
        return int(decoded.get("bdy_width", fallback)), str(path), "decode_wrfbdy"
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return int(fallback), None, "fallback"


def _field_sides_3d(field: np.ndarray, width: int = 1) -> dict[str, np.ndarray]:
    """Return WRF-ordered `(bdy_width, z, side_index)` side strips."""

    data = np.asarray(field)
    x_width = min(int(width), int(data.shape[2]))
    y_width = min(int(width), int(data.shape[1]))
    return {
        "W": np.moveaxis(data[:, :, :x_width], 2, 0),
        "E": np.moveaxis(np.flip(data[:, :, -x_width:], axis=2), 2, 0),
        "S": np.moveaxis(data[:, :y_width, :], 1, 0),
        "N": np.moveaxis(np.flip(data[:, -y_width:, :], axis=1), 1, 0),
    }


def _field_sides_2d(field: np.ndarray, width: int = 1) -> dict[str, np.ndarray]:
    """Return WRF-ordered `(bdy_width, side_index)` side strips."""

    data = np.asarray(field)
    x_width = min(int(width), int(data.shape[1]))
    y_width = min(int(width), int(data.shape[0]))
    return {
        "W": data[:, :x_width].T,
        "E": np.flip(data[:, -x_width:], axis=1).T,
        "S": data[:y_width, :],
        "N": np.flip(data[-y_width:, :], axis=0),
    }


def _pack_history_3d(
    run: Gen2Run,
    domain: str,
    var: str,
    *,
    ntimes: int,
    z_len: int,
    max_side: int,
    bdy_width: int,
    dtype: Any,
    transform=None,
) -> np.ndarray:
    _debug(f"pack history start {domain}:{var} ntimes={ntimes} z_len={z_len} max_side={max_side}")
    packed = np.zeros((ntimes, 4, int(bdy_width), z_len, max_side), dtype=dtype)
    for time_index in range(ntimes):
        data = np.asarray(_load(run, domain, var, time_index), dtype=dtype)
        if transform is not None:
            data = np.asarray(transform(run, domain, data, time_index), dtype=dtype)
        for side, values in _field_sides_3d(data, int(bdy_width)).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], : values.shape[1], : values.shape[2]] = values
    _debug(f"pack history done {domain}:{var} {_shape_dtype(packed)}")
    return packed


def _pack_history_mu(run: Gen2Run, domain: str, *, ntimes: int, max_side: int, bdy_width: int) -> np.ndarray:
    _debug(f"pack history start {domain}:MU+MUB ntimes={ntimes} max_side={max_side}")
    packed = np.zeros((ntimes, 4, int(bdy_width), 1, max_side), dtype=np.float64)
    for time_index in range(ntimes):
        data = np.asarray(_load(run, domain, "MU", time_index) + _load(run, domain, "MUB", time_index), dtype=np.float64)
        for side, values in _field_sides_2d(data, int(bdy_width)).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], 0, : values.shape[1]] = values
    _debug(f"pack history done {domain}:MU+MUB {_shape_dtype(packed)}")
    return packed


def load_history_boundary_leaves(
    run: Gen2Run,
    grid: GridSpec,
    *,
    domain: str = "d02",
    ntimes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build real lateral replay leaves from the Gen2 d02 hourly wrfout history."""

    _debug(f"load boundary leaves start domain={domain}")
    history_count = len(run.history_files(domain))
    _debug(f"history files counted domain={domain} count={history_count}")
    if history_count < 2:
        raise FileNotFoundError(f"{run.path} has fewer than two wrfout_{domain} history files")
    n = min(history_count, int(ntimes if ntimes is not None else history_count))
    max_side = int(max(grid.nx + 1, grid.ny + 1))
    bdy_width, wrfbdy_path, width_source = _wrfbdy_width_for_run(run, fallback=5)

    def add_theta(_run: Gen2Run, _domain: str, data: np.ndarray, _time_index: int) -> np.ndarray:
        return data + P0_THETA_OFFSET_K

    def add_phb(_run: Gen2Run, _domain: str, data: np.ndarray, time_index: int) -> np.ndarray:
        return data + np.asarray(_load(_run, _domain, "PHB", time_index), dtype=data.dtype)

    leaves_np = {
        "u_bdy": _pack_history_3d(run, domain, "U", ntimes=n, z_len=grid.nz, max_side=max_side, bdy_width=bdy_width, dtype=np.float32),
        "v_bdy": _pack_history_3d(run, domain, "V", ntimes=n, z_len=grid.nz, max_side=max_side, bdy_width=bdy_width, dtype=np.float32),
        "theta_bdy": _pack_history_3d(
            run,
            domain,
            "T",
            ntimes=n,
            z_len=grid.nz,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float32,
            transform=add_theta,
        ),
        "qv_bdy": _pack_history_3d(
            run,
            domain,
            "QVAPOR",
            ntimes=n,
            z_len=grid.nz,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float32,
        ),
        "ph_bdy": _pack_history_3d(
            run,
            domain,
            "PH",
            ntimes=n,
            z_len=grid.nz + 1,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
            transform=add_phb,
        ),
        "mu_bdy": _pack_history_mu(run, domain, ntimes=n, max_side=max_side, bdy_width=bdy_width),
    }
    _debug("device_put boundary leaves start")
    leaves = {name: jax.device_put(jnp.asarray(value)) for name, value in leaves_np.items()}
    _debug("device_put boundary leaves done")
    meta = {
        "source": "Gen2 d02 hourly wrfout side-history replay",
        "source_run_dir": str(run.path),
        "times": int(n),
        "side_order": list(SIDES),
        "padded_side_length": max_side,
        "bdy_width": int(bdy_width),
        "wrfbdy_width_source": width_source,
        "wrfbdy_path": wrfbdy_path,
        "strip_order": "WRF wrfbdy order: outer-to-inner bdy_width, vertical, side_index",
        "schema": "history-strip-pack-v2",
    }
    return leaves, meta


def _nested_axis_coords(child_meta: Any, *, y_len: int, x_len: int) -> tuple[np.ndarray, np.ndarray]:
    ratio = int(getattr(child_meta, "parent_grid_ratio"))
    if ratio <= 1:
        raise ValueError(f"nested child domain requires parent_grid_ratio > 1, got {ratio}")
    y0 = float(int(getattr(child_meta, "j_parent_start")) - 1)
    x0 = float(int(getattr(child_meta, "i_parent_start")) - 1)
    return (
        y0 + np.arange(int(y_len), dtype=np.float64) / float(ratio),
        x0 + np.arange(int(x_len), dtype=np.float64) / float(ratio),
    )


def _interp_parent_horizontal(field: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray) -> np.ndarray:
    data = np.asarray(field)
    if data.ndim not in {2, 3}:
        raise ValueError(f"nested parent interpolation expects 2D or 3D field, got shape {data.shape}")
    y_size = int(data.shape[-2])
    x_size = int(data.shape[-1])
    if y_size < 2 or x_size < 2:
        raise ValueError(f"parent field too small for bilinear interpolation: {data.shape}")

    y = np.clip(np.asarray(y_coords, dtype=np.float64), 0.0, float(y_size - 1))
    x = np.clip(np.asarray(x_coords, dtype=np.float64), 0.0, float(x_size - 1))
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, y_size - 1)
    x1 = np.clip(x0 + 1, 0, x_size - 1)
    wy = y - y0.astype(np.float64)
    wx = x - x0.astype(np.float64)

    if data.ndim == 3:
        f00 = np.take(np.take(data, y0, axis=1), x0, axis=2)
        f10 = np.take(np.take(data, y1, axis=1), x0, axis=2)
        f01 = np.take(np.take(data, y0, axis=1), x1, axis=2)
        f11 = np.take(np.take(data, y1, axis=1), x1, axis=2)
        wy3 = wy[None, :, None]
        wx3 = wx[None, None, :]
        out = (1.0 - wy3) * ((1.0 - wx3) * f00 + wx3 * f01) + wy3 * ((1.0 - wx3) * f10 + wx3 * f11)
    else:
        f00 = np.take(np.take(data, y0, axis=0), x0, axis=1)
        f10 = np.take(np.take(data, y1, axis=0), x0, axis=1)
        f01 = np.take(np.take(data, y0, axis=0), x1, axis=1)
        f11 = np.take(np.take(data, y1, axis=0), x1, axis=1)
        wy2 = wy[:, None]
        wx2 = wx[None, :]
        out = (1.0 - wy2) * ((1.0 - wx2) * f00 + wx2 * f01) + wy2 * ((1.0 - wx2) * f10 + wx2 * f11)
    return np.asarray(out, dtype=data.dtype)


def _pack_nested_parent_history_3d(
    run: Gen2Run,
    child_meta: Any,
    parent_domain: str,
    var: str,
    *,
    ntimes: int,
    child_shape: tuple[int, int, int],
    max_side: int,
    bdy_width: int,
    dtype: Any,
    transform=None,
) -> np.ndarray:
    z_len, y_len, x_len = (int(item) for item in child_shape)
    y_coords, x_coords = _nested_axis_coords(child_meta, y_len=y_len, x_len=x_len)
    packed = np.zeros((ntimes, 4, int(bdy_width), z_len, max_side), dtype=dtype)
    _debug(
        f"pack nested parent history start {parent_domain}:{var} ntimes={ntimes} "
        f"child_shape={(z_len, y_len, x_len)} max_side={max_side}"
    )
    for time_index in range(ntimes):
        parent = np.asarray(_load(run, parent_domain, var, time_index), dtype=dtype)
        if transform is not None:
            parent = np.asarray(transform(run, parent_domain, parent, time_index), dtype=dtype)
        child = _interp_parent_horizontal(parent, y_coords, x_coords)
        for side, values in _field_sides_3d(child, int(bdy_width)).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], : values.shape[1], : values.shape[2]] = values
    _debug(f"pack nested parent history done {parent_domain}:{var} {_shape_dtype(packed)}")
    return packed


def _pack_nested_parent_history_mu(
    run: Gen2Run,
    child_meta: Any,
    parent_domain: str,
    *,
    ntimes: int,
    child_shape: tuple[int, int],
    max_side: int,
    bdy_width: int,
) -> np.ndarray:
    y_len, x_len = (int(item) for item in child_shape)
    y_coords, x_coords = _nested_axis_coords(child_meta, y_len=y_len, x_len=x_len)
    packed = np.zeros((ntimes, 4, int(bdy_width), 1, max_side), dtype=np.float64)
    _debug(
        f"pack nested parent history start {parent_domain}:MU+MUB ntimes={ntimes} "
        f"child_shape={(y_len, x_len)} max_side={max_side}"
    )
    for time_index in range(ntimes):
        parent = np.asarray(
            _load(run, parent_domain, "MU", time_index) + _load(run, parent_domain, "MUB", time_index),
            dtype=np.float64,
        )
        child = _interp_parent_horizontal(parent, y_coords, x_coords)
        for side, values in _field_sides_2d(child, int(bdy_width)).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], 0, : values.shape[1]] = values
    _debug(f"pack nested parent history done {parent_domain}:MU+MUB {_shape_dtype(packed)}")
    return packed


def load_nested_parent_boundary_leaves(
    run: Gen2Run,
    grid: GridSpec,
    *,
    child_domain: str = "d02",
    parent_domain: str = "d01",
    ntimes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build child-sized boundary leaves by interpolating L2 parent-domain history."""

    _debug(f"load nested parent boundary leaves start child={child_domain} parent={parent_domain}")
    child_meta = run.grid(child_domain)
    parent_meta = run.grid(parent_domain)
    parent_count = len(run.history_files(parent_domain))
    child_count = len(run.history_files(child_domain))
    if parent_count < 2:
        raise FileNotFoundError(f"{run.path} has fewer than two wrfout_{parent_domain} history files")
    n = min(parent_count, int(ntimes if ntimes is not None else parent_count))
    parent_times = run.time_axis(parent_domain)
    child_times = run.time_axis(child_domain)
    if parent_times and child_times and parent_times[0] != child_times[0]:
        raise ValueError(
            f"parent/child history start mismatch: {parent_domain}={parent_times[0].isoformat()} "
            f"{child_domain}={child_times[0].isoformat()}"
        )
    max_side = int(max(grid.nx + 1, grid.ny + 1))
    bdy_width, wrfbdy_path, width_source = _wrfbdy_width_for_run(run, fallback=5)

    def add_theta(_run: Gen2Run, _domain: str, data: np.ndarray, _time_index: int) -> np.ndarray:
        return data + P0_THETA_OFFSET_K

    def add_phb(_run: Gen2Run, _domain: str, data: np.ndarray, time_index: int) -> np.ndarray:
        return data + np.asarray(_load(_run, _domain, "PHB", time_index), dtype=data.dtype)

    leaves_np = {
        "u_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "U",
            ntimes=n,
            child_shape=(grid.nz, grid.ny, grid.nx + 1),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float32,
        ),
        "v_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "V",
            ntimes=n,
            child_shape=(grid.nz, grid.ny + 1, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float32,
        ),
        "theta_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "T",
            ntimes=n,
            child_shape=(grid.nz, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float32,
            transform=add_theta,
        ),
        "qv_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "QVAPOR",
            ntimes=n,
            child_shape=(grid.nz, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float32,
        ),
        "ph_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "PH",
            ntimes=n,
            child_shape=(grid.nz + 1, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
            transform=add_phb,
        ),
        "mu_bdy": _pack_nested_parent_history_mu(
            run,
            child_meta,
            parent_domain,
            ntimes=n,
            child_shape=(grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
        ),
    }
    leaves = {name: jax.device_put(jnp.asarray(value)) for name, value in leaves_np.items()}
    meta = {
        "source": "Gen2 L2 parent-domain hourly wrfout nested interpolation",
        "source_run_dir": str(run.path),
        "child_domain": child_domain,
        "parent_domain": parent_domain,
        "times": int(n),
        "child_history_file_count": int(child_count),
        "parent_history_file_count": int(parent_count),
        "side_order": list(SIDES),
        "padded_side_length": max_side,
        "bdy_width": int(bdy_width),
        "wrfbdy_width_source": width_source,
        "wrfbdy_path": wrfbdy_path,
        "strip_order": "WRF wrfbdy order: outer-to-inner bdy_width, vertical, side_index",
        "schema": "nested-parent-strip-pack-v2",
        "parent_grid": {
            "mass_shape": [int(parent_meta.mass_nz), int(parent_meta.mass_ny), int(parent_meta.mass_nx)],
            "dx_m": float(parent_meta.dx_m),
            "dy_m": float(parent_meta.dy_m),
        },
        "child_grid": {
            "mass_shape": [int(grid.nz), int(grid.ny), int(grid.nx)],
            "dx_m": float(grid.projection.dx_m),
            "dy_m": float(grid.projection.dy_m),
        },
        "nesting": {
            "parent_grid_ratio": int(child_meta.parent_grid_ratio),
            "i_parent_start": int(child_meta.i_parent_start),
            "j_parent_start": int(child_meta.j_parent_start),
            "horizontal_interpolation": "bilinear on native WRF parent index coordinates",
        },
    }
    _debug("load nested parent boundary leaves done")
    return leaves, meta


def build_replay_case(
    run_dir: str | Path = DEFAULT_REPLAY_RUN_DIR,
    *,
    domain: str = "d02",
    boundary_domain: str | None = None,
) -> ReplayCase:
    """Load a Gen2 d02 initial state with WRF perturbation/base splits preserved."""

    _debug(f"build_replay_case start run_dir={run_dir} domain={domain} boundary_domain={boundary_domain}")
    run = Gen2Run(run_dir)
    _debug("Gen2Run created")
    grid = run.grid(domain).as_grid_spec()
    _debug(f"grid loaded mass_shape={(grid.nz, grid.ny, grid.nx)}")
    state = State.zeros(grid)
    _debug("State.zeros complete")
    tendencies = Tendencies.zeros(grid)
    _debug("Tendencies.zeros complete")
    metrics = load_wrfinput_metrics(run.wrfinput_file(domain))
    _debug("load_wrfinput_metrics complete")
    land = load_prescribed_land_state(run, domain=domain, time=0)
    _debug("load_prescribed_land_state complete")
    source_domain = boundary_domain or domain
    if source_domain == domain:
        boundary_leaves, boundary_meta = load_history_boundary_leaves(run, grid, domain=domain)
        _debug("load_history_boundary_leaves complete")
    else:
        boundary_leaves, boundary_meta = load_nested_parent_boundary_leaves(
            run,
            grid,
            child_domain=domain,
            parent_domain=source_domain,
        )
        _debug("load_nested_parent_boundary_leaves complete")

    p_perturbation = _load(run, domain, "P", 0)
    pb = _load(run, domain, "PB", 0)
    ph_perturbation = _load(run, domain, "PH", 0)
    phb = _load(run, domain, "PHB", 0)
    mu_perturbation = _load(run, domain, "MU", 0)
    mub = _load(run, domain, "MUB", 0)
    theta = _load(run, domain, "T", 0) + P0_THETA_OFFSET_K
    theta_base = jnp.full_like(theta, P0_THETA_OFFSET_K)

    state = state.replace(
        u=_load(run, domain, "U", 0),
        v=_load(run, domain, "V", 0),
        w=_load(run, domain, "W", 0),
        theta=theta,
        qv=_load(run, domain, "QVAPOR", 0),
        p_total=pb + p_perturbation,
        p_perturbation=p_perturbation,
        ph_total=phb + ph_perturbation,
        ph_perturbation=ph_perturbation,
        mu_total=mub + mu_perturbation,
        mu_perturbation=mu_perturbation,
        qc=_optional_load(run, domain, "QCLOUD", 0, jnp.zeros_like(state.qc)),
        qr=_optional_load(run, domain, "QRAIN", 0, jnp.zeros_like(state.qr)),
        qi=_optional_load(run, domain, "QICE", 0, jnp.zeros_like(state.qi)),
        qs=_optional_load(run, domain, "QSNOW", 0, jnp.zeros_like(state.qs)),
        qg=_optional_load(run, domain, "QGRAUP", 0, jnp.zeros_like(state.qg)),
        Ni=_optional_load(run, domain, "QNICE", 0, jnp.zeros_like(state.Ni)),
        Nr=_optional_load(run, domain, "QNRAIN", 0, jnp.zeros_like(state.Nr)),
        Ns=jnp.zeros_like(state.Ns),
        Ng=jnp.zeros_like(state.Ng),
        qke=_optional_load(run, domain, "QKE", 0, jnp.zeros_like(state.qke)),
        t_skin=land.t_skin,
        soil_moisture=land.soil_moisture[0],
        xland=land.xland,
        lakemask=land.lakemask,
        mavail=land.mavail,
        roughness_m=land.roughness_m,
        **boundary_leaves,
    )
    _debug("state.replace with initial fields complete")
    base = BaseState(
        pb=pb.astype(state.p_total.dtype),
        phb=phb.astype(state.ph_total.dtype),
        mub=mub.astype(state.mu_total.dtype),
        t0=theta_base.astype(state.theta.dtype),
        theta_base=theta_base.astype(state.theta.dtype),
    )
    metadata = {
        "run_id": run.run_id,
        "run_dir": str(run.path),
        "domain": domain,
        "run_start_label": run_start_label(run, domain),
        "grid": {
            "mass_shape": [int(grid.nz), int(grid.ny), int(grid.nx)],
            "wrf_staggered_extent": [int(grid.nz + 1), int(grid.ny + 1), int(grid.nx + 1)],
            "dx_m": float(grid.projection.dx_m),
            "dy_m": float(grid.projection.dy_m),
        },
        "boundary": boundary_meta,
        "prescribed_land": {
            "source_file": land.source.get("source_file"),
            "roughness_note": land.source.get("roughness_note"),
            "mavail_note": land.source.get("mavail_note"),
            "missing_optional_variables": land.source.get("missing_optional_variables", []),
        },
        "base_state": {
            "theta_base_k": P0_THETA_OFFSET_K,
            "pressure_split": "p_total=PB+P, p_perturbation=P",
            "geopotential_split": "ph_total=PHB+PH, ph_perturbation=PH",
            "mu_split": "mu_total=MUB+MU, mu_perturbation=MU",
        },
    }
    _debug("build_replay_case done")
    return ReplayCase(run, state, tendencies, grid, metrics, base, state.p_perturbation, metadata)


def build_l2_d02_replay_case(run_dir: str | Path, *, domain: str = "d02", parent_domain: str = "d01") -> ReplayCase:
    """Load an L2 d02 replay case with d01-derived one-way-nest boundary forcing."""

    case = build_replay_case(run_dir, domain=domain, boundary_domain=parent_domain)
    actual_yx = (int(case.grid.ny), int(case.grid.nx))
    if actual_yx != EXPECTED_L2_D02_MASS_SHAPE_YX:
        raise ValueError(
            f"L2 {domain} mass grid must be {EXPECTED_L2_D02_MASS_SHAPE_YX} for drop-in M7 d02 replay; "
            f"got {actual_yx} from {run_dir}"
        )
    metadata = dict(case.metadata)
    metadata["l2_replay_adapter"] = {
        "domain": domain,
        "parent_domain": parent_domain,
        "expected_mass_shape_yx": list(EXPECTED_L2_D02_MASS_SHAPE_YX),
        "actual_mass_shape_yx": list(actual_yx),
        "ic_source": "wrfout snapshot at t=0 plus wrfinput metrics/land state",
        "boundary_source": "parent d01 hourly wrfout interpolated to child d02 side strips",
    }
    return ReplayCase(
        case.run,
        case.state,
        case.tendencies,
        case.grid,
        case.metrics,
        case.base_state,
        case.previous_pressure,
        metadata,
    )


def _total_steps(replay_config: ReplayConfig) -> int:
    raw = float(replay_config.duration_s) / float(replay_config.dt_s)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(
            f"duration_s={replay_config.duration_s:g} is not an integer number of dt_s={replay_config.dt_s:g}"
        )
    return rounded


def replay_steps(replay_config: ReplayConfig) -> int:
    return _total_steps(replay_config)


def _finite_after_sanitize(state: State):
    leaves = jax.tree_util.tree_leaves(state)
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


def _step_diagnostics(state: State, stats) -> StepDiagnostics:
    return StepDiagnostics(
        finite_after_sanitize=_finite_after_sanitize(state),
        candidate_nonfinite_count=stats.nonfinite_count,
        candidate_clip_count=stats.clip_count,
        candidate_changed_count=stats.changed_count,
        w_abs_max_m_s=jnp.max(jnp.abs(state.w)),
        theta_min_k=jnp.min(state.theta),
        theta_max_k=jnp.max(state.theta),
    )


def _sanitize_replay_candidate(candidate: State, previous: State, base_state: BaseState) -> tuple[State, Any]:
    sanitized, stats = sanitize_state_with_stats(candidate, previous)
    sanitized = sanitized.replace(
        p_perturbation=sanitized.p_total - base_state.pb,
        ph_perturbation=sanitized.ph_total - base_state.phb,
        mu_perturbation=sanitized.mu_total - base_state.mub,
    )
    return sanitized, stats


def _acoustic_config(grid: GridSpec, n_acoustic: int, rayleigh_coefficient: float) -> AcousticConfig:
    return AcousticConfig(
        n_substeps=int(n_acoustic),
        dx_m=float(grid.projection.dx_m),
        dy_m=float(grid.projection.dy_m),
        non_hydrostatic=True,
        top_lid=True,
        mu_continuity=True,
        epssm=0.1,
        rayleigh=RayleighConfig(enabled=rayleigh_coefficient != 0.0, coefficient=float(rayleigh_coefficient)),
    )


def _rk3_stage(origin: State, stage_state: State, base_tendencies: Tendencies, grid: GridSpec, dt_stage: float) -> State:
    tendencies = compute_advection_tendencies(stage_state, base_tendencies, grid)
    return add_scaled_tendencies(origin, tendencies, dt_stage)


def _dycore_step_adr023(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
) -> tuple[State, Any]:
    acoustic = _acoustic_config(grid, replay_config.n_acoustic, replay_config.rayleigh_coefficient)
    s0 = apply_halo(state, halo_spec(grid))
    s1 = _rk3_stage(s0, s0, tendencies, grid, float(replay_config.dt_s) / 3.0)
    s1 = apply_halo(s1, halo_spec(grid))
    s2 = _rk3_stage(s0, s1, tendencies, grid, float(replay_config.dt_s) / 2.0)
    s2, previous_pressure = run_acoustic_scan(
        s2,
        previous_pressure,
        metrics,
        acoustic,
        float(replay_config.dt_s) / 2.0,
        base_state,
    )
    s2 = apply_halo(s2, halo_spec(grid))
    s3 = _rk3_stage(s0, s2, tendencies, grid, float(replay_config.dt_s))
    s3, previous_pressure = run_acoustic_scan(
        s3,
        previous_pressure,
        metrics,
        acoustic,
        float(replay_config.dt_s),
        base_state,
    )
    return apply_halo(s3, halo_spec(grid)), previous_pressure


def _candidate_timestep_adr023(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    global_step,
    replay_config: ReplayConfig,
    *,
    run_radiation: bool,
) -> tuple[State, Any]:
    next_state, next_previous_pressure = _dycore_step_adr023(
        state,
        previous_pressure,
        tendencies,
        grid,
        metrics,
        base_state,
        replay_config,
    )
    next_state = thompson_adapter(next_state, float(replay_config.dt_s))
    next_state = mynn_adapter(next_state, float(replay_config.dt_s), grid)
    next_state = surface_adapter(next_state, float(replay_config.dt_s))
    if bool(run_radiation):
        next_state = rrtmg_adapter(next_state, float(replay_config.dt_s), grid)
    lead_seconds = global_step.astype(jnp.float64) * float(replay_config.dt_s)
    next_state = apply_lateral_boundaries(next_state, lead_seconds, float(replay_config.dt_s), replay_config.boundary_config)
    return next_state, next_previous_pressure


def _empty_step_diagnostics() -> StepDiagnostics:
    return StepDiagnostics(
        finite_after_sanitize=jnp.zeros((0,), dtype=jnp.bool_),
        candidate_nonfinite_count=jnp.zeros((0,), dtype=jnp.int64),
        candidate_clip_count=jnp.zeros((0,), dtype=jnp.int64),
        candidate_changed_count=jnp.zeros((0,), dtype=jnp.int64),
        w_abs_max_m_s=jnp.zeros((0,), dtype=jnp.float64),
        theta_min_k=jnp.zeros((0,), dtype=jnp.float64),
        theta_max_k=jnp.zeros((0,), dtype=jnp.float64),
    )


def _stack_step_diagnostics(diagnostics: StepDiagnostics) -> StepDiagnostics:
    return StepDiagnostics(*(jnp.reshape(leaf, (1,)) for leaf in diagnostics))


def _concat_step_diagnostics(chunks: list[StepDiagnostics]) -> StepDiagnostics:
    if not chunks:
        return _empty_step_diagnostics()
    if len(chunks) == 1:
        return chunks[0]
    return StepDiagnostics(
        *(
            jnp.concatenate([getattr(chunk, name) for chunk in chunks], axis=0)
            for name in StepDiagnostics._fields
        )
    )


@partial(jax.jit, static_argnames=("grid", "replay_config", "start_step", "steps"))
def _run_replay_scan_no_radiation(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    start_step: int,
    steps: int,
) -> tuple[State, Any, StepDiagnostics]:
    """Run a no-radiation replay segment as one shape-stable device scan."""

    indices = jnp.arange(int(steps), dtype=jnp.int32) + jnp.asarray(int(start_step), dtype=jnp.int32) + 1

    def body(carry, global_step):
        carry_state, carry_previous_pressure = carry
        candidate, next_previous_pressure = _candidate_timestep_adr023(
            carry_state,
            carry_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            global_step,
            replay_config,
            run_radiation=False,
        )
        sanitized, stats = _sanitize_replay_candidate(candidate, carry_state, base_state)
        return (sanitized, next_previous_pressure), _step_diagnostics(sanitized, stats)

    (final_state, final_previous_pressure), diagnostics = jax.lax.scan(
        body,
        (state, previous_pressure),
        indices,
    )
    return final_state, final_previous_pressure, diagnostics


@partial(jax.jit, static_argnames=("grid", "replay_config", "run_radiation"))
def _run_replay_one_step(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    global_step,
    *,
    run_radiation: bool,
) -> tuple[State, Any, StepDiagnostics]:
    """Run one replay timestep with radiation as a static Python branch."""

    candidate, next_previous_pressure = _candidate_timestep_adr023(
        state,
        previous_pressure,
        tendencies,
        grid,
        metrics,
        base_state,
        global_step,
        replay_config,
        run_radiation=run_radiation,
    )
    sanitized, stats = _sanitize_replay_candidate(candidate, state, base_state)
    return sanitized, next_previous_pressure, _step_diagnostics(sanitized, stats)


def _run_no_radiation_segment(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    *,
    completed_steps: int,
    steps: int,
) -> tuple[State, Any, StepDiagnostics]:
    if int(steps) <= 0:
        return state, previous_pressure, _empty_step_diagnostics()
    _debug(f"no-radiation segment start completed_steps={completed_steps} steps={steps}")
    result = _run_replay_scan_no_radiation(
        state,
        previous_pressure,
        tendencies,
        grid,
        metrics,
        base_state,
        replay_config,
        int(completed_steps),
        int(steps),
    )
    _debug(f"no-radiation segment dispatched completed_steps={completed_steps} steps={steps}")
    return result


def _run_static_one_step(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    *,
    step_number: int,
    run_radiation: bool,
) -> tuple[State, Any, StepDiagnostics]:
    _debug(f"one-step segment start step={step_number} run_radiation={run_radiation}")
    state, previous_pressure, diagnostics = _run_replay_one_step(
        state,
        previous_pressure,
        tendencies,
        grid,
        metrics,
        base_state,
        replay_config,
        jnp.asarray(int(step_number), dtype=jnp.int32),
        run_radiation=bool(run_radiation),
    )
    _debug(f"one-step segment dispatched step={step_number} run_radiation={run_radiation}")
    return state, previous_pressure, _stack_step_diagnostics(diagnostics)


@partial(jax.jit, static_argnames=("grid", "replay_config", "start_step", "blocks", "cadence"))
def _run_replay_radiation_blocks(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    start_step: int,
    blocks: int,
    cadence: int,
) -> tuple[State, Any, StepDiagnostics]:
    """Run full radiation-cadence blocks without unrolling each block in Python."""

    def no_radiation_body(carry, global_step):
        carry_state, carry_previous_pressure = carry
        candidate, next_previous_pressure = _candidate_timestep_adr023(
            carry_state,
            carry_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            global_step,
            replay_config,
            run_radiation=False,
        )
        sanitized, stats = _sanitize_replay_candidate(candidate, carry_state, base_state)
        return (sanitized, next_previous_pressure), _step_diagnostics(sanitized, stats)

    def block(carry, block_index):
        carry_state, carry_previous_pressure = carry
        block_completed = jnp.asarray(int(start_step), dtype=jnp.int32) + block_index * int(cadence)
        no_rad_indices = jnp.arange(int(cadence) - 1, dtype=jnp.int32) + block_completed + 1
        (carry_state, carry_previous_pressure), no_rad_diags = jax.lax.scan(
            no_radiation_body,
            (carry_state, carry_previous_pressure),
            no_rad_indices,
        )
        candidate, next_previous_pressure = _candidate_timestep_adr023(
            carry_state,
            carry_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            block_completed + int(cadence),
            replay_config,
            run_radiation=True,
        )
        sanitized, stats = _sanitize_replay_candidate(candidate, carry_state, base_state)
        block_diags = _concat_step_diagnostics([no_rad_diags, _stack_step_diagnostics(_step_diagnostics(sanitized, stats))])
        return (sanitized, next_previous_pressure), block_diags

    (final_state, final_previous_pressure), block_diags = jax.lax.scan(
        block,
        (state, previous_pressure),
        jnp.arange(int(blocks), dtype=jnp.int32),
    )
    flat_diags = StepDiagnostics(
        *(leaf.reshape((int(blocks) * int(cadence),)) for leaf in block_diags)
    )
    return final_state, final_previous_pressure, flat_diags


@partial(jax.jit, static_argnames=("grid", "replay_config"))
def run_replay_scan(
    state: State,
    previous_pressure,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
) -> tuple[State, Any, StepDiagnostics]:
    """Run the coupled ADR-023 replay with static radiation-cadence segmentation."""

    total = _total_steps(replay_config)
    cadence = int(replay_config.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")
    if total <= 0:
        return state, previous_pressure, _empty_step_diagnostics()

    current = state
    current_previous_pressure = previous_pressure
    completed = 0
    remaining = int(total)
    chunks: list[StepDiagnostics] = []

    full_blocks = remaining // cadence
    if full_blocks > 0:
        _debug(f"radiation block scan start completed_steps={completed} blocks={full_blocks} cadence={cadence}")
        current, current_previous_pressure, diagnostics = _run_replay_radiation_blocks(
            current,
            current_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            replay_config,
            completed,
            full_blocks,
            cadence,
        )
        _debug(f"radiation block scan dispatched completed_steps={completed} blocks={full_blocks} cadence={cadence}")
        chunks.append(diagnostics)
        completed += full_blocks * cadence
        remaining -= full_blocks * cadence

    if remaining > 0:
        current, current_previous_pressure, diagnostics = _run_no_radiation_segment(
            current,
            current_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            replay_config,
            completed_steps=completed,
            steps=remaining - 1,
        )
        if remaining - 1 > 0:
            chunks.append(diagnostics)
            completed += remaining - 1
            remaining -= remaining - 1
        tail_radiation = bool(replay_config.final_radiation and completed + 1 == total)
        current, current_previous_pressure, diagnostics = _run_static_one_step(
            current,
            current_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            replay_config,
            step_number=completed + 1,
            run_radiation=tail_radiation,
        )
        chunks.append(diagnostics)

    return current, current_previous_pressure, _concat_step_diagnostics(chunks)


def _first_nonfinite_step(diagnostics: StepDiagnostics) -> int | None:
    finite = np.asarray(jax.device_get(diagnostics.finite_after_sanitize), dtype=bool)
    bad = ~finite
    indices = np.flatnonzero(bad)
    return int(indices[0] + 1) if indices.size else None


def diagnostics_summary(state: State, diagnostics: StepDiagnostics) -> dict[str, Any]:
    state_diag = state_diagnostics(state)
    nonfinite = np.asarray(jax.device_get(diagnostics.candidate_nonfinite_count), dtype=np.int64)
    clips = np.asarray(jax.device_get(diagnostics.candidate_clip_count), dtype=np.int64)
    changed = np.asarray(jax.device_get(diagnostics.candidate_changed_count), dtype=np.int64)
    candidate_bad = np.flatnonzero(nonfinite > 0)
    w_abs = np.asarray(jax.device_get(diagnostics.w_abs_max_m_s), dtype=np.float64)
    theta_min = np.asarray(jax.device_get(diagnostics.theta_min_k), dtype=np.float64)
    theta_max = np.asarray(jax.device_get(diagnostics.theta_max_k), dtype=np.float64)
    return {
        **state_diag,
        "first_nonfinite_step": _first_nonfinite_step(diagnostics),
        "first_candidate_nonfinite_step": int(candidate_bad[0] + 1) if candidate_bad.size else None,
        "candidate_nonfinite_steps": int(np.count_nonzero(nonfinite)),
        "candidate_nonfinite_count_total": int(nonfinite.sum()),
        "candidate_clip_count_total": int(clips.sum()),
        "candidate_changed_count_total": int(changed.sum()),
        "peak_w_abs_m_s": float(np.nanmax(w_abs)) if w_abs.size else state_diag["w_abs_max_m_s"],
        "theta_min_over_run_k": float(np.nanmin(theta_min)) if theta_min.size else state_diag["theta_min_k"],
        "theta_max_over_run_k": float(np.nanmax(theta_max)) if theta_max.size else state_diag["theta_max_k"],
    }


def _surface_fields(state: State) -> dict[str, Any]:
    surface = _surface_diagnostics_for_output(state)
    return {"T2": surface.t2, "U10": surface.u10, "V10": surface.v10}


def _reference_fields(run: Gen2Run, domain: str, lead_hours: float) -> tuple[dict[str, Any], Path, str]:
    history = run.history_files(domain)
    index = int(round(float(lead_hours)))
    if index >= len(history):
        raise FileNotFoundError(f"{run.path} has no wrfout_{domain} at lead {lead_hours:g}h")
    fields = {
        "T2": _load(run, domain, "T2", index),
        "U10": _load(run, domain, "U10", index),
        "V10": _load(run, domain, "V10", index),
        "w_k20": _load(run, domain, "W", index)[20, :, :],
        "theta_k20": _load(run, domain, "T", index)[20, :, :] + P0_THETA_OFFSET_K,
    }
    times = run.time_axis(domain)
    if index < len(times):
        valid_time = times[index].isoformat()
    else:
        valid_time = normalize_valid_time(history[index].name[-19:]).isoformat()
    return fields, history[index], valid_time


def forecast_comparison(state: State, run: Gen2Run, *, domain: str = "d02", lead_hours: float = 1.0) -> dict[str, Any]:
    surface = _surface_fields(state)
    forecast = {
        "T2": surface["T2"],
        "U10": surface["U10"],
        "V10": surface["V10"],
        "w_k20": state.w[20, :, :],
        "theta_k20": state.theta[20, :, :],
    }
    reference, source_path, valid_time = _reference_fields(run, domain, lead_hours)
    units = {"T2": "K", "U10": "m s-1", "V10": "m s-1", "w_k20": "m s-1", "theta_k20": "K"}
    rmse: dict[str, Any] = {}
    spatial_mean_drift: dict[str, Any] = {}
    max_abs_error: dict[str, Any] = {}
    shapes: dict[str, Any] = {}
    for name, predicted in forecast.items():
        ref = reference[name]
        error = predicted.astype(jnp.float64) - ref.astype(jnp.float64)
        rmse[name] = {"value": float(np.asarray(jnp.sqrt(jnp.mean(error * error)))), "units": units[name]}
        max_abs_error[name] = {"value": float(np.asarray(jnp.max(jnp.abs(error)))), "units": units[name]}
        shapes[name] = {"forecast": list(predicted.shape), "reference": list(ref.shape)}
        if name in {"T2", "U10", "V10"}:
            spatial_mean_drift[name] = {"value": float(np.asarray(jnp.mean(error))), "units": units[name]}
    return {
        "lead_hours": float(lead_hours),
        "valid_time_utc": valid_time,
        "gen2_reference_path": str(source_path),
        "rmse": rmse,
        "spatial_mean_drift": spatial_mean_drift,
        "max_abs_error": max_abs_error,
        "shapes": shapes,
    }


def static_transfer_audit(case: ReplayCase, replay_config: ReplayConfig) -> dict[str, Any]:
    _debug("static_transfer_audit start")
    jaxpr_text = str(
        jax.make_jaxpr(run_replay_scan, static_argnums=(3, 6))(
            case.state,
            case.previous_pressure,
            case.tendencies,
            case.grid,
            case.metrics,
            case.base_state,
            replay_config,
        )
    ).lower()
    forbidden = ("host_callback", "io_callback", "pure_callback")
    _debug(f"static_transfer_audit done jaxpr_bytes={len(jaxpr_text.encode('utf-8'))}")
    return {
        "method": "JAXPR callback scan for the exact ADR-023 replay scan",
        "host_callback_free": all(token not in jaxpr_text for token in forbidden),
        "forbidden_tokens": list(forbidden),
        "jaxpr_bytes": len(jaxpr_text.encode("utf-8")),
    }


def trace_transfer_audit(case: ReplayCase, replay_config: ReplayConfig, trace_dir: str | Path) -> dict[str, Any]:
    trace_path = Path(trace_dir)
    _debug(f"trace_transfer_audit warmup start trace_dir={trace_path}")
    block_until_ready(
        run_replay_scan(
            case.state,
            case.previous_pressure,
            case.tendencies,
            case.grid,
            case.metrics,
            case.base_state,
            replay_config,
        )
    )
    _debug("trace_transfer_audit warmup done")
    if trace_path.exists():
        shutil.rmtree(trace_path)
    trace_path.mkdir(parents=True, exist_ok=True)
    try:
        _debug("trace_transfer_audit profiled run start")
        with jax.profiler.trace(str(trace_path), create_perfetto_link=False):
            result = run_replay_scan(
                case.state,
                case.previous_pressure,
                case.tendencies,
                case.grid,
                case.metrics,
                case.base_state,
                replay_config,
            )
            block_until_ready(result)
    except TypeError:
        _debug("trace_transfer_audit retrying profiler.trace without create_perfetto_link")
        with jax.profiler.trace(str(trace_path)):
            result = run_replay_scan(
                case.state,
                case.previous_pressure,
                case.tendencies,
                case.grid,
                case.metrics,
                case.base_state,
                replay_config,
            )
            block_until_ready(result)
    _debug("trace_transfer_audit profiled run done")
    h2d, d2h, files = _count_replay_transfer_bytes(trace_path)
    _debug(f"trace_transfer_audit done h2d={h2d} d2h={d2h} files={len(files)}")
    return {
        "method": "jax.profiler.trace on warmed replay scan",
        "host_to_device_bytes_post_init": int(h2d),
        "device_to_host_bytes_post_init": int(d2h),
        "post_init_total_bytes": int(h2d + d2h),
        "trace_dir": str(trace_path),
        "trace_transfer_event_files": files,
    }


def peak_gpu_memory() -> dict[str, Any]:
    for device in jax.devices():
        if device.platform != "gpu" or not hasattr(device, "memory_stats"):
            continue
        stats = device.memory_stats() or {}
        return {
            "device": str(device),
            "bytes_in_use": stats.get("bytes_in_use"),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use") or stats.get("peak_bytes_reserved"),
            "raw_keys": sorted(stats.keys()),
        }
    return {"device": visible_gpu_name(), "bytes_in_use": None, "peak_bytes_in_use": None, "raw_keys": []}


def invoked_schemes(replay_config: ReplayConfig) -> list[dict[str, str]]:
    return [
        {
            "component": "dycore_large_step",
            "name": "WRF-shaped RK3 advection stages",
            "implementation": "gpuwrf.dynamics.advection.compute_advection_tendencies + add_scaled_tendencies",
        },
        {
            "component": "horizontal_pgf",
            "name": "c2-A2 WRF-shaped horizontal pressure-gradient force",
            "implementation": "gpuwrf.dynamics.acoustic_wrf.horizontal_pressure_gradient via run_acoustic_scan",
        },
        {
            "component": "vertical_acoustic",
            "name": "ADR-023 conservative tridiagonal column solver",
            "implementation": "gpuwrf.dynamics.acoustic_wrf.run_acoustic_scan / vertical_acoustic_update",
        },
        {
            "component": "mass_continuity",
            "name": "c2-A2 mu_continuity",
            "implementation": "gpuwrf.dynamics.acoustic_wrf.mu_continuity_tendency, AcousticConfig(mu_continuity=True)",
        },
        {
            "component": "microphysics",
            "name": "M5 Thompson column adapter",
            "implementation": "gpuwrf.coupling.physics_couplers.thompson_adapter",
        },
        {
            "component": "pbl",
            "name": "M5 MYNN PBL column adapter",
            "implementation": "gpuwrf.coupling.physics_couplers.mynn_adapter",
        },
        {
            "component": "surface",
            "name": "M6-S3 MM5 sfclay surface layer with prescribed Noah-MP Option A land leaves",
            "implementation": "gpuwrf.coupling.physics_couplers.surface_adapter + gpuwrf.io.land_state.load_prescribed_land_state",
        },
        {
            "component": "radiation",
            "name": f"M5 RRTMG SW+LW at cadence {int(replay_config.radiation_cadence_steps)} steps plus final={bool(replay_config.final_radiation)}",
            "implementation": "gpuwrf.coupling.physics_couplers.rrtmg_adapter",
        },
        {
            "component": "lateral_boundary",
            "name": "Gen2 d02 hourly side-history boundary replay",
            "implementation": "gpuwrf.coupling.boundary_apply.apply_lateral_boundaries",
        },
        {
            "component": "surface_lower_boundary",
            "name": "Bounded prescribed Noah-MP state, non-prognostic Option A",
            "implementation": "gpuwrf.physics.noah_mp.PrescribedNoahMPState",
        },
    ]


def write_output_fields(path: str | Path, state: State, comparison: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    surface = _surface_fields(state)
    np.savez(
        target,
        T2=np.asarray(jax.device_get(surface["T2"]), dtype=np.float32),
        U10=np.asarray(jax.device_get(surface["U10"]), dtype=np.float32),
        V10=np.asarray(jax.device_get(surface["V10"]), dtype=np.float32),
        w_k20=np.asarray(jax.device_get(state.w[20, :, :]), dtype=np.float32),
        theta_k20=np.asarray(jax.device_get(state.theta[20, :, :]), dtype=np.float32),
        gen2_reference_path=np.asarray(comparison["gen2_reference_path"]),
    )
    return target


def run_replay_proof(
    *,
    run_dir: str | Path = DEFAULT_REPLAY_RUN_DIR,
    output_fields_path: str | Path | None = DEFAULT_OUTPUT_FIELD_PATH,
    replay_config: ReplayConfig = ReplayConfig(),
    domain: str = "d02",
    trace_dir: str | Path | None = None,
    include_trace_audit: bool = True,
    include_static_audit: bool = True,
) -> dict[str, Any]:
    _debug(
        "run_replay_proof start "
        f"run_dir={run_dir} duration_s={replay_config.duration_s:g} dt_s={replay_config.dt_s:g} "
        f"n_acoustic={replay_config.n_acoustic}"
    )
    case = build_replay_case(run_dir, domain=domain)
    _debug("initial block_until_ready start")
    block_until_ready((case.state, case.previous_pressure, case.tendencies, case.metrics, case.base_state))
    _debug("initial block_until_ready done")
    start = time.perf_counter()
    _debug("run_replay_scan dispatch start")
    final_state, final_previous_pressure, step_diags = run_replay_scan(
        case.state,
        case.previous_pressure,
        case.tendencies,
        case.grid,
        case.metrics,
        case.base_state,
        replay_config,
    )
    _debug("run_replay_scan dispatch returned; block_until_ready start")
    block_until_ready((final_state, final_previous_pressure, step_diags))
    wall_s = time.perf_counter() - start
    _debug(f"run_replay_scan block_until_ready done wall_s={wall_s:.3f}")
    _debug("forecast_comparison start")
    comparison = forecast_comparison(final_state, case.run, domain=domain, lead_hours=float(replay_config.duration_s) / 3600.0)
    _debug("forecast_comparison done")
    _debug("diagnostics_summary start")
    diag = diagnostics_summary(final_state, step_diags)
    _debug("diagnostics_summary done")
    output_path = None
    if output_fields_path is not None:
        _debug(f"write_output_fields start path={output_fields_path}")
        output_path = write_output_fields(output_fields_path, final_state, comparison)
        _debug("write_output_fields done")
    static_audit = (
        static_transfer_audit(case, replay_config)
        if include_static_audit
        else {
            "method": "not run for this caller",
            "host_callback_free": True,
            "forbidden_tokens": ["host_callback", "io_callback", "pure_callback"],
            "jaxpr_bytes": None,
        }
    )
    _debug("static audit branch complete")
    trace = (
        trace_transfer_audit(
            case,
            replay_config,
            trace_dir
            or (DEFAULT_TRACE_ROOT / f"trace_m6x_d02_replay_{int(round(float(replay_config.duration_s)))}s"),
        )
        if include_trace_audit
        else {
            "method": "not run for this caller",
            "host_to_device_bytes_post_init": 0,
            "device_to_host_bytes_post_init": 0,
            "post_init_total_bytes": 0,
            "trace_dir": None,
            "trace_transfer_event_files": [],
        }
    )
    _debug("trace audit branch complete")

    payload = {
        "status": "PASS"
        if diag["first_nonfinite_step"] is None
        and static_audit["host_callback_free"]
        and trace["post_init_total_bytes"] == 0
        else "FAIL",
        "objective": "M6.x ADR-023 F6 rung 4: 1h Gen2 d02 boundary replay",
        "run": case.metadata,
        "duration_s": float(replay_config.duration_s),
        "dt_s": float(replay_config.dt_s),
        "steps": int(replay_steps(replay_config)),
        "n_acoustic": int(replay_config.n_acoustic),
        "wall_time_s": float(wall_s),
        "forecast_throughput_x_realtime": float(replay_config.duration_s) / wall_s if wall_s > 0.0 else None,
        "first_nonfinite_step": diag["first_nonfinite_step"],
        "diagnostics": diag,
        "comparison": comparison,
        "transfer_audit": {
            "static": static_audit,
            "trace": trace,
            "host_device_transfer_bytes_post_init": int(trace["post_init_total_bytes"]),
        },
        "peak_gpu_memory": peak_gpu_memory(),
        "invoked_schemes": invoked_schemes(replay_config),
        "output_fields_npz": str(output_path) if output_path is not None else None,
        "proof_notes": [
            "RMSE values are informational for this sprint; no threshold is applied.",
            "Boundary forcing is built from real Gen2 d02 hourly side histories for the same run used as truth.",
            "Sanitize statistics are reported separately; first_nonfinite_step is based on post-guard State leaves, and first_candidate_nonfinite_step records pre-guard candidate failures.",
        ],
    }
    return payload


__all__ = [
    "DEFAULT_OUTPUT_FIELD_PATH",
    "DEFAULT_REPLAY_RUN_DIR",
    "ReplayCase",
    "ReplayConfig",
    "build_l2_d02_replay_case",
    "build_replay_case",
    "diagnostics_summary",
    "forecast_comparison",
    "invoked_schemes",
    "load_history_boundary_leaves",
    "load_nested_parent_boundary_leaves",
    "replay_steps",
    "run_replay_proof",
    "run_replay_scan",
    "static_transfer_audit",
    "trace_transfer_audit",
]
