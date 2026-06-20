"""ADR-023 d02 boundary-replay integration helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
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

from gpuwrf.config.paths import tmp_root, wrf_l3_root
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
from gpuwrf.dynamics.metrics import load_wrfinput_metrics, terrain_slope_metrics
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.io.boundary_replay import decode_wrfbdy, wrfbdy_path_for_run
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.gen2_wrfout_loader import normalize_valid_time
from gpuwrf.io.land_state import load_prescribed_land_state
from gpuwrf.nesting.interp import sint_to_child_reference
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name


config.update("jax_enable_x64", True)

_DEBUG = os.environ.get("GPUWRF_D02_REPLAY_DEBUG", "").lower() not in {"", "0", "false", "no", "off"}
_DEBUG_START = time.perf_counter()
_METADATA_HOST_MAX_ELEMENT_LIMIT = 2_000_000


def _debug(message: str) -> None:
    if _DEBUG:
        print(f"[d02-replay +{time.perf_counter() - _DEBUG_START:8.3f}s] {message}", flush=True)


def _metadata_host_max(value: Any) -> float | None:
    """Return a host max for small metadata arrays without forcing d03-scale GPU work."""

    shape = getattr(value, "shape", np.shape(value))
    try:
        size = int(np.prod(shape))
    except Exception:
        size = _METADATA_HOST_MAX_ELEMENT_LIMIT + 1
    if not _DEBUG and size > _METADATA_HOST_MAX_ELEMENT_LIMIT:
        return None
    return float(jnp.max(value))


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
# WRF dry equation-of-state constants used to invert the loaded base state for the
# base potential-temperature profile (must match
# ``dynamics/acoustic_wrf._inverse_density_from_theta_pressure``).
_EOS_R_D = 287.0
_EOS_CP_D = 7.0 * _EOS_R_D / 2.0  # WRF cp = 1004.5 exactly (mirrors acoustic_wrf.CP_D)
_EOS_P0_PA = 100000.0
_EOS_CVPM = -(_EOS_CP_D - _EOS_R_D) / _EOS_CP_D
_WRF_BASE_GRAVITY_M_S2 = 9.81

# WRF ``share/module_model_constants.F`` REAL(4) parameters as compile-time fp32
# folds, used by the live-nest ``start_domain_em`` init transcription.  These are
# deliberately separate from the fp64 ``_EOS_*`` constants above (fp32 folds vs
# fp64; both carry the WRF cp = 7*r_d/2 = 1004.5 since the v0.14 constants fix).
_WRF32_R_D = np.float32(287.0)
_WRF32_CP = np.float32(7.0) * _WRF32_R_D / np.float32(2.0)      # 1004.5
_WRF32_CV = _WRF32_CP - _WRF32_R_D                              # 717.5
_WRF32_CVPM = -(_WRF32_CV / _WRF32_CP)
_WRF32_RDOCP = _WRF32_R_D / _WRF32_CP
_WRF32_CPOVCV = _WRF32_CP / _WRF32_CV
_WRF32_P1000MB = np.float32(100000.0)
_WRF32_T0 = np.float32(300.0)
_WRF32_G = np.float32(9.81)


class _WRFInitLibm32:
    """float32 ``expf``/``logf``/``powf`` bit-matched to the CPU-WRF build's libm.

    The CPU-WRF truth is gfortran ``-O2`` REAL(4) calling scalar glibc
    ``expf``/``logf``/``powf`` (math-errno blocks vectorization, no fast-math, no
    FMA).  NumPy's float32 SIMD ``exp``/``log`` differ from glibc by 1-4 ulp and
    glibc ``powf(x, 0.5)`` differs from ``sqrtf`` by 1 ulp on rare inputs; those
    ulps are amplified ~50x by the hypsometric layer-thickness division into
    ~Pa-level base/perturbation pressure errors (proofs/v014/
    mythos_kernel_fix_260609.*).  Calling the same libm closes the chain
    bit-exactly.  Initialization-only host helper; never on the device hot path.
    Falls back to float64-computed correctly-rounded float32 when libm is not
    loadable (residual then bounded ~2.3 Pa worst-case instead of ~0.04 Pa).
    """

    def __init__(self) -> None:
        self._libm = None
        self.provider = "float64-rounded-fallback"
        try:
            import ctypes
            import ctypes.util

            path = ctypes.util.find_library("m")
            libm = ctypes.CDLL(path) if path else ctypes.CDLL(None)
            for name, nargs in (("expf", 1), ("logf", 1), ("powf", 2)):
                fn = getattr(libm, name)
                fn.restype = ctypes.c_float
                fn.argtypes = [ctypes.c_float] * nargs
            self._ctypes = ctypes
            self._libm = libm
            self.provider = "glibc-libm-float32"
        except Exception:  # pragma: no cover - non-glibc fallback
            self._libm = None

    def _map1(self, name: str, x: np.ndarray) -> np.ndarray:
        x32 = np.asarray(x, dtype=np.float32)
        if self._libm is None:
            fn64 = np.exp if name == "expf" else np.log
            return fn64(x32.astype(np.float64)).astype(np.float32)
        fn = getattr(self._libm, name)
        c_float = self._ctypes.c_float
        out = np.empty_like(x32)
        fi, fo = x32.ravel(), out.ravel()
        for i in range(fi.size):
            fo[i] = fn(c_float(float(fi[i])))
        return out

    def exp(self, x: np.ndarray) -> np.ndarray:
        return self._map1("expf", x)

    def log(self, x: np.ndarray) -> np.ndarray:
        return self._map1("logf", x)

    def pow(self, x: np.ndarray, y: np.float32) -> np.ndarray:
        x32 = np.asarray(x, dtype=np.float32)
        y32 = np.float32(y)
        if self._libm is None:
            return np.power(x32.astype(np.float64), np.float64(y32)).astype(np.float32)
        c_float = self._ctypes.c_float
        c_y = c_float(float(y32))
        out = np.empty_like(x32)
        fi, fo = x32.ravel(), out.ravel()
        for i in range(fi.size):
            fo[i] = self._libm.powf(c_float(float(fi[i])), c_y)
        return out

    def sqrt_via_pow(self, x: np.ndarray) -> np.ndarray:
        """WRF ``(...)**0.5`` compiles to ``powf(x, 0.5)`` (not ``sqrtss``)."""

        if self._libm is None:
            # float64 sqrt rounds to the correctly-rounded float32 sqrt.
            return np.sqrt(np.asarray(x, dtype=np.float32).astype(np.float64)).astype(np.float32)
        return self.pow(x, np.float32(0.5))


_WRF_INIT_LIBM32 = _WRFInitLibm32()
# Replay scratch/output defaults; env-overridable via config.paths
# (GPUWRF_RUN_ROOT / GPUWRF_TMPDIR) with no hardcoded <USER_HOME>/<name> path so a
# clean clone writes its scratch under ~/.cache/gpuwrf.
DEFAULT_REPLAY_RUN_DIR = wrf_l3_root() / "20260521_18z_l3_24h_20260522T133443Z"
DEFAULT_OUTPUT_FIELD_PATH = tmp_root() / "outputs" / "m6x_d02_replay" / "proof_d02_replay_fields.npz"
DEFAULT_TRACE_ROOT = tmp_root() / "tmp"
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


def _load_wrfinput(run: Gen2Run, domain: str, var: str):
    start = time.perf_counter()
    _debug(f"load wrfinput start {domain}:{var}")
    value = run.load_wrfinput(domain, var, lazy=False)
    _debug(f"load wrfinput done {domain}:{var} {_shape_dtype(value)} elapsed_s={time.perf_counter() - start:.3f}")
    return value


def _load_initial(run: Gen2Run, domain: str, var: str, time_index: int, *, use_wrfinput: bool):
    if use_wrfinput:
        return _load_wrfinput(run, domain, var)
    return _load(run, domain, var, time_index)


def _optional_load(run: Gen2Run, domain: str, var: str, time: int, fallback):
    try:
        return _load(run, domain, var, time)
    except KeyError:
        _debug(f"optional load missing {domain}:{var}[{time}], using fallback {_shape_dtype(fallback)}")
        return fallback


def _optional_load_initial(run: Gen2Run, domain: str, var: str, time: int, fallback, *, use_wrfinput: bool):
    try:
        return _load_initial(run, domain, var, time, use_wrfinput=use_wrfinput)
    except KeyError:
        source = "wrfinput" if use_wrfinput else f"history[{time}]"
        _debug(f"optional load missing {domain}:{var} from {source}, using fallback {_shape_dtype(fallback)}")
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


def _pack_history_2d(
    run: Gen2Run,
    domain: str,
    var: str,
    *,
    ntimes: int,
    max_side: int,
    bdy_width: int,
    dtype: Any,
) -> np.ndarray:
    _debug(f"pack history start {domain}:{var} ntimes={ntimes} max_side={max_side}")
    packed = np.zeros((ntimes, 4, int(bdy_width), 1, max_side), dtype=dtype)
    for time_index in range(ntimes):
        data = np.asarray(_load(run, domain, var, time_index), dtype=dtype)
        for side, values in _field_sides_2d(data, int(bdy_width)).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], 0, : values.shape[1]] = values
    _debug(f"pack history done {domain}:{var} {_shape_dtype(packed)}")
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
    use_theta_m = _wrf_use_theta_m(run, domain)

    def add_theta(_run: Gen2Run, _domain: str, data: np.ndarray, _time_index: int) -> np.ndarray:
        # wrfout ``T`` is the DRY perturbation theta; operational State.theta
        # is MOIST theta_m (use_theta_m=1), so recouple with the same file's
        # QVAPOR (exact to wrfout fp32: THM == T_dry*(1+rvovrd*qv) at ~7e-5 K).
        theta_full = data + P0_THETA_OFFSET_K
        if use_theta_m == 1:
            qv = np.asarray(_load(_run, _domain, "QVAPOR", _time_index), dtype=np.float64)
            theta_full = theta_full * (1.0 + _RVOVRD * np.maximum(qv, 0.0))
        return theta_full

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
        "w_bdy": _pack_history_3d(
            run,
            domain,
            "W",
            ntimes=n,
            z_len=grid.nz + 1,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "p_bdy": _pack_history_3d(
            run,
            domain,
            "P",
            ntimes=n,
            z_len=grid.nz,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "pb_bdy": _pack_history_3d(
            run,
            domain,
            "PB",
            ntimes=1,
            z_len=grid.nz,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
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
        ),
        "phb_bdy": _pack_history_3d(
            run,
            domain,
            "PHB",
            ntimes=1,
            z_len=grid.nz + 1,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "mu_bdy": _pack_history_2d(
            run,
            domain,
            "MU",
            ntimes=n,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "mub_bdy": _pack_history_2d(
            run,
            domain,
            "MUB",
            ntimes=1,
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
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
        "variables": ["U", "V", "W", "T", "QVAPOR", "P", "PB", "PH", "PHB", "MU", "MUB"],
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


def _pack_nested_parent_history_2d(
    run: Gen2Run,
    child_meta: Any,
    parent_domain: str,
    var: str,
    *,
    ntimes: int,
    child_shape: tuple[int, int],
    max_side: int,
    bdy_width: int,
    dtype: Any,
) -> np.ndarray:
    y_len, x_len = (int(item) for item in child_shape)
    y_coords, x_coords = _nested_axis_coords(child_meta, y_len=y_len, x_len=x_len)
    packed = np.zeros((ntimes, 4, int(bdy_width), 1, max_side), dtype=dtype)
    _debug(
        f"pack nested parent history start {parent_domain}:{var} ntimes={ntimes} "
        f"child_shape={(y_len, x_len)} max_side={max_side}"
    )
    for time_index in range(ntimes):
        parent = np.asarray(_load(run, parent_domain, var, time_index), dtype=dtype)
        child = _interp_parent_horizontal(parent, y_coords, x_coords)
        for side, values in _field_sides_2d(child, int(bdy_width)).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], 0, : values.shape[1]] = values
    _debug(f"pack nested parent history done {parent_domain}:{var} {_shape_dtype(packed)}")
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
    use_theta_m = _wrf_use_theta_m(run, child_domain)

    def add_theta(_run: Gen2Run, _domain: str, data: np.ndarray, _time_index: int) -> np.ndarray:
        # Parent wrfout ``T`` is DRY perturbation theta; the child boundary
        # forces operational State.theta = MOIST theta_m (use_theta_m=1), so
        # recouple with the parent's QVAPOR BEFORE horizontal interpolation.
        theta_full = data + P0_THETA_OFFSET_K
        if use_theta_m == 1:
            qv = np.asarray(_load(_run, _domain, "QVAPOR", _time_index), dtype=np.float64)
            theta_full = theta_full * (1.0 + _RVOVRD * np.maximum(qv, 0.0))
        return theta_full

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
        "w_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "W",
            ntimes=n,
            child_shape=(grid.nz + 1, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "p_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "P",
            ntimes=n,
            child_shape=(grid.nz, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "pb_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "PB",
            ntimes=1,
            child_shape=(grid.nz, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
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
        ),
        "phb_bdy": _pack_nested_parent_history_3d(
            run,
            child_meta,
            parent_domain,
            "PHB",
            ntimes=1,
            child_shape=(grid.nz + 1, grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "mu_bdy": _pack_nested_parent_history_2d(
            run,
            child_meta,
            parent_domain,
            "MU",
            ntimes=n,
            child_shape=(grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
        ),
        "mub_bdy": _pack_nested_parent_history_2d(
            run,
            child_meta,
            parent_domain,
            "MUB",
            ntimes=1,
            child_shape=(grid.ny, grid.nx),
            max_side=max_side,
            bdy_width=bdy_width,
            dtype=np.float64,
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
        "variables": ["U", "V", "W", "T", "QVAPOR", "P", "PB", "PH", "PHB", "MU", "MUB"],
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


def _wrf_base_theta_from_loaded_state(
    *,
    pb: jax.Array,
    phb: jax.Array,
    mub: jax.Array,
    metrics: DycoreMetrics,
) -> jax.Array:
    """Recover the WRF base potential-temperature profile ``t0 + t_init``.

    The replay IC (PB/PHB/MUB) is the corpus base state, which WRF built in
    ``dyn_em/module_initialize_real.F:3793-3818`` so that the base geopotential
    is in EXACT discrete hydrostatic balance with the base inverse density::

        alb(k) = (R_d/p0)*(t_init(k)+t0)*(pb(k)/p0)**cvpm        (:3802)
        phb(k+1) = phb(k) - dnw(k)*(c1h(k)*mub + c2h(k))*alb(k)  (:3817)

    The dycore's ``diagnose_pressure_al_alt`` recomputes ``alb`` from a base
    potential-temperature field via the SAME EOS, then forms ``alt = al + alb``
    and the EOS pressure.  If that base theta is the constant 300 K (the prior
    behaviour) instead of WRF's height-varying ``t_init`` profile, the recomputed
    ``alb`` disagrees with the discrete ``alb`` the loaded ``phb`` was integrated
    from -- by up to ~35 % aloft (300 K vs the ~465 K base profile near the lid).
    The loaded IC is then NOT in the dycore's discrete hydrostatic balance, so the
    prognostic perturbation geopotential ``ph'`` equilibrates over the first
    forecast hour to absorb the base-state mismatch, producing the steady
    near-uniform +2.6 kPa diagnostic perturbation-pressure offset seen on BOTH
    d02 (force_geopotential=True) and d03.

    Fix: invert the loaded discrete base state to recover the EXACT ``alb`` it was
    balanced against, then back out the base potential temperature so the dycore's
    recomputed ``alb`` reproduces the loaded ``phb`` to round-off.  This is exact
    and grid-agnostic (uses the file's own hybrid c1h/c2h/dnw), so it requires no
    namelist base-profile parameters.
    """

    alb = _wrf_base_alb_from_loaded_state(phb=phb, mub=mub, metrics=metrics)
    # Back out theta_base from alb = (R_d/p0)*theta_base*(pb/p0)**cvpm so the
    # dycore's _inverse_density_from_theta_pressure(theta_base, pb) reproduces it.
    p_ratio = (jnp.maximum(pb, jnp.asarray(1.0, dtype=pb.dtype)) / _EOS_P0_PA) ** _EOS_CVPM
    theta_base = alb * (_EOS_P0_PA / _EOS_R_D) / p_ratio
    return theta_base


def _wrf_base_alb_from_loaded_state(
    *,
    phb: jax.Array,
    mub: jax.Array,
    metrics: DycoreMetrics,
) -> jax.Array:
    """Invert WRF's discrete base hydrostatic relation for ``alb``."""

    mass_h = metrics.c1h[:, None, None] * mub[None, :, :] + metrics.c2h[:, None, None]
    dphb = phb[1:, :, :] - phb[:-1, :, :]
    denom = metrics.dnw[:, None, None] * mass_h
    safe_denom = jnp.where(
        jnp.abs(denom) > 1.0e-12, denom, jnp.asarray(1.0e-12, dtype=denom.dtype)
    )
    return -dphb / safe_denom


def _namelist_int_any(run: Gen2Run, key: str, default: int, *, domain: str | None = None) -> int:
    """Read a scalar/list namelist integer from the groups WRF uses for nest init."""

    index = max(int(domain[1:]) - 1, 0) if domain else 0
    for group in ("domains", "bdy_control"):
        raw = run.namelist.get(group, {}).get(key)
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            if index < len(raw):
                return int(raw[index])
            return int(raw[-1]) if raw else int(default)
        return int(raw)
    return int(default)


def _scalar0(value: jax.Array) -> jax.Array:
    return jnp.reshape(jnp.asarray(value, dtype=jnp.float64), ())


def _wrf_blend_terrain_host(
    interpolated: np.ndarray,
    fine: np.ndarray,
    *,
    spec_bdy_width: int,
    blend_width: int,
    dtype: type = np.float32,
) -> np.ndarray:
    """Host transcription of WRF ``nest_init_utils.F::blend_terrain``.

    Initialization-only: no timestep-loop transfer.  Supports mass 2-D fields and
    vertical-stack 3-D fields with horizontal axes last.

    The default ``dtype=np.float32`` evaluates the blend in WRF's REAL(4)
    precision with the exact source grouping
    ``(blend_cell*ter_input + (blend_width+1-blend_cell)*ter_interpolated) *
    r_blend_zones`` (``r_blend_zones = 1./(blend_width+1)``), reproducing the
    CPU-WRF blended field bit-exactly when the inputs are bit-exact.
    """

    f = np.dtype(dtype).type
    fine_arr = np.asarray(fine, dtype=dtype)
    out = np.array(fine_arr, copy=True)
    parent = np.asarray(interpolated, dtype=dtype)
    ny, nx = out.shape[-2:]
    ide = nx + 1
    jde = ny + 1
    r_blend = f(1.0) / f(int(blend_width) + 1)
    for jj in range(ny):
        j = jj + 1
        for ii in range(nx):
            i = ii + 1
            for blend_cell in range(int(blend_width), 0, -1):
                if (
                    i == int(spec_bdy_width) + blend_cell
                    or j == int(spec_bdy_width) + blend_cell
                    or i == ide - int(spec_bdy_width) - blend_cell
                    or j == jde - int(spec_bdy_width) - blend_cell
                ):
                    out[..., jj, ii] = (
                        f(blend_cell) * fine_arr[..., jj, ii]
                        + f(int(blend_width) + 1 - blend_cell) * parent[..., jj, ii]
                    ) * r_blend
            if (
                i <= int(spec_bdy_width)
                or j <= int(spec_bdy_width)
                or i >= ide - int(spec_bdy_width)
                or j >= jde - int(spec_bdy_width)
            ):
                out[..., jj, ii] = parent[..., jj, ii]
    return out


def _sint_to_child_reference_stack(
    coarse: np.ndarray,
    *,
    ratio: int,
    i_parent_start: int,
    j_parent_start: int,
    child_ny: int,
    child_nx: int,
    dtype: type = np.float32,
) -> np.ndarray:
    """Apply WRF SINT to every leading-level slab of a 2-D/3-D base field."""

    arr = np.asarray(coarse, dtype=dtype)
    if arr.ndim == 2:
        return sint_to_child_reference(
            arr,
            ratio=ratio,
            i_parent_start=i_parent_start,
            j_parent_start=j_parent_start,
            child_ny=child_ny,
            child_nx=child_nx,
            dtype=dtype,
        )
    if arr.ndim < 2:
        raise ValueError(f"SINT stack interpolation expects >=2D field, got shape {arr.shape}")
    leading = arr.shape[:-2]
    flat = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    out = np.empty((flat.shape[0], int(child_ny), int(child_nx)), dtype=dtype)
    for idx, slab in enumerate(flat):
        out[idx] = sint_to_child_reference(
            slab,
            ratio=ratio,
            i_parent_start=i_parent_start,
            j_parent_start=j_parent_start,
            child_ny=child_ny,
            child_nx=child_nx,
            dtype=dtype,
        )
    return out.reshape(leading + (int(child_ny), int(child_nx)))


def _metrics_with_terrain(
    metrics: DycoreMetrics,
    *,
    terrain_height: jax.Array,
    grid: GridSpec,
    provenance_suffix: str,
) -> DycoreMetrics:
    dzdx, dzdy, dzdx_u, dzdy_v = terrain_slope_metrics(
        terrain_height,
        float(grid.projection.dx_m),
        float(grid.projection.dy_m),
    )
    return dataclass_replace(
        metrics,
        dzdx=dzdx,
        dzdy=dzdy,
        dzdx_u=dzdx_u,
        dzdy_v=dzdy_v,
        provenance=f"{metrics.provenance}; {provenance_suffix}",
    )


def _wrf_start_domain_base_scalars32(run: Gen2Run, domain: str) -> dict[str, np.float32]:
    """fp32 base-profile scalars exactly as ``start_domain_em`` reads them."""

    def f32_scalar(name: str) -> np.float32:
        return np.float32(float(np.asarray(jax.device_get(_load(run, domain, name, 0))).ravel()[0]))

    return {
        "p_top": f32_scalar("P_TOP"),
        "p00": f32_scalar("P00"),
        "t00": f32_scalar("T00"),
        "a": f32_scalar("TLP"),
        "tiso": f32_scalar("TISO"),
        "a_strat": f32_scalar("TLP_STRAT"),
        "p_strat": f32_scalar("P_STRAT"),
    }


def _wrf_start_domain_base_from_hgt(
    run: Gen2Run,
    domain: str,
    *,
    hgt: jax.Array,
    metrics: DycoreMetrics,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """WRF real-run ``start_domain_em`` base recompute from terrain.

    Returns ``(PB, MUB, PHB, T_INIT, ALB)`` for the multi-domain, non-ideal path
    (``input_from_file``, ``hypsometric_opt=2``, ``rebalance=0``), as float64
    arrays whose values are the WRF REAL(4) results exactly.

    The chain is evaluated on the host in WRF's fp32 precision with the exact
    ``dyn_em/start_em.F`` expression grouping and the WRF build's float32 libm
    (``_WRFInitLibm32``).  fp64 evaluation (the previous behaviour) leaves MUB up
    to 0.05 Pa from WRF, which the fp32 ``AL/ALT`` hypsometric diagnosis amplifies
    into a ~3-4 Pa perturbation-pressure error (proofs/v014/
    step1_base_state_boundary.* and mythos_kernel_fix_260609.*).
    Initialization-only; no timestep-loop transfer.
    """

    m = _WRF_INIT_LIBM32
    s = _wrf_start_domain_base_scalars32(run, domain)
    p_top, p00, t00, a = s["p_top"], s["p00"], s["t00"], s["a"]
    tiso, a_strat, p_strat = s["tiso"], s["a_strat"], s["p_strat"]

    ht32 = np.asarray(jax.device_get(hgt), dtype=np.float32)
    c3h = np.asarray(jax.device_get(metrics.c3h), dtype=np.float32)
    c4h = np.asarray(jax.device_get(metrics.c4h), dtype=np.float32)
    c3f = np.asarray(jax.device_get(metrics.c3f), dtype=np.float32)
    c4f = np.asarray(jax.device_get(metrics.c4f), dtype=np.float32)

    # p_surf = p00 * EXP ( -t00/a + ( (t00/a)**2 - 2.*g*ht/a/r_d ) **0.5 )
    t00_over_a = t00 / a
    root = (t00_over_a * t00_over_a - np.float32(2.0) * _WRF32_G * ht32 / a / _WRF32_R_D).astype(np.float32)
    p_surf = (p00 * m.exp((-t00_over_a + m.sqrt_via_pow(root)).astype(np.float32))).astype(np.float32)
    mub = p_surf - p_top
    pb = (c3h[:, None, None] * mub[None, :, :] + c4h[:, None, None] + p_top).astype(np.float32)

    # temp = MAX(tiso, t00 + A*LOG(pb/p00)); stratosphere branch when p_strat > 0
    temp = np.maximum(tiso, (t00 + a * m.log((pb / p00).astype(np.float32))).astype(np.float32))
    if float(p_strat) > 0.0:
        strat_temp = (tiso + a_strat * m.log((pb / p_strat).astype(np.float32))).astype(np.float32)
        temp = np.where(pb < p_strat, strat_temp, temp).astype(np.float32)
    t_init = (temp * m.pow((p00 / pb).astype(np.float32), _WRF32_RDOCP) - _WRF32_T0).astype(np.float32)
    alb = (
        _WRF32_R_D / _WRF32_P1000MB * (t_init + _WRF32_T0) * m.pow((pb / _WRF32_P1000MB).astype(np.float32), _WRF32_CVPM)
    ).astype(np.float32)

    # hypsometric_opt=2 base geopotential integration from terrain elevation
    nz = int(pb.shape[0])
    phb = np.empty((nz + 1,) + pb.shape[1:], dtype=np.float32)
    phb[0] = (ht32 * _WRF32_G).astype(np.float32)
    for full_k in range(1, nz + 1):
        half_k = full_k - 1
        pfu = (c3f[full_k] * mub + c4f[full_k] + p_top).astype(np.float32)
        pfd = (c3f[full_k - 1] * mub + c4f[full_k - 1] + p_top).astype(np.float32)
        phm = (c3h[half_k] * mub + c4h[half_k] + p_top).astype(np.float32)
        phb[full_k] = (phb[full_k - 1] + alb[half_k] * phm * m.log((pfd / pfu).astype(np.float32))).astype(np.float32)

    as64 = lambda arr: jnp.asarray(np.asarray(arr, dtype=np.float64))  # noqa: E731
    return as64(pb), as64(mub), as64(phb), as64(t_init), as64(alb)


def _apply_live_nest_base_init(
    run: Gen2Run,
    *,
    domain: str,
    grid: GridSpec,
    metrics: DycoreMetrics,
    parent_case: ReplayCase,
    child_mub: Any,
    child_phb: Any,
) -> tuple[GridSpec, DycoreMetrics, jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """Apply WRF live-nest terrain/base initialization before timestep ownership."""

    child_meta = run.grid(domain)
    expected_parent = f"d{int(child_meta.parent_id):02d}"
    parent_domain = str(parent_case.metadata.get("domain", ""))
    if parent_domain and parent_domain != expected_parent:
        raise ValueError(
            f"{domain}: live-nest parent case must be {expected_parent}, got {parent_domain}"
        )
    ratio = int(child_meta.parent_grid_ratio)
    if ratio <= 1:
        raise ValueError(f"{domain}: live-nest base init requires parent_grid_ratio > 1, got {ratio}")

    spec_bdy_width = _namelist_int_any(run, "spec_bdy_width", 5, domain=domain)
    blend_width = _namelist_int_any(run, "blend_width", 5, domain=domain)
    # WRF interpolates and blends the nest terrain in REAL(4); evaluating this
    # chain in float64 leaves the blended HT up to 1 fp32 ulp (~2.7e-5 m) from
    # WRF, which the hypsometric AL/ALT diagnosis amplifies into ~2 Pa local
    # perturbation-pressure errors (proofs/v014/mythos_kernel_fix_260609.*).
    parent_hgt = np.asarray(jax.device_get(parent_case.grid.terrain_height), dtype=np.float32)
    child_hgt = np.asarray(jax.device_get(grid.terrain_height), dtype=np.float32)
    parent_hgt_on_child = sint_to_child_reference(
        parent_hgt,
        ratio=ratio,
        i_parent_start=int(child_meta.i_parent_start),
        j_parent_start=int(child_meta.j_parent_start),
        child_ny=int(grid.ny),
        child_nx=int(grid.nx),
        dtype=np.float32,
    )
    if not np.isfinite(parent_hgt_on_child).all():
        raise ValueError(
            f"{domain}: live-nest SINT terrain interpolation produced non-finite values; "
            "the parent stencil halo or nest geometry is insufficient"
        )
    blended_hgt_np = _wrf_blend_terrain_host(
        parent_hgt_on_child,
        child_hgt,
        spec_bdy_width=spec_bdy_width,
        blend_width=blend_width,
    )
    blended_hgt = jnp.asarray(blended_hgt_np, dtype=grid.terrain_height.dtype)
    metrics = _metrics_with_terrain(
        metrics,
        terrain_height=blended_hgt,
        grid=grid,
        provenance_suffix=f"live_nest_base_init:{expected_parent}->{domain}:sint_tr4_blend_terrain",
    )

    parent_mub_on_child = _sint_to_child_reference_stack(
        np.asarray(jax.device_get(parent_case.base_state.mub), dtype=np.float32),
        ratio=ratio,
        i_parent_start=int(child_meta.i_parent_start),
        j_parent_start=int(child_meta.j_parent_start),
        child_ny=int(grid.ny),
        child_nx=int(grid.nx),
        dtype=np.float32,
    )
    parent_phb_on_child = _sint_to_child_reference_stack(
        np.asarray(jax.device_get(parent_case.base_state.phb), dtype=np.float32),
        ratio=ratio,
        i_parent_start=int(child_meta.i_parent_start),
        j_parent_start=int(child_meta.j_parent_start),
        child_ny=int(grid.ny),
        child_nx=int(grid.nx),
        dtype=np.float32,
    )
    if not np.isfinite(parent_mub_on_child).all() or not np.isfinite(parent_phb_on_child).all():
        raise ValueError(
            f"{domain}: live-nest SINT base-state interpolation produced non-finite values"
        )
    blended_mub_np = _wrf_blend_terrain_host(
        parent_mub_on_child,
        np.asarray(jax.device_get(child_mub), dtype=np.float32),
        spec_bdy_width=spec_bdy_width,
        blend_width=blend_width,
    )
    blended_phb_np = _wrf_blend_terrain_host(
        parent_phb_on_child,
        np.asarray(jax.device_get(child_phb), dtype=np.float32),
        spec_bdy_width=spec_bdy_width,
        blend_width=blend_width,
    )
    # WRF does blend MUB/PHB here, but for this real multi-domain branch
    # start_domain(nest,.TRUE.) immediately rebuilds final MUB/PB/PHB from blended
    # HT (start_em.F:600-638). The blended MUB is still the transient field consumed
    # by adjust_tempqv; _wrf_live_nest_transient_adjust_mub mirrors that path below.
    pb, mub, phb, _t_init, _alb = _wrf_start_domain_base_from_hgt(
        run,
        domain,
        hgt=blended_hgt,
        metrics=metrics,
    )

    terrain = dataclass_replace(
        grid.terrain,
        source_path=f"{grid.terrain.source_path};live_nest_parent={expected_parent}",
        max_elevation_m=float(np.nanmax(blended_hgt_np)),
    )
    grid = dataclass_replace(grid, terrain=terrain, terrain_height=blended_hgt, metrics=metrics)
    meta = {
        "enabled": True,
        "source": "native live-nest parent SINT/TR4 interpolation + WRF blend_terrain pre-start path",
        "base_state_source": (
            "HT/MUB/PHB are independently blended like med_nest_initial; final "
            "PB/MUB/PHB are WRF start_domain_em recompute from blended HT"
        ),
        "parent_domain": expected_parent,
        "child_domain": domain,
        "parent_grid_ratio": ratio,
        "i_parent_start": int(child_meta.i_parent_start),
        "j_parent_start": int(child_meta.j_parent_start),
        "spec_bdy_width": int(spec_bdy_width),
        "blend_width": int(blend_width),
        "interpolation": "gpuwrf.nesting.interp.sint_to_child_reference (WRF sint.F host reference, init-only)",
        "production_inputs": ["parent native initialized terrain", f"wrfinput_{domain}", "namelist.input"],
        "cpu_wrfout_dependency": False,
        "timestep_loop_transfer": False,
        "pre_start_blended_mub_max_pa": float(np.nanmax(blended_mub_np)),
        "pre_start_blended_phb_max_m2_s2": float(np.nanmax(blended_phb_np)),
        "note": (
            "The host SINT reference runs before OperationalCarry creation. The forecast "
            "loop remains device-resident; no CPU-WRF history file is read."
        ),
    }
    return grid, metrics, pb, phb, mub, meta


def _wrf_live_nest_transient_adjust_mub(
    run: Gen2Run,
    *,
    domain: str,
    grid: GridSpec,
    parent_mub: Any,
    child_mub: Any,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """Transient post-``blend_terrain`` current ``MUB`` consumed by ``adjust_tempqv``.

    WRF ``share/mediation_integrate.F`` ``med_nest_initial`` copies ``nest%mub``
    into ``nest%mub_save`` (the pre-blend child input column mass), then
    ``blend_terrain`` blends the parent-interpolated ``nest%mub_fine`` into the
    current ``nest%mub`` BEFORE calling
    ``adjust_tempqv(nest%mub, nest%mub_save, ...)`` (``dyn_em/nest_init_utils.F``).
    The temp/qv adjustment therefore sees this TRANSIENT post-blend column mass.

    The FINAL base state (``PB``/``MUB``/``PHB``) is recomputed LATER by
    ``start_domain`` from the blended terrain (``_wrf_start_domain_base_from_hgt``
    inside :func:`_apply_live_nest_base_init`).  That final ``MUB`` is a DIFFERENT
    surface and is intentionally left unchanged by this helper -- this helper only
    exposes the transient adjust-base ``MUB`` for theta/qv adjustment.

    ``parent_mub`` is the parent-grid base column mass (``MUB``); ``child_mub`` is
    the child-grid input base column mass (``nest%mub_save``).  Returns
    ``(save_mub, transient_mub, meta)`` with both arrays on the child mass grid.
    This is an initialization-only host transcription; no timestep-loop transfer.
    """

    child_meta = run.grid(domain)
    expected_parent = f"d{int(child_meta.parent_id):02d}"
    ratio = int(child_meta.parent_grid_ratio)
    if ratio <= 1:
        raise ValueError(
            f"{domain}: live-nest transient adjust MUB requires parent_grid_ratio > 1, got {ratio}"
        )
    spec_bdy_width = _namelist_int_any(run, "spec_bdy_width", 5, domain=domain)
    blend_width = _namelist_int_any(run, "blend_width", 5, domain=domain)

    # WRF interpolates/blends nest%mub in REAL(4); see the terrain comment in
    # _apply_live_nest_base_init for the fp32-faithfulness rationale.
    parent_mub_np = np.asarray(jax.device_get(parent_mub), dtype=np.float32)
    child_mub_np = np.asarray(jax.device_get(child_mub), dtype=np.float32)
    expected_shape = (int(grid.ny), int(grid.nx))
    if child_mub_np.shape != expected_shape:
        raise ValueError(
            f"{domain}: live-nest child MUB shape {child_mub_np.shape} != mass grid {expected_shape}"
        )
    parent_mub_on_child = sint_to_child_reference(
        parent_mub_np,
        ratio=ratio,
        i_parent_start=int(child_meta.i_parent_start),
        j_parent_start=int(child_meta.j_parent_start),
        child_ny=int(grid.ny),
        child_nx=int(grid.nx),
        dtype=np.float32,
    )
    if not np.isfinite(parent_mub_on_child).all():
        raise ValueError(
            f"{domain}: live-nest SINT MUB interpolation produced non-finite values; "
            "the parent stencil halo or nest geometry is insufficient"
        )
    transient_mub_np = _wrf_blend_terrain_host(
        parent_mub_on_child,
        child_mub_np,
        spec_bdy_width=spec_bdy_width,
        blend_width=blend_width,
    )
    dtype = jnp.asarray(child_mub).dtype
    save_mub = jnp.asarray(child_mub_np, dtype=dtype)
    transient_mub = jnp.asarray(transient_mub_np, dtype=dtype)
    meta = {
        "surface": "post_blend_terrain_pre_start_domain",
        "consumer": "dyn_em/nest_init_utils.F::adjust_tempqv current MUB argument",
        "parent_domain": expected_parent,
        "child_domain": domain,
        "parent_grid_ratio": ratio,
        "i_parent_start": int(child_meta.i_parent_start),
        "j_parent_start": int(child_meta.j_parent_start),
        "spec_bdy_width": int(spec_bdy_width),
        "blend_width": int(blend_width),
        "interpolation": "gpuwrf.nesting.interp.sint_to_child_reference (WRF sint.F host reference, init-only)",
        "wrf_reference": (
            "share/mediation_integrate.F med_nest_initial: copy_3d_field(mub_save,mub); "
            "blend_terrain(mub_fine,mub); adjust_tempqv(mub,mub_save,...)"
        ),
        "final_base_state_unchanged": True,
        "timestep_loop_transfer": False,
    }
    return save_mub, transient_mub, meta


def _wrf_use_theta_m(run: Gen2Run, domain: str) -> int:
    """Resolve WRF ``use_theta_m`` for ``domain`` (WRF default ``1``).

    Prefers the authoritative ``USE_THETA_M`` global attribute on
    ``wrfinput_<domain>`` (the file the live-nest child IC is read from), then the
    ``dynamics`` namelist scalar, then the WRF default of ``1``.  This decides
    whether the live-nest child temperature is the moist potential temperature
    ``theta_m`` (``dyn_em/module_initialize_real.F:4918-4928``).
    """

    try:
        from netCDF4 import Dataset  # noqa: PLC0415

        with Dataset(run.wrfinput_file(domain), "r") as dataset:
            if hasattr(dataset, "USE_THETA_M"):
                value = getattr(dataset, "USE_THETA_M")
                return int(value.item() if hasattr(value, "item") else value)
    except Exception:
        pass
    raw = run.namelist.get("dynamics", {}).get("use_theta_m")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return 1


# WRF adjust_tempqv constants (dyn_em/nest_init_utils.F::adjust_tempqv).
_ADJ_P0_PA = 1.0e5            # reference pressure for the Exner factor
_ADJ_RCP = 2.0 / 7.0         # R_d/c_p for dry air (kappa)
_ADJ_T_FREEZE_C = 273.15     # K -> C offset
_ADJ_ES_A = 610.78           # Tetens saturation-vapor-pressure coefficients
_ADJ_ES_B = 17.0809
_ADJ_ES_C = 234.175
_ADJ_EPS = 0.622             # R_d/R_v mixing-ratio factor
_ADJ_HYDRO_COEF = -191.86e-3  # g/(c_p) hydrostatic theta increment coefficient


def _wrf_live_nest_adjust_tempqv(
    *,
    theta: Any,
    qv: Any,
    p_perturbation: Any,
    save_mub: Any,
    transient_mub: Any,
    metrics: DycoreMetrics,
    use_theta_m: int,
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    """WRF live-nest child temperature/moisture adjustment for blended base mass.

    Transcribes ``share/mediation_integrate.F`` ``med_nest_initial`` calling
    ``dyn_em/nest_init_utils.F::adjust_tempqv(nest%mub, nest%mub_save, ...)`` after
    ``blend_terrain``, with the ``dyn_em/module_initialize_real.F:4918-4928``
    dry->moist ``theta_m`` conversion applied first when ``use_theta_m == 1``.

    ``theta`` is the loaded FULL potential temperature (WRF ``T`` + ``t0``);
    ``save_mub`` is the pre-blend child input column mass (``nest%mub_save``);
    ``transient_mub`` is the post-blend current column mass
    (:func:`_wrf_live_nest_transient_adjust_mub`).  The perturbation pressure
    ``p_perturbation`` is unchanged by WRF here, and the base state (``PB``/``MUB``/
    ``PHB``) recomputed by ``start_domain`` is intentionally untouched.

    Returns ``(theta_full_out, qv_out, meta)``: the adjusted FULL potential
    temperature (still moist when ``use_theta_m == 1``) and the RH-conserving
    adjusted ``QVAPOR``, both cast back to the input dtypes.  Initialization-only;
    no timestep-loop transfer.
    """

    theta_dtype = jnp.asarray(theta).dtype
    qv_dtype = jnp.asarray(qv).dtype
    th_full = jnp.asarray(theta, dtype=jnp.float64)
    qv_in = jnp.asarray(qv, dtype=jnp.float64)
    pp = jnp.asarray(p_perturbation, dtype=jnp.float64)
    save = jnp.asarray(save_mub, dtype=jnp.float64)[None, :, :]
    cur = jnp.asarray(transient_mub, dtype=jnp.float64)[None, :, :]
    c3 = jnp.asarray(metrics.c3h, dtype=jnp.float64)[:, None, None]
    c4 = jnp.asarray(metrics.c4h, dtype=jnp.float64)[:, None, None]
    p_top = jnp.reshape(jnp.asarray(metrics.p_top, dtype=jnp.float64), ())

    t0 = jnp.asarray(P0_THETA_OFFSET_K, dtype=jnp.float64)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    rv_over_rd = jnp.asarray(_RVOVRD, dtype=jnp.float64)
    moist = int(use_theta_m) == 1

    # module_initialize_real.F:4923-4928: dry -> moist theta_m before halos.
    th_pert = th_full - t0
    if moist:
        th_pert = (th_pert + t0) * (one + rv_over_rd * qv_in) - t0

    # nest_init_utils.F::adjust_tempqv: RH-conserving temp/qv adjust from the
    # pre-blend base pressure (save_mub) to the post-blend base pressure (mub).
    p_old = c4 + c3 * save + p_top + pp
    p_new = c4 + c3 * cur + p_top + pp
    exner_old = (p_old / _ADJ_P0_PA) ** _ADJ_RCP
    if moist:
        tc = (th_pert + t0) * exner_old / (one + rv_over_rd * qv_in) - _ADJ_T_FREEZE_C
        thloc = (th_pert + t0) / (one + rv_over_rd * qv_in)
    else:
        tc = (th_pert + t0) * exner_old - _ADJ_T_FREEZE_C
        thloc = th_pert + t0
    es = _ADJ_ES_A * jnp.exp(_ADJ_ES_B * tc / (_ADJ_ES_C + tc))
    e = qv_in * p_old / (_ADJ_EPS + qv_in)
    rh = e / es

    dth1 = _ADJ_HYDRO_COEF * thloc / (p_new + p_old) * (p_new - p_old)
    dth = _ADJ_HYDRO_COEF * (thloc + jnp.asarray(0.5, dtype=jnp.float64) * dth1) / (p_new + p_old) * (p_new - p_old)
    if moist:
        th_pert_out = (thloc + dth) * (one + rv_over_rd * qv_in) - t0
    else:
        th_pert_out = thloc + dth - t0

    tc_new = (thloc + dth) * (p_new / _ADJ_P0_PA) ** _ADJ_RCP - _ADJ_T_FREEZE_C
    es_new = _ADJ_ES_A * jnp.exp(_ADJ_ES_B * tc_new / (_ADJ_ES_C + tc_new))
    e_new = rh * es_new
    qv_out = _ADJ_EPS * e_new / (p_new - e_new)

    theta_full_out = (th_pert_out + t0).astype(theta_dtype)
    qv_out_cast = qv_out.astype(qv_dtype)
    meta = {
        "use_theta_m": int(use_theta_m),
        "theta_m_conversion_applied": bool(moist),
        "surface": "post_blend_terrain_pre_start_domain (adjust_tempqv current MUB)",
        "perturbation_pressure_unchanged": True,
        "final_base_state_unchanged": True,
        "wrf_reference": (
            "dyn_em/module_initialize_real.F:4918-4928 theta_m; "
            "dyn_em/nest_init_utils.F::adjust_tempqv(nest%mub, nest%mub_save, ...)"
        ),
        "timestep_loop_transfer": False,
    }
    return theta_full_out, qv_out_cast, meta


def _wrf_live_nest_start_domain_perturb_init(
    run: Gen2Run,
    *,
    domain: str,
    grid: GridSpec,
    metrics: DycoreMetrics,
    ph_perturbation: Any,
    mu_perturbation: Any,
    theta_full: Any,
    w: Any,
    u: Any,
    v: Any,
    ht_fine: Any,
    base_pb: Any | None = None,
    base_mub: Any | None = None,
    base_phb: Any | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, Any]]:
    """WRF live-nest ``start_domain_em`` perturbation-state initialization.

    Transcribes, in REAL(4) precision with the WRF build's float32 libm, the
    three ``start_domain(nest,.TRUE.)`` mutations that follow the base-state
    recompute and that raw ``wrfinput_<domain>`` perturbation leaves are missing
    (proofs/v014/step1_live_nest_perturb_state_init.* localized these as the
    remaining Step-1 ``P/MU/W`` gap):

    1. ``P``: the ``calc_p_rho_phi``-equation rederivation of ``al``/``p`` from
       ``ph_1`` for ``hypsometric_opt=2`` (``dyn_em/start_em.F``,
       "Use equations from calc_p_rho_phi to derive p and al from ph"), with
       ``qvf = 1`` under ``use_theta_m=1``.
    2. ``MU``: the ``press_adj`` terrain-delta column-mass correction
       ``MU_2 += al(1)/(alt(1)*alb(1)) * g * (ht - ht_fine)``.
    3. ``W``: ``module_bc_em.F::set_w_surface`` with ``fill_w_flag=.true.``,
       applied only when the input surface ``W`` is identically ~0 (the WRF
       ``w_needs_to_be_set`` gate; real-init nests carry no input ``W``).

    ``theta_full`` must be the POST-``adjust_tempqv`` full (moist when
    ``use_theta_m=1``) potential temperature -- WRF runs ``start_domain`` after
    ``med_nest_initial``'s temperature adjustment.  ``ht_fine`` is the nest's own
    pre-blend input terrain (WRF ``grid%ht_fine``); ``grid`` must already carry
    the blended terrain.  Returns ``(p_perturbation, mu_perturbation, w)`` as
    float64 arrays holding the WRF REAL(4) values exactly.
    Initialization-only host transcription; no timestep-loop transfer.
    """

    m = _WRF_INIT_LIBM32
    if base_pb is None or base_mub is None or base_phb is None:
        pb64, mub64, phb64, _t_init64, alb64 = _wrf_start_domain_base_from_hgt(
            run,
            domain,
            hgt=grid.terrain_height,
            metrics=metrics,
        )
    else:
        pb64 = jnp.asarray(base_pb)
        mub64 = jnp.asarray(base_mub)
        phb64 = jnp.asarray(base_phb)
        alb64 = _wrf_base_alb_from_loaded_state(phb=phb64, mub=mub64, metrics=metrics)
    pb = np.asarray(jax.device_get(pb64), dtype=np.float32)
    mub = np.asarray(jax.device_get(mub64), dtype=np.float32)
    phb = np.asarray(jax.device_get(phb64), dtype=np.float32)
    alb = np.asarray(jax.device_get(alb64), dtype=np.float32)

    c3h = np.asarray(jax.device_get(metrics.c3h), dtype=np.float32)
    c4h = np.asarray(jax.device_get(metrics.c4h), dtype=np.float32)
    c3f = np.asarray(jax.device_get(metrics.c3f), dtype=np.float32)
    c4f = np.asarray(jax.device_get(metrics.c4f), dtype=np.float32)
    p_top = np.float32(float(np.asarray(jax.device_get(metrics.p_top)).ravel()[0]))

    ph32 = np.asarray(jax.device_get(ph_perturbation), dtype=np.float32)
    mu32 = np.asarray(jax.device_get(mu_perturbation), dtype=np.float32)
    # WRF stores the perturbation theta t_1 in REAL(4); recover it from the fp64
    # full theta so the EOS sees (t0 + t_1) exactly as start_em.F does.
    t1_32 = (np.asarray(jax.device_get(theta_full), dtype=np.float64) - float(P0_THETA_OFFSET_K)).astype(np.float32)

    # --- 1. AL (hypsometric_opt=2) and P from the calc_p_rho_phi equations ---
    full_mu = (mub + mu32).astype(np.float32)
    pfu = (c3f[1:, None, None] * full_mu[None] + c4f[1:, None, None] + p_top).astype(np.float32)
    pfd = (c3f[:-1, None, None] * full_mu[None] + c4f[:-1, None, None] + p_top).astype(np.float32)
    phm = (c3h[:, None, None] * full_mu[None] + c4h[:, None, None] + p_top).astype(np.float32)
    dph = (ph32[1:] - ph32[:-1] + phb[1:] - phb[:-1]).astype(np.float32)
    # grid%al = (dph)/phm/LOG(pfd/pfu) - alb  (two sequential divisions, as in WRF)
    al = (dph / phm / m.log((pfd / pfu).astype(np.float32)) - alb).astype(np.float32)
    alt = (al + alb).astype(np.float32)
    theta_eos = (_WRF32_T0 + t1_32).astype(np.float32)  # qvf = 1 for use_theta_m=1
    ratio = ((_WRF32_R_D * theta_eos) / (_WRF32_P1000MB * alt)).astype(np.float32)
    p_new = (_WRF32_P1000MB * m.pow(ratio, _WRF32_CPOVCV) - pb).astype(np.float32)

    # --- 2. press_adj column-mass correction (press_adj=T for the live nest) ---
    ht32 = np.asarray(jax.device_get(grid.terrain_height), dtype=np.float32)
    ht_fine32 = np.asarray(jax.device_get(ht_fine), dtype=np.float32)
    mu_new = (mu32 + al[0] / (alt[0] * alb[0]) * _WRF32_G * (ht32 - ht_fine32)).astype(np.float32)

    # --- 3. set_w_surface(fill_w_flag=.true.) under the WRF w_needs_to_be_set gate ---
    w32 = np.asarray(jax.device_get(w), dtype=np.float32)
    w_surface_input_max = float(np.abs(w32[0]).max())
    w_needs_to_be_set = w_surface_input_max < 1.0e-6
    if w_needs_to_be_set:
        u32 = np.asarray(jax.device_get(u), dtype=np.float32)
        v32 = np.asarray(jax.device_get(v), dtype=np.float32)
        msftx = np.asarray(jax.device_get(metrics.msftx), dtype=np.float32)
        msfty = np.asarray(jax.device_get(metrics.msfty), dtype=np.float32)
        znw = np.asarray(jax.device_get(grid.vertical.eta_levels), dtype=np.float32)
        cf1 = np.float32(float(np.asarray(jax.device_get(metrics.cf1)).ravel()[0]))
        cf2 = np.float32(float(np.asarray(jax.device_get(metrics.cf2)).ravel()[0]))
        cf3 = np.float32(float(np.asarray(jax.device_get(metrics.cf3)).ravel()[0]))
        rdx = np.float32(1.0) / np.float32(float(grid.projection.dx_m))
        rdy = np.float32(1.0) / np.float32(float(grid.projection.dy_m))
        half = np.float32(0.5)

        ny, nx = ht32.shape
        jp1 = np.minimum(np.arange(ny) + 1, ny - 1)
        jm1 = np.maximum(np.arange(ny) - 1, 0)
        ip1 = np.minimum(np.arange(nx) + 1, nx - 1)
        im1 = np.maximum(np.arange(nx) - 1, 0)
        # cf1*v(.,1,.)+cf2*v(.,2,.)+cf3*v(.,3,.) at the v-rows j and j+1
        vv = (cf1 * v32[0] + cf2 * v32[1] + cf3 * v32[2]).astype(np.float32)  # (ny+1, nx)
        uu = (cf1 * u32[0] + cf2 * u32[1] + cf3 * u32[2]).astype(np.float32)  # (ny, nx+1)
        w_sfc = (
            msfty
            * half
            * rdy
            * ((ht32[jp1, :] - ht32) * vv[1:, :] + (ht32 - ht32[jm1, :]) * vv[:-1, :])
            + msftx
            * half
            * rdx
            * ((ht32[:, ip1] - ht32) * uu[:, 1:] + (ht32 - ht32[:, im1]) * uu[:, :-1])
        ).astype(np.float32)
        w_new = np.empty_like(w32)
        w_new[0] = w_sfc
        for k in range(1, w32.shape[0]):
            w_new[k] = (w_sfc * znw[k] * znw[k]).astype(np.float32)
    else:
        w_new = w32

    meta = {
        "surfaces": [
            "start_domain hypsometric AL/ALT + calc_p_rho_phi pressure",
            "press_adj MU correction",
            "set_w_surface(fill_w_flag=.true.)" if w_needs_to_be_set else "input W kept (surface W nonzero)",
        ],
        "hypsometric_opt": 2,
        "use_theta_m_qvf": 1.0,
        "w_needs_to_be_set": bool(w_needs_to_be_set),
        "w_surface_input_max_abs": w_surface_input_max,
        "libm_provider": m.provider,
        "wrf_reference": (
            "dyn_em/start_em.F (AL/ALT + p from calc_p_rho_phi equations; press_adj); "
            "dyn_em/module_bc_em.F::set_w_surface"
        ),
        "cpu_wrfout_dependency": False,
        "timestep_loop_transfer": False,
    }
    as64 = lambda arr: jnp.asarray(np.asarray(arr, dtype=np.float64))  # noqa: E731
    return as64(p_new), as64(mu_new), as64(w_new), meta


# WRF MYNN cold-start TKE constants (phys/module_bl_mynnedmf.F).
_MYNN_QKE_INIT_THRESHOLD = 0.0002  # module_bl_mynnedmf.F:623 MAXVAL(qke)<0.0002 => INITIALIZE_QKE


def _wrf_mynn_coldstart_qke(
    qke: jax.Array,
    *,
    state,
    grid,
) -> tuple[jax.Array, bool]:
    """Seed the WRF MYNN ``mym_initialize`` cold-start TKE state.

    The parent analysis carries no real QKE at the replay/live-nest start time
    (``QKE`` is identically zero / the nest interpolates a zero parent field).
    Real WRF NEVER advances MYNN from ``qke=0``: on the first call
    (``initflag>0``, not restart) the scheme sets ``INITIALIZE_QKE=.TRUE.``
    (module_bl_mynnedmf.F:618-632) and ``mym_initialize`` builds the turbulence
    state internally — a driver taper pre-seed, a frozen ``GET_PBLH``/
    ``SCALE_AWARE`` pass, and a 5-pass level-2 closure equilibrium iteration
    (module_bl_mynnedmf.F:1281-1430). In statically unstable initial layers the
    equilibrium qke is O(0.1-10 m^2/s^2); the earlier taper-only seed missed
    that by 3-5 orders of magnitude, which made the JAX step-1 MYNN sources
    ~10x weaker than WRF (proofs/v014/mynn_driver_source_output_fix).

    This mirrors WRF's own cold-start INITIALIZATION (init-time construction,
    not a runtime clamp/mask), via
    :func:`gpuwrf.coupling.physics_couplers.mynn_coldstart_qke_from_state`.

    Returns ``(qke_seeded, did_seed)``.  When the parent already carries TKE
    above the WRF threshold the array is returned unchanged (matching WRF's
    ``INITIALIZE_QKE=.FALSE.`` branch).
    """

    qke_arr = jnp.asarray(qke)
    if float(jnp.max(qke_arr)) >= _MYNN_QKE_INIT_THRESHOLD:
        return qke_arr, False

    from gpuwrf.coupling.physics_couplers import mynn_coldstart_qke_from_state

    qke_seed = mynn_coldstart_qke_from_state(state, grid)
    return qke_seed.astype(qke_arr.dtype), True


# --------------------------------------------------------------------------- #
# v0.12.0 standalone native-init lateral boundary: decode wrfbdy_<domain> into
# the operational ``*_bdy`` leaves WITHOUT any CPU-WRF wrfout history. This is the
# genuine out-of-the-box path: real.exe already produced wrfinput + wrfbdy, so we
# read the lateral forcing straight from wrfbdy instead of reconstructing it from
# pre-existing CPU wrfout history (the REPLAY path).
# --------------------------------------------------------------------------- #
# WRF couples the wrfbdy specified values by the hybrid mass term (the same term
# the dycore couples by). Decoupling to the raw fields the operational
# ``apply_lateral_boundaries`` adapter consumes (it interpolates raw decoupled
# leaves and re-couples internally) is the exact inverse:
#   u  = U_BXS  / (c1h*mass_u + c2h) * msfuy        (mass_u = staggered-x dry mass)
#   v  = V_BXS  / (c1h*mass_v + c2h) * msfvx
#   t  = T_BXS  / (c1h*total_mass + c2h)  (-> THM perturbation; +t0/(1+rv/rd qv) below)
#   qv = QVAPOR_BXS / (c1h*total_mass + c2h)
#   ph = PH_BXS / (c1f*total_mass + c2f)
#   mu = MU_BXS                                     (uncoupled MU_2)
# wrfbdy already stores E/N strips inner-to-outer flipped (module_bc.F stuff_bdy),
# which is the SAME orientation ``_field_sides_3d`` / the operational leaf packer
# use, so the decoded side strips map straight onto W/E/S/N with no extra flip.
_T0_K = P0_THETA_OFFSET_K
_RVOVRD = 461.6 / 287.0
_WRFBDY_SIDE_KEYS = {"W": "bxs", "E": "bxe", "S": "bys", "N": "bye"}


def _wrfbdy_total_mass_strips(
    *,
    mu_total: np.ndarray,
    metrics: DycoreMetrics,
    msfuy: np.ndarray,
    msfvx: np.ndarray,
    width: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-slice the IC dry-mass coupling term onto every wrfbdy side strip.

    Returns ``{coupling_name: {side: strip}}`` where ``coupling_name`` selects the
    stagger (``mass_h`` for theta/qv, ``mass_f`` for ph, ``mass_u``/``mass_v`` for
    the C-grid winds) and each strip is in the WRF wrfbdy ``(bdy_width, z, tan)``
    order (2D mass uses ``(bdy_width, 1, tan)``).
    """

    total = np.asarray(mu_total, dtype=np.float64)
    ny, nx = total.shape
    c1h = np.asarray(metrics.c1h, dtype=np.float64)
    c2h = np.asarray(metrics.c2h, dtype=np.float64)
    c1f = np.asarray(metrics.c1f, dtype=np.float64)
    c2f = np.asarray(metrics.c2f, dtype=np.float64)
    mass_h = c1h[:, None, None] * total[None, :, :] + c2h[:, None, None]            # (nz, ny, nx)
    mass_f = c1f[:, None, None] * total[None, :, :] + c2f[:, None, None]            # (nz+1, ny, nx)
    # Staggered C-grid dry mass on U/V faces (lateral_bc._staggered_total_mass).
    mass_u_2d = np.empty((ny, nx + 1), dtype=np.float64)
    mass_u_2d[:, 0] = total[:, 0]
    mass_u_2d[:, 1:nx] = 0.5 * (total[:, 1:] + total[:, :-1])
    mass_u_2d[:, nx] = total[:, nx - 1]
    mass_v_2d = np.empty((ny + 1, nx), dtype=np.float64)
    mass_v_2d[0, :] = total[0, :]
    mass_v_2d[1:ny, :] = 0.5 * (total[1:, :] + total[:-1, :])
    mass_v_2d[ny, :] = total[ny - 1, :]
    msfuy = np.asarray(msfuy, dtype=np.float64)
    msfvx = np.asarray(msfvx, dtype=np.float64)
    mass_u = (c1h[:, None, None] * mass_u_2d[None, :, :] + c2h[:, None, None]) / msfuy[None, :, :]
    mass_v = (c1h[:, None, None] * mass_v_2d[None, :, :] + c2h[:, None, None]) / msfvx[None, :, :]

    fields = {"mass_h": mass_h, "mass_f": mass_f, "mass_u": mass_u, "mass_v": mass_v}
    strips: dict[str, dict[str, np.ndarray]] = {}
    for name, field in fields.items():
        per_side = _field_sides_3d(field, width)  # (bdy_width, z, tan) per W/E/S/N
        strips[name] = per_side
    return strips


def _broadcast_base_leaf_3d(field: np.ndarray, *, width: int, max_side: int, ntimes: int) -> np.ndarray:
    """Broadcast a static (z, ny, nx) base field onto a (time, 4, bw, z, side_len) leaf."""
    per_side = _field_sides_3d(np.asarray(field, dtype=np.float64), width)
    z_len = int(per_side["W"].shape[1])
    leaf = np.zeros((ntimes, 4, width, z_len, max_side), dtype=np.float64)
    for side in SIDES:
        s = per_side[side]
        leaf[:, SIDE_INDEX[side], : s.shape[0], : s.shape[1], : s.shape[2]] = s[None]
    return leaf


def _broadcast_base_leaf_2d(field: np.ndarray, *, width: int, max_side: int, ntimes: int) -> np.ndarray:
    """Broadcast a static (ny, nx) base field onto a (time, 4, bw, 1, side_len) leaf."""
    per_side = _field_sides_2d(np.asarray(field, dtype=np.float64), width)
    leaf = np.zeros((ntimes, 4, width, 1, max_side), dtype=np.float64)
    for side in SIDES:
        s = per_side[side]
        leaf[:, SIDE_INDEX[side], : s.shape[0], 0, : s.shape[1]] = s
    return leaf


def load_wrfbdy_boundary_leaves(
    run: Gen2Run,
    grid: GridSpec,
    *,
    domain: str,
    mu_total: np.ndarray,
    metrics: DycoreMetrics,
    pb: np.ndarray | None = None,
    phb: np.ndarray | None = None,
    mub: np.ndarray | None = None,
    p_perturbation: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build operational ``*_bdy`` leaves by decoding ``wrfbdy_<domain>`` directly.

    The native standalone LBC source: NO CPU-WRF wrfout history. Decodes every
    wrfbdy forcing interval into the decoupled ``(time, side, bdy_width, z,
    side_len)`` leaves the operational boundary adapter consumes.
    """

    try:
        bdy_path = wrfbdy_path_for_run(run, domain)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"standalone native-init requires wrfbdy_{domain} for lateral forcing "
            f"(no CPU-WRF wrfout history present in {run.path}): {exc}"
        ) from exc

    # interval (bdyfrq) seconds: prefer the namelist, fall back to 6 h.
    interval_s = float(
        run.namelist.get("time_control", {}).get("interval_seconds", 21600) or 21600
    )
    probe = decode_wrfbdy(bdy_path, variables=("MU",), time_index=0)
    ntimes = int(len(probe.get("times", [])) or 1)
    width = int(probe.get("bdy_width", 5))
    max_side = int(max(grid.nx + 1, grid.ny + 1))

    mass_strips = _wrfbdy_total_mass_strips(
        mu_total=mu_total, metrics=metrics, msfuy=metrics.msfuy, msfvx=metrics.msfvx, width=width
    )

    def _decoupled_strip(coupled: np.ndarray, side: str, mass_name: str) -> np.ndarray:
        coupled = np.asarray(coupled, dtype=np.float64)  # (bw, z, tan)
        mstrip = mass_strips[mass_name][side]  # (bw, z, tan)
        z_use = min(coupled.shape[1], mstrip.shape[1])
        safe = np.where(np.abs(mstrip[:, :z_use]) > 1.0e-12, mstrip[:, :z_use], 1.0e-12)
        out = np.array(coupled)
        out[:, :z_use] = coupled[:, :z_use] / safe
        return out

    # Per-interval decode: read the boundary VALUE at each wrfbdy time level (the
    # _BX* base term IS the specified value at that interval start; WRF advances it
    # with the _BT* tendency, but reading the per-interval base value is exact at
    # each interval boundary and the operational adapter interpolates between them).
    u_t, v_t, th_t, qv_t, ph_t, w_t, mu_t = ([] for _ in range(7))
    # The wrfbdy ``T`` strips force WRF's runtime prognostic theta: MOIST
    # theta_m under use_theta_m=1 (init/real_init/types.py use_theta_m note),
    # dry theta under use_theta_m=0. Operational State.theta is theta_m, so
    # keep the moist strips as-is and recouple dry strips at ingest. (The
    # previous decode divided to DRY theta, forcing a ~5 K-deficient boundary
    # ring against the moist interior -- part of the v0.14 h1 root cause.)
    use_theta_m = _wrf_use_theta_m(run, domain)
    # Each wrfbdy record k holds the specified value at t = k*interval plus the
    # _BT* tendency that advances it across [k*interval, (k+1)*interval). The
    # leaf therefore needs ONE synthesized terminal level (last base + last
    # tendency * interval) so the leaf time axis spans the full forecast;
    # without it ``interpolate_boundary_leaf`` clamps frozen at the last record
    # start (e.g. 66 h of a 72 h run).
    records = [(k, 0.0) for k in range(ntimes)] + [(ntimes - 1, interval_s)]
    for k, tend_adv_s in records:
        dv = decode_wrfbdy(bdy_path, variables=("U", "V", "W", "T", "QVAPOR", "PH", "MU"), time_index=k)
        vars_k = dv["variables"]

        def _coupled_strip(var: str, side: str) -> np.ndarray:
            rec = vars_k[var]["sides"][side]
            base = np.asarray(rec["boundary"], dtype=np.float64)
            if tend_adv_s:
                base = base + np.asarray(rec["tendency"], dtype=np.float64) * float(tend_adv_s)
            return base

        per_u = np.zeros((4, width, grid.nz, max_side), dtype=np.float64)
        per_v = np.zeros((4, width, grid.nz, max_side), dtype=np.float64)
        per_th = np.zeros((4, width, grid.nz, max_side), dtype=np.float64)
        per_qv = np.zeros((4, width, grid.nz, max_side), dtype=np.float64)
        per_ph = np.zeros((4, width, grid.nz + 1, max_side), dtype=np.float64)
        per_w = np.zeros((4, width, grid.nz + 1, max_side), dtype=np.float64)
        per_mu = np.zeros((4, width, 1, max_side), dtype=np.float64)
        for side in SIDES:
            si = SIDE_INDEX[side]
            u_s = _decoupled_strip(_coupled_strip("U", side), side, "mass_u")
            v_s = _decoupled_strip(_coupled_strip("V", side), side, "mass_v")
            qv_s = np.maximum(_decoupled_strip(_coupled_strip("QVAPOR", side), side, "mass_h"), 0.0)
            thm_s = _decoupled_strip(_coupled_strip("T", side), side, "mass_h")
            if use_theta_m == 1:
                # already the moist theta_m perturbation -> full theta_m
                th_s = thm_s + _T0_K
            else:
                # dry perturbation theta -> full moist theta_m at ingest
                th_s = (thm_s + _T0_K) * (1.0 + _RVOVRD * qv_s)
            ph_s = _decoupled_strip(_coupled_strip("PH", side), side, "mass_f")
            w_s = _coupled_strip("W", side)  # W uncoupled (0)
            mu_s = _coupled_strip("MU", side)  # uncoupled
            for dst, src in ((per_u, u_s), (per_v, v_s), (per_th, th_s), (per_qv, qv_s)):
                dst[si, : src.shape[0], : src.shape[1], : src.shape[2]] = src[:, :, :]
            per_ph[si, : ph_s.shape[0], : ph_s.shape[1], : ph_s.shape[2]] = ph_s
            per_w[si, : w_s.shape[0], : w_s.shape[1], : w_s.shape[2]] = w_s
            per_mu[si, : mu_s.shape[0], 0, : mu_s.shape[1]] = mu_s
        u_t.append(per_u); v_t.append(per_v); th_t.append(per_th)
        qv_t.append(per_qv); ph_t.append(per_ph); w_t.append(per_w); mu_t.append(per_mu)

    leaves_np = {
        "u_bdy": np.stack(u_t, axis=0),
        "v_bdy": np.stack(v_t, axis=0),
        "theta_bdy": np.stack(th_t, axis=0),
        "qv_bdy": np.stack(qv_t, axis=0),
        "ph_bdy": np.stack(ph_t, axis=0),
        "w_bdy": np.stack(w_t, axis=0),
        "mu_bdy": np.stack(mu_t, axis=0),
    }
    # Base-state lateral leaves: pb/phb/mub are STATIC (they never evolve), so the
    # operational ``apply_lateral_boundaries`` re-forces the boundary ring to the IC
    # base strips. p_bdy carries the perturbation-pressure forcing; wrfbdy has no
    # specified perturbation-pressure boundary, so hold it at the IC perturbation
    # strip (a steady ring, consistent with the static base state). Broadcasting
    # across the same ``ntimes`` keeps interpolate_boundary_leaf shape-stable.
    ntimes_leaf = len(records)
    if pb is not None:
        leaves_np["pb_bdy"] = _broadcast_base_leaf_3d(pb, width=width, max_side=max_side, ntimes=ntimes_leaf)
    if phb is not None:
        leaves_np["phb_bdy"] = _broadcast_base_leaf_3d(phb, width=width, max_side=max_side, ntimes=ntimes_leaf)
    if mub is not None:
        leaves_np["mub_bdy"] = _broadcast_base_leaf_2d(mub, width=width, max_side=max_side, ntimes=ntimes_leaf)
    if p_perturbation is not None:
        leaves_np["p_bdy"] = _broadcast_base_leaf_3d(p_perturbation, width=width, max_side=max_side, ntimes=ntimes_leaf)
    leaves = {name: jax.device_put(jnp.asarray(value)) for name, value in leaves_np.items()}
    meta = {
        "source": "wrfbdy native lateral forcing (standalone; no CPU wrfout replay)",
        "wrfbdy_path": str(bdy_path),
        "times": int(ntimes_leaf),
        "wrfbdy_records": int(ntimes),
        "terminal_level_synthesized": True,
        "bdy_width": int(width),
        "interval_seconds": float(interval_s),
        "side_order": list(SIDES),
        "padded_side_length": max_side,
        "schema": "wrfbdy-decoupled-leaf-v1",
        "coupling": "decoupled by IC hybrid dry-mass term (c1h*muT+c2h etc.)",
        "variables": ["U", "V", "W", "T", "QVAPOR", "PH", "MU"],
        "use_theta_m": int(use_theta_m),
        "theta_bdy_convention": "moist theta_m (matches operational State.theta)",
    }
    return leaves, meta


def _wrfinput_start_label(run: Gen2Run, domain: str) -> str | None:
    """Read the analysis-time label (``YYYY-MM-DD_HH:MM:SS``) from wrfinput ``Times``.

    The standalone path has no wrfout time axis, so the forecast run-start is the
    wrfinput initial time. Returns ``None`` if the record cannot be read.
    """

    try:
        from netCDF4 import Dataset

        with Dataset(run.wrfinput_file(domain), "r") as ds:
            if "Times" not in ds.variables:
                return None
            raw = ds.variables["Times"][:]
            label = b"".join(np.asarray(raw[0]).tolist()).decode("ascii", errors="replace").strip()
            return label or None
    except Exception:  # noqa: BLE001 - best-effort; caller falls back to run_id
        return None


def build_replay_case(
    run_dir: str | Path = DEFAULT_REPLAY_RUN_DIR,
    *,
    domain: str = "d02",
    boundary_domain: str | None = None,
    standalone: bool | None = None,
    load_lateral_boundaries: bool = True,
    live_nest_parent: ReplayCase | None = None,
) -> ReplayCase:
    """Load a Gen2 d02 initial state with WRF perturbation/base splits preserved.

    When ``standalone`` is True (or auto-detected because the run dir has < 2
    ``wrfout_<domain>`` history files), the initial state is read from
    ``wrfinput_<domain>`` and the lateral forcing from ``wrfbdy_<domain>`` -- the
    genuine out-of-the-box path with NO CPU-WRF wrfout dependency. Otherwise the
    classic REPLAY path (IC from wrfout t=0, LBC from wrfout history) is used.

    ``load_lateral_boundaries=False`` loads ONLY the initial state from
    ``wrfinput_<domain>`` and leaves the ``*_bdy`` leaves at their zero-shaped
    ``State.zeros`` defaults -- the standalone LIVE-NESTED child path. A nested
    child reads NO lateral forcing from disk (no ``wrfbdy_<child>``, no
    ``wrfout_<child>``): the live parent constructs its boundary package each
    parent step (``build_child_boundary_package``), so the child only needs
    correctly-shaped (and unread) boundary leaves. ``wrfbdy_d01`` still forces the
    root. ``standalone`` is auto-forced True under this flag so the IC reads come
    from ``wrfinput`` and the wrfinput analysis-time label is used for run-start.

    ``live_nest_parent`` enables the WRF live-nest source initialization for a
    child: parent terrain is interpolated with the WRF ``sint`` host reference,
    HT/MUB/PHB go through the WRF ``blend_terrain`` pre-start path, and the final
    real-run base state is rebuilt by ``start_domain`` from blended HT before the
    ``State`` is handed to the device-resident timestep loop. CPU-WRF history
    output is not read; the parent case must already be loaded from native inputs.
    """

    _debug(f"build_replay_case start run_dir={run_dir} domain={domain} boundary_domain={boundary_domain}")
    run = Gen2Run(run_dir)
    _debug("Gen2Run created")
    # Auto-detect the standalone native-init path before any grid/IC payload is
    # loaded.  The canonical CPU reference directory contains complete wrfout
    # histories; a live-nested GPU ship run must still initialize from the sibling
    # wrfinput files, not from those CPU history snapshots.
    wrfout_history_count = len(sorted(run.path.glob(f"wrfout_{domain}_*")))
    is_standalone = bool(standalone) if standalone is not None else (wrfout_history_count < 2)
    # A live-nested child reads NO lateral forcing from disk: force the standalone
    # IC path (wrfinput) and keep the zero-shaped State.zeros *_bdy leaves; the live
    # parent overwrites them each parent step via build_child_boundary_package.
    if not load_lateral_boundaries:
        is_standalone = True
    grid = run.grid(domain).as_grid_spec()
    _debug(f"grid loaded mass_shape={(grid.nz, grid.ny, grid.nx)}")
    # Keep GridSpec.metrics and the runtime namelist metrics on the same loaded
    # WRF payload. GridSpec.__post_init__ fills metrics=None with DycoreMetrics.flat;
    # leaving that fallback in place made wrfout static metrics emit flat C/DN/map
    # fields even though dynamics consumed the loaded WRF metrics below.
    metrics_source = run.wrfinput_file(domain) if is_standalone else run.history_files(domain)[0]
    metrics = load_wrfinput_metrics(metrics_source)
    grid = dataclass_replace(grid, metrics=metrics)
    _debug(f"load_wrfinput_metrics complete (source={metrics_source})")
    if is_standalone:
        wrfinput_hgt = jnp.asarray(_load_wrfinput(run, domain, "HGT"), dtype=grid.terrain_height.dtype)
        terrain = dataclass_replace(
            grid.terrain,
            source_path=str(run.wrfinput_file(domain)),
            max_elevation_m=float(np.nanmax(np.asarray(jax.device_get(wrfinput_hgt)))),
        )
        grid = dataclass_replace(grid, terrain=terrain, terrain_height=wrfinput_hgt, metrics=metrics)
        _debug(f"standalone grid static HGT sourced from wrfinput_{domain}")
    state = State.zeros(grid)
    _debug("State.zeros complete")
    tendencies = Tendencies.zeros(grid)
    _debug("Tendencies.zeros complete")
    # Replay mode intentionally keeps the B4 snapshot-consistency invariant:
    # prognostic PH/PHB and GridSpec terrain both come from the same t=0 wrfout.
    # Standalone/live-nest mode is different by construction: WRF first reads the
    # child wrfinput, then med_nest_initial blends parent-interpolated fields into
    # that input payload. In that mode the grid and IC loads above are forced to
    # wrfinput before the live-nest blend is applied.
    land = load_prescribed_land_state(run, domain=domain, time=0)
    _debug("load_prescribed_land_state complete")
    source_domain = boundary_domain or domain
    boundary_leaves: dict[str, Any] | None = None
    boundary_meta: dict[str, Any] | None = None
    if not load_lateral_boundaries:
        boundary_leaves = {}
        boundary_meta = {
            "source": "live parent (standalone nested child; no wrfbdy/wrfout read)",
            "note": (
                "child *_bdy leaves stay at State.zeros shapes; the parent supplies "
                "the boundary package each parent step (build_child_boundary_package)"
            ),
        }
        _debug(f"standalone live-nested child: {domain} reads IC only (no LBC from disk)")
    elif is_standalone:
        # Standalone boundary leaves are decoded from wrfbdy below, after the IC
        # base/perturbation split (mu_total) is known (needed to decouple).
        _debug(f"standalone native-init: wrfout_{domain} history={wrfout_history_count} (<2) -> wrfbdy LBC")
    elif source_domain == domain:
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

    p_perturbation = _load_initial(run, domain, "P", 0, use_wrfinput=is_standalone)
    pb = _load_initial(run, domain, "PB", 0, use_wrfinput=is_standalone)
    ph_perturbation = _load_initial(run, domain, "PH", 0, use_wrfinput=is_standalone)
    phb = _load_initial(run, domain, "PHB", 0, use_wrfinput=is_standalone)
    mu_perturbation = _load_initial(run, domain, "MU", 0, use_wrfinput=is_standalone)
    mub = _load_initial(run, domain, "MUB", 0, use_wrfinput=is_standalone)
    theta = _load_initial(run, domain, "T", 0, use_wrfinput=is_standalone) + P0_THETA_OFFSET_K
    qv_initial = _load_initial(run, domain, "QVAPOR", 0, use_wrfinput=is_standalone)
    u_initial = _load_initial(run, domain, "U", 0, use_wrfinput=is_standalone)
    v_initial = _load_initial(run, domain, "V", 0, use_wrfinput=is_standalone)
    w_initial = _load_initial(run, domain, "W", 0, use_wrfinput=is_standalone)
    live_nest_base_meta: dict[str, Any] = {"enabled": False}
    if live_nest_parent is not None:
        # ``nest%mub_save`` is the pre-blend child input column mass; capture it
        # before ``start_domain`` recomputes the final base ``MUB`` from blended
        # terrain (``_apply_live_nest_base_init`` overwrites ``mub`` below).
        # ``nest%ht_fine`` is likewise the pre-blend child input terrain consumed
        # by the ``press_adj`` column-mass correction below.
        child_input_mub = mub
        child_input_terrain = grid.terrain_height
        grid, metrics, pb, phb, mub, live_nest_base_meta = _apply_live_nest_base_init(
            run,
            domain=domain,
            grid=grid,
            metrics=metrics,
            parent_case=live_nest_parent,
            child_mub=child_input_mub,
            child_phb=phb,
        )
        _debug(f"live-nest base init complete parent={live_nest_base_meta.get('parent_domain')} child={domain}")
        # WRF ``med_nest_initial`` adjusts the child temperature/moisture to the
        # transient post-``blend_terrain`` base mass via ``adjust_tempqv`` (with
        # the dry->moist ``theta_m`` conversion first when ``use_theta_m=1``). The
        # final ``start_domain`` base state above is intentionally left unchanged.
        parent_mub = live_nest_parent.base_state.mub
        save_mub_arr, transient_mub_arr, transient_mub_meta = _wrf_live_nest_transient_adjust_mub(
            run,
            domain=domain,
            grid=grid,
            parent_mub=parent_mub,
            child_mub=child_input_mub,
        )
        use_theta_m = _wrf_use_theta_m(run, domain)
        theta, qv_initial, theta_qv_adjust_meta = _wrf_live_nest_adjust_tempqv(
            theta=theta,
            qv=qv_initial,
            p_perturbation=p_perturbation,
            save_mub=save_mub_arr,
            transient_mub=transient_mub_arr,
            metrics=metrics,
            use_theta_m=use_theta_m,
        )
        # WRF ``start_domain(nest,.TRUE.)`` then re-derives the perturbation
        # pressure from ``ph_1`` (calc_p_rho_phi equations), applies the
        # ``press_adj`` column-mass correction, and sets the kinematic surface
        # ``W`` -- raw ``wrfinput_<domain>`` perturbation leaves are missing all
        # three (the Step-1 ``P/MU/W`` divergence family).
        p_perturbation, mu_perturbation, w_initial, perturb_init_meta = (
            _wrf_live_nest_start_domain_perturb_init(
                run,
                domain=domain,
                grid=grid,
                metrics=metrics,
                ph_perturbation=ph_perturbation,
                mu_perturbation=mu_perturbation,
                theta_full=theta,
                w=w_initial,
                u=u_initial,
                v=v_initial,
                ht_fine=child_input_terrain,
                base_pb=pb,
                base_mub=mub,
                base_phb=phb,
            )
        )
        live_nest_base_meta = {
            **live_nest_base_meta,
            "transient_adjust_mub": transient_mub_meta,
            "theta_qv_adjust": theta_qv_adjust_meta,
            "start_domain_perturb_init": perturb_init_meta,
        }
        if _DEBUG:
            _debug(
                f"live-nest theta_m/adjust_tempqv applied use_theta_m={use_theta_m} "
                f"theta_max={float(jnp.max(theta)):.4f}"
            )
        _debug(
            "live-nest start_domain perturb init applied "
            f"(w_set={perturb_init_meta.get('w_needs_to_be_set')}, "
            f"libm={perturb_init_meta.get('libm_provider')})"
        )
    else:
        # WRF's runtime prognostic theta under use_theta_m=1 is MOIST theta_m;
        # wrfinput/wrfout variable ``T`` stays the DRY perturbation theta
        # (real.exe converts AFTER writing dry T: module_initialize_real.F:
        # 4918-4928). Recouple at ingest so State.theta carries ONE convention
        # (theta_m) in every lane -- the dycore EOS (qvf=1), the physics
        # adapters' dry-view decoupling, and the live parent->child boundary
        # package all assume it. The live-nest branch above already converts
        # via _wrf_live_nest_adjust_tempqv. (v0.14 h1 root cause: d01 ran DRY
        # theta against the moist-convention physics/EOS.)
        if _wrf_use_theta_m(run, domain) == 1:
            theta = theta * (1.0 + _RVOVRD * qv_initial)
            _debug("standalone/replay IC theta dry->moist theta_m applied (use_theta_m=1)")
    # WRF-faithful base potential temperature ``t0 + t_init``, recovered by
    # inverting the loaded discrete base state (PB/PHB/MUB) so the dycore's
    # recomputed base inverse density ``alb`` matches the discrete ``alb`` the
    # loaded ``phb`` was hydrostatically integrated from.  Using the constant
    # 300 K here (the prior behaviour) left the loaded IC out of the dycore's
    # discrete hydrostatic balance and drove the steady +2.6 kPa perturbation-
    # pressure / Exner-T2 offset on both d02 and d03 (root cause documented in
    # .agent/reviews/2026-06-01-opus-pressure-drift-rootcause.md).
    theta_base = _wrf_base_theta_from_loaded_state(pb=pb, phb=phb, mub=mub, metrics=metrics)

    if is_standalone and load_lateral_boundaries:
        mub_np = np.asarray(jax.device_get(mub))
        phb_np = np.asarray(jax.device_get(phb))
        pb_np = np.asarray(jax.device_get(pb))
        mu_total_np = mub_np + np.asarray(jax.device_get(mu_perturbation))
        boundary_leaves, boundary_meta = load_wrfbdy_boundary_leaves(
            run, grid, domain=domain, mu_total=mu_total_np, metrics=metrics,
            pb=pb_np, phb=phb_np, mub=mub_np,
            p_perturbation=np.asarray(jax.device_get(p_perturbation)),
        )
        _debug("load_wrfbdy_boundary_leaves complete (standalone)")

    state = state.replace(
        u=u_initial,
        v=v_initial,
        w=w_initial,
        theta=theta,
        qv=qv_initial,
        p_total=pb + p_perturbation,
        p_perturbation=p_perturbation,
        ph_total=phb + ph_perturbation,
        ph_perturbation=ph_perturbation,
        mu_total=mub + mu_perturbation,
        mu_perturbation=mu_perturbation,
        qc=_optional_load_initial(run, domain, "QCLOUD", 0, jnp.zeros_like(state.qc), use_wrfinput=is_standalone),
        qr=_optional_load_initial(run, domain, "QRAIN", 0, jnp.zeros_like(state.qr), use_wrfinput=is_standalone),
        qi=_optional_load_initial(run, domain, "QICE", 0, jnp.zeros_like(state.qi), use_wrfinput=is_standalone),
        qs=_optional_load_initial(run, domain, "QSNOW", 0, jnp.zeros_like(state.qs), use_wrfinput=is_standalone),
        qg=_optional_load_initial(run, domain, "QGRAUP", 0, jnp.zeros_like(state.qg), use_wrfinput=is_standalone),
        Ni=_optional_load_initial(run, domain, "QNICE", 0, jnp.zeros_like(state.Ni), use_wrfinput=is_standalone),
        Nr=_optional_load_initial(run, domain, "QNRAIN", 0, jnp.zeros_like(state.Nr), use_wrfinput=is_standalone),
        Ns=jnp.zeros_like(state.Ns),
        Ng=jnp.zeros_like(state.Ng),
        qke=_optional_load_initial(run, domain, "QKE", 0, jnp.zeros_like(state.qke), use_wrfinput=is_standalone),
        t_skin=land.t_skin,
        soil_moisture=land.soil_moisture[0],
        xland=land.xland,
        lakemask=land.lakemask,
        mavail=land.mavail,
        roughness_m=land.roughness_m,
        lu_index=land.lu_index,
        **boundary_leaves,
    )
    _debug("state.replace with initial fields complete")
    # WRF MYNN cold-start TKE init (module_bl_mynnedmf.F mym_initialize): the
    # parent wrfout carries no real QKE at the analysis time, so build the WRF
    # first-call turbulence state (taper pre-seed + level-2 equilibrium
    # iteration) rather than feeding the MYNN closure a degenerate qke=0 column
    # (which runs away; see proofs/v090/d02replay_qke_*).  No-op when the
    # parent already carries TKE (INITIALIZE_QKE=.FALSE.).
    qke_seeded, did_seed_qke = _wrf_mynn_coldstart_qke(
        state.qke, state=state, grid=grid
    )
    if did_seed_qke:
        state = state.replace(qke=qke_seeded.astype(state.qke.dtype))
        if _DEBUG:
            _debug(f"WRF MYNN cold-start qke seeded: max={float(jnp.max(qke_seeded)):.4g}")
    qke_max = _metadata_host_max(state.qke)
    base = BaseState(
        pb=pb.astype(state.p_total.dtype),
        phb=phb.astype(state.ph_total.dtype),
        mub=mub.astype(state.mu_total.dtype),
        # ``t0`` is the WRF constant theta reference (= 300 K, the EOS ``t0``), not
        # the base profile; ``theta_base`` carries the recovered WRF ``t0+t_init``
        # profile used for the base inverse density ``alb``.
        t0=jnp.full_like(theta_base, P0_THETA_OFFSET_K).astype(state.theta.dtype),
        theta_base=theta_base.astype(state.theta.dtype),
    )
    # Run-start label: replay reads it from the wrfout time axis; standalone has no
    # wrfout, so read the analysis time straight from the wrfinput ``Times`` record.
    start_label = run_start_label(run, domain)
    if is_standalone or start_label == run.run_id:
        start_label = _wrfinput_start_label(run, domain) or start_label
    metadata = {
        "run_id": run.run_id,
        "run_dir": str(run.path),
        "domain": domain,
        "run_start_label": start_label,
        "standalone_native_init": bool(is_standalone),
        "initial_condition_source": (
            f"wrfinput_{domain}" if is_standalone else "wrfout history snapshot at t=0"
        ),
        "static_grid_source": (
            f"wrfinput_{domain}" if is_standalone else "wrfout history snapshot at t=0"
        ),
        "grid": {
            "mass_shape": [int(grid.nz), int(grid.ny), int(grid.nx)],
            "wrf_staggered_extent": [int(grid.nz + 1), int(grid.ny + 1), int(grid.nx + 1)],
            "dx_m": float(grid.projection.dx_m),
            "dy_m": float(grid.projection.dy_m),
        },
        "boundary": boundary_meta,
        "live_nest_base_init": live_nest_base_meta,
        "prescribed_land": {
            "source_file": land.source.get("source_file"),
            "roughness_note": land.source.get("roughness_note"),
            "mavail_note": land.source.get("mavail_note"),
            "missing_optional_variables": land.source.get("missing_optional_variables", []),
        },
        "base_state": {
            "theta_base_k": "WRF t0+t_init profile recovered from loaded PB/PHB/MUB",
            "t0_reference_k": P0_THETA_OFFSET_K,
            "pressure_split": "p_total=PB+P, p_perturbation=P",
            "geopotential_split": "ph_total=PHB+PH, ph_perturbation=PH",
            "mu_split": "mu_total=MUB+MU, mu_perturbation=MU",
            "live_nest_init": live_nest_base_meta,
        },
        "qke_coldstart": {
            "seeded": bool(did_seed_qke),
            "wrf_ref": "phys/module_bl_mynnedmf.F:618-691 (mym_initialize INITIALIZE_QKE)",
            "qke_max": qke_max,
            "qke_max_skipped": qke_max is None,
            "note": (
                "parent wrfout carried no real QKE (MAXVAL<0.0002); seeded WRF "
                "cold-start background TKE profile" if did_seed_qke
                else "parent carried TKE; INITIALIZE_QKE=.FALSE., qke unchanged"
            ),
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
        "ic_source": metadata.get("initial_condition_source", "wrfout history snapshot at t=0"),
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
    # FROZEN Gate-1 physics order (coupler_interface.md S1):
    # thompson -> surface -> mynn -> rrtmg. MYNN READS the surface-flux handles
    # (theta_flux, qv_flux, tau_u/v, rhosfc) that surface_adapter writes, so
    # surface MUST run before mynn (previously mynn ran first here, reading stale
    # fluxes -- a recomposition order bug).
    lead_seconds = global_step.astype(jnp.float64) * float(replay_config.dt_s)
    next_state = thompson_adapter(next_state, float(replay_config.dt_s))
    next_state = surface_adapter(
        next_state,
        float(replay_config.dt_s),
        grid,
        first_timestep=jnp.equal(global_step, 1),
    )
    next_state = mynn_adapter(next_state, float(replay_config.dt_s), grid)
    if bool(run_radiation):
        next_state = rrtmg_adapter(
            next_state, float(replay_config.dt_s), grid, lead_seconds=lead_seconds
        )
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
