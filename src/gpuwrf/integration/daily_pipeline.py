"""M7 daily end-to-end pipeline composition.

This module deliberately wires existing M7 components together. It does not
implement forecast physics, checkpoint internals, NetCDF schema logic, or
station-score algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from gpuwrf.config import paths
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.io.data_inventory import parse_wrfout_valid_time
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_state import load_hourly_land_state
from gpuwrf.io.radiation_static import load_radiation_static
from gpuwrf.io.gwdo_static import load_gwdo_statics
from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
from gpuwrf.io.auxhist_stream import (
    AuxhistStreamConfig,
    auxhist_substeps_per_hour,
    coerce_auxhist_streams,
)
from gpuwrf.io.wrfout_writer import (
    MINIMUM_WRFOUT_VARIABLES,
    prepare_wrfout_payload,
    write_prepared_wrfout,
    write_wrfout_netcdf,
)
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.checkpoint import read_checkpoint, write_checkpoint
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _commit_to_operational_device,
    compute_m9_diagnostics,
    dealias_state_buffers,
    run_forecast_operational,
    run_forecast_operational_segmented,
)
from gpuwrf.validation.forecast_vs_obs import (
    DEFAULT_AEMET_ROOT,
    compute_station_scores,
    interpolate_to_stations,
    inventory_aemet_observations,
    load_aemet_observations,
)


ROOT = Path(__file__).resolve().parents[3]
SPRINT_DIR = ROOT / ".agent" / "sprints" / "2026-05-27-m7-daily-pipeline-integration"
# Path indirection (see gpuwrf.config.paths): environment-overridable so the
# pipeline default run root is not the private <DATA_ROOT> path on a clean clone.
# The CLI passes an absolute --input-dir that bypasses this default entirely.
RUN_ROOT = paths.wrf_l3_root()
DEFAULT_RUN_ID = "20260521_18z_l3_24h_20260522T133443Z"
DEFAULT_OUTPUT_ROOT = Path("/tmp/m7_pipeline_runs/20260521")
DT_S = 10.0
CPU_TIMING_RE = re.compile(
    r"Timing for main: time "
    r"(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}) "
    r"on domain\s+(?P<domain>\d+):\s+"
    r"(?P<elapsed>[0-9.]+) elapsed seconds"
)
WRITER_LATLON_FIELDS: tuple[str, ...] = (
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
)


class PipelineBlocked(RuntimeError):
    """Raised when the full contracted run cannot be completed honestly."""

    def __init__(self, message: str, payload: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = dict(payload or {})


@dataclass(frozen=True)
class DailyPipelineConfig:
    run_id: str = DEFAULT_RUN_ID
    hours: int = 24
    output_dir: Path = DEFAULT_OUTPUT_ROOT
    proof_dir: Path = SPRINT_DIR
    run_root: Path = RUN_ROOT
    aemet_root: Path = DEFAULT_AEMET_ROOT
    score: bool = False
    restart_at_hour: int | None = None
    repeat: bool = False
    domain: str = "d02"
    dt_s: float = DT_S
    acoustic_substeps: int = 10
    radiation_cadence_steps: int = 180
    refresh_land_state_hourly: bool = True
    # v0.2.0 S6b: ACTIVATE prognostic Noah-MP over land. When True, the land
    # surface is the prognostic Noah-MP coupler (land EVOLVES inside the forecast
    # scan) instead of the prescribed bulk path -- and the hourly land snap
    # (``_refresh_hourly_land_state``) is DISABLED over land (the prognostic state
    # replaces the hourly corpus re-snap; re-snapping would clobber the evolved
    # land carry). The forecast must be advanced through the carry-threading
    # segmented entry so ``noahmp_land`` is threaded across the run (the v0.1.0
    # single-shot-per-hour ``run_forecast_operational`` path does NOT carry land
    # state across the hour boundary). See proofs/noahmp/s6b_activate_validate.py
    # and proofs/m20/tost_ensemble_runner.py for the carry-threading drivers.
    use_noahmp: bool = False
    # v0.2.0 wall-clock win #3: write each hour's wrfout on a background thread
    # while the GPU advances the next hour (double-buffered output). The device
    # ->host pull still happens synchronously on the main thread; only the NetCDF
    # write is overlapped, so output bytes/ordering are unchanged. Set False to
    # restore the fully-synchronous writer (e.g. for an A/B wall-clock baseline).
    async_output: bool = True
    # Optional WRF auxiliary-history (``auxhist``) secondary output streams. WRF
    # drives up to 24 INDEPENDENT auxhist streams (auxhist1..auxhist24), each with
    # its own interval / outname / frames / variable subset. This field accepts:
    #   * ``None`` (the default)      -> NO second stream; the forecast/output path
    #                                    is byte-for-byte the hourly-wrfout-only
    #                                    behaviour (OFF by default).
    #   * a single AuxhistStreamConfig -> one secondary stream (legacy form).
    #   * a list/tuple of configs      -> N secondary streams, each fired at its OWN
    #                                    interval (e.g. a 15-min surface subset AND a
    #                                    60-min full stream). The forecast hour is
    #                                    advanced at gcd(60, intervals)-minute
    #                                    granularity so every stream frame is genuine
    #                                    GPU output at its lead, never interpolated.
    # Use ``auxhist_streams`` for the normalized tuple. See gpuwrf.io.auxhist_stream.
    auxhist: AuxhistStreamConfig | Sequence[AuxhistStreamConfig] | None = None

    @property
    def auxhist_streams(self) -> tuple[AuxhistStreamConfig, ...]:
        """The configured auxhist streams as a normalized tuple (``()`` when OFF).

        Accepts the back-compatible single-config form and the multi-stream
        list form; enforces unique ``stream_id`` across the list.
        """

        return coerce_auxhist_streams(self.auxhist)


@dataclass(frozen=True)
class DailyCase:
    state: Any
    grid: Any
    namelist: Any
    run_start: datetime
    metadata: dict[str, Any]
    writer_diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ForecastSequenceResult:
    status: str
    run_id: str
    run_dir: Path
    output_dir: Path
    hours: int
    output_files: list[Path]
    per_hour_wall_s: list[float]
    final_state: Any
    run_start: datetime
    metadata: dict[str, Any]
    finite_summary: dict[str, Any]
    checkpoint: dict[str, Any] | None = None
    # Secondary WRF auxhist stream files (empty when no auxhist stream is configured).
    auxhist_files: list[Path] = field(default_factory=list)

    @property
    def final_wrfout(self) -> Path | None:
        return self.output_files[-1] if self.output_files else None

    @property
    def forecast_wall_s(self) -> float:
        return float(sum(self.per_hour_wall_s))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, set):
        return sorted(value)
    if pd.isna(value):
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def resolve_run_dir(run_id: str, run_root: str | Path = RUN_ROOT) -> Path:
    candidate = Path(run_id).expanduser()
    if candidate.is_dir():
        return candidate
    return Path(run_root) / run_id


def _coerce_run_start(value: str) -> datetime:
    text = value.strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _domain_namelist_value(run: Gen2Run, group: str, key: str, domain: str, default: Any) -> Any:
    value = run.namelist.get(group, {}).get(key, default)
    if isinstance(value, list):
        index = max(int(domain[1:]) - 1, 0)
        if index < len(value):
            return value[index]
        return value[-1] if value else default
    return value


def _lookup_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _static_latlon_expected_shapes(grid: Any | None) -> dict[str, tuple[int, int]]:
    nx = _lookup_attr(grid, "nx", None)
    ny = _lookup_attr(grid, "ny", None)
    projection = _lookup_attr(grid, "projection", None)
    if nx is None:
        nx = _lookup_attr(projection, "nx", None)
    if ny is None:
        ny = _lookup_attr(projection, "ny", None)
    if nx is None or ny is None:
        return {}
    nx_i = int(nx)
    ny_i = int(ny)
    return {
        "XLAT": (ny_i, nx_i),
        "XLONG": (ny_i, nx_i),
        "XLAT_U": (ny_i, nx_i + 1),
        "XLONG_U": (ny_i, nx_i + 1),
        "XLAT_V": (ny_i + 1, nx_i),
        "XLONG_V": (ny_i + 1, nx_i),
    }


def _read_static_latlon(path: Path, name: str) -> np.ndarray | None:
    with Dataset(path, "r") as dataset:
        if name not in dataset.variables:
            return None
        variable = dataset.variables[name]
        data = variable[0] if variable.dimensions and variable.dimensions[0] == "Time" else variable[:]
        return np.asarray(np.ma.filled(data, np.nan), dtype=np.float32)


def _candidate_static_latlon_paths(run: Gen2Run, domain: str) -> list[Path]:
    paths: list[Path] = []
    try:
        paths.append(run.wrfinput_file(domain))
    except (FileNotFoundError, ValueError):
        pass
    try:
        paths.extend(run.history_files(domain))
    except (FileNotFoundError, ValueError):
        pass
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = Path(path).resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(Path(path))
    return deduped


def _load_static_latlon_writer_diagnostics(
    run: Gen2Run,
    domain: str,
    *,
    grid: Any | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Load real WRF static lat/lon fields for host-only wrfout diagnostics.

    The payload is intentionally kept out of ``State``/``GridSpec``/``Namelist``:
    it is writer metadata, loaded once at case setup, and merged into the existing
    diagnostics map at output boundaries.
    """

    candidates = _candidate_static_latlon_paths(run, domain)
    expected_shapes = _static_latlon_expected_shapes(grid)
    fields: dict[str, np.ndarray] = {}
    meta_fields: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in WRITER_LATLON_FIELDS:
        for path in candidates:
            try:
                value = _read_static_latlon(path, name)
            except Exception as exc:  # noqa: BLE001 -- keep trying other static sources.
                errors[f"{path}:{name}"] = f"{type(exc).__name__}: {exc}"
                continue
            if value is None:
                continue
            expected = expected_shapes.get(name)
            if expected is not None and tuple(value.shape) != expected:
                raise ValueError(
                    f"{path}:{name} shape {tuple(value.shape)} does not match grid shape {expected}"
                )
            fields[name] = value.astype(np.float32, copy=False)
            finite = np.isfinite(value)
            meta_fields[name] = {
                "source_file": str(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite_fraction": float(finite.sum() / max(value.size, 1)),
                "min": float(np.nanmin(value)) if value.size else None,
                "max": float(np.nanmax(value)) if value.size else None,
            }
            break

    missing = [name for name in WRITER_LATLON_FIELDS if name not in fields]
    has_mass_pair = "XLAT" in fields and "XLONG" in fields
    status = "loaded" if has_mass_pair else ("missing_mass_pair" if candidates else "no_candidate_files")
    meta = {
        "status": status,
        "domain": str(domain),
        "candidate_files": [str(path) for path in candidates],
        "loaded_fields": sorted(fields),
        "missing_fields": missing,
        "fields": meta_fields,
        "errors": errors,
        "note": "host-only wrfout diagnostics; not part of State/GridSpec/OperationalNamelist",
    }
    return (fields if fields else None), meta


def _merge_output_diagnostics(*diagnostics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in diagnostics:
        if item:
            merged.update(item)
    return merged or None


def _build_real_case(config: DailyPipelineConfig) -> tuple[DailyCase, Path]:
    run_dir = resolve_run_dir(config.run_id, config.run_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run directory: {run_dir}")
    replay = build_replay_case(run_dir, domain=config.domain)
    # Seed the transitional legacy aliases (p/ph/mu) from the authoritative totals.
    # CRITICAL (donate-safety): ``State.replace(p=p_total, ...)`` aliases ``p`` to
    # the SAME device buffer as ``p_total`` -- a forecast entry compiled with
    # ``donate_argnums=(0,)`` then tries to donate one buffer twice and crashes
    # ("Attempt to donate the same buffer twice"). ``dealias_state_buffers`` makes
    # every leaf that shares a buffer a distinct copy, so the donate path is safe by
    # construction (and a second dedup guard lives at the JIT entry as a backstop).
    state = dealias_state_buffers(
        replay.state.replace(p=replay.state.p_total, ph=replay.state.ph_total, mu=replay.state.mu_total)
    )
    radiation_static, radiation_static_meta = load_radiation_static(
        replay.run,
        config.domain,
        grid=replay.grid,
        metrics=replay.metrics,
    )
    topo_shading = int(_domain_namelist_value(replay.run, "physics", "topo_shading", config.domain, 0))
    slope_rad = int(_domain_namelist_value(replay.run, "physics", "slope_rad", config.domain, 0))
    # Orographic gravity-wave drag (gwd_opt=1): read the case's own gwd_opt;
    # when on, build the per-run GWDOStatics from the geo_em sub-grid orography
    # (VAR/CON/OA1-4/OL1-4 in wrfinput) so the operational dispatch
    # (operational_mode, ``gwd_opt==1 AND gwdo_statics is not None``) actually
    # runs the drag.  gwd_opt=0 (default) or absent statics keep GWD a no-op.
    # ``gwd_opt`` is a WRF &dynamics control (module_check_a_mundo / Registry);
    # fall back to &physics for namelists that place it there.
    gwd_opt = int(_domain_namelist_value(replay.run, "dynamics", "gwd_opt", config.domain, 0))
    if gwd_opt == 0:
        gwd_opt = int(_domain_namelist_value(replay.run, "physics", "gwd_opt", config.domain, 0))
    gwdo_statics = None
    gwdo_static_meta: dict[str, Any] = {"requested": bool(gwd_opt == 1)}
    # v0.12.0: GWD operational coupling is GATED OFF BY DEFAULT on the operational
    # path.  In v0.12.0 this was gated OFF by default because the 24h nested-1km + GWD
    # run exceeded the single-GPU fp64 VRAM ceiling (~28 GB) at ~sim-hr 7.  v0.13's
    # RRTMG g-point + optics/taumol chunking cut SW peak VRAM -88.6% / LW -43.6%, so the
    # 24h nested-1km + GWD now FITS and runs GREEN (proofs/v013/gwd_nested_24h_gate.json:
    # PIPELINE_GREEN, 24/24 wrfout, all-finite, ~6-18 GB peak).  GWD is therefore ENABLED
    # BY DEFAULT now (gwd_opt=1 in a real namelist is honoured); set GPUWRF_GWD_NESTED=0
    # to force it off for a memory-tighter config.  Kernel oracle-validated (fp64 ~1e-13).
    _gwd_enabled = os.environ.get("GPUWRF_GWD_NESTED", "1") != "0"
    if gwd_opt == 1 and not _gwd_enabled:
        gwdo_static_meta = {
            "requested": True,
            "status": "force_disabled_env",
            "reason": (
                "GWD operational coupling is ON by default in v0.13 (24h nested-1km + "
                "GWD fits after the RRTMG optics/taumol VRAM chunking); forced off here "
                "via GPUWRF_GWD_NESTED=0."
            ),
        }
        gwd_opt = 0
    if gwd_opt == 1:
        try:
            gwdo_statics, gwdo_static_meta = load_gwdo_statics(
                replay.run, config.domain, grid=replay.grid, metrics=replay.metrics
            )
            gwdo_static_meta["requested"] = True
        except Exception as exc:  # noqa: BLE001 -- GWD is opt-in; never block init.
            gwdo_statics = None
            gwdo_static_meta = {"requested": True, "status": "error", "reason": str(exc)}
        # Fail closed: if the statics could not be built, demote to a no-op so the
        # run does not silently claim GWD that the kernel cannot apply.
        if gwdo_statics is None:
            gwd_opt = 0
    # Sprint U (P0-1): the real-case operational path MUST use the same F7-closed
    # dycore operators that the idealized gates use, not the pre-F7 primitive
    # advection path.  Concretely:
    #   * use_flux_advection=True  -> WRF flux-form mass-coupled advect_u/v/w +
    #     advect_scalar (incl. the F7N vertical-momentum sign fix), NOT the legacy
    #     primitive compute_advection_tendencies path.
    #   * force_fp64=True          -> the F7 gates require fp64 for the acoustic
    #     solve (fp32 loses the perturbation and detonates; perf downcast is a
    #     later gated decision, ADR-007).
    #   * diff_6th_opt=2 / 0.12    -> the operational d02 numerical filter (WRF
    #     diff_6th_opt=2, diff_6th_factor=0.12) that suppresses 2dx noise.
    #   * w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2 -> the Gen2 d02 WRF
    #     upper-level Rayleigh + vertical-velocity damping.
    #   * epssm=0.5                -> MATCH the real Gen2 d02 namelist.input
    #     (&dynamics epssm=0.5).  The dataclass default (0.1) is WRF's generic
    #     default, but the operational d02 run off-centres the vertically-implicit
    #     acoustic solve at 0.5 (eps_m=1-epssm weights the explicit old-time
    #     pressure/w in advance_w / calc_coef_w; cof=(0.5*dts*g*(1+epssm))^2,
    #     module_small_step_em.F:624,646).  Leaving it at 0.1 under-damps the
    #     vertical acoustic / top-face mode on the real init.
    #   * top_lid=True             -> RIGID lid at the model top.  Root-cause
    #     (proofs/dycore_realinit/): with top_lid=False (open top) the WRF-faithful
    #     advance_w top-face equation (module_small_step_em.F:1421-1429) produces a
    #     ~300 m/s spurious w on the model-top face on the very first step from the
    #     real d02 upper-level state; with run_boundary on, the lateral relaxation
    #     re-feeds that top-corner w mode faster than damp_opt=3 removes it, and the
    #     horizontal u-mode runs away to ~200 m/s.  top_lid=True (w(kde)=0, the SAME
    #     top BC every idealized validation gate used) zeroes the top face and the
    #     dycore-only + real-boundary run is STABLE for the full hour (|w|~14,
    #     |u|~31, theta physical over 360 steps).  proofs/dycore_realinit/
    #     step4_fix_longrun.json + step2_bc_isolation.json + step5_opentop_bndy.json.
    # Guards stay ON for the production real path (the operational safety net); the
    # guards-off stability of the dycore itself is proven separately (Sprint U P1-6).
    namelist = OperationalNamelist.from_grid(
        replay.grid,
        tendencies=replay.tendencies,
        metrics=replay.metrics,
        dt_s=float(config.dt_s),
        acoustic_substeps=int(config.acoustic_substeps),
        radiation_cadence_steps=int(config.radiation_cadence_steps),
        use_vertical_solver=True,
        use_flux_advection=True,
        force_fp64=True,
        diff_6th_opt=2,
        diff_6th_factor=0.12,
        w_damping=1,
        damp_opt=3,
        zdamp=5000.0,
        dampcoef=0.2,
        epssm=0.5,
        # v0.14 WRF-native advance_w oracle (proofs/v014/wrf_native_advance_w_dump):
        # the WRF truth runs an OPEN top (config top_lid F); the rigid lid zeroes
        # the top-face rhs/w of the implicit (w,phi) solve and the Thomas back-
        # substitution carries that error down the whole column (~half the h36
        # stage ph error once rw_tend is fixed).  The 2026-05-30 open-top
        # instability that motivated top_lid=True predates the F7J/F7H/v0.14
        # acoustic fixes; the h36 stage gate + short forecast now run stably open
        # (see the sprint review).  GPUWRF_TOP_LID=1 restores the rigid lid for
        # bisection.
        top_lid=os.environ.get("GPUWRF_TOP_LID", "0") == "1",
        # WRF Registry default hypsometric_opt=2 (LOG form) -- every real case runs
        # with it; the replay metrics carry the true wrfinput C3F/C4F/C3H/C4H/P_TOP.
        # v0.14 Switzerland h36 mass-venting root cause was the linear (1) form.
        hypsometric_opt=2,
        # v0.14 acoustic continuation: the case's WRF &dynamics h_sca_adv_order
        # (Registry default 5) drives rhs_ph's real-case horizontal phi
        # advection; the legacy hardwired order-2/periodic operator broke the
        # ~65:1 horizontal/vertical cancellation over steep terrain.
        h_sca_adv_order=int(_domain_namelist_value(replay.run, "dynamics", "h_sca_adv_order", config.domain, 5)),
        # v0.14 stage3/wrapper-cadence sprint: WRF SPECIFIED-domain per-stage
        # relax + per-substep spec boundary cadence (operational_mode docstring).
        # Opt-in via env until the hourly venting gate proves it for default-on.
        specified_bdy_cadence=os.environ.get("GPUWRF_SPECIFIED_BDY_CADENCE", "0") == "1",
        # candidate A: WRF specified-boundary advection-order degradation for
        # the flux-form operators (opt-in for the same gate).
        specified_adv_degrade=os.environ.get("GPUWRF_SPECIFIED_ADV_DEGRADE", "0") == "1",
        radiation_static=radiation_static,
        topo_shading=topo_shading,
        slope_rad=slope_rad,
        topo_shadow_length_m=float(
            _domain_namelist_value(replay.run, "physics", "shadlen", config.domain, 25000.0)
        ),
        gwd_opt=gwd_opt,
        gwdo_statics=gwdo_statics,
        # v0.14 venting-residual fix (proofs/v014/switzerland_uv_lane_decomposition
        # + _contributors): the WRF-native advance_uv oracle proved the staged
        # ru/rv_tend and t_tend were missing the ENTIRE physics *_tendf fold
        # (PBL drag 57%/72% rel of ru/rv_tend, RTHRATEN+RTHBLTEN 54% of t_tend)
        # because rad_rk_tendf defaulted 0 (legacy lumped/post-dycore cadence)
        # and _dry_physics_tendencies_from_state_delta is a deliberate empty
        # bridge.  WRF integrates the physics sources inside the RK/acoustic
        # loop (rk_addtend_dry); route them the same way by default on the real
        # case.  GPUWRF_PHYS_RK_TENDF=0 restores the legacy cadence for
        # bisection.
        rad_rk_tendf=0 if os.environ.get("GPUWRF_PHYS_RK_TENDF", "1") == "0" else 1,
        # Same sprint: honour the case's own WRF &dynamics horizontal-diffusion
        # config (Switzerland/Gen2 run diff_opt=1, km_opt=4 2-D Smagorinsky; the
        # JAX side ran 0/0, dropping the Smag u/v/theta/w tendencies WRF folds
        # into the same large-step lane).  GPUWRF_SMAG2D=0 forces it off.
        diff_opt=(
            0
            if os.environ.get("GPUWRF_SMAG2D", "1") == "0"
            else int(_domain_namelist_value(replay.run, "dynamics", "diff_opt", config.domain, 0))
        ),
        km_opt=(
            0
            if os.environ.get("GPUWRF_SMAG2D", "1") == "0"
            else int(_domain_namelist_value(replay.run, "dynamics", "km_opt", config.domain, 0))
        ),
    )
    run_start = _coerce_run_start(str(replay.metadata["run_start_label"]))
    writer_diagnostics, writer_static_latlon_meta = _load_static_latlon_writer_diagnostics(
        replay.run, config.domain, grid=replay.grid
    )
    metadata = {
        "run_id": replay.metadata.get("run_id"),
        "run_dir": str(run_dir),
        "domain": config.domain,
        "grid": replay.metadata.get("grid", {}),
        "boundary": replay.metadata.get("boundary", {}),
        "namelist": {
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "rk_order": int(namelist.rk_order),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
            # Sprint U (P0-1): F7 dycore operators on the real-case operational path.
            "use_flux_advection": bool(namelist.use_flux_advection),
            "force_fp64": bool(namelist.force_fp64),
            "diff_6th_opt": int(namelist.diff_6th_opt),
            "diff_6th_factor": float(namelist.diff_6th_factor),
            "w_damping": int(namelist.w_damping),
            "damp_opt": int(namelist.damp_opt),
            "zdamp": float(namelist.zdamp),
            "dampcoef": float(namelist.dampcoef),
            "top_lid": bool(namelist.top_lid),
            "disable_guards": bool(namelist.disable_guards),
            "topo_shading": int(namelist.topo_shading),
            "slope_rad": int(namelist.slope_rad),
            "topo_shadow_length_m": float(namelist.topo_shadow_length_m),
            "gwd_opt": int(namelist.gwd_opt),
        },
        "radiation_static": radiation_static_meta,
        "gwdo_static": gwdo_static_meta,
        "writer_static_latlon": writer_static_latlon_meta,
        "source": "gpuwrf.integration.d02_replay.build_replay_case",
        # Propagate the auto-detected init mode (replay vs standalone native-init)
        # so the forecast driver and the hourly land-refresh gate can branch on it.
        "standalone_native_init": bool(replay.metadata.get("standalone_native_init", False)),
        "init_mode": "standalone_native_init"
        if replay.metadata.get("standalone_native_init", False)
        else "cpu_wrf_replay",
    }
    return (
        DailyCase(
            state=state,
            grid=replay.grid,
            namelist=namelist,
            run_start=run_start,
            metadata=metadata,
            writer_diagnostics=writer_diagnostics,
        ),
        run_dir,
    )


def _field_items(obj: Any) -> Iterable[tuple[str, Any]]:
    slots = getattr(obj, "__slots__", ())
    if slots:
        for name in slots:
            if hasattr(obj, name):
                yield str(name), getattr(obj, name)
        return
    if hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            yield str(name), value


def finite_summary(state: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    all_finite = True
    for name, value in _field_items(state):
        try:
            array = np.asarray(value)
        except Exception:
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        finite = np.isfinite(array)
        ok = bool(finite.all())
        all_finite = all_finite and ok
        fields[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": ok,
            "nonfinite_count": int(array.size - int(finite.sum())),
            "min": float(np.nanmin(array)) if array.size else None,
            "max": float(np.nanmax(array)) if array.size else None,
        }
    return {"all_finite": bool(all_finite), "field_count": int(len(fields)), "fields": fields}


_ALL_FINITE_DEVICE_FN = None


def _all_finite_device(arrays: tuple):
    """ONE on-device all-finite reduction over the given float arrays.

    Lowers to per-array ``isfinite``+``all`` reduces fused by XLA; only a single
    boolean scalar crosses the device->host boundary (vs ``finite_summary``'s
    full-state pull). Pure read -- touches no state value. The jit is built
    lazily so this module keeps its jax-free import surface.
    """

    global _ALL_FINITE_DEVICE_FN
    if _ALL_FINITE_DEVICE_FN is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _fn(arrs):
            flags = [jnp.isfinite(a).all() for a in arrs]
            return jnp.stack(flags).all() if flags else jnp.asarray(True)

        _ALL_FINITE_DEVICE_FN = _fn
    return _ALL_FINITE_DEVICE_FN(arrays)


def finite_guard_summary(state: Any) -> dict[str, Any]:
    """``finite_summary`` with a device-side fast path (v0.15 Stream-A).

    The per-hour pipeline guards only consume ``summary["all_finite"]`` on the
    success path, but ``finite_summary`` pulls EVERY state leaf to host (a full
    device->host state transfer 2x/forecast-hour) just to compute it. This
    wrapper first runs one jitted on-device all-finite reduce (one bool scalar
    transferred). Only when that reduce reports a non-finite value -- or the
    device path cannot run -- does it fall back to the full host
    ``finite_summary``, so every FAILURE payload remains byte-identical
    (fail-closed semantics and report content unchanged). Identity: the model
    state is never modified; success-path consumers read only ``all_finite``.
    """

    try:
        import jax

        device_arrays = []
        host_ok = True
        n_fields = 0
        for _, value in _field_items(state):
            if isinstance(value, jax.Array):
                if np.issubdtype(np.dtype(value.dtype), np.inexact):
                    device_arrays.append(value)
                    n_fields += 1
                elif np.issubdtype(np.dtype(value.dtype), np.number):
                    n_fields += 1  # integer leaves are always finite
                continue
            # host-resident leaves (numpy/scalars): check here, no transfer needed
            try:
                array = np.asarray(value)
            except Exception:
                continue
            if not np.issubdtype(array.dtype, np.number):
                continue
            n_fields += 1
            if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
                host_ok = False
        if not device_arrays and host_ok:
            return finite_summary(state)
        ok = host_ok and bool(_all_finite_device(tuple(device_arrays)))
    except Exception:  # noqa: BLE001 -- guard must fail closed to the host path
        return finite_summary(state)
    if not ok:
        return finite_summary(state)
    return {"all_finite": True, "field_count": int(n_fields), "fields": {}}


def _wrfout_name(valid_time: datetime, domain: str) -> str:
    return f"wrfout_{domain}_{valid_time:%Y-%m-%d_%H:%M:%S}"


def _auxhist_substeps_per_hour(config: DailyPipelineConfig) -> int:
    """Number of equal sub-hour forecast segments needed per forecast hour.

    With no auxhist stream the loop steps a full hour at a time (1 segment), so the
    main-wrfout path is byte-for-byte the existing hourly behaviour. With one OR
    MORE auxhist streams the loop steps at ``gcd(60, i1, i2, ...)``-minute
    granularity so BOTH the hourly main-wrfout boundary and every stream's
    interval boundary land on a real model-state snapshot -- the auxhist frames are
    genuine GPU output, never interpolated/fabricated. Delegates to the shared
    multi-stream cadence helper in :mod:`gpuwrf.io.auxhist_stream`.
    """

    return auxhist_substeps_per_hour(config.auxhist_streams)


def _emit_auxhist_frame(
    *,
    stream: AuxhistStreamConfig,
    config: DailyPipelineConfig,
    state: Any,
    case: DailyCase,
    output_dir: Path,
    lead_minutes: float,
    diagnostics: Mapping[str, Any] | None,
    writer: AsyncWrfoutWriter | None,
    prepared: Any | None = None,
) -> Path | None:
    """Write one frame for ``stream`` if ``lead_minutes`` is its output boundary.

    Reuses the main wrfout writer in stream-generic mode: it materializes the
    payload once (host numpy) -- or REUSES the supplied ``prepared`` payload when
    the caller already pulled this state to host for another stream / the main
    wrfout (so concurrent streams at the same boundary share ONE device->host
    pull) -- and writes ONLY ``stream``'s variable subset to its WRF-named file.
    Returns the written path, or ``None`` when this lead is not ``stream``'s
    boundary.
    """

    if not stream.fires_at(lead_minutes):
        return None
    valid_time = case.run_start + timedelta(minutes=float(lead_minutes))
    aux_path = output_dir / stream.filename(valid_time, config.domain)
    output_diagnostics = _merge_output_diagnostics(case.writer_diagnostics, diagnostics)
    if prepared is None:
        prepared = prepare_wrfout_payload(
            state,
            case.grid,
            case.namelist,
            aux_path,
            valid_time=valid_time,
            lead_hours=float(lead_minutes) / 60.0,
            run_start=case.run_start,
            diagnostics=output_diagnostics,
        )
    subset = stream.variable_subset
    if writer is not None:
        writer.submit_subset(prepared, variable_subset=subset, target=aux_path)
    else:
        write_prepared_wrfout(prepared, variable_subset=subset, target_override=aux_path)
    return aux_path


def _default_forecast_fn(state: Any, namelist: Any, hours: float) -> Any:
    # Donate-safe by construction: run_forecast_operational de-aliases the State's
    # shared buffers before the donate JIT boundary.
    result = run_forecast_operational(
        _commit_to_operational_device(state), namelist, float(hours)
    )
    block_until_ready(result)
    return result


def _segmented_forecast_fn(state: Any, namelist: Any, hours: float) -> Any:
    """Standalone native-init forecast driver.

    Uses ``run_forecast_operational_segmented`` -- the long-run-safe, NO-outer-donate
    entry the v0.4.0 native-init forecast gate proved (proofs/v040/, proofs/perf/
    segscan_equiv.json). Keeps compile O(segment) and peak GPU memory bounded
    regardless of forecast length, and carries State across host-loop segments.
    """

    result = run_forecast_operational_segmented(
        _commit_to_operational_device(state), namelist, float(hours)
    )
    block_until_ready(result)
    return result


# Surface fields the operational M9 diagnostics supply to the wrfout writer.
# Each entry maps a wrfout variable name to its M9Diagnostics attribute. The
# writer keeps its own default for any field omitted or returned as None.
_M9_OUTPUT_FIELDS: tuple[tuple[str, str], ...] = (
    ("T2", "t2"),
    ("U10", "u10"),
    ("V10", "v10"),
    ("Q2", None),  # M9Diagnostics carries no q2; resolved from surf below.
    ("PSFC", "psfc"),
    ("SWDOWN", "swdown"),
    ("GLW", "glw"),
    ("PBLH", "pblh"),
    ("TSK", "tsk"),
    # --- B1 (v0.12.0) RRTMG up/down all-sky surface + TOA flux diagnostics. ---
    # All-sky only: the RRTMG port runs no separate clear-sky pass, so the WRF
    # clear-sky ``...C`` flux vars are deliberately absent (no fabrication). OLR
    # (== LWUPT) is derived in the writer from LWUPT. COSZEN is NOT routed here --
    # the writer already emits it from its own ``_compute_coszen`` solar-geometry
    # path; the M9Diagnostics.coszen leaf carries the same value for internal use.
    ("SWDNB", "swdnb"),
    ("SWUPB", "swupb"),
    ("LWDNB", "lwdnb"),
    ("LWUPB", "lwupb"),
    ("SWDNT", "swdnt"),
    ("SWUPT", "swupt"),
    ("LWDNT", "lwdnt"),
    ("LWUPT", "lwupt"),
    ("SWNORM", "swnorm"),
)


def _surface_diagnostics_for_output(
    state: Any, namelist: Any, run_start: Any, *, lead_seconds: float
) -> dict[str, np.ndarray] | None:
    """Recompute the operational surface map for the wrfout writer.

    Returns a ``{wrfout_name: numpy_array}`` mapping of the physically diagnosed
    surface fields (mass-point ``(ny, nx)``, K / m s^-1 / Pa / W m^-2) so the
    writer overrides its raw lowest-level fallbacks. Mirrors the d02-validate
    diagnostic source exactly (``compute_m9_diagnostics``). ``time_utc`` is
    threaded onto the namelist so SWDOWN/GLW follow the real forecast diurnal
    clock; ``lead_seconds`` is the elapsed forecast time at this output cadence.

    Degrades to ``None`` (writer keeps its defaults, no regression) when the
    namelist is not an operational namelist with a ``grid`` -- e.g. dict-driven
    or synthetic test callers.
    """

    if not hasattr(namelist, "grid"):
        return None
    import jax  # local import: keeps module import light for non-GPU callers.

    clock_namelist = namelist
    if getattr(namelist, "time_utc", None) is None and hasattr(namelist, "__dataclass_fields__"):
        clock_namelist = replace(namelist, time_utc=run_start)
    try:
        m9 = compute_m9_diagnostics(state, clock_namelist, lead_seconds)
    except Exception:  # noqa: BLE001 -- diagnostics are best-effort; never block output.
        return None

    # Q2 (2-m mixing ratio) is a surface-layer field; M9Diagnostics does not
    # expose it, so source it directly from surface_layer_diagnostics.
    q2 = None
    try:
        from gpuwrf.runtime.operational_mode import surface_layer_diagnostics

        q2 = getattr(surface_layer_diagnostics(state, clock_namelist.grid), "q2", None)
    except Exception:  # noqa: BLE001
        q2 = None

    out: dict[str, np.ndarray] = {}
    for wrf_name, attr in _M9_OUTPUT_FIELDS:
        value = q2 if wrf_name == "Q2" else (getattr(m9, attr, None) if attr else None)
        if value is None:
            continue
        out[wrf_name] = np.asarray(jax.device_get(value))
    return out or None


# Every State lateral-boundary leaf with a leading time axis. Kept in sync with
# gpuwrf.contracts.state.State (the *_bdy block); a missing/None leaf is skipped.
_BOUNDARY_LEAF_FIELDS: tuple[str, ...] = (
    "u_bdy",
    "v_bdy",
    "theta_bdy",
    "qv_bdy",
    "ph_bdy",
    "mu_bdy",
    "w_bdy",
    "p_bdy",
    "pb_bdy",
    "phb_bdy",
    "mub_bdy",
)


def _boundary_window_cadence_s(namelist: Any) -> float:
    config = getattr(namelist, "boundary_config", None)
    return float(getattr(config, "update_cadence_s", 3600.0) or 3600.0)


def _capture_boundary_leaves(state: Any, namelist: Any) -> dict[str, np.ndarray]:
    """Pull the full-time-axis lateral-boundary leaves to host, once per run.

    Returns ``{}`` (windowing disabled, loop unchanged) for cases without an
    active lateral boundary or without a pytree ``State`` -- idealized runs and
    the synthetic dict/SimpleNamespace test cases.
    """

    if not bool(getattr(namelist, "run_boundary", False)) or not hasattr(state, "replace"):
        return {}
    import jax  # local import: keeps module import light for non-GPU callers.

    leaves: dict[str, np.ndarray] = {}
    for name in _BOUNDARY_LEAF_FIELDS:
        leaf = getattr(state, name, None)
        if leaf is None:
            continue
        array = np.asarray(jax.device_get(leaf))
        if array.ndim < 2 or array.shape[0] < 1:
            continue
        leaves[name] = array
    return leaves


def _boundary_leaf_value_at(leaf: np.ndarray, t_s: float, record_cadence_s: float) -> np.ndarray:
    """Evaluate one boundary leaf at global forecast time ``t_s``.

    Host mirror of :func:`gpuwrf.coupling.boundary_apply.interpolate_boundary_leaf`
    (piecewise-linear along the leading time axis, clamped at both ends). Exact
    record levels are returned by reference so integer-hour windows stay
    bit-identical to the stored leaf levels.
    """

    max_index = int(leaf.shape[0]) - 1
    lead_index = float(t_s) / float(record_cadence_s)
    lower = min(max(int(math.floor(lead_index)), 0), max_index)
    upper = min(lower + 1, max_index)
    alpha = min(max(lead_index - float(lower), 0.0), 1.0)
    if alpha == 0.0 or lower == upper:
        return leaf[lower]
    return (leaf[lower] * (1.0 - alpha) + leaf[upper] * alpha).astype(leaf.dtype)


def _rewindow_boundary_leaves(
    state: Any,
    full_leaves: Mapping[str, np.ndarray],
    *,
    segment_start_s: float,
    record_cadence_s: float,
    window_s: float,
) -> Any:
    """Re-anchor the State's lateral-boundary leaves to the current segment.

    ``run_forecast_operational`` (and the segmented variant) restart their step
    clock at 1 on EVERY call, so the in-scan ``interpolate_boundary_leaf`` walk
    always begins at lead 0. Driving the hourly loop with the full-time-axis
    leaves therefore re-forces the lateral boundary toward leaf level 1 each
    hour -- the boundary stays pinned at the hour-1 value for the whole run
    (the v0.14 Switzerland d01 72h FAIL: GPU spec-zone ring == CPU truth h01
    bit-exact at every lead through h72).

    The fix: before each segment, swap in a 2-level leaf holding the exact
    piecewise-linear boundary values at global ``segment_start_s`` and
    ``segment_start_s + window_s``. The in-call walk (cadence ``window_s``)
    then reproduces the true global-time forcing exactly: record times are
    multiples of 3600 s and segment lengths divide 3600 s, so no record kink
    can fall inside the consumed sub-window. Hour 1 keeps the previously
    validated forcing bit-identical (window == leaf levels 0 and 1).
    """

    if not full_leaves:
        return state
    updates = {
        name: np.stack(
            (
                _boundary_leaf_value_at(leaf, segment_start_s, record_cadence_s),
                _boundary_leaf_value_at(leaf, segment_start_s + window_s, record_cadence_s),
            ),
            axis=0,
        )
        for name, leaf in full_leaves.items()
    }
    return state.replace(**updates)


def _refresh_hourly_land_state(
    state: Any, run_dir: Path, domain: str, hour: int, *, use_noahmp: bool = False
) -> tuple[Any, dict[str, Any]]:
    if not hasattr(state, "replace"):
        return state, {"status": "SKIPPED", "reason": "state has no replace method", "hour": int(hour)}
    land = load_hourly_land_state(Gen2Run(run_dir), domain=domain, time=int(hour))
    top_soil = land.soil_moisture[0] if getattr(land.soil_moisture, "ndim", 0) == 3 else land.soil_moisture
    updates = {
        "t_skin": land.t_skin,
        "soil_moisture": top_soil,
        "xland": land.xland,
        "lakemask": land.lakemask,
        "mavail": land.mavail,
        "roughness_m": land.roughness_m,
    }
    if use_noahmp:
        # v0.2.0 S6b: prognostic Noah-MP owns the LAND tile (it evolves inside the
        # scan). Re-snapping land would clobber that. Only WATER columns keep the
        # prescribed-SST hourly refresh; over land the current (prognostic) values
        # are preserved via where(is_water, corpus, current).
        import jax.numpy as _jnp
        is_water = _jnp.asarray(land.xland) >= 1.5
        masked = {}
        for name, corpus_val in updates.items():
            if name in ("xland", "lakemask"):
                masked[name] = corpus_val  # static masks: identical, safe to refresh
                continue
            if not hasattr(state, name):
                continue
            cur = _jnp.asarray(getattr(state, name))
            cv = _jnp.asarray(corpus_val)
            if cv.shape == cur.shape and is_water.shape == cur.shape:
                masked[name] = _jnp.where(is_water, cv, cur)
            else:
                masked[name] = cur  # shape mismatch -> preserve prognostic
        available = {name: value for name, value in masked.items() if hasattr(state, name)}
        refreshed = state.replace(**available)
        return refreshed, {
            "status": "PASS", "hour": int(hour), "mode": "noahmp_water_only",
            "source_file": land.source.get("source_file"),
            "fields_refreshed_water_only": sorted(available),
            "note": "prognostic Noah-MP owns LAND; only water SST/roughness re-snapped",
        }
    available = {name: value for name, value in updates.items() if hasattr(state, name)}
    refreshed = state.replace(**available)
    return refreshed, {
        "status": "PASS",
        "hour": int(hour),
        "source_file": land.source.get("source_file"),
        "requested_time_index": land.source.get("requested_time_index"),
        "time_index": land.source.get("time_index"),
        "fields_refreshed": sorted(available),
        "full_hourly_payload": ["TSK", "SST", "SMOIS", "SH2O", "TSLB"],
    }


def _run_forecast_sequence(
    config: DailyPipelineConfig,
    *,
    output_dir: Path,
    checkpoint_at_hour: int | None = None,
    forecast_fn: Callable[[Any, Any, float], Any] = _default_forecast_fn,
    case_builder: Callable[[DailyPipelineConfig], tuple[DailyCase, Path]] = _build_real_case,
) -> ForecastSequenceResult:
    if int(config.hours) <= 0:
        raise ValueError("--hours must be positive")
    if checkpoint_at_hour is not None and not (0 < int(checkpoint_at_hour) < int(config.hours)):
        raise ValueError("--restart-at-hour must be between 1 and hours - 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    case, run_dir = case_builder(config)
    state = case.state
    # Lateral-boundary clock fix (v0.14 Switzerland d01): hold the full-time-axis
    # *_bdy leaves on host and feed every forecast segment a 2-level window
    # anchored at the segment's GLOBAL start time (see _rewindow_boundary_leaves).
    # ``record_cadence_s`` is the true spacing of the stored leaf levels: hourly
    # wrfout history for replay; ``interval_seconds`` for native-init wrfbdy
    # leaves (whose 6x/3x-fast consumption on this path is fixed by the same
    # windowing). ``window_s`` is the in-call walk cadence.
    boundary_leaves = _capture_boundary_leaves(state, case.namelist)
    boundary_window_s = _boundary_window_cadence_s(case.namelist)
    boundary_record_cadence_s = float(
        (case.metadata.get("boundary") or {}).get("interval_seconds") or boundary_window_s
    )
    files: list[Path] = []
    auxhist_files: list[Path] = []
    per_hour_wall_s: list[float] = []
    checkpoint_payload: dict[str, Any] | None = None
    land_refresh_records: list[dict[str, Any]] = []
    # Configured auxhist streams (empty tuple == OFF). Each stream fires at its OWN
    # interval; multiple streams may fire at the same boundary (e.g. a 15-min and a
    # 60-min stream both fire at :00) and then share ONE device->host pull.
    streams = config.auxhist_streams
    # Sub-hour stepping granularity. 1 segment/hour (a full-hour advance) unless an
    # auxhist stream needs finer cadence, in which case the hour is advanced in
    # equal gcd(60, intervals)-minute segments so each auxhist frame is GENUINE
    # GPU output at that lead time (never interpolated). With no auxhist this is 1,
    # so the advance/output path is byte-for-byte the existing hourly behaviour.
    substeps = _auxhist_substeps_per_hour(config)
    segment_minutes = 60.0 / substeps
    # Per-stream emitted-file lists for the run metadata (the flat ``auxhist_files``
    # combines all streams for back-compat / the result object).
    auxhist_files_by_stream: dict[int, list[Path]] = {int(s.stream_id): [] for s in streams}
    # win #3: double-buffered output -- write hour N's wrfout on a background
    # thread while the GPU advances hour N+1. The device->host pull stays on the
    # main thread (prepare_wrfout_payload) so no device buffer is read off-thread;
    # only the NetCDF write is overlapped. join()ed below before any wrfout is read.
    writer = AsyncWrfoutWriter() if bool(config.async_output) else None

    def _flush_writer() -> None:
        # Drain/join the background writer so all submitted wrfouts are on disk
        # (and any writer error is surfaced) before the run returns OR fails. The
        # PipelineBlocked failure paths call this so partial output is flushed and
        # ``output_files_before_failure`` is accurate. join() is idempotent.
        if writer is not None:
            writer.join()

    def _emit_streams_at(
        lead_minutes: float,
        *,
        cur_state: Any,
        diagnostics: Mapping[str, Any] | None,
        prepared: Any | None = None,
    ) -> None:
        # Emit one frame for EVERY configured stream whose interval boundary falls
        # on ``lead_minutes``. All firing streams share the single ``prepared``
        # host payload (materialized once below) so concurrent streams cost no
        # extra device->host pull. Appends each written path to the flat list and
        # the per-stream metadata list.
        firing = [s for s in streams if s.fires_at(lead_minutes)]
        if not firing:
            return
        shared = prepared
        if shared is None:
            valid_time = case.run_start + timedelta(minutes=float(lead_minutes))
            shared = prepare_wrfout_payload(
                cur_state,
                case.grid,
                case.namelist,
                output_dir / firing[0].filename(valid_time, config.domain),
                valid_time=valid_time,
                lead_hours=float(lead_minutes) / 60.0,
                run_start=case.run_start,
                diagnostics=_merge_output_diagnostics(case.writer_diagnostics, diagnostics),
            )
        for stream in firing:
            aux_path = _emit_auxhist_frame(
                stream=stream, config=config, state=cur_state, case=case,
                output_dir=output_dir, lead_minutes=lead_minutes,
                diagnostics=diagnostics, writer=writer, prepared=shared,
            )
            if aux_path is not None:
                auxhist_files.append(aux_path)
                auxhist_files_by_stream[int(stream.stream_id)].append(aux_path)

    for hour in range(1, int(config.hours) + 1):
        start = time.perf_counter()
        # Advance the forecast across this hour in ``substeps`` equal segments. The
        # common case (no auxhist) is a single full-hour advance, identical to the
        # legacy path. When an auxhist stream is configured, each intermediate
        # segment boundary produces a genuine sub-hour model state at which the
        # auxhist frame (a variable subset) can be written.
        for seg in range(1, substeps + 1):
            if boundary_leaves:
                segment_start_s = ((hour - 1) * 60.0 + (seg - 1) * segment_minutes) * 60.0
                state = _rewindow_boundary_leaves(
                    state,
                    boundary_leaves,
                    segment_start_s=segment_start_s,
                    record_cadence_s=boundary_record_cadence_s,
                    window_s=boundary_window_s,
                )
            state = forecast_fn(state, case.namelist, segment_minutes / 60.0)
            if seg == substeps:
                break  # the hour boundary is handled by the main path below
            lead_minutes = (hour - 1) * 60.0 + seg * segment_minutes
            if any(s.fires_at(lead_minutes) for s in streams):
                seg_summary = finite_guard_summary(state)
                if not seg_summary["all_finite"]:
                    _flush_writer()
                    raise PipelineBlocked(
                        f"nonfinite model state at auxhist lead {lead_minutes:g} min",
                        {
                            "failure_mode": "NONFINITE_STATE",
                            "failed_lead_minutes": float(lead_minutes),
                            "finite_summary": seg_summary,
                            "output_files_before_failure": [str(path) for path in files],
                        },
                    )
                aux_surface_diag = _surface_diagnostics_for_output(
                    state, case.namelist, case.run_start, lead_seconds=lead_minutes * 60.0
                )
                aux_diag = _merge_output_diagnostics(case.writer_diagnostics, aux_surface_diag)
                _emit_streams_at(lead_minutes, cur_state=state, diagnostics=aux_diag)
        elapsed = time.perf_counter() - start
        per_hour_wall_s.append(float(elapsed))

        summary = finite_guard_summary(state)
        if not summary["all_finite"]:
            _flush_writer()
            raise PipelineBlocked(
                f"nonfinite model state after forecast hour {hour}",
                {
                    "failure_mode": "NONFINITE_STATE",
                    "failed_hour": int(hour),
                    "finite_summary": summary,
                    "output_files_before_failure": [str(path) for path in files],
                },
            )

        # Hourly land-state refresh reads CPU-WRF wrfout history; the standalone
        # native-init path has NONE, so it is skipped there (the prescribed wrfinput
        # land state stands, and prognostic Noah-MP evolves it if enabled).
        if (
            bool(config.refresh_land_state_hourly)
            and case.metadata.get("source") == "gpuwrf.integration.d02_replay.build_replay_case"
            and not bool(case.metadata.get("standalone_native_init"))
        ):
            state, land_record = _refresh_hourly_land_state(
                state, run_dir, config.domain, hour, use_noahmp=bool(config.use_noahmp))
            land_refresh_records.append(land_record)
            summary = finite_guard_summary(state)
            if not summary["all_finite"]:
                _flush_writer()
                raise PipelineBlocked(
                    f"nonfinite model state after land-state refresh hour {hour}",
                    {
                        "failure_mode": "NONFINITE_LAND_STATE_REFRESH",
                        "failed_hour": int(hour),
                        "finite_summary": summary,
                        "land_refresh_record": land_record,
                        "output_files_before_failure": [str(path) for path in files],
                    },
                )

        valid_time = case.run_start + timedelta(hours=hour)
        wrfout = output_dir / _wrfout_name(valid_time, config.domain)
        # Route the operational surface-layer diagnostics into the writer so the
        # wrfout surface map (T2/U10/V10/Q2/PSFC/SWDOWN/GLW/PBLH/TSK) is the
        # physically diagnosed 2-m/10-m/skin field, NOT the raw lowest-level
        # fallback (which reads far too warm/strong over terrain -- the d03 1km
        # T2 +8K-mean / +34K-summit bias). Mirrors the d02-validate diagnostic
        # source: compute_m9_diagnostics(state, namelist_with_clock, lead_seconds).
        # time_utc is threaded so SWDOWN/GLW follow the real forecast diurnal clock.
        surface_diagnostics = _surface_diagnostics_for_output(
            state, case.namelist, case.run_start, lead_seconds=float(hour) * 3600.0
        )
        diagnostics = _merge_output_diagnostics(case.writer_diagnostics, surface_diagnostics)
        prepared = None
        if writer is not None:
            # Pull this hour to host now (synchronous device->host), then write on
            # the background thread while the GPU starts the next hour.
            prepared = prepare_wrfout_payload(
                state,
                case.grid,
                case.namelist,
                wrfout,
                valid_time=valid_time,
                lead_hours=float(hour),
                run_start=case.run_start,
                diagnostics=diagnostics,
            )
            writer.submit(prepared)
        else:
            write_wrfout_netcdf(
                state,
                case.grid,
                case.namelist,
                wrfout,
                valid_time=valid_time,
                lead_hours=float(hour),
                run_start=case.run_start,
                diagnostics=diagnostics,
            )
        files.append(wrfout)

        # Auxhist frames at this hourly boundary: every stream whose interval
        # divides the hour fires here (e.g. a 60-min stream every hour, a 15-min
        # stream's :00 frame). Reuses the same diagnostics/state the main wrfout
        # used; when the async writer already materialized the hour to host
        # (``prepared``), ALL firing streams reuse that payload (no 2nd pull).
        hour_lead_minutes = float(hour) * 60.0
        _emit_streams_at(
            hour_lead_minutes, cur_state=state, diagnostics=diagnostics, prepared=prepared
        )

        if checkpoint_at_hour is not None and hour == int(checkpoint_at_hour):
            checkpoint_path = output_dir / "checkpoints" / f"{config.run_id}_hour{hour:02d}.pkl"
            write_checkpoint(
                state,
                case.namelist,
                case.grid,
                hour_steps(hour, float(config.dt_s)),
                checkpoint_path,
            )
            state, restored_namelist, restored_grid, restored_step = read_checkpoint(checkpoint_path)
            case = replace(case, state=state, namelist=restored_namelist, grid=restored_grid)
            checkpoint_payload = {
                "path": str(checkpoint_path),
                "checkpoint_hour": int(hour),
                "step_index": int(restored_step),
                "restored": True,
            }

    # Flush all background wrfout writes before returning: downstream
    # validation/inventory/scoring reads these files immediately after this call.
    _flush_writer()

    metadata = dict(case.metadata)
    metadata["land_state_refresh"] = {
        "enabled": bool(config.refresh_land_state_hourly),
        "cadence": "hourly forecast output boundary",
        "records": land_refresh_records,
    }
    if streams:
        stream_meta = [
            {
                "stream_id": int(s.stream_id),
                "interval_minutes": int(s.interval_minutes),
                "frames_per_file": int(s.frames_per_file),
                "io_form": int(s.io_form),
                "outname_pattern": s.outname_pattern,
                "variables": list(s.variables),
                "frame_count": int(len(auxhist_files_by_stream[int(s.stream_id)])),
                "files": [str(path) for path in auxhist_files_by_stream[int(s.stream_id)]],
            }
            for s in streams
        ]
        # Multi-stream metadata (one entry per configured auxhist stream).
        metadata["auxhist_streams"] = stream_meta
        # Back-compat: a single-stream run keeps the legacy ``auxhist_stream`` key.
        if len(stream_meta) == 1:
            metadata["auxhist_stream"] = stream_meta[0]
    return ForecastSequenceResult(
        status="PASS",
        run_id=config.run_id,
        run_dir=run_dir,
        output_dir=output_dir,
        hours=int(config.hours),
        output_files=files,
        per_hour_wall_s=per_hour_wall_s,
        final_state=state,
        run_start=case.run_start,
        metadata=metadata,
        finite_summary=finite_summary(state),
        checkpoint=checkpoint_payload,
        auxhist_files=auxhist_files,
    )


def hour_steps(hour: int, dt_s: float = DT_S) -> int:
    raw = int(hour) * 3600.0 / float(dt_s)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-8:
        raise ValueError(f"hour={hour} does not align with dt={dt_s:g}s")
    return rounded


def build_wrfout_inventory(wrfout_paths: Sequence[str | Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    all_readable = True
    all_minimum_present = True
    for item in wrfout_paths:
        path = Path(item)
        record: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "exists": path.is_file(),
            "readable": False,
            "size_bytes": int(path.stat().st_size) if path.is_file() else 0,
            "missing_minimum_variables": list(MINIMUM_WRFOUT_VARIABLES),
            "minimum_variable_count": int(len(MINIMUM_WRFOUT_VARIABLES)),
            "present_minimum_variable_count": 0,
        }
        try:
            with Dataset(path, "r") as dataset:
                variables = set(dataset.variables)
                missing = [name for name in MINIMUM_WRFOUT_VARIABLES if name not in variables]
                record.update(
                    {
                        "readable": True,
                        "file_format": dataset.file_format,
                        "valid_time_utc": parse_wrfout_valid_time(path).isoformat(),
                        "variable_count": int(len(variables)),
                        "missing_minimum_variables": missing,
                        "present_minimum_variable_count": int(len(MINIMUM_WRFOUT_VARIABLES) - len(missing)),
                        "dimensions": {name: int(len(dim)) for name, dim in dataset.dimensions.items()},
                    }
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        all_readable = all_readable and bool(record["readable"])
        all_minimum_present = all_minimum_present and not record["missing_minimum_variables"]
        records.append(record)
    expected_count = len(wrfout_paths)
    status = "PASS" if all_readable and all_minimum_present else "FAIL"
    return {
        "schema": "M7DailyPipelineWrfoutInventory",
        "schema_version": 1,
        "status": status,
        "expected_wrfout_count": int(expected_count),
        "actual_wrfout_count": int(len(records)),
        "minimum_variable_count": int(len(MINIMUM_WRFOUT_VARIABLES)),
        "all_readable": bool(all_readable),
        "all_minimum_variables_present": bool(all_minimum_present),
        "files": records,
    }


def _station_metadata(observations: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in ("station_id", "lat", "lon", "elev_m") if column in observations.columns]
    if not keep:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "elev_m"])
    return observations[keep].dropna(subset=["station_id", "lat", "lon"]).drop_duplicates(subset=["station_id"])


def score_wrfouts_against_aemet(
    wrfout_paths: Sequence[str | Path],
    *,
    aemet_root: str | Path = DEFAULT_AEMET_ROOT,
    variables: Sequence[str] = ("T2", "U10", "V10"),
) -> dict[str, Any]:
    files = [Path(path) for path in wrfout_paths]
    if not files:
        return {
            "schema": "M7DailyPipelineStationScores",
            "schema_version": 1,
            "status": "NO_WRFOUTS",
            "joined_rows": 0,
            "scores": {},
            "acceptance": {"finite_scores": False, "joined_rows_at_least_100": False},
        }
    start_time = parse_wrfout_valid_time(files[0])
    end_time = parse_wrfout_valid_time(files[-1])
    inventory = inventory_aemet_observations(aemet_root)
    observations = load_aemet_observations(aemet_root, variables=variables, start_time=start_time, end_time=end_time)
    stations = _station_metadata(observations)
    forecast_frames = [
        interpolate_to_stations(path, stations, variables=variables, valid_time=parse_wrfout_valid_time(path))
        for path in files
    ]
    forecasts = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    report = compute_station_scores(forecasts, observations, variables=variables).to_dict()
    finite_scores = True
    for variable in variables:
        entry = report["scores"].get(variable, {})
        if int(entry.get("sample_count", 0)) <= 0:
            finite_scores = False
            continue
        for key in ("bias", "rmse", "mae"):
            value = entry.get(key)
            finite_scores = finite_scores and isinstance(value, (int, float)) and math.isfinite(float(value))
    joined_ok = int(report["joined_rows"]) >= 100
    status = "PASS" if finite_scores and joined_ok else "FAIL"
    return {
        "schema": "M7DailyPipelineStationScores",
        "schema_version": 1,
        "status": status,
        "valid_time_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
        "aemet_root": str(aemet_root),
        "aemet_inventory": {
            "file_count": inventory.get("file_count"),
            "station_count": inventory.get("station_count"),
            "temporal_coverage": inventory.get("temporal_coverage"),
            "variables_present": inventory.get("variables_present"),
        },
        "station_observation_rows": int(len(observations)),
        "station_count_scored": int(stations["station_id"].nunique()) if not stations.empty else 0,
        "forecast_rows": int(len(forecasts)),
        "joined_rows": int(report["joined_rows"]),
        "variables": list(variables),
        "scores": report["scores"],
        "acceptance": {
            "finite_scores": bool(finite_scores),
            "joined_rows_at_least_100": bool(joined_ok),
            "minimum_joined_rows": 100,
        },
    }


def compare_wrfouts_xarray(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left = Path(left_path)
    right = Path(right_path)
    fields: dict[str, Any] = {}
    passed = True
    with xr.open_dataset(left, engine="netcdf4", decode_times=False) as left_ds:
        with xr.open_dataset(right, engine="netcdf4", decode_times=False) as right_ds:
            for name in sorted(set(left_ds.variables) | set(right_ds.variables)):
                if name not in left_ds.variables or name not in right_ds.variables:
                    fields[name] = {"status": "MISSING", "pass": False}
                    passed = False
                    continue
                left_values = np.asarray(left_ds[name].values)
                right_values = np.asarray(right_ds[name].values)
                if left_values.dtype.kind in {"S", "U"} or right_values.dtype.kind in {"S", "U"}:
                    equal = bool(np.array_equal(left_values, right_values))
                    max_delta = 0.0 if equal else float("inf")
                    tolerance = 0.0
                else:
                    diff = right_values.astype(np.float64) - left_values.astype(np.float64)
                    max_delta = float(np.nanmax(np.abs(diff))) if diff.size else 0.0
                    dtype = np.promote_types(left_values.dtype, right_values.dtype)
                    tolerance = 1.0e-12 if np.dtype(dtype) == np.dtype(np.float64) else 1.0e-6
                    equal = bool(np.isfinite(diff).all() and max_delta <= tolerance)
                fields[name] = {
                    "shape": list(left_values.shape),
                    "left_dtype": str(left_values.dtype),
                    "right_dtype": str(right_values.dtype),
                    "max_abs_delta": max_delta,
                    "tolerance": tolerance,
                    "pass": equal,
                }
                passed = passed and equal
    return {
        "status": "PASS" if passed else "FAIL",
        "left": str(left),
        "right": str(right),
        "field_count": int(len(fields)),
        "fields": fields,
    }


def _parse_cpu_timing_records(run_dir: Path, domain_id: int = 2) -> list[tuple[datetime, float, Path]]:
    records: list[tuple[datetime, float, Path]] = []
    for name in ("namelist.output", "rsl.error.0000", "rsl.out.0000"):
        path = run_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in CPU_TIMING_RE.finditer(text):
            if int(match.group("domain")) != int(domain_id):
                continue
            stamp = datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
            records.append((stamp, float(match.group("elapsed")), path))
    return records


def cpu_baseline_24h_from_logs(run_dir: str | Path, *, domain_id: int = 2) -> dict[str, Any]:
    source = Path(run_dir)
    records = _parse_cpu_timing_records(source, domain_id=domain_id)
    if records:
        stamps = sorted({stamp for stamp, _, _ in records})
        deltas = [
            (right - left).total_seconds()
            for left, right in zip(stamps[:-1], stamps[1:])
            if (right - left).total_seconds() > 0
        ]
        dt_s = float(np.median(deltas)) if deltas else DT_S
        elapsed = np.asarray([value for _, value, _ in records], dtype=np.float64)
        steps_per_24h = int(round(24.0 * 3600.0 / dt_s))
        wall_24h = float(np.mean(elapsed) * steps_per_24h)
        return {
            "status": "PASS",
            "method": "mean WRF per-step Timing for main on d02 multiplied by 24h step count",
            "run_dir": str(source),
            "domain_id": int(domain_id),
            "record_count": int(len(records)),
            "source_files": sorted({str(path) for _, _, path in records}),
            "median_model_step_s": dt_s,
            "mean_step_wall_s": float(np.mean(elapsed)),
            "median_step_wall_s": float(np.median(elapsed)),
            "steps_per_24h": steps_per_24h,
            "cpu_wall_24h_s": wall_24h,
        }

    files = sorted(source.glob("wrfout_d02_*"))
    if len(files) >= 2:
        first_time = parse_wrfout_valid_time(files[0])
        last_time = parse_wrfout_valid_time(files[-1])
        simulated_hours = max((last_time - first_time).total_seconds() / 3600.0, 1.0)
        observed_wall = max(files[-1].stat().st_mtime - files[0].stat().st_mtime, 0.0)
        return {
            "status": "PASS",
            "method": "fallback wrfout file mtime extrapolated to 24h",
            "run_dir": str(source),
            "observed_wrfout_count": int(len(files)),
            "observed_simulated_hours": float(simulated_hours),
            "observed_wall_s": float(observed_wall),
            "cpu_wall_24h_s": float(observed_wall * 24.0 / simulated_hours),
        }
    return {
        "status": "FAIL",
        "method": "no WRF timing records or wrfout fallback files found",
        "run_dir": str(source),
        "cpu_wall_24h_s": None,
    }


def speedup_vs_cpu_24h(
    run_dir: str | Path,
    *,
    pipeline_wall_s: float,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline = cpu_baseline_24h_from_logs(run_dir)
    cpu_wall = baseline.get("cpu_wall_24h_s")
    speedup = float(cpu_wall) / float(pipeline_wall_s) if cpu_wall and pipeline_wall_s > 0 else None
    if speedup is None:
        status = "FAIL"
    elif speedup >= 4.0:
        status = "PASS"
    else:
        status = "BELOW_TARGET"
    payload = {
        "schema": "M7DailyPipelineSpeedupVsCpu24h",
        "schema_version": 1,
        "status": status,
        "target_speedup_band": [4.0, 8.0],
        "pipeline_wall_24h_s": float(pipeline_wall_s),
        "cpu_baseline": baseline,
        "speedup": speedup,
        "note": (
            "CPU denominator is derived from existing Gen2 WRF logs for the contracted run. "
            "Pipeline wall includes IC load, hourly GPU forecast calls, wrfout writes, inventory, and scoring."
        ),
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def _artifact_paths(proof_dir: Path) -> dict[str, Path]:
    return {
        "pipeline": proof_dir / "pipeline_run_20260521.json",
        "inventory": proof_dir / "wrfout_inventory.json",
        "scores": proof_dir / "station_scores_20260521.json",
        "restart": proof_dir / "restart_in_pipeline.json",
        "repeatability": proof_dir / "repeatability.json",
        "speedup": proof_dir / "speedup_vs_cpu_24h.json",
    }


def _main_pipeline_payload(
    config: DailyPipelineConfig,
    main: ForecastSequenceResult,
    *,
    total_wall_s: float,
    inventory: Mapping[str, Any],
    scores: Mapping[str, Any] | None,
    speedup: Mapping[str, Any] | None,
) -> dict[str, Any]:
    score_summary = None
    if scores is not None:
        score_summary = {
            "status": scores.get("status"),
            "joined_rows": scores.get("joined_rows"),
            "scores": scores.get("scores", {}),
            "acceptance": scores.get("acceptance", {}),
        }
    # PIPELINE_GREEN requires the forecast to RUN and write valid wrfout. Optional
    # comparisons that were not requested or have no reference available are NOT
    # failures: ``--score`` not given (NOT_RUN), and the CPU-speedup baseline being
    # absent on a fresh standalone case (no CPU logs to divide by) are both expected
    # for a standalone out-of-the-box run and must NOT downgrade the verdict.
    verdict = "PIPELINE_GREEN"
    if inventory.get("status") != "PASS":
        verdict = "PIPELINE_PARTIAL"
    if scores is not None and scores.get("status") not in (None, "PASS", "NOT_RUN"):
        verdict = "PIPELINE_PARTIAL"
    if speedup is not None and speedup.get("status") == "FAIL":
        # FAIL means "no CPU baseline" (no logs / <2 wrfout): not a regression on a
        # standalone case. BELOW_TARGET (a real measured speedup under target) is
        # also reported but non-blocking; only inventory/scores can downgrade.
        speedup = dict(speedup)
        speedup["blocking"] = False
    return {
        "schema": "M7DailyPipelineRun",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": config.run_id,
        "domain": config.domain,
        "hours": int(config.hours),
        "device": visible_gpu_name(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "output_dir": str(config.output_dir),
        "run_dir": str(main.run_dir),
        "run_start_utc": main.run_start.isoformat(),
        "wall_clock_total_s": float(total_wall_s),
        "wall_clock_per_hour_s": main.per_hour_wall_s,
        "wall_clock_forecast_only_s": main.forecast_wall_s,
        "wall_clock_per_forecast_hour_s": float(total_wall_s / max(int(config.hours), 1)),
        "wrfout_files": [str(path) for path in main.output_files],
        "wrfout_inventory_status": inventory.get("status"),
        "station_score_summary": score_summary,
        "all_finite_check": main.finite_summary,
        "speedup_status": None if speedup is None else speedup.get("status"),
        "metadata": main.metadata,
    }


def detect_init_mode(config: DailyPipelineConfig) -> str:
    """Return ``"standalone_native_init"`` or ``"cpu_wrf_replay"`` for a run config.

    Cheap filesystem check (no JAX import): a run dir with FEWER THAN TWO
    ``wrfout_<domain>`` history files cannot be replayed, so it takes the standalone
    native-init path (IC from ``wrfinput_<domain>``, LBC from ``wrfbdy``). Two or
    more wrfout history files -> the classic CPU-WRF replay path.
    """

    run_dir = resolve_run_dir(config.run_id, config.run_root)
    wrfout_count = len(sorted(Path(run_dir).glob(f"wrfout_{config.domain}_*")))
    return "cpu_wrf_replay" if wrfout_count >= 2 else "standalone_native_init"


def execute_daily_pipeline(
    config: DailyPipelineConfig,
    *,
    forecast_fn: Callable[[Any, Any, float], Any] | None = None,
    case_builder: Callable[[DailyPipelineConfig], tuple[DailyCase, Path]] = _build_real_case,
) -> dict[str, Any]:
    paths = _artifact_paths(config.proof_dir)
    config.proof_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()

    # Auto-select the forecast driver by init mode (callers may still override).
    # Standalone native-init uses the long-run-safe segmented entry (no outer
    # donate, bounded compile/VRAM); replay uses the donate-safe single-scan entry.
    init_mode = detect_init_mode(config)
    if forecast_fn is None:
        forecast_fn = (
            _segmented_forecast_fn
            if init_mode == "standalone_native_init"
            else _default_forecast_fn
        )

    try:
        main = _run_forecast_sequence(
            config,
            output_dir=config.output_dir,
            forecast_fn=forecast_fn,
            case_builder=case_builder,
        )
        inventory = build_wrfout_inventory(main.output_files)
        write_json(paths["inventory"], inventory)
        scores: dict[str, Any] | None = None
        if config.score:
            scores = score_wrfouts_against_aemet(main.output_files, aemet_root=config.aemet_root, variables=("T2", "U10", "V10"))
        else:
            scores = {
                "schema": "M7DailyPipelineStationScores",
                "schema_version": 1,
                "status": "NOT_RUN",
                "reason": "--score was not requested",
                "joined_rows": 0,
                "scores": {},
                "acceptance": {"finite_scores": False, "joined_rows_at_least_100": False},
            }
        write_json(paths["scores"], scores)
        main_wall_s = time.perf_counter() - overall_start
        speedup = speedup_vs_cpu_24h(main.run_dir, pipeline_wall_s=main_wall_s, output_path=paths["speedup"])
        pipeline_payload = _main_pipeline_payload(
            config,
            main,
            total_wall_s=main_wall_s,
            inventory=inventory,
            scores=scores,
            speedup=speedup,
        )
        write_json(paths["pipeline"], pipeline_payload)

        restart_payload = _run_restart_probe(config, main, forecast_fn=forecast_fn, case_builder=case_builder)
        write_json(paths["restart"], restart_payload)
        repeat_payload = _run_repeatability_probe(config, main, forecast_fn=forecast_fn, case_builder=case_builder)
        write_json(paths["repeatability"], repeat_payload)

        final_verdict = pipeline_payload["verdict"]
        if config.restart_at_hour is not None and restart_payload.get("status") != "PASS":
            final_verdict = "PIPELINE_PARTIAL"
        if config.repeat and repeat_payload.get("status") != "PASS":
            final_verdict = "PIPELINE_PARTIAL"
        pipeline_payload["restart_probe_status"] = restart_payload.get("status")
        pipeline_payload["repeatability_status"] = repeat_payload.get("status")
        pipeline_payload["verdict"] = final_verdict
        write_json(paths["pipeline"], pipeline_payload)
        return pipeline_payload
    except PipelineBlocked as exc:
        return _write_blocked_artifacts(config, paths, "PIPELINE_BLOCKED", str(exc), exc.payload)
    except Exception as exc:
        return _write_blocked_artifacts(
            config,
            paths,
            "PIPELINE_BLOCKED",
            f"{type(exc).__name__}: {exc}",
            {"failure_mode": type(exc).__name__},
        )


def _run_restart_probe(
    config: DailyPipelineConfig,
    main: ForecastSequenceResult,
    *,
    forecast_fn: Callable[[Any, Any, float], Any],
    case_builder: Callable[[DailyPipelineConfig], tuple[DailyCase, Path]],
) -> dict[str, Any]:
    if config.restart_at_hour is None:
        return {
            "schema": "M7DailyPipelineRestartProbe",
            "schema_version": 1,
            "status": "NOT_RUN",
            "reason": "--restart-at-hour was not requested",
        }
    start = time.perf_counter()
    restart_dir = config.output_dir / "restart_probe"
    restarted = _run_forecast_sequence(
        config,
        output_dir=restart_dir,
        checkpoint_at_hour=config.restart_at_hour,
        forecast_fn=forecast_fn,
        case_builder=case_builder,
    )
    if main.final_wrfout is None or restarted.final_wrfout is None:
        comparison = {"status": "FAIL", "reason": "missing final wrfout"}
    else:
        comparison = compare_wrfouts_xarray(main.final_wrfout, restarted.final_wrfout)
    return {
        "schema": "M7DailyPipelineRestartProbe",
        "schema_version": 1,
        "status": "PASS" if comparison.get("status") == "PASS" and restarted.checkpoint else "FAIL",
        "restart_at_hour": int(config.restart_at_hour),
        "wall_clock_s": float(time.perf_counter() - start),
        "checkpoint": restarted.checkpoint,
        "continuous_final_wrfout": str(main.final_wrfout),
        "restarted_final_wrfout": str(restarted.final_wrfout),
        "comparison": comparison,
    }


def _run_repeatability_probe(
    config: DailyPipelineConfig,
    main: ForecastSequenceResult,
    *,
    forecast_fn: Callable[[Any, Any, float], Any],
    case_builder: Callable[[DailyPipelineConfig], tuple[DailyCase, Path]],
) -> dict[str, Any]:
    if not config.repeat:
        return {
            "schema": "M7DailyPipelineRepeatability",
            "schema_version": 1,
            "status": "NOT_RUN",
            "reason": "--repeat was not requested",
        }
    start = time.perf_counter()
    repeat_dir = config.output_dir / "repeat_2"
    repeat = _run_forecast_sequence(
        config,
        output_dir=repeat_dir,
        forecast_fn=forecast_fn,
        case_builder=case_builder,
    )
    if main.final_wrfout is None or repeat.final_wrfout is None:
        comparison = {"status": "FAIL", "reason": "missing final wrfout"}
    else:
        comparison = compare_wrfouts_xarray(main.final_wrfout, repeat.final_wrfout)
    return {
        "schema": "M7DailyPipelineRepeatability",
        "schema_version": 1,
        "status": "PASS" if comparison.get("status") == "PASS" else "FAIL",
        "wall_clock_s": float(time.perf_counter() - start),
        "run1_final_wrfout": str(main.final_wrfout),
        "run2_final_wrfout": str(repeat.final_wrfout),
        "comparison": comparison,
    }


def _write_blocked_artifacts(
    config: DailyPipelineConfig,
    paths: Mapping[str, Path],
    verdict: str,
    reason: str,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "M7DailyPipelineRun",
        "schema_version": 1,
        "verdict": verdict,
        "run_id": config.run_id,
        "hours": int(config.hours),
        "output_dir": str(config.output_dir),
        "reason": reason,
        "detail": dict(detail),
    }
    write_json(paths["pipeline"], payload)
    for key, schema in (
        ("inventory", "M7DailyPipelineWrfoutInventory"),
        ("scores", "M7DailyPipelineStationScores"),
        ("restart", "M7DailyPipelineRestartProbe"),
        ("repeatability", "M7DailyPipelineRepeatability"),
        ("speedup", "M7DailyPipelineSpeedupVsCpu24h"),
    ):
        write_json(
            paths[key],
            {
                "schema": schema,
                "schema_version": 1,
                "status": "BLOCKED",
                "reason": reason,
                "upstream_verdict": verdict,
            },
        )
    return payload


__all__ = [
    "DailyCase",
    "DailyPipelineConfig",
    "PipelineBlocked",
    "build_wrfout_inventory",
    "compare_wrfouts_xarray",
    "cpu_baseline_24h_from_logs",
    "execute_daily_pipeline",
    "finite_summary",
    "hour_steps",
    "resolve_run_dir",
    "score_wrfouts_against_aemet",
    "speedup_vs_cpu_24h",
    "write_json",
]
