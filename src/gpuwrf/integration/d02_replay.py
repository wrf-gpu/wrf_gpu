"""ADR-023 d02 boundary-replay integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import os
from pathlib import Path
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
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.gen2_wrfout_loader import normalize_valid_time
from gpuwrf.io.land_state import load_prescribed_land_state
from gpuwrf.profiling.transfer_audit import block_until_ready, count_transfer_bytes, visible_gpu_name


config.update("jax_enable_x64", True)

P0_THETA_OFFSET_K = 300.0
DEFAULT_REPLAY_RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DEFAULT_OUTPUT_FIELD_PATH = Path("/home/enric/.cache/gpuwrf_outputs/m6x_d02_replay/proof_d02_replay_fields.npz")
DEFAULT_TRACE_ROOT = Path(os.environ.get("GPUWRF_TMPDIR", "/home/enric/.cache/gpuwrf_tmp"))


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


def _load(run: Gen2Run, domain: str, var: str, time: int):
    return run.load(domain, var, time=time, lazy=False)


def _optional_load(run: Gen2Run, domain: str, var: str, time: int, fallback):
    try:
        return _load(run, domain, var, time)
    except KeyError:
        return fallback


def _field_sides_3d(field: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "W": field[:, :, 0],
        "E": field[:, :, -1],
        "S": field[:, 0, :],
        "N": field[:, -1, :],
    }


def _field_sides_2d(field: np.ndarray) -> dict[str, np.ndarray]:
    return {"W": field[:, 0], "E": field[:, -1], "S": field[0, :], "N": field[-1, :]}


def _pack_history_3d(
    run: Gen2Run,
    domain: str,
    var: str,
    *,
    ntimes: int,
    z_len: int,
    max_side: int,
    dtype: Any,
    transform=None,
) -> np.ndarray:
    packed = np.zeros((ntimes, 4, z_len, max_side), dtype=dtype)
    for time_index in range(ntimes):
        data = np.asarray(_load(run, domain, var, time_index), dtype=dtype)
        if transform is not None:
            data = np.asarray(transform(run, domain, data, time_index), dtype=dtype)
        for side, values in _field_sides_3d(data).items():
            packed[time_index, SIDE_INDEX[side], : values.shape[0], : values.shape[1]] = values
    return packed


def _pack_history_mu(run: Gen2Run, domain: str, *, ntimes: int, max_side: int) -> np.ndarray:
    packed = np.zeros((ntimes, 4, 1, max_side), dtype=np.float64)
    for time_index in range(ntimes):
        data = np.asarray(_load(run, domain, "MU", time_index) + _load(run, domain, "MUB", time_index), dtype=np.float64)
        for side, values in _field_sides_2d(data).items():
            packed[time_index, SIDE_INDEX[side], 0, : values.shape[0]] = values
    return packed


def load_history_boundary_leaves(
    run: Gen2Run,
    grid: GridSpec,
    *,
    domain: str = "d02",
    ntimes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build real lateral replay leaves from the Gen2 d02 hourly wrfout history."""

    history_count = len(run.history_files(domain))
    if history_count < 2:
        raise FileNotFoundError(f"{run.path} has fewer than two wrfout_{domain} history files")
    n = min(history_count, int(ntimes if ntimes is not None else history_count))
    max_side = int(max(grid.nx + 1, grid.ny + 1))

    def add_theta(_run: Gen2Run, _domain: str, data: np.ndarray, _time_index: int) -> np.ndarray:
        return data + P0_THETA_OFFSET_K

    def add_phb(_run: Gen2Run, _domain: str, data: np.ndarray, time_index: int) -> np.ndarray:
        return data + np.asarray(_load(_run, _domain, "PHB", time_index), dtype=data.dtype)

    leaves_np = {
        "u_bdy": _pack_history_3d(run, domain, "U", ntimes=n, z_len=grid.nz, max_side=max_side, dtype=np.float32),
        "v_bdy": _pack_history_3d(run, domain, "V", ntimes=n, z_len=grid.nz, max_side=max_side, dtype=np.float32),
        "theta_bdy": _pack_history_3d(
            run,
            domain,
            "T",
            ntimes=n,
            z_len=grid.nz,
            max_side=max_side,
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
            dtype=np.float32,
        ),
        "ph_bdy": _pack_history_3d(
            run,
            domain,
            "PH",
            ntimes=n,
            z_len=grid.nz + 1,
            max_side=max_side,
            dtype=np.float64,
            transform=add_phb,
        ),
        "mu_bdy": _pack_history_mu(run, domain, ntimes=n, max_side=max_side),
    }
    leaves = {name: jax.device_put(jnp.asarray(value)) for name, value in leaves_np.items()}
    meta = {
        "source": "Gen2 d02 hourly wrfout side-history replay",
        "source_run_dir": str(run.path),
        "times": int(n),
        "side_order": list(SIDES),
        "padded_side_length": max_side,
        "schema": "history-side-pack-v1",
    }
    return leaves, meta


def build_replay_case(run_dir: str | Path = DEFAULT_REPLAY_RUN_DIR, *, domain: str = "d02") -> ReplayCase:
    """Load a Gen2 d02 initial state with WRF perturbation/base splits preserved."""

    run = Gen2Run(run_dir)
    grid = run.grid(domain).as_grid_spec()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    metrics = load_wrfinput_metrics(run.wrfinput_file(domain))
    land = load_prescribed_land_state(run, domain=domain, time=0)
    boundary_leaves, boundary_meta = load_history_boundary_leaves(run, grid, domain=domain)

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
    return ReplayCase(run, state, tendencies, grid, metrics, base, state.p_perturbation, metadata)


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
    run_radiation = (global_step % int(replay_config.radiation_cadence_steps)) == 0
    run_radiation = run_radiation | (
        bool(replay_config.final_radiation) & (global_step == _total_steps(replay_config))
    )
    next_state = jax.lax.cond(
        run_radiation,
        lambda value: rrtmg_adapter(value, float(replay_config.dt_s), grid),
        lambda value: value,
        next_state,
    )
    lead_seconds = global_step.astype(jnp.float64) * float(replay_config.dt_s)
    next_state = apply_lateral_boundaries(next_state, lead_seconds, float(replay_config.dt_s), replay_config.boundary_config)
    return next_state, next_previous_pressure


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
    """Run the coupled ADR-023 replay as one device-resident scan."""

    def body(carry, local_index):
        carry_state, carry_previous_pressure = carry
        global_step = local_index.astype(jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
        candidate, next_previous_pressure = _candidate_timestep_adr023(
            carry_state,
            carry_previous_pressure,
            tendencies,
            grid,
            metrics,
            base_state,
            global_step,
            replay_config,
        )
        sanitized, stats = _sanitize_replay_candidate(candidate, carry_state, base_state)
        return (sanitized, next_previous_pressure), _step_diagnostics(sanitized, stats)

    (final_state, final_previous_pressure), diagnostics = jax.lax.scan(
        body,
        (state, previous_pressure),
        jnp.arange(_total_steps(replay_config), dtype=jnp.int32),
    )
    return final_state, final_previous_pressure, diagnostics


def _first_nonfinite_step(diagnostics: StepDiagnostics) -> int | None:
    finite = np.asarray(jax.device_get(diagnostics.finite_after_sanitize), dtype=bool)
    bad = ~finite
    indices = np.flatnonzero(bad)
    return int(indices[0] + 1) if indices.size else None


def diagnostics_summary(state: State, diagnostics: StepDiagnostics) -> dict[str, Any]:
    nonfinite = np.asarray(jax.device_get(diagnostics.candidate_nonfinite_count), dtype=np.int64)
    clips = np.asarray(jax.device_get(diagnostics.candidate_clip_count), dtype=np.int64)
    changed = np.asarray(jax.device_get(diagnostics.candidate_changed_count), dtype=np.int64)
    candidate_bad = np.flatnonzero(nonfinite > 0)
    w_abs = np.asarray(jax.device_get(diagnostics.w_abs_max_m_s), dtype=np.float64)
    theta_min = np.asarray(jax.device_get(diagnostics.theta_min_k), dtype=np.float64)
    theta_max = np.asarray(jax.device_get(diagnostics.theta_max_k), dtype=np.float64)
    return {
        **state_diagnostics(state),
        "first_nonfinite_step": _first_nonfinite_step(diagnostics),
        "first_candidate_nonfinite_step": int(candidate_bad[0] + 1) if candidate_bad.size else None,
        "candidate_nonfinite_steps": int(np.count_nonzero(nonfinite)),
        "candidate_nonfinite_count_total": int(nonfinite.sum()),
        "candidate_clip_count_total": int(clips.sum()),
        "candidate_changed_count_total": int(changed.sum()),
        "peak_w_abs_m_s": float(np.nanmax(w_abs)),
        "theta_min_over_run_k": float(np.nanmin(theta_min)),
        "theta_max_over_run_k": float(np.nanmax(theta_max)),
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
    return {
        "method": "JAXPR callback scan for the exact ADR-023 replay scan",
        "host_callback_free": all(token not in jaxpr_text for token in forbidden),
        "forbidden_tokens": list(forbidden),
        "jaxpr_bytes": len(jaxpr_text.encode("utf-8")),
    }


def trace_transfer_audit(case: ReplayCase, replay_config: ReplayConfig, trace_dir: str | Path) -> dict[str, Any]:
    trace_path = Path(trace_dir)
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
    if trace_path.exists():
        shutil.rmtree(trace_path)
    trace_path.mkdir(parents=True, exist_ok=True)
    try:
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
    h2d, d2h, files = count_transfer_bytes(trace_path)
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
    case = build_replay_case(run_dir, domain=domain)
    block_until_ready((case.state, case.previous_pressure, case.tendencies, case.metrics, case.base_state))
    start = time.perf_counter()
    final_state, final_previous_pressure, step_diags = run_replay_scan(
        case.state,
        case.previous_pressure,
        case.tendencies,
        case.grid,
        case.metrics,
        case.base_state,
        replay_config,
    )
    block_until_ready((final_state, final_previous_pressure, step_diags))
    wall_s = time.perf_counter() - start
    comparison = forecast_comparison(final_state, case.run, domain=domain, lead_hours=float(replay_config.duration_s) / 3600.0)
    diag = diagnostics_summary(final_state, step_diags)
    output_path = None
    if output_fields_path is not None:
        output_path = write_output_fields(output_fields_path, final_state, comparison)
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
    "build_replay_case",
    "diagnostics_summary",
    "forecast_comparison",
    "invoked_schemes",
    "load_history_boundary_leaves",
    "replay_steps",
    "run_replay_proof",
    "run_replay_scan",
    "static_transfer_audit",
    "trace_transfer_audit",
]
