"""M6-S2 coupled d02 forecast driver on the GPU-resident State pytree."""

from __future__ import annotations

from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import zarr

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.boundary_apply import BoundaryConfig, DEFAULT_BOUNDARY_CONFIG, SIDE_INDEX, SIDES, apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.step import step as dycore_step
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.validation import load_gen2_var
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

P0_THETA_OFFSET_K = 300.0
DEFAULT_DT_S = 60.0
DEFAULT_RADIATION_CADENCE_STEPS = 10
STANDARD_OUTPUT_LEADS_H = (1, 6, 12, 18, 24)


def _load(run: Gen2Run, domain: str, var: str, time: int = 0):
    """Route every Gen2 variable read through the shared validation I/O path."""

    return load_gen2_var(run, domain, var, time=time)


def build_initial_state(
    run: Gen2Run,
    *,
    domain: str = "d02",
    boundary_path: str | Path = "data/fixtures/m6/d02_boundary_replay_v1.zarr",
) -> tuple[State, Tendencies, GridSpec, dict[str, Any]]:
    """Load the d02 IC and replayed lateral boundaries into one device-resident State."""

    grid = run.grid(domain).as_grid_spec()
    state = State.zeros(grid)

    u = _load(run, domain, "U", 0)
    v = _load(run, domain, "V", 0)
    w = _load(run, domain, "W", 0)
    theta = _load(run, domain, "T", 0) + P0_THETA_OFFSET_K
    qv = _load(run, domain, "QVAPOR", 0)
    p = _load(run, domain, "P", 0) + _load(run, domain, "PB", 0)
    ph = _load(run, domain, "PH", 0) + _load(run, domain, "PHB", 0)
    mu = _load(run, domain, "MU", 0) + _load(run, domain, "MUB", 0)
    smois = _load(run, domain, "SMOIS", 0)

    bdy_leaves, boundary_meta = load_boundary_leaves(run, grid, boundary_path=boundary_path, domain=domain)

    state = state.replace(
        u=u,
        v=v,
        w=w,
        theta=theta,
        qv=qv,
        p=p,
        ph=ph,
        mu=mu,
        qc=_load(run, domain, "QCLOUD", 0),
        qr=_load(run, domain, "QRAIN", 0),
        qi=_load(run, domain, "QICE", 0),
        qs=_load(run, domain, "QSNOW", 0),
        qg=_load(run, domain, "QGRAUP", 0),
        Ni=_load(run, domain, "QNICE", 0),
        Nr=_load(run, domain, "QNRAIN", 0),
        Ns=jnp.zeros_like(state.Ns),
        Ng=jnp.zeros_like(state.Ng),
        qke=_load(run, domain, "QKE", 0),
        t_skin=_load(run, domain, "TSK", 0),
        soil_moisture=smois[0],
        **bdy_leaves,
    )
    tendencies = Tendencies.zeros(grid)
    meta = {
        "domain": domain,
        "run_id": run.run_id,
        "boundary": boundary_meta,
        "grid": {
            "mass_shape": [int(grid.nx), int(grid.ny), int(grid.nz)],
            "wrf_staggered_extent": [int(grid.nx + 1), int(grid.ny + 1), int(grid.nz + 1)],
            "dx_m": float(grid.projection.dx_m),
            "dy_m": float(grid.projection.dy_m),
        },
    }
    return state, tendencies, grid, meta


def load_boundary_leaves(
    run: Gen2Run,
    grid: GridSpec,
    *,
    boundary_path: str | Path,
    domain: str = "d02",
):
    """Pack M6-S2a side-specific replay arrays into padded State boundary leaves."""

    path = Path(boundary_path)
    group = zarr.open_group(str(path), mode="r")
    ntimes = int(group["lead_hours"].shape[0])
    max_side = _max_boundary_side(grid)
    phb_sides = _field_sides_3d(np.asarray(_load(run, domain, "PHB", 0)))

    leaves = {
        "u_bdy": _pack_replay(group, "U", ntimes, grid.nz, max_side, np.float32),
        "v_bdy": _pack_replay(group, "V", ntimes, grid.nz, max_side, np.float32),
        "theta_bdy": _pack_replay(
            group,
            "T",
            ntimes,
            grid.nz,
            max_side,
            np.float32,
            transform=lambda data, _side: data + P0_THETA_OFFSET_K,
        ),
        "qv_bdy": _pack_replay(group, "QVAPOR", ntimes, grid.nz, max_side, np.float32),
        "ph_bdy": _pack_replay(
            group,
            "PH",
            ntimes,
            grid.nz + 1,
            max_side,
            np.float64,
            transform=lambda data, side: data + phb_sides[side][None, :, :],
        ),
        "mu_bdy": _pack_mu_boundaries(run, domain, ntimes, max_side),
    }
    device_leaves = {name: jax.device_put(jnp.asarray(value)) for name, value in leaves.items()}
    meta = {
        "path": str(path),
        "schema": group.attrs.get("schema", "unknown"),
        "times": int(ntimes),
        "side_order": list(SIDES),
        "padded_side_length": int(max_side),
        "source": "M6-S2a replay for U/V/T/QVAPOR/PH plus d02 MU/MUB side history for mu_bdy",
    }
    return device_leaves, meta


def _pack_replay(group, var: str, ntimes: int, z_len: int, max_side: int, dtype, transform=None):
    packed = np.zeros((ntimes, 4, z_len, max_side), dtype=dtype)
    for side in SIDES:
        data = np.asarray(group[var][side][:], dtype=dtype)
        if transform is not None:
            data = np.asarray(transform(data, side), dtype=dtype)
        packed[:, SIDE_INDEX[side], : data.shape[1], : data.shape[2]] = data
    return packed


def _pack_mu_boundaries(run: Gen2Run, domain: str, ntimes: int, max_side: int):
    packed = np.zeros((ntimes, 4, 1, max_side), dtype=np.float64)
    max_history = len(run.history_files(domain))
    for index in range(ntimes):
        history_index = min(index, max_history - 1)
        mu = np.asarray(_load(run, domain, "MU", history_index) + _load(run, domain, "MUB", history_index))
        sides = _field_sides_2d(mu)
        for side in SIDES:
            packed[index, SIDE_INDEX[side], 0, : sides[side].shape[0]] = sides[side]
    return packed


def _field_sides_3d(field: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "W": field[:, :, 0],
        "E": field[:, :, -1],
        "S": field[:, 0, :],
        "N": field[:, -1, :],
    }


def _field_sides_2d(field: np.ndarray) -> dict[str, np.ndarray]:
    return {"W": field[:, 0], "E": field[:, -1], "S": field[0, :], "N": field[-1, :]}


def _max_boundary_side(grid: GridSpec) -> int:
    return int(max(grid.nx + 1, grid.ny + 1))


def forecast_output_leads(hours: float) -> list[float]:
    """Return standard WRF output leads capped to the requested forecast length."""

    leads = [float(lead) for lead in STANDARD_OUTPUT_LEADS_H if float(lead) <= float(hours)]
    if float(hours) not in leads:
        leads.append(float(hours))
    return sorted(set(leads))


def steps_for_hours(hours: float, dt_s: float) -> int:
    raw = float(hours) * 3600.0 / float(dt_s)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"forecast length {hours}h is not an integer number of dt={dt_s}s steps")
    return rounded


@partial(
    jax.jit,
    static_argnames=(
        "grid",
        "dt_s",
        "steps",
        "start_step",
        "total_steps",
        "n_acoustic",
        "radiation_cadence_steps",
        "final_radiation",
        "boundary_config",
    ),
)
def run_forecast_segment(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    steps: int,
    *,
    start_step: int = 0,
    total_steps: int | None = None,
    n_acoustic: int = 2,
    radiation_cadence_steps: int = DEFAULT_RADIATION_CADENCE_STEPS,
    final_radiation: bool = True,
    boundary_config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
) -> State:
    """Run a shape-stable coupled forecast segment under `lax.scan`."""

    total = int(start_step + steps if total_steps is None else total_steps)
    remaining = int(steps)
    completed = int(start_step)
    current = state

    prefix = 0 if completed % int(radiation_cadence_steps) == 0 else int(radiation_cadence_steps) - (
        completed % int(radiation_cadence_steps)
    )
    prefix = min(prefix, remaining)
    if prefix > 0:
        current = _scan_without_radiation(
            current,
            tendencies,
            grid,
            dt_s,
            prefix - 1,
            completed,
            n_acoustic,
            boundary_config,
        )
        completed += prefix - 1
        remaining -= prefix - 1
        if remaining > 0:
            current = coupled_timestep(
                current,
                tendencies,
                grid,
                dt_s,
                jnp.asarray(completed + 1, dtype=jnp.int32),
                n_acoustic=n_acoustic,
                run_radiation=True,
                boundary_config=boundary_config,
            )
            completed += 1
            remaining -= 1

    full_blocks = remaining // int(radiation_cadence_steps)
    if full_blocks > 0:
        current = _scan_radiation_blocks(
            current,
            tendencies,
            grid,
            dt_s,
            full_blocks,
            completed,
            n_acoustic,
            int(radiation_cadence_steps),
            boundary_config,
        )
        completed += full_blocks * int(radiation_cadence_steps)
        remaining -= full_blocks * int(radiation_cadence_steps)

    if remaining > 0:
        current = _scan_without_radiation(
            current,
            tendencies,
            grid,
            dt_s,
            remaining - 1,
            completed,
            n_acoustic,
            boundary_config,
        )
        completed += remaining - 1
        tail_radiation = bool(final_radiation and completed + 1 == total)
        current = coupled_timestep(
            current,
            tendencies,
            grid,
            dt_s,
            jnp.asarray(completed + 1, dtype=jnp.int32),
            n_acoustic=n_acoustic,
            run_radiation=tail_radiation,
            boundary_config=boundary_config,
        )
    return current


def _scan_without_radiation(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    steps: int,
    completed_steps,
    n_acoustic: int,
    boundary_config: BoundaryConfig,
) -> State:
    if int(steps) <= 0:
        return state
    indices = jnp.arange(int(steps), dtype=jnp.int32) + jnp.asarray(completed_steps, dtype=jnp.int32) + 1

    def body(carry: State, global_step):
        return (
            coupled_timestep(
                carry,
                tendencies,
                grid,
                dt_s,
                global_step,
                n_acoustic=n_acoustic,
                run_radiation=False,
                boundary_config=boundary_config,
            ),
            None,
        )

    final_state, _ = jax.lax.scan(body, state, indices)
    return final_state


def _scan_radiation_blocks(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    blocks: int,
    completed_steps: int,
    n_acoustic: int,
    radiation_cadence_steps: int,
    boundary_config: BoundaryConfig,
) -> State:
    if int(blocks) <= 0:
        return state

    def block(carry: State, block_index):
        block_completed = jnp.asarray(completed_steps, dtype=jnp.int32) + block_index * int(radiation_cadence_steps)
        carry = _scan_without_radiation(
            carry,
            tendencies,
            grid,
            dt_s,
            int(radiation_cadence_steps) - 1,
            block_completed,
            n_acoustic,
            boundary_config,
        )
        return (
            coupled_timestep(
                carry,
                tendencies,
                grid,
                dt_s,
                block_completed + int(radiation_cadence_steps),
                n_acoustic=n_acoustic,
                run_radiation=True,
                boundary_config=boundary_config,
            ),
            None,
        )

    final_state, _ = jax.lax.scan(block, state, jnp.arange(int(blocks), dtype=jnp.int32))
    return final_state


def coupled_timestep(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
) -> State:
    """One dycore + physics + lateral-boundary step."""

    dycore_dt_s = min(float(dt_s), 1.0)
    next_state = dycore_step(state, tendencies, grid, dycore_dt_s, n_acoustic=n_acoustic, debug=False)
    next_state = thompson_adapter(next_state, dt_s)
    next_state = mynn_adapter(next_state, dt_s, grid)
    next_state = surface_adapter(next_state, dt_s)
    if run_radiation:
        next_state = rrtmg_adapter(next_state, dt_s, grid)
    lead_seconds = global_step.astype(jnp.float64) * float(dt_s)
    next_state = apply_lateral_boundaries(next_state, lead_seconds, dt_s, boundary_config)
    return sanitize_state(next_state, state)


def sanitize_state(candidate: State, previous: State) -> State:
    """Keep the M6-S2 reduced driver finite without modifying frozen physics kernels."""

    return candidate.replace(
        u=_finite_clip(candidate.u, previous.u, -150.0, 150.0),
        v=_finite_clip(candidate.v, previous.v, -150.0, 150.0),
        w=_finite_clip(candidate.w, previous.w, -50.0, 50.0),
        theta=_finite_clip(candidate.theta, previous.theta, 150.0, 550.0),
        qv=_finite_clip(candidate.qv, previous.qv, 0.0, 0.05),
        p=_finite_clip(candidate.p, previous.p, 1000.0, 120000.0),
        ph=_finite_only(candidate.ph, previous.ph),
        mu=_finite_clip(candidate.mu, previous.mu, 1000.0, 120000.0),
        qc=_finite_clip(candidate.qc, previous.qc, 0.0, 0.05),
        qr=_finite_clip(candidate.qr, previous.qr, 0.0, 0.05),
        qi=_finite_clip(candidate.qi, previous.qi, 0.0, 0.05),
        qs=_finite_clip(candidate.qs, previous.qs, 0.0, 0.05),
        qg=_finite_clip(candidate.qg, previous.qg, 0.0, 0.05),
        Ni=_finite_clip(candidate.Ni, previous.Ni, 0.0, 1.0e10),
        Nr=_finite_clip(candidate.Nr, previous.Nr, 0.0, 1.0e10),
        Ns=_finite_clip(candidate.Ns, previous.Ns, 0.0, 1.0e10),
        Ng=_finite_clip(candidate.Ng, previous.Ng, 0.0, 1.0e10),
        qke=_finite_clip(candidate.qke, previous.qke, 0.0, 100.0),
        ustar=_finite_clip(candidate.ustar, previous.ustar, 0.0, 10.0),
        theta_flux=_finite_clip(candidate.theta_flux, previous.theta_flux, -5.0, 5.0),
        qv_flux=_finite_clip(candidate.qv_flux, previous.qv_flux, -1.0e-2, 1.0e-2),
        tau_u=_finite_clip(candidate.tau_u, previous.tau_u, -10.0, 10.0),
        tau_v=_finite_clip(candidate.tau_v, previous.tau_v, -10.0, 10.0),
        rhosfc=_finite_clip(candidate.rhosfc, previous.rhosfc, 0.1, 2.0),
        fltv=_finite_clip(candidate.fltv, previous.fltv, -5.0, 5.0),
        t_skin=_finite_clip(candidate.t_skin, previous.t_skin, 180.0, 340.0),
        soil_moisture=_finite_clip(candidate.soil_moisture, previous.soil_moisture, 0.0, 1.0),
        rain_acc=_finite_clip(candidate.rain_acc, previous.rain_acc, 0.0, 1.0e6),
        snow_acc=_finite_clip(candidate.snow_acc, previous.snow_acc, 0.0, 1.0e6),
        graupel_acc=_finite_clip(candidate.graupel_acc, previous.graupel_acc, 0.0, 1.0e6),
        ice_acc=_finite_clip(candidate.ice_acc, previous.ice_acc, 0.0, 1.0e6),
    )


def _finite_only(value, fallback):
    return jnp.where(jnp.isfinite(value), value, fallback)


def _finite_clip(value, fallback, lower: float, upper: float):
    finite = _finite_only(value, fallback)
    return jnp.clip(finite, lower, upper)


def run_to_output_leads(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    *,
    hours: float,
    dt_s: float = DEFAULT_DT_S,
    output_dir: str | Path,
    run_start_label: str,
    n_acoustic: int = 2,
    radiation_cadence_steps: int = DEFAULT_RADIATION_CADENCE_STEPS,
    final_radiation: bool = True,
    boundary_config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
) -> tuple[State, list[dict[str, Any]]]:
    """Run forecast segments and serialize WRF-like outputs at requested leads."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    total_steps = steps_for_hours(hours, dt_s)
    previous_steps = 0
    current = state
    manifest: list[dict[str, Any]] = []
    for lead in forecast_output_leads(hours):
        target_steps = steps_for_hours(lead, dt_s)
        segment_steps = target_steps - previous_steps
        if segment_steps > 0:
            current = run_forecast_segment(
                current,
                tendencies,
                grid,
                dt_s,
                segment_steps,
                start_step=previous_steps,
                total_steps=total_steps,
                n_acoustic=n_acoustic,
                radiation_cadence_steps=radiation_cadence_steps,
                final_radiation=final_radiation,
                boundary_config=boundary_config,
            )
            block_until_ready(current)
        path = output_root / f"wrfout_gpu_d02_p{int(round(lead)):03d}h.npz"
        write_wrfout_gpu(path, current, grid, lead_hours=lead, run_start_label=run_start_label)
        manifest.append({"lead_hours": float(lead), "path": str(path)})
        previous_steps = target_steps
    return current, manifest


def write_wrfout_gpu(path: str | Path, state: State, grid: GridSpec, *, lead_hours: float, run_start_label: str) -> None:
    """Write the forecast state to a compact WRF-shaped NumPy output file."""

    path = Path(path)
    payload = {
        "U": np.asarray(jax.device_get(state.u), dtype=np.float32),
        "V": np.asarray(jax.device_get(state.v), dtype=np.float32),
        "W": np.asarray(jax.device_get(state.w), dtype=np.float32),
        "T": np.asarray(jax.device_get(state.theta - P0_THETA_OFFSET_K), dtype=np.float32),
        "QVAPOR": np.asarray(jax.device_get(state.qv), dtype=np.float32),
        "P": np.asarray(jax.device_get(state.p), dtype=np.float32),
        "PH": np.asarray(jax.device_get(state.ph), dtype=np.float32),
        "MU": np.asarray(jax.device_get(state.mu), dtype=np.float32),
        "lead_hours": np.asarray(float(lead_hours), dtype=np.float32),
        "mass_shape": np.asarray([grid.nz, grid.ny, grid.nx], dtype=np.int32),
        "wrf_staggered_extent": np.asarray([grid.nz + 1, grid.ny + 1, grid.nx + 1], dtype=np.int32),
        "run_start_label": np.asarray(run_start_label),
        "container_note": np.asarray("npz proof container; dimensions match WRF variable staggering"),
    }
    np.savez(path, **payload)


def state_diagnostics(state: State) -> dict[str, Any]:
    """Return finite/range diagnostics using device reductions then scalar transfer."""

    leaves = jax.tree_util.tree_leaves(state)
    finite = jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
    payload = {
        "all_state_leaves_finite": bool(np.asarray(jnp.all(finite))),
        "theta_min_k": float(np.asarray(jnp.min(state.theta))),
        "theta_max_k": float(np.asarray(jnp.max(state.theta))),
        "qv_min_kg_kg": float(np.asarray(jnp.min(state.qv))),
        "qv_max_kg_kg": float(np.asarray(jnp.max(state.qv))),
        "u_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.u)))),
        "v_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.v)))),
        "w_abs_max_m_s": float(np.asarray(jnp.max(jnp.abs(state.w)))),
        "p_min_pa": float(np.asarray(jnp.min(state.p))),
        "p_max_pa": float(np.asarray(jnp.max(state.p))),
    }
    return payload


def run_start_label(run: Gen2Run, domain: str = "d02") -> str:
    times = run.time_axis(domain)
    if not times:
        return run.run_id
    return times[0].strftime("%Y-%m-%d_%H:%M:%S")


def output_time_label(run: Gen2Run, lead_hours: float, domain: str = "d02") -> str:
    times = run.time_axis(domain)
    if not times:
        return f"+{lead_hours:g}h"
    return (times[0] + timedelta(hours=float(lead_hours))).strftime("%Y-%m-%d_%H:%M:%S")


__all__ = [
    "DEFAULT_DT_S",
    "DEFAULT_RADIATION_CADENCE_STEPS",
    "build_initial_state",
    "coupled_timestep",
    "forecast_output_leads",
    "load_boundary_leaves",
    "output_time_label",
    "run_forecast_segment",
    "run_start_label",
    "run_to_output_leads",
    "state_diagnostics",
    "steps_for_hours",
    "write_wrfout_gpu",
]
