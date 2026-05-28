"""M6-S2 coupled d02 forecast driver on the GPU-resident State pytree."""

from __future__ import annotations

from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import zarr

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import BoundaryConfig, DEFAULT_BOUNDARY_CONFIG, SIDE_INDEX, SIDES, apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.acoustic import forward_backward_acoustic
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.dynamics.step import step as dycore_step
from gpuwrf.io.gen2_accessor import DEFAULT_M6_BOUNDARY_REPLAY, Gen2Run
from gpuwrf.io.land_state import load_prescribed_land_state
from gpuwrf.io.validation import load_gen2_var
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

P0_THETA_OFFSET_K = 300.0
GRAVITY_M_S2 = 9.80665
DEFAULT_DT_S = 10.0
MAX_LIFTED_DYCORE_DT_S = 12.0
DEFAULT_RADIATION_CADENCE_STEPS = 60
STANDARD_OUTPUT_LEADS_H = (1, 6, 12, 18, 24)


class BoundarySnapshot(NamedTuple):
    """Boundary-relevant leaves captured immediately before boundary replay."""

    u: object
    v: object
    theta: object
    qv: object
    ph: object
    mu: object


class PreSanitizeTap(NamedTuple):
    """Per-step state captured before `sanitize_state`, plus boundary replay terms."""

    state: State
    pre_boundary: BoundarySnapshot
    boundary_tendency: BoundarySnapshot


class SanitizeStats(NamedTuple):
    """Scalar counts describing whether `sanitize_state` changed a candidate."""

    nonfinite_count: object
    clip_count: object
    changed_count: object
    total_count: object


class BisectionConfig(NamedTuple):
    """Static diagnostic switches for empirical instability bisection."""

    disable_sanitize: bool = False
    disable_thompson: bool = False
    disable_mynn: bool = False
    disable_surface: bool = False
    disable_rrtmg: bool = False
    disable_boundary: bool = False
    disable_advection: bool = False
    disable_acoustic: bool = False
    disable_mu_continuity: bool = False


DEFAULT_BISECTION_CONFIG = BisectionConfig()


def _load(run: Gen2Run, domain: str, var: str, time: int = 0):
    """Route every Gen2 variable read through the shared validation I/O path."""

    return load_gen2_var(run, domain, var, time=time)


def build_initial_state(
    run: Gen2Run,
    *,
    domain: str = "d02",
    boundary_path: str | Path = DEFAULT_M6_BOUNDARY_REPLAY,
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
    land = load_prescribed_land_state(run, domain=domain, time=0)

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
        t_skin=land.t_skin,
        soil_moisture=land.soil_moisture[0],
        xland=land.xland,
        lakemask=land.lakemask,
        mavail=land.mavail,
        roughness_m=land.roughness_m,
        **bdy_leaves,
    )
    tendencies = Tendencies.zeros(grid)
    meta = {
        "domain": domain,
        "run_id": run.run_id,
        "boundary": boundary_meta,
        "prescribed_land": {
            "source_file": land.source.get("source_file"),
            "roughness_note": land.source.get("roughness_note"),
            "mavail_note": land.source.get("mavail_note"),
            "missing_optional_variables": land.source.get("missing_optional_variables", []),
        },
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


def validate_lifted_coupled_dt(dt_s: float) -> float:
    """Return a Path-B coupled dt that is safe to pass directly to the dycore."""

    value = float(dt_s)
    if value <= 0.0:
        raise ValueError(f"coupled dt_s must be positive, got {dt_s}")
    if value > MAX_LIFTED_DYCORE_DT_S:
        raise ValueError(
            f"coupled dt_s={value:g}s exceeds the M6-S5 Path-B dycore limit of "
            f"{MAX_LIFTED_DYCORE_DT_S:g}s; lower dt_s instead of reintroducing a cap"
        )
    return value


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
        "capture_pre_sanitize",
        "bisection_config",
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
    capture_pre_sanitize: bool = False,
    bisection_config: BisectionConfig = DEFAULT_BISECTION_CONFIG,
) -> State | tuple[State, PreSanitizeTap]:
    """Run a shape-stable coupled forecast segment under `lax.scan`."""

    if bool(capture_pre_sanitize):
        return _run_forecast_segment_with_pre_sanitize_tap(
            state,
            tendencies,
            grid,
            dt_s,
            steps,
            start_step=start_step,
            total_steps=total_steps,
            n_acoustic=n_acoustic,
            radiation_cadence_steps=radiation_cadence_steps,
            final_radiation=final_radiation,
            boundary_config=boundary_config,
            bisection_config=bisection_config,
        )

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
            bisection_config,
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
                bisection_config=bisection_config,
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
            bisection_config,
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
            bisection_config,
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
            bisection_config=bisection_config,
        )
    return current


def _run_forecast_segment_with_pre_sanitize_tap(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    steps: int,
    *,
    start_step: int,
    total_steps: int | None,
    n_acoustic: int,
    radiation_cadence_steps: int,
    final_radiation: bool,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig,
) -> tuple[State, PreSanitizeTap]:
    """Run the same static cadence segmentation while returning pre-sanitize states."""

    total = int(start_step + steps if total_steps is None else total_steps)
    remaining = int(steps)
    completed = int(start_step)
    current = state
    taps: list[PreSanitizeTap] = []

    prefix = 0 if completed % int(radiation_cadence_steps) == 0 else int(radiation_cadence_steps) - (
        completed % int(radiation_cadence_steps)
    )
    prefix = min(prefix, remaining)
    if prefix > 0:
        if prefix - 1 > 0:
            current, tap = _scan_without_radiation_tap(
                current,
                tendencies,
                grid,
                dt_s,
                prefix - 1,
                completed,
                n_acoustic,
                boundary_config,
                bisection_config,
            )
            taps.append(tap)
            completed += prefix - 1
            remaining -= prefix - 1
        if remaining > 0:
            current, tap = coupled_timestep_with_pre_sanitize(
                current,
                tendencies,
                grid,
                dt_s,
                jnp.asarray(completed + 1, dtype=jnp.int32),
                n_acoustic=n_acoustic,
                run_radiation=True,
                boundary_config=boundary_config,
                bisection_config=bisection_config,
            )
            taps.append(_stack_single_tap(tap))
            completed += 1
            remaining -= 1

    full_blocks = remaining // int(radiation_cadence_steps)
    if full_blocks > 0:
        current, tap = _scan_radiation_blocks_tap(
            current,
            tendencies,
            grid,
            dt_s,
            full_blocks,
            completed,
            n_acoustic,
            int(radiation_cadence_steps),
            boundary_config,
            bisection_config,
        )
        taps.append(tap)
        completed += full_blocks * int(radiation_cadence_steps)
        remaining -= full_blocks * int(radiation_cadence_steps)

    if remaining > 0:
        if remaining - 1 > 0:
            current, tap = _scan_without_radiation_tap(
                current,
                tendencies,
                grid,
                dt_s,
                remaining - 1,
                completed,
                n_acoustic,
                boundary_config,
                bisection_config,
            )
            taps.append(tap)
            completed += remaining - 1
        tail_radiation = bool(final_radiation and completed + 1 == total)
        current, tap = coupled_timestep_with_pre_sanitize(
            current,
            tendencies,
            grid,
            dt_s,
            jnp.asarray(completed + 1, dtype=jnp.int32),
            n_acoustic=n_acoustic,
            run_radiation=tail_radiation,
            boundary_config=boundary_config,
            bisection_config=bisection_config,
        )
        taps.append(_stack_single_tap(tap))

    if not taps:
        return current, _empty_tap_like(state)
    return current, _concat_taps(taps)


def _scan_without_radiation(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    steps: int,
    completed_steps,
    n_acoustic: int,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig,
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
                bisection_config=bisection_config,
            ),
            None,
        )

    final_state, _ = jax.lax.scan(body, state, indices)
    return final_state


def _scan_without_radiation_tap(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    steps: int,
    completed_steps,
    n_acoustic: int,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig,
) -> tuple[State, PreSanitizeTap]:
    if int(steps) <= 0:
        return state, _empty_tap_like(state)
    indices = jnp.arange(int(steps), dtype=jnp.int32) + jnp.asarray(completed_steps, dtype=jnp.int32) + 1

    def body(carry: State, global_step):
        return coupled_timestep_with_pre_sanitize(
            carry,
            tendencies,
            grid,
            dt_s,
            global_step,
            n_acoustic=n_acoustic,
            run_radiation=False,
            boundary_config=boundary_config,
            bisection_config=bisection_config,
        )

    final_state, tap = jax.lax.scan(body, state, indices)
    return final_state, tap


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
    bisection_config: BisectionConfig,
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
            bisection_config,
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
                bisection_config=bisection_config,
            ),
            None,
        )

    final_state, _ = jax.lax.scan(block, state, jnp.arange(int(blocks), dtype=jnp.int32))
    return final_state


def _scan_radiation_blocks_tap(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    blocks: int,
    completed_steps: int,
    n_acoustic: int,
    radiation_cadence_steps: int,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig,
) -> tuple[State, PreSanitizeTap]:
    if int(blocks) <= 0:
        return state, _empty_tap_like(state)

    def block(carry: State, block_index):
        block_completed = jnp.asarray(completed_steps, dtype=jnp.int32) + block_index * int(radiation_cadence_steps)
        carry, no_rad_tap = _scan_without_radiation_tap(
            carry,
            tendencies,
            grid,
            dt_s,
            int(radiation_cadence_steps) - 1,
            block_completed,
            n_acoustic,
            boundary_config,
            bisection_config,
        )
        carry, rad_tap = coupled_timestep_with_pre_sanitize(
            carry,
            tendencies,
            grid,
            dt_s,
            block_completed + int(radiation_cadence_steps),
            n_acoustic=n_acoustic,
            run_radiation=True,
            boundary_config=boundary_config,
            bisection_config=bisection_config,
        )
        return carry, _concat_taps((no_rad_tap, _stack_single_tap(rad_tap)))

    final_state, block_taps = jax.lax.scan(block, state, jnp.arange(int(blocks), dtype=jnp.int32))
    flat_taps = jax.tree_util.tree_map(
        lambda leaf: leaf.reshape((int(blocks) * int(radiation_cadence_steps),) + leaf.shape[2:]),
        block_taps,
    )
    return final_state, flat_taps


def _boundary_snapshot(state: State) -> BoundarySnapshot:
    return BoundarySnapshot(state.u, state.v, state.theta, state.qv, state.ph, state.mu)


def _boundary_tendency(after: State, before: State, dt_s: float) -> BoundarySnapshot:
    inv_dt = 1.0 / float(dt_s)
    return BoundarySnapshot(
        (after.u - before.u) * inv_dt,
        (after.v - before.v) * inv_dt,
        (after.theta - before.theta) * inv_dt,
        (after.qv - before.qv) * inv_dt,
        (after.ph - before.ph) * inv_dt,
        (after.mu - before.mu) * inv_dt,
    )


def _stack_single_tap(tap: PreSanitizeTap) -> PreSanitizeTap:
    return jax.tree_util.tree_map(lambda leaf: jnp.expand_dims(leaf, axis=0), tap)


def _empty_tap_like(state: State) -> PreSanitizeTap:
    tap = PreSanitizeTap(
        state=state,
        pre_boundary=_boundary_snapshot(state),
        boundary_tendency=_boundary_snapshot(state.replace(
            u=jnp.zeros_like(state.u),
            v=jnp.zeros_like(state.v),
            theta=jnp.zeros_like(state.theta),
            qv=jnp.zeros_like(state.qv),
            ph=jnp.zeros_like(state.ph),
            mu=jnp.zeros_like(state.mu),
        )),
    )
    return jax.tree_util.tree_map(lambda leaf: leaf[None, ...][:0], tap)


def _concat_taps(taps) -> PreSanitizeTap:
    taps = tuple(taps)
    if len(taps) == 1:
        return taps[0]
    return jax.tree_util.tree_map(lambda *leaves: jnp.concatenate(leaves, axis=0), *taps)


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
    bisection_config: BisectionConfig = DEFAULT_BISECTION_CONFIG,
) -> State:
    """One dycore + physics + lateral-boundary step."""

    candidate = _candidate_timestep(
        state,
        tendencies,
        grid,
        dt_s,
        global_step,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        boundary_config=boundary_config,
        bisection_config=bisection_config,
    )
    if bool(bisection_config.disable_sanitize):
        return candidate
    return sanitize_state(candidate, state)


def coupled_timestep_with_pre_sanitize(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig = DEFAULT_BISECTION_CONFIG,
) -> tuple[State, PreSanitizeTap]:
    """One coupled step returning the state immediately before `sanitize_state`."""

    before_boundary = _candidate_before_boundary(
        state,
        tendencies,
        grid,
        dt_s,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        bisection_config=bisection_config,
    )
    lead_seconds = global_step.astype(jnp.float64) * float(dt_s)
    if bool(bisection_config.disable_boundary):
        candidate = before_boundary
    else:
        candidate = apply_lateral_boundaries(before_boundary, lead_seconds, dt_s, boundary_config)
    tap = PreSanitizeTap(
        state=candidate,
        pre_boundary=_boundary_snapshot(before_boundary),
        boundary_tendency=_boundary_tendency(candidate, before_boundary, dt_s),
    )
    if bool(bisection_config.disable_sanitize):
        return candidate, tap
    return sanitize_state(candidate, state), tap


def coupled_timestep_with_sanitize_stats(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig = DEFAULT_BISECTION_CONFIG,
) -> tuple[State, SanitizeStats]:
    """One coupled step returning scalar sanitize-change counts."""

    candidate = _candidate_timestep(
        state,
        tendencies,
        grid,
        dt_s,
        global_step,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        boundary_config=boundary_config,
        bisection_config=bisection_config,
    )
    if bool(bisection_config.disable_sanitize):
        zero = jnp.asarray(0, dtype=jnp.int64)
        total = sum((leaf.size for leaf in jax.tree_util.tree_leaves(candidate)), 0)
        return candidate, SanitizeStats(zero, zero, zero, jnp.asarray(total, dtype=jnp.int64))
    return sanitize_state_with_stats(candidate, state)


def _candidate_timestep(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    global_step,
    *,
    n_acoustic: int,
    run_radiation: bool,
    boundary_config: BoundaryConfig,
    bisection_config: BisectionConfig = DEFAULT_BISECTION_CONFIG,
) -> State:
    before_boundary = _candidate_before_boundary(
        state,
        tendencies,
        grid,
        dt_s,
        n_acoustic=n_acoustic,
        run_radiation=run_radiation,
        bisection_config=bisection_config,
    )
    if bool(bisection_config.disable_boundary):
        return before_boundary
    lead_seconds = global_step.astype(jnp.float64) * float(dt_s)
    return apply_lateral_boundaries(before_boundary, lead_seconds, dt_s, boundary_config)


def _candidate_before_boundary(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt_s: float,
    *,
    n_acoustic: int,
    run_radiation: bool,
    bisection_config: BisectionConfig = DEFAULT_BISECTION_CONFIG,
) -> State:
    dycore_dt_s = validate_lifted_coupled_dt(dt_s)
    next_state = _dycore_step_for_bisection(
        state,
        tendencies,
        grid,
        dycore_dt_s,
        n_acoustic=n_acoustic,
        bisection_config=bisection_config,
    )
    if not bool(bisection_config.disable_thompson):
        next_state = thompson_adapter(next_state, dt_s)
    if not bool(bisection_config.disable_mynn):
        next_state = mynn_adapter(next_state, dt_s, grid)
    if not bool(bisection_config.disable_surface):
        next_state = surface_adapter(next_state, dt_s)
    if run_radiation and not bool(bisection_config.disable_rrtmg):
        next_state = rrtmg_adapter(next_state, dt_s, grid)
    return next_state


def _dycore_step_for_bisection(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt: float,
    *,
    n_acoustic: int,
    bisection_config: BisectionConfig,
) -> State:
    if not (
        bool(bisection_config.disable_advection)
        or bool(bisection_config.disable_acoustic)
        or bool(bisection_config.disable_mu_continuity)
    ):
        return dycore_step(state, tendencies, grid, dt, n_acoustic=n_acoustic, debug=False)

    s0 = apply_halo(state, halo_spec(grid))
    s1 = _rk3_bisection_stage(
        s0,
        s0,
        tendencies,
        grid,
        dt / 3.0,
        disable_advection=bool(bisection_config.disable_advection),
    )
    s1 = apply_halo(s1, halo_spec(grid))
    s2 = _rk3_bisection_stage(
        s0,
        s1,
        tendencies,
        grid,
        dt / 2.0,
        disable_advection=bool(bisection_config.disable_advection),
    )
    if not bool(bisection_config.disable_acoustic):
        s2 = forward_backward_acoustic(s2, grid, dt / 2.0, n_acoustic)
    s2 = apply_halo(s2, halo_spec(grid))
    s3 = _rk3_bisection_stage(
        s0,
        s2,
        tendencies,
        grid,
        dt,
        disable_advection=bool(bisection_config.disable_advection),
    )
    if not bool(bisection_config.disable_acoustic):
        s3 = forward_backward_acoustic(s3, grid, dt, n_acoustic)
    s3 = apply_halo(s3, halo_spec(grid))
    if bool(bisection_config.disable_mu_continuity):
        s3 = s3.replace(mu=state.mu)
    return s3


def _rk3_bisection_stage(
    origin: State,
    stage_state: State,
    base_tendencies: Tendencies,
    grid: GridSpec,
    dt_stage: float,
    *,
    disable_advection: bool,
) -> State:
    tendencies = base_tendencies if disable_advection else compute_advection_tendencies(stage_state, base_tendencies, grid)
    return add_scaled_tendencies(origin, tendencies, dt_stage)


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


def sanitize_state_with_stats(candidate: State, previous: State) -> tuple[State, SanitizeStats]:
    """Return sanitized state plus scalar nonfinite/clip counts for audit runs."""

    counts = []

    def finite_clip(value, fallback, lower: float, upper: float):
        sanitized, stats = _finite_clip_with_stats(value, fallback, lower, upper)
        counts.append(stats)
        return sanitized

    def finite_only(value, fallback):
        sanitized, stats = _finite_only_with_stats(value, fallback)
        counts.append(stats)
        return sanitized

    state = candidate.replace(
        u=finite_clip(candidate.u, previous.u, -150.0, 150.0),
        v=finite_clip(candidate.v, previous.v, -150.0, 150.0),
        w=finite_clip(candidate.w, previous.w, -50.0, 50.0),
        theta=finite_clip(candidate.theta, previous.theta, 150.0, 550.0),
        qv=finite_clip(candidate.qv, previous.qv, 0.0, 0.05),
        p=finite_clip(candidate.p, previous.p, 1000.0, 120000.0),
        ph=finite_only(candidate.ph, previous.ph),
        mu=finite_clip(candidate.mu, previous.mu, 1000.0, 120000.0),
        qc=finite_clip(candidate.qc, previous.qc, 0.0, 0.05),
        qr=finite_clip(candidate.qr, previous.qr, 0.0, 0.05),
        qi=finite_clip(candidate.qi, previous.qi, 0.0, 0.05),
        qs=finite_clip(candidate.qs, previous.qs, 0.0, 0.05),
        qg=finite_clip(candidate.qg, previous.qg, 0.0, 0.05),
        Ni=finite_clip(candidate.Ni, previous.Ni, 0.0, 1.0e10),
        Nr=finite_clip(candidate.Nr, previous.Nr, 0.0, 1.0e10),
        Ns=finite_clip(candidate.Ns, previous.Ns, 0.0, 1.0e10),
        Ng=finite_clip(candidate.Ng, previous.Ng, 0.0, 1.0e10),
        qke=finite_clip(candidate.qke, previous.qke, 0.0, 100.0),
        ustar=finite_clip(candidate.ustar, previous.ustar, 0.0, 10.0),
        theta_flux=finite_clip(candidate.theta_flux, previous.theta_flux, -5.0, 5.0),
        qv_flux=finite_clip(candidate.qv_flux, previous.qv_flux, -1.0e-2, 1.0e-2),
        tau_u=finite_clip(candidate.tau_u, previous.tau_u, -10.0, 10.0),
        tau_v=finite_clip(candidate.tau_v, previous.tau_v, -10.0, 10.0),
        rhosfc=finite_clip(candidate.rhosfc, previous.rhosfc, 0.1, 2.0),
        fltv=finite_clip(candidate.fltv, previous.fltv, -5.0, 5.0),
        t_skin=finite_clip(candidate.t_skin, previous.t_skin, 180.0, 340.0),
        soil_moisture=finite_clip(candidate.soil_moisture, previous.soil_moisture, 0.0, 1.0),
        rain_acc=finite_clip(candidate.rain_acc, previous.rain_acc, 0.0, 1.0e6),
        snow_acc=finite_clip(candidate.snow_acc, previous.snow_acc, 0.0, 1.0e6),
        graupel_acc=finite_clip(candidate.graupel_acc, previous.graupel_acc, 0.0, 1.0e6),
        ice_acc=finite_clip(candidate.ice_acc, previous.ice_acc, 0.0, 1.0e6),
    )
    return state, _sum_sanitize_stats(counts)


def _finite_only(value, fallback):
    return jnp.where(jnp.isfinite(value), value, fallback)


def _finite_clip(value, fallback, lower: float, upper: float):
    finite = _finite_only(value, fallback)
    return jnp.clip(finite, lower, upper)


def _finite_only_with_stats(value, fallback):
    finite_mask = jnp.isfinite(value)
    sanitized = jnp.where(finite_mask, value, fallback)
    nonfinite = jnp.sum(~finite_mask, dtype=jnp.int64)
    return sanitized, SanitizeStats(
        nonfinite_count=nonfinite,
        clip_count=jnp.asarray(0, dtype=jnp.int64),
        changed_count=nonfinite,
        total_count=jnp.asarray(value.size, dtype=jnp.int64),
    )


def _finite_clip_with_stats(value, fallback, lower: float, upper: float):
    finite_mask = jnp.isfinite(value)
    finite = jnp.where(finite_mask, value, fallback)
    clipped_mask = finite_mask & ((finite < float(lower)) | (finite > float(upper)))
    sanitized = jnp.clip(finite, lower, upper)
    nonfinite = jnp.sum(~finite_mask, dtype=jnp.int64)
    clipped = jnp.sum(clipped_mask, dtype=jnp.int64)
    return sanitized, SanitizeStats(
        nonfinite_count=nonfinite,
        clip_count=clipped,
        changed_count=nonfinite + clipped,
        total_count=jnp.asarray(value.size, dtype=jnp.int64),
    )


def _sum_sanitize_stats(stats: list[SanitizeStats]) -> SanitizeStats:
    return SanitizeStats(
        nonfinite_count=sum((item.nonfinite_count for item in stats), jnp.asarray(0, dtype=jnp.int64)),
        clip_count=sum((item.clip_count for item in stats), jnp.asarray(0, dtype=jnp.int64)),
        changed_count=sum((item.changed_count for item in stats), jnp.asarray(0, dtype=jnp.int64)),
        total_count=sum((item.total_count for item in stats), jnp.asarray(0, dtype=jnp.int64)),
    )


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
    surface = _surface_diagnostics_for_output(state)
    payload = {
        "U": np.asarray(jax.device_get(state.u), dtype=np.float32),
        "V": np.asarray(jax.device_get(state.v), dtype=np.float32),
        "W": np.asarray(jax.device_get(state.w), dtype=np.float32),
        "T": np.asarray(jax.device_get(state.theta - P0_THETA_OFFSET_K), dtype=np.float32),
        "QVAPOR": np.asarray(jax.device_get(state.qv), dtype=np.float32),
        "P": np.asarray(jax.device_get(state.p), dtype=np.float32),
        "PH": np.asarray(jax.device_get(state.ph), dtype=np.float32),
        "MU": np.asarray(jax.device_get(state.mu), dtype=np.float32),
        "U10": np.asarray(jax.device_get(surface.u10), dtype=np.float32),
        "V10": np.asarray(jax.device_get(surface.v10), dtype=np.float32),
        "T2": np.asarray(jax.device_get(surface.t2), dtype=np.float32),
        "Q2": np.asarray(jax.device_get(surface.q2), dtype=np.float32),
        "UST": np.asarray(jax.device_get(surface.fluxes.ustar), dtype=np.float32),
        "HFX_KIN": np.asarray(jax.device_get(surface.fluxes.theta_flux), dtype=np.float32),
        "QFX_KIN": np.asarray(jax.device_get(surface.fluxes.qv_flux), dtype=np.float32),
        "lead_hours": np.asarray(float(lead_hours), dtype=np.float32),
        "mass_shape": np.asarray([grid.nz, grid.ny, grid.nx], dtype=np.int32),
        "wrf_staggered_extent": np.asarray([grid.nz + 1, grid.ny + 1, grid.nx + 1], dtype=np.int32),
        "run_start_label": np.asarray(run_start_label),
        "container_note": np.asarray("npz proof container; dimensions match WRF variable staggering"),
    }
    np.savez(path, **payload)


class _SurfaceOutputState(NamedTuple):
    u: object
    v: object
    theta: object
    qv: object
    p: object
    dz: object
    t_skin: object
    soil_moisture: object
    xland: object
    lakemask: object
    mavail: object
    roughness_m: object
    ustar: object


def _surface_diagnostics_for_output(state: State):
    u_mass = 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])
    v_mass = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    dz = jnp.maximum((state.ph[1:, :, :] - state.ph[:-1, :, :]) / GRAVITY_M_S2, 1.0)
    column_state = _SurfaceOutputState(
        u=jnp.moveaxis(u_mass, 0, -1),
        v=jnp.moveaxis(v_mass, 0, -1),
        theta=jnp.moveaxis(state.theta, 0, -1),
        qv=jnp.moveaxis(state.qv, 0, -1),
        p=jnp.moveaxis(state.p, 0, -1),
        dz=jnp.moveaxis(dz, 0, -1),
        t_skin=state.t_skin,
        soil_moisture=state.soil_moisture,
        xland=state.xland,
        lakemask=state.lakemask,
        mavail=state.mavail,
        roughness_m=state.roughness_m,
        ustar=state.ustar,
    )
    return surface_layer_with_diagnostics(column_state)


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
    "BisectionConfig",
    "BoundarySnapshot",
    "DEFAULT_BISECTION_CONFIG",
    "DEFAULT_DT_S",
    "DEFAULT_RADIATION_CADENCE_STEPS",
    "MAX_LIFTED_DYCORE_DT_S",
    "PreSanitizeTap",
    "SanitizeStats",
    "build_initial_state",
    "coupled_timestep",
    "coupled_timestep_with_pre_sanitize",
    "coupled_timestep_with_sanitize_stats",
    "forecast_output_leads",
    "load_boundary_leaves",
    "output_time_label",
    "run_forecast_segment",
    "run_start_label",
    "run_to_output_leads",
    "sanitize_state_with_stats",
    "state_diagnostics",
    "steps_for_hours",
    "validate_lifted_coupled_dt",
    "write_wrfout_gpu",
]
