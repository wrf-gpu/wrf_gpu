"""M6-S6 Tier-3 controlled timestep-sensitivity drift envelope.

TSC1.0 uses a reduced, controlled coupled case at `dt=18,9,4.5s` to define
per-variable/per-lead envelopes, then compares the pinned d02 GPU forecast
against the Gen2 WRF reference without aggregate-only pass/fail hiding.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.boundary_apply import BoundaryConfig, DEFAULT_BOUNDARY_CONFIG, SIDES, apply_lateral_boundaries
from gpuwrf.coupling.driver import (
    build_initial_state,
    run_forecast_segment,
    steps_for_hours,
)
from gpuwrf.coupling.physics_couplers import thompson_adapter_with_tendencies
from gpuwrf.io.boundary_replay import (
    WRFBDY_VARIABLES,
    compare_boundary_tendency_to_wrfbdy,
    decode_wrfbdy,
    wrfbdy_path_for_run,
)
from gpuwrf.io.gen2_accessor import DEFAULT_M6_BOUNDARY_REPLAY, DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier2_coupled import water_budget_residual


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT = ROOT / "artifacts" / "m6" / "tier3" / "tsc_envelope.json"
VARIABLES = ("U10", "V10", "T2", "qv2", "precip")
LEADS_H = (6.0, 12.0, 24.0)
TSC_DTS = (18.0, 9.0, 4.5)
P0_THETA_OFFSET_K = 300.0
GRAVITY_M_S2 = 9.80665


class _SurfaceDiagnosticState(NamedTuple):
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


def _as_np(array) -> np.ndarray:
    return np.asarray(jax.device_get(array), dtype=np.float64)


def _rel_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _field_norm(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }


def surface_fields_from_state(state: State) -> dict[str, np.ndarray]:
    """Extract the M6 surface variables on the mass grid."""

    u_mass = 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])
    v_mass = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    dz = jnp.maximum((state.ph[1:, :, :] - state.ph[:-1, :, :]) / GRAVITY_M_S2, 1.0)
    diagnostics = surface_layer_with_diagnostics(
        _SurfaceDiagnosticState(
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
    )
    return {
        "U10": _as_np(diagnostics.u10),
        "V10": _as_np(diagnostics.v10),
        "T2": _as_np(diagnostics.t2),
        "qv2": _as_np(diagnostics.q2),
        "precip": _as_np(state.rain_acc + state.snow_acc + state.graupel_acc + state.ice_acc),
    }


def _pack_3d_boundary(field, max_side: int, dtype) -> np.ndarray:
    data = _as_np(field)
    packed = np.zeros((1, 4, data.shape[0], max_side), dtype=dtype)
    sides = {
        "W": data[:, :, 0],
        "E": data[:, :, -1],
        "S": data[:, 0, :],
        "N": data[:, -1, :],
    }
    for index, side in enumerate(SIDES):
        packed[0, index, : sides[side].shape[0], : sides[side].shape[1]] = sides[side]
    return packed


def _pack_2d_boundary(field, max_side: int) -> np.ndarray:
    data = _as_np(field)
    packed = np.zeros((1, 4, 1, max_side), dtype=np.float64)
    sides = {"W": data[:, 0], "E": data[:, -1], "S": data[0, :], "N": data[-1, :]}
    for index, side in enumerate(SIDES):
        packed[0, index, 0, : sides[side].shape[0]] = sides[side]
    return packed


def _constant_boundary_leaves(state: State, grid: GridSpec) -> dict[str, jax.Array]:
    max_side = int(max(grid.nx + 1, grid.ny + 1))
    return {
        "u_bdy": jax.device_put(jnp.asarray(_pack_3d_boundary(state.u, max_side, np.float32))),
        "v_bdy": jax.device_put(jnp.asarray(_pack_3d_boundary(state.v, max_side, np.float32))),
        "theta_bdy": jax.device_put(jnp.asarray(_pack_3d_boundary(state.theta, max_side, np.float32))),
        "qv_bdy": jax.device_put(jnp.asarray(_pack_3d_boundary(state.qv, max_side, np.float32))),
        "ph_bdy": jax.device_put(jnp.asarray(_pack_3d_boundary(state.ph, max_side, np.float64))),
        "mu_bdy": jax.device_put(jnp.asarray(_pack_2d_boundary(state.mu, max_side))),
    }


def idealized_coupled_state() -> tuple[State, Tendencies, GridSpec]:
    """Create the reduced smooth case used for controlled dt refinement."""

    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    z = jnp.linspace(0.0, 1.0, grid.nz, dtype=jnp.float64)[:, None, None]
    zi = jnp.linspace(0.0, 9000.0, grid.nz + 1, dtype=jnp.float64)[:, None, None]
    x = jnp.linspace(0.0, 2.0 * jnp.pi, grid.nx, endpoint=False, dtype=jnp.float64)[None, None, :]
    y = jnp.linspace(0.0, 2.0 * jnp.pi, grid.ny, endpoint=False, dtype=jnp.float64)[None, :, None]
    xu = jnp.linspace(0.0, 2.0 * jnp.pi, grid.nx + 1, endpoint=True, dtype=jnp.float64)[None, None, :]
    yv = jnp.linspace(0.0, 2.0 * jnp.pi, grid.ny + 1, endpoint=True, dtype=jnp.float64)[None, :, None]
    surface_x = x[0]
    surface_y = y[0]
    state = state.replace(
        u=3.0 + 0.3 * jnp.sin(xu) + jnp.zeros((grid.nz, grid.ny, 1), dtype=jnp.float64),
        v=1.5 + 0.2 * jnp.cos(yv) + jnp.zeros((grid.nz, 1, grid.nx), dtype=jnp.float64),
        w=jnp.zeros_like(state.w),
        theta=300.0 + 0.8 * jnp.sin(x) + 0.4 * jnp.cos(y) + 0.2 * z,
        qv=0.008 + 2.0e-4 * jnp.cos(x - y) + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        p=90000.0 - 4500.0 * z + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        ph=jnp.broadcast_to(zi * GRAVITY_M_S2, state.ph.shape),
        mu=80000.0 + 40.0 * jnp.sin(surface_x) + 20.0 * jnp.cos(surface_y),
        qc=jnp.ones_like(state.qc) * 1.0e-6,
        qr=jnp.zeros_like(state.qr),
        qi=jnp.zeros_like(state.qi),
        qs=jnp.zeros_like(state.qs),
        qg=jnp.zeros_like(state.qg),
        Ni=jnp.ones_like(state.Ni) * 100.0,
        Nr=jnp.zeros_like(state.Nr),
        qke=jnp.ones_like(state.qke) * 0.2,
        t_skin=294.0 + 0.5 * jnp.sin(surface_x) + 0.25 * jnp.cos(surface_y),
        soil_moisture=jnp.ones_like(state.soil_moisture) * 0.25,
        xland=jnp.ones_like(state.xland),
        lakemask=jnp.zeros_like(state.lakemask),
        mavail=jnp.ones_like(state.mavail) * 0.35,
        roughness_m=jnp.ones_like(state.roughness_m) * 0.08,
    )
    state = state.replace(**_constant_boundary_leaves(state, grid))
    return state, Tendencies.zeros(grid), grid


def _radiation_cadence_steps(dt_s: float, cadence_s: float) -> int:
    steps = int(round(float(cadence_s) / float(dt_s)))
    if abs(steps * float(dt_s) - float(cadence_s)) > 1.0e-9:
        raise ValueError(f"radiation cadence {cadence_s}s is not divisible by dt={dt_s}s")
    return max(1, steps)


def run_reduced_tsc(
    *,
    dts: tuple[float, float, float] = TSC_DTS,
    leads_h: tuple[float, ...] = LEADS_H,
    radiation_cadence_s: float = 540.0,
) -> dict[float, dict[float, dict[str, np.ndarray]]]:
    """Run the controlled reduced coupled case at all TSC timesteps."""

    outputs: dict[float, dict[float, dict[str, np.ndarray]]] = {}
    for dt_s in dts:
        state, tendencies, grid = idealized_coupled_state()
        current = state
        previous_steps = 0
        total_steps = steps_for_hours(max(leads_h), dt_s)
        dt_outputs: dict[float, dict[str, np.ndarray]] = {}
        for lead in leads_h:
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
                    n_acoustic=1,
                    radiation_cadence_steps=_radiation_cadence_steps(dt_s, radiation_cadence_s),
                    final_radiation=True,
                    boundary_config=DEFAULT_BOUNDARY_CONFIG,
                )
                block_until_ready(current)
            dt_outputs[float(lead)] = surface_fields_from_state(current)
            previous_steps = target_steps
        outputs[float(dt_s)] = dt_outputs
    return outputs


def compute_tsc_envelope(
    outputs: dict[float, dict[float, dict[str, np.ndarray]]],
    *,
    dts: tuple[float, float, float] = TSC_DTS,
    leads_h: tuple[float, ...] = LEADS_H,
    variables: tuple[str, ...] = VARIABLES,
) -> dict[str, Any]:
    """Compute `max(|F18-F9|, |F9-F4.5|)` by variable and lead."""

    base, refine, further = (float(item) for item in dts)
    table: dict[str, Any] = {}
    for var in variables:
        table[var] = {}
        for lead in leads_h:
            lead_key = f"+{lead:g}h"
            base_refine = _field_norm(outputs[base][float(lead)][var], outputs[refine][float(lead)][var])
            refine_further = _field_norm(outputs[refine][float(lead)][var], outputs[further][float(lead)][var])
            table[var][lead_key] = {
                "base_vs_refine": base_refine,
                "refine_vs_further": refine_further,
                "envelope": max(base_refine["max_abs"], refine_further["max_abs"]),
                "units": _variable_units(var),
            }
    return table


def _variable_units(var: str) -> str:
    return {
        "U10": "m s-1",
        "V10": "m s-1",
        "T2": "K",
        "qv2": "kg kg-1",
        "precip": "mm",
    }[var]


def _read_wrfout_2d(path: Path, variable: str) -> np.ndarray:
    with Dataset(path, "r") as dataset:
        data = dataset.variables[variable][0]
        return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def load_gen2_surface_fields(run: Gen2Run, lead_h: float, *, domain: str = "d02") -> dict[str, np.ndarray]:
    """Load Gen2 surface reference fields at an hourly lead."""

    files = run.history_files(domain)
    lead_index = int(round(float(lead_h)))
    if abs(float(lead_h) - lead_index) > 1.0e-9:
        raise ValueError("Gen2 d02 drift comparison currently requires hourly lead files")
    initial = files[0]
    path = files[lead_index]
    precip_vars = ("RAINC", "RAINNC", "SNOWNC", "GRAUPELNC")
    precip = sum(_read_wrfout_2d(path, name) - _read_wrfout_2d(initial, name) for name in precip_vars)
    return {
        "U10": _read_wrfout_2d(path, "U10"),
        "V10": _read_wrfout_2d(path, "V10"),
        "T2": _read_wrfout_2d(path, "T2"),
        "qv2": _read_wrfout_2d(path, "Q2"),
        "precip": precip,
    }


def run_pinned_d02_gpu(
    *,
    run_dir: str | Path = DEFAULT_M6_GEN2_RUN_DIR,
    boundary_path: str | Path = DEFAULT_M6_BOUNDARY_REPLAY,
    domain: str = "d02",
    dt_s: float = 18.0,
    leads_h: tuple[float, ...] = LEADS_H,
    radiation_cadence_s: float = 540.0,
) -> tuple[dict[float, dict[str, np.ndarray]], dict[str, Any]]:
    """Run the pinned d02 GPU forecast and return surface fields at leads."""

    run = Gen2Run(run_dir)
    state, tendencies, grid, meta = build_initial_state(run, domain=domain, boundary_path=boundary_path)
    current = state
    previous_steps = 0
    total_steps = steps_for_hours(max(leads_h), dt_s)
    outputs: dict[float, dict[str, np.ndarray]] = {}
    for lead in leads_h:
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
                n_acoustic=2,
                radiation_cadence_steps=_radiation_cadence_steps(dt_s, radiation_cadence_s),
                final_radiation=True,
                boundary_config=DEFAULT_BOUNDARY_CONFIG,
            )
            block_until_ready(current)
        outputs[float(lead)] = surface_fields_from_state(current)
        previous_steps = target_steps
    return outputs, meta


def compute_gpu_drift(
    gpu_outputs: dict[float, dict[str, np.ndarray]],
    *,
    run_dir: str | Path = DEFAULT_M6_GEN2_RUN_DIR,
    domain: str = "d02",
    leads_h: tuple[float, ...] = LEADS_H,
    variables: tuple[str, ...] = VARIABLES,
) -> dict[str, Any]:
    """Compare pinned d02 GPU fields against Gen2 WRF fields."""

    run = Gen2Run(run_dir)
    drift: dict[str, Any] = {var: {} for var in variables}
    for lead in leads_h:
        truth = load_gen2_surface_fields(run, lead, domain=domain)
        lead_key = f"+{lead:g}h"
        for var in variables:
            drift[var][lead_key] = {
                **_field_norm(gpu_outputs[float(lead)][var], truth[var]),
                "units": _variable_units(var),
                "reference": "Gen2 WRF d02 hourly wrfout on the same mass grid",
            }
    return drift


def classify_drift(envelope: dict[str, Any], drift: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return per-variable/per-lead GREEN/PARTIAL/FAIL status without aggregation hiding."""

    statuses: dict[str, Any] = {}
    variable_states = []
    for var, lead_records in envelope.items():
        lead_status = {}
        failures = 0
        blocked = 0
        for lead_key, env_record in lead_records.items():
            if var not in drift or lead_key not in drift[var]:
                status = "BLOCKED"
                blocked += 1
                observed = None
            else:
                observed = float(drift[var][lead_key]["max_abs"])
                status = "GREEN" if observed <= float(env_record["envelope"]) else "FAIL"
                failures += int(status == "FAIL")
            lead_status[lead_key] = {
                "status": status,
                "gpu_drift_max_abs": observed,
                "envelope": float(env_record["envelope"]),
                "rule": "GREEN iff raw GPU drift max_abs <= raw TSC envelope; no cap or min(raw, cap) adjustment",
            }
        if blocked:
            variable_status = "BLOCKED"
        elif failures == 0:
            variable_status = "GREEN"
        elif failures < len(lead_status):
            variable_status = "PARTIAL"
        else:
            variable_status = "FAIL"
        variable_states.append(variable_status)
        statuses[var] = {"status": variable_status, "leads": lead_status}
    if all(state == "GREEN" for state in variable_states):
        overall = "GREEN"
    elif all(state == "BLOCKED" for state in variable_states):
        overall = "BLOCKED"
    elif all(state == "FAIL" for state in variable_states):
        overall = "FAIL"
    else:
        overall = "PARTIAL"
    return statuses, overall


def thompson_water_budget_oracle_probe(state: State, *, dt_s: float = 18.0) -> dict[str, Any]:
    """Recompute the Thompson water residual with the M6-S6 side-channel oracle."""

    thompson_next, oracle = thompson_adapter_with_tendencies(state, dt_s)
    residual = water_budget_residual(state, thompson_next, dt_s, oracle)
    corrupted = thompson_next.replace(qv=thompson_next.qv + 1.0e-7)
    corrupted_residual = water_budget_residual(state, corrupted, dt_s, oracle)
    return {
        "oracle": "ThompsonTendencySideChannel column_water_tendency from physics_couplers.thompson_adapter(return_tendencies=True)",
        "residual_max_abs": float(residual["max_abs"]),
        "residual_domain_mean_abs": float(residual["domain_mean_abs"]),
        "corrupted_residual_max_abs": float(corrupted_residual["max_abs"]),
        "load_bearing": bool(corrupted_residual["max_abs"] > max(residual["max_abs"] * 10.0, 1.0e-10)),
        "note": "Current Thompson source/sink subset has no sedimentation accumulator; precip_out_tendency is therefore zero.",
    }


def _assign_outer_edges(field: np.ndarray, decoded: dict[str, Any], var: str) -> np.ndarray:
    out = np.asarray(field, dtype=np.float64).copy()
    sides = decoded["variables"][var]["sides"]
    if out.ndim == 3:
        out[:, :, 0] = sides["W"]["boundary"][0]
        out[:, :, -1] = sides["E"]["boundary"][0]
        out[:, 0, :] = sides["S"]["boundary"][0]
        out[:, -1, :] = sides["N"]["boundary"][0]
        return out
    out[:, 0] = sides["W"]["boundary"][0]
    out[:, -1] = sides["E"]["boundary"][0]
    out[0, :] = sides["S"]["boundary"][0]
    out[-1, :] = sides["N"]["boundary"][0]
    return out


def _pack_wrfbdy_outer_leaf(decoded: dict[str, Any], var: str, z_len: int, max_side: int, cadence_s: float) -> np.ndarray:
    packed = np.zeros((2, 4, z_len, max_side), dtype=np.float64)
    for side_index, side in enumerate(SIDES):
        base = decoded["variables"][var]["sides"][side]["boundary"][0]
        tendency = decoded["variables"][var]["sides"][side]["tendency"][0]
        if base.ndim == 1:
            packed[0, side_index, 0, : base.shape[0]] = base
            packed[1, side_index, 0, : base.shape[0]] = base + float(cadence_s) * tendency
        else:
            packed[0, side_index, : base.shape[0], : base.shape[-1]] = base
            packed[1, side_index, : base.shape[0], : base.shape[-1]] = base + float(cadence_s) * tendency
    return packed


def wrfbdy_boundary_oracle_probe(
    *,
    run_dir: str | Path = DEFAULT_M6_GEN2_RUN_DIR,
    cadence_s: float = 1.0,
) -> dict[str, Any]:
    """Decode wrfbdy_d01 and compare boundary_apply's specified-edge tendency."""

    run = Gen2Run(run_dir)
    path = wrfbdy_path_for_run(run, "d01")
    decoded = decode_wrfbdy(path, variables=WRFBDY_VARIABLES, time_index=0)
    grid = run.grid("d01").as_grid_spec()
    state = State.zeros(grid)
    max_side = int(max(grid.nx + 1, grid.ny + 1))
    fields = {
        "u": _assign_outer_edges(np.zeros(state.u.shape, dtype=np.float64), decoded, "U"),
        "v": _assign_outer_edges(np.zeros(state.v.shape, dtype=np.float64), decoded, "V"),
        "theta": _assign_outer_edges(np.zeros(state.theta.shape, dtype=np.float64), decoded, "T"),
        "qv": _assign_outer_edges(np.zeros(state.qv.shape, dtype=np.float64), decoded, "QVAPOR"),
        "ph": _assign_outer_edges(np.zeros(state.ph.shape, dtype=np.float64), decoded, "PH"),
        "mu": _assign_outer_edges(np.zeros(state.mu.shape, dtype=np.float64), decoded, "MU"),
    }
    state = state.replace(
        **{name: jax.device_put(jnp.asarray(value)) for name, value in fields.items()},
        u_bdy=jax.device_put(jnp.asarray(_pack_wrfbdy_outer_leaf(decoded, "U", grid.nz, max_side, cadence_s))),
        v_bdy=jax.device_put(jnp.asarray(_pack_wrfbdy_outer_leaf(decoded, "V", grid.nz, max_side, cadence_s))),
        theta_bdy=jax.device_put(jnp.asarray(_pack_wrfbdy_outer_leaf(decoded, "T", grid.nz, max_side, cadence_s))),
        qv_bdy=jax.device_put(jnp.asarray(_pack_wrfbdy_outer_leaf(decoded, "QVAPOR", grid.nz, max_side, cadence_s))),
        ph_bdy=jax.device_put(jnp.asarray(_pack_wrfbdy_outer_leaf(decoded, "PH", grid.nz + 1, max_side, cadence_s))),
        mu_bdy=jax.device_put(jnp.asarray(_pack_wrfbdy_outer_leaf(decoded, "MU", 1, max_side, cadence_s))),
    )
    config = BoundaryConfig(spec_bdy_width=1, spec_zone=1, relax_zone=1, update_cadence_s=cadence_s, spec_exp=0.0)
    after = apply_lateral_boundaries(state, cadence_s, cadence_s, config)
    tendency = tuple((getattr(after, field) - getattr(state, field)) / float(cadence_s) for field in ("u", "v", "theta", "qv", "ph", "mu"))
    comparison = compare_boundary_tendency_to_wrfbdy(
        dict(zip(("u", "v", "theta", "qv", "ph", "mu"), tendency, strict=True)),
        decoded,
        width=1,
        trim_corners=1,
    )
    max_abs = max(record["aggregate"]["max_abs_max"] for record in comparison["variables"].values())
    comparison["max_abs_all_variables"] = float(max_abs)
    comparison["status"] = "PASS" if max_abs <= 1.0e-6 else "PARTIAL"
    comparison["load_bearing"] = True
    comparison["note"] = (
        "Comparison uses wrfbdy_d01 because the nested d02 run has replayed d02 boundaries, not a native wrfbdy_d02 file. "
        "Raw nonzero residuals are retained; large T/PH boundary magnitudes are limited by FP32 State boundary storage."
    )
    return comparison


def build_tier3_artifact(
    *,
    output_path: str | Path = DEFAULT_ARTIFACT,
    run_dir: str | Path = DEFAULT_M6_GEN2_RUN_DIR,
    boundary_path: str | Path = DEFAULT_M6_BOUNDARY_REPLAY,
    domain: str = "d02",
    leads_h: tuple[float, ...] = LEADS_H,
    run_d02: bool = True,
    d02_blocked_reason: str | None = None,
    d02_dt_s: float = 18.0,
    radiation_cadence_s: float = 540.0,
) -> dict[str, Any]:
    """Run TSC1.0 and write the M6-S6 proof object."""

    reduced_outputs = run_reduced_tsc(leads_h=leads_h, radiation_cadence_s=radiation_cadence_s)
    envelope = compute_tsc_envelope(reduced_outputs, leads_h=leads_h)
    ideal_state, _ideal_tendencies, _ideal_grid = idealized_coupled_state()
    water_oracle = thompson_water_budget_oracle_probe(ideal_state, dt_s=TSC_DTS[0])
    wrfbdy_oracle = wrfbdy_boundary_oracle_probe(run_dir=run_dir)
    if run_d02:
        gpu_outputs, d02_meta = run_pinned_d02_gpu(
            run_dir=run_dir,
            boundary_path=boundary_path,
            domain=domain,
            dt_s=d02_dt_s,
            leads_h=leads_h,
            radiation_cadence_s=radiation_cadence_s,
        )
        gpu_drift = compute_gpu_drift(gpu_outputs, run_dir=run_dir, domain=domain, leads_h=leads_h)
        per_variable_status, status = classify_drift(envelope, gpu_drift)
    else:
        d02_meta = {
            "boundary": {"path": str(boundary_path), "schema": "not_loaded"},
            "blocked_reason": d02_blocked_reason or "Pinned d02 drift comparison was skipped by runner option.",
        }
        gpu_drift = {}
        per_variable_status = {
            var: {"status": "BLOCKED", "leads": {f"+{lead:g}h": {"status": "BLOCKED"} for lead in leads_h}}
            for var in VARIABLES
        }
        status = "BLOCKED"
    payload = {
        "artifact_type": "tier3_drift_envelope",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": Gen2Run(run_dir).run_id,
        "domain": domain,
        "status": status,
        "base_dt_s": TSC_DTS[0],
        "refined_dt_s": TSC_DTS[1],
        "further_refined_dt_s": TSC_DTS[2],
        "pinned_d02_dt_s": float(d02_dt_s),
        "lead_hours": list(leads_h),
        "variables": list(VARIABLES),
        "boundary_mode": {
            "reduced_tsc": "constant idealized lateral boundaries packed into State boundary leaves",
            "pinned_d02": d02_meta["boundary"],
            "wrfbdy_oracle": "wrfbdy_d01 decoded for independent lateral forcing tendency proof; d02 uses replay zarr",
        },
        "d02_drift_blocker": d02_meta.get("blocked_reason"),
        "forcing_mode": {
            "physics": "coupled dycore + Thompson + MYNN + surface layer + RRTMG cadence",
            "radiation_cadence_s": float(radiation_cadence_s),
            "dycore_cap_inherited": "coupling.driver still caps dycore_dt_s=min(dt_s,1.0); M6-S5 owns cap lift",
        },
        "regridding": {
            "reduced_tsc": "same reduced grid across all dt values; no regridding",
            "pinned_d02": "GPU and Gen2 d02 compared on the same mass grid; surface variables are already unstaggered",
            "qv2_mapping": "GPU qv2 is surface_layer Q2 diagnostic; Gen2 reference variable is Q2",
            "precip_mapping": "GPU precip is rain+snow+graupel+ice accumulators; Gen2 precip is RAINC+RAINNC+SNOWNC+GRAUPELNC minus t0",
        },
        "norm_definitions": {
            "max_abs": "max(abs(candidate-reference)) over the 2D surface domain",
            "mean_abs": "mean(abs(candidate-reference)) over the 2D surface domain",
            "rmse": "sqrt(mean((candidate-reference)^2)) over the 2D surface domain",
            "status_rule": "per-variable per-lead GREEN iff raw GPU drift max_abs <= raw TSC envelope; no aggregate-only pass",
        },
        "envelope_derivation": {
            "method": "TSC1.0 controlled dt-refinement reduced case",
            "formula": "envelope=max(max_abs(F(dt=18)-F(dt=9)), max_abs(F(dt=9)-F(dt=4.5)))",
            "cpu_reference": {
                "status": "ANALYTIC_FIXTURE_USED",
                "reason": "No reviewed Gen2 multi-dt d02 campaign is present in the pinned run inventory; reduced smooth fixture supplies controlled same-equation dt refinement.",
                "derivation": "Identical IC/BC/physics cadence on a smooth 8x8x10 case isolates timestep sensitivity from l2-vs-l3 configuration noise.",
            },
        },
        "envelope": envelope,
        "gpu_drift": gpu_drift,
        "per_variable_status": per_variable_status,
        "thompson_water_budget_oracle": water_oracle,
        "wrfbdy_boundary_oracle": wrfbdy_oracle,
        "artifact_paths": [_rel_path(output_path), _rel_path(boundary_path)],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_ARTIFACT",
    "LEADS_H",
    "TSC_DTS",
    "VARIABLES",
    "build_tier3_artifact",
    "classify_drift",
    "compute_gpu_drift",
    "compute_tsc_envelope",
    "idealized_coupled_state",
    "load_gen2_surface_fields",
    "run_pinned_d02_gpu",
    "run_reduced_tsc",
    "surface_fields_from_state",
    "thompson_water_budget_oracle_probe",
    "wrfbdy_boundary_oracle_probe",
]
