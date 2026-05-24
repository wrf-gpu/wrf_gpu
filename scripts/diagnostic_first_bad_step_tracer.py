#!/usr/bin/env python3
"""Sanitizer-bypass d02 replay tracer for the M6.x S3 operator bug hunt."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from functools import partial
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import BaseState, State, Tendencies
from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics import acoustic_wrf
from gpuwrf.dynamics.acoustic_wrf import AcousticConfig, AcousticScanCarry
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.dynamics.damping import RayleighConfig
from gpuwrf.integration import d02_replay
from gpuwrf.integration.d02_replay import ReplayCase, ReplayConfig, build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready


config.update("jax_enable_x64", True)

SANITIZER_LIMITS: dict[str, tuple[float, float]] = {
    "u": (-150.0, 150.0),
    "v": (-150.0, 150.0),
    "w": (-50.0, 50.0),
    "theta": (150.0, 550.0),
    "qv": (0.0, 0.05),
    "p": (1000.0, 120000.0),
    "mu": (1000.0, 120000.0),
    "qc": (0.0, 0.05),
    "qr": (0.0, 0.05),
    "qi": (0.0, 0.05),
    "qs": (0.0, 0.05),
    "qg": (0.0, 0.05),
    "Ni": (0.0, 1.0e10),
    "Nr": (0.0, 1.0e10),
    "Ns": (0.0, 1.0e10),
    "Ng": (0.0, 1.0e10),
    "qke": (0.0, 100.0),
    "ustar": (0.0, 10.0),
    "theta_flux": (-5.0, 5.0),
    "qv_flux": (-1.0e-2, 1.0e-2),
    "tau_u": (-10.0, 10.0),
    "tau_v": (-10.0, 10.0),
    "rhosfc": (0.1, 2.0),
    "fltv": (-5.0, 5.0),
    "t_skin": (180.0, 340.0),
    "soil_moisture": (0.0, 1.0),
    "rain_acc": (0.0, 1.0e6),
    "snow_acc": (0.0, 1.0e6),
    "graupel_acc": (0.0, 1.0e6),
    "ice_acc": (0.0, 1.0e6),
}

INSPECT_FIELDS = tuple(name for name in State.__slots__ if not name.endswith("_bdy"))

SOURCE_CITATIONS = {
    "wrf_calc_coef_w": "module_small_step_em.F:619-651",
    "wrf_mu": "module_small_step_em.F:1094-1119",
    "wrf_w": "module_small_step_em.F:1340-1597",
    "mpas_coefficients": "mpas_atm_time_integration.F:1589-1656",
    "mpas_recurrence": "mpas_atm_time_integration.F:2146-2208",
    "mpas_w_metric": "mpas_atm_time_integration.F:2491-2495",
    "current_recurrence": "src/gpuwrf/dynamics/acoustic_wrf.py:763-827",
    "current_mu": "src/gpuwrf/dynamics/acoustic_wrf.py:473-495,926-930",
    "current_metric": "src/gpuwrf/dynamics/acoustic_wrf.py:512-534",
    "current_staging": "src/gpuwrf/integration/d02_replay.py:489-520",
    "current_sanitizer": "src/gpuwrf/integration/d02_replay.py:461-468,727-765",
}


@dataclass(frozen=True)
class DiagnosticToggle:
    """One-suspect diagnostic switch for a sanitizer-off replay."""

    name: str = "baseline"
    description: str = "Current operator, sanitizer bypassed."
    n_acoustic: int | None = None
    mu_mode: str = "bounded"  # bounded, zero, raw
    metric_mode: str = "current"  # current, reference_column
    recurrence_mode: str = "current"  # current, cofwr_sign_flip
    physics_enabled: bool = True
    boundary_enabled: bool = True
    pressure_scale_override: float | None = None
    final_radiation: bool | None = None


def now_label() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def replay_config_for_steps(
    steps: int,
    *,
    dt_s: float = 1.0,
    n_acoustic: int = 4,
    template: ReplayConfig | None = None,
    final_radiation: bool | None = None,
) -> ReplayConfig:
    base = template or ReplayConfig()
    return ReplayConfig(
        dt_s=float(dt_s),
        duration_s=float(steps) * float(dt_s),
        n_acoustic=int(n_acoustic),
        radiation_cadence_steps=int(base.radiation_cadence_steps),
        final_radiation=bool(base.final_radiation if final_radiation is None else final_radiation),
        boundary_config=base.boundary_config,
        rayleigh_coefficient=float(base.rayleigh_coefficient),
    )


def load_default_case(run_dir: str | Path | None = None, *, domain: str = "d02") -> ReplayCase:
    case = build_replay_case(run_dir or d02_replay.DEFAULT_REPLAY_RUN_DIR, domain=domain)
    block_until_ready((case.state, case.previous_pressure, case.tendencies, case.metrics, case.base_state))
    return case


def _json_scalar(value: Any) -> float | int | str | None:
    if value is None:
        return None
    scalar = np.asarray(value).item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, (float, np.floating)):
        return float(scalar) if np.isfinite(float(scalar)) else str(float(scalar))
    return str(scalar)


def _value_at(array: Any, index: tuple[int, ...]) -> float | int | str | None:
    if array is None:
        return None
    arr = np.asarray(jax.device_get(array))
    if arr.shape == ():
        return _json_scalar(arr)
    if len(index) != arr.ndim:
        return None
    return _json_scalar(arr[index])


def _location(index: np.ndarray) -> dict[str, Any]:
    raw = [int(item) for item in index.tolist()]
    payload: dict[str, Any] = {"raw_indices": raw}
    if len(raw) >= 3:
        payload["i_j_k"] = [raw[-1], raw[-2], raw[0]]
    elif len(raw) == 2:
        payload["i_j_k"] = [raw[-1], raw[-2], None]
    elif len(raw) == 1:
        payload["i_j_k"] = [raw[0], None, None]
    else:
        payload["i_j_k"] = []
    return payload


def _issue_from_array(
    *,
    step: int,
    stage: str,
    field: str,
    value: Any,
    previous_value: Any | None = None,
    limits: tuple[float, float] | None = None,
    source: str = "state",
) -> dict[str, Any] | None:
    nonfinite_count = int(np.asarray(jax.device_get(jnp.sum(~jnp.isfinite(value)))))
    if nonfinite_count:
        arr = np.asarray(jax.device_get(value))
        nonfinite = ~np.isfinite(arr)
        index = np.argwhere(nonfinite)[0]
        idx = tuple(int(item) for item in index.tolist())
        return {
            "step": int(step),
            "stage": stage,
            "source": source,
            "field": field,
            "kind": "nonfinite",
            "location": _location(index),
            "value": _value_at(value, idx),
            "previous_value": _value_at(previous_value, idx),
            "limits": None,
        }
    if limits is None:
        return None
    lower, upper = limits
    cap_count = int(
        np.asarray(
            jax.device_get(
                jnp.sum(jnp.isfinite(value) & ((value <= float(lower)) | (value >= float(upper))))
            )
        )
    )
    if not cap_count:
        return None
    arr = np.asarray(jax.device_get(value))
    finite = np.isfinite(arr)
    capped = finite & ((arr <= lower) | (arr >= upper))
    index = np.argwhere(capped)[0]
    idx = tuple(int(item) for item in index.tolist())
    return {
        "step": int(step),
        "stage": stage,
        "source": source,
        "field": field,
        "kind": "guard_limit",
        "location": _location(index),
        "value": _value_at(value, idx),
        "previous_value": _value_at(previous_value, idx),
        "limits": [float(lower), float(upper)],
    }


def first_state_issue(state: State, previous: State | None, *, step: int, stage: str) -> dict[str, Any] | None:
    for field in INSPECT_FIELDS:
        value = getattr(state, field)
        previous_value = getattr(previous, field) if previous is not None and hasattr(previous, field) else None
        issue = _issue_from_array(
            step=step,
            stage=stage,
            field=field,
            value=value,
            previous_value=previous_value,
            limits=SANITIZER_LIMITS.get(field),
        )
        if issue is not None:
            return issue
    return None


def _issue_from_array_host(
    *,
    step: int,
    stage: str,
    field: str,
    value: Any,
    previous_value: Any | None = None,
    limits: tuple[float, float] | None = None,
    source: str = "state",
) -> dict[str, Any] | None:
    arr = np.asarray(jax.device_get(value))
    nonfinite = ~np.isfinite(arr)
    if np.any(nonfinite):
        index = np.argwhere(nonfinite)[0]
        idx = tuple(int(item) for item in index.tolist())
        return {
            "step": int(step),
            "stage": stage,
            "source": source,
            "field": field,
            "kind": "nonfinite",
            "location": _location(index),
            "value": _json_scalar(arr[idx]),
            "previous_value": _value_at(previous_value, idx),
            "limits": None,
        }
    if limits is None:
        return None
    lower, upper = limits
    finite = np.isfinite(arr)
    capped = finite & ((arr <= lower) | (arr >= upper))
    if not np.any(capped):
        return None
    index = np.argwhere(capped)[0]
    idx = tuple(int(item) for item in index.tolist())
    return {
        "step": int(step),
        "stage": stage,
        "source": source,
        "field": field,
        "kind": "guard_limit",
        "location": _location(index),
        "value": _json_scalar(arr[idx]),
        "previous_value": _value_at(previous_value, idx),
        "limits": [float(lower), float(upper)],
    }


def first_state_issue_host(state: State, previous: State | None, *, step: int, stage: str) -> dict[str, Any] | None:
    for field in INSPECT_FIELDS:
        value = getattr(state, field)
        previous_value = getattr(previous, field) if previous is not None and hasattr(previous, field) else None
        issue = _issue_from_array_host(
            step=step,
            stage=stage,
            field=field,
            value=value,
            previous_value=previous_value,
            limits=SANITIZER_LIMITS.get(field),
        )
        if issue is not None:
            return issue
    return None


def first_term_issue(terms: dict[str, Any], *, step: int, stage: str) -> dict[str, Any] | None:
    for name, value in terms.items():
        issue = _issue_from_array(step=step, stage=stage, field=name, value=value, source="term")
        if issue is not None:
            return issue
    return None


def cap_fields(state: State) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for field, limits in SANITIZER_LIMITS.items():
        lower, upper = limits
        value = getattr(state, field)
        count = int(
            np.asarray(
                jax.device_get(
                    jnp.sum(jnp.isfinite(value) & ((value <= float(lower)) | (value >= float(upper))))
                )
            )
        )
        if count:
            fields.append({"field": field, "count": count, "limits": [float(lower), float(upper)]})
    return fields


def cap_fields_host(state: State) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for field, limits in SANITIZER_LIMITS.items():
        arr = np.asarray(jax.device_get(getattr(state, field)))
        lower, upper = limits
        finite = np.isfinite(arr)
        count = int(np.count_nonzero(finite & ((arr <= lower) | (arr >= upper))))
        if count:
            fields.append({"field": field, "count": count, "limits": [float(lower), float(upper)]})
    return fields


def nonfinite_fields(state: State) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for field in INSPECT_FIELDS:
        value = getattr(state, field)
        count = int(np.asarray(jax.device_get(jnp.sum(~jnp.isfinite(value)))))
        if count:
            fields.append({"field": field, "count": count})
    return fields


def nonfinite_fields_host(state: State) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for field in INSPECT_FIELDS:
        arr = np.asarray(jax.device_get(getattr(state, field)))
        count = int(np.count_nonzero(~np.isfinite(arr)))
        if count:
            fields.append({"field": field, "count": count})
    return fields


def state_extrema(state: State) -> dict[str, float | str]:
    return {
        "w_abs_max_m_s": _json_scalar(jnp.max(jnp.abs(state.w))),
        "theta_min_k": _json_scalar(jnp.min(state.theta)),
        "theta_max_k": _json_scalar(jnp.max(state.theta)),
        "u_abs_max_m_s": _json_scalar(jnp.max(jnp.abs(state.u))),
        "v_abs_max_m_s": _json_scalar(jnp.max(jnp.abs(state.v))),
        "p_min_pa": _json_scalar(jnp.min(state.p)),
        "p_max_pa": _json_scalar(jnp.max(state.p)),
        "mu_min_pa": _json_scalar(jnp.min(state.mu)),
        "mu_max_pa": _json_scalar(jnp.max(state.mu)),
    }


def _acoustic_config(grid: GridSpec, replay_config: ReplayConfig, toggle: DiagnosticToggle) -> AcousticConfig:
    n_acoustic = int(toggle.n_acoustic or replay_config.n_acoustic)
    return AcousticConfig(
        n_substeps=n_acoustic,
        dx_m=float(grid.projection.dx_m),
        dy_m=float(grid.projection.dy_m),
        non_hydrostatic=True,
        top_lid=True,
        mu_continuity=toggle.mu_mode != "zero",
        epssm=0.1,
        rayleigh=RayleighConfig(
            enabled=float(replay_config.rayleigh_coefficient) != 0.0,
            coefficient=float(replay_config.rayleigh_coefficient),
        ),
    )


def _mpas_recurrence_terms(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float,
    buoyancy_scale: float,
    recurrence_mode: str = "current",
) -> dict[str, Any]:
    theta_base = acoustic_wrf._base_theta(base_state, state)
    theta_perturbation = state.theta - theta_base
    rho_pp = acoustic_wrf._density_perturbation_from_pressure(state, base_state)
    mpas_w_metric = acoustic_wrf._mpas_w_metric_faces(state, base_state, metrics)
    rw_p = state.w * mpas_w_metric
    dz = acoustic_wrf._layer_thickness_m(state, base_state, metrics)
    cofrz, cofwr, cofwz, coftz, cofwt, rdzw, a, b, c = acoustic_wrf.build_epssm_column_coefficients(
        state.theta,
        dz,
        dt=dt,
        epssm=epssm,
    )
    resm = (1.0 - float(epssm)) / (1.0 + float(epssm))
    rs = rho_pp - cofrz * resm * (rw_p[1:, :, :] - rw_p[:-1, :, :])
    ts = theta_perturbation - resm * rdzw * (
        coftz[1:, :, :] * rw_p[1:, :, :] - coftz[:-1, :, :] * rw_p[:-1, :, :]
    )
    buoyancy_face = acoustic_wrf._vertical_buoyancy_acceleration(state, base_state)
    cofwr_sign = 1.0 if recurrence_mode == "cofwr_sign_flip" else -1.0
    rhs_interior = (
        rw_p[1:-1, :, :]
        + float(dt) * float(buoyancy_scale) * buoyancy_face[1:-1, :, :]
        - cofwz[1:-1, :, :]
        * (
            (ts[1:, :, :] - ts[:-1, :, :])
            + resm * (theta_perturbation[1:, :, :] - theta_perturbation[:-1, :, :])
        )
        + cofwr_sign
        * cofwr[1:-1, :, :]
        * ((rs[1:, :, :] + rs[:-1, :, :]) + resm * (rho_pp[1:, :, :] + rho_pp[:-1, :, :]))
        + cofwt[1:, :, :] * (ts[1:, :, :] + resm * theta_perturbation[1:, :, :])
        + cofwt[:-1, :, :] * (ts[:-1, :, :] + resm * theta_perturbation[:-1, :, :])
    )
    return {
        "rho_pp": rho_pp,
        "mpas_w_metric": mpas_w_metric,
        "rw_p": rw_p,
        "dz": dz,
        "cofrz": cofrz,
        "cofwr": cofwr,
        "cofwz": cofwz,
        "coftz": coftz,
        "cofwt": cofwt,
        "rdzw": rdzw,
        "tri_a": a,
        "tri_b": b,
        "tri_c": c,
        "rs": rs,
        "ts": ts,
        "buoyancy_face": buoyancy_face,
        "rhs_interior": rhs_interior,
    }


def _record_issue(
    issues: list[dict[str, Any]],
    issue: dict[str, Any] | None,
    *,
    substep: int | None = None,
    acoustic_pass: str | None = None,
) -> None:
    if issue is None:
        return
    if substep is not None:
        issue["acoustic_substep"] = int(substep)
    if acoustic_pass is not None:
        issue["acoustic_pass"] = acoustic_pass
    issues.append(issue)


def _acoustic_substep_carry_staged(
    carry: AcousticScanCarry,
    metrics: DycoreMetrics,
    acoustic_config: AcousticConfig,
    dt: float,
    base_state: BaseState | None,
    toggle: DiagnosticToggle,
    *,
    step: int,
    substep: int,
    acoustic_pass: str,
) -> tuple[AcousticScanCarry, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    pressure_diag, al, alt = acoustic_wrf.diagnose_pressure_al_alt(carry.state, base_state, metrics)
    keep_resident_pressure = bool(acoustic_config.non_hydrostatic) and base_state is not None
    pressure_source = carry.state.p_perturbation if keep_resident_pressure else pressure_diag
    cqu, cqv = acoustic_wrf.moisture_coupling_factors(carry.state)
    pressure_next = acoustic_wrf.apply_smdiv_pressure(pressure_source, carry.previous_pressure, acoustic_config.smdiv)
    pressure_state = carry.state if keep_resident_pressure else acoustic_wrf._replace_pressure(carry.state, pressure_next, base_state)
    du_dt, dv_dt, _, _ = acoustic_wrf.horizontal_pressure_gradient(
        pressure_state,
        base_state,
        metrics,
        pressure_next,
        al,
        alt,
        cqu,
        cqv,
        dx_m=acoustic_config.dx_m,
        dy_m=acoustic_config.dy_m,
        non_hydrostatic=acoustic_config.non_hydrostatic,
        top_lid=acoustic_config.top_lid,
    )
    pre_vertical = pressure_state.replace(
        u=pressure_state.u + float(dt) * du_dt,
        v=pressure_state.v + float(dt) * dv_dt,
        w=acoustic_wrf.apply_rayleigh_w(pressure_state.w, acoustic_config.rayleigh),
    )
    _record_issue(
        issues,
        first_state_issue(pre_vertical, pressure_state, step=step, stage="pre-vertical-recurrence"),
        substep=substep,
        acoustic_pass=acoustic_pass,
    )

    pressure_scale = (
        float(toggle.pressure_scale_override)
        if toggle.pressure_scale_override is not None
        else (-1.0 if bool(acoustic_config.non_hydrostatic) else 1.0)
    )
    buoyancy_scale = (
        acoustic_wrf.SLICE_ONLY_MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE
        if bool(acoustic_config.non_hydrostatic) and str(metrics.provenance).startswith("analytic")
        else acoustic_wrf.SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE
    )
    if pressure_scale <= 0.0:
        terms = _mpas_recurrence_terms(
            pre_vertical,
            base_state,
            metrics,
            dt=dt,
            epssm=acoustic_config.epssm,
            buoyancy_scale=buoyancy_scale,
            recurrence_mode=toggle.recurrence_mode,
        )
        _record_issue(
            issues,
            first_term_issue(terms, step=step, stage="inside-recurrence"),
            substep=substep,
            acoustic_pass=acoustic_pass,
        )
    vertical_state = acoustic_wrf.vertical_acoustic_update(
        pre_vertical,
        base_state,
        metrics,
        dt=dt,
        epssm=acoustic_config.epssm,
        top_lid=acoustic_config.top_lid,
        pressure_scale=pressure_scale,
        buoyancy_scale=buoyancy_scale,
        convective_buoyancy_gain=0.0,
    )
    _record_issue(
        issues,
        first_state_issue(vertical_state, pre_vertical, step=step, stage="post-recurrence"),
        substep=substep,
        acoustic_pass=acoustic_pass,
    )

    next_state = vertical_state
    if bool(acoustic_config.mu_continuity):
        mu_source_state = pressure_state if bool(acoustic_config.non_hydrostatic) else next_state
        dmu_dt = acoustic_wrf.mu_continuity_tendency(
            mu_source_state,
            base_state,
            metrics,
            dx_m=acoustic_config.dx_m,
            dy_m=acoustic_config.dy_m,
        )
        if toggle.mu_mode == "raw":
            dmu = float(dt) * dmu_dt
        else:
            dmu = acoustic_wrf._mu_continuity_increment(next_state, base_state, dmu_dt, dt=dt)
        _record_issue(
            issues,
            first_term_issue({"dmu_dt": dmu_dt, "dmu": dmu}, step=step, stage="mu-update"),
            substep=substep,
            acoustic_pass=acoustic_pass,
        )
        next_state = acoustic_wrf._replace_mu(next_state, next_state.mu_perturbation + dmu, base_state)
        _record_issue(
            issues,
            first_state_issue(next_state, vertical_state, step=step, stage="mu-update"),
            substep=substep,
            acoustic_pass=acoustic_pass,
        )

    final_pressure, final_al, final_alt = acoustic_wrf.diagnose_pressure_al_alt(next_state, base_state, metrics)
    final_state = next_state if keep_resident_pressure else acoustic_wrf._replace_pressure(next_state, final_pressure, base_state)
    final_cqu, final_cqv = acoustic_wrf.moisture_coupling_factors(final_state)
    return AcousticScanCarry(final_state, pressure_source, final_al, final_alt, final_cqu, final_cqv), issues


def _run_acoustic_scan_staged(
    state: State,
    previous_pressure: Any,
    metrics: DycoreMetrics,
    acoustic_config: AcousticConfig,
    dt: float,
    base_state: BaseState | None,
    toggle: DiagnosticToggle,
    *,
    step: int,
    acoustic_pass: str,
) -> tuple[State, Any, list[dict[str, Any]]]:
    dt_sub = float(dt) / float(acoustic_config.n_substeps)
    carry = acoustic_wrf.initialize_acoustic_carry(state, previous_pressure, metrics, base_state, acoustic_config)
    issues: list[dict[str, Any]] = []
    for substep in range(1, int(acoustic_config.n_substeps) + 1):
        carry, sub_issues = _acoustic_substep_carry_staged(
            carry,
            metrics,
            acoustic_config,
            dt_sub,
            base_state,
            toggle,
            step=step,
            substep=substep,
            acoustic_pass=acoustic_pass,
        )
        issues.extend(sub_issues)
    return carry.state, carry.previous_pressure, issues


def _dycore_step_staged(
    state: State,
    previous_pressure: Any,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    toggle: DiagnosticToggle,
    *,
    step: int,
) -> tuple[State, Any, list[dict[str, Any]]]:
    acoustic_config = _acoustic_config(grid, replay_config, toggle)
    issues: list[dict[str, Any]] = []
    s0 = apply_halo(state, halo_spec(grid))
    s1 = d02_replay._rk3_stage(s0, s0, tendencies, grid, float(replay_config.dt_s) / 3.0)
    s1 = apply_halo(s1, halo_spec(grid))
    s2 = d02_replay._rk3_stage(s0, s1, tendencies, grid, float(replay_config.dt_s) / 2.0)
    s2, previous_pressure, pass_issues = _run_acoustic_scan_staged(
        s2,
        previous_pressure,
        metrics,
        acoustic_config,
        float(replay_config.dt_s) / 2.0,
        base_state,
        toggle,
        step=step,
        acoustic_pass="rk2-half-step",
    )
    issues.extend(pass_issues)
    s2 = apply_halo(s2, halo_spec(grid))
    s3 = d02_replay._rk3_stage(s0, s2, tendencies, grid, float(replay_config.dt_s))
    s3, previous_pressure, pass_issues = _run_acoustic_scan_staged(
        s3,
        previous_pressure,
        metrics,
        acoustic_config,
        float(replay_config.dt_s),
        base_state,
        toggle,
        step=step,
        acoustic_pass="rk3-full-step",
    )
    issues.extend(pass_issues)
    return apply_halo(s3, halo_spec(grid)), previous_pressure, issues


def _run_radiation_for_step(replay_config: ReplayConfig, *, step: int, total_steps: int, toggle: DiagnosticToggle) -> bool:
    final_radiation = replay_config.final_radiation if toggle.final_radiation is None else bool(toggle.final_radiation)
    cadence = int(replay_config.radiation_cadence_steps)
    cadence_hit = cadence > 0 and step % cadence == 0
    return bool(cadence_hit or (final_radiation and step == total_steps))


def _candidate_timestep_staged(
    state: State,
    previous_pressure: Any,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    toggle: DiagnosticToggle,
    *,
    step: int,
    total_steps: int,
) -> tuple[State, Any, list[dict[str, Any]]]:
    candidate, next_previous_pressure, issues = _dycore_step_staged(
        state,
        previous_pressure,
        tendencies,
        grid,
        metrics,
        base_state,
        replay_config,
        toggle,
        step=step,
    )
    if bool(toggle.physics_enabled):
        before_physics = candidate
        candidate = thompson_adapter(candidate, float(replay_config.dt_s))
        candidate = mynn_adapter(candidate, float(replay_config.dt_s), grid)
        candidate = surface_adapter(candidate, float(replay_config.dt_s))
        if _run_radiation_for_step(replay_config, step=step, total_steps=total_steps, toggle=toggle):
            candidate = rrtmg_adapter(candidate, float(replay_config.dt_s), grid)
        _record_issue(issues, first_state_issue(candidate, before_physics, step=step, stage="physics"))
    if bool(toggle.boundary_enabled):
        before_boundary = candidate
        lead_seconds = jnp.asarray(step, dtype=jnp.float64) * float(replay_config.dt_s)
        candidate = apply_lateral_boundaries(candidate, lead_seconds, float(replay_config.dt_s), replay_config.boundary_config)
        _record_issue(issues, first_state_issue(candidate, before_boundary, step=step, stage="boundary-application"))
    return candidate, next_previous_pressure, issues


def _first_step_of_kind(per_step: list[dict[str, Any]], kind: str) -> int | None:
    for row in per_step:
        issue = row.get("first_issue")
        if issue and issue.get("kind") == kind:
            return int(row["step"])
    return None


def _candidate_timestep_fast(
    state: State,
    previous_pressure: Any,
    tendencies: Tendencies,
    grid: GridSpec,
    metrics: DycoreMetrics,
    base_state: BaseState,
    replay_config: ReplayConfig,
    toggle: DiagnosticToggle,
    *,
    step: int,
    total_steps: int,
) -> tuple[State, Any]:
    candidate, next_previous_pressure = d02_replay._dycore_step_adr023(
        state,
        previous_pressure,
        tendencies,
        grid,
        metrics,
        base_state,
        replay_config,
    )
    if bool(toggle.physics_enabled):
        candidate = thompson_adapter(candidate, float(replay_config.dt_s))
        candidate = mynn_adapter(candidate, float(replay_config.dt_s), grid)
        candidate = surface_adapter(candidate, float(replay_config.dt_s))
        if _run_radiation_for_step(replay_config, step=step, total_steps=total_steps, toggle=toggle):
            candidate = rrtmg_adapter(candidate, float(replay_config.dt_s), grid)
    if bool(toggle.boundary_enabled):
        lead_seconds = jnp.asarray(step, dtype=jnp.float64) * float(replay_config.dt_s)
        candidate = apply_lateral_boundaries(candidate, lead_seconds, float(replay_config.dt_s), replay_config.boundary_config)
    return candidate, next_previous_pressure


def _replay_to_step_start(
    case: ReplayCase,
    replay_config: ReplayConfig,
    toggle: DiagnosticToggle,
    *,
    step: int,
) -> tuple[State, Any]:
    state = case.state
    previous_pressure = case.previous_pressure
    if step <= 1:
        return state, previous_pressure
    with patched_operator(toggle):
        for index in range(1, int(step)):
            state, previous_pressure = _candidate_timestep_fast(
                state,
                previous_pressure,
                case.tendencies,
                case.grid,
                case.metrics,
                case.base_state,
                replay_config,
                toggle,
                step=index,
                total_steps=int(replay_config.duration_s / replay_config.dt_s),
            )
            block_until_ready((state, previous_pressure))
    return state, previous_pressure


def localize_first_bad_stage(
    case: ReplayCase,
    replay_config: ReplayConfig,
    toggle: DiagnosticToggle,
    *,
    step: int,
) -> dict[str, Any]:
    state, previous_pressure = _replay_to_step_start(case, replay_config, toggle, step=step)
    with patched_operator(toggle):
        dycore, next_previous_pressure = d02_replay._dycore_step_adr023(
            state,
            previous_pressure,
            case.tendencies,
            case.grid,
            case.metrics,
            case.base_state,
            replay_config,
        )
        block_until_ready((dycore, next_previous_pressure))
    dycore_issue = first_state_issue_host(dycore, state, step=step, stage="post-recurrence")
    if dycore_issue is not None:
        no_mu_toggle = dataclass_replace(toggle, name=f"{toggle.name}_stage_no_mu", mu_mode="zero")
        with patched_operator(no_mu_toggle):
            no_mu, _ = d02_replay._dycore_step_adr023(
                state,
                previous_pressure,
                case.tendencies,
                case.grid,
                case.metrics,
                case.base_state,
                replay_config,
            )
            block_until_ready(no_mu)
        no_mu_issue = first_state_issue_host(no_mu, state, step=step, stage="post-recurrence")
        if no_mu_issue is None and toggle.mu_mode != "zero":
            issue = dict(dycore_issue)
            issue["stage"] = "mu-update"
            return {
                "classified_stage": "mu-update",
                "issue": issue,
                "evidence": {
                    "dycore_with_mu_issue": dycore_issue,
                    "dycore_without_mu_issue": None,
                },
            }
        return {
            "classified_stage": "post-recurrence",
            "issue": dycore_issue,
            "evidence": {
                "dycore_with_mu_issue": dycore_issue,
                "dycore_without_mu_issue": no_mu_issue,
            },
        }

    candidate = dycore
    if bool(toggle.physics_enabled):
        before_physics = candidate
        candidate = thompson_adapter(candidate, float(replay_config.dt_s))
        candidate = mynn_adapter(candidate, float(replay_config.dt_s), case.grid)
        candidate = surface_adapter(candidate, float(replay_config.dt_s))
        if _run_radiation_for_step(replay_config, step=step, total_steps=int(replay_config.duration_s / replay_config.dt_s), toggle=toggle):
            candidate = rrtmg_adapter(candidate, float(replay_config.dt_s), case.grid)
        block_until_ready(candidate)
        physics_issue = first_state_issue_host(candidate, before_physics, step=step, stage="physics")
        if physics_issue is not None:
            return {"classified_stage": "physics", "issue": physics_issue, "evidence": {"physics_issue": physics_issue}}
    if bool(toggle.boundary_enabled):
        before_boundary = candidate
        lead_seconds = jnp.asarray(step, dtype=jnp.float64) * float(replay_config.dt_s)
        candidate = apply_lateral_boundaries(candidate, lead_seconds, float(replay_config.dt_s), replay_config.boundary_config)
        block_until_ready(candidate)
        boundary_issue = first_state_issue_host(candidate, before_boundary, step=step, stage="boundary-application")
        if boundary_issue is not None:
            return {
                "classified_stage": "boundary-application",
                "issue": boundary_issue,
                "evidence": {"boundary_issue": boundary_issue},
            }
    return {"classified_stage": "candidate", "issue": None, "evidence": {"note": "No stage issue reproduced"}}


def run_sanitizer_off_replay(
    case: ReplayCase,
    replay_config: ReplayConfig,
    *,
    steps: int,
    toggle: DiagnosticToggle = DiagnosticToggle(),
    abort_on_first_bad: bool = True,
    localize_stage: bool = False,
) -> dict[str, Any]:
    state = case.state
    previous_pressure = case.previous_pressure
    first_issue: dict[str, Any] | None = None
    first_nonfinite_issue: dict[str, Any] | None = None
    per_step: list[dict[str, Any]] = []
    start = time.perf_counter()
    with patched_operator(toggle):
        for step in range(1, int(steps) + 1):
            previous_state = state
            state, previous_pressure = _candidate_timestep_fast(
                state,
                previous_pressure,
                case.tendencies,
                case.grid,
                case.metrics,
                case.base_state,
                replay_config,
                toggle,
                step=step,
                total_steps=int(steps),
            )
            block_until_ready((state, previous_pressure))
            step_issue = first_state_issue_host(state, previous_state, step=step, stage="candidate")
            if first_issue is None and step_issue is not None:
                first_issue = step_issue
            if first_nonfinite_issue is None and step_issue is not None and step_issue.get("kind") == "nonfinite":
                first_nonfinite_issue = step_issue
            caps = cap_fields_host(state)
            nonfinite = nonfinite_fields_host(state)
            per_step.append(
                {
                    "step": int(step),
                    "first_issue": step_issue,
                    "stage_issue_count": 0,
                    "nonfinite_fields": nonfinite,
                    "fields_on_cap": caps,
                    "extrema": state_extrema(state),
                }
            )
            if step_issue is not None and step_issue.get("kind") == "nonfinite" and bool(abort_on_first_bad):
                break
    elapsed = time.perf_counter() - start
    first_nonfinite_step = None
    first_guard_step = None
    for row in per_step:
        issue = row.get("first_issue")
        if issue and issue.get("kind") == "nonfinite" and first_nonfinite_step is None:
            first_nonfinite_step = int(row["step"])
        if issue and issue.get("kind") == "guard_limit" and first_guard_step is None:
            first_guard_step = int(row["step"])
    terminal_caps = per_step[-1]["fields_on_cap"] if per_step else []
    terminal_nonfinite = per_step[-1]["nonfinite_fields"] if per_step else []
    stage_localization = None
    if localize_stage and first_nonfinite_step is not None:
        stage_localization = localize_first_bad_stage(case, replay_config, toggle, step=first_nonfinite_step)
        if stage_localization.get("issue") is not None:
            first_nonfinite_issue = stage_localization["issue"]
    return {
        "toggle": toggle.name,
        "description": toggle.description,
        "source_citations": SOURCE_CITATIONS,
        "steps_requested": int(steps),
        "steps_completed": len(per_step),
        "abort_on_first_bad": bool(abort_on_first_bad),
        "wall_time_s": float(elapsed),
        "first_issue": first_nonfinite_issue or first_issue,
        "first_guard_or_nonfinite_issue": first_issue,
        "stage_localization": stage_localization,
        "first_bad_step": int((first_nonfinite_issue or first_issue)["step"]) if (first_nonfinite_issue or first_issue) else None,
        "first_nonfinite_step": first_nonfinite_step,
        "first_guard_limit_step": first_guard_step,
        "terminal_nonfinite_fields": terminal_nonfinite,
        "terminal_fields_on_cap": terminal_caps,
        "ten_step_sanitize_off_acceptance": (
            len(per_step) == int(steps)
            and first_nonfinite_step is None
            and first_guard_step is None
            and not terminal_nonfinite
            and not terminal_caps
        ),
        "per_step": per_step,
    }


def coefficient_sanity(case: ReplayCase, *, dt: float = 0.25, epssm: float = 0.1) -> dict[str, Any]:
    j = int(case.grid.ny // 2)
    i = int(case.grid.nx // 2)
    state = case.state
    dz = acoustic_wrf._layer_thickness_m(state, case.base_state, case.metrics)
    coeffs = acoustic_wrf.build_epssm_column_coefficients(state.theta, dz, dt=dt, epssm=epssm)
    names = ["cofrz", "cofwr", "cofwz", "coftz", "cofwt", "rdzw", "tri_a", "tri_b", "tri_c"]
    metric = acoustic_wrf._mpas_w_metric_faces(state, case.base_state, case.metrics)
    payload: dict[str, Any] = {
        "created_utc": now_label(),
        "column": {"i": i, "j": j},
        "dt": float(dt),
        "epssm": float(epssm),
        "source_citations": {
            "wrf": SOURCE_CITATIONS["wrf_calc_coef_w"],
            "mpas": SOURCE_CITATIONS["mpas_coefficients"],
            "current": "src/gpuwrf/dynamics/vertical_implicit_solver.py:18-82",
        },
        "checks": {},
        "coefficients": {},
    }
    for name, value in zip(names, coeffs):
        column = np.asarray(jax.device_get(value[:, j, i]))
        payload["coefficients"][name] = {
            "min": _json_scalar(np.nanmin(column)),
            "max": _json_scalar(np.nanmax(column)),
            "mean": _json_scalar(np.nanmean(column)),
            "abs_max": _json_scalar(np.nanmax(np.abs(column))),
            "finite": bool(np.all(np.isfinite(column))),
            "values": [_json_scalar(item) for item in column.tolist()],
        }
    metric_column = np.asarray(jax.device_get(metric[:, j, i]))
    tri_b = np.asarray(jax.device_get(coeffs[7][:, j, i]))
    tri_a = np.asarray(jax.device_get(coeffs[6][:, j, i]))
    tri_c = np.asarray(jax.device_get(coeffs[8][:, j, i]))
    payload["metric_column"] = {
        "min": _json_scalar(np.nanmin(metric_column)),
        "max": _json_scalar(np.nanmax(metric_column)),
        "finite": bool(np.all(np.isfinite(metric_column))),
        "values": [_json_scalar(item) for item in metric_column.tolist()],
    }
    payload["checks"] = {
        "all_coefficients_finite": all(item["finite"] for item in payload["coefficients"].values()),
        "tri_b_positive": bool(np.all(tri_b > 0.0)),
        "tri_diagonal_not_zero": bool(np.all(np.abs(tri_b) > 1.0e-12)),
        "metric_finite_positive": bool(np.all(np.isfinite(metric_column)) and np.all(metric_column > 0.0)),
        "weak_diagonal_dominance": bool(np.all(np.abs(tri_b) >= (np.abs(tri_a) + np.abs(tri_c)))),
    }
    return payload


def _reference_metric_function(original):
    def reference_metric(state: State, base_state: BaseState | None, metrics: DycoreMetrics):
        current = original(state, base_state, metrics)
        j = current.shape[1] // 2
        i = current.shape[2] // 2
        ref = current[:, j : j + 1, i : i + 1]
        return jnp.broadcast_to(ref, current.shape)

    return reference_metric


@partial(jax.jit, static_argnames=("dt", "epssm", "top_lid", "buoyancy_scale"))
def _mpas_recurrence_vertical_update_cofwr_sign_flip(
    state: State,
    base_state: BaseState | None,
    metrics: DycoreMetrics,
    *,
    dt: float,
    epssm: float = 0.1,
    top_lid: bool = True,
    buoyancy_scale: float = acoustic_wrf.SOURCE_BACKED_COLUMN_BUOYANCY_TENDENCY_SCALE,
) -> State:
    theta_base = acoustic_wrf._base_theta(base_state, state)
    theta_perturbation = state.theta - theta_base
    rho_pp = acoustic_wrf._density_perturbation_from_pressure(state, base_state)
    mpas_w_metric = acoustic_wrf._mpas_w_metric_faces(state, base_state, metrics)
    rw_p = state.w * mpas_w_metric
    dz = acoustic_wrf._layer_thickness_m(state, base_state, metrics)
    cofrz, cofwr, cofwz, coftz, cofwt, rdzw, a, b, c = acoustic_wrf.build_epssm_column_coefficients(
        state.theta,
        dz,
        dt=dt,
        epssm=epssm,
    )
    resm = (1.0 - float(epssm)) / (1.0 + float(epssm))
    rs = rho_pp - cofrz * resm * (rw_p[1:, :, :] - rw_p[:-1, :, :])
    ts = theta_perturbation - resm * rdzw * (
        coftz[1:, :, :] * rw_p[1:, :, :] - coftz[:-1, :, :] * rw_p[:-1, :, :]
    )
    buoyancy_face = acoustic_wrf._vertical_buoyancy_acceleration(state, base_state)
    rhs = rw_p
    rhs_interior = (
        rw_p[1:-1, :, :]
        + float(dt) * float(buoyancy_scale) * buoyancy_face[1:-1, :, :]
        - cofwz[1:-1, :, :]
        * (
            (ts[1:, :, :] - ts[:-1, :, :])
            + resm * (theta_perturbation[1:, :, :] - theta_perturbation[:-1, :, :])
        )
        + cofwr[1:-1, :, :] * ((rs[1:, :, :] + rs[:-1, :, :]) + resm * (rho_pp[1:, :, :] + rho_pp[:-1, :, :]))
        + cofwt[1:, :, :] * (ts[1:, :, :] + resm * theta_perturbation[1:, :, :])
        + cofwt[:-1, :, :] * (ts[:-1, :, :] + resm * theta_perturbation[:-1, :, :])
    )
    rhs = rhs.at[1:-1, :, :].set(rhs_interior)
    rhs = rhs.at[0, :, :].set(0.0)
    if bool(top_lid):
        rhs = rhs.at[-1, :, :].set(0.0)
    rw_next = acoustic_wrf.solve_tridiagonal(a, b, c, rhs)
    rw_next = rw_next.at[0, :, :].set(0.0)
    if bool(top_lid):
        rw_next = rw_next.at[-1, :, :].set(0.0)
    w_next = rw_next / mpas_w_metric
    rho_next = rs - cofrz * (rw_next[1:, :, :] - rw_next[:-1, :, :])
    theta_perturbation_next = ts - rdzw * (
        coftz[1:, :, :] * rw_next[1:, :, :] - coftz[:-1, :, :] * rw_next[:-1, :, :]
    )
    theta_next = theta_base + theta_perturbation_next
    ph_perturbation = state.ph_perturbation + acoustic_wrf.GRAVITY_M_S2 * float(dt) * (
        0.5 * (1.0 - float(epssm)) * state.w + 0.5 * (1.0 + float(epssm)) * w_next
    )
    ph_base = acoustic_wrf._base_geopotential(base_state, state)
    if base_state is None:
        pressure_perturbation = state.p_perturbation
        pressure_total = state.p_total
    else:
        pressure_perturbation = acoustic_wrf._pressure_from_density_perturbation(state, base_state, rho_next)
        pressure_total = base_state.pb + pressure_perturbation
    return state.replace(
        w=w_next,
        theta=theta_next,
        ph_perturbation=ph_perturbation,
        ph_total=ph_base + ph_perturbation,
        p_perturbation=pressure_perturbation,
        p_total=pressure_total,
    )


@contextlib.contextmanager
def patched_operator(toggle: DiagnosticToggle) -> Iterator[None]:
    original_metric = acoustic_wrf._mpas_w_metric_faces
    original_recurrence = acoustic_wrf._mpas_recurrence_vertical_update
    original_mu_increment = acoustic_wrf._mu_continuity_increment
    original_acoustic_config = d02_replay._acoustic_config
    original_vertical_update = acoustic_wrf.vertical_acoustic_update
    changed = False
    if toggle.metric_mode == "reference_column":
        acoustic_wrf._mpas_w_metric_faces = _reference_metric_function(original_metric)
        changed = True
    if toggle.recurrence_mode == "cofwr_sign_flip":
        acoustic_wrf._mpas_recurrence_vertical_update = _mpas_recurrence_vertical_update_cofwr_sign_flip
        changed = True
    if toggle.mu_mode == "raw":
        def raw_mu_increment(state: State, base_state: BaseState | None, dmu_dt: Any, *, dt: float):
            del state, base_state
            return float(dt) * dmu_dt

        acoustic_wrf._mu_continuity_increment = raw_mu_increment
        changed = True
    if toggle.mu_mode == "zero":
        def no_mu_acoustic_config(grid: GridSpec, n_acoustic: int, rayleigh_coefficient: float) -> AcousticConfig:
            return dataclass_replace(
                original_acoustic_config(grid, n_acoustic, rayleigh_coefficient),
                mu_continuity=False,
            )

        d02_replay._acoustic_config = no_mu_acoustic_config
        changed = True
    if toggle.pressure_scale_override is not None:
        def forced_vertical_update(
            state: State,
            base_state: BaseState | None,
            metrics: DycoreMetrics,
            *,
            dt: float,
            epssm: float = 0.1,
            top_lid: bool = True,
            pressure_scale: float = 1.0,
            buoyancy_scale: float = 1.0,
            convective_buoyancy_gain: float = acoustic_wrf.CONVECTIVE_BUOYANCY_GAIN,
        ) -> State:
            del pressure_scale
            return original_vertical_update(
                state,
                base_state,
                metrics,
                dt=dt,
                epssm=epssm,
                top_lid=top_lid,
                pressure_scale=float(toggle.pressure_scale_override),
                buoyancy_scale=buoyancy_scale,
                convective_buoyancy_gain=convective_buoyancy_gain,
            )

        acoustic_wrf.vertical_acoustic_update = forced_vertical_update
        changed = True
    if changed:
        jax.clear_caches()
    try:
        yield
    finally:
        if changed:
            acoustic_wrf._mpas_w_metric_faces = original_metric
            acoustic_wrf._mpas_recurrence_vertical_update = original_recurrence
            acoustic_wrf._mu_continuity_increment = original_mu_increment
            d02_replay._acoustic_config = original_acoustic_config
            acoustic_wrf.vertical_acoustic_update = original_vertical_update
            jax.clear_caches()
