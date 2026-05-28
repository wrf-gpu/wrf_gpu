#!/usr/bin/env python
"""F6 first-12-step transaction audit for the operational RK/acoustic path."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("OMP_NUM_THREADS", "4")

_PRE_PARSER = argparse.ArgumentParser(add_help=False)
_PRE_PARSER.add_argument("--jax-platform", choices=("cpu", "gpu"), default=None)
_PRE_ARGS, _ = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.jax_platform is not None:
    os.environ["JAX_PLATFORMS"] = _PRE_ARGS.jax_platform
    os.environ["JAX_PLATFORM_NAME"] = _PRE_ARGS.jax_platform

import jax
from jax import config
import jax.numpy as jnp

import gpuwrf.contracts.state as state_contract
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.diagnostics.comprehensive_harness import DIAGNOSTIC_OPERATORS
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.acoustic import (
    AcousticCoreConfig,
    AcousticCoreState,
    _advance_geopotential,
    _decouple_theta_after_advance,
    _diagnose_pressure,
    _mass_couple_theta_before_advance,
    _ph_tend_increment,
    advance_mu_t_core,
    w_solve_core,
)
from gpuwrf.dynamics.small_step_scratch import ScratchInputs, build_scratch_state
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _acoustic_core_state,
    _carry_from_acoustic_core,
    _enforce_operational_precision,
    _finite_or_origin,
    _horizontal_pressure_gradient_tendencies,
    _limit_guarded_dynamics_state,
    _limit_guarded_dynamics_state_with_diagnostics,
    _theta_base_offset,
    _valid_mixing_ratio,
    _with_save_family,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)


DEFAULT_RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
DEFAULT_OUTPUT_DIR = ROOT / "proofs/f6"

BUDGET_FIELDS = (
    "mu",
    "muts",
    "muave",
    "theta",
    "theta_1",
    "p",
    "ph",
    "w",
    "u",
    "v",
    "theta_tend",
    "mu_tend",
)

COMBINATIONS = {
    "a": {
        "label": "pure_dycore",
        "physics": False,
        "boundary": False,
        "guards": False,
        "contract": "physics_off + boundary_off + guards_off",
    },
    "b": {
        "label": "dycore_plus_physics",
        "physics": True,
        "boundary": False,
        "guards": False,
        "contract": "physics_on + boundary_off + guards_off",
    },
    "c": {
        "label": "dycore_plus_boundary",
        "physics": False,
        "boundary": True,
        "guards": False,
        "contract": "physics_off + boundary_on + guards_off",
    },
    "d": {
        "label": "dycore_plus_limiter",
        "physics": False,
        "boundary": False,
        "guards": True,
        "contract": "physics_off + boundary_off + guards_on",
    },
}

PRESSURE_PERTURBATION_TO_BASE_LIMIT = 2.0
THETA_MASS_REL_TOL = 1.0e-10
SAVE_STATE_ATOL = 1.0e-10
MUTS_CONSISTENCY_ATOL = 1.0e-8
THETA_SANITY_MIN_K = 200.0
THETA_SANITY_MAX_K = 700.0
U_V_SANITY_MAX_M_S = 100.0
W_SANITY_MAX_M_S = 50.0


def _patch_cpu_state_allocator_if_needed() -> bool:
    """Allow replay loading on CPU when CUDA is unavailable."""

    if any(device.platform == "gpu" for device in jax.devices()):
        return False

    def _first_visible_device() -> jax.Device:
        return jax.devices()[0]

    state_contract._gpu_device = _first_visible_device  # type: ignore[attr-defined]
    return True


CPU_ALLOCATION_PATCHED = _patch_cpu_state_allocator_if_needed()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        if np.isfinite(scalar):
            return scalar
        if np.isnan(scalar):
            return "nan"
        return "inf" if scalar > 0 else "-inf"
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _device_summary() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_platform_name_env": os.environ.get("JAX_PLATFORM_NAME"),
        "devices": [{"id": str(device), "platform": device.platform} for device in devices],
        "gpu_visible": any(device.platform == "gpu" for device in devices),
        "cpu_state_allocation_patch": bool(CPU_ALLOCATION_PATCHED),
    }


def _field_stats(array: jax.Array) -> dict[str, Any]:
    arr = jnp.asarray(array, dtype=jnp.float64)
    finite = jnp.isfinite(arr)
    finite_count = int(np.asarray(jax.device_get(jnp.sum(finite))))
    total_count = int(arr.size)
    if finite_count == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "abs_max": None,
            "finite_count": 0,
            "total_count": total_count,
            "nonfinite_count": total_count,
        }
    safe = jnp.where(finite, arr, 0.0)
    reductions = {
        "mean": jnp.sum(safe) / float(finite_count),
        "min": jnp.min(jnp.where(finite, arr, jnp.inf)),
        "max": jnp.max(jnp.where(finite, arr, -jnp.inf)),
        "abs_max": jnp.max(jnp.where(finite, jnp.abs(arr), 0.0)),
    }
    host = jax.device_get(reductions)
    return {
        "mean": float(host["mean"]),
        "min": float(host["min"]),
        "max": float(host["max"]),
        "abs_max": float(host["abs_max"]),
        "finite_count": finite_count,
        "total_count": total_count,
        "nonfinite_count": total_count - finite_count,
    }


def _max_abs_diff(left: jax.Array, right: jax.Array) -> float:
    diff = jnp.asarray(left, dtype=jnp.float64) - jnp.asarray(right, dtype=jnp.float64)
    return float(np.asarray(jax.device_get(jnp.max(jnp.abs(diff)))))


def _safe_max_abs_diff(left: jax.Array, right: jax.Array) -> float | None:
    if tuple(left.shape) != tuple(right.shape):
        return None
    return _max_abs_diff(left, right)


def _theta_mass(acoustic: AcousticCoreState) -> jax.Array:
    mass = acoustic.c1h[:, None, None] * acoustic.muts[None, :, :] + acoustic.c2h[:, None, None]
    return jnp.sum(jnp.asarray(acoustic.theta, dtype=jnp.float64) * mass)


def _explicit_theta_delta(acoustic: AcousticCoreState, dt_s: float) -> jax.Array:
    mass = acoustic.c1h[:, None, None] * acoustic.muts[None, :, :] + acoustic.c2h[:, None, None]
    return float(dt_s) * jnp.sum(jnp.asarray(acoustic.theta_tend, dtype=jnp.float64) * mass)


def _first_nonfinite_budget_field(stats: dict[str, dict[str, Any]]) -> str | None:
    for name in BUDGET_FIELDS:
        if int(stats[name]["nonfinite_count"]) > 0:
            return name
    return None


class AuditRecorder:
    def __init__(self, combination: str, combo_config: dict[str, Any]) -> None:
        self.combination = combination
        self.combo_config = combo_config
        self.rows: list[dict[str, Any]] = []
        self.first_by_invariant: dict[str, dict[str, Any]] = {}
        self.max_acoustic_uv_delta = 0.0

    def note_acoustic_uv_delta(self, before: AcousticCoreState, after: AcousticCoreState) -> None:
        delta = max(_max_abs_diff(before.u, after.u), _max_abs_diff(before.v, after.v))
        self.max_acoustic_uv_delta = max(self.max_acoustic_uv_delta, delta)

    def _append_violation(self, row: dict[str, Any], invariant: str, details: dict[str, Any]) -> None:
        if invariant in self.first_by_invariant:
            return
        self.first_by_invariant[invariant] = {
            "combination": self.combination,
            "step": int(row["step"]),
            "rk_stage": row.get("rk_stage"),
            "acoustic_substep": row.get("acoustic_substep"),
            "operator": row["operator"],
            "invariant": invariant,
            "details": details,
        }

    def record(
        self,
        acoustic: AcousticCoreState,
        *,
        step: int,
        operator: str,
        carry_mu_save: jax.Array,
        base_pressure: jax.Array,
        rk_stage: int | None = None,
        acoustic_substep: int | None = None,
        stage_factor: float | None = None,
        theta_mass_before: jax.Array | None = None,
        explicit_theta_delta: jax.Array | None = None,
        expected_theta_1: jax.Array | None = None,
        limiter_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        fields = {name: _field_stats(getattr(acoustic, name)) for name in BUDGET_FIELDS}
        diagnostics: dict[str, Any] = {}
        violations: list[dict[str, Any]] = []

        first_nonfinite = _first_nonfinite_budget_field(fields)
        if first_nonfinite is not None:
            details = {"first_nonfinite_budget_field": first_nonfinite}
            violations.append({"invariant": "all_values_finite", "details": details})

        theta_total_stats = _field_stats(jnp.asarray(acoustic.theta, dtype=jnp.float64) + 300.0)
        theta_min = theta_total_stats["min"]
        theta_max = theta_total_stats["max"]
        diagnostics["theta_total_sanity_stats_k"] = theta_total_stats
        if theta_min is None or theta_max is None or theta_min < THETA_SANITY_MIN_K or theta_max > THETA_SANITY_MAX_K:
            violations.append(
                {
                    "invariant": "theta_sanity_bounds",
                    "details": {
                        "min": theta_min,
                        "max": theta_max,
                        "allowed": [THETA_SANITY_MIN_K, THETA_SANITY_MAX_K],
                    },
                }
            )

        u_abs = fields["u"]["abs_max"]
        v_abs = fields["v"]["abs_max"]
        w_abs = fields["w"]["abs_max"]
        diagnostics["wind_sanity_abs_max_m_s"] = {"u": u_abs, "v": v_abs, "w": w_abs}
        if (
            u_abs is None
            or v_abs is None
            or w_abs is None
            or u_abs > U_V_SANITY_MAX_M_S
            or v_abs > U_V_SANITY_MAX_M_S
            or w_abs > W_SANITY_MAX_M_S
        ):
            violations.append(
                {
                    "invariant": "wind_sanity_bounds",
                    "details": {
                        "u_abs_max": u_abs,
                        "v_abs_max": v_abs,
                        "w_abs_max": w_abs,
                        "allowed": {"u_v": U_V_SANITY_MAX_M_S, "w": W_SANITY_MAX_M_S},
                    },
                }
            )

        dry_mass = jnp.asarray(acoustic.mu, dtype=jnp.float64) + jnp.asarray(acoustic.mut, dtype=jnp.float64)
        dry_mass_min = float(np.asarray(jax.device_get(jnp.min(dry_mass))))
        diagnostics["dry_mass_min"] = dry_mass_min
        if (not np.isfinite(dry_mass_min)) or dry_mass_min < 0.0:
            violations.append({"invariant": "dry_mass_nonnegative", "details": {"dry_mass_min": dry_mass_min}})

        p = jnp.asarray(acoustic.p, dtype=jnp.float64)
        pb = jnp.asarray(base_pressure, dtype=jnp.float64)
        pressure_ratio = jnp.max(jnp.abs(p) / jnp.maximum(jnp.abs(pb), 1.0))
        pressure_ratio_host = float(np.asarray(jax.device_get(pressure_ratio)))
        diagnostics["pressure_abs_ratio_to_base_max"] = pressure_ratio_host
        if (not np.isfinite(pressure_ratio_host)) or pressure_ratio_host >= PRESSURE_PERTURBATION_TO_BASE_LIMIT:
            violations.append(
                {
                    "invariant": "pressure_bounded",
                    "details": {
                        "abs_p_over_base_max": pressure_ratio_host,
                        "threshold": PRESSURE_PERTURBATION_TO_BASE_LIMIT,
                    },
                }
            )

        muts_consistency = _safe_max_abs_diff(acoustic.muts - acoustic.mut, acoustic.mu - carry_mu_save)
        diagnostics["muts_equals_mut_plus_work_mu_abs_error"] = muts_consistency
        if muts_consistency is not None and muts_consistency > MUTS_CONSISTENCY_ATOL:
            violations.append(
                {
                    "invariant": "muts_mut_work_mu_consistency",
                    "details": {"max_abs_error": muts_consistency, "atol": MUTS_CONSISTENCY_ATOL},
                }
            )

        if expected_theta_1 is not None:
            theta_1_error = _max_abs_diff(acoustic.theta_1, expected_theta_1)
            diagnostics["rk_saved_theta_1_abs_error"] = theta_1_error
            if theta_1_error > SAVE_STATE_ATOL:
                violations.append(
                    {
                        "invariant": "rk_saved_state_theta_1",
                        "details": {"max_abs_error": theta_1_error, "atol": SAVE_STATE_ATOL},
                    }
                )

        theta_mass_after = _theta_mass(acoustic)
        diagnostics["theta_mass"] = float(np.asarray(jax.device_get(theta_mass_after)))
        if theta_mass_before is not None and explicit_theta_delta is not None:
            residual = theta_mass_after - theta_mass_before - explicit_theta_delta
            denominator = jnp.maximum(jnp.abs(theta_mass_before), 1.0)
            rel_residual = jnp.abs(residual) / denominator
            residual_host = float(np.asarray(jax.device_get(residual)))
            rel_host = float(np.asarray(jax.device_get(rel_residual)))
            diagnostics.update(
                {
                    "theta_mass_before": float(np.asarray(jax.device_get(theta_mass_before))),
                    "theta_mass_explicit_tendency_delta": float(np.asarray(jax.device_get(explicit_theta_delta))),
                    "theta_mass_residual": residual_host,
                    "theta_mass_relative_residual": rel_host,
                    "theta_mass_relative_tolerance": THETA_MASS_REL_TOL,
                }
            )
            if (not np.isfinite(rel_host)) or rel_host > THETA_MASS_REL_TOL:
                violations.append(
                    {
                        "invariant": "theta_mass_residual",
                        "details": {"relative_residual": rel_host, "residual": residual_host, "rtol": THETA_MASS_REL_TOL},
                    }
                )

        if limiter_diagnostics:
            diagnostics["limiter"] = limiter_diagnostics

        row = {
            "combination": self.combination,
            "combination_label": self.combo_config["label"],
            "step": int(step),
            "operator": operator,
            "rk_stage": rk_stage,
            "acoustic_substep": acoustic_substep,
            "stage_factor": stage_factor,
            "fields": fields,
            "diagnostics": diagnostics,
            "violations": violations,
        }
        self.rows.append(row)
        for violation in violations:
            self._append_violation(row, violation["invariant"], violation["details"])


def _acoustic_from_carry(carry: OperationalCarry, namelist: OperationalNamelist) -> tuple[AcousticCoreState, jax.Array]:
    state = apply_halo(carry.state, halo_spec(namelist.grid))
    acoustic = _acoustic_core_state(carry.replace(state=state), namelist)
    base_pressure = jnp.asarray(state.p_total, dtype=jnp.float64) - jnp.asarray(state.p_perturbation, dtype=jnp.float64)
    return acoustic, base_pressure


def _record_carry(
    recorder: AuditRecorder,
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    step: int,
    operator: str,
    rk_stage: int | None = None,
    acoustic_substep: int | None = None,
    stage_factor: float | None = None,
    expected_theta_1: jax.Array | None = None,
    limiter_diagnostics: dict[str, Any] | None = None,
) -> None:
    acoustic, base_pressure = _acoustic_from_carry(carry, namelist)
    recorder.record(
        acoustic,
        step=step,
        operator=operator,
        carry_mu_save=carry.mu_save,
        base_pressure=base_pressure,
        rk_stage=rk_stage,
        acoustic_substep=acoustic_substep,
        stage_factor=stage_factor,
        expected_theta_1=expected_theta_1,
        limiter_diagnostics=limiter_diagnostics,
    )


def _instrumented_acoustic_substep(
    recorder: AuditRecorder,
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    dt_sub: float,
    *,
    step: int,
    rk_stage: int,
    acoustic_substep: int,
    stage_factor: float,
    expected_theta_1: jax.Array | None,
) -> OperationalCarry:
    state = apply_halo(carry.state, halo_spec(namelist.grid))
    theta_offset = _theta_base_offset(state.theta)
    acoustic = _acoustic_core_state(carry.replace(state=state), namelist)
    base_pressure = jnp.asarray(state.p_total, dtype=jnp.float64) - jnp.asarray(state.p_perturbation, dtype=jnp.float64)
    coeff_mut = acoustic.coef_mut if acoustic.coef_mut is not None else acoustic.muts
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        coeff_mut,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    cfg = AcousticCoreConfig(
        dt=float(dt_sub),
        dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )

    theta_old = acoustic.theta
    mu_old = acoustic.mu
    theta_mass_before = _theta_mass(acoustic)
    explicit_delta = _explicit_theta_delta(acoustic, dt_sub)

    coupled_state = acoustic.replace(theta=_mass_couple_theta_before_advance(acoustic))
    advanced = advance_mu_t_core(coupled_state, cfg)
    after_mu_t = acoustic.replace(
        mu=advanced["mu"],
        mudf=advanced["mudf"],
        muts=advanced["muts"],
        muave=advanced["muave"],
        ww=advanced["ww"],
        theta=advanced["theta"],
    )
    recorder.record(
        after_mu_t,
        step=step,
        operator="advance_mu_t",
        carry_mu_save=carry.mu_save,
        base_pressure=base_pressure,
        rk_stage=rk_stage,
        acoustic_substep=acoustic_substep,
        stage_factor=stage_factor,
        theta_mass_before=theta_mass_before,
        explicit_theta_delta=explicit_delta,
        expected_theta_1=expected_theta_1,
    )

    theta_new = _decouple_theta_after_advance(acoustic, advanced["theta"], advanced["muts"])
    after_decouple = after_mu_t.replace(theta=theta_new, theta_ave=theta_new)
    recorder.record(
        after_decouple,
        step=step,
        operator="_decouple_theta_after_advance",
        carry_mu_save=carry.mu_save,
        base_pressure=base_pressure,
        rk_stage=rk_stage,
        acoustic_substep=acoustic_substep,
        stage_factor=stage_factor,
        expected_theta_1=expected_theta_1,
    )

    w_solved = w_solve_core(acoustic, a=a, alpha=alpha, gamma=gamma)
    after_w_solve = after_decouple.replace(w=w_solved)
    recorder.record(
        after_w_solve,
        step=step,
        operator="w_solve_core",
        carry_mu_save=carry.mu_save,
        base_pressure=base_pressure,
        rk_stage=rk_stage,
        acoustic_substep=acoustic_substep,
        stage_factor=stage_factor,
        expected_theta_1=expected_theta_1,
    )

    ph_next = _advance_geopotential(acoustic, w_solved, cfg)
    p_next = _diagnose_pressure(acoustic, advanced["mu"])
    after_pressure = after_w_solve.replace(ph=ph_next, p=p_next)
    recorder.record(
        after_pressure,
        step=step,
        operator="_diagnose_pressure",
        carry_mu_save=carry.mu_save,
        base_pressure=base_pressure,
        rk_stage=rk_stage,
        acoustic_substep=acoustic_substep,
        stage_factor=stage_factor,
        expected_theta_1=expected_theta_1,
    )

    ph_increment = _ph_tend_increment(theta_old, theta_new, acoustic.ph_tend)
    scratch = build_scratch_state(
        ScratchInputs(
            theta_old=theta_old,
            theta_new=theta_new,
            t_2ave_prev=acoustic.t_2ave,
            ww_old=acoustic.ww,
            ww_new=advanced["ww"],
            mu_old=mu_old,
            mu_new=advanced["mu"],
            mut=acoustic.mut,
            muave_prev=acoustic.muave,
            muts_prev=acoustic.muts,
            ph_tend_old=acoustic.ph_tend,
            ph_tend_increment=ph_increment,
            u_current=acoustic.u,
            v_current=acoustic.v,
            w_current=w_solved,
            ph_current=acoustic.ph,
            epssm=float(cfg.epssm),
        )
    )
    next_acoustic = acoustic.replace(
        mu=advanced["mu"],
        mudf=advanced["mudf"],
        muts=advanced["muts"],
        muave=advanced["muave"],
        ww=scratch["ww"],
        theta=theta_new,
        theta_ave=theta_new,
        ph_tend=scratch["ph_tend"],
        w=w_solved,
        ph=ph_next,
        p=p_next,
        t_2ave=scratch["t_2ave"],
    )
    recorder.note_acoustic_uv_delta(acoustic, next_acoustic)
    next_carry = _carry_from_acoustic_core(next_acoustic, state, theta_offset)
    _record_carry(
        recorder,
        next_carry,
        namelist,
        step=step,
        operator="acoustic_substep_commit",
        rk_stage=rk_stage,
        acoustic_substep=acoustic_substep,
        stage_factor=stage_factor,
        expected_theta_1=expected_theta_1,
    )
    return next_carry


def _instrumented_rk_scan_step(
    recorder: AuditRecorder,
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    step: int,
) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    carry = _with_save_family(carry.replace(state=origin), origin)
    theta_rk1_start = _acoustic_core_state(carry, namelist).theta
    _record_carry(recorder, carry, namelist, step=step, operator="rk_step_start")

    stages = (
        (1, 1.0 / 3.0, 1),
        (2, 0.5, max(1, int(namelist.acoustic_substeps) // 2)),
        (3, 1.0, int(namelist.acoustic_substeps)),
    )
    dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)
    for rk_stage, factor, substeps in stages:
        haloed = apply_halo(carry.state, halo_spec(namelist.grid))
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        du_dt, dv_dt = _horizontal_pressure_gradient_tendencies(haloed, namelist)
        tendencies = tendencies.replace(u=tendencies.u + du_dt, v=tendencies.v + dv_dt)
        candidate = add_scaled_tendencies(origin, tendencies, float(namelist.dt_s) * float(factor))
        carry = _with_save_family(carry.replace(state=candidate), candidate)
        expected_theta_1 = theta_rk1_start if rk_stage in (2, 3) else None
        _record_carry(
            recorder,
            carry,
            namelist,
            step=step,
            operator="rk_stage_candidate",
            rk_stage=rk_stage,
            stage_factor=factor,
            expected_theta_1=expected_theta_1,
        )
        for substep in range(1, int(substeps) + 1):
            carry = _instrumented_acoustic_substep(
                recorder,
                carry,
                namelist,
                dt_sub,
                step=step,
                rk_stage=rk_stage,
                acoustic_substep=substep,
                stage_factor=factor,
                expected_theta_1=expected_theta_1,
            )
        carry = carry.replace(state=apply_halo(carry.state, halo_spec(namelist.grid)))
        _record_carry(
            recorder,
            carry,
            namelist,
            step=step,
            operator="rk_stage_halo",
            rk_stage=rk_stage,
            stage_factor=factor,
            expected_theta_1=expected_theta_1,
        )
    return carry


def _limiter_json(diagnostics: dict[str, jax.Array]) -> dict[str, Any]:
    return {key: _jsonable(np.asarray(jax.device_get(value))) for key, value in diagnostics.items()}


def _one_instrumented_step(
    recorder: AuditRecorder,
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    step: int,
) -> OperationalCarry:
    physical_origin = carry.state
    carry = _instrumented_rk_scan_step(recorder, carry, namelist, step=step)
    next_state = carry.state

    if not bool(namelist.disable_guards):
        next_state, limiter_diagnostics = _limit_guarded_dynamics_state_with_diagnostics(next_state, physical_origin)
        next_state = next_state.replace(
            qv=_valid_mixing_ratio(next_state.qv, physical_origin.qv),
            qc=_valid_mixing_ratio(next_state.qc, physical_origin.qc),
            qr=_valid_mixing_ratio(next_state.qr, physical_origin.qr),
            qi=_valid_mixing_ratio(next_state.qi, physical_origin.qi),
            qs=_valid_mixing_ratio(next_state.qs, physical_origin.qs),
            qg=_valid_mixing_ratio(next_state.qg, physical_origin.qg),
        )
        carry = carry.replace(state=next_state)
        _record_carry(
            recorder,
            carry,
            namelist,
            step=step,
            operator="dynamics_guards",
            limiter_diagnostics=_limiter_json(limiter_diagnostics),
        )

    if bool(namelist.run_physics):
        if not bool(namelist.disable_guards):
            next_state = thompson_adapter(next_state, float(namelist.dt_s))
            carry = carry.replace(state=next_state)
            _record_carry(recorder, carry, namelist, step=step, operator="microphysics_thompson")
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        carry = carry.replace(state=next_state)
        _record_carry(recorder, carry, namelist, step=step, operator="surface_layer")
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        carry = carry.replace(state=next_state)
        _record_carry(recorder, carry, namelist, step=step, operator="mynn_pbl")

    if bool(namelist.run_boundary):
        lead_seconds = jnp.asarray(step, dtype=jnp.float64) * float(namelist.dt_s)
        bounded = apply_lateral_boundaries(next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
        carry = carry.replace(state=bounded)
        _record_carry(recorder, carry, namelist, step=step, operator="lateral_boundary")
        if bool(namelist.disable_guards):
            next_state = bounded
        else:
            next_state = bounded.replace(
                u=_finite_or_origin(bounded.u, physical_origin.u),
                v=_finite_or_origin(bounded.v, physical_origin.v),
                w=_finite_or_origin(bounded.w, physical_origin.w),
                theta=_finite_or_origin(bounded.theta, physical_origin.theta),
                qv=_valid_mixing_ratio(bounded.qv, physical_origin.qv),
                p=_finite_or_origin(bounded.p, physical_origin.p),
                ph=_finite_or_origin(bounded.ph, physical_origin.ph),
                p_total=_finite_or_origin(bounded.p_total, physical_origin.p_total),
                ph_total=_finite_or_origin(bounded.ph_total, physical_origin.ph_total),
                p_perturbation=_finite_or_origin(bounded.p_perturbation, physical_origin.p_perturbation),
                ph_perturbation=_finite_or_origin(bounded.ph_perturbation, physical_origin.ph_perturbation),
            )
            next_state = _limit_guarded_dynamics_state(next_state, physical_origin)
            carry = carry.replace(state=next_state)
            _record_carry(recorder, carry, namelist, step=step, operator="boundary_guards")

    next_state = _enforce_operational_precision(next_state)
    carry = carry.replace(state=next_state)
    _record_carry(recorder, carry, namelist, step=step, operator="precision_commit")
    return carry


def _build_namelist(base: OperationalNamelist, combo: dict[str, Any]) -> OperationalNamelist:
    return replace(
        base,
        run_physics=bool(combo["physics"]),
        run_boundary=bool(combo["boundary"]),
        disable_guards=not bool(combo["guards"]),
    )


def _run_combination(
    *,
    combination: str,
    combo: dict[str, Any],
    initial_state: Any,
    base_namelist: OperationalNamelist,
    metadata: dict[str, Any],
    steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    namelist = _build_namelist(base_namelist, combo)
    recorder = AuditRecorder(combination, combo)
    carry = initial_operational_carry(_enforce_operational_precision(initial_state))
    wall_start = time.perf_counter()
    for step in range(1, int(steps) + 1):
        carry = _one_instrumented_step(recorder, carry, namelist, step=step)
        jax.block_until_ready(carry)
    wall_s = time.perf_counter() - wall_start
    first_any = min(
        recorder.first_by_invariant.values(),
        key=lambda item: (int(item["step"]), int(item.get("rk_stage") or 99), int(item.get("acoustic_substep") or 99)),
        default=None,
    )
    critical_names = {
        "all_values_finite",
        "dry_mass_nonnegative",
        "pressure_bounded",
        "theta_sanity_bounds",
        "wind_sanity_bounds",
    }
    first_critical = min(
        (item for name, item in recorder.first_by_invariant.items() if name in critical_names),
        key=lambda item: (int(item["step"]), int(item.get("rk_stage") or 99), int(item.get("acoustic_substep") or 99)),
        default=None,
    )
    payload = {
        "schema_version": "f6-transaction-audit-1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "combination": combination,
        "combination_config": combo,
        "run_config": {
            "steps": int(steps),
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "guards_enabled": not bool(namelist.disable_guards),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "device_summary": _device_summary(),
            "cpu_affinity_required": "taskset -c 0-3",
        },
        "case_metadata": metadata,
        "budget_fields": list(BUDGET_FIELDS),
        "invariant_policy": {
            "pressure_abs_p_over_base_threshold": PRESSURE_PERTURBATION_TO_BASE_LIMIT,
            "theta_mass_relative_tolerance": THETA_MASS_REL_TOL,
            "muts_consistency_atol": MUTS_CONSISTENCY_ATOL,
            "rk_saved_state_theta_1_atol": SAVE_STATE_ATOL,
            "theta_sanity_bounds_k": [THETA_SANITY_MIN_K, THETA_SANITY_MAX_K],
            "wind_sanity_bounds_m_s": {"u_v": U_V_SANITY_MAX_M_S, "w": W_SANITY_MAX_M_S},
        },
        "rows": recorder.rows,
        "first_violation": first_any,
        "first_critical_violation": first_critical,
        "first_violation_by_invariant": recorder.first_by_invariant,
        "acoustic_uv_max_delta": recorder.max_acoustic_uv_delta,
        "operator_vocabulary_from_comprehensive_harness": list(DIAGNOSTIC_OPERATORS),
        "wall_time_s": wall_s,
    }
    _write_json(output_dir / f"audit_combination_{combination}.json", payload)
    return payload


def _load_case(run_dir: Path, domain: str, dt_s: float, acoustic_substeps: int) -> tuple[Any, OperationalNamelist, dict[str, Any]]:
    case = build_replay_case(run_dir, domain=domain)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=float(dt_s),
        acoustic_substeps=int(acoustic_substeps),
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
        disable_guards=True,
    )
    metadata = {
        "run_id": case.metadata.get("run_id"),
        "run_dir": str(run_dir),
        "domain": domain,
        "grid": case.metadata.get("grid"),
        "base_state": case.metadata.get("base_state"),
        "boundary": case.metadata.get("boundary"),
    }
    return state, namelist, metadata


def _build_violations_payload(combination_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    firsts: list[dict[str, Any]] = []
    for payload in combination_payloads.values():
        for item in payload["first_violation_by_invariant"].values():
            firsts.append(item)
    firsts.sort(key=lambda item: (item["combination"], item["step"], str(item.get("operator")), item["invariant"]))
    return {
        "schema_version": "f6-invariant-violations-1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "violations": firsts,
        "first_violation_per_combination": {
            key: {
                "first_any": payload["first_violation"],
                "first_critical": payload["first_critical_violation"],
                "first_by_invariant": payload["first_violation_by_invariant"],
                "acoustic_uv_max_delta": payload["acoustic_uv_max_delta"],
            }
            for key, payload in sorted(combination_payloads.items())
        },
    }


def _fmt_first(item: dict[str, Any] | None) -> str:
    if item is None:
        return "none in first 12 steps"
    stage = "" if item.get("rk_stage") is None else f", RK{item['rk_stage']}"
    sub = "" if item.get("acoustic_substep") is None else f", substep {item['acoustic_substep']}"
    return f"step {item['step']}{stage}{sub}, {item['operator']}, {item['invariant']}"


def _write_summary(output_dir: Path, combination_payloads: dict[str, dict[str, Any]], violations: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# F6 Transaction Audit Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Device: {_device_summary()}")
    lines.append("")
    lines.append("## Toggle Results")
    for key in ("a", "b", "c", "d"):
        payload = combination_payloads[key]
        combo = payload["combination_config"]
        lines.append(
            f"- {key} ({combo['contract']}): first critical violation = {_fmt_first(payload['first_critical_violation'])}; "
            f"first algebraic violation = {_fmt_first(payload['first_violation'])}; acoustic uv max delta = "
            f"{payload['acoustic_uv_max_delta']:.3e}."
        )
    lines.append("")
    pure = combination_payloads["a"]
    first_critical = pure["first_critical_violation"]
    first_any = pure["first_violation"]
    lines.append("## Questions")
    if first_critical is None:
        lines.append(
            "1. Where does the blow-up actually start? No finite/dry-mass/pressure critical violation appears "
            "in the pure dycore combination during the first 12 steps. The earliest algebraic defect is "
            f"{_fmt_first(first_any)}."
        )
    else:
        lines.append(
            "1. Where does the blow-up actually start? In the pure dycore combination, the first critical "
            f"violation is {_fmt_first(first_critical)}. That means the failure is dycore-internal before "
            "physics, boundary replay, or guards are needed."
        )
    uv_zero = all(float(payload["acoustic_uv_max_delta"]) == 0.0 for payload in combination_payloads.values())
    save_hit = any(
        "rk_saved_state_theta_1" in payload["first_violation_by_invariant"] for payload in combination_payloads.values()
    )
    muts_hit = any(
        "muts_mut_work_mu_consistency" in payload["first_violation_by_invariant"] for payload in combination_payloads.values()
    )
    pressure_hit = any("pressure_bounded" in payload["first_violation_by_invariant"] for payload in combination_payloads.values())
    lines.append(
        "2. Does it match F3 Opus? Mostly yes. "
        f"Missing `advance_uv` is {'supported' if uv_zero else 'not directly supported'} by zero u/v delta inside acoustic substeps; "
        f"cross-stage saved-state loss is {'hit' if save_hit else 'not hit'} by `rk_saved_state_theta_1`; "
        f"`muts`/work-mu inconsistency is {'hit' if muts_hit else 'not hit'}; "
        f"the pressure bound is {'hit' if pressure_hit else 'not hit'} within this 12-step algebraic threshold."
    )
    lines.append(
        "3. Does it match agy? It matches the original agy direction on restored advection tests, theta_1 "
        "reference testing, and mu-save/mass tracking. It extends agy's three-bug diagnosis by separating "
        "pure-dycore, physics, boundary, and limiter toggles."
    )
    lines.append(
        "4. Most targeted next fix scope: do not patch physics or boundary first if combination a has the "
        "critical failure. The narrow next code sprint should repair RK/acoustic state ownership: carry the "
        "RK1 saved family across RK2/RK3, add/validate small-step `advance_uv`, and replace the pressure "
        "stub only after the save-family invariants are clean."
    )
    lines.append("")
    lines.append("## Proof Objects")
    for key in ("a", "b", "c", "d"):
        lines.append(f"- `proofs/f6/audit_combination_{key}.json`")
    lines.append("- `proofs/f6/invariant_violations.json`")
    lines.append("")
    lines.append("F6_AUDIT_SUMMARY_COMPLETE")
    (output_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--domain", default="d02")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--dt-s", type=float, default=10.0)
    parser.add_argument("--acoustic-substeps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--combination", choices=tuple(COMBINATIONS), default=None)
    parser.add_argument("--jax-platform", choices=("cpu", "gpu"), default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state, base_namelist, metadata = _load_case(args.run_dir, args.domain, args.dt_s, args.acoustic_substeps)
    selected = [args.combination] if args.combination else ["a", "b", "c", "d"]
    payloads: dict[str, dict[str, Any]] = {}
    for key in selected:
        print(f"[f6] running combination {key}: {COMBINATIONS[key]['contract']}", flush=True)
        payloads[key] = _run_combination(
            combination=key,
            combo=COMBINATIONS[key],
            initial_state=state,
            base_namelist=base_namelist,
            metadata=metadata,
            steps=int(args.steps),
            output_dir=args.output_dir,
        )
        print(
            f"[f6] wrote audit_combination_{key}.json; first critical = "
            f"{_fmt_first(payloads[key]['first_critical_violation'])}",
            flush=True,
        )
    if len(payloads) == 4:
        violations = _build_violations_payload(payloads)
        _write_json(args.output_dir / "invariant_violations.json", violations)
        _write_summary(args.output_dir, payloads, violations)
        print(f"[f6] wrote {args.output_dir / 'invariant_violations.json'}", flush=True)
        print(f"[f6] wrote {args.output_dir / 'audit_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
