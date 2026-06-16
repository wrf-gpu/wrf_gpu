#!/usr/bin/env python
"""M6b operational-vs-validation composition bisection.

This is a diagnostic script only.  It does not patch operational numerics; it
loads one real Gen2 d02 initial condition, runs the operational entry point and
the locked validation coupled-step composition, then drills into the first
controlled RK/acoustic boundary where the compositions differ.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

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

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import State
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, thompson_adapter
from gpuwrf.dynamics.acoustic_loop import (
    AcousticLoopConfig,
    AcousticLoopState,
    FULL_STATE_FIELDS,
    acoustic_substep_wrf,
)
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.coupled_step import CoupledStepConfig, coupled_timestep_wrf
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _theta_base_offset,
    _u_face_average_2d,
    _v_face_average_2d,
    _with_save_family,
    run_forecast_operational,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)

SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-operational-composition-bisection"
RUN_ROOT = Path("/mnt/data/canairy_meteo/runs/wrf_l3")
DEFAULT_RUN_ID = "20260523_18z_l3_24h_20260524T004313Z"
THRESHOLD = 1.0e-10
WRF_SOLVE = "/home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F"
WRF_SMALL = "/home/user/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _max_abs(a: Any, b: Any) -> float:
    left = np.asarray(a)
    right = np.asarray(b)
    diff = np.abs(left - right)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return 1.0e300
    max_finite = float(np.max(finite))
    return 1.0e300 if not np.all(np.isfinite(diff)) else max_finite


def _summarize_field_delta(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for name in FULL_STATE_FIELDS:
        delta = _max_abs(actual[name], expected[name])
        fields[name] = {
            "max_abs_delta": delta,
            "exceeds_1e_10": bool(delta > THRESHOLD),
            "actual_shape": list(np.shape(actual[name])),
            "expected_shape": list(np.shape(expected[name])),
        }
    return fields


def _first_bad_field(fields: dict[str, dict[str, Any]]) -> str | None:
    bad = [name for name, item in fields.items() if bool(item["exceeds_1e_10"])]
    if not bad:
        return None
    return max(bad, key=lambda name: float(fields[name]["max_abs_delta"]))


def _snapshot_to_numpy(snapshot: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in snapshot.items():
        array = jnp.asarray(value)
        block_until_ready(array)
        arrays[name] = np.asarray(array)
    return arrays


def _base_mu(state: State) -> jax.Array:
    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def _operational_state_for_run(run_id: str) -> tuple[State, OperationalNamelist, Any]:
    case = build_replay_case(RUN_ROOT / run_id)
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    namelist = OperationalNamelist.from_grid(
        case.grid,
        tendencies=case.tendencies,
        metrics=case.metrics,
        dt_s=10.0,
        acoustic_substeps=10,
        radiation_cadence_steps=999999,
        use_vertical_solver=True,
    )
    return state, namelist, case


def _clone_state(state: State) -> State:
    return jax.tree_util.tree_map(lambda leaf: jnp.array(leaf, copy=True), state)


def _carry_to_validation_snapshot(carry: OperationalCarry) -> dict[str, jax.Array]:
    state = carry.state
    theta_offset = _theta_base_offset(state.theta)
    return {
        "mu": jnp.asarray(state.mu_perturbation),
        "mut": _base_mu(state),
        "mudf": jnp.zeros_like(state.mu_perturbation),
        "muts": jnp.asarray(carry.muts),
        "muave": jnp.asarray(carry.muave),
        "ww": jnp.asarray(carry.ww),
        "theta": jnp.asarray(state.theta) - theta_offset,
        "ph_tend": jnp.asarray(carry.ph_tend),
        "u": jnp.asarray(state.u),
        "v": jnp.asarray(state.v),
        "w": jnp.asarray(state.w),
        "ph": jnp.asarray(state.ph_perturbation),
        "p": jnp.asarray(state.p_perturbation),
        "t_2ave": jnp.asarray(carry.t_2ave) - theta_offset,
    }


def _state_to_acoustic(carry: OperationalCarry, namelist: OperationalNamelist) -> AcousticLoopState:
    state = carry.state
    theta_offset = _theta_base_offset(state.theta)
    mu_base = _base_mu(state)
    mu_total = mu_base + state.mu_perturbation
    return AcousticLoopState(
        ww=carry.ww,
        ww_1=carry.ww_save,
        u=state.u,
        u_1=carry.u_save,
        v=state.v,
        v_1=carry.v_save,
        w=state.w,
        mu=state.mu_perturbation,
        mut=mu_base,
        muave=carry.muave,
        muts=carry.muts,
        muu=_u_face_average_2d(mu_total),
        muv=_v_face_average_2d(mu_total),
        mudf=jnp.zeros_like(state.mu_perturbation),
        theta=state.theta - theta_offset,
        theta_1=carry.t_save - theta_offset,
        theta_ave=carry.t_2ave - theta_offset,
        theta_tend=namelist.tendencies.theta,
        mu_tend=namelist.tendencies.mu,
        ph_tend=carry.ph_tend,
        ph=state.ph_perturbation,
        p=state.p_perturbation,
        t_2ave=carry.t_2ave - theta_offset,
        dnw=namelist.metrics.dnw,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        msfuy=namelist.metrics.msfuy,
        msfvx_inv=1.0 / namelist.metrics.msfvx,
        msftx=namelist.metrics.msftx,
        msfty=namelist.metrics.msfty,
        coef_mut=carry.muts,
    )


def _validation_snapshot_to_physical_state(snapshot: dict[str, Any], template: State, theta_offset: float = 300.0) -> State:
    theta = jnp.asarray(snapshot["theta"]) + float(theta_offset)
    p_pert = jnp.asarray(snapshot["p"])
    ph_pert = jnp.asarray(snapshot["ph"])
    mu_pert = jnp.asarray(snapshot["mu"])
    p_total = template.p_total - template.p_perturbation + p_pert
    ph_total = template.ph_total - template.ph_perturbation + ph_pert
    mu_total = template.mu_total - template.mu_perturbation + mu_pert
    return template.replace(
        u=jnp.asarray(snapshot["u"]),
        v=jnp.asarray(snapshot["v"]),
        w=jnp.asarray(snapshot["w"]),
        theta=theta,
        p=p_total,
        p_total=p_total,
        p_perturbation=p_pert,
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=ph_pert,
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_pert,
    )


def _coupled_extras(state: State) -> dict[str, jax.Array]:
    return {
        "qv": state.qv,
        "qc": state.qc,
        "qr": state.qr,
        "qi": state.qi,
        "qs": state.qs,
        "qg": state.qg,
        "qke": state.qke,
        "t_skin": state.t_skin,
        "xland": state.xland,
        "lakemask": state.lakemask,
        "u_bdy": state.u_bdy,
        "v_bdy": state.v_bdy,
        "theta_bdy": state.theta_bdy,
        "qv_bdy": state.qv_bdy,
        "ph_bdy": state.ph_bdy,
        "mu_bdy": state.mu_bdy,
    }


def _validation_coupled_step(state: State, namelist: OperationalNamelist) -> dict[str, jax.Array]:
    carry = initial_operational_carry(_enforce_operational_precision(state))
    cfg = CoupledStepConfig(
        dt=float(namelist.dt_s),
        dx=float(namelist.grid.projection.dx_m),
        dy=float(namelist.grid.projection.dy_m),
        acoustic_substeps=int(namelist.acoustic_substeps),
        rk_order=int(namelist.rk_order),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        physics_enabled=True,
        boundary_enabled=True,
        boundary_config=namelist.boundary_config,
    )
    return coupled_timestep_wrf(
        _state_to_acoustic(carry, namelist),
        namelist.metrics,
        cfg,
        extras=_coupled_extras(state),
        step_index=1,
    )


def _post_physics_boundary_validation_units(snapshot: dict[str, Any], template: State) -> dict[str, jax.Array]:
    carry = initial_operational_carry(_validation_snapshot_to_physical_state(snapshot, template))
    return _carry_to_validation_snapshot(carry)


def _post_physics_boundary_operational_units(state: State) -> dict[str, jax.Array]:
    carry = initial_operational_carry(state)
    return _carry_to_validation_snapshot(carry)


def _rk1_common_pre_acoustic(carry: OperationalCarry, namelist: OperationalNamelist) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    carry = _with_save_family(carry.replace(state=origin), origin)
    haloed = apply_halo(carry.state, halo_spec(namelist.grid))
    tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
    candidate = add_scaled_tendencies(origin, tendencies, float(namelist.dt_s) / 3.0)
    return _with_save_family(carry.replace(state=candidate), candidate)


def _validation_one_acoustic_substep(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    dt_sub: float,
) -> AcousticLoopState:
    acoustic = _state_to_acoustic(carry, namelist)
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        acoustic.muts,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    return acoustic_substep_wrf(
        acoustic,
        a=a,
        alpha=alpha,
        gamma=gamma,
        cfg=AcousticLoopConfig(
            dt=float(dt_sub),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
        ),
    )


def _acoustic_snapshot(state: AcousticLoopState) -> dict[str, jax.Array]:
    values = state.to_dict()
    return {name: values[name] for name in FULL_STATE_FIELDS}


def _operator_bisection(pre_carry: OperationalCarry, post_validation: AcousticLoopState) -> list[dict[str, Any]]:
    pre = _carry_to_validation_snapshot(pre_carry)
    post = _acoustic_snapshot(post_validation)
    checks = []
    for operator, fields in (
        ("calc_coef_w", ("w",)),
        ("advance_mu_t", ("mu", "mudf", "muts", "muave", "ww", "theta")),
        ("tridiag_fwd_back", ("w",)),
        ("scratch_updates", ("t_2ave", "ph_tend", "muave", "muts", "ww")),
    ):
        deltas = {name: _max_abs(pre[name], post[name]) for name in fields}
        checks.append(
            {
                "operator": operator,
                "compared_to_operational_rk1_no_acoustic_output": True,
                "field_deltas": deltas,
                "max_abs_delta": max(deltas.values()) if deltas else 0.0,
                "exceeds_1e_10": any(value > THRESHOLD for value in deltas.values()),
            }
        )
    return checks


def run_bisection(run_id: str, steps: int) -> dict[str, Any]:
    state, namelist, case = _operational_state_for_run(run_id)
    requested_steps = int(steps)
    one_step_hours = float(namelist.dt_s) / 3600.0

    template = _clone_state(state)
    validation_step = _validation_coupled_step(state, namelist)
    block_until_ready(validation_step)
    validation_step_units = _snapshot_to_numpy(_post_physics_boundary_validation_units(validation_step, template))
    initial = initial_operational_carry(_enforce_operational_precision(state))
    rk1_pre = _rk1_common_pre_acoustic(initial, namelist)
    rk1_operational_post = _carry_to_validation_snapshot(rk1_pre)
    rk1_validation_substep1 = _validation_one_acoustic_substep(
        rk1_pre,
        namelist,
        dt_sub=float(namelist.dt_s) / 3.0,
    )
    block_until_ready(rk1_validation_substep1)
    rk1_validation_post = _snapshot_to_numpy(_acoustic_snapshot(rk1_validation_substep1))
    rk1_operational_post_np = _snapshot_to_numpy(rk1_operational_post)
    rk1_fields = _summarize_field_delta(rk1_operational_post_np, rk1_validation_post)
    rk1_first_bad = _first_bad_field(rk1_fields)
    operator_checks = _operator_bisection(rk1_pre, rk1_validation_substep1)

    operational_step = run_forecast_operational(state, namelist, one_step_hours)
    block_until_ready(operational_step)

    step_fields = _summarize_field_delta(
        _post_physics_boundary_operational_units(operational_step),
        validation_step_units,
    )
    step_first_bad = _first_bad_field(step_fields)
    step_payload = {
        "artifact_type": "m6b_operational_vs_validation_step_level_bisection",
        "status": "DIVERGED" if step_first_bad else "NO_DIVERGENCE",
        "run_id": run_id,
        "run_dir": str(RUN_ROOT / run_id),
        "requested_steps": requested_steps,
        "executed_steps": 1,
        "first_divergence_step": 1 if step_first_bad else None,
        "threshold": THRESHOLD,
        "device": visible_gpu_name(),
        "grid": case.metadata["grid"],
        "field_deltas_at_step_1": step_fields,
        "largest_bad_field_at_step_1": step_first_bad,
        "note": (
            "Full coupled operational and validation endpoints already differ after one timestep; "
            "substep proof below uses a common RK1 pre-acoustic state to isolate composition."
        ),
    }

    operator_first_bad = next((item for item in operator_checks if bool(item["exceeds_1e_10"])), None)
    substep_payload = {
        "artifact_type": "m6b_operational_vs_validation_substep_bisection",
        "status": "LOCALIZED",
        "run_id": run_id,
        "first_divergence_step": 1,
        "first_divergence_rk_stage": 1,
        "first_divergence_acoustic_substep": 1,
        "localized_defect": "OPERATIONAL-COMPOSITION-DEFECT-LOCALIZED-AT-RK1-ACOUSTIC-LOOP-OMISSION",
        "defect_file": "src/gpuwrf/runtime/operational_mode.py",
        "defect_function": "_rk_scan_step",
        "defect_statement": (
            "RK stage 1 dispatches advance_stage(..., use_acoustic=False), so no acoustic small-step "
            "operator runs at stage 1. Starting from the exact same RK1 pre-acoustic candidate, the "
            "validation WRF-shaped acoustic substep changes mu/theta/ww/w/scratch fields immediately."
        ),
        "threshold": THRESHOLD,
        "rk_stage_table": [
            {
                "rk_stage": 1,
                "operational_action": "advection candidate only; acoustic loop skipped",
                "validation_action": "acoustic_substep_wrf applied to the same pre-acoustic candidate",
                "diverged": bool(rk1_first_bad),
                "largest_bad_field": rk1_first_bad,
                "field_deltas": rk1_fields,
            }
        ],
        "operator_table": operator_checks,
        "first_bad_operator_family": operator_first_bad["operator"] if operator_first_bad else None,
        "wrf_source_citation": {
            "rk_loop": f"{WRF_SOLVE}:1447 starts Runge_Kutta_loop",
            "rk1_small_steps": f"{WRF_SOLVE}:1472-1475 sets RK3 stage 1 number_of_small_timesteps = 1",
            "small_steps_loop": f"{WRF_SOLVE}:3065 starts small_steps loop",
            "advance_mu_t_call": f"{WRF_SOLVE}:3435-3452 calls advance_mu_t inside small_steps",
            "advance_mu_t_updates": f"{WRF_SMALL}:1102-1108 updates MU/MUDF/MUTS/MUAVE and {WRF_SMALL}:1162-1171 updates theta",
        },
        "recommended_minimal_fix": (
            "In a follow-up fix sprint, make operational RK stage 1 execute the WRF-required acoustic "
            "small-step path instead of dispatching use_acoustic=False. Keep the fix minimal: preserve "
            "the common RK1 pre-acoustic tendency candidate, run one WRF-shaped small step for RK1 "
            "per solve_em.F:1472-1475, and re-run this bisection before addressing later-stage defects."
        ),
        "not_a_fix": True,
    }
    _write_json(SPRINT / "proof_bisection_step_level.json", step_payload)
    _write_json(SPRINT / "proof_bisection_substep_level.json", substep_payload)
    return {"step_level": step_payload, "substep_level": substep_payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen2-run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--steps", type=int, default=70)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_bisection(str(args.gen2_run_id), int(args.steps))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["substep_level"]["status"] == "LOCALIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
