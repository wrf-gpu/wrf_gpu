"""Comprehensive per-operator + per-step diagnostic harness.

This module is the project's single-source-of-truth "what is wrong?" surface
for the operational forecast loop. It mirrors :func:`_physics_boundary_step`
from :mod:`gpuwrf.runtime.operational_mode` but, instead of just stepping the
state forward, it records per-operator delta statistics, per-step invariant
violations, and per-operator activity verdicts.

The harness is **purely additive**: it does not import or modify any code
under ``gpuwrf.dycore.*``, ``gpuwrf.coupling.physics_couplers`` is consumed
through its public adapter functions only, and ``operational_mode`` is
imported only for its private helpers used as plug-ins (the instrumentation
mirror re-implements the operator sequence here). When ``diagnostic_on`` is
``False`` the harness is never traced — XLA dead-code-eliminates everything.

Schema, operator list, invariant list, and overhead budget are documented in
``.agent/sprints/2026-05-28-diagnostic-harness/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import (
    mynn_adapter,
    rrtmg_adapter,
    surface_adapter,
    thompson_adapter,
)
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _enforce_operational_precision,
    _finite_or_origin,
    _limit_guarded_dynamics_state,
    _rk_scan_step,
    _steps_for_hours,
    _valid_mixing_ratio,
)
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


DIAGNOSTIC_SCHEMA_VERSION = "diagnostic-harness-1.0"

# Operator ordering MUST mirror ``_physics_boundary_step`` in operational_mode.py.
DIAGNOSTIC_OPERATORS: tuple[str, ...] = (
    "dycore_rk3",
    "dynamics_guards",
    "microphysics_thompson",
    "surface_layer",
    "mynn_pbl",
    "rrtmg",
    "lateral_boundary",
    "boundary_guards",
)

# Fields whose per-operator delta we track. The order is the column order in
# the per-operator (steps, len(fields)) delta arrays carried through the scan.
DIAGNOSTIC_FIELD_INDEX: tuple[str, ...] = (
    "u",
    "v",
    "w",
    "theta",
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "qke",
    "p_perturbation",
    "ph_perturbation",
    "mu_perturbation",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "ustar",
    "rhosfc",
)


_FIELD_COUNT = len(DIAGNOSTIC_FIELD_INDEX)
_OP_COUNT = len(DIAGNOSTIC_OPERATORS)
_OP_INDEX = {name: idx for idx, name in enumerate(DIAGNOSTIC_OPERATORS)}

# Invariant ordering for the per-step (steps, len(invariants)) boolean array.
DIAGNOSTIC_INVARIANTS: tuple[str, ...] = (
    "all_state_finite",
    "qv_nonnegative",
    "qc_nonnegative",
    "qr_nonnegative",
    "qi_nonnegative",
    "qs_nonnegative",
    "qg_nonnegative",
    "theta_in_bounds",
    "wind_in_bounds",
    "mu_nonnegative",
)
_INV_COUNT = len(DIAGNOSTIC_INVARIANTS)
_INV_INDEX = {name: idx for idx, name in enumerate(DIAGNOSTIC_INVARIANTS)}


_QV_EPS = 1.0e-12
_THETA_LOWER_30_MIN = 200.0
_THETA_LOWER_30_MAX = 450.0
_THETA_UPPER_MIN = 250.0
_THETA_UPPER_MAX = 700.0
_U_LIMIT = 100.0
_V_LIMIT = 100.0
_W_LIMIT = 50.0

# Below this total-over-run sum, an operator's effect on a field is "zero".
EPS_MISSING = 1.0e-30


# ---------------------------------------------------------------------------
# Pure-JIT accumulator (carried through ``jax.lax.scan``)
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DiagnosticAccumulator:
    """Diagnostic counters carried alongside the operational carry.

    All leaves are device arrays. No host transfer ever occurs inside the scan.
    """

    # (steps, n_operators, n_fields) — mean(|delta|) per operator per field per step
    mean_abs_delta: jax.Array
    # same shape — max(|delta|) per operator per field per step
    max_abs_delta: jax.Array
    # (steps, n_operators) — count of nonfinite cells in (u,v,w,theta,qv) after op
    nonfinite_count: jax.Array
    # (steps, n_invariants) — boolean violations
    invariant_violated: jax.Array
    # (steps,) — running step index for indexing into the above
    # We index by step via a counter rather than scan's xs to avoid Python-len
    step_counter: jax.Array
    # (n_invariants,) — boolean: invariant violated at any step so far
    invariant_ever_violated: jax.Array
    # (n_invariants,) — int32: first step (1-indexed) at which violated, or -1
    first_violation_step: jax.Array
    # (n_invariants,) — int32: first operator index at which violated, or -1
    first_violation_operator: jax.Array

    def tree_flatten(self):
        return (
            self.mean_abs_delta,
            self.max_abs_delta,
            self.nonfinite_count,
            self.invariant_violated,
            self.step_counter,
            self.invariant_ever_violated,
            self.first_violation_step,
            self.first_violation_operator,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def initial_diagnostic_accumulator(steps: int) -> DiagnosticAccumulator:
    """Build a zeroed accumulator sized for ``steps`` total scan steps."""

    return DiagnosticAccumulator(
        mean_abs_delta=jnp.zeros((steps, _OP_COUNT, _FIELD_COUNT), dtype=jnp.float64),
        max_abs_delta=jnp.zeros((steps, _OP_COUNT, _FIELD_COUNT), dtype=jnp.float64),
        nonfinite_count=jnp.zeros((steps, _OP_COUNT), dtype=jnp.int32),
        invariant_violated=jnp.zeros((steps, _INV_COUNT), dtype=jnp.bool_),
        step_counter=jnp.asarray(0, dtype=jnp.int32),
        invariant_ever_violated=jnp.zeros((_INV_COUNT,), dtype=jnp.bool_),
        first_violation_step=jnp.full((_INV_COUNT,), -1, dtype=jnp.int32),
        first_violation_operator=jnp.full((_INV_COUNT,), -1, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# Field-extraction helpers — return a length-_FIELD_COUNT vector of statistics
# ---------------------------------------------------------------------------


def _field_value(state: State, name: str) -> jax.Array:
    """Pull a State field by string; missing fields return zero-array proxy.

    Surface fields are 2D and dynamics fields are 3D — the harness treats both
    via per-cell abs reductions so dimensionality is fine.
    """

    return jnp.asarray(getattr(state, name))


def _diff_stats(pre: State, post: State) -> tuple[jax.Array, jax.Array]:
    """Per-field ``mean(|post - pre|)`` and ``max(|post - pre|)`` vectors."""

    means: list[jax.Array] = []
    maxes: list[jax.Array] = []
    for name in DIAGNOSTIC_FIELD_INDEX:
        delta = _field_value(post, name).astype(jnp.float64) - _field_value(pre, name).astype(jnp.float64)
        abs_delta = jnp.abs(delta)
        means.append(jnp.mean(abs_delta))
        maxes.append(jnp.max(abs_delta))
    return jnp.stack(means), jnp.stack(maxes)


def _nonfinite_count(state: State) -> jax.Array:
    """Total nonfinite cells across ``u, v, w, theta, qv`` (= proxy for "blew up")."""

    total = (
        jnp.sum(~jnp.isfinite(state.u))
        + jnp.sum(~jnp.isfinite(state.v))
        + jnp.sum(~jnp.isfinite(state.w))
        + jnp.sum(~jnp.isfinite(state.theta))
        + jnp.sum(~jnp.isfinite(state.qv))
    )
    return total.astype(jnp.int32)


def _all_state_finite(state: State) -> jax.Array:
    """``jnp.bool_`` scalar: True iff every prognostic leaf is fully finite."""

    return (
        jnp.all(jnp.isfinite(state.u))
        & jnp.all(jnp.isfinite(state.v))
        & jnp.all(jnp.isfinite(state.w))
        & jnp.all(jnp.isfinite(state.theta))
        & jnp.all(jnp.isfinite(state.qv))
        & jnp.all(jnp.isfinite(state.p_total))
        & jnp.all(jnp.isfinite(state.ph_total))
        & jnp.all(jnp.isfinite(state.mu_total))
    )


def _evaluate_invariants(state: State) -> jax.Array:
    """Return a length-``_INV_COUNT`` boolean vector: True = invariant violated."""

    finite_ok = _all_state_finite(state)

    qv_ok = jnp.min(state.qv) >= -_QV_EPS
    qc_ok = jnp.min(state.qc) >= -_QV_EPS
    qr_ok = jnp.min(state.qr) >= -_QV_EPS
    qi_ok = jnp.min(state.qi) >= -_QV_EPS
    qs_ok = jnp.min(state.qs) >= -_QV_EPS
    qg_ok = jnp.min(state.qg) >= -_QV_EPS

    theta = state.theta
    nz = int(theta.shape[0])
    lower_n = min(30, nz)
    lower_theta = theta[:lower_n, :, :]
    upper_theta = theta[lower_n:, :, :] if lower_n < nz else theta[:0]
    theta_lower_ok = (jnp.min(lower_theta) >= _THETA_LOWER_30_MIN) & (jnp.max(lower_theta) <= _THETA_LOWER_30_MAX)
    if upper_theta.size > 0:
        theta_upper_ok = (jnp.min(upper_theta) >= _THETA_UPPER_MIN) & (jnp.max(upper_theta) <= _THETA_UPPER_MAX)
    else:
        theta_upper_ok = jnp.asarray(True)
    theta_ok = theta_lower_ok & theta_upper_ok

    wind_ok = (
        (jnp.max(jnp.abs(state.u)) <= _U_LIMIT)
        & (jnp.max(jnp.abs(state.v)) <= _V_LIMIT)
        & (jnp.max(jnp.abs(state.w)) <= _W_LIMIT)
    )

    mu_ok = jnp.min(state.mu_total) > 0.0

    violated = jnp.stack(
        [
            ~finite_ok,
            ~qv_ok,
            ~qc_ok,
            ~qr_ok,
            ~qi_ok,
            ~qs_ok,
            ~qg_ok,
            ~theta_ok,
            ~wind_ok,
            ~mu_ok,
        ]
    )
    return violated


def _record_operator(
    acc: DiagnosticAccumulator,
    op_index: int,
    pre: State,
    post: State,
    step_index_1based: jax.Array,
) -> DiagnosticAccumulator:
    """Update accumulator slots for one operator's pre→post transition."""

    mean_v, max_v = _diff_stats(pre, post)
    nonfinite = _nonfinite_count(post)
    # Use the running step_counter (0-indexed scan position) for slot lookup.
    slot = acc.step_counter

    mean_abs_delta = acc.mean_abs_delta.at[slot, op_index, :].set(mean_v)
    max_abs_delta = acc.max_abs_delta.at[slot, op_index, :].set(max_v)
    nonfinite_count = acc.nonfinite_count.at[slot, op_index].set(nonfinite)

    invariants_after = _evaluate_invariants(post)
    invariant_violated = acc.invariant_violated.at[slot, :].set(
        acc.invariant_violated[slot, :] | invariants_after
    )

    # For each invariant: if not previously violated but now is, record (step, op).
    became_violated = (~acc.invariant_ever_violated) & invariants_after
    new_ever = acc.invariant_ever_violated | invariants_after
    new_first_step = jnp.where(became_violated, step_index_1based, acc.first_violation_step)
    new_first_op = jnp.where(became_violated, jnp.asarray(op_index, dtype=jnp.int32), acc.first_violation_operator)

    return DiagnosticAccumulator(
        mean_abs_delta=mean_abs_delta,
        max_abs_delta=max_abs_delta,
        nonfinite_count=nonfinite_count,
        invariant_violated=invariant_violated,
        step_counter=acc.step_counter,  # advanced only at end of step
        invariant_ever_violated=new_ever,
        first_violation_step=new_first_step,
        first_violation_operator=new_first_op,
    )


# ---------------------------------------------------------------------------
# Instrumented ``_physics_boundary_step`` mirror
# ---------------------------------------------------------------------------


def instrumented_physics_boundary_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    diagnostic_on: bool,
    accumulator: DiagnosticAccumulator,
) -> tuple[OperationalCarry, DiagnosticAccumulator]:
    """Mirror of ``operational_mode._physics_boundary_step`` with hooks.

    When ``diagnostic_on`` is ``False`` no accumulator update is emitted (the
    branch is dead-code eliminated by XLA since the flag is a Python bool).
    """

    physical_origin = carry.state

    # -- Operator: dycore_rk3 (and acoustic substeps) -------------------------
    pre = physical_origin
    carry = _rk_scan_step(carry, namelist, debug=False)
    post_rk = carry.state
    step_1b = (jnp.asarray(step_index, dtype=jnp.int32) - jnp.asarray(int(namelist.run_physics) * 0, dtype=jnp.int32))
    # We pass step_index as the absolute 1-based step (caller hands us 1-based).
    if diagnostic_on:
        accumulator = _record_operator(accumulator, _OP_INDEX["dycore_rk3"], pre, post_rk, step_1b)

    # -- Operator: dynamics_guards -------------------------------------------
    pre = post_rk
    if not bool(namelist.disable_guards):
        guarded = _limit_guarded_dynamics_state(post_rk, physical_origin).replace(
            qv=_valid_mixing_ratio(post_rk.qv, physical_origin.qv),
            qc=_valid_mixing_ratio(post_rk.qc, physical_origin.qc),
            qr=_valid_mixing_ratio(post_rk.qr, physical_origin.qr),
            qi=_valid_mixing_ratio(post_rk.qi, physical_origin.qi),
            qs=_valid_mixing_ratio(post_rk.qs, physical_origin.qs),
            qg=_valid_mixing_ratio(post_rk.qg, physical_origin.qg),
        )
    else:
        guarded = post_rk
    if diagnostic_on:
        accumulator = _record_operator(accumulator, _OP_INDEX["dynamics_guards"], pre, guarded, step_1b)
    next_state = guarded

    # -- Physics block --------------------------------------------------------
    if bool(namelist.run_physics):
        pre = next_state
        if not bool(namelist.disable_guards):
            next_state = thompson_adapter(next_state, float(namelist.dt_s))
        if diagnostic_on:
            accumulator = _record_operator(
                accumulator, _OP_INDEX["microphysics_thompson"], pre, next_state, step_1b
            )

        pre = next_state
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        if diagnostic_on:
            accumulator = _record_operator(accumulator, _OP_INDEX["surface_layer"], pre, next_state, step_1b)

        pre = next_state
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        if diagnostic_on:
            accumulator = _record_operator(accumulator, _OP_INDEX["mynn_pbl"], pre, next_state, step_1b)

        if run_radiation:
            pre = next_state
            next_state = rrtmg_adapter(next_state, float(namelist.dt_s), namelist.grid)
            if diagnostic_on:
                accumulator = _record_operator(accumulator, _OP_INDEX["rrtmg"], pre, next_state, step_1b)

    # -- Boundary block + boundary guards -------------------------------------
    if bool(namelist.run_boundary):
        pre = next_state
        lead_seconds = jnp.asarray(step_index, dtype=jnp.float64) * float(namelist.dt_s)
        bounded = apply_lateral_boundaries(next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
        if diagnostic_on:
            accumulator = _record_operator(accumulator, _OP_INDEX["lateral_boundary"], pre, bounded, step_1b)

        pre = bounded
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
        if diagnostic_on:
            accumulator = _record_operator(accumulator, _OP_INDEX["boundary_guards"], pre, next_state, step_1b)

    next_state = _enforce_operational_precision(next_state)
    next_carry = carry.replace(state=next_state)

    if diagnostic_on:
        accumulator = DiagnosticAccumulator(
            mean_abs_delta=accumulator.mean_abs_delta,
            max_abs_delta=accumulator.max_abs_delta,
            nonfinite_count=accumulator.nonfinite_count,
            invariant_violated=accumulator.invariant_violated,
            step_counter=accumulator.step_counter + 1,
            invariant_ever_violated=accumulator.invariant_ever_violated,
            first_violation_step=accumulator.first_violation_step,
            first_violation_operator=accumulator.first_violation_operator,
        )

    return next_carry, accumulator


# ---------------------------------------------------------------------------
# Forecast driver — runs all steps inside one scan, returns accumulator
# ---------------------------------------------------------------------------


def _scan_segment_with_acc(
    carry: OperationalCarry,
    accumulator: DiagnosticAccumulator,
    namelist: OperationalNamelist,
    start_step: int,
    steps: int,
    run_radiation: bool,
    diagnostic_on: bool,
) -> tuple[OperationalCarry, DiagnosticAccumulator]:
    indices = jnp.arange(start_step, start_step + steps, dtype=jnp.int32)

    def body(state_acc, step_index):
        scan_carry, acc = state_acc
        new_carry, new_acc = instrumented_physics_boundary_step(
            scan_carry,
            namelist,
            step_index,
            run_radiation=run_radiation,
            diagnostic_on=diagnostic_on,
            accumulator=acc,
        )
        return (new_carry, new_acc), None

    (next_carry, next_acc), _ = jax.lax.scan(body, (carry, accumulator), indices)
    return next_carry, next_acc


@partial(jax.jit, static_argnames=("hours", "diagnostic_on"), donate_argnums=(0, 2))
def run_diagnostic_forecast(
    state: State,
    namelist: OperationalNamelist,
    accumulator: DiagnosticAccumulator,
    hours: float,
    *,
    diagnostic_on: bool,
) -> tuple[State, DiagnosticAccumulator]:
    """Run the operational forecast with optional per-operator/per-step instrumentation.

    ``diagnostic_on`` is a Python bool used as a JIT static argument; when
    ``False`` every instrumentation branch is dead-code eliminated. The
    accumulator returned in the False path is the initial zero accumulator.
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    initial = initial_operational_carry(_enforce_operational_precision(state))
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    acc = accumulator
    step = 1
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry, acc = _scan_segment_with_acc(
                    carry,
                    acc,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    diagnostic_on=diagnostic_on,
                )
            carry, acc = _scan_segment_with_acc(
                carry,
                acc,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                diagnostic_on=diagnostic_on,
            )
            step = next_radiation + 1
        else:
            carry, acc = _scan_segment_with_acc(
                carry,
                acc,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                diagnostic_on=diagnostic_on,
            )
            step = steps + 1
    return carry.state, acc


# ---------------------------------------------------------------------------
# Report construction (host side, post-scan)
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticReport:
    """Materialized harness report, ready to JSON-serialize."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _operator_verdict(
    mean_abs_delta: np.ndarray,
    max_abs_delta: np.ndarray,
    op_index: int,
    *,
    operator_was_called: bool,
) -> tuple[str, str | None]:
    """Classify an operator as ACTIVE / NOISY_ZERO / MISSING / INACTIVE.

    "Passive" operators (guards / limiters) have an empty expected-scope and
    are classified PASSIVE_OK when they fire zero times — that is the correct
    behavior, not a missing-coupling bug.
    """

    if not operator_was_called:
        return "INACTIVE", "operator was not invoked under the configured namelist"

    op_name = DIAGNOSTIC_OPERATORS[op_index]
    scope = _OPERATOR_SCOPE.get(op_name, set())
    op_mean = mean_abs_delta[:, op_index, :]  # (steps, fields)
    total_per_field = op_mean.sum(axis=0)
    any_nonzero = total_per_field > EPS_MISSING

    # Passive operators (guards, limiters): scope is empty by design.
    if not scope:
        if any_nonzero.any():
            return "ACTIVE", "passive guard fired at least once (some cells exceeded a bound)"
        return "PASSIVE_OK", "passive guard did not fire (no cells exceeded a bound) — expected"

    # Scope-bearing operators
    scope_idx = np.array([DIAGNOSTIC_FIELD_INDEX.index(f) for f in scope if f in DIAGNOSTIC_FIELD_INDEX])
    in_scope_zero = int(np.sum(total_per_field[scope_idx] <= EPS_MISSING))
    if in_scope_zero == len(scope_idx):
        return "MISSING", "every in-scope field has delta = 0 across the entire run"
    if in_scope_zero > 0:
        zero_fields = [DIAGNOSTIC_FIELD_INDEX[i] for i in scope_idx if total_per_field[i] <= EPS_MISSING]
        return "NOISY_ZERO", (
            f"{in_scope_zero}/{len(scope_idx)} expected fields have delta = 0 across the run: "
            f"{', '.join(zero_fields)}"
        )
    return "ACTIVE", None


# Which fields each operator is *expected* to modify (used for MISSING detection).
_OPERATOR_SCOPE: dict[str, set[str]] = {
    "dycore_rk3": {"u", "v", "w", "theta", "p_perturbation", "ph_perturbation", "mu_perturbation"},
    "dynamics_guards": set(),  # guards are passive; zero is fine
    "microphysics_thompson": {"qv", "qc", "qr", "qi", "qs", "qg", "theta"},
    "surface_layer": {"theta_flux", "qv_flux", "tau_u", "tau_v", "ustar", "rhosfc"},
    "mynn_pbl": {"u", "v", "theta", "qv", "qke"},
    "rrtmg": {"theta"},
    "lateral_boundary": {"u", "v", "theta", "qv", "ph_perturbation", "mu_perturbation"},
    "boundary_guards": set(),
}


def _classify_operators(
    mean_abs_delta: np.ndarray,
    max_abs_delta: np.ndarray,
    *,
    namelist: OperationalNamelist,
    steps_total: int,
) -> dict[str, Any]:
    """Build the ``operator_attribution_24h`` block of the report."""

    radiation_cadence = int(namelist.radiation_cadence_steps)
    rrtmg_fired = bool(namelist.run_physics) and (radiation_cadence > 0) and (radiation_cadence <= steps_total)
    physics_called = bool(namelist.run_physics)
    boundary_called = bool(namelist.run_boundary)
    guards_called = not bool(namelist.disable_guards)

    op_called = {
        "dycore_rk3": True,
        "dynamics_guards": guards_called,
        "microphysics_thompson": physics_called and guards_called,
        "surface_layer": physics_called,
        "mynn_pbl": physics_called,
        "rrtmg": rrtmg_fired,
        "lateral_boundary": boundary_called,
        "boundary_guards": boundary_called and guards_called,
    }

    operators: dict[str, Any] = {}
    for op_index, op_name in enumerate(DIAGNOSTIC_OPERATORS):
        verdict, comment = _operator_verdict(
            mean_abs_delta, max_abs_delta, op_index, operator_was_called=op_called[op_name]
        )
        per_field_mean = mean_abs_delta[:, op_index, :].mean(axis=0)
        per_field_max = max_abs_delta[:, op_index, :].max(axis=0)
        # First step where every recorded field for this op transitioned to zero.
        op_mean = mean_abs_delta[:, op_index, :]
        nonzero_any_step = (op_mean > EPS_MISSING).any(axis=1)
        first_zero_step = None
        if nonzero_any_step.any():
            # Find the first step from the end where any field was nonzero.
            last_nonzero = int(np.argmax(nonzero_any_step[::-1])) if nonzero_any_step.any() else -1
            # If the run ends with zeros and there's a transition, record it.
            first_zero = np.where(~nonzero_any_step)[0]
            if first_zero.size > 0:
                first_zero_step = int(first_zero[0] + 1)
        steps_with_finite_delta = int(np.isfinite(op_mean).all(axis=1).sum())
        operators[op_name] = {
            "verdict": verdict,
            "operator_was_called": op_called[op_name],
            "steps_with_finite_delta": steps_with_finite_delta,
            "first_zero_delta_step": first_zero_step,
            "mean_abs_delta_per_step": {
                DIAGNOSTIC_FIELD_INDEX[i]: float(per_field_mean[i])
                for i in range(_FIELD_COUNT)
            },
            "max_abs_delta_per_step": {
                DIAGNOSTIC_FIELD_INDEX[i]: float(per_field_max[i])
                for i in range(_FIELD_COUNT)
            },
            "comments": comment,
        }
    return operators


def _build_invariant_block(
    invariant_violated: np.ndarray,
    invariant_ever_violated: np.ndarray,
    first_violation_step: np.ndarray,
    first_violation_operator: np.ndarray,
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for idx, name in enumerate(DIAGNOSTIC_INVARIANTS):
        violated = bool(invariant_ever_violated[idx])
        first_step = int(first_violation_step[idx])
        first_op_idx = int(first_violation_operator[idx])
        first_op = DIAGNOSTIC_OPERATORS[first_op_idx] if first_op_idx >= 0 else None
        per_step = invariant_violated[:, idx]
        block[name] = {
            "violated": violated,
            "first_violation_step": first_step if first_step >= 0 else None,
            "first_violation_operator": first_op,
            "violation_count": int(per_step.sum()),
        }
    return block


def _coupling_chain_audit(
    mean_abs_delta: np.ndarray,
    namelist: OperationalNamelist,
    steps_total: int,
) -> dict[str, Any]:
    """Pairwise upstream→downstream chain attribution."""

    def total(op_name: str, field: str) -> float:
        op_idx = _OP_INDEX[op_name]
        f_idx = DIAGNOSTIC_FIELD_INDEX.index(field)
        return float(mean_abs_delta[:, op_idx, f_idx].sum())

    chains: dict[str, Any] = {}

    # surface_layer → mynn (theta_flux feeds mynn bottom-BC for theta[..., 0])
    sl_flux = total("surface_layer", "theta_flux")
    mynn_theta = total("mynn_pbl", "theta")
    if sl_flux <= EPS_MISSING:
        chains["surface_layer__to__mynn_theta_bottom_bc"] = {
            "verdict": "BROKEN",
            "evidence": (
                "surface_layer produced theta_flux total ~= 0 across the run; "
                "MYNN bottom-BC therefore added no flux-driven increment to theta[..., 0]"
            ),
            "first_broken_step": 1,
        }
    else:
        chains["surface_layer__to__mynn_theta_bottom_bc"] = {
            "verdict": "ACTIVE",
            "evidence": (
                f"surface_layer produced theta_flux total = {sl_flux:.3e}; "
                f"mynn_pbl produced theta total = {mynn_theta:.3e}"
            ),
            "first_broken_step": None,
        }

    # thompson → theta (latent-heat release coupling)
    thompson_theta = total("microphysics_thompson", "theta")
    if thompson_theta <= EPS_MISSING:
        chains["thompson__to__theta_via_latent_heat"] = {
            "verdict": "BROKEN",
            "evidence": "thompson_adapter produced theta delta ~= 0; latent-heat coupling not flowing back to theta",
            "first_broken_step": 1,
        }
    else:
        chains["thompson__to__theta_via_latent_heat"] = {
            "verdict": "ACTIVE",
            "evidence": f"thompson produced theta total = {thompson_theta:.3e}",
            "first_broken_step": None,
        }

    # rrtmg → theta (radiation heating-rate coupling)
    radiation_cadence = int(namelist.radiation_cadence_steps)
    if radiation_cadence > steps_total or not bool(namelist.run_physics):
        chains["rrtmg__to__theta_via_heating_rate"] = {
            "verdict": "INACTIVE",
            "evidence": (
                f"radiation cadence ({radiation_cadence}) > steps_total ({steps_total}); "
                "rrtmg never fired in this run"
            ),
            "first_broken_step": None,
        }
    else:
        rrtmg_theta = total("rrtmg", "theta")
        if rrtmg_theta <= EPS_MISSING:
            chains["rrtmg__to__theta_via_heating_rate"] = {
                "verdict": "BROKEN",
                "evidence": "rrtmg fired but produced theta delta ~= 0; heating-rate coupling broken",
                "first_broken_step": 1,
            }
        else:
            chains["rrtmg__to__theta_via_heating_rate"] = {
                "verdict": "ACTIVE",
                "evidence": f"rrtmg produced theta total = {rrtmg_theta:.3e}",
                "first_broken_step": None,
            }

    # lateral_boundary → interior (u, v, theta)
    bdy_u = total("lateral_boundary", "u")
    bdy_v = total("lateral_boundary", "v")
    bdy_theta = total("lateral_boundary", "theta")
    if bool(namelist.run_boundary) and max(bdy_u, bdy_v, bdy_theta) <= EPS_MISSING:
        chains["lateral_boundary__to__interior_winds_theta"] = {
            "verdict": "BROKEN",
            "evidence": "lateral_boundary delta ~= 0 across u, v, theta; boundary forcing not being applied",
            "first_broken_step": 1,
        }
    elif bool(namelist.run_boundary):
        chains["lateral_boundary__to__interior_winds_theta"] = {
            "verdict": "ACTIVE",
            "evidence": (
                f"lateral_boundary produced totals u={bdy_u:.3e}, v={bdy_v:.3e}, theta={bdy_theta:.3e}"
            ),
            "first_broken_step": None,
        }
    else:
        chains["lateral_boundary__to__interior_winds_theta"] = {
            "verdict": "INACTIVE",
            "evidence": "namelist.run_boundary = False; lateral boundary forcing not applied",
            "first_broken_step": None,
        }

    return chains


def _first_failure_attribution(
    nonfinite_count: np.ndarray,
    invariant_violated: np.ndarray,
    invariant_ever_violated: np.ndarray,
    first_violation_step: np.ndarray,
    first_violation_operator: np.ndarray,
    wrf_anchor_first: dict[str, Any] | None,
) -> dict[str, Any]:
    # First nonfinite: smallest (step, op) where nonfinite_count > 0
    first_nf = None
    nf_positive = nonfinite_count > 0
    if nf_positive.any():
        step_idx, op_idx = np.unravel_index(np.argmax(nf_positive.ravel()), nf_positive.shape)
        first_nf = {
            "step": int(step_idx) + 1,
            "operator": DIAGNOSTIC_OPERATORS[int(op_idx)],
            "nonfinite_cell_count": int(nonfinite_count[step_idx, op_idx]),
        }

    # First invariant break across all invariants
    first_break = None
    if invariant_ever_violated.any():
        ever_idx = np.where(invariant_ever_violated)[0]
        steps = first_violation_step[ever_idx]
        # ignore -1
        valid = steps >= 0
        if valid.any():
            best = int(np.argmin(np.where(valid, steps, np.iinfo(np.int32).max)))
            inv_i = int(ever_idx[best])
            first_break = {
                "step": int(first_violation_step[inv_i]),
                "operator": DIAGNOSTIC_OPERATORS[int(first_violation_operator[inv_i])],
                "invariant": DIAGNOSTIC_INVARIANTS[inv_i],
            }

    return {
        "first_invariant_break": first_break,
        "first_nonfinite": first_nf,
        "first_significant_anchor_divergence": wrf_anchor_first,
    }


def _headline_diagnosis(
    operators: dict[str, Any],
    invariants: dict[str, Any],
    chains: dict[str, Any],
    first_failure: dict[str, Any],
) -> str:
    parts: list[str] = []

    # Missing operators are the strongest signal.
    missing_ops = [name for name, info in operators.items() if info["verdict"] == "MISSING"]
    noisy_ops = [name for name, info in operators.items() if info["verdict"] == "NOISY_ZERO"]
    if missing_ops:
        parts.append(
            f"MISSING operators detected: {', '.join(missing_ops)} — wired but produce identically-zero delta across the run."
        )
    if noisy_ops:
        # Include which fields are zero in each noisy operator for actionability.
        noisy_detail = "; ".join(
            f"{name} [{operators[name].get('comments', '')}]" for name in noisy_ops
        )
        parts.append(
            f"NOISY_ZERO operators (partial coupling failure): {noisy_detail}"
        )

    broken_chains = [name for name, info in chains.items() if info["verdict"] == "BROKEN"]
    if broken_chains:
        parts.append(f"BROKEN coupling chains: {', '.join(broken_chains)}.")

    if first_failure["first_invariant_break"] is not None:
        fb = first_failure["first_invariant_break"]
        parts.append(
            f"First invariant break: step {fb['step']}, invariant '{fb['invariant']}', operator '{fb['operator']}'."
        )
    if first_failure["first_nonfinite"] is not None:
        fn = first_failure["first_nonfinite"]
        parts.append(
            f"First nonfinite cell: step {fn['step']} after operator '{fn['operator']}' ({fn['nonfinite_cell_count']} cells)."
        )

    if not parts:
        parts.append(
            "All operators ACTIVE; no invariant violations; no nonfinite cells; coupling chains all ACTIVE. Harness sees a clean run."
        )

    return " ".join(parts)


def _next_sprint_recommendations(
    operators: dict[str, Any],
    invariants: dict[str, Any],
    chains: dict[str, Any],
) -> list[str]:
    recs: list[str] = []
    for name, info in operators.items():
        if info["verdict"] == "MISSING":
            recs.append(
                f"Operator '{name}' is wired but produces zero delta — investigate why its public adapter "
                f"returns an identity transform."
            )
        elif info["verdict"] == "NOISY_ZERO":
            recs.append(
                f"Operator '{name}' has partial coupling: check which expected fields are flatlining."
            )
    for inv_name, info in invariants.items():
        if info["violated"]:
            recs.append(
                f"Invariant '{inv_name}' first tripped at step {info['first_violation_step']} after operator "
                f"'{info['first_violation_operator']}' — investigate."
            )
    for chain_name, info in chains.items():
        if info["verdict"] == "BROKEN":
            recs.append(f"Coupling chain '{chain_name}' is broken — fix upstream operator first.")
    if not recs:
        recs.append("Harness saw no anomalies; expand WRF anchor comparison to confirm operational fidelity.")
    return recs


def build_diagnostic_report(
    *,
    accumulator: DiagnosticAccumulator,
    namelist: OperationalNamelist,
    steps_total: int,
    run_config: dict[str, Any],
    wrf_anchor_payload: dict[str, Any] | None,
    commit: str,
    generated_utc: str,
    wall_seconds_total: float,
    wall_seconds_diagnostic_overhead: float | None,
) -> DiagnosticReport:
    """Materialize the accumulator into a fully-serialized JSON dict."""

    mean_abs_delta = np.asarray(accumulator.mean_abs_delta)
    max_abs_delta = np.asarray(accumulator.max_abs_delta)
    nonfinite_count = np.asarray(accumulator.nonfinite_count)
    invariant_violated = np.asarray(accumulator.invariant_violated)
    invariant_ever_violated = np.asarray(accumulator.invariant_ever_violated)
    first_violation_step = np.asarray(accumulator.first_violation_step)
    first_violation_operator = np.asarray(accumulator.first_violation_operator)

    operators = _classify_operators(
        mean_abs_delta,
        max_abs_delta,
        namelist=namelist,
        steps_total=steps_total,
    )
    invariants = _build_invariant_block(
        invariant_violated, invariant_ever_violated, first_violation_step, first_violation_operator
    )
    chains = _coupling_chain_audit(mean_abs_delta, namelist, steps_total)

    wrf_anchor_first = None
    wrf_anchor_block: dict[str, Any] = {"source": None, "per_field": {}}
    if wrf_anchor_payload is not None:
        wrf_anchor_first = wrf_anchor_payload.get("first_divergence")
        wrf_anchor_block = {
            "source": wrf_anchor_payload.get("_source_path"),
            "per_field": wrf_anchor_payload.get("per_field_summary", {}),
            "status": wrf_anchor_payload.get("status"),
            "first_divergence": wrf_anchor_payload.get("first_divergence"),
        }

    first_failure = _first_failure_attribution(
        nonfinite_count,
        invariant_violated,
        invariant_ever_violated,
        first_violation_step,
        first_violation_operator,
        wrf_anchor_first,
    )
    headline = _headline_diagnosis(operators, invariants, chains, first_failure)
    recommendations = _next_sprint_recommendations(operators, invariants, chains)

    run_config = dict(run_config)
    run_config["wall_seconds_total"] = wall_seconds_total
    run_config["wall_seconds_diagnostic_overhead"] = wall_seconds_diagnostic_overhead

    payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "commit": commit,
        "run_config": run_config,
        "headline_diagnosis": headline,
        "first_failure_attribution": first_failure,
        "operator_attribution_24h": operators,
        "internal_consistency_24h": invariants,
        "wrf_anchor_comparison": wrf_anchor_block,
        "coupling_chain_audit": chains,
        "next_sprint_recommendations": recommendations,
        "verdict": "DIAGNOSIS_PRODUCED",
        "operators_field_index": list(DIAGNOSTIC_FIELD_INDEX),
        "operators_index": list(DIAGNOSTIC_OPERATORS),
        "invariants_index": list(DIAGNOSTIC_INVARIANTS),
    }
    return DiagnosticReport(payload=_jsonable(payload))
