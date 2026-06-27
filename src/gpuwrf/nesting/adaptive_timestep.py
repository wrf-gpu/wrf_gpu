"""WRF ``adapt_timestep_em.F``-referenced adaptive timestep planner.

The planner is host metadata: it computes the next domain timestep from CFL
diagnostics and schedule constraints, leaving the compiled dycore stepper to be
called by the surrounding runtime.  It mirrors WRF's practical rules:

* vertical and horizontal CFL each propose a candidate via ``calc_dt``;
* the smaller candidate wins;
* growth is capped by ``max_step_increase_pct``;
* the result is rounded to the nearest 1/100 s, then min/max clamped;
* optional boundary/output time alignment shortens the next step to avoid a
  very short following step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AdaptiveTimeStepConfig:
    target_cfl: float = 1.2
    target_hcfl: float = 0.84
    min_time_step_s: float = 1.0
    max_time_step_s: float = 60.0
    max_step_increase_pct: float = 20.0
    precision: int = 100
    step_to_output_time: bool = False
    history_interval_s: float | None = None
    boundary_interval_s: float | None = None
    adapt_step_using_child: bool = False

    def __post_init__(self) -> None:
        if int(self.precision) <= 0:
            raise ValueError("precision must be positive")
        if float(self.min_time_step_s) <= 0.0:
            raise ValueError("min_time_step_s must be positive")
        if float(self.max_time_step_s) < float(self.min_time_step_s):
            raise ValueError("max_time_step_s must be >= min_time_step_s")
        if float(self.target_cfl) <= 0.0 or float(self.target_hcfl) <= 0.0:
            raise ValueError("target CFL values must be positive")

    @property
    def max_increase_factor(self) -> float:
        return 1.0 + float(self.max_step_increase_pct) / 100.0


@dataclass(frozen=True)
class AdaptiveTimeStepState:
    dt_s: float
    last_dt_s: float
    max_vert_cfl: float
    max_horiz_cfl: float
    current_seconds: float = 0.0
    dtbc_s: float = 0.0
    last_max_vert_cfl: float = 0.0
    last_max_horiz_cfl: float = 0.0
    stepping_to_time: bool = False
    advance_count: int = 1
    restart: bool = False


@dataclass(frozen=True)
class AdaptiveTimeStepResult:
    dt_s: float
    last_dt_s: float
    stepping_to_time: bool
    used_last2: bool
    reason: str
    max_vert_cfl: float
    max_horiz_cfl: float
    num_small_steps: int | None = None
    parent_dt_s: float | None = None


def _round_dt(value: float, precision: int) -> float:
    return round(float(value) * int(precision)) / float(precision)


def calc_dt_candidate(
    *,
    max_cfl: float,
    target_cfl: float,
    last_dt_s: float,
    max_increase_factor: float,
    precision: int = 100,
) -> float:
    """WRF ``calc_dt`` reduced to scalar seconds."""

    cfl = float(max_cfl)
    if cfl < 0.001:
        factor = float(max_increase_factor)
    elif cfl > float(target_cfl):
        factor = (float(target_cfl) - 0.5 * (cfl - float(target_cfl))) / cfl
        factor = max(0.1, factor)
    else:
        factor = float(target_cfl) / cfl
    return _round_dt(float(last_dt_s) * factor, int(precision))


def _limit_to_schedule(
    dt_s: float,
    *,
    current_seconds: float,
    interval_s: float | None,
    precision: int,
) -> tuple[float, bool, str | None]:
    if interval_s is None or float(interval_s) <= 0.0:
        return float(dt_s), False, None
    interval = float(interval_s)
    elapsed = float(current_seconds)
    rem = interval - math.fmod(elapsed, interval)
    if abs(rem - interval) <= 1.0e-12:
        rem = interval
    rem = _round_dt(rem, precision)
    if rem < 2.0 * float(dt_s) and rem > float(dt_s):
        return rem / 2.0, True, "half_step_to_schedule"
    if rem <= float(dt_s):
        target = round((elapsed + rem) / interval) * interval
        return max(0.0, target - elapsed), True, "step_to_schedule"
    return float(dt_s), False, None


def adapt_timestep(
    state: AdaptiveTimeStepState,
    config: AdaptiveTimeStepConfig,
    *,
    nested_parent_dt_s: float | None = None,
) -> AdaptiveTimeStepResult:
    """Compute one WRF-style adaptive timestep update."""

    precision = int(config.precision)
    reason = "cfl"
    if int(state.advance_count) == 0 and not bool(state.restart):
        dt = float(state.dt_s)
        last_dt = 0.0
        reason = "starting_time_step"
    else:
        last_dt = float(state.last_dt_s if state.last_dt_s > 0.0 else state.dt_s)
        max_vert = float(state.last_max_vert_cfl if state.stepping_to_time else state.max_vert_cfl)
        max_horiz = float(state.last_max_horiz_cfl if state.stepping_to_time else state.max_horiz_cfl)
        vert_dt = calc_dt_candidate(
            max_cfl=max_vert,
            target_cfl=float(config.target_cfl),
            last_dt_s=last_dt,
            max_increase_factor=config.max_increase_factor,
            precision=precision,
        )
        horiz_dt = calc_dt_candidate(
            max_cfl=max_horiz,
            target_cfl=float(config.target_hcfl),
            last_dt_s=last_dt,
            max_increase_factor=config.max_increase_factor,
            precision=precision,
        )
        dt = min(vert_dt, horiz_dt)

    cap = last_dt * config.max_increase_factor if last_dt > 0.0 else dt
    if float(state.current_seconds) != 0.0 and dt > cap:
        dt = cap
        reason = "max_increase_cap"
    dt = _round_dt(dt, precision)
    dt = min(float(config.max_time_step_s), max(float(config.min_time_step_s), dt))

    num_small_steps = None
    parent_dt = None
    if nested_parent_dt_s is not None:
        parent_dt = float(nested_parent_dt_s)
        if parent_dt <= 0.0:
            raise ValueError("nested_parent_dt_s must be positive")
        if not bool(config.adapt_step_using_child):
            num_small_steps = max(1, int(math.ceil(parent_dt / max(dt, 1.0e-12))))
            dt = parent_dt / float(num_small_steps)
            reason = "nested_parent_divisor"
        else:
            num_small_steps = max(1, int(math.floor(parent_dt / max(dt, 1.0e-12))))
            parent_dt = dt * float(num_small_steps)
            reason = "nested_child_controls_parent"

    used_last2 = False
    stepping = False
    if config.boundary_interval_s is not None:
        dt, stepping, sched_reason = _limit_to_schedule(
            dt,
            current_seconds=float(state.current_seconds) + float(state.dtbc_s),
            interval_s=float(config.boundary_interval_s),
            precision=precision,
        )
        if stepping:
            reason = sched_reason or reason
            used_last2 = True
    if config.step_to_output_time and not stepping and config.history_interval_s is not None:
        dt, stepping, sched_reason = _limit_to_schedule(
            dt,
            current_seconds=float(state.current_seconds),
            interval_s=float(config.history_interval_s),
            precision=precision,
        )
        if stepping:
            reason = sched_reason or reason
            used_last2 = True

    if used_last2:
        next_last_dt = last_dt
    else:
        next_last_dt = float(dt)
    return AdaptiveTimeStepResult(
        dt_s=float(dt),
        last_dt_s=float(next_last_dt),
        stepping_to_time=bool(stepping),
        used_last2=bool(used_last2),
        reason=reason,
        max_vert_cfl=float(state.max_vert_cfl),
        max_horiz_cfl=float(state.max_horiz_cfl),
        num_small_steps=num_small_steps,
        parent_dt_s=parent_dt,
    )


__all__ = [
    "AdaptiveTimeStepConfig",
    "AdaptiveTimeStepResult",
    "AdaptiveTimeStepState",
    "adapt_timestep",
    "calc_dt_candidate",
]
