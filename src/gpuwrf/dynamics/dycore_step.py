"""Validation-only WRF-shaped full dycore timestep composition.

This module wraps the M6B4 acoustic loop in WRF's RK3 outer-loop cadence for
M6B5 savepoint parity. It is intentionally not imported by operational runtime.

WRF ordering anchors:
- ``solve_em.F:1447`` starts ``Runge_Kutta_loop: DO rk_step = 1, rk_order``.
- ``solve_em.F:2409-2738`` builds acoustic coefficients inside each RK stage.
- ``solve_em.F:3065-4363`` runs ``small_steps`` for the acoustic loop.
- ``solve_em.F:6765`` closes the RK predictor-corrector loop.
- ``solve_em.F:8174`` calls ``after_all_rk_steps`` after the dycore step.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import config

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.dynamics.acoustic_loop import (
    AcousticLoopConfig,
    AcousticLoopState,
    FULL_STATE_FIELDS,
    acoustic_loop_wrf,
    snapshot_full_state,
)


config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class DycoreStepConfig:
    """Static validation config for M6B5 RK3-over-acoustic composition."""

    dt: float
    dx: float
    dy: float
    acoustic_substeps: int = 10
    rk_order: int = 3
    epssm: float = 0.1
    top_lid: bool = False
    physics_enabled: bool = False
    boundary_enabled: bool = False

    def acoustic_config(self) -> AcousticLoopConfig:
        return AcousticLoopConfig(
            dt=float(self.dt),
            dx=float(self.dx),
            dy=float(self.dy),
            epssm=float(self.epssm),
            top_lid=bool(self.top_lid),
        )


def _assert_m6b5_mode(cfg: DycoreStepConfig) -> None:
    if int(cfg.rk_order) != 3:
        raise ValueError("M6B5 dycore_step validation requires RK3")
    if bool(cfg.physics_enabled) or bool(cfg.boundary_enabled):
        raise ValueError("M6B5 dycore_step validation requires physics and boundary disabled")


def dycore_timestep_wrf(
    state: AcousticLoopState,
    metrics: DycoreMetrics,
    cfg: DycoreStepConfig,
) -> tuple[list[dict[str, jax.Array]], dict[str, jax.Array]]:
    """Run one validation-only full dycore timestep with RK3 outer cadence."""

    _assert_m6b5_mode(cfg)
    current = state
    rk_snapshots: list[dict[str, jax.Array]] = []
    acoustic_cfg = cfg.acoustic_config()
    for _rk_stage in range(1, int(cfg.rk_order) + 1):
        _substeps, loop_snapshot, _coefficients = acoustic_loop_wrf(
            current,
            metrics,
            acoustic_cfg,
            substeps=int(cfg.acoustic_substeps),
        )
        current = current.replace(**loop_snapshot)
        rk_snapshots.append(snapshot_full_state(current))
    return rk_snapshots, snapshot_full_state(current)


def dycore_timesteps_wrf(
    state: AcousticLoopState,
    metrics: DycoreMetrics,
    cfg: DycoreStepConfig,
    *,
    steps: int,
) -> tuple[list[dict[str, jax.Array]], list[list[dict[str, jax.Array]]]]:
    """Run repeated full dycore timesteps for validation comparisons."""

    _assert_m6b5_mode(cfg)
    current = state
    step_snapshots: list[dict[str, jax.Array]] = []
    rk_snapshots_by_step: list[list[dict[str, jax.Array]]] = []
    for _step in range(1, int(steps) + 1):
        rk_snapshots, step_snapshot = dycore_timestep_wrf(current, metrics, cfg)
        current = current.replace(**step_snapshot)
        rk_snapshots_by_step.append(rk_snapshots)
        step_snapshots.append(step_snapshot)
    return step_snapshots, rk_snapshots_by_step


__all__ = [
    "DycoreStepConfig",
    "FULL_STATE_FIELDS",
    "dycore_timestep_wrf",
    "dycore_timesteps_wrf",
]
