"""Pure shared WRF-shaped full dycore timestep core.

This module composes the shared acoustic core in WRF's RK3 outer-loop cadence.

WRF ordering anchors:
- ``solve_em.F:1447`` starts ``Runge_Kutta_loop: DO rk_step = 1, rk_order``.
- ``solve_em.F:2409-2738`` builds acoustic coefficients inside each RK stage.
- ``solve_em.F:3065-4363`` runs ``small_steps`` for the acoustic loop.
- ``solve_em.F:6765`` closes the RK predictor-corrector loop.
- ``solve_em.F:8174`` calls ``after_all_rk_steps`` after the dycore step.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass

import jax
from jax import config

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.dynamics.core.acoustic import (
    AcousticCoreConfig,
    AcousticCoreState,
    FULL_STATE_FIELDS,
    acoustic_scan_core,
    snapshot_full_state,
)


configure_jax_x64()


@dataclass(frozen=True)
class DycoreCoreConfig:
    """Static shared core config for M6B5 RK3-over-acoustic composition."""

    dt: float
    dx: float
    dy: float
    acoustic_substeps: int = 10
    rk_order: int = 3
    epssm: float = 0.1
    top_lid: bool = False
    physics_enabled: bool = False
    boundary_enabled: bool = False
    periodic_x: bool = True
    specified: bool = False
    nested: bool = False

    def acoustic_config(self) -> AcousticCoreConfig:
        return AcousticCoreConfig(
            dt=float(self.dt),
            dx=float(self.dx),
            dy=float(self.dy),
            epssm=float(self.epssm),
            top_lid=bool(self.top_lid),
            periodic_x=bool(self.periodic_x),
            specified=bool(self.specified),
            nested=bool(self.nested),
        )


def _assert_m6b5_mode(cfg: DycoreCoreConfig) -> None:
    if int(cfg.rk_order) != 3:
        raise ValueError("M6B dycore core requires RK3")
    if bool(cfg.physics_enabled) or bool(cfg.boundary_enabled):
        raise ValueError("M6B dycore core requires physics and boundary disabled")


def rk_stage_core(
    state: AcousticCoreState,
    metrics: DycoreMetrics,
    cfg: DycoreCoreConfig,
) -> tuple[dict[str, jax.Array], AcousticCoreState]:
    """Run one shared RK-stage acoustic loop."""

    _substeps, loop_snapshot, _coefficients = acoustic_scan_core(
        state,
        metrics,
        cfg.acoustic_config(),
        substeps=int(cfg.acoustic_substeps),
    )
    return loop_snapshot, state.replace(**loop_snapshot)


def dycore_timestep_core(
    state: AcousticCoreState,
    metrics: DycoreMetrics,
    cfg: DycoreCoreConfig,
) -> tuple[list[dict[str, jax.Array]], dict[str, jax.Array]]:
    """Run one full dycore timestep with RK3 outer cadence."""

    _assert_m6b5_mode(cfg)
    current = state
    rk_snapshots: list[dict[str, jax.Array]] = []
    for _rk_stage in range(1, int(cfg.rk_order) + 1):
        loop_snapshot, current = rk_stage_core(current, metrics, cfg)
        rk_snapshots.append(snapshot_full_state(current))
    return rk_snapshots, snapshot_full_state(current)


def dycore_timesteps_core(
    state: AcousticCoreState,
    metrics: DycoreMetrics,
    cfg: DycoreCoreConfig,
    *,
    steps: int,
) -> tuple[list[dict[str, jax.Array]], list[list[dict[str, jax.Array]]]]:
    """Run repeated full dycore timesteps for core callers."""

    _assert_m6b5_mode(cfg)
    current = state
    step_snapshots: list[dict[str, jax.Array]] = []
    rk_snapshots_by_step: list[list[dict[str, jax.Array]]] = []
    for _step in range(1, int(steps) + 1):
        rk_snapshots, step_snapshot = dycore_timestep_core(current, metrics, cfg)
        current = current.replace(**step_snapshot)
        rk_snapshots_by_step.append(rk_snapshots)
        step_snapshots.append(step_snapshot)
    return step_snapshots, rk_snapshots_by_step


__all__ = [
    "DycoreCoreConfig",
    "FULL_STATE_FIELDS",
    "dycore_timestep_core",
    "dycore_timesteps_core",
    "rk_stage_core",
    "DycoreStepConfig",
    "dycore_timestep_wrf",
    "dycore_timesteps_wrf",
]


DycoreStepConfig = DycoreCoreConfig
dycore_timestep_wrf = dycore_timestep_core
dycore_timesteps_wrf = dycore_timesteps_core
