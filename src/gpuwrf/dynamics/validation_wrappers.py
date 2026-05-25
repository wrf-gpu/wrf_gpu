"""Validation-only compatibility wrappers around :mod:`gpuwrf.dynamics.core`.

This module keeps the savepoint-ladder API names stable while the numerical
math lives in ``dynamics.core``. Operational runtime must import core directly,
not this wrapper module.
"""

from gpuwrf.dynamics.core.acoustic import (
    AcousticCoreConfig,
    AcousticCoreState,
    AcousticLoopConfig,
    AcousticLoopState,
    FULL_STATE_FIELDS,
    _advance_inputs,
    acoustic_loop_wrf,
    acoustic_scan_core,
    acoustic_substep_core,
    acoustic_substep_wrf,
    advance_mu_t_core,
    snapshot_full_state,
    w_solve_core,
)
from gpuwrf.dynamics.core.coupled import (
    COUPLED_STATE_FIELDS,
    NAMELIST_PHYSICS_BOUNDARY_ON,
    PHYSICS_TENDENCY_FIELDS,
    CoupledCoreConfig,
    CoupledStepConfig,
    coupled_timestep_core,
    coupled_timestep_wrf,
    coupled_timesteps_core,
    coupled_timesteps_wrf,
)
from gpuwrf.dynamics.core.dycore import (
    DycoreCoreConfig,
    DycoreStepConfig,
    dycore_timestep_core,
    dycore_timestep_wrf,
    dycore_timesteps_core,
    dycore_timesteps_wrf,
    rk_stage_core,
)

__all__ = [
    "AcousticCoreConfig",
    "AcousticCoreState",
    "AcousticLoopConfig",
    "AcousticLoopState",
    "COUPLED_STATE_FIELDS",
    "CoupledCoreConfig",
    "CoupledStepConfig",
    "DycoreCoreConfig",
    "DycoreStepConfig",
    "FULL_STATE_FIELDS",
    "NAMELIST_PHYSICS_BOUNDARY_ON",
    "PHYSICS_TENDENCY_FIELDS",
    "_advance_inputs",
    "acoustic_loop_wrf",
    "acoustic_scan_core",
    "acoustic_substep_core",
    "acoustic_substep_wrf",
    "advance_mu_t_core",
    "coupled_timestep_core",
    "coupled_timestep_wrf",
    "coupled_timesteps_core",
    "coupled_timesteps_wrf",
    "dycore_timestep_core",
    "dycore_timestep_wrf",
    "dycore_timesteps_core",
    "dycore_timesteps_wrf",
    "rk_stage_core",
    "snapshot_full_state",
    "w_solve_core",
]
