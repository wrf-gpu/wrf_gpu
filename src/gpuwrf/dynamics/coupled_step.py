"""Validation-only coupled-step wrapper API.

The coupled numerical composition moved to ``gpuwrf.dynamics.core.coupled``.
This module remains only for savepoint-ladder and comparison-script
compatibility. Operational runtime must import ``dynamics.core`` directly.
"""

from gpuwrf.dynamics.validation_wrappers import (
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

__all__ = [
    "COUPLED_STATE_FIELDS",
    "NAMELIST_PHYSICS_BOUNDARY_ON",
    "PHYSICS_TENDENCY_FIELDS",
    "CoupledCoreConfig",
    "CoupledStepConfig",
    "coupled_timestep_core",
    "coupled_timestep_wrf",
    "coupled_timesteps_core",
    "coupled_timesteps_wrf",
]
