"""Validation-only dycore-step wrapper API.

The RK/acoustic numerical composition moved to ``gpuwrf.dynamics.core.dycore``.
This module remains only for savepoint-ladder and comparison-script
compatibility. Operational runtime must import ``dynamics.core`` directly.
"""

from gpuwrf.dynamics.validation_wrappers import (
    DycoreCoreConfig,
    DycoreStepConfig,
    FULL_STATE_FIELDS,
    dycore_timestep_core,
    dycore_timestep_wrf,
    dycore_timesteps_core,
    dycore_timesteps_wrf,
    rk_stage_core,
)

__all__ = [
    "DycoreCoreConfig",
    "DycoreStepConfig",
    "FULL_STATE_FIELDS",
    "dycore_timestep_core",
    "dycore_timestep_wrf",
    "dycore_timesteps_core",
    "dycore_timesteps_wrf",
    "rk_stage_core",
]
