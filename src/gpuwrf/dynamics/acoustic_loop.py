"""Validation-only acoustic wrapper API.

The numerical recurrence moved to ``gpuwrf.dynamics.core.acoustic``. This
module remains only for savepoint-ladder and comparison-script compatibility.
Operational runtime must import ``dynamics.core`` directly.
"""

from gpuwrf.dynamics.validation_wrappers import (
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

__all__ = [
    "AcousticCoreConfig",
    "AcousticCoreState",
    "AcousticLoopConfig",
    "AcousticLoopState",
    "FULL_STATE_FIELDS",
    "_advance_inputs",
    "acoustic_loop_wrf",
    "acoustic_scan_core",
    "acoustic_substep_core",
    "acoustic_substep_wrf",
    "advance_mu_t_core",
    "snapshot_full_state",
    "w_solve_core",
]
